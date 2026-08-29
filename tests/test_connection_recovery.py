"""Regression tests for stale BLE connection recovery."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components import niimbot as integration
from custom_components.niimbot.niimprint import parser
from custom_components.niimbot.niimprint.parser import NiimbotDevice
from custom_components.niimbot.niimprint.printer import PrinterTimeout

ADDRESS = "AA:BB:CC:DD:EE:FF"


def run(coro):
    return asyncio.run(coro)


def _patch_recovery_apis(monkeypatch, lookup_results):
    lookup = MagicMock(side_effect=lookup_results)
    clear_history = MagicMock()
    rediscover = MagicMock()
    active_scan = AsyncMock()
    close_stale = AsyncMock()
    monkeypatch.setattr(
        integration.bluetooth, "async_ble_device_from_address", lookup
    )
    monkeypatch.setattr(
        integration.bluetooth,
        "async_clear_advertisement_history",
        clear_history,
        raising=False,
    )
    monkeypatch.setattr(
        integration.bluetooth, "async_rediscover_address", rediscover
    )
    monkeypatch.setattr(
        integration.bluetooth,
        "async_request_active_scan",
        active_scan,
        raising=False,
    )
    monkeypatch.setattr(
        integration, "close_stale_connections_by_address", close_stale
    )
    return lookup, clear_history, rediscover, active_scan, close_stale


def test_update_timeout_forces_disconnect_with_keep_connection(monkeypatch):
    async def _test():
        device = NiimbotDevice("AA:BB:CC:DD:EE:FF", keep_connection=True)
        device.ble_data.name = "D110_M"
        device.ble_data.serial_number = "serial"
        device.ble_data.hw_version = "1.0"
        device.ble_data.sw_version = "1.0"
        device.ble_data.devicetype = "2304"
        device._info_loaded = True

        stale_client = MagicMock()
        stale_client.is_connected = True
        stale_client.disconnect = AsyncMock()
        stale_printer = MagicMock()
        stale_printer.heartbeat_payload = b"\x01"
        stale_printer.stop_notify = AsyncMock()
        stale_printer.get_sound = AsyncMock(return_value=None)
        stale_printer.heartbeat = AsyncMock(
            side_effect=PrinterTimeout("No response for request 0xdc")
        )
        device.client = stale_client
        device._printer = stale_printer

        ble_device = MagicMock(address=device.address, name="D110_M")
        with pytest.raises(PrinterTimeout):
            await device.update_device(ble_device)

        stale_printer.stop_notify.assert_awaited_once()
        stale_client.disconnect.assert_awaited_once()
        assert device._printer is None
        assert device.client is None

        fresh_client = MagicMock()
        fresh_client.is_connected = True
        establish = AsyncMock(return_value=fresh_client)
        monkeypatch.setattr(parser, "establish_connection", establish)
        fresh_printer = MagicMock()
        fresh_printer.start_notify = AsyncMock()
        printer_factory = MagicMock(return_value=fresh_printer)
        monkeypatch.setattr(parser, "PrinterClient", printer_factory)

        assert await device._ensure_printer(ble_device) is fresh_printer
        establish.assert_awaited_once()
        fresh_printer.start_notify.assert_awaited_once()

    run(_test())


def test_disconnect_clears_state_when_cleanup_fails():
    async def _test():
        device = NiimbotDevice("AA:BB:CC:DD:EE:FF", keep_connection=True)
        client = MagicMock()
        client.is_connected = True
        client.disconnect = AsyncMock(side_effect=RuntimeError("GATT is stale"))
        printer = MagicMock()
        printer.heartbeat_payload = None
        printer.stop_notify = AsyncMock(side_effect=RuntimeError("notify is stale"))
        device.client = client
        device._printer = printer

        await device.disconnect()

        printer.stop_notify.assert_awaited_once()
        client.disconnect.assert_awaited_once()
        assert device._printer is None
        assert device.client is None

    run(_test())


def test_timeout_recovery_clears_connector_cache(monkeypatch):
    async def _test():
        device = MagicMock()
        device.disconnect = AsyncMock()
        close_stale = AsyncMock()
        monkeypatch.setattr(
            integration, "close_stale_connections_by_address", close_stale
        )

        await integration._recover_stale_connection(
            "AA:BB:CC:DD:EE:FF", device
        )

        device.disconnect.assert_awaited_once()
        close_stale.assert_awaited_once_with("AA:BB:CC:DD:EE:FF")

    run(_test())


def test_available_ble_device_updates_without_recovery(monkeypatch):
    async def _test():
        ble_device = MagicMock()
        fresh_data = MagicMock()
        device = MagicMock()
        device.update_device = AsyncMock(return_value=fresh_data)
        lookup = MagicMock(return_value=ble_device)
        recover = AsyncMock(return_value=None)
        monkeypatch.setattr(
            integration.bluetooth, "async_ble_device_from_address", lookup
        )
        monkeypatch.setattr(integration, "_recover_missing_ble_device", recover)
        last_recovery = MagicMock()
        state = integration._MissingBLERecoveryState(
            last_attempt=1,
            missing=True,
            ble_recovery_count=2,
            last_ble_recovery=last_recovery,
            last_ble_recovery_result="still_unavailable",
        )

        result = await integration._async_update_niimbot(
            MagicMock(), ADDRESS, device, state
        )

        assert result is fresh_data
        device.update_device.assert_awaited_once_with(ble_device)
        recover.assert_not_awaited()
        assert state.last_attempt is None
        assert state.missing is False
        assert state.ble_recovery_count == 2
        assert state.last_ble_recovery is last_recovery
        assert state.last_ble_recovery_result == "still_unavailable"

    run(_test())


def test_missing_ble_device_runs_bounded_recovery(monkeypatch, caplog):
    async def _test():
        caplog.set_level(logging.DEBUG, logger=integration.__name__)
        hass = MagicMock()
        last_data = MagicMock()
        device = MagicMock(ble_data=last_data)
        device.disconnect = AsyncMock()
        device.update_device = AsyncMock()
        (
            lookup,
            clear_history,
            rediscover,
            active_scan,
            close_stale,
        ) = _patch_recovery_apis(monkeypatch, [None, None, None])

        state = integration._MissingBLERecoveryState()
        result = await integration._async_update_niimbot(
            hass, ADDRESS, device, state
        )

        assert result is last_data
        assert lookup.call_count == 3
        device.disconnect.assert_awaited_once()
        close_stale.assert_awaited_once_with(ADDRESS)
        clear_history.assert_called_once_with(hass, ADDRESS)
        rediscover.assert_called_once_with(hass, ADDRESS)
        active_scan.assert_awaited_once_with(
            hass, integration._MISSING_BLE_ACTIVE_SCAN_DURATION
        )
        device.update_device.assert_not_awaited()
        assert state.ble_recovery_count == 1
        assert state.last_ble_recovery is not None
        assert state.last_ble_recovery_result == "still_unavailable"
        assert any(
            record.levelno == logging.DEBUG
            and "BLE device not available" in record.message
            for record in caplog.records
        )
        assert not any(
            record.levelno >= logging.WARNING
            and "BLE device not available" in record.message
            for record in caplog.records
        )

    run(_test())


def test_missing_ble_device_updates_immediately_after_recovery(monkeypatch):
    async def _test():
        ble_device = MagicMock()
        fresh_data = MagicMock()
        device = MagicMock()
        device.disconnect = AsyncMock()
        device.update_device = AsyncMock(return_value=fresh_data)
        lookup, _, _, active_scan, _ = _patch_recovery_apis(
            monkeypatch, [None, None, ble_device]
        )
        state = integration._MissingBLERecoveryState()

        result = await integration._async_update_niimbot(
            MagicMock(), ADDRESS, device, state
        )

        assert result is fresh_data
        assert lookup.call_count == 3
        active_scan.assert_awaited_once()
        device.update_device.assert_awaited_once_with(ble_device)
        assert state.last_attempt is None
        assert state.missing is False
        assert state.ble_recovery_count == 1
        assert state.last_ble_recovery_result == "success"

    run(_test())


def test_missing_ble_device_returns_previous_data(monkeypatch, caplog):
    async def _test():
        caplog.set_level(logging.DEBUG, logger=integration.__name__)
        last_data = MagicMock()
        device = MagicMock(ble_data=last_data)
        lookup = MagicMock(side_effect=[None, None])
        recover = AsyncMock(return_value=None)
        monkeypatch.setattr(
            integration.bluetooth, "async_ble_device_from_address", lookup
        )
        monkeypatch.setattr(integration, "_recover_missing_ble_device", recover)

        state = integration._MissingBLERecoveryState()
        result = await integration._async_update_niimbot(
            MagicMock(), ADDRESS, device, state
        )

        assert result is last_data
        recover.assert_awaited_once()
        assert state.last_ble_recovery_result == "still_unavailable"
        assert not any(
            record.levelno >= logging.WARNING
            and "BLE device" in record.message
            for record in caplog.records
        )

    run(_test())


def test_missing_ble_recovery_respects_cooldown(monkeypatch, caplog):
    async def _test():
        caplog.set_level(logging.DEBUG, logger=integration.__name__)
        last_data = MagicMock()
        device = MagicMock(ble_data=last_data)
        lookup = MagicMock(return_value=None)
        recover = AsyncMock(return_value=None)
        monkeypatch.setattr(
            integration.bluetooth, "async_ble_device_from_address", lookup
        )
        monkeypatch.setattr(integration, "_recover_missing_ble_device", recover)
        monkeypatch.setattr(integration, "monotonic", MagicMock(return_value=101))
        state = integration._MissingBLERecoveryState(last_attempt=100, missing=True)

        result = await integration._async_update_niimbot(
            MagicMock(), ADDRESS, device, state
        )

        assert result is last_data
        lookup.assert_called_once()
        recover.assert_not_awaited()
        assert state.ble_recovery_count == 0
        assert any(
            record.levelno == logging.DEBUG
            and "cooldown is active" in record.message
            for record in caplog.records
        )

    run(_test())


def test_missing_ble_recovery_runs_after_cooldown(monkeypatch, caplog):
    async def _test():
        caplog.set_level(logging.DEBUG, logger=integration.__name__)
        device = MagicMock()
        lookup = MagicMock(side_effect=[None, None])
        recover = AsyncMock(return_value=None)
        monkeypatch.setattr(
            integration.bluetooth, "async_ble_device_from_address", lookup
        )
        monkeypatch.setattr(integration, "_recover_missing_ble_device", recover)
        monkeypatch.setattr(
            integration,
            "monotonic",
            MagicMock(return_value=integration._MISSING_BLE_RECOVERY_COOLDOWN + 1),
        )
        state = integration._MissingBLERecoveryState(last_attempt=0, missing=True)

        await integration._async_update_niimbot(
            MagicMock(), ADDRESS, device, state
        )

        recover.assert_awaited_once()
        lookup.assert_called_once()
        assert state.ble_recovery_count == 1
        assert state.last_ble_recovery_result == "still_unavailable"
        assert not any(
            record.levelno >= logging.WARNING
            and "BLE device" in record.message
            for record in caplog.records
        )

    run(_test())


def test_coordinator_timeout_recovery_is_preserved(monkeypatch, caplog):
    async def _test():
        caplog.set_level(logging.WARNING, logger=integration.__name__)
        ble_device = MagicMock()
        last_data = MagicMock()
        device = MagicMock(ble_data=last_data)
        device.update_device = AsyncMock(
            side_effect=PrinterTimeout("No response for request 0xdc")
        )
        device.disconnect = AsyncMock()
        close_stale = AsyncMock()
        monkeypatch.setattr(
            integration.bluetooth,
            "async_ble_device_from_address",
            MagicMock(return_value=ble_device),
        )
        monkeypatch.setattr(
            integration, "close_stale_connections_by_address", close_stale
        )

        state = integration._MissingBLERecoveryState()
        result = await integration._async_update_niimbot(
            MagicMock(), ADDRESS, device, state
        )

        assert result is last_data
        device.disconnect.assert_awaited_once()
        close_stale.assert_awaited_once_with(ADDRESS)
        assert state.ble_recovery_count == 1
        assert state.last_ble_recovery_result == "timeout_cleanup"
        assert any(
            record.levelno == logging.WARNING
            and "Printer timed out" in record.message
            for record in caplog.records
        )

    run(_test())


def test_unexpected_update_error_remains_a_warning(monkeypatch, caplog):
    async def _test():
        caplog.set_level(logging.WARNING, logger=integration.__name__)
        ble_device = MagicMock()
        last_data = MagicMock()
        device = MagicMock(ble_data=last_data)
        device.update_device = AsyncMock(side_effect=RuntimeError("GATT failed"))
        monkeypatch.setattr(
            integration.bluetooth,
            "async_ble_device_from_address",
            MagicMock(return_value=ble_device),
        )

        result = await integration._async_update_niimbot(
            MagicMock(), ADDRESS, device, integration._MissingBLERecoveryState()
        )

        assert result is last_data
        assert any(
            record.levelno == logging.WARNING
            and "Unable to fetch data" in record.message
            for record in caplog.records
        )

    run(_test())


def test_missing_recovery_forces_keep_connection_cleanup(monkeypatch):
    async def _test():
        device = NiimbotDevice(ADDRESS, keep_connection=True)
        client = MagicMock(is_connected=True)
        client.disconnect = AsyncMock()
        printer = MagicMock(heartbeat_payload=None)
        printer.stop_notify = AsyncMock()
        device.client = client
        device._printer = printer
        _, _, _, _, close_stale = _patch_recovery_apis(
            monkeypatch, [None, None]
        )

        await integration._recover_missing_ble_device(
            MagicMock(), ADDRESS, device
        )

        printer.stop_notify.assert_awaited_once()
        client.disconnect.assert_awaited_once()
        close_stale.assert_awaited_once_with(ADDRESS)
        assert device._printer is None
        assert device.client is None

    run(_test())


def test_active_scan_failure_is_safe_and_retriable(monkeypatch):
    async def _test():
        last_data = MagicMock()
        device = MagicMock(ble_data=last_data)
        _, _, _, active_scan, _ = _patch_recovery_apis(
            monkeypatch, [None, None, None, None, None, None]
        )
        active_scan.side_effect = RuntimeError("scanner unavailable")
        monkeypatch.setattr(
            integration,
            "monotonic",
            MagicMock(
                side_effect=[0, integration._MISSING_BLE_RECOVERY_COOLDOWN + 1]
            ),
        )
        state = integration._MissingBLERecoveryState()

        first = await integration._async_update_niimbot(
            MagicMock(), ADDRESS, device, state
        )
        second = await integration._async_update_niimbot(
            MagicMock(), ADDRESS, device, state
        )

        assert first is last_data
        assert second is last_data
        assert active_scan.await_count == 2
        assert state.ble_recovery_count == 2
        assert state.last_ble_recovery_result == "scan_failed"

    run(_test())
