"""Regression tests for stale BLE connection recovery."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.niimbot.niimprint import parser
from custom_components.niimbot.niimprint.parser import NiimbotDevice
from custom_components.niimbot.niimprint.printer import PrinterTimeout


def run(coro):
    return asyncio.run(coro)


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
