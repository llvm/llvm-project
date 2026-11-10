import io
import json
from typing import List, Optional

from lldbsuite.test.tools.lldb_dap.types import RawMessage
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.utils import DAPConnection, MessageHandler, Transport


class EchoClient:
    def __init__(self):
        self.seen_messages: List[RawMessage] = []

    def on_message(self, msg: RawMessage):
        self.seen_messages.append(msg)


class _BytesTransport:
    """A transport reading from `data`.

    `read_error` if set is raised by read and readline instead of returning data.
    `reader_closed` makes `is_alive` report False, as if the peer had already
    closed the connection.
    """

    def __init__(
        self,
        data: bytes,
        read_error: Optional[Exception] = None,
        reader_closed: bool = False,
    ):
        self._in = io.BytesIO(data)
        self._read_error = read_error
        self._reader_closed = reader_closed
        self._closed = False
        self._out: List[str] = []

    def write(self, data: bytes):
        self._out.append(data.decode("utf-8"))

    def readline(self) -> bytes:
        if self._read_error:
            raise self._read_error
        return self._in.readline()

    def read(self, n: int) -> bytes:
        if self._read_error:
            raise self._read_error
        return self._in.read(n)

    def close(self):
        self._closed = True

    @property
    def is_alive(self) -> bool:
        return not self._closed and not self._reader_closed


class TestDAPUtils_DAPConnection(DAPTestCaseBase):
    USE_DEFAULT_DEBUG_ADAPTER = False

    def test_round_trip(self):
        expected_messages = self.get_sample_dap_log()
        transport = self.create_transport(expected_messages)
        connection = DAPConnection("conn0", transport)

        client = EchoClient()
        handler = MessageHandler(
            on_event=client.on_message,
            on_response=client.on_message,
            on_reverse_request=client.on_message,
        )
        connection.start(handler)
        # The sample log ends with the adapter closing the connection.
        self.assertFalse(connection.is_alive())

        received_messages = client.seen_messages
        self.assertEqual(len(expected_messages), len(received_messages))
        for received, expected in zip(received_messages, expected_messages):
            self.assertEqual(received, expected)

    def test_data_sent_before_the_adapter_closed_is_received(self):
        messages = self.get_sample_dap_log()
        encoded = b"".join(DAPConnection.encode_message(m) for m in messages)
        transport = _BytesTransport(encoded, reader_closed=True)
        connection = DAPConnection("conn0", transport)
        self.assertFalse(connection.is_alive())

        client = EchoClient()
        handler = MessageHandler(
            on_event=client.on_message,
            on_response=client.on_message,
            on_reverse_request=client.on_message,
        )
        connection.start(handler)

        self.assertEqual(client.seen_messages, messages)

    def test_read_error_is_reported_to_on_close(self):
        error = ConnectionResetError("reset by peer")
        connection = DAPConnection("conn0", self.create_failing_transport(error))

        closed_errors: List[Optional[Exception]] = []
        handler = MessageHandler(
            on_event=lambda m: None,
            on_response=lambda m: None,
            on_reverse_request=lambda m: None,
            on_close=closed_errors.append,
        )
        connection.start(handler)

        self.assertEqual(closed_errors, [error])
        self.assertFalse(connection.is_alive())

    def test_truncated_message_is_reported_to_on_close(self):
        encoded = DAPConnection.encode_message(
            {"type": "event", "seq": 1, "event": "initialized"}
        )
        # Cut the message in the middle of its body.
        connection = DAPConnection("conn0", _BytesTransport(encoded[:-5]))

        client = EchoClient()
        closed_errors: List[Optional[Exception]] = []
        handler = MessageHandler(
            on_event=client.on_message,
            on_response=client.on_message,
            on_reverse_request=client.on_message,
            on_close=closed_errors.append,
        )
        connection.start(handler)

        self.assertEqual(client.seen_messages, [])
        self.assertEqual(len(closed_errors), 1)
        self.assertIsInstance(closed_errors[0], EOFError)

    def test_encode_message_framing(self):
        payload = {"type": "request", "seq": 1, "command": "initialize"}
        data = DAPConnection.encode_message(payload)
        header, _, body = data.partition(b"\r\n\r\n")
        self.assertTrue(header.startswith(b"Content-Length:"))
        content_length = int(header.split(b":")[1].strip())
        self.assertEqual(content_length, len(body))
        self.assertEqual(json.loads(body), payload)

    def test_encode_message_round_trips(self):
        payload = {"nested": {"a": 1, "b": [1, 2, 3]}}
        _, _, body = DAPConnection.encode_message(payload).partition(b"\r\n\r\n")
        self.assertEqual(json.loads(body), payload)

    @staticmethod
    def create_transport(data: List[RawMessage]) -> Transport:
        encoded_data = b"".join(DAPConnection.encode_message(m) for m in data)
        return _BytesTransport(encoded_data)

    @staticmethod
    def create_failing_transport(error: Exception) -> Transport:
        return _BytesTransport(b"", read_error=error)

    def get_sample_dap_log(self) -> List[dict]:
        message_log = self.getSourcePath("sample_dap_log.json")
        with open(message_log, "r", encoding="utf-8") as file:
            raw_events = json.loads(file.read())
        return raw_events
