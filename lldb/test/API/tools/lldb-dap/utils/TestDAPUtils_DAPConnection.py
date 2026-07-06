import io
import json
from typing import List

from lldbsuite.test.tools.lldb_dap.dap_types import RawMessage
from lldbsuite.test.tools.lldb_dap.lldb_dap_testcase import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.utils import DAPConnection, MessageHandler, Transport


class EchoClient:
    def __init__(self):
        self.seen_messages: List[RawMessage] = []

    def on_message(self, msg: RawMessage):
        self.seen_messages.append(msg)


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

        received_messages = client.seen_messages
        self.assertEqual(len(expected_messages), len(received_messages))
        for received, expected in zip(received_messages, expected_messages):
            self.assertEqual(received, expected)

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
        class BinaryTransport:
            def __init__(self, data: List[RawMessage]):
                encoded_data = [
                    DAPConnection.encode_message(message) for message in data
                ]
                self._in = io.BytesIO(b"".join(encoded_data))
                self._out: List[str] = []

            def write(self, data: bytes):
                self._out.append(data.decode("utf-8"))

            def readline(self):
                return self._in.readline()

            def read(self, n: int) -> bytes:
                return self._in.read(n)

            def close(self):
                self._in.close()

            @property
            def is_alive(self) -> bool:
                return not self._in.closed

        return BinaryTransport(data)

    def get_sample_dap_log(self) -> List[dict]:
        message_log = self.getSourcePath("sample_dap_log.json")
        with open(message_log, "r", encoding="utf-8") as file:
            raw_events = json.loads(file.read())
        return raw_events
