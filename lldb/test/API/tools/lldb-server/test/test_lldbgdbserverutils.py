import unittest2
import re

from lldbgdbserverutils import *


class TestLldbGdbServerUtils(unittest2.TestCase):
    def test_entry_exact_payload_match(self):
        entry = GdbRemoteEntry(is_send_to_remote=False, exact_payload="$OK#9a")
        entry.assert_match(self, "$OK#9a")

    def test_entry_exact_payload_match_ignores_checksum(self):
        entry = GdbRemoteEntry(is_send_to_remote=False, exact_payload="$OK#9a")
        entry.assert_match(self, "$OK#00")

    def test_entry_creates_context(self):
        entry = GdbRemoteEntry(is_send_to_remote=False, exact_payload="$OK#9a")
        context = entry.assert_match(self, "$OK#9a")
        self.assertIsNotNone(context)

    def test_entry_regex_matches(self):
        entry = GdbRemoteEntry(
            is_send_to_remote=False,
            regex=re.compile(r"^\$QC([0-9a-fA-F]+)#"),
            capture={1: "thread_id"},
        )
        context = entry.assert_match(self, "$QC980#00")

    def test_entry_regex_saves_match(self):
        entry = GdbRemoteEntry(
            is_send_to_remote=False,
            regex=re.compile(r"^\$QC([0-9a-fA-F]+)#"),
            capture={1: "thread_id"},
        )
        context = entry.assert_match(self, "$QC980#00")
        self.assertEqual(context["thread_id"], "980")

    def test_entry_regex_expect_captures_success(self):
        context = {"thread_id": "980"}
        entry = GdbRemoteEntry(
            is_send_to_remote=False,
            regex=re.compile(r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+)"),
            expect_captures={2: "thread_id"},
        )
        entry.assert_match(self, "$T11thread:980;", context=context)

    def test_entry_regex_expect_captures_raises_on_fail(self):
        context = {"thread_id": "980"}
        entry = GdbRemoteEntry(
            is_send_to_remote=False,
            regex=re.compile(r"^\$T([0-9a-fA-F]{2})thread:([0-9a-fA-F]+)"),
            expect_captures={2: "thread_id"},
        )
        try:
            entry.assert_match(self, "$T11thread:970;", context=context)
            self.fail()
        except AssertionError:
            # okay
            return None
