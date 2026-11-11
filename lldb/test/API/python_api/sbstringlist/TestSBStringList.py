"""Test the SBStringList API."""

from typing import List

import lldb
from lldbsuite.test.lldbtest import *


def make_list(items: List[str]):
    slist = lldb.SBStringList()
    for item in items:
        slist.AppendString(item)
    return slist


class SBStringListAPICase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_repr(self):
        # Empty list.
        strings = lldb.SBStringList()
        self.assertEqual(repr(strings), "[]")

        strings = make_list(["foo"])
        self.assertEqual(repr(strings), "['foo']")

        strings = make_list(["foo", "bar", "baz"])
        self.assertEqual(repr(strings), "['foo', 'bar', 'baz']")

    def test_getitem_positive_index(self):
        strings = make_list(["foo", "bar", "baz"])
        self.assertEqual(strings[0], "foo")
        self.assertEqual(strings[1], "bar")
        self.assertEqual(strings[2], "baz")

        # Negative index
        self.assertEqual(strings[-1], "baz")
        self.assertEqual(strings[-2], "bar")
        self.assertEqual(strings[-3], "foo")

        # Out of range.
        with self.assertRaises(IndexError):
            strings[4]
        with self.assertRaises(IndexError):
            strings[100]

    def test_getitem_wrong_type(self):
        strings = make_list(["foo", "bar"])
        with self.assertRaises(TypeError):
            strings["not-an-index"]
        with self.assertRaises(TypeError):
            strings[1.5]

    def test_getitem_slice(self):
        strings = make_list(["a", "b", "c", "d", "e"])
        self.assertEqual(strings[1:4], ["b", "c", "d"])

        # Copy.
        items = ["a", "b", "c"]
        strings = make_list(items)
        self.assertEqual(strings[:], items)

        # Stepping.
        strings = make_list(["a", "b", "c", "d", "e"])
        self.assertEqual(strings[::2], ["a", "c", "e"])
        self.assertEqual(strings[1::2], ["b", "d"])

        self.assertEqual(strings[-3:], ["c", "d", "e"])
        self.assertEqual(strings[::-1], ["e", "d", "c", "b", "a"])

        # Out of range slice.
        strings = make_list(["a", "b", "c"])
        self.assertEqual(strings[10:20], [])
        self.assertEqual(lldb.SBStringList()[:], [])
