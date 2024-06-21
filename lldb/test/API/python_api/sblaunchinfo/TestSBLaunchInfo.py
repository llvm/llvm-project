"""
Test SBLaunchInfo
"""


from lldbsuite.test.lldbtest import *


def lookup(info, key):
    for i in range(info.GetNumEnvironmentEntries()):
        KeyEqValue = info.GetEnvironmentEntryAtIndex(i)
        Key, Value = KeyEqValue.split("=")
        if Key == key:
            return Value
    return ""


class TestSBLaunchInfo(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_environment_getset(self):
        info = lldb.SBLaunchInfo(None)
        info.SetEnvironmentEntries(["FOO=BAR"], False)
        self.assertEqual(1, info.GetNumEnvironmentEntries())
        info.SetEnvironmentEntries(["BAR=BAZ"], True)
        self.assertEqual(2, info.GetNumEnvironmentEntries())
        self.assertEqual("BAR", lookup(info, "FOO"))
        self.assertEqual("BAZ", lookup(info, "BAR"))
