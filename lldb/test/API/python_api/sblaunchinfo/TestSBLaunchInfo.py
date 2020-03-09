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

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def test_environment_getset(self):
        info = lldb.SBLaunchInfo(None)
        info.SetEnvironmentEntries(["FOO=BAR"], False)
        self.assertEquals(1, info.GetNumEnvironmentEntries())
        info.SetEnvironmentEntries(["BAR=BAZ"], True)
        self.assertEquals(2, info.GetNumEnvironmentEntries())
        self.assertEquals("BAR", lookup(info, "FOO"))
        self.assertEquals("BAZ", lookup(info, "BAR"))
