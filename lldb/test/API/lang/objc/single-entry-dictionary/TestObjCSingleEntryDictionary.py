"""Test that we properly vend children for a single entry NSDictionary"""



import unittest2


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ObjCSingleEntryDictionaryTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.m', '// break here')

    @expectedFailureAll(oslist=['watchos'], bugnumber="rdar://problem/34642736") # bug in NSDictionary formatting on watchos
    def test_single_entry_dict(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the foo function which takes a bar_ptr argument.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        d1 = self.frame().FindVariable("d1")
        d1.SetPreferSyntheticValue(True)
        d1.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)

        self.assertTrue(
            d1.GetNumChildren() == 1,
            "dictionary has != 1 child elements")
        pair = d1.GetChildAtIndex(0)
        pair.SetPreferSyntheticValue(True)
        pair.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)

        self.assertTrue(
            pair.GetNumChildren() == 2,
            "pair has != 2 child elements")

        key = pair.GetChildMemberWithName("key")
        value = pair.GetChildMemberWithName("value")

        key.SetPreferSyntheticValue(True)
        key.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
        value.SetPreferSyntheticValue(True)
        value.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)

        self.assertTrue(
            key.GetSummary() == '@"key"',
            "key doesn't contain key")
        self.assertTrue(
            value.GetSummary() == '@"value"',
            "value doesn't contain value")
