"""
Test a synthetic child that is not owned by its parent's ClusterManager.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SyntheticForeignChildTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_child_outlives_its_last_reference(self):
        """The provider keeps no reference to the child it returns, so the
        parent has to keep it alive for later lookups."""
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.c")
        )
        # The ownership check replaces a foreign child with an error; turn it
        # off to see the child as a user does.
        self.runCmd("settings set target.check-vo-ownership false")
        self.runCmd(
            "command script import " + os.path.join(self.getSourceDir(), "provider.py")
        )
        self.runCmd(
            "type synthetic add -l provider.ForeignChildProvider 'struct Wrapper'"
        )

        wrapper = thread.GetSelectedFrame().FindVariable("wrapper")
        self.assertEqual(wrapper.GetNumChildren(), 1)
        # Look the child up several times without holding on to it, so only
        # the parent's cache keeps it alive between lookups.
        for _ in range(3):
            self.assertEqual(wrapper.GetChildAtIndex(0).GetValueAsSigned(), 47)
        self.assertEqual(wrapper.GetChildMemberWithName("child").GetValueAsSigned(), 47)
        self.expect("frame variable wrapper", substrs=["child = 47"])
        self.expect("frame variable wrapper", substrs=["child = 47"])
