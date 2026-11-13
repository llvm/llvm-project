"""
Test synthetic children that are not owned by their parent's ClusterManager.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SyntheticForeignChildTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def foreign_child_warnings(self, listener):
        messages = []
        event = lldb.SBEvent()
        while listener.GetNextEvent(event):
            diagnostic = lldb.SBDebugger.GetDiagnosticFromEvent(event)
            message = diagnostic.GetValueForKey("message").GetStringValue(1000)
            if "synthetic child provider" in message:
                messages.append(message)
        return messages

    def test_foreign_children(self):
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
        self.runCmd(
            "type synthetic add -l provider.ChildOfAnotherValueProvider "
            "'struct Outer'"
        )
        broadcaster = self.dbg.GetBroadcaster()
        listener = lldbutil.start_listening_from(
            broadcaster, lldb.SBDebugger.eBroadcastBitWarning
        )
        frame = thread.GetSelectedFrame()

        # ForeignChildProvider keeps no reference to the root value it returns,
        # so the parent has to keep it alive between lookups.
        wrapper = frame.FindVariable("wrapper")
        self.assertEqual(wrapper.GetNumChildren(), 1)
        for _ in range(3):
            self.assertEqual(wrapper.GetChildAtIndex(0).GetValueAsSigned(), 47)
        self.assertEqual(wrapper.GetChildMemberWithName("child").GetValueAsSigned(), 47)
        self.expect("frame variable wrapper", substrs=["child = 47"])
        self.expect("frame variable wrapper", substrs=["child = 47"])
        warnings = self.foreign_child_warnings(listener)
        self.assertEqual(len(warnings), 1, warnings)
        self.assertIn("'Wrapper' returned 'child'", warnings[0])
        self.assertIn("is not owned by its parent", warnings[0])

        # A child of another value is not kept alive by the parent, but gets a
        # stronger warning.
        outer = frame.FindVariable("outer")
        for _ in range(3):
            self.assertEqual(outer.GetChildAtIndex(0).GetValueAsSigned(), 48)
        self.expect("frame variable outer", substrs=["value = 48"])
        warnings = self.foreign_child_warnings(listener)
        self.assertEqual(len(warnings), 1, warnings)
        self.assertIn("'Outer' returned 'value'", warnings[0])
        self.assertIn("may be freed while it is still in use", warnings[0])

        # Each warning is reported once per session.
        self.expect(
            "frame variable wrapper outer", substrs=["child = 47", "value = 48"]
        )
        self.assertEqual(self.foreign_child_warnings(listener), [])
