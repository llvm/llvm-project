import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil

class TestCase(lldbtest.TestBase):

    @swiftTest
    @skipIf(oslist=['windows', 'linux'])
    def test(self):
        """Test `frame variable` in async functions"""
        self.build()

        source_file = lldb.SBFileSpec("main.swift")
        target, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "// break one", source_file)

        # At "break one", only the `a` variable should have a value.
        frame = process.GetSelectedThread().frames[0]
        a = frame.FindVariable("a")
        self.assertTrue(a.IsValid())
        self.assertGreater(a.unsigned, 0)
        b = frame.FindVariable("b")
        self.assertTrue(b.IsValid())
        self.assertEqual(b.unsigned, 0)

        # The first breakpoint resolves to multiple locations, but only the
        # first location is needed. Now that we've stopped, delete it to
        # prevent the other locations from interrupting the test.
        target.DeleteAllBreakpoints()

        # Setup, and run to, the next breakpoint.
        target.BreakpointCreateBySourceRegex("// break two", source_file)
        self.setAsync(False)
        process.Continue()

        # At "break two", both `a` and `b` should have values.
        frame = process.GetSelectedThread().frames[0]
        a = frame.FindVariable("a")
        self.assertTrue(a.IsValid())
        self.assertGreater(a.unsigned, 0)
        b = frame.FindVariable("b")
        self.assertTrue(b.IsValid())
        self.assertGreater(b.unsigned, 0)
