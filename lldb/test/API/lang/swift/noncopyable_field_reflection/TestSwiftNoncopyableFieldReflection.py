import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftNoncopyableFieldReflection(lldbtest.TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @skipEmbeddedSwift
    @swiftTest
    def test(self):
        """Test that LLDB surfaces an error when a resilient class
        has a noncopyable stored property whose reflection type ref is
        an accessor-function symbolic reference. Such a field type ref
        cannot be resolved by passive reflection (LLDB cannot run the
        accessor in the target), so laying out the class fails.
        """
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec("main.swift"),
            extra_images=["Lib"])

        self.runCmd(
            "settings set symbols.swift-typesystem-compiler-fallback false")

        # Inspect a plain, fully reflectable struct first. This succeeds and
        # resets the reflection reader's transient "last demangle failure"
        # state to make we're not picking up a stale error.
        self.expect("frame variable p", substrs=["value = 7"])

        var_h = thread.frames[0].FindVariable("h")
        self.assertTrue(var_h.IsValid(), "'h' is a valid variable")
        self.expect("frame variable h",
                    substrs=["non-copyable fields in resilient types with a "
                             "deployment target < 27.0 are not supported"])
