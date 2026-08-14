import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedRuntimeFailureMessage(TestBase):
    """Tests how the message of a Swift runtime failure reaches the debugger in
    embedded Swift.

    The message is recorded in the debug info of the trap, where the
    Swift runtime failure recognizer finds it and reports it as the
    stop reason. With assertions the standard library sends it to a
    platform hook to, e.g., print it, before trapping. A message that
    cannot be folded into the trap is replaced by a generic one.
    """

    SHARED_BUILD_TESTCASE = False
    message = "index must not be negative"

    def run_to_failure(self, extra_flags):
        """Run until the program traps and return the stopped thread."""
        self.build(dictionary={"SWIFTFLAGS_EXTRAS": extra_flags})
        target = lldbutil.run_to_breakpoint_make_target(self)
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertState(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
        return process.GetSelectedThread()

    def failure_line(self, extra_flags):
        """The line doomed() fails on, which depends on the API it calls."""
        if "ASSERTION_FAILURE" in extra_flags:
            call = "assertionFailure(message)"
        elif "ASSERT" in extra_flags:
            call = "assert(x >= 0, message)"
        elif "PRECONDITION_FAILURE" in extra_flags:
            call = "preconditionFailure(message)"
        elif "PRECONDITION" in extra_flags:
            call = "precondition(x >= 0, message)"
        else:
            call = "fatalError(message)"
        return line_number("main.swift", call)

    def check_frame(self, thread, line):
        frame = thread.GetSelectedFrame()
        self.assertIn("doomed", frame.GetDisplayFunctionName())
        line_entry = frame.GetLineEntry()
        self.assertEqual(line_entry.GetFileSpec().GetFilename(), "main.swift")
        self.assertEqual(line_entry.GetLine(), line)

    def check_stop_reason(self, config_flags, extra_flags, message=None):
        thread = self.run_to_failure(config_flags + " " + extra_flags)
        self.assertEqual(
            thread.GetStopDescription(1024),
            "Swift runtime failure: " + (message or self.message),
        )
        self.check_frame(thread, self.failure_line(extra_flags))

    def check_without_asserts(self, extra_flags, message=None):
        self.check_stop_reason("-O", extra_flags, message)

    def check_with_asserts(self, extra_flags, message=None):
        self.check_stop_reason("-O -assert-config Debug", extra_flags, message)

    @skipUnlessDarwin
    @swiftTest
    @skipUnlessEmbeddedSwift
    def test_fatal_error_no_asserts(self):
        self.check_without_asserts("")

    @skipUnlessDarwin
    @swiftTest
    @skipUnlessEmbeddedSwift
    def test_fatal_error_asserts(self):
        self.check_with_asserts("")

    @skipUnlessDarwin
    @swiftTest
    @skipUnlessEmbeddedSwift
    def test_precondition_no_asserts(self):
        self.check_without_asserts("-D PRECONDITION")

    @skipUnlessDarwin
    @swiftTest
    @skipUnlessEmbeddedSwift
    def test_precondition_asserts(self):
        self.check_with_asserts("-D PRECONDITION")

    @skipUnlessDarwin
    @swiftTest
    @skipUnlessEmbeddedSwift
    def test_precondition_failure_no_asserts(self):
        self.check_without_asserts("-D PRECONDITION_FAILURE")

    @skipUnlessDarwin
    @swiftTest
    @skipUnlessEmbeddedSwift
    def test_precondition_failure_asserts(self):
        self.check_with_asserts("-D PRECONDITION_FAILURE")

    @skipUnlessDarwin
    @swiftTest
    @skipUnlessEmbeddedSwift
    def test_fallback_no_asserts(self):
        self.check_without_asserts("-D FALLBACK", "unknown program error")

    @skipUnlessDarwin
    @swiftTest
    @skipUnlessEmbeddedSwift
    def test_fallback_asserts(self):
        self.check_with_asserts("-D FALLBACK", "unknown program error")

    # assert and assertionFailure have no effect without assertions, so they only
    # have a failure to report in this configuration.

    @skipUnlessDarwin
    @swiftTest
    @skipUnlessEmbeddedSwift
    def test_assert_asserts(self):
        self.check_with_asserts("-D ASSERT")

    @skipUnlessDarwin
    @swiftTest
    @skipUnlessEmbeddedSwift
    def test_assertion_failure_asserts(self):
        self.check_with_asserts("-D ASSERTION_FAILURE")
