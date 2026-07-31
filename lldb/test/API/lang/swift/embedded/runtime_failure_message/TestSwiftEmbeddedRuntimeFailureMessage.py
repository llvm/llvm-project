import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedRuntimeFailureMessage(TestBase):
    """
    Tests how the message of a Swift runtime failure reaches the debugger in
    embedded Swift.

    Without assertions the message is recorded only in the debug info of the
    trap, and the Swift runtime failure recognizer reports it as the stop
    reason. With assertions the standard library prints the message at runtime
    instead, and the stop reason is a plain trap. A message that cannot be
    folded into the trap is replaced by a generic one.
    """

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
        if "PRECONDITION_FAILURE" in extra_flags:
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

    def check_message_in_stop_reason(self, extra_flags, message=None):
        thread = self.run_to_failure("-O " + extra_flags)
        self.assertEqual(
            thread.GetStopDescription(1024),
            "Swift runtime failure: " + (message or self.message),
        )
        self.check_frame(thread, self.failure_line(extra_flags))

    def check_no_message_in_stop_reason(self, extra_flags):
        thread = self.run_to_failure("-O -assert-config Debug " + extra_flags)
        self.assertNotIn(self.message, thread.GetStopDescription(1024))
        self.check_frame(thread, self.failure_line(extra_flags))

    @skipUnlessDarwin
    @swiftTest
    @skipUnlessEmbeddedSwift
    def test_fatal_error_no_asserts(self):
        self.check_message_in_stop_reason("")

    @skipUnlessDarwin
    @swiftTest
    @skipUnlessEmbeddedSwift
    def test_fatal_error_asserts(self):
        self.check_no_message_in_stop_reason("")

    @skipUnlessDarwin
    @swiftTest
    @skipUnlessEmbeddedSwift
    def test_precondition_no_asserts(self):
        self.check_message_in_stop_reason("-D PRECONDITION")

    @skipUnlessDarwin
    @swiftTest
    @skipUnlessEmbeddedSwift
    def test_precondition_asserts(self):
        self.check_no_message_in_stop_reason("-D PRECONDITION")

    @skipUnlessDarwin
    @swiftTest
    @skipUnlessEmbeddedSwift
    def test_precondition_failure_no_asserts(self):
        self.check_message_in_stop_reason("-D PRECONDITION_FAILURE")

    @skipUnlessDarwin
    @swiftTest
    @skipUnlessEmbeddedSwift
    def test_precondition_failure_asserts(self):
        self.check_no_message_in_stop_reason("-D PRECONDITION_FAILURE")

    @skipUnlessDarwin
    @swiftTest
    @skipUnlessEmbeddedSwift
    def test_fallback_no_asserts(self):
        self.check_message_in_stop_reason("-D FALLBACK", "unknown program error")
