import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftRuntimeInstrumentationRecognizer(lldbtest.TestBase):
    @swiftTest
    def test(self):
        """Test Swift Runtime Instrumentation Recognizer"""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"))
        self.runCmd("process launch")

        self.expect("frame recognizer list",
                    substrs=['Swift Runtime Instrumentation StackFrame Recognizer, symbol _swift_runtime_on_report (regexp)'])

        self.expect("frame recognizer info 0",
                    substrs=['frame 0 is recognized by Swift Runtime Instrumentation StackFrame Recognizer'])

        self.expect("thread info",
                    substrs=['stop reason = Fatal error: Division by zero'])

        self.expect('bt')
        self.expect("frame info",
                    patterns=['frame #(.*)`testit(.*)at RuntimeError\.swift'])
