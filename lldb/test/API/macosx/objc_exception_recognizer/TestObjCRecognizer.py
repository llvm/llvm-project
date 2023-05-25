"""
Test that the built in ObjC exception throw recognizer works
"""

import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestObjCRecognizer(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    def test_exception_recognizer_sub_class(self):
        """There can be many tests in a test case - describe this test here."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.m")
        self.objc_recognizer_test(True)

    @skipUnlessDarwin
    def test_exception_recognizer_plain(self):
        """There can be many tests in a test case - describe this test here."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.m")
        self.objc_recognizer_test(False)

    def objc_recognizer_test(self, sub_class):
        """Make sure we stop at the exception and get all the fields out of the recognizer.
        If sub_class is True, we make a subclass of NSException and throw that."""
        if sub_class:
            bkpt_string = "Set a breakpoint here for MyException"
        else:
            bkpt_string = "Set a breakpoint here for plain exception"

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, bkpt_string, self.main_source_file
        )

        # Now turn on the ObjC Exception breakpoint and continue to hit it:
        exception_bkpt = target.BreakpointCreateForException(
            lldb.eLanguageTypeObjC, False, True
        )
        self.assertTrue(
            exception_bkpt.GetNumLocations() > 0, "Got some exception locations"
        )

        threads = lldbutil.continue_to_breakpoint(process, exception_bkpt)
        self.assertEqual(len(threads), 1, "One thread hit exception breakpoint")
        frame = threads[0].frame[0]

        var_opts = lldb.SBVariablesOptions()
        var_opts.SetIncludeRecognizedArguments(True)
        var_opts.SetUseDynamic(True)
        vars = frame.GetVariables(var_opts)
        self.assertEqual(len(vars), 1, "Got the synthetic argument")
        self.assertTrue(vars[0].IsValid(), "Got a valid Exception variable")

        # This will be a pointer

        ns_exception_children = [
            ValueCheck(type="NSObject"),
            ValueCheck(name="name", summary='"NSException"'),
            ValueCheck(name="reason", summary='"Simple Reason"'),
            ValueCheck(name="userInfo"),
            ValueCheck(name="reserved"),
        ]
        ns_exception = ValueCheck(type="NSException", children=ns_exception_children)
        if not sub_class:
            simple_check = ValueCheck(name="exception", dereference=ns_exception)
            simple_check.check_value(self, vars[0], "Simple exception is right")
        else:
            my_exception_children = [
                ns_exception,
                ValueCheck(name="extra_info", type="int", value="100"),
            ]
            my_exception = ValueCheck(
                type="MyException", children=my_exception_children
            )
            sub_check = ValueCheck(
                name="exception", type="MyException *", dereference=my_exception
            )
            sub_check.check_value(self, vars[0], "Subclass exception is right")
