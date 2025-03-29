"""
Test that the FrameRecognizer for __abort_with_payload
works properly
"""


import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestAbortWithPayload(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessAppleSilicon
    def test_abort_with_payload(self):
        """There can be many tests in a test case - describe this test here."""
        self.build()
        self.abort_with_test(True)

    @skipUnlessAppleSilicon
    def test_abort_with_reason(self):
        """There can be many tests in a test case - describe this test here."""
        self.build()
        self.abort_with_test(False)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.main_source_file = lldb.SBFileSpec("main.c")

    def abort_with_test(self, with_payload):
        """If with_payload is True, we test the abort_with_payload call,
        if false, we test abort_with_reason."""
        launch_info = lldb.SBLaunchInfo([])
        if not with_payload:
            launch_info.SetArguments(["use_reason"], True)
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self,
            "Stop here before abort",
            self.main_source_file,
            launch_info=launch_info,
        )

        frame = thread.GetFrameAtIndex(0)
        payload_str_var = frame.FindVariable("payload_string")
        self.assertSuccess(payload_str_var.GetError(), "Got payload string var")
        payload_var_addr = payload_str_var.unsigned

        payload_size_var = frame.FindVariable("payload_string_len")
        self.assertSuccess(payload_size_var.GetError(), "Got payload string len var")
        payload_size_val = payload_size_var.unsigned

        # Not let it run to crash:
        process.Continue()

        # At this point we should have stopped at the internal function.
        # Make sure we selected the right thread:
        sel_thread = process.GetSelectedThread()
        self.assertEqual(thread, sel_thread, "Selected the original thread")
        # Make sure the stop reason is right:
        self.assertEqual(
            thread.GetStopDescription(100),
            "abort with payload or reason",
            "Description was right",
        )
        frame_0 = thread.frames[0]
        self.assertEqual(frame_0.name, "__abort_with_payload", "Frame 0 was right")

        # Now check the recognized argument values and the ExtendedCrashInformation version:
        options = lldb.SBVariablesOptions()
        options.SetIncludeRecognizedArguments(True)
        options.SetIncludeArguments(False)
        options.SetIncludeLocals(False)
        options.SetIncludeStatics(False)
        options.SetIncludeRuntimeSupportValues(False)

        arguments = frame_0.GetVariables(options)

        correct_values = {
            "namespace": 5,
            "code": 100,
            "payload_addr": payload_var_addr,
            "payload_size": payload_size_val,
            "payload_string": '"This is a payload that happens to be a string"',
            "reason_string": '"This is the reason string"',
            "reason_no_quote": "This is the reason string",
            "flags": 0x85,
        }

        # First check the recognized argument values:
        self.assertEqual(len(arguments), 6, "Got all six values")
        self.assertEqual(arguments[0].name, "namespace")
        self.assertEqual(
            arguments[0].unsigned,
            correct_values["namespace"],
            "Namespace value correct",
        )

        self.assertEqual(arguments[1].name, "code")
        self.assertEqual(
            arguments[1].unsigned, correct_values["code"], "code value correct"
        )

        # We always stop at __abort_with_payload, regardless of whether the caller
        # was abort_with_reason or abort_with_payload or any future API that
        # funnels here. Since I don't want to have to know too much about the
        # callers, I just always report what is in the function I've 
        # 
        # add the payload ones if it is the payload not the reason function.
        self.assertEqual(arguments[2].name, "payload_addr")
        self.assertEqual(arguments[3].name, "payload_size")
        if with_payload:
            self.assertEqual(
                arguments[2].unsigned,
                correct_values["payload_addr"],
                "Payload matched variable address",
            )
            # We've made a payload that is a string, try to fetch that:
            char_ptr_type = target.FindFirstType("char").GetPointerType()
            self.assertTrue(char_ptr_type.IsValid(), "Got char ptr type")

            str_val = arguments[2].Cast(char_ptr_type)
            self.assertEqual(
                str_val.summary, correct_values["payload_string"], "Got payload string"
            )

            self.assertEqual(
                arguments[3].unsigned,
                correct_values["payload_size"],
                "payload size value correct",
            )
        else:
            self.assertEqual(
                arguments[2].unsigned, 0, "Got 0 payload addr for reason call"
            )
            self.assertEqual(
                arguments[3].unsigned, 0, "Got 0 payload size for reason call"
            )

        self.assertEqual(arguments[4].name, "reason")
        self.assertEqual(
            arguments[4].summary,
            correct_values["reason_string"],
            "Reason value correct",
        )

        self.assertEqual(arguments[5].name, "flags")
        self.assertEqual(
            arguments[5].unsigned, correct_values["flags"], "Flags value correct"
        )

        # Also check that the same info was stored in the ExtendedCrashInformation dict:
        dict = process.GetExtendedCrashInformation()
        self.assertTrue(dict.IsValid(), "Got extended crash information dict")
        self.assertEqual(
            dict.GetType(), lldb.eStructuredDataTypeDictionary, "It is a dictionary"
        )

        abort_dict = dict.GetValueForKey("abort_with_payload")
        self.assertTrue(abort_dict.IsValid(), "Got an abort_with_payload dict")
        self.assertEqual(
            abort_dict.GetType(),
            lldb.eStructuredDataTypeDictionary,
            "It is a dictionary",
        )

        namespace_val = abort_dict.GetValueForKey("namespace")
        self.assertTrue(namespace_val.IsValid(), "Got a valid namespace")
        self.assertEqual(
            namespace_val.GetIntegerValue(0),
            correct_values["namespace"],
            "Namespace value correct",
        )

        code_val = abort_dict.GetValueForKey("code")
        self.assertTrue(code_val.IsValid(), "Got a valid code")
        self.assertEqual(
            code_val.GetIntegerValue(0), correct_values["code"], "Code value correct"
        )

        if with_payload:
            addr_val = abort_dict.GetValueForKey("payload_addr")
            self.assertTrue(addr_val.IsValid(), "Got a payload_addr")
            self.assertEqual(
                addr_val.GetIntegerValue(0),
                correct_values["payload_addr"],
                "payload_addr right in dictionary",
            )

            size_val = abort_dict.GetValueForKey("payload_size")
            self.assertTrue(size_val.IsValid(), "Got a payload size value")
            self.assertEqual(
                size_val.GetIntegerValue(0),
                correct_values["payload_size"],
                "payload size right in dictionary",
            )

        reason_val = abort_dict.GetValueForKey("reason")
        self.assertTrue(reason_val.IsValid(), "Got a reason key")
        self.assertEqual(
            reason_val.GetStringValue(100),
            correct_values["reason_no_quote"],
            "reason right in dictionary",
        )

        flags_val = abort_dict.GetValueForKey("flags")
        self.assertTrue(flags_val.IsValid(), "Got a flags value")
        self.assertEqual(
            flags_val.GetIntegerValue(0),
            correct_values["flags"],
            "flags right in dictionary",
        )
