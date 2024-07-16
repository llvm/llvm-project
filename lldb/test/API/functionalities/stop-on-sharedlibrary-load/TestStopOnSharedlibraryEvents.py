""" Test that stop-on-sharedlibrary-events works and cooperates with breakpoints. """
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestStopOnSharedlibraryEvents(TestBase):
    @skipIfRemote
    @skipIfWindows
    @no_debug_info_test
    def test_stopping_breakpoints(self):
        self.do_test()

    @skipIfRemote
    @skipIfWindows
    @no_debug_info_test
    def test_auto_continue(self):
        def auto_continue(bkpt):
            bkpt.SetAutoContinue(True)

        self.do_test(auto_continue)

    @skipIfRemote
    @skipIfWindows
    @no_debug_info_test
    def test_failing_condition(self):
        def condition(bkpt):
            bkpt.SetCondition("1 == 2")

        self.do_test(condition)

    @skipIfRemote
    @skipIfWindows
    @no_debug_info_test
    def test_continue_callback(self):
        def bkpt_callback(bkpt):
            bkpt.SetScriptCallbackBody("return False")

        self.do_test(bkpt_callback)

    def do_test(self, bkpt_modifier=None):
        self.build()
        main_spec = lldb.SBFileSpec("main.cpp")
        # Launch and stop before the dlopen call.
        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self,
            "// Set a breakpoint here",
            main_spec,
            extra_images=["load_a", "load_b"],
        )

        # Now turn on shared library events, continue and make sure we stop for the event.
        self.runCmd("settings set target.process.stop-on-sharedlibrary-events 1")
        self.addTearDownHook(
            lambda: self.runCmd(
                "settings set target.process.stop-on-sharedlibrary-events 0"
            )
        )

        # Since I don't know how to check that we are at the "right place" to stop for
        # shared library events, make an breakpoint after the load is done and
        # make sure we don't stop there:
        backstop_bkpt_1 = target.BreakpointCreateBySourceRegex(
            "Set another here - we should not hit this one", main_spec
        )
        self.assertGreater(
            backstop_bkpt_1.GetNumLocations(), 0, "Set our second breakpoint"
        )

        process.Continue()
        self.assertState(
            process.GetState(), lldb.eStateStopped, "We didn't stop for the load"
        )
        self.assertEqual(
            backstop_bkpt_1.GetHitCount(), 0, "Hit our backstop breakpoint"
        )

        # We should be stopped after the library is loaded, check that:
        found_it = False
        for module in target.modules:
            if module.file.basename.find("load_a") > -1:
                found_it = True
                break
        self.assertTrue(found_it, "Found the loaded module.")

        # Now capture the place where we stopped so we can set a breakpoint and make
        # sure the breakpoint there works correctly:
        load_address = process.GetSelectedThread().frames[0].addr
        load_bkpt = target.BreakpointCreateBySBAddress(load_address)
        self.assertGreater(load_bkpt.GetNumLocations(), 0, "Set the load breakpoint")

        backstop_bkpt_1.SetEnabled(False)

        backstop_bkpt_2 = target.BreakpointCreateBySourceRegex(
            "Set a third here - we should not hit this one", main_spec
        )
        self.assertGreater(
            backstop_bkpt_2.GetNumLocations(), 0, "Set our third breakpoint"
        )

        if bkpt_modifier is None:
            process.Continue()
            self.assertState(
                process.GetState(), lldb.eStateStopped, "We didn't stop for the load"
            )
            self.assertEqual(
                backstop_bkpt_2.GetHitCount(), 0, "Hit our backstop breakpoint"
            )
            self.assertStopReason(
                thread.stop_reason,
                lldb.eStopReasonBreakpoint,
                "We attributed the stop to the breakpoint",
            )
            self.assertEqual(
                load_bkpt.GetHitCount(), 1, "We hit our breakpoint at the load address"
            )
        else:
            bkpt_modifier(load_bkpt)
            process.Continue()
            self.assertState(process.GetState(), lldb.eStateStopped, "We didn't stop")
            self.assertTrue(thread.IsValid(), "Our thread was no longer valid.")
            self.assertStopReason(
                thread.stop_reason,
                lldb.eStopReasonBreakpoint,
                "We didn't hit some breakpoint",
            )
            self.assertEqual(
                backstop_bkpt_2.GetHitCount(), 1, "We continued to the right breakpoint"
            )
