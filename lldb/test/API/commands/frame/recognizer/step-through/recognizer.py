# encoding: utf-8

import lldb
from lldb.plugins.scripted_thread_plan import ScriptedThreadPlan
from lldb.plugins.scripted_stackframe_recognizer import ScriptedStackFrameRecognizer


class NestedFrameRecognizer(ScriptedStackFrameRecognizer):
    """Exercises the step-through feature of frame recognizers."""

    def get_step_through_plan(self, thread):
        """Step through from baz to bar"""
        frame = thread.frames[0]
        print(f"Asked to step through at {frame.name}")
        if frame.name.startswith("baz"):
            target = thread.process.target
            bar_funcs = target.FindFunctions("bar", lldb.eFunctionNameTypeFull)
            if bar_funcs.GetSize() == 0:
                print("Found no functions matching bar")
                return None
            if bar_funcs.GetSize() != 1:
                print("Found more than one function matching bar")
                return None
            bar_func = bar_funcs.functions[0]
            address = bar_func.addr
            if not address.IsValid():
                print("Didn't get a valid address for bar")
                return None

            load_addr = address.GetLoadAddress(target)

            dict = {
                "class_name": "recognizer.StepThrough",
                "extra_args": {"address": str(load_addr)},
            }

            return dict


class StepThrough(ScriptedThreadPlan):
    def __init__(
        self, thread_plan: lldb.SBThreadPlan, extra_args: lldb.SBStructuredData
    ):
        super().__init__(thread_plan)

        target = thread_plan.GetThread().process.target

        addr_val = extra_args.GetValueForKey("address")
        if not addr_val.IsValid():
            print("Missing addr_val key")
            thread_plan.SetPlanComplete(False)
            return

        strm = lldb.SBStream()
        addr_val.GetDescription(strm)
        addr_str = addr_val.GetStringValue(32)
        addr_int = int(addr_str)
        if addr_int == 0:
            print("Got zero value for addr_int")
            thread_plan.SetPlanComplete(False)
            return

        address = lldb.SBAddress(addr_int, target)
        if not address.IsValid():
            print("Got invalid value for address")
            thread_plan.SetPlanComplete(False)
            return

        error = lldb.SBError()
        self.addr_plan = thread_plan.QueueThreadPlanForRunToAddress(address, error)
        if error.Fail():
            print(f"Couldn't queue run to address plan: {error.description}")
            thread_plan.SetPlanComplete(False)
            return

    def explains_stop(self, event: lldb.SBEvent):
        if self.addr_plan.IsPlanComplete():
            self.thread_plan.SetPlanComplete(True)
            return True
        else:
            return False

    def should_stop(self):
        return self.thread_plan.IsPlanComplete()

    def should_step(self):
        return False

    def stop_description(self, stream: lldb.SBStream):
        stream.Print("stepped through from baz to bar")
