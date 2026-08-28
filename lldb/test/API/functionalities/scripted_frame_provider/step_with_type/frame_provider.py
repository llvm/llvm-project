"""
This frame provider just adds a step plan.  We use it to ensure that the 
scripted frames can change the step behavior of the frame.
"""

import lldb
import struct

from lldb.plugins.scripted_process import ScriptedFrame
from lldb.plugins.scripted_frame_provider import ScriptedFrameProvider
from lldb.plugins.scripted_thread_plan import ScriptedThreadPlan

class BaseStepFrame(ScriptedFrame):
    """A frame that wraps a real frame but changes the meaning of stepping."""

    def __init__(self, thread, orig_frame, idx):
        args = lldb.SBStructuredData()
        super().__init__(thread, args)

        self.orig_frame = orig_frame
        self.idx = idx

    def get_id(self):
        return self.idx

    def is_artificial(self):
        # lldb won't step out to artificial frames, but these
        # mirror real frames so they aren't artificial.
        return False

    def get_cfa(self):
        return self.orig_frame.GetCFA()
        
    def get_pc(self):
        pc = self.orig_frame.GetPC()
        return pc

    def get_symbol_context(self):
        return self.orig_frame.GetSymbolContext(lldb.eSymbolContextEverything)

    def get_function_name(self):
        return self.orig_frame.GetFunctionName() or "<wrapped>"

    def get_register_context(self):
        """Forward the wrapped frame's GPRs, packed in register_info order."""
        regs = {}
        for reg_set in self.orig_frame.registers:
            if "general purpose" in reg_set.name.lower():
                for reg in reg_set:
                    regs[reg.name] = (
                        int(reg.value, 16) if reg.value else 0,
                        reg.GetByteSize(),
                    )
                break
        if not regs:
            return None

        info = self.get_register_info()["registers"]
        
        def read(entry):
            # A register set reports a register under the name LLDB displays,
            # which can be an alias of the architectural name the register info
            # uses. The register info carries that alias in "alt-name".
            if entry["name"] in regs:
                return regs[entry["name"]]

            try:
                return regs[entry["alt-name"]]
            except KeyError:
                return 0, entry["bitsize"] // 8

        struct_format = ""
        struct_data = []
        sizes = {1: "B", 2: "H", 4: "I", 8: "Q"}

        for reg in info:
            value, size = read(reg)
            struct_format += sizes[size]
            struct_data.append(value)

        return struct.pack(struct_format, *struct_data)

class StepTypeFrame (BaseStepFrame):
    def get_plan_for_step_type(self, step_type):
        dict = {
            "class_name" : "frame_provider.StepTwice",
            "extra_args" : {"step_type" : str(step_type)},
        }
        return dict

class BadStepFrame (BaseStepFrame):
    def get_plan_for_step_type(self, step_type):
        dict = {
            "class_name" : "frame_provider.Oops",
            "extra_args" : {"step_type" : str(step_type)},
        }
        return dict

class NoStepFrame (BaseStepFrame):
    def get_plan_for_step_type(self, step_type):
        dict = {
            "class_name" : "",
            "extra_args" : {"step_type" : str(step_type)},
        }
        return dict

class BaseFrameProvider(ScriptedFrameProvider):
    """
    Provider that passes through every frame from its parent StackFrameList
    but adds a prefix to each function name.

    This verifies that the provider can freely access its input_frames
    (the parent list) without hitting circular dependencies or deadlocks.
    """

    PREFIX = "my_custom_"

    def __init__(self, input_frames, args):
        super().__init__(input_frames, args)

    @staticmethod
    def get_description():
        return "Provider that prefixes all function names with 'my_custom_'"

class CorrectStepProvider(BaseFrameProvider):
    def get_frame_at_index(self, idx):
        if idx < len(self.input_frames):
            frame = self.input_frames[idx]
            try:
                returned_frame = StepTypeFrame(self.thread, frame, idx)
            except Exception as err:
                print(f"Got err: {str(err)}")
            return returned_frame
        return None

class NoStepProvider(BaseFrameProvider):
    def get_frame_at_index(self, idx):
        if idx < len(self.input_frames):
            frame = self.input_frames[idx]
            try:
                returned_frame = NoStepFrame(self.thread, frame, idx)
            except Exception as err:
                print(f"Got err: {str(err)}")
            return returned_frame
        return None
    
class BadStepProvider(BaseFrameProvider):
    def get_frame_at_index(self, idx):
        if idx < len(self.input_frames):
            frame = self.input_frames[idx]
            try:
                returned_frame = BadStepFrame(self.thread, frame, idx)
            except Exception as err:
                print(f"Got err: {str(err)}")
            return returned_frame
        return None
    

class StepTwice(ScriptedThreadPlan):
    """ This thread plan does whatever it is asked to do twice. """
    def __init__(
            self, thread_plan: lldb.SBThreadPlan, extra_args: lldb.SBStructuredData
    ):
        super().__init__(thread_plan)
        self.counter = 1
        self.thread = self.thread_plan.GetThread()
        step_type = extra_args.GetValueForKey("step_type")
        if not step_type.IsValid():
            thread_plan.SetPlanComplete(False)
            return
        
        step_str = step_type.GetStringValue()
        self.step_val = int(step_str)
        self.queue_thread_plan()

    def queue_thread_plan(self):
        stop_frame = self.thread.frames[0]
        target = self.thread.process.target
        error = lldb.SBError()
        self.curr_plan = None

        line_entry = stop_frame.GetSymbolContext(lldb.eSymbolContextEverything).line_entry
        curr_addr = stop_frame.addr.GetLoadAddress(target)
        pc_addr = stop_frame.GetPC()

        length = line_entry.end_addr.GetLoadAddress(target) - curr_addr
        
        
        if self.step_val == lldb.eStepTypeOver:
            self.curr_plan = self.thread_plan.QueueThreadPlanForStepOverRange(stop_frame.addr, length, error)
            
        if self.step_val == lldb.eStepTypeInto:
            self.curr_plan = self.thread_plan.QueueThreadPlanForStepInRange(stop_frame.addr, length, error)

        if self.step_val == lldb.eStepTypeOut:
            self.curr_plan = self.thread_plan.QueueThreadPlanForStepOut(0, False, error)

        if self.curr_plan == None:
            print(f"Didn't make a plan for {self.step_val}")

        if error.Fail():
            print(f"Couldn't queue run plan for {step_type}: {error.description}")
            self.thread_plan.SetPlanComplete(False)
            return

    def explains_stop(self, event: lldb.SBEvent):
        if self.curr_plan.IsPlanComplete():
            return True
        else:
            return False

    def should_stop(self):
        if self.thread_plan.IsPlanComplete():
            return True
        else:
            if self.counter == 2:
                self.thread_plan.SetPlanComplete(True)
                return True
            else:
                self.counter += 1
                self.queue_thread_plan()
                return False

    def should_step(self):
        return False

    def stop_description(self, stream: lldb.SBStream):
        stream.Print("double-step engage")
