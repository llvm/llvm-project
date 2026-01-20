import lldb


class WrapStepOver:
    def __init__(self, thread_plan, args_data):
        self.plan = thread_plan
        thread = thread_plan.GetThread()
        target = thread.GetProcess().GetTarget()
        frame_0 = thread.frames[0]
        line_entry = frame_0.line_entry
        start_addr = line_entry.addr
        end_addr = line_entry.end_addr
        range_size = end_addr.GetLoadAddress(target) - start_addr.GetLoadAddress(target)
        error = lldb.SBError()
        self.sub_plan = thread_plan.QueueThreadPlanForStepOverRange(
            start_addr, range_size
        )

    def should_step(self):
        return False

    def should_stop(self, event):
        if self.sub_plan.IsPlanComplete():
            self.plan.SetPlanComplete(True)
            return True
        else:
            return False
