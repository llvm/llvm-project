import lldb


class RunToAddress:
    """The shape the issue reports: a scripted plan whose only job is to queue a
    run-to-address sub-plan for an address it is handed."""

    def __init__(self, thread_plan, args_data):
        self.thread_plan = thread_plan
        target = thread_plan.GetThread().GetProcess().GetTarget()
        addr = args_data.GetValueForKey("addr").GetUnsignedIntegerValue()
        self.sub_plan = thread_plan.QueueThreadPlanForRunToAddress(
            lldb.SBAddress(addr, target), lldb.SBError()
        )

    def explains_stop(self, event):
        return False

    def should_stop(self, event):
        self.thread_plan.SetPlanComplete(True)
        return True
