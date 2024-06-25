import lldb


class StepWithChild:
    def __init__(self, thread_plan):
        self.thread_plan = thread_plan
        self.child_thread_plan = self.queue_child_thread_plan()

    def explains_stop(self, event):
        return False

    def should_stop(self, event):
        if not self.child_thread_plan.IsPlanComplete():
            return False

        self.thread_plan.SetPlanComplete(True)

        return True

    def should_step(self):
        return False

    def stop_description(self, stream):
        if self.child_thread_plan.IsPlanComplete():
            return self.child_thread_plan.GetDescription(stream)
        return True

    def queue_child_thread_plan(self):
        return None


class StepOut(StepWithChild):
    def __init__(self, thread_plan, dict):
        StepWithChild.__init__(self, thread_plan)

    def queue_child_thread_plan(self):
        return self.thread_plan.QueueThreadPlanForStepOut(0)


class StepScripted(StepWithChild):
    def __init__(self, thread_plan, dict):
        StepWithChild.__init__(self, thread_plan)

    def queue_child_thread_plan(self):
        return self.thread_plan.QueueThreadPlanForStepScripted("Steps.StepOut")


# This plan does a step-over until a variable changes value.
class StepUntil(StepWithChild):
    def __init__(self, thread_plan, args_data, dict):
        self.thread_plan = thread_plan
        self.frame = thread_plan.GetThread().frames[0]
        self.target = thread_plan.GetThread().GetProcess().GetTarget()
        var_entry = args_data.GetValueForKey("variable_name")

        if not var_entry.IsValid():
            print("Did not get a valid entry for variable_name")
        self.var_name = var_entry.GetStringValue(100)

        self.value = self.frame.FindVariable(self.var_name)
        if self.value.GetError().Fail():
            print("Failed to get foo value: %s" % (self.value.GetError().GetCString()))

        StepWithChild.__init__(self, thread_plan)

    def queue_child_thread_plan(self):
        le = self.frame.GetLineEntry()
        start_addr = le.GetStartAddress()
        start = start_addr.GetLoadAddress(self.target)
        end = le.GetEndAddress().GetLoadAddress(self.target)
        return self.thread_plan.QueueThreadPlanForStepOverRange(start_addr, end - start)

    def should_stop(self, event):
        if not self.child_thread_plan.IsPlanComplete():
            return False

        # If we've stepped out of this frame, stop.
        if not self.frame.IsValid():
            self.thread_plan.SetPlanComplete(True)
            return True

        if not self.value.IsValid():
            self.thread_plan.SetPlanComplete(True)
            return True

        if not self.value.GetValueDidChange():
            self.child_thread_plan = self.queue_child_thread_plan()
            return False
        else:
            self.thread_plan.SetPlanComplete(True)
            return True

    def stop_description(self, stream):
        stream.Print(f"Stepped until {self.var_name} changed.")


# This plan does nothing, but sets stop_mode to the
# value of GetStopOthers for this plan.
class StepReportsStopOthers:
    stop_mode_dict = {}

    def __init__(self, thread_plan, args_data, dict):
        self.thread_plan = thread_plan
        self.key = str(args_data.GetValueForKey("token").GetUnsignedIntegerValue(1000))

    def should_stop(self, event):
        self.thread_plan.SetPlanComplete(True)
        StepReportsStopOthers.stop_mode_dict[
            self.key
        ] = self.thread_plan.GetStopOthers()
        return True

    def should_step(self):
        return True

    def explains_stop(self, event):
        return True
