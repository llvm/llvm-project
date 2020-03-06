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
        self.frame = thread_plan.GetThread().frames[0]
        self.target = thread_plan.GetThread().GetProcess().GetTarget()
        func_entry = args_data.GetValueForKey("variable_name")
        
        if not func_entry.IsValid():
            print("Did not get a valid entry for variable_name")
        func_name = func_entry.GetStringValue(100)

        self.value = self.frame.FindVariable(func_name)
        if self.value.GetError().Fail():
            print("Failed to get foo value: %s"%(self.value.GetError().GetCString()))
        else:
            print("'foo' value: %d"%(self.value.GetValueAsUnsigned()))

        StepWithChild.__init__(self, thread_plan)


    def queue_child_thread_plan(self):
        le = self.frame.GetLineEntry()
        start_addr = le.GetStartAddress() 
        start = start_addr.GetLoadAddress(self.target)
        end = le.GetEndAddress().GetLoadAddress(self.target)
        return self.thread_plan.QueueThreadPlanForStepOverRange(start_addr,
                                                                end - start)

    def should_stop(self, event):
        if not self.child_thread_plan.IsPlanComplete():
            return False

        # If we've stepped out of this frame, stop.
        if not self.frame.IsValid():
            return True

        if not self.value.IsValid():
            return True

        print("Got next value: %d"%(self.value.GetValueAsUnsigned()))
        if not self.value.GetValueDidChange():
            self.child_thread_plan = self.queue_child_thread_plan()
            return False
        else:
            return True
