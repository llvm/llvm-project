"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import lldb


def fuzz_obj(obj):
    obj.GetAddress()
    obj.GetLoadAddress()
    obj.SetEnabled(True)
    obj.IsEnabled()
    obj.SetCondition("i >= 10")
    obj.GetCondition()
    obj.SetThreadID(0)
    obj.GetThreadID()
    obj.SetThreadIndex(0)
    obj.GetThreadIndex()
    obj.SetThreadName("worker thread")
    obj.GetThreadName()
    obj.SetQueueName("my queue")
    obj.GetQueueName()
    obj.IsResolved()
    obj.GetDescription(lldb.SBStream(), lldb.eDescriptionLevelVerbose)
    breakpoint = obj.GetBreakpoint()
    # Do fuzz testing on the breakpoint obj, it should not crash lldb.
    import sb_breakpoint

    sb_breakpoint.fuzz_obj(breakpoint)
