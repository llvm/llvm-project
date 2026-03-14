"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import lldb


def fuzz_obj(obj):
    obj.BroadcastEventByType(lldb.eBreakpointEventTypeInvalidType, True)
    obj.BroadcastEvent(lldb.SBEvent(), False)
    listener = lldb.SBListener("fuzz_testing")
    obj.AddInitialEventsToListener(listener, 0xFFFFFFFF)
    obj.AddInitialEventsToListener(listener, 0)
    obj.AddListener(listener, 0xFFFFFFFF)
    obj.AddListener(listener, 0)
    obj.GetName()
    obj.EventTypeHasListeners(0)
    obj.RemoveListener(listener, 0xFFFFFFFF)
    obj.RemoveListener(listener, 0)
    obj.Clear()
