"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import lldb


def fuzz_obj(obj):
    obj.AddEvent(lldb.SBEvent())
    obj.StartListeningForEvents(lldb.SBBroadcaster(), 0xFFFFFFFF)
    obj.StopListeningForEvents(lldb.SBBroadcaster(), 0xFFFFFFFF)
    event = lldb.SBEvent()
    broadcaster = lldb.SBBroadcaster()
    obj.WaitForEvent(5, event)
    obj.WaitForEventForBroadcaster(5, broadcaster, event)
    obj.WaitForEventForBroadcasterWithType(5, broadcaster, 0xFFFFFFFF, event)
    obj.PeekAtNextEvent(event)
    obj.PeekAtNextEventForBroadcaster(broadcaster, event)
    obj.PeekAtNextEventForBroadcasterWithType(broadcaster, 0xFFFFFFFF, event)
    obj.GetNextEvent(event)
    obj.GetNextEventForBroadcaster(broadcaster, event)
    obj.GetNextEventForBroadcasterWithType(broadcaster, 0xFFFFFFFF, event)
    obj.HandleBroadcastEvent(event)
