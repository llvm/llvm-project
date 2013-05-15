"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import sys
import lldb

def fuzz_obj(obj):
    obj.GetDataFlavor()
    obj.GetType()
    broadcaster = obj.GetBroadcaster()
    # Do fuzz testing on the broadcaster obj, it should not crash lldb.
    import sb_broadcaster
    sb_broadcaster.fuzz_obj(broadcaster)
    obj.BroadcasterMatchesRef(broadcaster)
    obj.GetDescription(lldb.SBStream())
    obj.Clear()
