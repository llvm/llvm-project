"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import sys
import lldb

def fuzz_obj(obj):
    obj.AppendString("another string")
    obj.AppendString(None)
    obj.AppendList(None, 0)
    obj.AppendList(lldb.SBStringList())
    obj.GetSize()
    obj.GetStringAtIndex(0xffffffff)
    obj.Clear()
    for str in obj:
        print str
