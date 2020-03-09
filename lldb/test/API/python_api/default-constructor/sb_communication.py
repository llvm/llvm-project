"""
Fuzz tests an object after the default construction to make sure it does not crash lldb.
"""

import lldb


def fuzz_obj(obj):
    broadcaster = obj.GetBroadcaster()
    # Do fuzz testing on the broadcaster obj, it should not crash lldb.
    import sb_broadcaster
    sb_broadcaster.fuzz_obj(broadcaster)
    obj.AdoptFileDesriptor(0, False)
    obj.AdoptFileDesriptor(1, False)
    obj.AdoptFileDesriptor(2, False)
    obj.Connect("file:/tmp/myfile")
    obj.Connect(None)
    obj.Disconnect()
    obj.IsConnected()
    obj.GetCloseOnEOF()
    obj.SetCloseOnEOF(True)
    obj.SetCloseOnEOF(False)
    #obj.Write(None, sys.maxint, None)
    #obj.Read(None, sys.maxint, 0xffffffff, None)
    obj.ReadThreadStart()
    obj.ReadThreadStop()
    obj.ReadThreadIsRunning()
    obj.SetReadThreadBytesReceivedCallback(None, None)
