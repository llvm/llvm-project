import lldb


def write_ok(frame, bp_loc, internal_dict):
    """Write "OK" into the buffer pointed to by 'buf' and auto-continue."""
    buf = frame.FindVariable("buf")
    addr = buf.GetValueAsUnsigned()
    if addr != 0:
        error = lldb.SBError()
        frame.GetThread().GetProcess().WriteMemory(addr, b"OK", error)
    return False
