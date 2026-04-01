import lldb


def __lldb_init_module(debugger, internal_dict):
    lldb.LOADED = True
