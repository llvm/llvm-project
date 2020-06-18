import lldb

def test(debugger, command, result, internal_dict):
    return int(lldb.SBAddress())

def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand('command script add -f sbaddress.test test')
