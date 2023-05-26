import lldb


def StepOver(debugger, args, result, dict):
    """
    Step over a given number of times instead of only just once
    """
    arg_split = args.split(" ")
    print(type(arg_split))
    count = int(arg_split[0])
    for i in range(0, count):
        debugger.GetSelectedTarget().GetProcess().GetSelectedThread().StepOver(
            lldb.eOnlyThisThread
        )
        print("step<%d>" % i)


def __lldb_init_module(debugger, session_dict):
    # by default, --synchronicity is set to synchronous
    debugger.HandleCommand("command script add -f mysto.StepOver mysto")
    return None
