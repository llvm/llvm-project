import lldb


@lldb.command("tagged-pointer-collision")
def noop(dbg, cmdstr, ctx, result, _):
    print("ran tagged-pointer-collision", file=result)
