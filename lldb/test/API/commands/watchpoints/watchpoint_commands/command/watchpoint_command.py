import lldb

num_hits = 0


def watchpoint_command(frame, wp, dict):
    global num_hits
    if num_hits == 0:
        print("I stopped the first time")
        frame.EvaluateExpression("cookie = 888")
        num_hits += 1
        return True
    if num_hits == 1:
        print("I stopped the second time, but with no return")
        frame.EvaluateExpression("cookie = 666")
        num_hits += 1
    else:
        print("I stopped the %d time" % (num_hits))
        frame.EvaluateExpression("cookie = 999")
        return False  # This cause the process to continue.
