import time
import lldb


@lldb.command(command_name="busy-loop")
def busy_loop(debugger, command, exe_ctx, result, internal_dict):
    """Test helper as a busy loop."""
    if not command:
        command = "10"
    count = int(command)
    print("Starting loop...", count)
    for i in range(count):
        if debugger.InterruptRequested():
            print("interrupt requested, stopping loop", i)
            break
        print("No interrupted requested, sleeping", i)
        time.sleep(1)
