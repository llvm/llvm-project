import lldb
import time

class StopHook:
    # These dictionaries are used to pass data back to the test case.
    # Since these are global, we need to know which test run is which.
    # The test passes a key in the extra_args, we use that as the key
    # for these dictionaries, and then the test can fetch out the right
    # one.
    counter = {}
    non_stops = {}
    def __init__(self, target, extra_args, dict):
        self.target = target
        self.regs = {}
        self.instance = extra_args.GetValueForKey("instance").GetStringValue(100)
        StopHook.counter[self.instance] = 0
        StopHook.non_stops[self.instance] = 0
        
    def handle_stop(self, exe_ctx, stream):
        import time
        # All this stop hook does is sleep a bit and count.  There was a bug
        # where we were sending the secondary listener events when the
        # private state thread's DoOnRemoval completed, rather than when
        # the primary public process Listener consumes the event.  That
        # became really clear when a stop hook artificially delayed the
        # delivery of the primary listener's event - since IT had to come
        # after the stop hook ran.
        time.sleep(0.5)
        StopHook.counter[self.instance] += 1
        # When we were sending events too early, one symptom was the stop
        # event would get triggered before the state had been changed.
        # Watch for that here.
        if exe_ctx.process.GetState() != lldb.eStateStopped:
            StopHook.non_stops[self.instance] += 1

