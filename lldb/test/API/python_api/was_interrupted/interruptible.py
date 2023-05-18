import lldb
import threading

local_data = None


class BarrierContainer(threading.local):
    def __init__(self, before_interrupt_barrier, after_interrupt_barrier, event):
        self.event = event
        self.before_interrupt_barrier = before_interrupt_barrier
        self.after_interrupt_barrier = after_interrupt_barrier


class WelcomeCommand(object):
    def __init__(self, debugger, session_dict):
        return

    def get_short_help(self):
        return "A command that waits for an interrupt before returning."

    def check_was_interrupted(self, debugger, use_interpreter):
        if use_interpreter:
            self.was_interrupted = debugger.GetCommandInterpreter().WasInterrupted()
        else:
            self.was_interrupted = debugger.InterruptRequested()
        if local_data.event:
            self.was_canceled = local_data.event.is_set()

    def __call__(self, debugger, args, exe_ctx, result):
        """Command arguments:
        {interp/debugger} - Whether to use SBCommandInterpreter::WasInterrupted
                            of SBDebugger::InterruptRequested().
        check - Don't do the rendevous, just check if an interrupt was requested.
                If check is not provided, we'll do the lock and then check.
        poll  - Should we poll once after the rendevous or spin waiting for the
                interruption to happen.

        For the interrupt cases, the command waits serially on the barriers
        passed to it in local data, giving the test runner a chance to set the
        interrupt.  Once the barriers are passed, it waits for the interrupt
        or the event.
        If it finds an interrupt, it returns "Command was interrupted". If it gets an
        event before seeing the interrupt it returns "Command was not interrupted."
        For the "poll" case, it waits on the rendevous, then checks once.
        For the "check" case, it doesn't wait, but just returns whether there was
        an interrupt in force or not."""

        if local_data == None:
            result.SetError("local data was not set.")
            result.SetStatus(lldb.eReturnStatusFailed)
            return

        use_interpreter = "interp" in args
        if not use_interpreter:
            if not "debugger" in args:
                result.SetError("Must pass either 'interp' or 'debugger'")
                result.SetStatus(lldb.eReturnStatusFailed)
                return

        self.was_interrupted = False
        self.was_canceled = False

        if "check" in args:
            self.check_was_interrupted(debugger, use_interpreter)
            if self.was_interrupted:
                result.Print("Command was interrupted")
            else:
                result.Print("Command was not interrupted")
        else:
            # Wait here to rendevous in the test before it sets the interrupt.
            local_data.before_interrupt_barrier.wait()
            # Now the test will set the interrupt, and we can continue:
            local_data.after_interrupt_barrier.wait()

            if "poll" in args:
                self.check_was_interrupted(debugger, use_interpreter)
            else:
                while not self.was_interrupted and not self.was_canceled:
                    self.check_was_interrupted(debugger, use_interpreter)

            if self.was_interrupted:
                result.Print("Command was interrupted")
            else:
                result.Print("Command was not interrupted")

            if self.was_canceled:
                result.Print("Command was canceled")
        result.SetStatus(lldb.eReturnStatusSuccessFinishResult)
        return True
