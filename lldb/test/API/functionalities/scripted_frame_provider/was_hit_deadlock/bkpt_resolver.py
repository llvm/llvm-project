"""
Scripted breakpoint resolver whose was_hit callback calls EvaluateExpression.

This reproduces the deadlock seen in the sample where:
1. BreakpointResolverScripted::WasHit runs on the private state thread
2. was_hit calls SBFrame.EvaluateExpression -> RunThreadPlan -> Halt ->
   WaitForProcessToStop (holds a mutex, waits for state event)
3. The override private state thread handles the stop and loads a scripted
   frame provider
4. The provider's __init__ calls SBThread.__bool__ -> GetStoppedExecutionContext
   -> tries to acquire the same mutex -> DEADLOCK
"""

import lldb


class ExprEvalResolver:
    """Scripted breakpoint resolver that evaluates an expression in was_hit."""

    def __init__(self, bkpt, extra_args, dict):
        self.bkpt = bkpt
        sym_name = extra_args.GetValueForKey("symbol").GetStringValue(100)
        self.sym_name = sym_name
        self.facade_loc = None

    def __callback__(self, sym_ctx):
        sym = sym_ctx.module.FindSymbol(self.sym_name, lldb.eSymbolTypeCode)
        if sym.IsValid():
            self.bkpt.AddLocation(sym.GetStartAddress())
            self.facade_loc = self.bkpt.AddFacadeLocation()

    def get_short_help(self):
        return f"ExprEvalResolver for {self.sym_name}"

    def was_hit(self, frame, bp_loc):
        # This runs on the private state thread. Calling EvaluateExpression
        # here triggers RunThreadPlan -> Halt -> WaitForProcessToStop, which
        # holds a mutex and waits for a state change event.
        options = lldb.SBExpressionOptions()
        options.SetStopOthers(True)
        options.SetTryAllThreads(False)

        result = frame.EvaluateExpression("increment()", options)
        if not result.error.success:
            return lldb.LLDB_INVALID_BREAK_ID

        return self.facade_loc
