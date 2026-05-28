import lldb


class OverrideExample:
    def __init__(
        self, bkpt: lldb.SBBreakpoint, extra_args: lldb.SBStructuredData, dict
    ):
        self.bkpt = bkpt
        self.extra_args = extra_args
        self.set_bkpt = False
        symbol_value = extra_args.GetValueForKey("symbol")
        self.alternate_loc = symbol_value.GetStringValue(1000)

    def __callback__(self, sym_ctx: lldb.SBSymbolContext):
        """This callback only sets a breakpoint in one place,
        no matter what file and line you ask for"""
        if self.set_bkpt == True:
            return
        # FIXME: Do this better...
        alternate_sym_list = sym_ctx.module.FindFunctions(self.alternate_loc)
        if len(alternate_sym_list.symbols) == 0:
            return
        alternate_sym = alternate_sym_list.symbols[0]
        start_addr = alternate_sym.addr
        self.bkpt.AddLocation(start_addr)
        self.set_bkpt = True

    def get_short_help(self):
        return f"I am an override resolver, resolving to {self.alternate_loc}."

    def set_breakpoint(self, bkpt: lldb.SBBreakpoint):
        self.bkpt = bkpt

    def overrides_resolver(
        self, target: lldb.SBTarget, initial_resolver: lldb.SBStructuredData
    ):
        strm = lldb.SBStream()
        initial_resolver.GetAsJSON(strm)
        type = initial_resolver.GetValueForKey("Type").GetStringValue(1000)
        if type == "FileAndLine":
            return True
        return False


class TrivialExample:
    def __init__(
        self, bkpt: lldb.SBBreakpoint, extra_args: lldb.SBStructuredData, dict
    ):
        self.bkpt = bkpt
        self.extra_args = extra_args
        self.set_bkpt = False

    def __callback__(self, sym_ctx: lldb.SBSymbolContext):
        """This one's trivial, it does nothing"""
        return

    def get_short_help(self):
        return f"I am an triial resolver, doing nothing."

    def set_breakpoint(self, bkpt: lldb.SBBreakpoint):
        self.bkpt = bkpt

    def overrides_resolver(
        self, target: lldb.SBTarget, initial_resolver: lldb.SBStructuredData
    ):
        """Trivial - overrides nothing"""
        return False
