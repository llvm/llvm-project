import lldb

# These count how many times the resolver override was called.
# Used to check that the masks actually work.

override_count = 0
trivial_count = 0
override_not_file = 0
trivial_not_name = 0


class CheckerCommand:
    def __init__(self, debugger, internal_dict):
        self.debugger = debugger

    def get_short_help(self):
        return "A command the checks how many times the resolvers were called"

    def __call__(self, debugger, command, exe_ctx, result):
        global override_count
        global trivial_count
        global override_not_file
        global trivial_not_name

        result.SetStatus(lldb.eReturnStatusSuccessFinishResult)
        if command == "trivial":
            result.AppendMessage(str(trivial_count))
            return
        if command == "override":
            result.AppendMessage(str(override_count))
            return
        if command == "override_not_file":
            result.AppendMessage(str(override_not_file))
            return
        if command == "trivial_not_name":
            result.AppendMessage(str(trivial_not_name))
            return

        result.AppendError(f"unknown check type: {command}")


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
        global override_count
        global override_not_file

        override_count += 1

        strm = lldb.SBStream()

        initial_resolver.GetAsJSON(strm)
        type = initial_resolver.GetValueForKey("Type").GetStringValue(1000)
        if type == "FileAndLine":
            return True
        else:
            override_not_file += 1
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
        global trivial_count
        global trivial_not_name
        trivial_count += 1

        strm = lldb.SBStream()

        initial_resolver.GetAsJSON(strm)
        type = initial_resolver.GetValueForKey("Type").GetStringValue(1000)
        if type != "SymbolName":
            trivial_not_name += 1

        """Trivial - overrides nothing"""
        return False


def __lldb_init_module(debugger, dict):
    print(f"About to run: command script add -c {__name__}.CheckerCommand checker")
    debugger.HandleCommand(f"command script add -c {__name__}.CheckerCommand checker")
