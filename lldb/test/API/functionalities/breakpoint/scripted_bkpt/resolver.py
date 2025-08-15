import lldb


class Resolver:
    got_files = 0
    func_list = []

    def __init__(self, bkpt, extra_args, dict):
        self.bkpt = bkpt
        self.extra_args = extra_args

        Resolver.func_list = []
        Resolver.got_files = 0

    def __callback__(self, sym_ctx):
        print("Resolver callback called")
        sym_name = "not_a_real_function_name"
        sym_item = self.extra_args.GetValueForKey("symbol")
        if sym_item.IsValid():
            sym_name = sym_item.GetStringValue(1000)
        else:
            print("Didn't have a value for key 'symbol'")

        if sym_ctx.compile_unit.IsValid():
            Resolver.got_files = 1
        else:
            Resolver.got_files = 2

        if sym_ctx.function.IsValid():
            Resolver.got_files = 3
            func_name = sym_ctx.function.GetName()
            Resolver.func_list.append(func_name)
            if sym_name == func_name:
                self.bkpt.AddLocation(sym_ctx.function.GetStartAddress())
            return

        if sym_ctx.module.IsValid():
            print(f"Looking for {sym_name}")
            sym = sym_ctx.module.FindSymbol(sym_name, lldb.eSymbolTypeCode)
            if sym.IsValid():
                print(f"Adding location at {sym.GetStartAddress()} to {self.bkpt}")
                self.bkpt.AddLocation(sym.GetStartAddress())
                print(f"After addition: {self.bkpt}")
            else:
                print("Didn't find it, however...")

        print(f"GotFiles: {Resolver.got_files}")
        for func in Resolver.func_list:
            print(f"Function: func")

    def get_short_help(self):
        return "I am a python breakpoint resolver"

class ResolverModuleDepth(Resolver):
    def __get_depth__(self):
        return lldb.eSearchDepthModule

    def __callback__(self, sym_ctx):
        print(f"About to call the Resolver callback for {self.bkpt}")
        Resolver.__callback__(self, sym_ctx)
        print("Called the callback for ResolverModuleDepth")


class ResolverCUDepth(Resolver):
    def __get_depth__(self):
        return lldb.eSearchDepthCompUnit

    def __callback__(self, sym_ctx):
        Resolver.__callback__(self, sym_ctx)


class ResolverFuncDepth(Resolver):
    def __get_depth__(self):
        return lldb.eSearchDepthFunction

    def __callback__(self, sym_ctx):
        Resolver.__callback__(self, sym_ctx)


class ResolverBadDepth(Resolver):
    def __get_depth__(self):
        return lldb.kLastSearchDepthKind + 1

    def __callback__(self, sym_ctx):
        Resolver.__callback__(self, sym_ctx)
