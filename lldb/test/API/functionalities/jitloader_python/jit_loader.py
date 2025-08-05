import lldb

# Keep a global map of process ID to JITLoaderPlugin instances for the
# breakpoint callback function "jit_breakpoint_callback" below.
pid_to_jit_loader = {}


JIT_NOACTION = 0
JIT_LOAD = 1
JIT_UNLOAD = 2


def jit_breakpoint_callback(
    frame: lldb.SBFrame,
    bp_loc: lldb.SBBreakpointLocation,
    extra_args: lldb.SBStructuredData,
    dict,
):
    pid = frame.thread.process.GetProcessID()
    jit_loader = pid_to_jit_loader[pid]
    jit_loader.breakpoint_callback(frame, bp_loc, extra_args, dict)
    return False


class JITLoaderPlugin:
    def __init__(self, process: lldb.SBProcess):
        global pid_to_jit_loader
        self.process = process
        self.bp = None
        pid_to_jit_loader[process.GetProcessID()] = self

    def breakpoint_callback(
        self,
        frame: lldb.SBFrame,
        bp_loc: lldb.SBBreakpointLocation,
        extra_args: lldb.SBStructuredData,
        dict,
    ):
        thread = frame.thread
        process = thread.process
        target = process.target
        entry = frame.EvaluateExpression("entry")
        if entry.GetError().Fail():
            return
        action = frame.EvaluateExpression("action")
        if action.GetError().Fail():
            return
        action_value = action.unsigned
        if action_value == JIT_NOACTION:
            return
        # Get path from summary, but trim off the double quotes
        path = entry.member["path"].GetSummary()[1:-1]
        if action_value == JIT_LOAD:
            module_spec = lldb.SBModuleSpec()
            module_spec.SetFileSpec(lldb.SBFileSpec(path, False))
            module = lldb.SBModule(module_spec)
            if module.IsValid():
                target.AddModule(module)
                load_addr = entry.member["address"].unsigned
                target.SetModuleLoadAddress(module, load_addr)
        elif action_value == JIT_UNLOAD:
            module = target.module[path]
            if module.IsValid():
                target.ClearModuleLoadAddress(module)
                target.RemoveModule(module)

    def did_attach_or_launch(self):
        self.bp = self.process.target.BreakpointCreateByName("jit_module_action")
        quaklified_callback_name = f"{__name__}.jit_breakpoint_callback"
        error = self.bp.SetScriptCallbackFunction(quaklified_callback_name)

    def did_attach(self):
        self.did_attach_or_launch()

    def did_launch(self):
        self.did_attach_or_launch()

    def module_did_load(self, module: lldb.SBModule):
        pass
