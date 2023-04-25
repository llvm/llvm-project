# Usage:
# ./bin/lldb $LLVM/lldb/test/API/functionalities/interactive_scripted_process/main \
#   -o "br set -p 'Break here'" \
#   -o "command script import $LLVM/lldb/test/API/functionalities/interactive_scripted_process/interactive_scripted_process.py" \
#   -o "create_mux" \
#   -o "create_sub" \
#   -o "br set -p 'also break here'" -o 'continue'

import os, json, struct, signal, tempfile

from threading import Thread
from typing import Any, Dict

import lldb
from lldb.plugins.scripted_process import ScriptedProcess
from lldb.plugins.scripted_process import ScriptedThread


class PassthruScriptedProcess(ScriptedProcess):
    driving_target = None
    driving_process = None

    def __init__(
        self,
        exe_ctx: lldb.SBExecutionContext,
        args: lldb.SBStructuredData,
        launched_driving_process: bool = True,
    ):
        super().__init__(exe_ctx, args)

        self.driving_target = None
        self.driving_process = None

        self.driving_target_idx = args.GetValueForKey("driving_target_idx")
        if self.driving_target_idx and self.driving_target_idx.IsValid():
            if self.driving_target_idx.GetType() == lldb.eStructuredDataTypeInteger:
                idx = self.driving_target_idx.GetIntegerValue(42)
            if self.driving_target_idx.GetType() == lldb.eStructuredDataTypeString:
                idx = int(self.driving_target_idx.GetStringValue(100))
            self.driving_target = self.target.GetDebugger().GetTargetAtIndex(idx)

            if launched_driving_process:
                self.driving_process = self.driving_target.GetProcess()
                for driving_thread in self.driving_process:
                    structured_data = lldb.SBStructuredData()
                    structured_data.SetFromJSON(
                        json.dumps(
                            {
                                "driving_target_idx": idx,
                                "thread_idx": driving_thread.GetIndexID(),
                            }
                        )
                    )

                    self.threads[driving_thread.GetThreadID()] = PassthruScriptedThread(
                        self, structured_data
                    )

                for module in self.driving_target.modules:
                    path = module.file.fullpath
                    load_addr = module.GetObjectFileHeaderAddress().GetLoadAddress(
                        self.driving_target
                    )
                    self.loaded_images.append({"path": path, "load_addr": load_addr})

    def get_memory_region_containing_address(
        self, addr: int
    ) -> lldb.SBMemoryRegionInfo:
        mem_region = lldb.SBMemoryRegionInfo()
        error = self.driving_process.GetMemoryRegionInfo(addr, mem_region)
        if error.Fail():
            return None
        return mem_region

    def read_memory_at_address(
        self, addr: int, size: int, error: lldb.SBError
    ) -> lldb.SBData:
        data = lldb.SBData()
        bytes_read = self.driving_process.ReadMemory(addr, size, error)

        if error.Fail():
            return data

        data.SetDataWithOwnership(
            error,
            bytes_read,
            self.driving_target.GetByteOrder(),
            self.driving_target.GetAddressByteSize(),
        )

        return data

    def write_memory_at_address(
        self, addr: int, data: lldb.SBData, error: lldb.SBError
    ) -> int:
        return self.driving_process.WriteMemory(
            addr, bytearray(data.uint8.all()), error
        )

    def get_process_id(self) -> int:
        return 42

    def is_alive(self) -> bool:
        return True

    def get_scripted_thread_plugin(self) -> str:
        return f"{PassthruScriptedThread.__module__}.{PassthruScriptedThread.__name__}"


class MultiplexedScriptedProcess(PassthruScriptedProcess):
    def __init__(self, exe_ctx: lldb.SBExecutionContext, args: lldb.SBStructuredData):
        super().__init__(exe_ctx, args)
        self.multiplexer = None
        if isinstance(self.driving_process, lldb.SBProcess) and self.driving_process:
            parity = args.GetValueForKey("parity")
            # TODO: Change to Walrus operator (:=) with oneline if assignment
            # Requires python 3.8
            val = extract_value_from_structured_data(parity, 0)
            if val is not None:
                self.parity = val

            # Turn PassThruScriptedThread into MultiplexedScriptedThread
            for thread in self.threads.values():
                thread.__class__ = MultiplexedScriptedThread

    def get_process_id(self) -> int:
        return self.parity + 420

    def launch(self, should_stop: bool = True) -> lldb.SBError:
        self.first_launch = True
        return lldb.SBError()

    def resume(self, should_stop: bool) -> lldb.SBError:
        if self.first_launch:
            self.first_launch = False
            return super().resume()
        else:
            if not self.multiplexer:
                error = lldb.SBError("Multiplexer is not set.")
                return error
            return self.multiplexer.resume(should_stop)

    def get_threads_info(self) -> Dict[int, Any]:
        if not self.multiplexer:
            return super().get_threads_info()
        filtered_threads = self.multiplexer.get_threads_info(pid=self.get_process_id())
        # Update the filtered thread class from PassthruScriptedThread to MultiplexedScriptedThread
        return dict(
            map(
                lambda pair: (pair[0], MultiplexedScriptedThread(pair[1])),
                filtered_threads.items(),
            )
        )

    def create_breakpoint(self, addr, error, pid=None):
        if not self.multiplexer:
            error.SetErrorString("Multiplexer is not set.")
        return self.multiplexer.create_breakpoint(addr, error, self.get_process_id())

    def get_scripted_thread_plugin(self) -> str:
        return f"{MultiplexedScriptedThread.__module__}.{MultiplexedScriptedThread.__name__}"


class PassthruScriptedThread(ScriptedThread):
    def __init__(self, process, args):
        super().__init__(process, args)
        driving_target_idx = args.GetValueForKey("driving_target_idx")
        thread_idx = args.GetValueForKey("thread_idx")

        # TODO: Change to Walrus operator (:=) with oneline if assignment
        # Requires python 3.8
        val = extract_value_from_structured_data(thread_idx, 0)
        if val is not None:
            self.idx = val

        self.driving_target = None
        self.driving_process = None
        self.driving_thread = None

        # TODO: Change to Walrus operator (:=) with oneline if assignment
        # Requires python 3.8
        val = extract_value_from_structured_data(driving_target_idx, 42)
        if val is not None:
            self.driving_target = self.target.GetDebugger().GetTargetAtIndex(val)
            self.driving_process = self.driving_target.GetProcess()
            self.driving_thread = self.driving_process.GetThreadByIndexID(self.idx)

        if self.driving_thread:
            self.id = self.driving_thread.GetThreadID()

    def get_thread_id(self) -> int:
        return self.id

    def get_name(self) -> str:
        return f"{PassthruScriptedThread.__name__}.thread-{self.idx}"

    def get_stop_reason(self) -> Dict[str, Any]:
        stop_reason = {"type": lldb.eStopReasonInvalid, "data": {}}

        if (
            self.driving_thread
            and self.driving_thread.IsValid()
            and self.get_thread_id() == self.driving_thread.GetThreadID()
        ):
            stop_reason["type"] = lldb.eStopReasonNone

            if self.driving_thread.GetStopReason() != lldb.eStopReasonNone:
                if "arm64" in self.scripted_process.arch:
                    stop_reason["type"] = lldb.eStopReasonException
                    stop_reason["data"][
                        "desc"
                    ] = self.driving_thread.GetStopDescription(100)
                elif self.scripted_process.arch == "x86_64":
                    stop_reason["type"] = lldb.eStopReasonSignal
                    stop_reason["data"]["signal"] = signal.SIGTRAP
                else:
                    stop_reason["type"] = self.driving_thread.GetStopReason()

        return stop_reason

    def get_register_context(self) -> str:
        if not self.driving_thread or self.driving_thread.GetNumFrames() == 0:
            return None
        frame = self.driving_thread.GetFrameAtIndex(0)

        GPRs = None
        registerSet = frame.registers  # Returns an SBValueList.
        for regs in registerSet:
            if "general purpose" in regs.name.lower():
                GPRs = regs
                break

        if not GPRs:
            return None

        for reg in GPRs:
            self.register_ctx[reg.name] = int(reg.value, base=16)

        return struct.pack(f"{len(self.register_ctx)}Q", *self.register_ctx.values())


class MultiplexedScriptedThread(PassthruScriptedThread):
    def get_name(self) -> str:
        parity = "Odd" if self.scripted_process.parity % 2 else "Even"
        return f"{parity}{MultiplexedScriptedThread.__name__}.thread-{self.idx}"


class MultiplexerScriptedProcess(PassthruScriptedProcess):
    listener = None
    multiplexed_processes = None

    def wait_for_driving_process_to_stop(self):
        def handle_process_state_event():
            # Update multiplexer process
            log("Updating interactive scripted process threads")
            dbg = self.driving_target.GetDebugger()
            log("Clearing interactive scripted process threads")
            self.threads.clear()
            for driving_thread in self.driving_process:
                log(f"{len(self.threads)} New thread {hex(driving_thread.id)}")
                structured_data = lldb.SBStructuredData()
                structured_data.SetFromJSON(
                    json.dumps(
                        {
                            "driving_target_idx": dbg.GetIndexOfTarget(
                                self.driving_target
                            ),
                            "thread_idx": driving_thread.GetIndexID(),
                        }
                    )
                )

                self.threads[driving_thread.GetThreadID()] = PassthruScriptedThread(
                    self, structured_data
                )

            mux_process = self.target.GetProcess()
            mux_process.ForceScriptedState(lldb.eStateRunning)
            mux_process.ForceScriptedState(lldb.eStateStopped)

            for child_process in self.multiplexed_processes.values():
                child_process.ForceScriptedState(lldb.eStateRunning)
                child_process.ForceScriptedState(lldb.eStateStopped)

        event = lldb.SBEvent()
        while True:
            if self.listener.WaitForEvent(1, event):
                event_mask = event.GetType()
                if event_mask & lldb.SBProcess.eBroadcastBitStateChanged:
                    state = lldb.SBProcess.GetStateFromEvent(event)
                    log(f"Received public process state event: {state}")
                    if state == lldb.eStateStopped:
                        # If it's a stop event, iterate over the driving process
                        # thread, looking for a breakpoint stop reason, if internal
                        # continue.
                        handle_process_state_event()
            else:
                continue

    def __init__(self, exe_ctx: lldb.SBExecutionContext, args: lldb.SBStructuredData):
        super().__init__(exe_ctx, args, launched_driving_process=False)
        if isinstance(self.driving_target, lldb.SBTarget) and self.driving_target:
            self.listener = lldb.SBListener(
                "lldb.listener.multiplexer-scripted-process"
            )
            self.multiplexed_processes = {}

            # Copy breakpoints from real target to passthrough
            with tempfile.NamedTemporaryFile() as tf:
                bkpt_file = lldb.SBFileSpec(tf.name)
                error = self.driving_target.BreakpointsWriteToFile(bkpt_file)
                if error.Fail():
                    log(
                        "Failed to save breakpoints from driving target (%s)"
                        % error.GetCString()
                    )
                bkpts_list = lldb.SBBreakpointList(self.target)
                error = self.target.BreakpointsCreateFromFile(bkpt_file, bkpts_list)
                if error.Fail():
                    log(
                        "Failed create breakpoints from driving target \
                        (bkpt file: %s)"
                        % tf.name
                    )

            # Copy breakpoint from passthrough to real target
            if error.Success():
                self.driving_target.DeleteAllBreakpoints()
                for bkpt in self.target.breakpoints:
                    if bkpt.IsValid():
                        for bl in bkpt:
                            real_bpkt = self.driving_target.BreakpointCreateBySBAddress(
                                bl.GetAddress()
                            )
                            if not real_bpkt.IsValid():
                                log(
                                    "Failed to set breakpoint at address %s in \
                                    driving target"
                                    % hex(bl.GetLoadAddress())
                                )

            self.listener_thread = Thread(
                target=self.wait_for_driving_process_to_stop, daemon=True
            )
            self.listener_thread.start()

    def launch(self, should_stop: bool = True) -> lldb.SBError:
        if not self.driving_target:
            return lldb.SBError(
                f"{self.__class__.__name__}.resume: Invalid driving target."
            )

        if self.driving_process:
            return lldb.SBError(
                f"{self.__class__.__name__}.resume: Invalid driving process."
            )

        error = lldb.SBError()
        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetListener(self.listener)
        driving_process = self.driving_target.Launch(launch_info, error)

        if not driving_process or error.Fail():
            return error

        self.driving_process = driving_process

        for module in self.driving_target.modules:
            path = module.file.fullpath
            load_addr = module.GetObjectFileHeaderAddress().GetLoadAddress(
                self.driving_target
            )
            self.loaded_images.append({"path": path, "load_addr": load_addr})

        self.first_resume = True
        return error

    def resume(self, should_stop: bool = True) -> lldb.SBError:
        if self.first_resume:
            # When we resume the multiplexer process for the first time,
            # we shouldn't do anything because lldb's execution machinery
            # will resume the driving process by itself.

            # Also, no need to update the multiplexer scripted process state
            # here because since it's listening for the real process stop events.
            # Once it receives the stop event from the driving process,
            # `wait_for_driving_process_to_stop` will update the multiplexer
            # state for us.

            self.first_resume = False
            return lldb.SBError()

        if not self.driving_process:
            return lldb.SBError(
                f"{self.__class__.__name__}.resume: Invalid driving process."
            )

        return self.driving_process.Continue()

    def get_threads_info(self, pid: int = None) -> Dict[int, Any]:
        if not pid:
            return super().get_threads_info()
        parity = pid % 2
        return dict(filter(lambda pair: pair[0] % 2 == parity, self.threads.items()))

    def create_breakpoint(self, addr, error, pid=None):
        if not self.driving_target:
            error.SetErrorString("%s has no driving target." % self.__class__.__name__)
            return False

        def create_breakpoint_with_name(target, load_addr, name, error):
            addr = lldb.SBAddress(load_addr, target)
            if not addr.IsValid():
                error.SetErrorString("Invalid breakpoint address %s" % hex(load_addr))
                return False
            bkpt = target.BreakpointCreateBySBAddress(addr)
            if not bkpt.IsValid():
                error.SetErrorString(
                    "Failed to create breakpoint at address %s"
                    % hex(addr.GetLoadAddress())
                )
                return False
            error = bkpt.AddNameWithErrorHandling(name)
            return error.Success()

        name = (
            "multiplexer_scripted_process"
            if not pid
            else f"multiplexed_scripted_process_{pid}"
        )

        if pid is not None:
            # This means that this method has been called from one of the
            # multiplexed scripted process. That also means that the multiplexer
            # target doesn't have this breakpoint created.
            mux_error = lldb.SBError()
            bkpt = create_breakpoint_with_name(self.target, addr, name, mux_error)
            if mux_error.Fail():
                error.SetError(
                    "Failed to create breakpoint in multiplexer \
                               target: %s"
                    % mux_error.GetCString()
                )
                return False
        return create_breakpoint_with_name(self.driving_target, addr, name, error)


def multiplex(mux_process, muxed_process):
    muxed_process.GetScriptedImplementation().multiplexer = (
        mux_process.GetScriptedImplementation()
    )
    mux_process.GetScriptedImplementation().multiplexed_processes[
        muxed_process.GetProcessID()
    ] = muxed_process


def launch_scripted_process(target, class_name, dictionary):
    structured_data = lldb.SBStructuredData()
    structured_data.SetFromJSON(json.dumps(dictionary))

    launch_info = lldb.SBLaunchInfo(None)
    launch_info.SetProcessPluginName("ScriptedProcess")
    launch_info.SetScriptedProcessClassName(class_name)
    launch_info.SetScriptedProcessDictionary(structured_data)

    error = lldb.SBError()
    return target.Launch(launch_info, error)


def duplicate_target(driving_target):
    error = lldb.SBError()
    exe = driving_target.executable.fullpath
    triple = driving_target.triple
    debugger = driving_target.GetDebugger()
    return debugger.CreateTargetWithFileAndTargetTriple(exe, triple)


def extract_value_from_structured_data(data, default_val):
    if data and data.IsValid():
        if data.GetType() == lldb.eStructuredDataTypeInteger:
            return data.GetIntegerValue(default_val)
        if data.GetType() == lldb.eStructuredDataTypeString:
            return int(data.GetStringValue(100))
    return default_val


def create_mux_process(debugger, command, exe_ctx, result, dict):
    if not debugger.GetNumTargets() > 0:
        return result.SetError(
            "Interactive scripted processes requires one non scripted process."
        )

    debugger.SetAsync(True)

    driving_target = debugger.GetSelectedTarget()
    if not driving_target:
        return result.SetError("Driving target is invalid")

    # Create a seconde target for the multiplexer scripted process
    mux_target = duplicate_target(driving_target)
    if not mux_target:
        return result.SetError(
            "Couldn't duplicate driving target to launch multiplexer scripted process"
        )

    class_name = f"{__name__}.{MultiplexerScriptedProcess.__name__}"
    dictionary = {"driving_target_idx": debugger.GetIndexOfTarget(driving_target)}
    mux_process = launch_scripted_process(mux_target, class_name, dictionary)
    if not mux_process:
        return result.SetError("Couldn't launch multiplexer scripted process")


def create_child_processes(debugger, command, exe_ctx, result, dict):
    if not debugger.GetNumTargets() >= 2:
        return result.SetError("Scripted Multiplexer process not setup")

    debugger.SetAsync(True)

    # Create a seconde target for the multiplexer scripted process
    mux_target = debugger.GetSelectedTarget()
    if not mux_target:
        return result.SetError("Couldn't get multiplexer scripted process target")
    mux_process = mux_target.GetProcess()
    if not mux_process:
        return result.SetError("Couldn't get multiplexer scripted process")

    driving_target = mux_process.GetScriptedImplementation().driving_target
    if not driving_target:
        return result.SetError("Driving target is invalid")

    # Create a target for the multiplexed even scripted process
    even_target = duplicate_target(driving_target)
    if not even_target:
        return result.SetError(
            "Couldn't duplicate driving target to launch multiplexed even scripted process"
        )

    class_name = f"{__name__}.{MultiplexedScriptedProcess.__name__}"
    dictionary = {"driving_target_idx": debugger.GetIndexOfTarget(mux_target)}
    dictionary["parity"] = 0
    even_process = launch_scripted_process(even_target, class_name, dictionary)
    if not even_process:
        return result.SetError("Couldn't launch multiplexed even scripted process")
    multiplex(mux_process, even_process)

    # Create a target for the multiplexed odd scripted process
    odd_target = duplicate_target(driving_target)
    if not odd_target:
        return result.SetError(
            "Couldn't duplicate driving target to launch multiplexed odd scripted process"
        )

    dictionary["parity"] = 1
    odd_process = launch_scripted_process(odd_target, class_name, dictionary)
    if not odd_process:
        return result.SetError("Couldn't launch multiplexed odd scripted process")
    multiplex(mux_process, odd_process)


def log(message):
    # FIXME: For now, we discard the log message until we can pass it to an lldb
    # logging channel.
    should_log = False
    if should_log:
        print(message)


def __lldb_init_module(dbg, dict):
    dbg.HandleCommand(
        "command script add -o -f interactive_scripted_process.create_mux_process create_mux"
    )
    dbg.HandleCommand(
        "command script add -o -f interactive_scripted_process.create_child_processes create_sub"
    )
