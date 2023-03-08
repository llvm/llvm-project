# Usage:
# ./bin/lldb $LLVM/lldb/test/API/functionalities/interactive_scripted_process/main \
#   -o "br set -p 'Break here'" -o "run" \
#   -o "command script import
#   $LLVM/lldb/test/API/functionalities/interactive_scripted_process/interactive_scripted_process.py" \
#   -o "br set -p 'also break here'" -o 'continue'

import os,json,struct,signal

from threading import Thread
from typing import Any, Dict

import lldb
from lldb.plugins.scripted_process import ScriptedProcess
from lldb.plugins.scripted_process import ScriptedThread

class PassthruScriptedProcess(ScriptedProcess):
    driving_target = None
    driving_process = None

    def __init__(self, exe_ctx: lldb.SBExecutionContext, args : lldb.SBStructuredData):
        super().__init__(exe_ctx, args)

        self.driving_target = None
        self.driving_process = None

        self.driving_target_idx = args.GetValueForKey("driving_target_idx")
        if (self.driving_target_idx and self.driving_target_idx.IsValid()):
            if self.driving_target_idx.GetType() == lldb.eStructuredDataTypeInteger:
                idx = self.driving_target_idx.GetIntegerValue(42)
            if self.driving_target_idx.GetType() == lldb.eStructuredDataTypeString:
                idx = int(self.driving_target_idx.GetStringValue(100))
            self.driving_target = self.target.GetDebugger().GetTargetAtIndex(idx)
            self.driving_process = self.driving_target.GetProcess()
            for driving_thread in self.driving_process:
                structured_data = lldb.SBStructuredData()
                structured_data.SetFromJSON(json.dumps({
                    "driving_target_idx" : idx,
                    "thread_idx" : driving_thread.GetIndexID()
                }))

                self.threads[driving_thread.GetThreadID()] = PassthruScriptedThread(self, structured_data)

            for module in self.driving_target.modules:
                path = module.file.fullpath
                load_addr = module.GetObjectFileHeaderAddress().GetLoadAddress(self.driving_target)
                self.loaded_images.append({"path": path, "load_addr": load_addr})

    def get_memory_region_containing_address(self, addr: int) -> lldb.SBMemoryRegionInfo:
        mem_region = lldb.SBMemoryRegionInfo()
        error = self.driving_process.GetMemoryRegionInfo(addr, mem_region)
        if error.Fail():
            return None
        return mem_region

    def read_memory_at_address(self, addr: int, size: int, error: lldb.SBError) -> lldb.SBData:
        data = lldb.SBData()
        bytes_read = self.driving_process.ReadMemory(addr, size, error)

        if error.Fail():
            return data

        data.SetDataWithOwnership(error, bytes_read,
                                  self.driving_target.GetByteOrder(),
                                  self.driving_target.GetAddressByteSize())

        return data

    def write_memory_at_address(self, addr, data, error):
        return self.driving_process.WriteMemory(addr,
                                                bytearray(data.uint8.all()),
                                                error)

    def get_loaded_images(self):
        return self.loaded_images

    def get_process_id(self) -> int:
        return 42

    def is_alive(self) -> bool:
        return True

    def get_scripted_thread_plugin(self):
        return PassthruScriptedThread.__module__ + "." + PassthruScriptedThread.__name__

class MultiplexedScriptedProcess(PassthruScriptedProcess):
    def __init__(self, exe_ctx: lldb.SBExecutionContext, args : lldb.SBStructuredData):
        super().__init__(exe_ctx, args)
        self.multiplexer = None
        if isinstance(self.driving_process, lldb.SBProcess) and self.driving_process:
            parity  = args.GetValueForKey("parity")
            #TODO: Change to Walrus operator (:=) with oneline if assignment
            # Requires python 3.8
            val = extract_value_from_structured_data(parity, 0)
            if val is not None:
                self.parity = val

            # Turn PassThruScriptedThread into MultiplexedScriptedThread
            for thread in self.threads.values():
                thread.__class__ = MultiplexedScriptedThread

    def get_process_id(self):
        return self.parity + 420

    def launch(self):
        self.first_launch = True
        return lldb.SBError()

    def resume(self, should_stop):
        if self.first_launch:
            self.first_launch = False
            return super().resume()
        else:
            if not self.multiplexer:
                error = lldb.SBError()
                error.SetErrorString("Multiplexer is not set.")
                return error
            return self.multiplexer.resume(pid=self.get_process_id())

    def get_threads_info(self):
        if not self.multiplexer:
            return super().get_threads_info()
        return self.multiplexer.get_threads_info(pid=self.get_process_id())

    def get_scripted_thread_plugin(self):
        return MultiplexedScriptedThread.__module__ + "." + MultiplexedScriptedThread.__name__

class PassthruScriptedThread(ScriptedThread):
    def __init__(self, process, args):
        super().__init__(process, args)
        driving_target_idx = args.GetValueForKey("driving_target_idx")
        thread_idx = args.GetValueForKey("thread_idx")

        #TODO: Change to Walrus operator (:=) with oneline if assignment
        # Requires python 3.8
        val = extract_value_from_structured_data(thread_idx, 0)
        if val is not None:
            self.idx = val

        self.driving_target = None
        self.driving_process = None
        self.driving_thread = None

        #TODO: Change to Walrus operator (:=) with oneline if assignment
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
        return PassthruScriptedThread.__name__ + ".thread-" + str(self.idx)

    def get_stop_reason(self) -> Dict[str, Any]:
        stop_reason = { "type": lldb.eStopReasonInvalid, "data": {  }}

        if self.driving_thread and self.driving_thread.IsValid() \
                and self.get_thread_id() == self.driving_thread.GetThreadID():
            stop_reason["type"] = lldb.eStopReasonNone

            if self.driving_thread.GetStopReason() != lldb.eStopReasonNone:
                if 'arm64' in self.scripted_process.arch:
                    stop_reason["type"] = lldb.eStopReasonException
                    stop_reason["data"]["desc"] = self.driving_thread.GetStopDescription(100)
                elif self.scripted_process.arch == 'x86_64':
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
        registerSet = frame.registers # Returns an SBValueList.
        for regs in registerSet:
            if 'general purpose' in regs.name.lower():
                GPRs = regs
                break

        if not GPRs:
            return None

        for reg in GPRs:
            self.register_ctx[reg.name] = int(reg.value, base=16)

        return struct.pack("{}Q".format(len(self.register_ctx)), *self.register_ctx.values())

class MultiplexedScriptedThread(PassthruScriptedThread):
    def get_name(self) -> str:
        parity = "Odd" if self.scripted_process.parity % 2 else "Even"
        return parity + MultiplexedScriptedThread.__name__ + ".thread-" + str(self.idx)

class MultiplexerScriptedProcess(PassthruScriptedProcess):
    listener = None
    multiplexed_processes = None

    def wait_for_driving_process_to_stop(self, originator_pid, stop_event_mask):
        event = lldb.SBEvent()

        done = False
        while not done:
            if self.listener.WaitForEvent(5, event):
                event_mask = event.GetType();
                if event.BroadcasterMatchesRef(self.driving_process.GetBroadcaster()):
                    if event_mask & lldb.SBProcess.eBroadcastBitStateChanged:
                        done = True;
            continue

        self.listener.StopListeningForEvents(self.driving_process.GetBroadcaster(),
                                             stop_event_mask)

        # Stop multiplexer process
        mux_process = self.target.GetProcess()
        mux_process.ForceScriptedState(lldb.eStateRunning);
        mux_process.ForceScriptedState(lldb.eStateStopped);

        child_process = self.multiplexed_processes[originator_pid]
        child_process.ForceScriptedState(lldb.eStateRunning);
        child_process.ForceScriptedState(lldb.eStateStopped);

    def __init__(self, exe_ctx: lldb.SBExecutionContext, args : lldb.SBStructuredData):
        super().__init__(exe_ctx, args)
        if isinstance(self.driving_process, lldb.SBProcess) and self.driving_process:
            self.listener = lldb.SBListener("lldb.listener.multiplexer-scripted-process")
            self.multiplexed_processes = {}

    def resume(self, should_stop=True, pid=None):
        if not pid or pid not in self.multiplexed_processes.keys():
            return super().resume()

        stop_event_mask = lldb.SBProcess.eBroadcastBitInterrupt | lldb.SBProcess.eBroadcastBitStateChanged
        self.listener.StartListeningForEvents(self.driving_process.GetBroadcaster(),
                                              stop_event_mask)

        listener_thread = Thread(target=self.wait_for_driving_process_to_stop,
                                 args=[pid, stop_event_mask])
        listener_thread.start()

        # Resume the driving process
        self.driving_process.Continue()

        # Update the scripted process state.
        return lldb.SBError()

    def get_threads_info(self, pid=None):
        # if not pid or pid not in self.multiplexed_processes.keys():
        if not pid:
            return super().get_threads_info()
        parity = pid % 2
        return dict(filter(lambda pair: pair[0] % 2 == parity, self.threads.items()))

def multiplex(mux_process, muxed_process):
    muxed_process.GetScriptedImplementation().multiplexer = mux_process.GetScriptedImplementation()
    mux_process.GetScriptedImplementation().multiplexed_processes[muxed_process.GetProcessID()] = muxed_process

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
    return None

def __lldb_init_module(debugger, dict):
    def error_out(message):
        print(message)
        return

    if not debugger.GetNumTargets() > 0:
        return error_out("Interactive scripted processes requires one non scripted process.")

    debugger.SetAsync(True)

    driving_target = debugger.GetSelectedTarget()
    if not driving_target:
        return error_out("Driving target is invalid")

    driving_process = driving_target.GetProcess()
    if not driving_process:
        return error_out("Driving process is invalid")

    # Check that the driving process is stopped somewhere.
    if not driving_process.state == lldb.eStateStopped:
        return error_out("Driving process isn't stopped")

    # Create a seconde target for the multiplexer scripted process
    mux_target = duplicate_target(driving_target)
    if not mux_target:
        return error_out("Couldn't duplicate driving target to launch multiplexer scripted process")

    class_name = __name__ + "." + MultiplexerScriptedProcess.__name__
    dictionary = {'driving_target_idx': debugger.GetIndexOfTarget(driving_target)}
    mux_process = launch_scripted_process(mux_target, class_name, dictionary)
    if not mux_process:
        return error_out("Couldn't launch multiplexer scripted process")

    # Create a target for the multiplexed even scripted process
    even_target = duplicate_target(driving_target)
    if not even_target:
        return error_out("Couldn't duplicate driving target to launch multiplexed even scripted process")

    class_name = __name__ + "." + MultiplexedScriptedProcess.__name__
    dictionary['parity'] = 0
    even_process = launch_scripted_process(even_target, class_name, dictionary)
    if not even_process:
        return error_out("Couldn't launch multiplexed even scripted process")
    multiplex(mux_process, even_process)

    # Create a target for the multiplexed odd scripted process
    odd_target = duplicate_target(driving_target)
    if not odd_target:
        return error_out("Couldn't duplicate driving target to launch multiplexed odd scripted process")

    dictionary['parity'] = 1
    odd_process = launch_scripted_process(odd_target, class_name, dictionary)
    if not odd_process:
        return error_out("Couldn't launch multiplexed odd scripted process")
    multiplex(mux_process, odd_process)
