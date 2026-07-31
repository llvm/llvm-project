//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProcessWasm.h"
#include "ThreadWasm.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Value.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "llvm/Support/ErrorExtras.h"
#include <cstring>

#include "lldb/Target/UnixSignals.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_gdb_remote;
using namespace lldb_private::wasm;

LLDB_PLUGIN_DEFINE(ProcessWasm)

ProcessWasm::ProcessWasm(lldb::TargetSP target_sp, ListenerSP listener_sp)
    : ProcessGDBRemote(target_sp, listener_sp) {
  assert(target_sp);
  // Wasm doesn't have any Unix-like signals as a platform concept, but pretend
  // like it does to appease LLDB.
  m_unix_signals_sp = UnixSignals::Create(target_sp->GetArchitecture());
  // FIXME: LLVM's RuntimeDyld doesn't support the Wasm object format, so we
  // can't JIT expressions for this target.
  SetCanJIT(false);
}

void ProcessWasm::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance,
                                DebuggerInitialize);
}

void ProcessWasm::DebuggerInitialize(Debugger &debugger) {
  ProcessGDBRemote::DebuggerInitialize(debugger);
}

llvm::StringRef ProcessWasm::GetPluginName() { return GetPluginNameStatic(); }

llvm::StringRef ProcessWasm::GetPluginNameStatic() { return "wasm"; }

llvm::StringRef ProcessWasm::GetPluginDescriptionStatic() {
  return "GDB Remote protocol based WebAssembly debugging plug-in.";
}

void ProcessWasm::Terminate() {
  PluginManager::UnregisterPlugin(ProcessWasm::CreateInstance);
}

lldb::ProcessSP ProcessWasm::CreateInstance(lldb::TargetSP target_sp,
                                            ListenerSP listener_sp,
                                            const FileSpec *crash_file_path,
                                            bool can_connect) {
  if (crash_file_path == nullptr)
    return std::make_shared<ProcessWasm>(target_sp, listener_sp);
  return {};
}

bool ProcessWasm::CanDebug(lldb::TargetSP target_sp,
                           bool plugin_specified_by_name) {
  if (plugin_specified_by_name)
    return true;

  if (Module *exe_module = target_sp->GetExecutableModulePointer()) {
    if (ObjectFile *exe_objfile = exe_module->GetObjectFile())
      return exe_objfile->GetArchitecture().GetTriple().isWasm();
  }

  // However, if there is no wasm module, we return false, otherwise,
  // we might use ProcessWasm to attach gdb remote.
  return false;
}

std::shared_ptr<ThreadGDBRemote> ProcessWasm::CreateThread(lldb::tid_t tid) {
  if (!GetTarget().GetArchitecture().GetTriple().isWasm())
    return ProcessGDBRemote::CreateThread(tid);

  return std::make_shared<ThreadWasm>(*this, tid);
}

size_t ProcessWasm::ReadGlobal(uint32_t module_id, uint32_t index, void *buf,
                               size_t size, Status &error) {
  // Looking for a frame drives the unwinder, so only pay for it when the read
  // has to go through one.
  const uint32_t frame_index = CanNameInstance(module_id)
                                   ? LLDB_INVALID_INDEX32
                                   : GetFallbackFrameIndex(module_id);

  llvm::Expected<lldb::DataBufferSP> buffer =
      GetWasmGlobal(module_id, index, frame_index);
  if (!buffer) {
    error = Status::FromError(buffer.takeError());
    return 0;
  }

  // A global comes back whole. Reading more than it holds would have to come
  // from somewhere else, and the next index is not adjacent storage.
  const size_t global_size = (*buffer)->GetByteSize();
  if (size > global_size) {
    error = Status::FromErrorStringWithFormatv(
        "Wasm global read failed: requested {0} bytes from a {1}-byte global",
        size, global_size);
    return 0;
  }

  std::memcpy(buf, (*buffer)->GetBytes(), size);
  return size;
}

uint32_t ProcessWasm::GetFallbackFrameIndex(uint32_t module_id) {
  ThreadSP thread = GetThreadList().GetSelectedThread();
  StackFrameSP frame =
      thread ? thread->GetSelectedFrame(DoNoSelectMostRelevantFrame) : nullptr;
  if (!frame)
    return LLDB_INVALID_INDEX32;

  // A frame can only stand in for the module the stub reports it executing.
  const uint32_t frame_index = frame->GetConcreteFrameIndex();
  ThreadWasm &wasm_thread = static_cast<ThreadWasm &>(*thread);
  if (GetWasmModuleID(wasm_thread.GetConcreteFramePC(frame_index)) != module_id)
    return LLDB_INVALID_INDEX32;

  return frame_index;
}

size_t ProcessWasm::ReadMemory(const ProcessAddress &process_addr, void *buf,
                               size_t size, Status &error) {
  // A caller may reuse one error across reads, as the overridden
  // Process::ReadMemory allows.
  error.Clear();

  lldb::addr_t vm_addr = process_addr.GetValue();
  wasm_addr_t wasm_addr(vm_addr);

  switch (wasm_addr.GetType()) {
  case WasmAddressType::Memory:
  case WasmAddressType::Object:
    return ProcessGDBRemote::ReadMemory(vm_addr, buf, size, error);
  case WasmAddressType::Global:
    return ReadGlobal(wasm_addr.GetModuleID(), wasm_addr.GetOffset(), buf, size,
                      error);
  case WasmAddressType::Invalid:
    break;
  }

  error = Status::FromErrorStringWithFormatv(
      "Wasm read failed for invalid address {0:x} (type = {1:x}, module = "
      "{2:x}, offset = {3:x})",
      vm_addr, wasm_addr.GetType(), wasm_addr.GetModuleID(),
      wasm_addr.GetOffset());
  return 0;
}

llvm::Expected<std::vector<lldb::addr_t>>
ProcessWasm::GetWasmCallStack(lldb::tid_t tid) {
  StreamString packet;
  packet.Printf("qWasmCallStack:");
  packet.Printf("%" PRIx64, tid);

  StringExtractorGDBRemote response;
  if (m_gdb_comm.SendPacketAndWaitForResponse(packet.GetString(), response) !=
      GDBRemoteCommunication::PacketResult::Success)
    return llvm::createStringError("failed to send qWasmCallStack");

  if (!response.IsNormalResponse())
    return llvm::createStringError("failed to get response for qWasmCallStack");

  WritableDataBufferSP data_buffer_sp =
      std::make_shared<DataBufferHeap>(response.GetStringRef().size() / 2, 0);
  const size_t bytes = response.GetHexBytes(data_buffer_sp->GetData(), '\xcc');
  if (bytes == 0 || bytes % sizeof(uint64_t) != 0)
    return llvm::createStringError("invalid response for qWasmCallStack");

  // To match the Wasm specification, the addresses are encoded in little endian
  // byte order.
  DataExtractor data(data_buffer_sp, lldb::eByteOrderLittle,
                     GetAddressByteSize());
  lldb::offset_t offset = 0;
  std::vector<lldb::addr_t> call_stack_pcs;
  while (offset < bytes)
    call_stack_pcs.push_back(data.GetU64(&offset));

  return call_stack_pcs;
}

llvm::Expected<lldb::DataBufferSP>
ProcessWasm::SendWasmValueQuery(llvm::StringRef packet) {
  StringExtractorGDBRemote response;
  if (m_gdb_comm.SendPacketAndWaitForResponse(packet, response) !=
      GDBRemoteCommunication::PacketResult::Success)
    return llvm::createStringErrorV("failed to send {0}", packet);

  if (!response.IsNormalResponse())
    return llvm::createStringErrorV("failed to get response for {0}", packet);

  WritableDataBufferSP buffer_sp(
      new DataBufferHeap(response.GetStringRef().size() / 2, 0));
  response.GetHexBytes(buffer_sp->GetData(), '\xcc');
  return buffer_sp;
}

llvm::Expected<lldb::DataBufferSP>
ProcessWasm::GetWasmVariable(WasmVirtualRegisterKinds kind,
                             uint32_t frame_index, uint32_t index) {
  switch (kind) {
  case eWasmTagLocal:
    return SendWasmValueQuery(
        llvm::formatv("qWasmLocal:{0};{1}", frame_index, index).str());
  case eWasmTagOperandStack:
    return SendWasmValueQuery(
        llvm::formatv("qWasmStackValue:{0};{1}", frame_index, index).str());
  case eWasmTagGlobal:
    // A global belongs to a module rather than a frame. See GetWasmGlobal.
    return llvm::createStringError("a Wasm global does not belong to a frame");
  case eWasmTagNotAWasmLocation:
    return llvm::createStringError("not a Wasm location");
  }
  llvm_unreachable("unhandled Wasm virtual register kind");
}

llvm::Expected<lldb::DataBufferSP>
ProcessWasm::GetWasmGlobal(uint32_t module_id, uint32_t index,
                           uint32_t frame_index) {
  // The global index space belongs to a module instance, so an index only names
  // a global together with the instance holding it.
  if (CanNameInstance(module_id))
    return SendWasmValueQuery(
        llvm::formatv("qWasmGlobal:{0};instance:{1};", index, module_id).str());

  // A frame stands in for the instance it is executing only where that instance
  // cannot be named.
  if (frame_index != LLDB_INVALID_INDEX32)
    return SendWasmValueQuery(
        llvm::formatv("qWasmGlobal:{0};{1}", frame_index, index).str());

  if (module_id == kWasmInvalidModuleID)
    return llvm::createStringErrorV(
        "global {0} belongs to no known module instance, and no frame is "
        "executing one to read it through",
        index);

  return llvm::createStringErrorV(
      "the Wasm stub can only read a global through a frame, and no frame is "
      "executing module {0:x} to read global {1} through",
      module_id, index);
}

bool ProcessWasm::CanNameInstance(uint32_t module_id) {
  return module_id != kWasmInvalidModuleID &&
         m_gdb_comm.GetWasmInstanceSupported();
}
