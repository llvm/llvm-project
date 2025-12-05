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
#include "lldb/Utility/DataBufferHeap.h"

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
}

void ProcessWasm::Initialize() {
  static llvm::once_flag g_once_flag;

  llvm::call_once(g_once_flag, []() {
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(), CreateInstance,
                                  DebuggerInitialize);
  });
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
      return exe_objfile->GetArchitecture().GetMachine() ==
             llvm::Triple::wasm32;
  }

  // However, if there is no wasm module, we return false, otherwise,
  // we might use ProcessWasm to attach gdb remote.
  return false;
}

std::shared_ptr<ThreadGDBRemote> ProcessWasm::CreateThread(lldb::tid_t tid) {
  return std::make_shared<ThreadWasm>(*this, tid);
}

size_t ProcessWasm::ReadMemory(lldb::addr_t vm_addr, void *buf, size_t size,
                               Status &error) {
  wasm_addr_t wasm_addr(vm_addr);

  switch (wasm_addr.GetType()) {
  case WasmAddressType::Memory:
  case WasmAddressType::Object:
    return ProcessGDBRemote::ReadMemory(vm_addr, buf, size, error);
  case WasmAddressType::Invalid:
    error.FromErrorStringWithFormat(
        "Wasm read failed for invalid address 0x%" PRIx64, vm_addr);
    return 0;
  }
  llvm_unreachable("Fully covered switch above");
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
ProcessWasm::GetWasmVariable(WasmVirtualRegisterKinds kind, int frame_index,
                             int index) {
  StreamString packet;
  switch (kind) {
  case eWasmTagLocal:
    packet.Printf("qWasmLocal:");
    break;
  case eWasmTagGlobal:
    packet.Printf("qWasmGlobal:");
    break;
  case eWasmTagOperandStack:
    packet.PutCString("qWasmStackValue:");
    break;
  case eWasmTagNotAWasmLocation:
    return llvm::createStringError("not a Wasm location");
  }
  packet.Printf("%d;%d", frame_index, index);

  StringExtractorGDBRemote response;
  if (m_gdb_comm.SendPacketAndWaitForResponse(packet.GetString(), response) !=
      GDBRemoteCommunication::PacketResult::Success)
    return llvm::createStringError("failed to send Wasm variable");

  if (!response.IsNormalResponse())
    return llvm::createStringError("failed to get response for Wasm variable");

  WritableDataBufferSP buffer_sp(
      new DataBufferHeap(response.GetStringRef().size() / 2, 0));
  response.GetHexBytes(buffer_sp->GetData(), '\xcc');
  return buffer_sp;
}
