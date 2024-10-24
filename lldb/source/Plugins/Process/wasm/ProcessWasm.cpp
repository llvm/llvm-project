//===-- ProcessWasm.cpp ---------------------------------------------------===//
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
#include "lldb/Utility/DataBufferHeap.h"

#include "lldb/Target/UnixSignals.h"
#include "llvm/ADT/ArrayRef.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_gdb_remote;
using namespace lldb_private::wasm;

LLDB_PLUGIN_DEFINE(ProcessWasm)

// ProcessGDBRemote constructor
ProcessWasm::ProcessWasm(lldb::TargetSP target_sp, ListenerSP listener_sp)
    : ProcessGDBRemote(target_sp, listener_sp) {
  /* always use linux signals for wasm process */
  m_unix_signals_sp =
      UnixSignals::Create(ArchSpec{"wasm32-unknown-unknown-wasm"});
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

llvm::StringRef ProcessWasm::GetPluginNameStatic() {
  static ConstString g_name("wasm");
  return g_name;
}

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
  lldb::ProcessSP process_sp;
  if (crash_file_path == nullptr)
    process_sp = std::make_shared<ProcessWasm>(target_sp, listener_sp);
  return process_sp;
}

bool ProcessWasm::CanDebug(lldb::TargetSP target_sp,
                           bool plugin_specified_by_name) {
  if (plugin_specified_by_name)
    return true;

  Module *exe_module = target_sp->GetExecutableModulePointer();
  if (exe_module) {
    ObjectFile *exe_objfile = exe_module->GetObjectFile();
    return exe_objfile->GetArchitecture().GetMachine() == llvm::Triple::wasm32;
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
    if (wasm_addr.module_id != 0) {
      if (WasmReadMemory(wasm_addr.module_id, wasm_addr.offset, buf, size)) {
        return size;
      }
      error.SetErrorStringWithFormat("Wasm memory read failed for 0x%" PRIx64,
                                     vm_addr);
      return 0;
    } else {
      return ProcessGDBRemote::ReadMemory(vm_addr, buf, size, error);
    }
  case WasmAddressType::Code:
    return ProcessGDBRemote::ReadMemory(wasm_addr, buf, size, error);
  case WasmAddressType::Invalid:
  default:
    error.SetErrorStringWithFormat(
        "Wasm read failed for invalid address 0x%" PRIx64, vm_addr);
    return 0;
  }
}

lldb::ModuleSP ProcessWasm::ReadModuleFromMemory(const FileSpec &file_spec,
                                                 lldb::addr_t header_addr,
                                                 size_t size_to_read) {
  wasm_addr_t wasm_addr(header_addr);
  wasm_addr.type = WasmAddressType::Code;
  return Process::ReadModuleFromMemory(file_spec, wasm_addr, size_to_read);
}

lldb::addr_t ProcessWasm::FixMemoryAddress(lldb::addr_t address,
                                           StackFrame *stack_frame) const {
  if (stack_frame) {
    assert(stack_frame->CalculateTarget()->GetArchitecture().GetMachine() ==
           llvm::Triple::wasm32);
    // Extract Wasm module ID from the program counter.
    wasm_addr_t wasm_addr(address);
    wasm_addr.module_id =
        wasm_addr_t(stack_frame->GetStackID().GetPC()).module_id;
    wasm_addr.type = WasmAddressType::Memory;
    return wasm_addr;
  }
  return address;
}

size_t ProcessWasm::WasmReadMemory(uint32_t wasm_module_id, lldb::addr_t addr,
                                   void *buf, size_t buffer_size) {
  char packet[64];
  int packet_len =
      ::snprintf(packet, sizeof(packet), "qWasmMem:%d;%" PRIx64 ";%" PRIx64,
                 wasm_module_id, static_cast<uint64_t>(addr),
                 static_cast<uint64_t>(buffer_size));
  assert(packet_len + 1 < (int)sizeof(packet));
  UNUSED_IF_ASSERT_DISABLED(packet_len);
  StringExtractorGDBRemote response;
  if (m_gdb_comm.SendPacketAndWaitForResponse(packet, response,
                                              GetInterruptTimeout()) ==
      GDBRemoteCommunication::PacketResult::Success) {
    if (response.IsNormalResponse()) {
      return response.GetHexBytes(llvm::MutableArrayRef<uint8_t>(
                                      static_cast<uint8_t *>(buf), buffer_size),
                                  '\xdd');
    }
  }
  return 0;
}

size_t ProcessWasm::WasmReadData(uint32_t wasm_module_id, lldb::addr_t addr,
                                 void *buf, size_t buffer_size) {
  char packet[64];
  int packet_len =
      ::snprintf(packet, sizeof(packet), "qWasmData:%d;%" PRIx64 ";%" PRIx64,
                 wasm_module_id, static_cast<uint64_t>(addr),
                 static_cast<uint64_t>(buffer_size));
  assert(packet_len + 1 < (int)sizeof(packet));
  UNUSED_IF_ASSERT_DISABLED(packet_len);
  StringExtractorGDBRemote response;
  if (m_gdb_comm.SendPacketAndWaitForResponse(packet, response,
                                              GetInterruptTimeout()) ==
      GDBRemoteCommunication::PacketResult::Success) {
    if (response.IsNormalResponse()) {
      return response.GetHexBytes(llvm::MutableArrayRef<uint8_t>(
                                      static_cast<uint8_t *>(buf), buffer_size),
                                  '\xdd');
    }
  }
  return 0;
}

bool ProcessWasm::GetWasmLocal(int frame_index, int index, void *buf,
                               size_t buffer_size, size_t &size) {
  StreamString packet;
  packet.Printf("qWasmLocal:");
  packet.Printf("%d;%d", frame_index, index);
  StringExtractorGDBRemote response;
  if (m_gdb_comm.SendPacketAndWaitForResponse(packet.GetString(), response) !=
      GDBRemoteCommunication::PacketResult::Success) {
    return false;
  }

  if (!response.IsNormalResponse()) {
    return false;
  }

  WritableDataBufferSP buffer_sp(
      new DataBufferHeap(response.GetStringRef().size() / 2, 0));
  response.GetHexBytes(buffer_sp->GetData(), '\xcc');
  size = buffer_sp->GetByteSize();
  if (size <= buffer_size) {
    memcpy(buf, buffer_sp->GetBytes(), size);
    return true;
  }

  return false;
}

bool ProcessWasm::GetWasmGlobal(int frame_index, int index, void *buf,
                                size_t buffer_size, size_t &size) {
  StreamString packet;
  packet.PutCString("qWasmGlobal:");
  packet.Printf("%d;%d", frame_index, index);
  StringExtractorGDBRemote response;
  if (m_gdb_comm.SendPacketAndWaitForResponse(packet.GetString(), response) !=
      GDBRemoteCommunication::PacketResult::Success) {
    return false;
  }

  if (!response.IsNormalResponse()) {
    return false;
  }

  WritableDataBufferSP buffer_sp(
      new DataBufferHeap(response.GetStringRef().size() / 2, 0));
  response.GetHexBytes(buffer_sp->GetData(), '\xcc');
  size = buffer_sp->GetByteSize();
  if (size <= buffer_size) {
    memcpy(buf, buffer_sp->GetBytes(), size);
    return true;
  }

  return false;
}

bool ProcessWasm::GetWasmStackValue(int frame_index, int index, void *buf,
                                    size_t buffer_size, size_t &size) {
  StreamString packet;
  packet.PutCString("qWasmStackValue:");
  packet.Printf("%d;%d", frame_index, index);
  StringExtractorGDBRemote response;
  if (m_gdb_comm.SendPacketAndWaitForResponse(packet.GetString(), response) !=
      GDBRemoteCommunication::PacketResult::Success) {
    return false;
  }

  if (!response.IsNormalResponse()) {
    return false;
  }

  WritableDataBufferSP buffer_sp(
      new DataBufferHeap(response.GetStringRef().size() / 2, 0));
  response.GetHexBytes(buffer_sp->GetData(), '\xcc');
  size = buffer_sp->GetByteSize();
  if (size <= buffer_size) {
    memcpy(buf, buffer_sp->GetBytes(), size);
    return true;
  }

  return false;
}

bool ProcessWasm::GetWasmCallStack(lldb::tid_t tid,
                                   std::vector<lldb::addr_t> &call_stack_pcs) {
  call_stack_pcs.clear();
  StreamString packet;
  packet.Printf("qWasmCallStack:");
  packet.Printf("%llx", tid);
  StringExtractorGDBRemote response;
  if (m_gdb_comm.SendPacketAndWaitForResponse(packet.GetString(), response) !=
      GDBRemoteCommunication::PacketResult::Success) {
    return false;
  }

  if (!response.IsNormalResponse()) {
    return false;
  }

  addr_t buf[1024 / sizeof(addr_t)];
  size_t bytes = response.GetHexBytes(
      llvm::MutableArrayRef<uint8_t>((uint8_t *)buf, sizeof(buf)), '\xdd');
  if (bytes == 0) {
    return false;
  }

  for (size_t i = 0; i < bytes / sizeof(addr_t); i++) {
    call_stack_pcs.push_back(buf[i]);
  }
  return true;
}
