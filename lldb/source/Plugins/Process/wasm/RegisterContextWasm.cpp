//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextWasm.h"
#include "Plugins/Process/gdb-remote/GDBRemoteRegisterContext.h"
#include "ProcessWasm.h"
#include "ThreadWasm.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "llvm/Support/Error.h"
#include <memory>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_gdb_remote;
using namespace lldb_private::wasm;

RegisterContextWasm::RegisterContextWasm(
    ThreadGDBRemote &thread, uint32_t concrete_frame_idx,
    GDBRemoteDynamicRegisterInfoSP reg_info_sp)
    : GDBRemoteRegisterContext(thread, concrete_frame_idx, reg_info_sp, false,
                               false) {}

RegisterContextWasm::~RegisterContextWasm() = default;

uint32_t RegisterContextWasm::ConvertRegisterKindToRegisterNumber(
    lldb::RegisterKind kind, uint32_t num) {
  return num;
}

size_t RegisterContextWasm::GetRegisterCount() {
  // Wasm has no registers.
  return 0;
}

const RegisterInfo *RegisterContextWasm::GetRegisterInfoAtIndex(size_t reg) {
  uint32_t tag = GetWasmVirtualRegisterTag(reg);
  if (tag == eWasmTagNotAWasmLocation)
    return m_reg_info_sp->GetRegisterInfoAtIndex(
        GetWasmVirtualRegisterIndex(reg));

  auto it = m_register_map.find(reg);
  if (it == m_register_map.end()) {
    WasmVirtualRegisterKinds kind = static_cast<WasmVirtualRegisterKinds>(tag);
    std::tie(it, std::ignore) = m_register_map.insert(
        {reg, std::make_unique<WasmVirtualRegisterInfo>(
                  kind, GetWasmVirtualRegisterIndex(reg))});
  }
  return it->second.get();
}

size_t RegisterContextWasm::GetRegisterSetCount() { return 0; }

const RegisterSet *RegisterContextWasm::GetRegisterSet(size_t reg_set) {
  // Wasm has no registers.
  return nullptr;
}

bool RegisterContextWasm::ReadRegister(const RegisterInfo *reg_info,
                                       RegisterValue &value) {
  // The only real registers is the PC.
  if (reg_info->name)
    return GDBRemoteRegisterContext::ReadRegister(reg_info, value);

  // Read the virtual registers.
  ThreadWasm *thread = static_cast<ThreadWasm *>(&GetThread());
  ProcessWasm *process = static_cast<ProcessWasm *>(thread->GetProcess().get());
  if (!thread)
    return false;

  uint32_t frame_index = m_concrete_frame_idx;
  WasmVirtualRegisterInfo *wasm_reg_info =
      static_cast<WasmVirtualRegisterInfo *>(
          const_cast<RegisterInfo *>(reg_info));

  llvm::Expected<DataBufferSP> maybe_buffer = process->GetWasmVariable(
      wasm_reg_info->kind, frame_index, wasm_reg_info->index);
  if (!maybe_buffer) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Process), maybe_buffer.takeError(),
                   "Failed to read Wasm local: {0}");
    return false;
  }

  DataBufferSP buffer_sp = *maybe_buffer;
  DataExtractor reg_data(buffer_sp, process->GetByteOrder(),
                         process->GetAddressByteSize());
  wasm_reg_info->byte_size = buffer_sp->GetByteSize();
  wasm_reg_info->encoding = lldb::eEncodingUint;

  Status error = value.SetValueFromData(
      *reg_info, reg_data, reg_info->byte_offset, /*partial_data_ok=*/false);
  return error.Success();
}

void RegisterContextWasm::InvalidateAllRegisters() {}

bool RegisterContextWasm::WriteRegister(const RegisterInfo *reg_info,
                                        const RegisterValue &value) {
  // The only real registers is the PC.
  if (reg_info->name)
    return GDBRemoteRegisterContext::WriteRegister(reg_info, value);
  return false;
}
