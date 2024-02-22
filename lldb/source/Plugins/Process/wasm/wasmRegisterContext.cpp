//===---- wasmRegisterContext.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "wasmRegisterContext.h"
#include "Plugins/Process/gdb-remote/GDBRemoteRegisterContext.h"
#include "ProcessWasm.h"
#include "ThreadWasm.h"
#include "lldb/Utility/RegisterValue.h"
#include <memory>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_gdb_remote;
using namespace lldb_private::wasm;

WasmRegisterContext::WasmRegisterContext(
    wasm::ThreadWasm &thread, uint32_t concrete_frame_idx,
    GDBRemoteDynamicRegisterInfoSP reg_info_sp)
    : GDBRemoteRegisterContext(thread, concrete_frame_idx, reg_info_sp, false,
                               false) {}

WasmRegisterContext::~WasmRegisterContext() = default;

uint32_t WasmRegisterContext::ConvertRegisterKindToRegisterNumber(
    lldb::RegisterKind kind, uint32_t num) {
  return num;
}

size_t WasmRegisterContext::GetRegisterCount() { return 0; }

const RegisterInfo *WasmRegisterContext::GetRegisterInfoAtIndex(size_t reg) {
  uint32_t tag = (reg >> kTagShift) & kTagMask;
  if (tag == WasmVirtualRegisterKinds::eNotAWasmLocation)
    return m_reg_info_sp->GetRegisterInfoAtIndex(reg & kIndexMask);

  auto it = m_register_map.find(reg);
  if (it == m_register_map.end()) {
    WasmVirtualRegisterKinds kind = static_cast<WasmVirtualRegisterKinds>(tag);
    std::tie(it, std::ignore) = m_register_map.insert(
        {reg,
         std::make_unique<WasmVirtualRegisterInfo>(kind, reg & kIndexMask)});
  }
  return it->second.get();
}

size_t WasmRegisterContext::GetRegisterSetCount() { return 0; }

const RegisterSet *WasmRegisterContext::GetRegisterSet(size_t reg_set) {
  return nullptr;
}

bool WasmRegisterContext::ReadRegister(const RegisterInfo *reg_info,
                                       RegisterValue &value) {
  if (reg_info->name)
    return GDBRemoteRegisterContext::ReadRegister(reg_info, value);

  ThreadWasm *thread = static_cast<ThreadWasm *>(&GetThread());
  ProcessWasm *process = static_cast<ProcessWasm *>(thread->GetProcess().get());
  if (!thread)
    return false;

  uint32_t frame_index = m_concrete_frame_idx;
  WasmVirtualRegisterInfo *wasm_reg_info =
      static_cast<WasmVirtualRegisterInfo *>(
          const_cast<RegisterInfo *>(reg_info));
  uint8_t buf[16];
  size_t size = 0;
  switch (wasm_reg_info->kind) {
  case eLocal:
    process->GetWasmLocal(frame_index, wasm_reg_info->index, buf, sizeof(buf),
                          size);
    break;
  case eGlobal:
    process->GetWasmGlobal(frame_index, wasm_reg_info->index, buf, sizeof(buf),
                           size);
    break;
  case eOperandStack:
    process->GetWasmStackValue(frame_index, wasm_reg_info->index, buf,
                               sizeof(buf), size);
    break;
  default:
    return false;
  }

  DataExtractor reg_data(buf, size, process->GetByteOrder(),
                         process->GetAddressByteSize());
  const bool partial_data_ok = false;
  wasm_reg_info->byte_size = size;
  wasm_reg_info->encoding = lldb::eEncodingUint;
  Status error(value.SetValueFromData(*reg_info, reg_data,
                                      reg_info->byte_offset, partial_data_ok));
  return error.Success();
}

void WasmRegisterContext::InvalidateAllRegisters() {}

bool WasmRegisterContext::WriteRegister(const RegisterInfo *reg_info,
                                        const RegisterValue &value) {
  return false;
}
