//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextWasm.h"
#include "lldb/Register/Arch/lldb-wasm-register-enums.h"
#include "Plugins/Process/gdb-remote/GDBRemoteRegisterContext.h"
#include "ProcessWasm.h"
#include "ThreadWasm.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Register/RegisterValue.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include <memory>


namespace reg_ids = lldb_private::registers::wasm::id;
using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_gdb_remote;
using namespace lldb_private::wasm;

RegisterContextWasm::RegisterContextWasm(
    ThreadGDBRemote &thread, uint32_t concrete_frame_idx,
    std::shared_ptr<const RegisterLayout> layout_sp)
    : GDBRemoteRegisterContext(thread, concrete_frame_idx, layout_sp, false,
                               false) {
  auto original = GetDescription();
  auto description = llvm::cantFail(RegisterDescription::Create(
      {original->GetRegisters().begin(), original->GetRegisters().end()}, {},
      {original->GetOwnedTypes().begin(), original->GetOwnedTypes().end()}));
  SetRegisterDescription(std::make_shared<RegisterDescription>(
      llvm::cantFail(description.WithUserRegisters({}))));
}

RegisterContextWasm::~RegisterContextWasm() = default;

uint32_t RegisterContextWasm::GetModuleID() {
  return GetWasmModuleID(GetPC(LLDB_INVALID_ADDRESS));
}

llvm::Expected<RegisterID>
RegisterContextWasm::ResolveRegisterNumber(lldb::RegisterKind kind,
                                           uint32_t number) const {
  // DW_OP_WASM_location encodes a VM location independently of the DWARF
  // expression's register numbering kind.
  if (reg_ids::GetWasmVirtualRegisterTag(number) !=
      reg_ids::eWasmTagNotAWasmLocation)
    return number;
  return RegisterContext::ResolveRegisterNumber(kind, number);
}

/// A local and an operand stack value belong to the frame, while a global
/// belongs to the module instance the frame is executing, which is named
/// directly wherever the stub accepts one.
static llvm::Expected<DataBufferSP>
ReadWasmValue(ProcessWasm &process, reg_ids::WasmVirtualRegisterKinds kind,
              uint32_t index, uint32_t frame_index, uint32_t module_id) {
  if (kind != reg_ids::eWasmTagGlobal)
    return process.GetWasmVariable(kind, frame_index, index);

  if (process.CanNameInstance(module_id))
    return process.GetWasmGlobalForModule(module_id, index);

  return process.GetWasmGlobalForFrame(frame_index, index);
}

llvm::Expected<RegisterValue> RegisterContextWasm::ReadRegister(RegisterID id) {
  auto &thread = static_cast<ThreadWasm &>(GetThread());
  const uint32_t tag = reg_ids::GetWasmVirtualRegisterTag(id);
  if (tag == reg_ids::eWasmTagNotAWasmLocation) {
    if (m_concrete_frame_idx > 0) {
      lldb::addr_t pc = thread.GetConcreteFramePC(m_concrete_frame_idx);
      if (pc != LLDB_INVALID_ADDRESS) {
        const auto *reg = GetDescription()->Find(id);
        if (!reg)
          return llvm::createStringError("unknown register ID");
        RegisterValue value;
        value.SetUInt(pc, reg->byte_size);
        return value;
      }
    }
    return GDBRemoteRegisterContext::ReadRegister(id);
  }
  auto process = static_cast<ProcessWasm *>(thread.GetProcess().get());
  if (!process)
    return llvm::createStringError("Wasm process is unavailable");
  auto buffer = ReadWasmValue(
      *process, static_cast<reg_ids::WasmVirtualRegisterKinds>(tag),
      reg_ids::GetWasmVirtualRegisterIndex(id), m_concrete_frame_idx,
      GetModuleID());
  if (!buffer)
    return buffer.takeError();
  RegisterValue value;
  value.SetBytes((*buffer)->GetBytes(), (*buffer)->GetByteSize(),
                 process->GetByteOrder());
  value.SetType(eEncodingUint, (*buffer)->GetByteSize());

  // VM locals and stack slots are discovered lazily. Publish a new immutable
  // generation after their value size is known; keep them out of register UI.
  auto previous = GetDescription();
  std::vector<RegisterDescriptor> registers(previous->GetRegisters().begin(),
                                            previous->GetRegisters().end());
  RegisterDescriptor reg;
  reg.id = id;
  reg.byte_size = value.GetByteSize();
  reg.numbers[eRegisterKindLLDB] = id;
  auto found = llvm::find_if(
      registers, [id](const auto &entry) { return entry.id == id; });
  if (found == registers.end())
    registers.push_back(std::move(reg));
  else
    *found = std::move(reg);
  auto description = RegisterDescription::Create(
      std::move(registers), {},
      {previous->GetOwnedTypes().begin(), previous->GetOwnedTypes().end()});
  if (!description)
    return description.takeError();
  auto presentation = description->WithUserRegisters({});
  if (!presentation)
    return presentation.takeError();
  SetRegisterDescription(
      std::make_shared<RegisterDescription>(std::move(*presentation)));
  return value;
}

void RegisterContextWasm::InvalidateAllRegisters() {}

llvm::Error RegisterContextWasm::WriteRegister(RegisterID id,
                                        const RegisterValue &value) {
  if (reg_ids::GetWasmVirtualRegisterTag(id) == reg_ids::eWasmTagNotAWasmLocation)
    return GDBRemoteRegisterContext::WriteRegister(id, value);
  return llvm::createStringError("Wasm value writes are unavailable");
}
