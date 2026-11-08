//===-- NativeRegisterContext.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/common/NativeRegisterContext.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Register/RegisterValue.h"

#include "lldb/Host/PosixApi.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Host/common/NativeThreadProtocol.h"

using namespace lldb;
using namespace lldb_private;

NativeRegisterContext::NativeRegisterContext(NativeThreadProtocol &thread)
    : m_thread(thread) {}

NativeRegisterContext::NativeRegisterContext(
    NativeThreadProtocol &thread,
    std::shared_ptr<const RegisterDescription> description)
    : m_thread(thread), m_description(std::move(description)) {}

// Destructor
NativeRegisterContext::~NativeRegisterContext() = default;

lldb::addr_t NativeRegisterContext::GetPC(lldb::addr_t fail_value) {
  Log *log = GetLog(LLDBLog::Thread);

  auto description = GetDescription();
  const auto *descriptor =
      description ? description->FindByNumber(eRegisterKindGeneric,
                                              LLDB_REGNUM_GENERIC_PC)
                  : nullptr;
  RegisterID reg = descriptor ? descriptor->id : LLDB_INVALID_REGNUM;
  LLDB_LOGF(log, "Using register ID %" PRIu32 " (default %" PRIu64 ")", reg,
            fail_value);

  const uint64_t retval = ReadRegisterAsUnsigned(reg, fail_value);

  LLDB_LOG(log, "PC value: {0}", retval);

  return retval;
}

lldb::addr_t
NativeRegisterContext::GetPCfromBreakpointLocation(lldb::addr_t fail_value) {
  return GetPC(fail_value);
}

Status NativeRegisterContext::SetPC(lldb::addr_t pc) {
  auto description = GetDescription();
  const auto *descriptor =
      description ? description->FindByNumber(eRegisterKindGeneric,
                                              LLDB_REGNUM_GENERIC_PC)
                  : nullptr;
  RegisterID reg = descriptor ? descriptor->id : LLDB_INVALID_REGNUM;
  return Status::FromError(WriteRegisterFromUnsigned(reg, pc));
}

lldb::addr_t NativeRegisterContext::GetSP(lldb::addr_t fail_value) {
  auto description = GetDescription();
  const auto *descriptor =
      description ? description->FindByNumber(eRegisterKindGeneric,
                                              LLDB_REGNUM_GENERIC_SP)
                  : nullptr;
  RegisterID reg = descriptor ? descriptor->id : LLDB_INVALID_REGNUM;
  return ReadRegisterAsUnsigned(reg, fail_value);
}

Status NativeRegisterContext::SetSP(lldb::addr_t sp) {
  auto description = GetDescription();
  const auto *descriptor =
      description ? description->FindByNumber(eRegisterKindGeneric,
                                              LLDB_REGNUM_GENERIC_SP)
                  : nullptr;
  RegisterID reg = descriptor ? descriptor->id : LLDB_INVALID_REGNUM;
  return Status::FromError(WriteRegisterFromUnsigned(reg, sp));
}

lldb::addr_t NativeRegisterContext::GetFP(lldb::addr_t fail_value) {
  auto description = GetDescription();
  const auto *descriptor =
      description ? description->FindByNumber(eRegisterKindGeneric,
                                              LLDB_REGNUM_GENERIC_FP)
                  : nullptr;
  RegisterID reg = descriptor ? descriptor->id : LLDB_INVALID_REGNUM;
  return ReadRegisterAsUnsigned(reg, fail_value);
}

Status NativeRegisterContext::SetFP(lldb::addr_t fp) {
  auto description = GetDescription();
  const auto *descriptor =
      description ? description->FindByNumber(eRegisterKindGeneric,
                                              LLDB_REGNUM_GENERIC_FP)
                  : nullptr;
  RegisterID reg = descriptor ? descriptor->id : LLDB_INVALID_REGNUM;
  return Status::FromError(WriteRegisterFromUnsigned(reg, fp));
}

lldb::addr_t NativeRegisterContext::GetReturnAddress(lldb::addr_t fail_value) {
  auto description = GetDescription();
  const auto *descriptor =
      description ? description->FindByNumber(eRegisterKindGeneric,
                                              LLDB_REGNUM_GENERIC_RA)
                  : nullptr;
  RegisterID reg = descriptor ? descriptor->id : LLDB_INVALID_REGNUM;
  return ReadRegisterAsUnsigned(reg, fail_value);
}

lldb::addr_t NativeRegisterContext::GetFlags(lldb::addr_t fail_value) {
  auto description = GetDescription();
  const auto *descriptor =
      description ? description->FindByNumber(eRegisterKindGeneric,
                                              LLDB_REGNUM_GENERIC_FLAGS)
                  : nullptr;
  RegisterID reg = descriptor ? descriptor->id : LLDB_INVALID_REGNUM;
  return ReadRegisterAsUnsigned(reg, fail_value);
}

lldb::tid_t NativeRegisterContext::GetThreadID() const {
  return m_thread.GetID();
}

uint32_t NativeRegisterContext::NumSupportedHardwareBreakpoints() { return 0; }

uint32_t NativeRegisterContext::SetHardwareBreakpoint(lldb::addr_t addr,
                                                      size_t size) {
  return LLDB_INVALID_INDEX32;
}

Status NativeRegisterContext::ClearAllHardwareBreakpoints() {
  return Status::FromErrorString("not implemented");
}

bool NativeRegisterContext::ClearHardwareBreakpoint(uint32_t hw_idx) {
  return false;
}

Status NativeRegisterContext::GetHardwareBreakHitIndex(uint32_t &bp_index,
                                                       lldb::addr_t trap_addr) {
  bp_index = LLDB_INVALID_INDEX32;
  return Status::FromErrorString("not implemented");
}

uint32_t NativeRegisterContext::NumSupportedHardwareWatchpoints() { return 0; }

uint32_t NativeRegisterContext::SetHardwareWatchpoint(lldb::addr_t addr,
                                                      size_t size,
                                                      uint32_t watch_flags) {
  return LLDB_INVALID_INDEX32;
}

bool NativeRegisterContext::ClearHardwareWatchpoint(uint32_t hw_index) {
  return false;
}

Status NativeRegisterContext::ClearWatchpointHit(uint32_t hw_index) {
  return Status::FromErrorString("not implemented");
}

Status NativeRegisterContext::ClearAllHardwareWatchpoints() {
  return Status::FromErrorString("not implemented");
}

Status NativeRegisterContext::IsWatchpointHit(uint32_t wp_index, bool &is_hit) {
  is_hit = false;
  return Status::FromErrorString("not implemented");
}

Status NativeRegisterContext::GetWatchpointHitIndex(uint32_t &wp_index,
                                                    lldb::addr_t trap_addr) {
  wp_index = LLDB_INVALID_INDEX32;
  return Status::FromErrorString("not implemented");
}

Status NativeRegisterContext::IsWatchpointVacant(uint32_t wp_index,
                                                 bool &is_vacant) {
  is_vacant = false;
  return Status::FromErrorString("not implemented");
}

lldb::addr_t NativeRegisterContext::GetWatchpointAddress(uint32_t wp_index) {
  return LLDB_INVALID_ADDRESS;
}

lldb::addr_t NativeRegisterContext::GetWatchpointHitAddress(uint32_t wp_index) {
  return LLDB_INVALID_ADDRESS;
}

bool NativeRegisterContext::HardwareSingleStep(bool enable) { return false; }

std::vector<RegisterID>
NativeRegisterContext::GetExpeditedRegisters(ExpeditedRegs expType) const {
  auto description = GetDescription();
  if (!description)
    return {};
  if (expType == ExpeditedRegs::Full) {
    if (description->GetGroups().empty())
      return {};
    return description->GetGroups().front().registers;
  }
  std::vector<RegisterID> ids;
  for (uint32_t role : {LLDB_REGNUM_GENERIC_PC, LLDB_REGNUM_GENERIC_SP,
                        LLDB_REGNUM_GENERIC_FP, LLDB_REGNUM_GENERIC_RA})
    if (const auto *reg = description->FindByNumber(eRegisterKindGeneric, role))
      ids.push_back(reg->id);
  return ids;
}

uint32_t NativeRegisterContext::GetRegisterValueOffset(RegisterID id) const {
  auto description = GetDescription();
  const auto *reg = description ? description->Find(id) : nullptr;
  if (!reg || reg->value_registers.size() != 1)
    return 0;
  const auto *parent = description->Find(reg->value_registers.front());
  if (!parent)
    return 0;
  if (m_arch.GetByteOrder() == eByteOrderBig &&
      reg->byte_size + reg->value_byte_offset <= parent->byte_size)
    return parent->byte_size - reg->byte_size - reg->value_byte_offset;
  return reg->value_byte_offset;
}

NativeRegisterContext::NativeRegisterContext(NativeThreadProtocol &thread,
                                             const ArchSpec &arch,
                                             RegisterLayout layout)
    : m_thread(thread), m_arch(arch) {
  SetRegisterLayout(std::move(layout));
}

void NativeRegisterContext::SetRegisterLayout(RegisterLayout layout) {
  m_description = layout.GetDescription();
  m_layout = std::make_shared<RegisterLayout>(std::move(layout));
}

std::shared_ptr<const RegisterDescription>
NativeRegisterContext::GetDescription() const {
  return m_description;
}
const RegisterLayout &NativeRegisterContext::GetLayout() const {
  assert(m_layout && "native context has no register layout");
  return (*m_layout);
}
size_t NativeRegisterContext::GetBankSize(RegisterBankID id) const {
  const auto *bank = m_layout ? (*m_layout).FindBank(id) : nullptr;
  return bank ? bank->byte_size : 0;
}

llvm::Expected<RegisterValue> NativeRegisterContext::ReadBankRegister(
    RegisterID id, llvm::ArrayRef<RegisterData> candidates) const {
  std::vector<RegisterData> banks;
  for (const auto &bank : candidates)
    if (GetLayout().FindBank(bank.bank))
      banks.push_back(bank);
  return GetLayout().Read(id, banks);
}
llvm::Error NativeRegisterContext::WriteBankRegister(
    RegisterID id, const RegisterValue &value,
    llvm::ArrayRef<MutableRegisterData> candidates) const {
  std::vector<MutableRegisterData> banks;
  for (const auto &bank : candidates)
    if (GetLayout().FindBank(bank.bank))
      banks.push_back(bank);
  return GetLayout().Write(id, value, banks);
}


