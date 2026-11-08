//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Metadata lookup and scalar access shared by register backends.
///
//===----------------------------------------------------------------------===//

#include "lldb/Register/RegisterContext.h"

using namespace lldb_private;

size_t RegisterContext::GetRegisterCount() const {
  auto description = GetDescription();
  return description ? description->GetRegisters().size() : 0;
}

size_t RegisterContext::GetUserRegisterCount() const {
  auto description = GetDescription();
  return description ? description->GetUserRegisters().size() : 0;
}

const RegisterDescriptor *
RegisterContext::GetRegisterAtIndex(size_t index) const {
  auto description = GetDescription();
  return description ? description->GetRegisterAtIndex(index) : nullptr;
}

const RegisterDescriptor *
RegisterContext::GetUserRegisterAtIndex(size_t index) const {
  auto description = GetDescription();
  if (!description || index >= description->GetUserRegisters().size())
    return nullptr;
  return description->Find(description->GetUserRegisters()[index]);
}

size_t RegisterContext::GetRegisterGroupCount() const {
  auto description = GetDescription();
  return description ? description->GetGroups().size() : 0;
}

const RegisterGroup *RegisterContext::GetRegisterGroup(size_t index) const {
  auto description = GetDescription();
  return description ? description->GetGroupAtIndex(index) : nullptr;
}

const RegisterDescriptor *
RegisterContext::GetRegisterByName(llvm::StringRef name,
                                   uint32_t start_index) const {
  auto description = GetDescription();
  return description ? description->FindByName(name, start_index) : nullptr;
}

const RegisterDescriptor *
RegisterContext::GetRegisterByNumber(uint32_t kind, uint32_t number) const {
  auto description = GetDescription();
  return description ? description->FindByNumber(
                           static_cast<lldb::RegisterKind>(kind), number)
                     : nullptr;
}

const char *RegisterContext::GetRegisterNameAtIndex(uint32_t index) const {
  const auto *reg = GetRegisterAtIndex(index);
  return reg ? reg->name.c_str() : nullptr;
}

RegisterID RegisterContext::GetRegisterIDForNumber(uint32_t kind,
                                                   uint32_t number) const {
  const auto *reg = GetRegisterByNumber(kind, number);
  return reg ? reg->id : LLDB_INVALID_REGNUM;
}

bool RegisterContext::ConvertBetweenRegisterKinds(lldb::RegisterKind source,
                                                  uint32_t number,
                                                  lldb::RegisterKind target,
                                                  uint32_t &result) const {
  auto description = GetDescription();
  result = description ? description->ConvertNumber(source, number, target)
                       : LLDB_INVALID_REGNUM;
  return result != LLDB_INVALID_REGNUM;
}

llvm::Expected<RegisterID>
RegisterContext::ResolveRegisterNumber(lldb::RegisterKind kind,
                                       uint32_t number) const {
  auto description = GetDescription();
  const auto *reg =
      description ? description->FindByNumber(kind, number) : nullptr;
  if (!reg)
    return llvm::createStringError("unknown register number");
  return reg->id;
}

uint64_t RegisterContext::ReadRegisterAsUnsigned(RegisterID id,
                                                 uint64_t fail_value) {
  auto value = ReadRegister(id);
  if (!value) {
    llvm::consumeError(value.takeError());
    return fail_value;
  }
  return value->GetAsUInt64(fail_value);
}

llvm::Error RegisterContext::WriteRegisterFromUnsigned(RegisterID id,
                                                       uint64_t value) {
  auto description = GetDescription();
  const auto *reg = description ? description->Find(id) : nullptr;
  if (!reg)
    return llvm::createStringError("unknown register ID");
  RegisterValue encoded;
  if (!encoded.SetUInt(value, reg->byte_size))
    return llvm::createStringError("register cannot hold an unsigned value");
  return WriteRegister(id, encoded);
}

RegisterContext::RegisterContext(std::shared_ptr<const void> owner)
    : m_owned_identity(owner ? nullptr : std::make_shared<int>(0)),
      m_checkpoint_owner(owner ? std::move(owner) : m_owned_identity) {}

llvm::Expected<RegisterCheckpoint> RegisterContext::SaveCheckpoint() {
  return llvm::createStringError("register checkpoints are unavailable");
}

llvm::Error
RegisterContext::RestoreCheckpoint(const RegisterCheckpoint &checkpoint) {
  // Compare control-block ownership: different adapter handles can alias
  // subobjects of the same backend owner.
  auto owner = checkpoint.m_owner.lock();
  auto current = m_checkpoint_owner.lock();
  if (!owner || !current || owner.owner_before(current) ||
      current.owner_before(owner))
    return llvm::createStringError("register checkpoint belongs to another owner");
  return checkpoint.m_restore();
}

RegisterCheckpoint
RegisterContext::CreateCheckpoint(std::function<llvm::Error()> restore) const {
  return RegisterCheckpoint(m_checkpoint_owner.lock(), std::move(restore));
}
