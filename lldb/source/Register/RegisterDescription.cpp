//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Validation and indexing of architecture register metadata.
///
//===----------------------------------------------------------------------===//

#include "lldb/Register/RegisterDescription.h"
#include "lldb/Utility/Args.h"
#include <algorithm>
#include <functional>
#include <iterator>
#include <set>

using namespace lldb_private;

llvm::Expected<RegisterDescription>
RegisterDescription::Create(std::vector<RegisterDescriptor> registers,
                            std::vector<RegisterGroup> groups,
                            OwnedTypes types) {
  RegisterDescription result;
  result.m_owned_types = std::move(types);
  result.m_registers = std::move(registers);
  result.m_groups = std::move(groups);
  for (size_t i = 0; i != result.m_registers.size(); ++i) {
    const auto &reg = result.m_registers[i];
    result.m_user_registers.push_back(reg.id);
    if (reg.id == LLDB_INVALID_REGNUM || !reg.byte_size ||
        (!reg.constant_value.empty() &&
         reg.constant_value.size() != reg.byte_size))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "invalid register description");
    if (!result.m_ids.emplace(reg.id, i).second)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "duplicate register identity");
    // A format may expose multiple slots with the same display name.
    // Exact lookup follows presentation order; identity remains unambiguous.
    if (!reg.name.empty())
      result.m_names.try_emplace(reg.name, i);
    for (size_t kind = 0; kind != lldb::kNumRegisterKinds; ++kind)
      if (reg.numbers[kind] != LLDB_INVALID_REGNUM)
        // Aliases can share an external number. Preserve the first exposed
        // value, as the context lookup did before descriptions owned the map.
        result.m_numbers[kind].try_emplace(reg.numbers[kind], i);
  }
  for (const auto &reg : result.m_registers) {
    if (reg.zero_extend_parent != LLDB_INVALID_REGNUM) {
      const auto *parent = result.Find(reg.zero_extend_parent);
      if (!parent || parent->id == reg.id || parent->byte_size <= reg.byte_size)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "invalid zero-extension parent");
    }
    for (auto refs : {llvm::ArrayRef<RegisterID>(reg.value_registers),
                      llvm::ArrayRef<RegisterID>(reg.invalidate_registers)})
      for (RegisterID id : refs)
        if (!result.Find(id))
          return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                         "unknown register relationship");
  }
  for (const auto &group : result.m_groups) {
    std::set<RegisterID> seen;
    for (RegisterID id : group.registers)
      if (!result.Find(id) || !seen.insert(id).second)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "invalid register group member");
  }
  // Every producer publishes through the same relation validation and
  // invalidation finalization, independent of its parser or CPU family.
  std::map<RegisterID, uint8_t> state;
  std::function<llvm::Error(RegisterID)> visit =
      [&](RegisterID id) -> llvm::Error {
    if (state[id] == 2)
      return llvm::Error::success();
    if (state[id] == 1)
      return llvm::createStringError("cyclic register value relationship");
    state[id] = 1;
    for (RegisterID parent : result.Find(id)->value_registers)
      if (auto error = visit(parent))
        return error;
    state[id] = 2;
    return llvm::Error::success();
  };
  std::map<RegisterID, std::vector<RegisterID>> invalidations;
  for (const auto &reg : result.m_registers) {
    if (auto error = visit(reg.id))
      return std::move(error);
    invalidations.emplace(reg.id, reg.invalidate_registers);
  }
  for (auto &reg : result.m_registers) {
    std::set<RegisterID> resolved(reg.invalidate_registers.begin(),
                                  reg.invalidate_registers.end());
    if (!reg.value_registers.empty()) {
      std::vector<RegisterID> pending(resolved.begin(), resolved.end());
      for (size_t i = 0; i < pending.size(); ++i)
        for (RegisterID next : invalidations.at(pending[i]))
          if (next != reg.id && resolved.insert(next).second)
            pending.push_back(next);
    }
    resolved.erase(reg.id);
    reg.invalidate_registers.assign(resolved.begin(), resolved.end());
  }
  return result;
}

llvm::Expected<RegisterDescription>
RegisterDescription::WithUserRegisters(llvm::ArrayRef<RegisterID> order) const {
  std::set<RegisterID> seen;
  for (RegisterID id : order)
    if (!Find(id) || !seen.insert(id).second)
      return llvm::createStringError("invalid register presentation");
  RegisterDescription result = *this;
  result.m_user_registers.assign(order.begin(), order.end());
  return result;
}

const RegisterDescriptor *RegisterDescription::Find(RegisterID id) const {
  auto it = m_ids.find(id);
  return it == m_ids.end() ? nullptr : &m_registers[it->second];
}

const RegisterDescriptor *
RegisterDescription::FindByNumber(lldb::RegisterKind kind,
                                  uint32_t number) const {
  if (static_cast<unsigned>(kind) >= lldb::kNumRegisterKinds ||
      number == LLDB_INVALID_REGNUM)
    return nullptr;
  const auto &numbers = m_numbers[kind];
  auto it = numbers.find(number);
  return it == numbers.end() ? nullptr : &m_registers[it->second];
}

const RegisterDescriptor *
RegisterDescription::FindByName(llvm::StringRef name,
                                size_t start_index) const {
  if (name.empty())
    return nullptr;
  uint32_t generic = Args::StringToGenericRegister(name);
  if (const auto *reg = FindByNumber(lldb::eRegisterKindGeneric, generic))
    return reg;
  for (size_t i = start_index; i < m_registers.size(); ++i) {
    const auto &reg = m_registers[i];
    if (name.equals_insensitive(reg.name) ||
        (!reg.alt_name.empty() && name.equals_insensitive(reg.alt_name)))
      return &reg;
  }
  return nullptr;
}

std::optional<size_t>
RegisterDescription::GetRegisterIndex(RegisterID id) const {
  auto it = m_ids.find(id);
  if (it == m_ids.end())
    return std::nullopt;
  return it->second;
}

uint32_t
RegisterDescription::ConvertNumber(lldb::RegisterKind source_kind,
                                   uint32_t number,
                                   lldb::RegisterKind target_kind) const {
  if (static_cast<unsigned>(target_kind) >= lldb::kNumRegisterKinds)
    return LLDB_INVALID_REGNUM;
  const auto *reg = FindByNumber(source_kind, number);
  return reg ? reg->numbers[target_kind] : LLDB_INVALID_REGNUM;
}

const RegisterDescriptor *
RegisterDescription::Find(llvm::StringRef name) const {
  auto it = m_names.find(name.str());
  if (it != m_names.end())
    return &m_registers[it->second];
  for (const auto &reg : m_registers)
    if (!reg.alt_name.empty() && reg.alt_name == name)
      return &reg;
  return nullptr;
}

const RegisterGroup *
RegisterDescription::FindGroup(llvm::StringRef short_name) const {
  for (const auto &group : m_groups)
    if (group.short_name == short_name)
      return &group;
  return nullptr;
}

bool RegisterDescription::IsInGroup(RegisterID id,
                                    llvm::StringRef short_name) const {
  const auto *group = FindGroup(short_name);
  return group && std::find(group->registers.begin(), group->registers.end(),
                            id) != group->registers.end();
}

const RegisterGroup *
RegisterDescription::FindGroupForRegister(RegisterID id) const {
  for (const auto &group : m_groups)
    if (std::find(group.registers.begin(), group.registers.end(), id) !=
        group.registers.end())
      return &group;
  return nullptr;
}
