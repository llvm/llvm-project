//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterPacketLayout.h"
#include <functional>
#include <limits>

using namespace lldb_private;
using namespace lldb_private::process_gdb_remote;

llvm::Expected<RegisterPacketLayout> RegisterPacketLayout::Create(
    std::shared_ptr<const RegisterDescription> description,
    llvm::ArrayRef<RegisterID> order, lldb::ByteOrder byte_order) {
  if (!description || (byte_order != lldb::eByteOrderLittle &&
                       byte_order != lldb::eByteOrderBig))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid register packet description");
  RegisterPacketLayout result;
  result.m_description = std::move(description);
  for (RegisterID id : order)
    if (!result.m_description->Find(id) ||
        !result.m_indices.emplace(id, result.m_indices.size()).second)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "invalid register packet order");

  for (RegisterID id : order) {
    const auto &reg = *result.m_description->Find(id);
    bool alias = reg.value_registers.size() == 1 &&
                 result.m_indices.count(reg.value_registers.front());
    uint32_t offset = 0;
    if (!alias) {
      if (reg.byte_size >
          std::numeric_limits<uint32_t>::max() - result.m_byte_size)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "register packet is too large");
      offset = result.m_byte_size;
      result.m_byte_size += reg.byte_size;
    }
    result.m_entries.push_back({id, offset, reg.byte_size, !alias});
  }

  std::vector<uint8_t> state(order.size(), 0);
  std::function<llvm::Error(size_t)> resolve =
      [&](size_t index) -> llvm::Error {
    if (state[index] == 2)
      return llvm::Error::success();
    if (state[index] == 1)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "cyclic register packet aliases");
    state[index] = 1;
    auto &entry = result.m_entries[index];
    const auto &reg = *result.m_description->Find(entry.id);
    if (!entry.primary) {
      size_t parent_index = result.m_indices.at(reg.value_registers.front());
      if (auto error = resolve(parent_index))
        return error;
      const auto &parent = result.m_entries[parent_index];
      if (reg.value_byte_offset > parent.byte_size ||
          entry.byte_size > parent.byte_size - reg.value_byte_offset)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "register alias exceeds its parent");
      uint32_t offset =
          byte_order == lldb::eByteOrderBig
              ? parent.byte_size - entry.byte_size - reg.value_byte_offset
              : reg.value_byte_offset;
      entry.byte_offset = parent.byte_offset + offset;
    }
    state[index] = 2;
    return llvm::Error::success();
  };
  for (size_t i = 0; i < order.size(); ++i)
    if (auto error = resolve(i))
      return std::move(error);
  return result;
}

std::optional<uint32_t>
RegisterPacketLayout::GetWireNumber(RegisterID id) const {
  auto position = m_indices.find(id);
  return position == m_indices.end()
             ? std::nullopt
             : std::optional<uint32_t>(position->second);
}
std::vector<uint32_t>
RegisterPacketLayout::GetWireNumbers(llvm::ArrayRef<RegisterID> ids) const {
  std::vector<uint32_t> numbers;
  for (RegisterID id : ids)
    if (auto number = GetWireNumber(id))
      numbers.push_back(*number);
  return numbers;
}
