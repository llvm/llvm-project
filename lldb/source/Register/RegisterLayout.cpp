//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Validation and byte transfer for concrete register storage mappings.
///
//===----------------------------------------------------------------------===//

#include "lldb/Register/RegisterLayout.h"
#include "lldb/Utility/Status.h"
#include <algorithm>
#include <cstring>

using namespace lldb_private;

llvm::Error RegisterLayout::AddBank(RegisterBankID id, uint64_t byte_size) {
  if (!m_banks.emplace(id, RegisterBank{id, byte_size}).second)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "duplicate register bank");
  m_description.reset();
  return llvm::Error::success();
}

llvm::Error RegisterLayout::Add(RegisterID id, RegisterLocation location) {
  if (id == LLDB_INVALID_REGNUM ||
      !m_locations.emplace(id, std::move(location)).second)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid or duplicate register location");
  m_description.reset();
  return llvm::Error::success();
}

const RegisterLocation *RegisterLayout::Find(RegisterID id) const {
  auto it = m_locations.find(id);
  return it == m_locations.end() ? nullptr : &it->second;
}

const RegisterBank *RegisterLayout::FindBank(RegisterBankID id) const {
  auto it = m_banks.find(id);
  return it == m_banks.end() ? nullptr : &it->second;
}

llvm::Error
RegisterLayout::Validate(const RegisterDescription &description) const {
  if (m_byte_order != lldb::eByteOrderLittle &&
      m_byte_order != lldb::eByteOrderBig)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid register storage byte order");
  for (const auto &[id, location] : m_locations) {
    const auto *reg = description.Find(id);
    if (!reg)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "unknown register location identity");
    if (location.kind != RegisterLocation::Stored) {
      if (location.zero_fill || !location.pieces.empty() ||
          (location.kind == RegisterLocation::Constant &&
           location.constant.size() != reg->byte_size) ||
          (location.kind == RegisterLocation::Unavailable &&
           !location.constant.empty()))
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "invalid non-storage register location");
      continue;
    }
    if (!location.constant.empty())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "stored register has a constant value");
    std::vector<bool> covered(reg->byte_size, false);
    for (const auto &piece : location.pieces) {
      const auto *bank = FindBank(piece.bank);
      if (!bank || !piece.byte_size || piece.byte_offset > bank->byte_size ||
          piece.byte_size > bank->byte_size - piece.byte_offset ||
          piece.value_offset > reg->byte_size ||
          piece.byte_size > reg->byte_size - piece.value_offset)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "register piece out of bounds");
      for (uint32_t i = 0; i != piece.byte_size; ++i) {
        if (covered[piece.value_offset + i])
          return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                         "overlapping register pieces");
        covered[piece.value_offset + i] = true;
      }
    }
    if (!location.zero_fill &&
        std::find(covered.begin(), covered.end(), false) != covered.end())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "incomplete register storage mapping");
  }
  return llvm::Error::success();
}

llvm::Expected<RegisterLayout>
RegisterLayout::Create(std::shared_ptr<const RegisterDescription> description,
                       RegisterLayout draft) {
  if (!description)
    return llvm::createStringError("missing register description");
  if (auto error = draft.Validate(*description))
    return std::move(error);
  draft.m_description = std::move(description);
  return draft;
}

namespace {
/// Validate the supplied bank views before copying any pieces. Read and write
/// paths share these checks so a missing or short bank is reported before the
/// layout starts modifying caller-owned bytes.
template <typename Bank>
llvm::Error CheckBanks(const RegisterLayout &layout,
                       const RegisterLocation &location,
                       llvm::ArrayRef<Bank> banks) {
  for (size_t i = 0; i != banks.size(); ++i) {
    for (size_t j = 0; j != i; ++j)
      if (banks[i].bank == banks[j].bank)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "duplicate register bank data");
    const auto *format = layout.FindBank(banks[i].bank);
    if (!format)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "truncated or unknown register bank");
  }
  for (const auto &piece : location.pieces) {
    auto bank = std::find_if(banks.begin(), banks.end(), [&](const Bank &bank) {
      return bank.bank == piece.bank;
    });
    if (bank == banks.end())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "missing register bank");
    if (piece.byte_offset > bank->bytes.size() ||
        piece.byte_size > bank->bytes.size() - piece.byte_offset)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "truncated register bank");
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<std::vector<RegisterBankID>>
RegisterLayout::GetRequiredBanks(RegisterID id, bool writing) const {
  if (!m_description)
    return llvm::createStringError("register layout was not finalized");
  const auto &description = *m_description;
  const auto &layout = *this;
  const auto *reg = description.Find(id);
  if (!reg)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unknown register");
  std::vector<RegisterBankID> banks;
  if (!reg->constant_value.empty()) {
    if (writing)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "register is read-only");
    return banks;
  }
  const auto *location =
      layout.Find(writing && reg->zero_extend_parent != LLDB_INVALID_REGNUM
                      ? reg->zero_extend_parent
                      : id);
  if (!location || location->kind == RegisterLocation::Unavailable)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "register has no storage location");
  if (writing && location->kind == RegisterLocation::Constant)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "register is read-only");
  for (const auto &piece : location->pieces)
    if (std::find(banks.begin(), banks.end(), piece.bank) == banks.end())
      banks.push_back(piece.bank);
  return banks;
}

llvm::Expected<RegisterValue>
RegisterLayout::Read(RegisterID id, llvm::ArrayRef<RegisterData> banks) const {
  if (!m_description)
    return llvm::createStringError("register layout was not finalized");
  const auto &description = *m_description;
  const auto &layout = *this;
  const auto *reg = description.Find(id);
  const auto *location = layout.Find(id);
  if (!reg || (reg->constant_value.empty() &&
               (!location || location->kind == RegisterLocation::Unavailable)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "register is unavailable");
  std::vector<uint8_t> bytes(reg->byte_size);
  if (!reg->constant_value.empty()) {
    bytes = reg->constant_value;
    if (layout.GetByteOrder() == lldb::eByteOrderBig)
      std::reverse(bytes.begin(), bytes.end());
  } else if (location->kind == RegisterLocation::Constant) {
    bytes = location->constant;
  } else {
    if (llvm::Error error = CheckBanks(layout, *location, banks))
      return std::move(error);
    for (const auto &piece : location->pieces)
      for (const auto &bank : banks)
        if (bank.bank == piece.bank)
          std::memcpy(bytes.data() + piece.value_offset,
                      bank.bytes.data() + piece.byte_offset, piece.byte_size);
  }
  RegisterValue result;
  Status error;
  result.SetFromMemoryData(reg->encoding, reg->byte_size, bytes.data(),
                           bytes.size(), layout.GetByteOrder(), error,
                           reg->name);
  if (error.Fail())
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s",
                                   error.AsCString());
  return result;
}

llvm::Error
RegisterLayout::Write(RegisterID id, const RegisterValue &value,
                      llvm::ArrayRef<MutableRegisterData> banks) const {
  if (!m_description)
    return llvm::createStringError("register layout was not finalized");
  const auto &description = *m_description;
  const auto &layout = *this;
  const auto *reg = description.Find(id);
  const auto *location = layout.Find(id);
  if (!reg || !reg->constant_value.empty() || !location ||
      location->kind != RegisterLocation::Stored)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "register is unavailable or read-only");
  if (llvm::Error error = CheckBanks(layout, *location, banks))
    return error;
  std::vector<uint8_t> bytes(reg->byte_size);
  Status error;
  if (value.GetAsMemoryData(reg->byte_size, bytes.data(), bytes.size(),
                            layout.GetByteOrder(), error,
                            reg->name) != bytes.size())
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s",
                                   error.AsCString());
  if (location->zero_fill) {
    std::vector<bool> stored(reg->byte_size, false);
    for (const auto &piece : location->pieces)
      for (uint32_t i = 0; i < piece.byte_size; ++i)
        stored[piece.value_offset + i] = true;
    for (size_t i = 0; i < stored.size(); ++i)
      if (!stored[i] && bytes[i])
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "cannot enable inactive register lanes");
  }
  if (reg->zero_extend_parent != LLDB_INVALID_REGNUM) {
    const auto *parent = description.Find(reg->zero_extend_parent);
    const auto *parent_location = layout.Find(reg->zero_extend_parent);
    if (!parent || !parent_location ||
        parent_location->kind != RegisterLocation::Stored)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "zero-extension parent is unavailable");
    if (llvm::Error error = CheckBanks(layout, *parent_location, banks))
      return error;
    std::vector<uint8_t> extended(parent->byte_size, 0);
    size_t offset = layout.GetByteOrder() == lldb::eByteOrderBig
                        ? parent->byte_size - reg->byte_size
                        : 0;
    std::copy(bytes.begin(), bytes.end(), extended.begin() + offset);
    bytes = std::move(extended);
    location = parent_location;
  }
  for (const auto &piece : location->pieces)
    for (const auto &bank : banks)
      if (bank.bank == piece.bank)
        std::memcpy(bank.bytes.data() + piece.byte_offset,
                    bytes.data() + piece.value_offset, piece.byte_size);
  return llvm::Error::success();
}

bool RegisterLayout::HasSameStorage(const RegisterLayout &other) const {
  if (!m_description || !other.m_description ||
      m_byte_order != other.m_byte_order ||
      m_banks.size() != other.m_banks.size() ||
      m_locations.size() != other.m_locations.size() ||
      m_description->GetRegisters().size() !=
          other.m_description->GetRegisters().size())
    return false;
  for (const auto &[id, bank] : m_banks) {
    const auto *rhs = other.FindBank(id);
    if (!rhs || bank.byte_size != rhs->byte_size)
      return false;
  }
  for (const auto &reg : m_description->GetRegisters()) {
    const auto *rhs = other.m_description->Find(reg.id);
    if (!rhs || reg.name != rhs->name || reg.byte_size != rhs->byte_size ||
        reg.encoding != rhs->encoding || reg.numbers != rhs->numbers ||
        reg.constant_value != rhs->constant_value ||
        reg.value_registers != rhs->value_registers ||
        reg.value_byte_offset != rhs->value_byte_offset ||
        reg.zero_extend_parent != rhs->zero_extend_parent)
      return false;
  }
  for (const auto &[id, location] : m_locations) {
    const auto *rhs = other.Find(id);
    if (!rhs || location.kind != rhs->kind ||
        location.zero_fill != rhs->zero_fill ||
        location.constant != rhs->constant ||
        location.pieces.size() != rhs->pieces.size())
      return false;
    for (size_t index = 0; index < location.pieces.size(); ++index) {
      const auto &a = location.pieces[index];
      const auto &b = rhs->pieces[index];
      if (a.bank != b.bank || a.byte_offset != b.byte_offset ||
          a.byte_size != b.byte_size || a.value_offset != b.value_offset)
        return false;
    }
  }
  return true;
}
