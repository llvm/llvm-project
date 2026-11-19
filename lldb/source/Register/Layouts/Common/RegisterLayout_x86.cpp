//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/Common/RegisterLayout_x86.h"
#include "lldb/Register/Arch/lldb-x86_32-register-enums.h"
#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Endian.h"
#include <cstddef>
#include <iterator>

using namespace lldb_private;

uint16_t
RegisterLayout_x86::AbridgedToFullTagWord(uint8_t abridged_tw, uint16_t sw,
                                          llvm::ArrayRef<MMSReg> st_regs) {
  // Tag word is using internal FPU register numbering rather than ST(i).
  // Mapping to ST(i): i = FPU regno - TOP (Status Word, bits 11:13).
  // Here we start with FPU reg 7 and go down.
  int st = 7 - ((sw >> 11) & 7);
  uint16_t tw = 0;

  for (uint8_t mask = 0x80; mask != 0; mask >>= 1) {
    tw <<= 2;

    if (abridged_tw & mask) {
      // The register is non-empty, so we need to check the value of ST(i).
      uint16_t exp =
          st_regs[st].comp.sign_exp & 0x7fff; // Discard the sign bit.
      if (exp == 0) {
        if (st_regs[st].comp.mantissa == 0)
          tw |= 1; // Zero
        else
          tw |= 2; // Denormal
      } else if (exp == 0x7fff)
        tw |= 2; // Infinity or NaN
      // 0 if normal number
    } else
      tw |= 3; // Empty register

    // Rotate ST down.
    st = (st - 1) & 7;
  }

  return tw;
}

// Convert the 16-bit FPU Tag Word to the abridged 8-bit value, to be written
// into FXSAVE.
uint8_t RegisterLayout_x86::FullToAbridgedTagWord(uint16_t tw) {
  uint8_t abridged_tw = 0;

  for (uint16_t mask = 0xc000; mask != 0; mask >>= 2) {
    abridged_tw <<= 1;

    // full TW uses 11 for empty registers, aTW uses 0
    if ((tw & mask) != mask)
      abridged_tw |= 1;
  }

  return abridged_tw;
}

llvm::Error
RegisterLayout_x86::ReadX87Tag(llvm::ArrayRef<RegisterData> banks,
                               lldb::ByteOrder byte_order,
                               llvm::MutableArrayRef<uint8_t> value) {
  if (byte_order != lldb::eByteOrderLittle)
    return llvm::createStringError(
        "x87 saved state requires little-endian storage");

  auto bank = llvm::find_if(banks, [](const auto &bank) {
    return bank.bank == RegisterLayout_x86::FP;
  });
  if (bank == banks.end() ||
      bank->bytes.size() < offsetof(FXSAVE, stmm) + sizeof(FXSAVE::stmm))
    return llvm::createStringError("truncated x87 saved state");

  const auto *bytes = bank->bytes.data();
  MMSReg st[8]{};

  for (size_t i = 0; i != 8; ++i) {
    const auto *slot = bytes + offsetof(FXSAVE, stmm) + i * sizeof(MMSReg);
    st[i].comp.mantissa = llvm::support::endian::read64le(slot);
    st[i].comp.sign_exp = llvm::support::endian::read16le(slot + 8);
  }

  uint16_t tag = AbridgedToFullTagWord(
      bytes[offsetof(FXSAVE, ftag)],
      llvm::support::endian::read16le(bytes + offsetof(FXSAVE, fstat)), st);
  llvm::support::endian::write16le(value.data(), tag);

  return llvm::Error::success();
}

llvm::Error
RegisterLayout_x86::WriteX87Tag(llvm::ArrayRef<uint8_t> value,
                                lldb::ByteOrder byte_order,
                                llvm::ArrayRef<MutableRegisterData> banks) {
  if (byte_order != lldb::eByteOrderLittle)
    return llvm::createStringError(
        "x87 saved state requires little-endian storage");

  auto bank = llvm::find_if(banks, [](const auto &bank) {
    return bank.bank == RegisterLayout_x86::FP;
  });
  if (bank == banks.end() || bank->bytes.size() <= offsetof(FXSAVE, ftag))
    return llvm::createStringError("truncated x87 saved state");

  bank->bytes[offsetof(FXSAVE, ftag)] =
      FullToAbridgedTagWord(llvm::support::endian::read16le(value.data()));

  return llvm::Error::success();
}

const RegisterConversion &RegisterLayout_x86::GetX87TagConversion() {
  static const RegisterConversion conversion{2, ReadX87Tag, WriteX87Tag};
  return conversion;
}

RegisterID RegisterLayout_x86::GetX86_64GPR(RegisterID id) {
  namespace r32 = registers::x86_32::id;
  namespace r64 = registers::x86_64::id;
  switch (id) {
  case r32::eax:
    return r64::rax;
  case r32::ebx:
    return r64::rbx;
  case r32::ecx:
    return r64::rcx;
  case r32::edx:
    return r64::rdx;
  case r32::esi:
    return r64::rsi;
  case r32::edi:
    return r64::rdi;
  case r32::ebp:
    return r64::rbp;
  case r32::esp:
    return r64::rsp;
  case r32::eip:
    return r64::rip;
  case r32::eflags:
    return r64::rflags;
  case r32::cs:
    return r64::cs;
  case r32::ss:
    return r64::ss;
  case r32::ds:
    return r64::ds;
  case r32::es:
    return r64::es;
  case r32::fs:
    return r64::fs;
  case r32::gs:
    return r64::gs;
  default:
    return LLDB_INVALID_REGNUM;
  }
}

std::vector<std::pair<RegisterID, uint32_t>>
RegisterLayout_x86::GetGPRWidths(RegisterID register_count,
                                 GPRLookup get_field) {
  std::vector<std::pair<RegisterID, uint32_t>> widths;
  for (RegisterID id = 0; id != register_count; ++id)
    if (auto field = get_field(id))
      widths.emplace_back(id, field->byte_size);

  return widths;
}

namespace {

using Info = RegisterLayout_x86;

struct HardwareRegisters {
  RegisterID st, mm, xmm, ymm, bnd, bndcfgu, bndstatus, dr, ftag;
  uint32_t vector_count;
  llvm::ArrayRef<RegisterID> scalars;
};

constexpr uint32_t ScalarOffsets[] = {offsetof(Info::FXSAVE, fctrl),
                                      offsetof(Info::FXSAVE, fstat),
                                      offsetof(Info::FXSAVE, ftag),
                                      offsetof(Info::FXSAVE, fop),
                                      offsetof(Info::FXSAVE, ptr.x86_32.fiseg),
                                      offsetof(Info::FXSAVE, ptr.x86_32.fioff),
                                      offsetof(Info::FXSAVE, ptr.x86_32.foseg),
                                      offsetof(Info::FXSAVE, ptr.x86_32.fooff),
                                      offsetof(Info::FXSAVE, mxcsr),
                                      offsetof(Info::FXSAVE, mxcsrmask),
                                      offsetof(Info::FXSAVE, ptr.x86_64.fip),
                                      offsetof(Info::FXSAVE, ptr.x86_64.fdp)};

namespace hardware_x86_32 {

using namespace registers::x86_32::id;

constexpr RegisterID Scalars[] = {fctrl,
                                  registers::x86_32::id::fstat,
                                  ftag,
                                  fop,
                                  fiseg,
                                  fioff,
                                  foseg,
                                  fooff,
                                  mxcsr,
                                  mxcsrmask,
                                  LLDB_INVALID_REGNUM,
                                  LLDB_INVALID_REGNUM};
static_assert(std::size(Scalars) == std::size(ScalarOffsets));

const HardwareRegisters Registers{st0,       mm0, xmm0, ymm0, bnd0,   bndcfgu,
                                  bndstatus, dr0, ftag, 8,    Scalars};

} // namespace hardware_x86_32

namespace hardware_x86_64 {

using namespace registers::x86_64::id;

constexpr RegisterID Scalars[] = {fctrl, registers::x86_64::id::fstat,
                                  ftag,  fop,
                                  fiseg, fioff,
                                  foseg, fooff,
                                  mxcsr, mxcsrmask,
                                  fip,   fdp};
static_assert(std::size(Scalars) == std::size(ScalarOffsets));

const HardwareRegisters Registers{st0,       mm0, xmm0, ymm0, bnd0,   bndcfgu,
                                  bndstatus, dr0, ftag, 16,   Scalars};

} // namespace hardware_x86_64

} // namespace

std::array<RegisterBank, 5>
RegisterLayout_x86::GetHardwareBanks(uint32_t debug_slot_size) {
  return {{{FP, sizeof(FXSAVE)},
           {YMMHigh, 256},
           {Bounds, 64},
           {BoundsControl, 16},
           {Debug, 8 * debug_slot_size}}};
}

std::optional<RegisterLocation> RegisterLayout_x86::GetRegisterLocation(
    const RegisterDescriptor &reg, GPRLookup get_field, bool is_64_bit,
    uint32_t debug_slot_size, TagWordView tag_word) {
  const RegisterID id = reg.id;
  if (auto field = get_field(id))
    return RegisterLocation::StoredAt(field->bank, field->byte_offset,
                                      reg.byte_size);

  const auto &ids =
      is_64_bit ? hardware_x86_64::Registers : hardware_x86_32::Registers;
  const auto stored = [&](RegisterBankID bank, uint64_t offset) {
    return RegisterLocation::StoredAt(bank, offset, reg.byte_size);
  };
  const auto in_range = [id](RegisterID first, uint32_t count) {
    return id >= first && id - first < count;
  };

  if (id == ids.ftag && tag_word == TagWordView::Full) {
    RegisterLocation tag;
    tag.kind = RegisterLocation::Stored;
    tag.conversion = &GetX87TagConversion();
    tag.inputs = {{FP, offsetof(FXSAVE, ftag), 1},
                  {FP, offsetof(FXSAVE, fstat), 2},
                  {FP, offsetof(FXSAVE, stmm), sizeof(FXSAVE::stmm)}};
    return tag;
  }

  for (size_t i = 0; i != ids.scalars.size(); ++i)
    if (ids.scalars[i] == id)
      return stored(FP, ScalarOffsets[i]);

  if (in_range(ids.st, 8))
    return stored(FP, offsetof(FXSAVE, stmm) + (id - ids.st) * sizeof(MMSReg));
  if (in_range(ids.mm, 8))
    return stored(FP, offsetof(FXSAVE, stmm) + (id - ids.mm) * sizeof(MMSReg));
  if (in_range(ids.dr, 8))
    return stored(Debug, (id - ids.dr) * debug_slot_size);
  if (in_range(ids.xmm, ids.vector_count))
    return stored(FP, offsetof(FXSAVE, xmm) + (id - ids.xmm) * sizeof(XMMReg));
  if (in_range(ids.ymm, ids.vector_count)) {
    const uint32_t i = id - ids.ymm;
    RegisterLocation ymm;
    ymm.kind = RegisterLocation::Stored;
    ymm.pieces = {{FP, offsetof(FXSAVE, xmm) + i * sizeof(XMMReg), 16, 0},
                  {YMMHigh, i * sizeof(YMMHReg), 16, 16}};
    return ymm;
  }
  if (in_range(ids.bnd, 4))
    return stored(Bounds, (id - ids.bnd) * sizeof(MPXReg));
  if (id == ids.bndcfgu)
    return stored(BoundsControl, 0);
  if (id == ids.bndstatus)
    return stored(BoundsControl, sizeof(MPXCsr));

  return std::nullopt;
}
