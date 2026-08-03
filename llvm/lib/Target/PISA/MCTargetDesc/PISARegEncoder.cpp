//===-- PISARegEncoder.cpp - Encode PISA virtual registers ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISARegEncoder.h"

using namespace llvm;
using namespace PISA;

const char *RegEncoder::getPrefixFromBank(RegBank Bank) {

  const auto NumBanks = static_cast<size_t>(RegBank::NUM_BANK);

  const char *const BankPrefixes[NumBanks] = {
      "%p",   "%b",   "%h",   "%w",   "%d",    "%q",    "%v2b", "%v2h", "%v2w",
      "%v2d", "%v3b", "%v3h", "%v3w", "%v3d",  "%v4b",  "%v4h", "%v4w", "%v4d",
      "%v5w", "%v6w", "%v7w", "%v8w", "%v16w", "%v32w", "%v64w"};

  auto Index = static_cast<size_t>(Bank);
  if (Index >= NumBanks)
    llvm_unreachable("Unknown register bank!");

  return BankPrefixes[Index];
}

RegEncoder::RegBank RegEncoder::getRegBank(unsigned NumElts, unsigned EltSize) {
  switch (NumElts) {
  case 1:
    switch (EltSize) {
    case 1:
      return RegBank::Reg1;
    case 8:
      return RegBank::Reg8;
    case 16:
      return RegBank::Reg16;
    case 32:
      return RegBank::Reg32;
    case 64:
      return RegBank::Reg64;
    case 128:
      return RegBank::Reg128;
    default:
      llvm_unreachable("Unknown element size!");
    }
    break;
  case 2:
    switch (EltSize) {
    case 8:
      return RegBank::RegV2_8;
    case 16:
      return RegBank::RegV2_16;
    case 32:
      return RegBank::RegV2_32;
    case 64:
      return RegBank::RegV2_64;
    default:
      llvm_unreachable("Unknown element size!");
    }
    break;
  case 3:
    switch (EltSize) {
    case 8:
      return RegBank::RegV3_8;
    case 16:
      return RegBank::RegV3_16;
    case 32:
      return RegBank::RegV3_32;
    case 64:
      return RegBank::RegV3_64;
    default:
      llvm_unreachable("Unknown element size!");
    }
    break;
  case 4:
    switch (EltSize) {
    case 8:
      return RegBank::RegV4_8;
    case 16:
      return RegBank::RegV4_16;
    case 32:
      return RegBank::RegV4_32;
    case 64:
      return RegBank::RegV4_64;
    default:
      llvm_unreachable("Unknown element size!");
    }
    break;
  case 5:
    if (EltSize == 32)
      return RegBank::RegV5_32;
    else
      llvm_unreachable("Unknown element size!");
    break;
  case 6:
    if (EltSize == 32)
      return RegBank::RegV6_32;
    else
      llvm_unreachable("Unknown element size!");
    break;
  case 7:
    if (EltSize == 32)
      return RegBank::RegV7_32;
    else
      llvm_unreachable("Unknown element size!");
    break;
  case 8:
    if (EltSize == 32)
      return RegBank::RegV8_32;
    else
      llvm_unreachable("Unknown element size!");
    break;
  case 16:
    if (EltSize == 32)
      return RegBank::RegV16_32;
    else
      llvm_unreachable("Unknown element size!");
    break;
  case 32:
    if (EltSize == 32)
      return RegBank::RegV32_32;
    else
      llvm_unreachable("Unknown element size!");
    break;
  case 64:
    if (EltSize == 32)
      return RegBank::RegV64_32;
    else
      llvm_unreachable("Unknown element size!");
    break;
  default:
    llvm_unreachable("Unknown number of elements!");
    return RegBank::NUM_BANK;
  }
}

// 3 bits used for type, 5 bits used for bank, 24 bits used for index
static constexpr unsigned NumTypeBits = 3;
static constexpr unsigned NumBankBits = 5;
static constexpr unsigned NumRegBits = 24;
static_assert(RegEncoder::NUM_TYPE < 5, "need to update encoding");
static_assert(static_cast<unsigned>(RegEncoder::RegBank::NUM_BANK) < 32,
              "need to update encoding");

RegEncoder::RegType RegEncoder::getRegType(uint8_t TSFlags) {
  unsigned TypeBitsMask = (1U << NumTypeBits) - 1;
  return static_cast<RegEncoder::RegType>(TSFlags & TypeBitsMask);
}

RegEncoder::RegType RegEncoder::getRegType(const TargetRegisterClass *RC) {
  return getRegType(RC->TSFlags);
}

RegEncoder::RegBank RegEncoder::getRegBank(uint8_t TSFlags) {
  unsigned BankBitsMask = (1U << NumBankBits) - 1;
  return static_cast<RegEncoder::RegBank>(TSFlags & BankBitsMask);
}

unsigned RegEncoder::encodeVirtualRegister(unsigned Idx, RegBank Bank,
                                           RegType Type) {
  // Check that NumRegBits is sufficient to encode the register Idx
  auto RegBank = static_cast<unsigned>(Bank);
  assert(Idx < (1U << NumRegBits) &&
         "Register index exceeds virtual register encoding limit");
  return Register::index2VirtReg((Type << (NumRegBits + NumBankBits)) |
                                 RegBank << (NumRegBits) |
                                 (Idx & ((1U << NumRegBits) - 1)));
}

std::pair<const char *, unsigned>
RegEncoder::decodeVirtualRegister(MCRegister Reg) {
  auto BankBits = (Reg >> NumRegBits) & ((1U << NumBankBits) - 1);
  const char *Prefix = getPrefixFromBank(getRegBank(BankBits));
  unsigned Num = Reg & ((1U << NumRegBits) - 1);
  return std::make_pair(Prefix, Num);
}

bool RegEncoder::isVirtualRegNo(unsigned RegNo) {
  return getRegType(RegNo >> (NumRegBits + NumBankBits)) != NONE;
}
