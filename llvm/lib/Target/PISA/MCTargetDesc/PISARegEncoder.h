//===-- PISARegEncoder.h - Encode PISA virtual registers ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISAREGENCODER_H
#define LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISAREGENCODER_H

#include "llvm/MC/MCRegister.h"

namespace llvm {

class MCRegisterClass;
using TargetRegisterClass = MCRegisterClass;

namespace PISA {

class RegEncoder {
public:
  enum RegType {
    NONE,
    // Keep in sync with TSFlags in PISARegisterInfo.td
    REG,
    PRED,
    NUM_TYPE
  };

  enum class RegBank : unsigned {
    Reg1,
    Reg8,
    Reg16,
    Reg32,
    Reg64,
    Reg128,
    RegV2_8,
    RegV2_16,
    RegV2_32,
    RegV2_64,
    RegV3_8,
    RegV3_16,
    RegV3_32,
    RegV3_64,
    RegV4_8,
    RegV4_16,
    RegV4_32,
    RegV4_64,
    RegV5_32,
    RegV6_32,
    RegV7_32,
    RegV8_32,
    RegV16_32,
    RegV32_32,
    RegV64_32,
    NUM_BANK
  };

  static RegBank getRegBank(unsigned NumElts, unsigned EltSize);
  static const char *getPrefixFromBank(RegBank Bank);
  static bool isVirtualRegNo(unsigned RegNo);
  static unsigned encodeVirtualRegister(unsigned Idx, RegBank Bank,
                                        RegType Type);
  static std::pair<const char *, unsigned>
  decodeVirtualRegister(MCRegister Reg);

protected:
  static RegEncoder::RegBank getRegBank(uint8_t TSFlags);
  static RegEncoder::RegType getRegType(uint8_t TSFlags);
  static RegEncoder::RegType getRegType(const TargetRegisterClass *RC);
};

} // namespace PISA
} // namespace llvm

#endif // LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISAREGENCODER_H
