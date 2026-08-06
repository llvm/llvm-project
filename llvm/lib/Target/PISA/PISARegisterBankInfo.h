//===-- PISARegisterBankInfo.h --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISAREGISTERBANKINFO_H
#define LLVM_LIB_TARGET_PISA_PISAREGISTERBANKINFO_H

#include "llvm/CodeGen/RegisterBankInfo.h"

#define GET_REGBANK_DECLARATIONS
#include "PISAGenRegisterBank.inc"

namespace llvm {

class TargetRegisterInfo;

class PISAGenRegisterBankInfo : public RegisterBankInfo {
protected:
#define GET_TARGET_REGBANK_CLASS
#include "PISAGenRegisterBank.inc"
};

// This class provides the information for the target register banks.
class PISARegisterBankInfo final : public PISAGenRegisterBankInfo {
public:
  const RegisterBank &getRegBankFromRegClass(const TargetRegisterClass &RC,
                                             LLT Ty) const override;

  const InstructionMapping &
  getInstrMapping(const MachineInstr &MI) const override;
};
} // namespace llvm
#endif // LLVM_LIB_TARGET_PISA_PISAREGISTERBANKINFO_H
