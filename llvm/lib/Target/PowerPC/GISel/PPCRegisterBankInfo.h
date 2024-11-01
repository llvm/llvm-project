//===-- PPCRegisterBankInfo.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the targeting of the RegisterBankInfo class for PowerPC.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PPC_GISEL_PPCREGISTERBANKINFO_H
#define LLVM_LIB_TARGET_PPC_GISEL_PPCREGISTERBANKINFO_H

#include "llvm/CodeGen/RegisterBank.h"
#include "llvm/CodeGen/RegisterBankInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

#define GET_REGBANK_DECLARATIONS
#include "PPCGenRegisterBank.inc"

namespace llvm {
class TargetRegisterInfo;

class PPCGenRegisterBankInfo : public RegisterBankInfo {
protected:
  enum PartialMappingIdx {
    PMI_None = -1,
    PMI_GPR64 = 1,
    PMI_FPR32 = 2,
    PMI_FPR64 = 3,
    PMI_Min = PMI_GPR64,
  };

  static RegisterBankInfo::PartialMapping PartMappings[];
  static RegisterBankInfo::ValueMapping ValMappings[];
  static PartialMappingIdx BankIDToCopyMapIdx[];

  /// Get the pointer to the ValueMapping representing the RegisterBank
  /// at \p RBIdx.
  ///
  /// The returned mapping works for instructions with the same kind of
  /// operands for up to 3 operands.
  ///
  /// \pre \p RBIdx != PartialMappingIdx::None
  static const RegisterBankInfo::ValueMapping *
  getValueMapping(PartialMappingIdx RBIdx);

  /// Get the pointer to the ValueMapping of the operands of a copy
  /// instruction from the \p SrcBankID register bank to the \p DstBankID
  /// register bank with a size of \p Size.
  static const RegisterBankInfo::ValueMapping *
  getCopyMapping(unsigned DstBankID, unsigned SrcBankID, unsigned Size);

#define GET_TARGET_REGBANK_CLASS
#include "PPCGenRegisterBank.inc"
};

class PPCRegisterBankInfo final : public PPCGenRegisterBankInfo {
public:
  PPCRegisterBankInfo(const TargetRegisterInfo &TRI);

  const RegisterBank &getRegBankFromRegClass(const TargetRegisterClass &RC,
                                             LLT Ty) const override;
  const InstructionMapping &
  getInstrMapping(const MachineInstr &MI) const override;

  InstructionMappings
  getInstrAlternativeMappings(const MachineInstr &MI) const override;
};
} // namespace llvm

#endif
