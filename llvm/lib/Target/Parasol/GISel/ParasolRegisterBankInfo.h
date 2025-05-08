//===-- ParasolRegisterBankInfo.h -----------------------------------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the targeting of the RegisterBankInfo class for Parasol.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Parasol_GISEL_ParasolREGISTERBANKINFO_H
#define LLVM_LIB_TARGET_Parasol_GISEL_ParasolREGISTERBANKINFO_H

#include "MCTargetDesc/ParasolMCTargetDesc.h"
#include "llvm/CodeGen/RegisterBankInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

#define GET_REGBANK_DECLARATIONS
#include "ParasolGenRegisterBank.inc"

namespace llvm {
class TargetRegisterInfo;

class ParasolGenRegisterBankInfo : public RegisterBankInfo {
protected:
  /// Get the pointer to the ValueMapping of the operands of a copy
  /// instruction from the \p SrcBankID register bank to the \p DstBankID
  /// register bank with a size of \p Size.
  static const RegisterBankInfo::ValueMapping *
  getCopyMapping(unsigned DstBankID, unsigned SrcBankID, unsigned Size);

#define GET_TARGET_REGBANK_CLASS
#include "ParasolGenRegisterBank.inc"
};

class ParasolRegisterBankInfo final : public ParasolGenRegisterBankInfo {
private:
  const RegisterBankInfo::InstructionMapping &
  getSameKindOfOperandsMapping(const MachineInstr &MI) const;

public:
  ParasolRegisterBankInfo(const TargetRegisterInfo &TRI);

  unsigned copyCost(const RegisterBank &A, const RegisterBank &B,
                    TypeSize Size) const override;

  const RegisterBank &getRegBankFromRegClass(const TargetRegisterClass &RC,
                                             LLT) const override;

  //   InstructionMappings
  //   getInstrAlternativeMappings(const MachineInstr &MI) const override;

  const InstructionMapping &
  getInstrMapping(const MachineInstr &MI) const override;
};
} // namespace llvm

#endif
