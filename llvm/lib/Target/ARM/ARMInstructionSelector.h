//===- ARMInstructionSelector -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file declares the targeting of the InstructionSelector class for ARM.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_ARMINSTRUCTIONSELECTOR_H
#define LLVM_LIB_TARGET_ARM_ARMINSTRUCTIONSELECTOR_H

#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"

namespace llvm {

class ARMBaseInstrInfo;
class ARMBaseRegisterInfo;
class ARMRegisterBankInfo;
class ARMSubtarget;

class ARMInstructionSelector : public InstructionSelector {
public:
  ARMInstructionSelector(const ARMSubtarget &STI,
                         const ARMRegisterBankInfo &RBI);

  bool select(MachineInstr &I) const override;

private:
  const ARMBaseInstrInfo &TII;
  const ARMBaseRegisterInfo &TRI;
  const ARMRegisterBankInfo &RBI;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_ARM_ARMINSTRUCTIONSELECTOR_H
