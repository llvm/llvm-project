//===- AArch64InstructionSelector --------------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares the targeting of the InstructionSelector class for
/// AArch64.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64INSTRUCTIONSELECTOR_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64INSTRUCTIONSELECTOR_H

#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"

namespace llvm {
class AArch64InstrInfo;
class AArch64RegisterBankInfo;
class AArch64RegisterInfo;
class AArch64Subtarget;

class AArch64InstructionSelector : public InstructionSelector {
public:
  AArch64InstructionSelector(const AArch64Subtarget &STI,
                             const AArch64RegisterBankInfo &RBI);

  virtual bool select(MachineInstr &I) const override;

private:
  const AArch64InstrInfo &TII;
  const AArch64RegisterInfo &TRI;
  const AArch64RegisterBankInfo &RBI;
};

} // End llvm namespace.
#endif
