//===-- PISACallLowering.h - Call lowering --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISACALLLOWERING_H
#define LLVM_LIB_TARGET_PISA_PISACALLLOWERING_H

#include "llvm/CodeGen/GlobalISel/CallLowering.h"

namespace llvm {

class PISATargetLowering;

class PISACallLowering : public CallLowering {

public:
  PISACallLowering(const PISATargetLowering &TLI);

  bool lowerReturn(MachineIRBuilder &MIRBuilder, const Value *Val,
                   ArrayRef<Register> VRegs, FunctionLoweringInfo &FLI,
                   Register SwiftErrorVReg) const override;

  bool lowerFormalArguments(MachineIRBuilder &MIRBuilder, const Function &F,
                            ArrayRef<ArrayRef<Register>> VRegs,
                            FunctionLoweringInfo &FLI) const override;

  // Build OpCall, or replace with a builtin function.
  bool lowerCall(MachineIRBuilder &MIRBuilder,
                 CallLoweringInfo &Info) const override;

private:
  unsigned getLoadParamOpcode(MachineIRBuilder &MIRBuilder, const Function &F,
                              const Register &VReg, Type *ArgType) const;
  void loadParamWithOpcode(MachineIRBuilder &MIRBuilder, const Function &F,
                           const Register &VReg, Type *ArgType, unsigned Opcode,
                           unsigned ArgNo, unsigned Offset) const;
};
} // end namespace llvm

#endif // LLVM_LIB_TARGET_PISA_PISACALLLOWERING_H
