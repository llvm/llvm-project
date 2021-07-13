//===- MipsCallLowering.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file describes how to lower LLVM calls to machine code calls.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MIPS_MIPSCALLLOWERING_H
#define LLVM_LIB_TARGET_MIPS_MIPSCALLLOWERING_H

#include "llvm/CodeGen/GlobalISel/CallLowering.h"

namespace llvm {

class MachineMemOperand;
class MipsTargetLowering;

class MipsCallLowering : public CallLowering {

public:
#if 0
  class MipsHandler {
  public:
    MipsHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI)
        : MIRBuilder(MIRBuilder), MRI(MRI) {}

    virtual ~MipsHandler() = default;

    bool handle(ArrayRef<CCValAssign> ArgLocs,
                ArrayRef<CallLowering::ArgInfo> Args);

  protected:
    bool assignVRegs(ArrayRef<Register> VRegs, ArrayRef<CCValAssign> ArgLocs,
                     unsigned ArgLocsStartIndex, const EVT &VT);

    void setLeastSignificantFirst(SmallVectorImpl<Register> &VRegs);

    MachineIRBuilder &MIRBuilder;
    MachineRegisterInfo &MRI;

  private:
    bool assign(Register VReg, const CCValAssign &VA, const EVT &VT);

    virtual Register getStackAddress(const CCValAssign &VA,
                                     MachineMemOperand *&MMO) = 0;

    virtual void assignValueToReg(Register ValVReg, const CCValAssign &VA,
                                  const EVT &VT) = 0;

    virtual void assignValueToAddress(Register ValVReg,
                                      const CCValAssign &VA) = 0;

    virtual bool handleSplit(SmallVectorImpl<Register> &VRegs,
                             ArrayRef<CCValAssign> ArgLocs,
                             unsigned ArgLocsStartIndex, Register ArgsReg,
                             const EVT &VT) = 0;
  };
#endif

  MipsCallLowering(const MipsTargetLowering &TLI);

  bool lowerReturn(MachineIRBuilder &MIRBuilder, const Value *Val,
                   ArrayRef<Register> VRegs,
                   FunctionLoweringInfo &FLI) const override;

  bool lowerFormalArguments(MachineIRBuilder &MIRBuilder, const Function &F,
                            ArrayRef<ArrayRef<Register>> VRegs,
                            FunctionLoweringInfo &FLI) const override;

  bool lowerCall(MachineIRBuilder &MIRBuilder,
                 CallLoweringInfo &Info) const override;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_MIPS_MIPSCALLLOWERING_H
