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

class MipsTargetLowering;

class MipsCallLowering : public CallLowering {

public:
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

  MipsCallLowering(const MipsTargetLowering &TLI);

  bool lowerReturn(MachineIRBuilder &MIRBuilder, const Value *Val,
                   ArrayRef<Register> VRegs) const override;

  bool lowerFormalArguments(MachineIRBuilder &MIRBuilder, const Function &F,
                            ArrayRef<ArrayRef<Register>> VRegs) const override;

  bool lowerCall(MachineIRBuilder &MIRBuilder,
                 CallLoweringInfo &Info) const override;

private:
  /// Based on registers available on target machine split or extend
  /// type if needed, also change pointer type to appropriate integer
  /// type.
  template <typename T>
  void subTargetRegTypeForCallingConv(const Function &F, ArrayRef<ArgInfo> Args,
                                      ArrayRef<unsigned> OrigArgIndices,
                                      SmallVectorImpl<T> &ISDArgs) const;

  /// Split structures and arrays, save original argument indices since
  /// Mips calling convention needs info about original argument type.
  void splitToValueTypes(const DataLayout &DL, const ArgInfo &OrigArg,
                         unsigned OriginalIndex,
                         SmallVectorImpl<ArgInfo> &SplitArgs,
                         SmallVectorImpl<unsigned> &SplitArgsOrigIndices) const;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_MIPS_MIPSCALLLOWERING_H
