//===- lib/Target/AMDGPU/AMDGPUCallLowering.h - Call lowering -*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes how to lower LLVM calls to machine code calls.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUCALLLOWERING_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUCALLLOWERING_H

#include "AMDGPU.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"

namespace llvm {

class AMDGPUTargetLowering;
class MachineInstrBuilder;

class AMDGPUCallLowering final : public CallLowering {
  void lowerParameterPtr(Register DstReg, MachineIRBuilder &B, Type *ParamTy,
                         uint64_t Offset) const;

  void lowerParameter(MachineIRBuilder &B, Type *ParamTy, uint64_t Offset,
                      Align Alignment, Register DstReg) const;

  /// A function of this type is used to perform value split action.
  using SplitArgTy = std::function<void(ArrayRef<Register>, Register, LLT, LLT, int)>;

  void splitToValueTypes(MachineIRBuilder &B,
                         const ArgInfo &OrigArgInfo,
                         SmallVectorImpl<ArgInfo> &SplitArgs,
                         const DataLayout &DL,
                         CallingConv::ID CallConv,
                         bool IsOutgoing,
                         SplitArgTy SplitArg) const;

  bool lowerReturnVal(MachineIRBuilder &B, const Value *Val,
                      ArrayRef<Register> VRegs, MachineInstrBuilder &Ret) const;

public:
  AMDGPUCallLowering(const AMDGPUTargetLowering &TLI);

  bool lowerReturn(MachineIRBuilder &B, const Value *Val,
                   ArrayRef<Register> VRegs) const override;

  bool lowerFormalArgumentsKernel(MachineIRBuilder &B, const Function &F,
                                  ArrayRef<ArrayRef<Register>> VRegs) const;

  bool lowerFormalArguments(MachineIRBuilder &B, const Function &F,
                            ArrayRef<ArrayRef<Register>> VRegs) const override;

  bool passSpecialInputs(MachineIRBuilder &MIRBuilder,
                         CCState &CCInfo,
                         SmallVectorImpl<std::pair<MCRegister, Register>> &ArgRegs,
                         CallLoweringInfo &Info) const;

  bool lowerCall(MachineIRBuilder &MIRBuilder,
                 CallLoweringInfo &Info) const override;

  static CCAssignFn *CCAssignFnForCall(CallingConv::ID CC, bool IsVarArg);
  static CCAssignFn *CCAssignFnForReturn(CallingConv::ID CC, bool IsVarArg);
};
} // End of namespace llvm;
#endif
