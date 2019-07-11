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

class AMDGPUCallLowering: public CallLowering {
  Register lowerParameterPtr(MachineIRBuilder &MIRBuilder, Type *ParamTy,
                             uint64_t Offset) const;

  void lowerParameter(MachineIRBuilder &MIRBuilder, Type *ParamTy,
                      uint64_t Offset, unsigned Align,
                      Register DstReg) const;

 public:
  AMDGPUCallLowering(const AMDGPUTargetLowering &TLI);

  bool lowerReturn(MachineIRBuilder &MIRBuilder, const Value *Val,
                   ArrayRef<Register> VRegs) const override;

  bool lowerFormalArgumentsKernel(MachineIRBuilder &MIRBuilder,
                                  const Function &F,
                                  ArrayRef<ArrayRef<Register>> VRegs) const;

  bool lowerFormalArguments(MachineIRBuilder &MIRBuilder, const Function &F,
                            ArrayRef<ArrayRef<Register>> VRegs) const override;
  static CCAssignFn *CCAssignFnForCall(CallingConv::ID CC, bool IsVarArg);
  static CCAssignFn *CCAssignFnForReturn(CallingConv::ID CC, bool IsVarArg);
};
} // End of namespace llvm;
#endif
