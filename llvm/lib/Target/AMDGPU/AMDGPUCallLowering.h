//===- lib/Target/AMDGPU/AMDGPUCallLowering.h - Call lowering -*- C++ -*---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes how to lower LLVM calls to machine code calls.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUCALLLOWERING_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUCALLLOWERING_H

#include "llvm/CodeGen/GlobalISel/CallLowering.h"

namespace llvm {

class AMDGPUTargetLowering;

class AMDGPUCallLowering: public CallLowering {

  unsigned lowerParameterPtr(MachineIRBuilder &MIRBuilder, Type *ParamTy,
                             unsigned Offset) const;

  void lowerParameter(MachineIRBuilder &MIRBuilder, Type *ParamTy,
                      unsigned Offset, unsigned DstReg) const;

 public:
  AMDGPUCallLowering(const AMDGPUTargetLowering &TLI);

  bool lowerReturn(MachineIRBuilder &MIRBuiler, const Value *Val,
                   unsigned VReg) const override;
  bool lowerFormalArguments(MachineIRBuilder &MIRBuilder, const Function &F,
                            ArrayRef<unsigned> VRegs) const override;
  CCAssignFn *CCAssignFnForCall(CallingConv::ID CC, bool IsVarArg) const;
};
} // End of namespace llvm;
#endif
