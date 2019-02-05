//===- llvm/lib/Target/X86/X86CallLowering.h - Call lowering ----*- C++ -*-===//
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

#ifndef LLVM_LIB_TARGET_X86_X86CALLLOWERING_H
#define LLVM_LIB_TARGET_X86_X86CALLLOWERING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include <functional>

namespace llvm {

class DataLayout;
class MachineRegisterInfo;
class X86TargetLowering;

class X86CallLowering : public CallLowering {
public:
  X86CallLowering(const X86TargetLowering &TLI);

  bool lowerReturn(MachineIRBuilder &MIRBuilder, const Value *Val,
                   ArrayRef<unsigned> VRegs) const override;

  bool lowerFormalArguments(MachineIRBuilder &MIRBuilder, const Function &F,
                            ArrayRef<unsigned> VRegs) const override;

  bool lowerCall(MachineIRBuilder &MIRBuilder, CallingConv::ID CallConv,
                 const MachineOperand &Callee, const ArgInfo &OrigRet,
                 ArrayRef<ArgInfo> OrigArgs) const override;

private:
  /// A function of this type is used to perform value split action.
  using SplitArgTy = std::function<void(ArrayRef<unsigned>)>;

  bool splitToValueTypes(const ArgInfo &OrigArgInfo,
                         SmallVectorImpl<ArgInfo> &SplitArgs,
                         const DataLayout &DL, MachineRegisterInfo &MRI,
                         SplitArgTy SplitArg) const;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_X86_X86CALLLOWERING_H
