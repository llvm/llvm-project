//===-- llvm/lib/Target/X86/X86CallLowering.h - Call lowering -----===//
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

#ifndef LLVM_LIB_TARGET_X86_X86CALLLOWERING
#define LLVM_LIB_TARGET_X86_X86CALLLOWERING

#include "llvm/ADT/ArrayRef.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"

namespace llvm {

class Function;
class MachineIRBuilder;
class X86TargetLowering;
class Value;

class X86CallLowering : public CallLowering {
public:
  X86CallLowering(const X86TargetLowering &TLI);

  bool lowerReturn(MachineIRBuilder &MIRBuiler, const Value *Val,
                   unsigned VReg) const override;

  bool lowerFormalArguments(MachineIRBuilder &MIRBuilder, const Function &F,
                            ArrayRef<unsigned> VRegs) const override;

  bool lowerCall(MachineIRBuilder &MIRBuilder, CallingConv::ID CallConv,
                 const MachineOperand &Callee, const ArgInfo &OrigRet,
                 ArrayRef<ArgInfo> OrigArgs) const override;

private:
  /// A function of this type is used to perform value split action.
  typedef std::function<void(ArrayRef<unsigned>)> SplitArgTy;

  bool splitToValueTypes(const ArgInfo &OrigArgInfo,
                         SmallVectorImpl<ArgInfo> &SplitArgs,
                         const DataLayout &DL, MachineRegisterInfo &MRI,
                         SplitArgTy SplitArg) const;
};
} // namespace llvm
#endif
