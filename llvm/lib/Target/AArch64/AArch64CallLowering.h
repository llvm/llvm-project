//===-- llvm/lib/Target/AArch64/AArch64CallLowering.h - Call lowering -----===//
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

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64CALLLOWERING
#define LLVM_LIB_TARGET_AARCH64_AARCH64CALLLOWERING

#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/ValueTypes.h"

namespace llvm {

class AArch64TargetLowering;

class AArch64CallLowering: public CallLowering {
 public:
  AArch64CallLowering(const AArch64TargetLowering &TLI);

  bool lowerReturn(MachineIRBuilder &MIRBuiler, const Value *Val,
                   unsigned VReg) const override;

  bool lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                            const Function::ArgumentListType &Args,
                            ArrayRef<unsigned> VRegs) const override;

  bool lowerCall(MachineIRBuilder &MIRBuilder, const MachineOperand &Callee,
                 ArrayRef<MVT> ResTys, ArrayRef<unsigned> ResRegs,
                 ArrayRef<MVT> ArgTys,
                 ArrayRef<unsigned> ArgRegs) const override;

private:
  typedef std::function<void(MachineIRBuilder &, unsigned, unsigned)>
      AssignFnTy;

  bool handleAssignments(MachineIRBuilder &MIRBuilder, CCAssignFn *AssignFn,
                         ArrayRef<MVT> ArgsTypes, ArrayRef<unsigned> ArgRegs,
                         AssignFnTy AssignValToReg) const;
};
} // End of namespace llvm;
#endif
