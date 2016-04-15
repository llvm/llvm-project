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

namespace llvm {

class AArch64TargetLowering;

class AArch64CallLowering: public CallLowering {
 public:
  AArch64CallLowering(const AArch64TargetLowering &TLI);

  bool lowerReturn(MachineIRBuilder &MIRBuiler, const Value *Val,
                   unsigned VReg) const override;
  bool
  lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                       const Function::ArgumentListType &Args,
                       const SmallVectorImpl<unsigned> &VRegs) const override;
};
} // End of namespace llvm;
#endif
