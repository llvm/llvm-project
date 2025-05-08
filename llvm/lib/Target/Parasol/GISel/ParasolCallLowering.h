//===-- ParasolCallLowering.h - Call lowering for GlobalISel --------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes how to lower LLVM calls to machine code calls.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Parasol_GISEL_ParasolCALLLOWERING_H
#define LLVM_LIB_TARGET_Parasol_GISEL_ParasolCALLLOWERING_H

#include "ParasolISelLowering.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/IR/CallingConv.h"

namespace llvm {

class ParasolTargetLowering;

class ParasolCallLowering : public CallLowering {
public:
  ParasolCallLowering(const ParasolTargetLowering &TLI);
  bool lowerReturn(MachineIRBuilder &MIRBuilder, const Value *Val,
                   ArrayRef<Register> VRegs, FunctionLoweringInfo &FLI,
                   Register SwiftErrorVReg) const override;
  bool lowerFormalArguments(MachineIRBuilder &MIRBuilder, const Function &F,
                            ArrayRef<ArrayRef<Register>> VRegs,
                            FunctionLoweringInfo &FLI) const override;
  bool lowerCall(MachineIRBuilder &MIRBuilder,
                 CallLoweringInfo &Info) const override;
};
} // namespace llvm

#endif
