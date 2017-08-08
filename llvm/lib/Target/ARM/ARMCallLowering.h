//===-- llvm/lib/Target/ARM/ARMCallLowering.h - Call lowering -------------===//
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

#ifndef LLVM_LIB_TARGET_ARM_ARMCALLLOWERING
#define LLVM_LIB_TARGET_ARM_ARMCALLLOWERING

#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/ValueTypes.h"

namespace llvm {

class ARMTargetLowering;
class MachineInstrBuilder;

class ARMCallLowering : public CallLowering {
public:
  ARMCallLowering(const ARMTargetLowering &TLI);

  bool lowerReturn(MachineIRBuilder &MIRBuiler, const Value *Val,
                   unsigned VReg) const override;

  bool lowerFormalArguments(MachineIRBuilder &MIRBuilder, const Function &F,
                            ArrayRef<unsigned> VRegs) const override;

  bool lowerCall(MachineIRBuilder &MIRBuilder, CallingConv::ID CallConv,
                 const MachineOperand &Callee, const ArgInfo &OrigRet,
                 ArrayRef<ArgInfo> OrigArgs) const override;

private:
  bool lowerReturnVal(MachineIRBuilder &MIRBuilder, const Value *Val,
                      unsigned VReg, MachineInstrBuilder &Ret) const;

  typedef std::function<void(unsigned Reg, uint64_t Offset)> SplitArgTy;

  /// Split an argument into one or more arguments that the CC lowering can cope
  /// with (e.g. replace pointers with integers).
  void splitToValueTypes(const ArgInfo &OrigArg,
                         SmallVectorImpl<ArgInfo> &SplitArgs,
                         MachineFunction &MF,
                         const SplitArgTy &PerformArgSplit) const;
};
} // End of namespace llvm
#endif
