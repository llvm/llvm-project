//===-- llvm/CodeGen/GlobalISel/CallLowering.h - Call lowering --*- C++ -*-===//
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

#ifndef LLVM_CODEGEN_GLOBALISEL_CALLLOWERING_H
#define LLVM_CODEGEN_GLOBALISEL_CALLLOWERING_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/Function.h"

namespace llvm {
// Forward declarations.
class MachineIRBuilder;
class MachineOperand;
class TargetLowering;
class Value;

class CallLowering {
  const TargetLowering *TLI;
 protected:
  /// Getter for generic TargetLowering class.
  const TargetLowering *getTLI() const {
    return TLI;
  }

  /// Getter for target specific TargetLowering class.
  template <class XXXTargetLowering>
    const XXXTargetLowering *getTLI() const {
    return static_cast<const XXXTargetLowering *>(TLI);
  }
 public:
  CallLowering(const TargetLowering *TLI) : TLI(TLI) {}
  virtual ~CallLowering() {}

  /// This hook must be implemented to lower outgoing return values, described
  /// by \p Val, into the specified virtual register \p VReg.
  /// This hook is used by GlobalISel.
  ///
  /// \return True if the lowering succeeds, false otherwise.
  virtual bool lowerReturn(MachineIRBuilder &MIRBuilder,
                           const Value *Val, unsigned VReg) const {
    return false;
  }

  /// This hook must be implemented to lower the incoming (formal)
  /// arguments, described by \p Args, for GlobalISel. Each argument
  /// must end up in the related virtual register described by VRegs.
  /// In other words, the first argument should end up in VRegs[0],
  /// the second in VRegs[1], and so on.
  /// \p MIRBuilder is set to the proper insertion for the argument
  /// lowering.
  ///
  /// \return True if the lowering succeeded, false otherwise.
  virtual bool
  lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                       const Function::ArgumentListType &Args,
                       ArrayRef<unsigned> VRegs) const {
    return false;
  }

  /// This hook must be implemented to lower the given call instruction,
  /// including argument and return value marshalling.
  ///
  /// \p Callee is the destination of the call. It should be either a register,
  /// globaladdress, or externalsymbol.
  ///
  /// \p ResTys is a list of the individual result types this function call will
  /// produce. The types are used to assign physical registers to each slot.
  ///
  /// \p ResRegs is a list of the virtual registers that we expect to be defined
  /// by this call, one per entry in \p ResTys.
  ///
  /// \p ArgTys is a list of the types each member of \p ArgRegs has; used by
  /// the target to decide which register/stack slot should be allocated.
  ///
  /// \p ArgRegs is a list of virtual registers containing each argument that
  /// needs to be passed.
  ///
  /// \return true if the lowering succeeded, false otherwise.
  virtual bool lowerCall(MachineIRBuilder &MIRBuilder,
                         const MachineOperand &Callee, ArrayRef<MVT> ResTys,
                         ArrayRef<unsigned> ResRegs, ArrayRef<MVT> ArgTys,
                         ArrayRef<unsigned> ArgRegs) const {
    return false;
  }

  /// This hook must be implemented to lower the given call instruction,
  /// including argument and return value marshalling.
  ///
  /// \p ResReg is a register where the call's return value should be stored (or
  /// 0 if there is no return value).
  ///
  /// \p ArgRegs is a list of virtual registers containing each argument that
  /// needs to be passed.
  ///
  /// \p GetCalleeReg is a callback to materialize a register for the callee if
  /// the target determines it cannot jump to the destination based purely on \p
  /// CI. This might be because \p CI is indirect, or because of the limited
  /// range of an immediate jump.
  ///
  /// \return true if the lowering succeeded, false otherwise.
  virtual bool lowerCall(MachineIRBuilder &MIRBuilder, const CallInst &CI,
                         unsigned ResReg, ArrayRef<unsigned> ArgRegs,
                         std::function<unsigned()> GetCalleeReg) const;
};
} // End namespace llvm.

#endif
