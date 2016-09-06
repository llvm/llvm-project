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
#include "llvm/IR/Function.h"

namespace llvm {
// Forward declarations.
class MachineIRBuilder;
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
  /// \p CalleeReg is a virtual-register containing the destination if
  /// `CI.getCalledFunction()` returns null (i.e. if the call is indirect);
  /// otherwise it is 0.
  ///
  /// \p ResReg is a register where the call's return value should be stored (or
  /// 0 if there is no return value).
  ///
  /// \p ArgRegs is a list of virtual registers containing each argument that
  /// needs to be passed.
  ///
  /// \return true if the lowering succeeded, false otherwise.
  virtual bool lowerCall(MachineIRBuilder &MIRBuilder, const CallInst &CI,
                         unsigned CalleeReg, unsigned ResReg,
                         ArrayRef<unsigned> ArgRegs) const {
    return false;
  }
};
} // End namespace llvm.

#endif
