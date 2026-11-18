//===- llvm/CodeGen/ProcessImplicitDefs.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PROCESSIMPLICITDEFS_H
#define LLVM_CODEGEN_PROCESSIMPLICITDEFS_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

/// Computes what ProcessImplicitDefs does to virtual registers defined by
/// IMPLICIT_DEF, without modifying the function:
///  - Each use of such a register becomes undef, unless its instruction reads
///    the register through both a tied operand and another operand.
///  - A copy, INSERT_SUBREG, REG_SEQUENCE or PHI whose uses all become undef
///    becomes an IMPLICIT_DEF.
///  - An IMPLICIT_DEF whose uses all become undef is erased.
class ImplicitDefInfo {
  SmallPtrSet<const MachineOperand *, 16> UndefUses;
  SmallPtrSet<const MachineInstr *, 16> NewImplicitDefs;
  SmallPtrSet<const MachineInstr *, 16> ErasedImplicitDefs;

public:
  LLVM_ABI explicit ImplicitDefInfo(const MachineFunction &MF);

  /// Returns true if \p MO is a use that becomes undef.
  bool isUndef(const MachineOperand &MO) const {
    return UndefUses.contains(&MO);
  }

  /// Returns true if \p MI becomes an IMPLICIT_DEF.
  bool becomesImplicitDef(const MachineInstr &MI) const {
    return NewImplicitDefs.contains(&MI);
  }

  /// Returns true if \p MI is, or becomes, an IMPLICIT_DEF that is erased.
  bool isErased(const MachineInstr &MI) const {
    return ErasedImplicitDefs.contains(&MI);
  }
};

class ProcessImplicitDefsPass
    : public RequiredPassInfoMixin<ProcessImplicitDefsPass> {
public:
  LLVM_ABI PreservedAnalyses run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM);
  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties().setIsSSA();
  }
};

} // namespace llvm

#endif // LLVM_CODEGEN_PROCESSIMPLICITDEFS_H
