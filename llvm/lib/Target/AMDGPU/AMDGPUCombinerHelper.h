//=== lib/CodeGen/GlobalISel/AMDGPUCombinerHelper.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This contains common combine transformations that may be used in a combine
/// pass.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUCOMBINERHELPER_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUCOMBINERHELPER_H

#include "GCNSubtarget.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"

namespace llvm {
class AMDGPUCombinerHelper : public CombinerHelper {
protected:
  const GCNSubtarget &STI;
  const SIInstrInfo &TII;

public:
  using CombinerHelper::CombinerHelper;
  AMDGPUCombinerHelper(GISelChangeObserver &Observer, MachineIRBuilder &B,
                       bool IsPreLegalize, GISelValueTracking *VT,
                       MachineDominatorTree *MDT, const LegalizerInfo *LI,
                       const GCNSubtarget &STI);

  bool matchFoldableFneg(MachineInstr &MI, MachineInstr *&MatchInfo) const;
  void applyFoldableFneg(MachineInstr &MI, MachineInstr *&MatchInfo) const;

  bool matchFoldFAbsFptrunc(MachineInstr &Fabs, MachineInstr &Fptrunc) const;
  void applyFoldFAbsFptrunc(MachineInstr &Fabs, MachineInstr &Fptrunc) const;

  bool matchExpandPromotedF16FMed3(MachineInstr &MI, Register Src0,
                                   Register Src1, Register Src2) const;
  void applyExpandPromotedF16FMed3(MachineInstr &MI, Register Src0,
                                   Register Src1, Register Src2) const;

  bool matchCombineFmulWithSelectToFldexp(
      MachineInstr &MI, MachineInstr &Sel,
      std::function<void(MachineIRBuilder &)> &MatchInfo) const;

  bool matchConstantIs32BitMask(Register Reg) const;

  /// fmin_legacy/fmax_legacy select s1 on NaN, and on a +0.0/-0.0 tie (s1 for
  /// min, s0 for max). Returns true if that tie cannot be observed: nsz on
  /// \p MI, or a known non-logical-zero \p LHS or \p RHS.
  bool canIgnoreLegacyMinMaxTies(const MachineInstr &MI, Register LHS,
                                 Register RHS) const;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUCOMBINERHELPER_H
