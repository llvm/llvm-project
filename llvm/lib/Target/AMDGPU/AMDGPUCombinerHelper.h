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

  /// Match an s64 and/or where one operand is a 32-bit mask (see
  /// matchConstantIs32BitMask), used to narrow it into two s32 ops. Bails when
  /// both operands are constants, since folding the binop to a single constant
  /// is strictly better than narrowing.
  bool matchBinopWith32BitMask(Register X, Register Y) const;

  /// fmin_legacy/fmax_legacy select s1 on NaN, and on a +0.0/-0.0 tie (s1 for
  /// min, s0 for max). Returns true if that tie cannot be observed: nsz on
  /// \p MI, or a known non-logical-zero \p LHS or \p RHS.
  bool canIgnoreLegacyMinMaxTies(const MachineInstr &MI, Register LHS,
                                 Register RHS) const;

  /// Operands of an add/sub-with-carry that replaces a conditional increment
  /// or decrement: Base + K + Carry, or Base - K - Carry, where K is 0 or -1.
  struct ConditionalCarryOp {
    bool IsAdd = true;
    int64_t K = 0;
    Register Base;
    Register Carry;
  };

  /// Return true if \p Reg is a boolean that is produced in SCC or VCC: the
  /// carry-out of an add/sub, a compare, a floating-point class test, or a
  /// logic op of such booleans.
  bool isCarryLikeBool(Register Reg, unsigned Depth = 0) const;

  /// Return true if \p Reg is a single-use zext, anyext or sext of a boolean
  /// that isCarryLikeBool accepts. If so, set \p Carry to the boolean and
  /// \p IsSExt to true for a sext. An anyext is treated as a zext.
  bool isFoldableCarryExt(Register Reg, Register *Carry = nullptr,
                          bool *IsSExt = nullptr) const;

  /// Match a conditional increment or decrement that one add/sub-with-carry
  /// chain can compute:
  ///   (select c, (add x, 1), x)  -> x + 0 + c
  ///   (select c, x, (add x, 1))  -> x - (-1) - c
  ///   (select c, (add x, -1), x) -> x - 0 - c
  ///   (select c, x, (add x, -1)) -> x + (-1) + c
  ///   (add x, (zext c)), (sub x, (sext c)) -> x + 0 + c
  ///   (add x, (anyext c)), (sub x, (anyext c)) are treated as zext.
  ///   (sub x, (zext c)), (add x, (sext c)) -> x - 0 - c
  /// Before legalization only 64-bit types are matched. After legalization
  /// 16-bit and 32-bit types are matched.
  bool matchConditionalCarryOp(MachineInstr &MI,
                               ConditionalCarryOp &MatchInfo) const;
  void applyConditionalCarryOp(MachineInstr &MI,
                               const ConditionalCarryOp &MatchInfo) const;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUCOMBINERHELPER_H
