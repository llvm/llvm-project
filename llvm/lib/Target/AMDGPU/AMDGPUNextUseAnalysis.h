//===---------------------- AMDGPUNextUseAnalysis.h  ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Next Use Analysis.
//
// For each register it goes over all uses and returns the estimated distance of
// the nearest use. This will be used for selecting which registers to spill
// before register allocation.
//
// This is based on ideas from the paper:
// "Register Spilling and Live-Range Splitting for SSA-Form Programs"
// Matthias Braun and Sebastian Hack, CC'09
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUNEXTUSEANALYSIS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUNEXTUSEANALYSIS_H

#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include <limits>
#include <optional>

namespace llvm {

class AMDGPUNextUseAnalysisImpl;

//==============================================================================
// NextUseDistance - Represents a distance in the next-use analysis. Currently
// wraps a 64-bit int with special encoding for loop depth and unreachable
// distances.
//==============================================================================
class NextUseDistance {
public:
  constexpr static NextUseDistance unreachable() {
    return NextUseDistance(std::numeric_limits<int64_t>::max());
  }

  constexpr static NextUseDistance fromSize(unsigned Size, unsigned Depth) {
    return NextUseDistance(Size).applyLoopWeight(Depth);
  }

  constexpr NextUseDistance(unsigned V) : Value(V) {}
  constexpr NextUseDistance(int V) : Value(V) {}
  constexpr NextUseDistance(const NextUseDistance &B) : Value(B.Value) {}

  constexpr bool isUnreachable() const { return *this == unreachable(); }
  constexpr bool isReachable() const { return !isUnreachable(); }

  //----------------------------------------------------------------------------
  // Assignment
  //----------------------------------------------------------------------------
  constexpr NextUseDistance &operator=(const NextUseDistance &B) {
    Value = B.Value;
    return *this;
  }

  constexpr NextUseDistance &operator=(unsigned V) {
    Value = V;
    return *this;
  }

  constexpr NextUseDistance &operator=(int V) {
    Value = V;
    return *this;
  }

  //----------------------------------------------------------------------------
  // Arithmetic operators
  //----------------------------------------------------------------------------
  constexpr NextUseDistance &operator+=(const NextUseDistance &B) {
    Value += B.Value;
    return *this;
  }

  constexpr NextUseDistance &operator-=(const NextUseDistance &B) {
    Value -= B.Value;
    return *this;
  }

  constexpr NextUseDistance operator-() const {
    return NextUseDistance(-Value);
  }

  constexpr NextUseDistance applyLoopWeight() const {
    NextUseDistance W = fromLoopDepth(1);
    if (W.isUnreachable())
      return unreachable();
    constexpr int64_t MaxVal = std::numeric_limits<int64_t>::max();
    if (Value != 0 && W.Value > MaxVal / Value)
      return unreachable();
    return NextUseDistance(Value * W.Value);
  }

  //----------------------------------------------------------------------------
  // Comparison operators
  //----------------------------------------------------------------------------
  constexpr bool operator<(const NextUseDistance &B) const {
    return Value < B.Value;
  }

  constexpr bool operator>(const NextUseDistance &B) const {
    return Value > B.Value;
  }

  constexpr bool operator<=(const NextUseDistance &B) const {
    return Value <= B.Value;
  }

  constexpr bool operator>=(const NextUseDistance &B) const {
    return Value >= B.Value;
  }

  constexpr bool operator==(const NextUseDistance &B) const {
    return Value == B.Value;
  }

  constexpr bool operator!=(const NextUseDistance &B) const {
    return Value != B.Value;
  }

  //----------------------------------------------------------------------------
  // Debugging
  //----------------------------------------------------------------------------
  format_object<int64_t> fmt() const { return format("%ld", Value); }

  void print(raw_ostream &OS) const {
    if (isUnreachable())
      OS << "<unreachable>";
    else
      OS << fmt();
  }

  json::Value toJsonValue() const {
    if (isUnreachable())
      return "<unreachable>";
    return Value;
  }

  std::string toString() const {
    std::string Str;
    llvm::raw_string_ostream OS(Str);
    print(OS);
    return OS.str();
  }

  constexpr int64_t getRawValue() const { return Value; }
  using RawValueType = int64_t;

private:
  friend class AMDGPUNextUseAnalysisImpl;
  int64_t Value;
  constexpr explicit NextUseDistance(int64_t V) : Value(V) {}

  constexpr static NextUseDistance fromLoopDepth(unsigned Depth) {
    const unsigned Shift = 7 * Depth;

    // Saturate?
    if (Shift >= 63)
      return unreachable();

    // This implementation is multiplicative (f(a+b) == f(a) * f(b)) which we
    // take advantage of below in applyLoopWeight(Depth).
    return NextUseDistance(int64_t(1) << Shift);
  }

  // Semantically: apply fromLoopDepth(1) Depth times (compositional).
  //
  // Optimized to take advantage of multiplicative implementation of
  // fromLoopDepth - a single multiply by fromLoopDepth(Depth) gives the same
  // result. If fromLoopDepth is changed to a non-multiplicative formula,
  // replace the body with something like:
  //
  //   NextUseDistance D = *this;
  //   for (unsigned I = 0; I < Depth; ++I) {
  //     D = D.applyLoopWeight();
  //     if (D.isUnreachable())
  //       return unreachable();
  //   }
  //   return D;
  //
  constexpr NextUseDistance applyLoopWeight(unsigned Depth) const {
    if (!Depth)
      return *this;
    NextUseDistance W = fromLoopDepth(Depth);
    if (W.isUnreachable())
      return unreachable();
    constexpr int64_t MaxVal = std::numeric_limits<int64_t>::max();
    if (Value != 0 && W.Value > MaxVal / Value)
      return unreachable();
    return NextUseDistance(Value * W.Value);
  }
};

constexpr inline NextUseDistance operator+(NextUseDistance A,
                                           const NextUseDistance &B) {
  return A += B;
}

constexpr inline NextUseDistance operator-(NextUseDistance A,
                                           const NextUseDistance &B) {
  return A -= B;
}

constexpr inline NextUseDistance min(NextUseDistance A, NextUseDistance B) {
  return A < B ? A : B;
}

constexpr inline NextUseDistance max(NextUseDistance A, NextUseDistance B) {
  return A > B ? A : B;
}

//==============================================================================
// AMDGPUNextUseAnalysis - Provides next-use distances for live registers or
// sub-registers at a given MachineInstruction suitable for making spilling
// decisions.
//==============================================================================
class AMDGPUNextUseAnalysis {
  friend class AMDGPUNextUseAnalysisLegacyPass;
  friend class AMDGPUNextUseAnalysisPrinterLegacyPass;
  friend class AMDGPUNextUseAnalysisPass;
  friend class AMDGPUNextUseAnalysisPrinterPass;

  std::unique_ptr<AMDGPUNextUseAnalysisImpl> Impl;

  AMDGPUNextUseAnalysis(const MachineFunction *, const MachineLoopInfo *);

public:
  AMDGPUNextUseAnalysis(AMDGPUNextUseAnalysis &&Other);
  ~AMDGPUNextUseAnalysis();

  AMDGPUNextUseAnalysis &operator=(AMDGPUNextUseAnalysis &&Other);

  // Configuration flags for controlling the distance model. Defaults correspond
  // to the Graphics preset.
  struct Config {
    // Count PHI instructions as having non-zero cost (distance and block
    // size). When false, all PHIs share ID 0 and don't contribute to block
    // size.
    bool CountPhis = true;

    // Restrict inter-block distances to forward-reachable paths only.
    // When false, distances through back-edges are also considered.
    bool ForwardOnly = true;

    // Model PHI uses as belonging to their incoming edge's block, and apply
    // full loop-aware reachability filtering including intermediate-def
    // checks. When false, a simple same-block / forward-reachable check is
    // used.
    bool PreciseUseModeling = false;

    // Promote uses that are inside a loop not yet entered or inside a directly
    // nested inner loop to the end of that loop's preheader. This models the
    // assumption that a spilled value will be reloaded at the preheader rather
    // than at the actual use site. When false, direct shortest distance to the
    // use is used instead.
    bool PromoteToPreheader = false;

    /// Named presets. See note in AMDGPUNextUseAnalysis.cpp associated with
    /// 'amdgpu-next-use-analysis-config' regarding the historical context for
    /// these.
    static Config Graphics() { return {}; }
    static Config Compute() {
      Config Cfg;
      Cfg.CountPhis = false;
      Cfg.ForwardOnly = false;
      Cfg.PreciseUseModeling = true;
      Cfg.PromoteToPreheader = true;
      return Cfg;
    }
  };

  Config getConfig() const;
  void setConfig(Config);

  void getReachableUses(Register LiveReg, LaneBitmask LaneMask,
                        const MachineInstr &MI,
                        SmallVector<const MachineOperand *> &Uses) const;

  /// \Returns the shortest next-use distance from \p CurMI for \p LiveReg.
  NextUseDistance
  getShortestDistance(Register LiveReg, const MachineInstr &CurMI,
                      const SmallVector<const MachineOperand *> &Uses,
                      const MachineOperand **ShortestUseOut = nullptr,
                      SmallVector<NextUseDistance> *Distances = nullptr) const;

  struct UseDistancePair {
    const MachineOperand *Use = nullptr;
    NextUseDistance Dist = 0;
    UseDistancePair() = default;
    UseDistancePair(const MachineOperand *Use, NextUseDistance Dist)
        : Use(Use), Dist(Dist) {}
  };

  void getNextUseDistances(const DenseMap<unsigned, LaneBitmask> &LiveRegs,
                           const MachineInstr &MI, UseDistancePair &Furthest,
                           UseDistancePair *FurthestSubreg = nullptr,
                           DenseMap<const MachineOperand *, UseDistancePair>
                               *RelevantUses = nullptr) const;
};

//==============================================================================
// AMDGPUNextUseAnalysisLegacyPass - Legacy and New pass wrapper around
// AMDGPUNextUseAnalysis
//==============================================================================
class AMDGPUNextUseAnalysisLegacyPass : public MachineFunctionPass {

public:
  static char ID;

  AMDGPUNextUseAnalysisLegacyPass();

  AMDGPUNextUseAnalysis &getNextUseAnalysis() { return *NUA; }
  const AMDGPUNextUseAnalysis &getNextUseAnalysis() const { return *NUA; }
  StringRef getPassName() const override;

protected:
  bool runOnMachineFunction(MachineFunction &) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  std::unique_ptr<AMDGPUNextUseAnalysis> NUA;
};

class AMDGPUNextUseAnalysisPass
    : public AnalysisInfoMixin<AMDGPUNextUseAnalysisPass> {
  friend AnalysisInfoMixin<AMDGPUNextUseAnalysisPass>;
  static AnalysisKey Key;

public:
  using Result = AMDGPUNextUseAnalysis;
  Result run(MachineFunction &MF, MachineFunctionAnalysisManager &MFAM);
};

//==============================================================================
// AMDGPUNextUseAnalysisPrinterLegacyPass - Legacy Pass for printing
// AMDGPUNextUseAnalysis results as JSON.
//==============================================================================
class AMDGPUNextUseAnalysisPrinterLegacyPass : public MachineFunctionPass {

public:
  static char ID;

  AMDGPUNextUseAnalysisPrinterLegacyPass();

  StringRef getPassName() const override;

protected:
  bool runOnMachineFunction(MachineFunction &) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

class AMDGPUNextUseAnalysisPrinterPass
    : public RequiredPassInfoMixin<AMDGPUNextUseAnalysisPrinterPass> {
  raw_ostream &OS;

public:
  explicit AMDGPUNextUseAnalysisPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

} // namespace llvm
#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUNEXTUSEANALYSIS_H
