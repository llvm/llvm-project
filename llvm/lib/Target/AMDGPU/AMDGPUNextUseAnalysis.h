//===---------------------- AMDGPUNextUseAnalysis.h  ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Next Use Analysis.
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
#include <cmath>
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

  constexpr static NextUseDistance fromLoopDepth(unsigned Depth) {
    constexpr int64_t LoopWeight = 1000;
    // FIXME: Is 24 a realistic limit?
    assert(Depth < 24 && "Loop depth exceeds limit (24)");
    int64_t v = LoopWeight * (1 << (2 * Depth));
    return NextUseDistance(v);
  }

  constexpr NextUseDistance(unsigned V) : Value(V) {}
  constexpr NextUseDistance(int V) : Value(V) {}
  constexpr NextUseDistance(const llvm::NextUseDistance &B) : Value(B.Value) {}

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

  constexpr inline NextUseDistance applyLoopWeight(unsigned Depth) const {
    NextUseDistance D = *this;
    if (Depth)
      D.Value *= fromLoopDepth(Depth).Value;
    return D;
  }

  // Extend this distance by 'Size' and reset it's depth to 'Depth'.
  constexpr NextUseDistance extend(unsigned Size, unsigned Depth) const {
    NextUseDistance D = *this;
    return D += NextUseDistance(Size).applyLoopWeight(Depth);
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

  double getRawValue() const { return Value; }

private:
  friend class AMDGPUNextUseAnalysisImpl;
  int64_t Value;
  constexpr explicit NextUseDistance(int64_t V) : Value(V) {}
};

constexpr inline NextUseDistance operator+(NextUseDistance A,
                                           const NextUseDistance &B) {
  return A += B;
}

constexpr inline NextUseDistance operator-(NextUseDistance A,
                                           const NextUseDistance &B) {
  return A -= B;
}

// Allow std::min/std::max with NextUseDistance
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

  enum CompatibilityMode { Compute, Graphics };

  CompatibilityMode getCompatibilityMode();
  void setCompatibilityMode(CompatibilityMode);

  /// \Returns the next-use distance for \p LiveReg.
  std::optional<NextUseDistance>
  getNextUseDistance(Register LiveReg, const MachineInstr &CurMI,
                     const SmallVector<const MachineOperand *> &Uses,
                     SmallVector<NextUseDistance> *Distances = nullptr,
                     const MachineOperand **UseOut = nullptr);

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

  void getUses(unsigned Register, LaneBitmask LaneMask, const MachineInstr &MI,
               SmallVector<const MachineOperand *> &Uses);
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
    : public PassInfoMixin<AMDGPUNextUseAnalysisPrinterPass> {
  raw_ostream &OS;

public:
  explicit AMDGPUNextUseAnalysisPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
  static bool isRequired() { return true; }
};

} // namespace llvm
#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUNEXTUSEANALYSIS_H
