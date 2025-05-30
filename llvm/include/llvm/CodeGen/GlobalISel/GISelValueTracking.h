//===- llvm/CodeGen/GlobalISel/GISelValueTracking.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Provides analysis for querying information about KnownBits during GISel
/// passes.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_GISELVALUETRACKING_H
#define LLVM_CODEGEN_GLOBALISEL_GISELVALUETRACKING_H

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/GlobalISel/GISelChangeObserver.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/KnownFPClass.h"

namespace llvm {

class TargetLowering;
class DataLayout;

class GISelValueTracking : public GISelChangeObserver {
  MachineFunction &MF;
  MachineRegisterInfo &MRI;
  const TargetLowering &TL;
  const DataLayout &DL;
  unsigned MaxDepth;
  /// Cache maintained during a computeKnownBits request.
  SmallDenseMap<Register, KnownBits, 16> ComputeKnownBitsCache;

  void computeKnownBitsMin(Register Src0, Register Src1, KnownBits &Known,
                           const APInt &DemandedElts, unsigned Depth = 0);

  unsigned computeNumSignBitsMin(Register Src0, Register Src1,
                                 const APInt &DemandedElts, unsigned Depth = 0);

  void computeKnownFPClass(Register R, KnownFPClass &Known,
                           FPClassTest InterestedClasses, unsigned Depth);

  void computeKnownFPClassForFPTrunc(const MachineInstr &MI,
                                     const APInt &DemandedElts,
                                     FPClassTest InterestedClasses,
                                     KnownFPClass &Known, unsigned Depth);

  void computeKnownFPClass(Register R, const APInt &DemandedElts,
                           FPClassTest InterestedClasses, KnownFPClass &Known,
                           unsigned Depth);

public:
  GISelValueTracking(MachineFunction &MF, unsigned MaxDepth = 6);
  virtual ~GISelValueTracking() = default;

  const MachineFunction &getMachineFunction() const { return MF; }

  const DataLayout &getDataLayout() const { return DL; }

  virtual void computeKnownBitsImpl(Register R, KnownBits &Known,
                                    const APInt &DemandedElts,
                                    unsigned Depth = 0);

  unsigned computeNumSignBits(Register R, const APInt &DemandedElts,
                              unsigned Depth = 0);
  unsigned computeNumSignBits(Register R, unsigned Depth = 0);

  // KnownBitsAPI
  KnownBits getKnownBits(Register R);
  KnownBits getKnownBits(Register R, const APInt &DemandedElts,
                         unsigned Depth = 0);

  // Calls getKnownBits for first operand def of MI.
  KnownBits getKnownBits(MachineInstr &MI);
  APInt getKnownZeroes(Register R);
  APInt getKnownOnes(Register R);

  /// \return true if 'V & Mask' is known to be zero in DemandedElts. We use
  /// this predicate to simplify operations downstream.
  /// Mask is known to be zero for bits that V cannot have.
  bool maskedValueIsZero(Register Val, const APInt &Mask) {
    return Mask.isSubsetOf(getKnownBits(Val).Zero);
  }

  /// \return true if the sign bit of Op is known to be zero.  We use this
  /// predicate to simplify operations downstream.
  bool signBitIsZero(Register Op);

  static void computeKnownBitsForAlignment(KnownBits &Known, Align Alignment) {
    // The low bits are known zero if the pointer is aligned.
    Known.Zero.setLowBits(Log2(Alignment));
  }

  /// \return The known alignment for the pointer-like value \p R.
  Align computeKnownAlignment(Register R, unsigned Depth = 0);

  /// Determine which floating-point classes are valid for \p V, and return them
  /// in KnownFPClass bit sets.
  ///
  /// This function is defined on values with floating-point type, values
  /// vectors of floating-point type, and arrays of floating-point type.

  /// \p InterestedClasses is a compile time optimization hint for which
  /// floating point classes should be queried. Queries not specified in \p
  /// InterestedClasses should be reliable if they are determined during the
  /// query.
  KnownFPClass computeKnownFPClass(Register R, const APInt &DemandedElts,
                                   FPClassTest InterestedClasses,
                                   unsigned Depth);

  KnownFPClass computeKnownFPClass(Register R,
                                   FPClassTest InterestedClasses = fcAllFlags,
                                   unsigned Depth = 0);

  /// Wrapper to account for known fast math flags at the use instruction.
  KnownFPClass computeKnownFPClass(Register R, const APInt &DemandedElts,
                                   uint32_t Flags,
                                   FPClassTest InterestedClasses,
                                   unsigned Depth);

  KnownFPClass computeKnownFPClass(Register R, uint32_t Flags,
                                   FPClassTest InterestedClasses,
                                   unsigned Depth);

  // Observer API. No-op for non-caching implementation.
  void erasingInstr(MachineInstr &MI) override {}
  void createdInstr(MachineInstr &MI) override {}
  void changingInstr(MachineInstr &MI) override {}
  void changedInstr(MachineInstr &MI) override {}

protected:
  unsigned getMaxDepth() const { return MaxDepth; }
};

/// To use KnownBitsInfo analysis in a pass,
/// KnownBitsInfo &Info = getAnalysis<GISelValueTrackingInfoAnalysis>().get(MF);
/// Add to observer if the Info is caching.
/// WrapperObserver.addObserver(Info);

/// Eventually add other features such as caching/ser/deserializing
/// to MIR etc. Those implementations can derive from GISelValueTracking
/// and override computeKnownBitsImpl.
class GISelValueTrackingAnalysisLegacy : public MachineFunctionPass {
  std::unique_ptr<GISelValueTracking> Info;

public:
  static char ID;
  GISelValueTrackingAnalysisLegacy() : MachineFunctionPass(ID) {
    initializeGISelValueTrackingAnalysisLegacyPass(
        *PassRegistry::getPassRegistry());
  }
  GISelValueTracking &get(MachineFunction &MF);
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;
  void releaseMemory() override { Info.reset(); }
};

class GISelValueTrackingAnalysis
    : public AnalysisInfoMixin<GISelValueTrackingAnalysis> {
  friend AnalysisInfoMixin<GISelValueTrackingAnalysis>;
  static AnalysisKey Key;

public:
  using Result = GISelValueTracking;

  Result run(MachineFunction &MF, MachineFunctionAnalysisManager &MFAM);
};

class GISelValueTrackingPrinterPass
    : public PassInfoMixin<GISelValueTrackingPrinterPass> {
  raw_ostream &OS;

public:
  GISelValueTrackingPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};
} // namespace llvm

#endif // LLVM_CODEGEN_GLOBALISEL_GISELVALUETRACKING_H
