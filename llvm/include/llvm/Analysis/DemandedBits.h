//===-- llvm/Analysis/DemandedBits.h - Determine demanded bits --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements a demanded bits analysis. A demanded bit is one that
// contributes to a result; bits that are not demanded can be either zero or
// one without affecting control or data flow. For example in this sequence:
//
//   %1 = add i32 %x, %y
//   %2 = trunc i32 %1 to i16
//
// Only the lowest 16 bits of %1 are demanded; the rest are removed by the
// trunc.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DEMANDED_BITS_H
#define LLVM_ANALYSIS_DEMANDED_BITS_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class FunctionPass;
class Function;
class Instruction;
class DominatorTree;
class AssumptionCache;
struct KnownBits;

class DemandedBits {
public:
  DemandedBits(Function &F, AssumptionCache &AC, DominatorTree &DT) :
    F(F), AC(AC), DT(DT), Analyzed(false) {}

  /// Return the bits demanded from instruction I.
  APInt getDemandedBits(Instruction *I);

  /// Return true if, during analysis, I could not be reached.
  bool isInstructionDead(Instruction *I);
  
  void print(raw_ostream &OS);

private:
  Function &F;
  AssumptionCache &AC;
  DominatorTree &DT;

  void performAnalysis();
  void determineLiveOperandBits(const Instruction *UserI,
    const Instruction *I, unsigned OperandNo,
    const APInt &AOut, APInt &AB,
    KnownBits &Known, KnownBits &Known2);

  bool Analyzed;

  // The set of visited instructions (non-integer-typed only).
  SmallPtrSet<Instruction*, 32> Visited;
  DenseMap<Instruction *, APInt> AliveBits;
};

class DemandedBitsWrapperPass : public FunctionPass {
private:
  mutable Optional<DemandedBits> DB;
public:
  static char ID; // Pass identification, replacement for typeid
  DemandedBitsWrapperPass();

  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  
  /// Clean up memory in between runs
  void releaseMemory() override;
  
  DemandedBits &getDemandedBits() { return *DB; }

  void print(raw_ostream &OS, const Module *M) const override;
};

/// An analysis that produces \c DemandedBits for a function.
class DemandedBitsAnalysis : public AnalysisInfoMixin<DemandedBitsAnalysis> {
  friend AnalysisInfoMixin<DemandedBitsAnalysis>;
  static AnalysisKey Key;

public:
  /// \brief Provide the result typedef for this analysis pass.
  typedef DemandedBits Result;

  /// \brief Run the analysis pass over a function and produce demanded bits
  /// information.
  DemandedBits run(Function &F, FunctionAnalysisManager &AM);
};

/// \brief Printer pass for DemandedBits
class DemandedBitsPrinterPass : public PassInfoMixin<DemandedBitsPrinterPass> {
  raw_ostream &OS;

public:
  explicit DemandedBitsPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// Create a demanded bits analysis pass.
FunctionPass *createDemandedBitsWrapperPass();

} // End llvm namespace

#endif
