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

#include "llvm/Pass.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace llvm {

class FunctionPass;
class Function;
class Instruction;
class DominatorTree;
class AssumptionCache;

struct DemandedBits : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  DemandedBits();

  bool runOnFunction(Function& F) override;
  void getAnalysisUsage(AnalysisUsage& AU) const override;
  void print(raw_ostream &OS, const Module *M) const override;
  
  /// Return the bits demanded from instruction I.
  APInt getDemandedBits(Instruction *I);

  /// Return true if, during analysis, I could not be reached.
  bool isInstructionDead(Instruction *I);

private:
  void performAnalysis();
  void determineLiveOperandBits(const Instruction *UserI,
                                const Instruction *I, unsigned OperandNo,
                                const APInt &AOut, APInt &AB,
                                APInt &KnownZero, APInt &KnownOne,
                                APInt &KnownZero2, APInt &KnownOne2);

  AssumptionCache *AC;
  DominatorTree *DT;
  Function *F;
  bool Analyzed;

  // The set of visited instructions (non-integer-typed only).
  SmallPtrSet<Instruction*, 32> Visited;
  DenseMap<Instruction *, APInt> AliveBits;
};

/// Create a demanded bits analysis pass.
FunctionPass *createDemandedBitsPass();

} // End llvm namespace

#endif
