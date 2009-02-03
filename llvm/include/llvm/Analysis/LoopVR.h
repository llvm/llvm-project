//===- LoopVR.cpp - Value Range analysis driven by loop information -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for the loop-driven value range pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOOPVR_H
#define LLVM_ANALYSIS_LOOPVR_H

#include "llvm/Pass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Support/ConstantRange.h"
#include <iosfwd>
#include <map>

namespace llvm {

/// LoopVR - This class maintains a mapping of Values to ConstantRanges.
/// There are interfaces to look up and update ranges by value, and for
/// accessing all values with range information.
///
class LoopVR : public FunctionPass {
public:
  static char ID; // Class identification, replacement for typeinfo

  LoopVR() : FunctionPass(intptr_t(&ID)) {}

  bool runOnFunction(Function &F);
  virtual void print(std::ostream &os, const Module *) const;
  void releaseMemory();

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequiredTransitive<LoopInfo>();
    AU.addRequiredTransitive<ScalarEvolution>();
    AU.setPreservesAll();
  }

  //===---------------------------------------------------------------------
  // Methods that are used to look up and update particular values.

  /// get - return the ConstantRange for a given Value of IntegerType.
  ConstantRange get(Value *V);

  /// remove - remove a value from this analysis.
  void remove(Value *V);

  /// narrow - improve our unterstanding of a Value by pointing out that it
  /// must fall within ConstantRange. To replace a range, remove it first.
  void narrow(Value *V, const ConstantRange &CR);

  //===---------------------------------------------------------------------
  // Methods that are used to iterate across all values with information.

  /// size - returns the number of Values with information
  unsigned size() const { return Map.size(); }

  typedef std::map<Value *, ConstantRange *>::iterator iterator;

  /// begin - return an iterator to the first Value, ConstantRange pair
  iterator begin() { return Map.begin(); }

  /// end - return an iterator one past the last Value, ConstantRange pair
  iterator end() { return Map.end(); }

  /// getValue - return the Value referenced by an iterator
  Value *getValue(iterator I) { return I->first; }

  /// getConstantRange - return the ConstantRange referenced by an iterator
  ConstantRange getConstantRange(iterator I) { return *I->second; }

private:
  ConstantRange compute(Value *V);

  ConstantRange getRange(SCEVHandle S, Loop *L, ScalarEvolution &SE);

  ConstantRange getRange(SCEVHandle S, SCEVHandle T, ScalarEvolution &SE);

  std::map<Value *, ConstantRange *> Map;
};

} // end llvm namespace

#endif
