#ifndef __YK_LIVENESS_H
#define __YK_LIVENESS_H

#include "llvm/IR/Instructions.h"

#include <map>
#include <set>

using namespace llvm;

namespace llvm {

// A liveness analysis for LLVM IR.
//
// This is based on the algorithm shown in Chapter 10 of the book:
//
//   Modern Compiler Implementation in Java (2nd edition)
//   by Andrew W. Appel
class LivenessAnalysis {
  std::map<Instruction *, std::set<Value *>> In;

  /// Find the successor instructions of the specified instruction.
  std::set<Instruction *> getSuccessorInstructions(Instruction *I);

  /// Replaces the set `S` with the set `R`, returning if the set changed.
  bool updateValueSet(std::set<Value *> &S, const std::set<Value *> &R);

public:
  LivenessAnalysis(Function *Func);

  /// Returns the set of live variables immediately before the specified
  /// instruction.
  std::set<Value *> getLiveVarsBefore(Instruction *I);
};

} // namespace llvm

#endif
