//===- DependencyGraph.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the dependency graph used by the vectorizer's instruction
// scheduler.
//
// The nodes of the graph are objects of the `DGNode` class. Each `DGNode`
// object points to an instruction.
// The edges between `DGNode`s are implicitly defined by an ordered set of
// predecessor nodes, to save memory.
// Finally the whole dependency graph is an object of the `DependencyGraph`
// class, which also provides the API for creating/extending the graph from
// input Sandbox IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_DEPENDENCYGRAPH_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_DEPENDENCYGRAPH_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Interval.h"

namespace llvm::sandboxir {

/// A DependencyGraph Node that points to an Instruction and contains memory
/// dependency edges.
class DGNode {
  Instruction *I;
  /// Memory predecessors.
  DenseSet<DGNode *> MemPreds;
  /// This is true if this may read/write memory, or if it has some ordering
  /// constraints, like with stacksave/stackrestore and alloca/inalloca.
  bool IsMem;

public:
  DGNode(Instruction *I) : I(I) {
    IsMem = I->isMemDepCandidate() ||
            (isa<AllocaInst>(I) && cast<AllocaInst>(I)->isUsedWithInAlloca()) ||
            I->isStackSaveOrRestoreIntrinsic();
  }
  Instruction *getInstruction() const { return I; }
  void addMemPred(DGNode *PredN) { MemPreds.insert(PredN); }
  /// \Returns all memory dependency predecessors.
  iterator_range<DenseSet<DGNode *>::const_iterator> memPreds() const {
    return make_range(MemPreds.begin(), MemPreds.end());
  }
  /// \Returns true if there is a memory dependency N->this.
  bool hasMemPred(DGNode *N) const { return MemPreds.count(N); }
  /// \Returns true if this may read/write memory, or if it has some ordering
  /// constraints, like with stacksave/stackrestore and alloca/inalloca.
  bool isMem() const { return IsMem; }
#ifndef NDEBUG
  void print(raw_ostream &OS, bool PrintDeps = true) const;
  friend raw_ostream &operator<<(DGNode &N, raw_ostream &OS) {
    N.print(OS);
    return OS;
  }
  LLVM_DUMP_METHOD void dump() const;
#endif // NDEBUG
};

class DependencyGraph {
private:
  DenseMap<Instruction *, std::unique_ptr<DGNode>> InstrToNodeMap;

public:
  DependencyGraph() {}

  DGNode *getNode(Instruction *I) const {
    auto It = InstrToNodeMap.find(I);
    return It != InstrToNodeMap.end() ? It->second.get() : nullptr;
  }
  DGNode *getOrCreateNode(Instruction *I) {
    auto [It, NotInMap] = InstrToNodeMap.try_emplace(I);
    if (NotInMap)
      It->second = std::make_unique<DGNode>(I);
    return It->second.get();
  }
  /// Build/extend the dependency graph such that it includes \p Instrs. Returns
  /// the interval spanning \p Instrs.
  Interval<Instruction> extend(ArrayRef<Instruction *> Instrs);
#ifndef NDEBUG
  void print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif // NDEBUG
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_DEPENDENCYGRAPH_H
