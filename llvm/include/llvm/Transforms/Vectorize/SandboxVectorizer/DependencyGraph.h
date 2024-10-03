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
#include "llvm/SandboxIR/Utils.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Interval.h"

namespace llvm::sandboxir {

class DependencyGraph;
class MemDGNode;

/// SubclassIDs for isa/dyn_cast etc.
enum class DGNodeID {
  DGNode,
  MemDGNode,
};

/// A DependencyGraph Node that points to an Instruction and contains memory
/// dependency edges.
class DGNode {
protected:
  Instruction *I;
  // TODO: Use a PointerIntPair for SubclassID and I.
  /// For isa/dyn_cast etc.
  DGNodeID SubclassID;
  /// Memory predecessors.
  DenseSet<MemDGNode *> MemPreds;

  DGNode(Instruction *I, DGNodeID ID) : I(I), SubclassID(ID) {}
  friend class MemDGNode; // For constructor.

public:
  DGNode(Instruction *I) : I(I), SubclassID(DGNodeID::DGNode) {
    assert(!isMemDepCandidate(I) && "Expected Non-Mem instruction, ");
  }
  DGNode(const DGNode &Other) = delete;
  virtual ~DGNode() = default;
  /// \Returns true if this is before \p Other in program order.
  bool comesBefore(const DGNode *Other) { return I->comesBefore(Other->I); }
  /// \Returns true if \p I is a memory dependency candidate instruction.
  static bool isMemDepCandidate(Instruction *I) {
    AllocaInst *Alloca;
    return Utils::isMemDepCandidate(I) ||
           ((Alloca = dyn_cast<AllocaInst>(I)) &&
            Alloca->isUsedWithInAlloca()) ||
           Utils::isStackSaveOrRestoreIntrinsic(I);
  }

  Instruction *getInstruction() const { return I; }
  void addMemPred(MemDGNode *PredN) { MemPreds.insert(PredN); }
  /// \Returns all memory dependency predecessors.
  iterator_range<DenseSet<MemDGNode *>::const_iterator> memPreds() const {
    return make_range(MemPreds.begin(), MemPreds.end());
  }
  /// \Returns true if there is a memory dependency N->this.
  bool hasMemPred(DGNode *N) const {
    if (auto *MN = dyn_cast<MemDGNode>(N))
      return MemPreds.count(MN);
    return false;
  }

#ifndef NDEBUG
  virtual void print(raw_ostream &OS, bool PrintDeps = true) const;
  friend raw_ostream &operator<<(DGNode &N, raw_ostream &OS) {
    N.print(OS);
    return OS;
  }
  LLVM_DUMP_METHOD void dump() const;
#endif // NDEBUG
};

/// A DependencyGraph Node for instructions that may read/write memory, or have
/// some ordering constraints, like with stacksave/stackrestore and
/// alloca/inalloca.
class MemDGNode final : public DGNode {
  MemDGNode *PrevMemN = nullptr;
  MemDGNode *NextMemN = nullptr;

  void setNextNode(MemDGNode *N) { NextMemN = N; }
  void setPrevNode(MemDGNode *N) { PrevMemN = N; }
  friend class DependencyGraph; // For setNextNode(), setPrevNode().

public:
  MemDGNode(Instruction *I) : DGNode(I, DGNodeID::MemDGNode) {
    assert(isMemDepCandidate(I) && "Expected Mem instruction!");
  }
  static bool classof(const DGNode *Other) {
    return Other->SubclassID == DGNodeID::MemDGNode;
  }
  /// \Returns the previous Mem DGNode in instruction order.
  MemDGNode *getPrevNode() const { return PrevMemN; }
  /// \Returns the next Mem DGNode in instruction order.
  MemDGNode *getNextNode() const { return NextMemN; }
};

/// Convenience builders for a MemDGNode interval.
class MemDGNodeIntervalBuilder {
public:
  /// Given \p Instrs it finds their closest mem nodes in the interval and
  /// returns the corresponding mem range. Note: BotN (or its neighboring mem
  /// node) is included in the range.
  static Interval<MemDGNode> make(const Interval<Instruction> &Instrs,
                                  DependencyGraph &DAG);
  static Interval<MemDGNode> makeEmpty() { return {}; }
};

class DependencyGraph {
private:
  DenseMap<Instruction *, std::unique_ptr<DGNode>> InstrToNodeMap;
  /// The DAG spans across all instructions in this interval.
  Interval<Instruction> DAGInterval;

public:
  DependencyGraph() {}

  DGNode *getNode(Instruction *I) const {
    auto It = InstrToNodeMap.find(I);
    return It != InstrToNodeMap.end() ? It->second.get() : nullptr;
  }
  /// Like getNode() but returns nullptr if \p I is nullptr.
  DGNode *getNodeOrNull(Instruction *I) const {
    if (I == nullptr)
      return nullptr;
    return getNode(I);
  }
  DGNode *getOrCreateNode(Instruction *I) {
    auto [It, NotInMap] = InstrToNodeMap.try_emplace(I);
    if (NotInMap) {
      if (DGNode::isMemDepCandidate(I))
        It->second = std::make_unique<MemDGNode>(I);
      else
        It->second = std::make_unique<DGNode>(I);
    }
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
