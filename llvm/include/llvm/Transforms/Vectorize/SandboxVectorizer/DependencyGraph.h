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
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/SandboxIR/IntrinsicInst.h"
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
  // TODO: Move MemPreds to MemDGNode.
  /// Memory predecessors.
  DenseSet<MemDGNode *> MemPreds;

  DGNode(Instruction *I, DGNodeID ID) : I(I), SubclassID(ID) {}
  friend class MemDGNode; // For constructor.

public:
  DGNode(Instruction *I) : I(I), SubclassID(DGNodeID::DGNode) {
    assert(!isMemDepNodeCandidate(I) && "Expected Non-Mem instruction, ");
  }
  DGNode(const DGNode &Other) = delete;
  virtual ~DGNode() = default;
  /// \Returns true if this is before \p Other in program order.
  bool comesBefore(const DGNode *Other) { return I->comesBefore(Other->I); }

  static bool isStackSaveOrRestoreIntrinsic(Instruction *I) {
    if (auto *II = dyn_cast<IntrinsicInst>(I)) {
      auto IID = II->getIntrinsicID();
      return IID == Intrinsic::stackrestore || IID == Intrinsic::stacksave;
    }
    return false;
  }

  /// \Returns true if intrinsic \p I touches memory. This is used by the
  /// dependency graph.
  static bool isMemIntrinsic(IntrinsicInst *I) {
    auto IID = I->getIntrinsicID();
    return IID != Intrinsic::sideeffect && IID != Intrinsic::pseudoprobe;
  }

  /// We consider \p I as a Memory Dependency Candidate instruction if it
  /// reads/write memory or if it has side-effects. This is used by the
  /// dependency graph.
  static bool isMemDepCandidate(Instruction *I) {
    IntrinsicInst *II;
    return I->mayReadOrWriteMemory() &&
           (!(II = dyn_cast<IntrinsicInst>(I)) || isMemIntrinsic(II));
  }

  /// \Returns true if \p I is fence like. It excludes non-mem intrinsics.
  static bool isFenceLike(Instruction *I) {
    IntrinsicInst *II;
    return I->isFenceLike() &&
           (!(II = dyn_cast<IntrinsicInst>(I)) || isMemIntrinsic(II));
  }

  /// \Returns true if \p I is a memory dependency candidate instruction.
  static bool isMemDepNodeCandidate(Instruction *I) {
    AllocaInst *Alloca;
    return isMemDepCandidate(I) ||
           ((Alloca = dyn_cast<AllocaInst>(I)) &&
            Alloca->isUsedWithInAlloca()) ||
           isStackSaveOrRestoreIntrinsic(I) || isFenceLike(I);
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
    assert(isMemDepNodeCandidate(I) && "Expected Mem instruction!");
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

  std::unique_ptr<BatchAAResults> BatchAA;

  enum class DependencyType {
    RAW,   ///> Read After Write
    WAW,   ///> Write After Write
    RAR,   ///> Read After Read
    WAR,   ///> Write After Read
    CTRL,  ///> Control-related dependencies, like with PHIs/Terminators
    OTHER, ///> Currently used for stack related instrs
    NONE,  ///> No memory/other dependency
  };
  /// \Returns the dependency type depending on whether instructions may
  /// read/write memory or whether they are some specific opcode-related
  /// restrictions.
  /// Note: It does not check whether a memory dependency is actually correct,
  /// as it won't call AA. Therefore it returns the worst-case dep type.
  static DependencyType getRoughDepType(Instruction *FromI, Instruction *ToI);

  // TODO: Implement AABudget.
  /// \Returns true if there is a memory/other dependency \p SrcI->DstI.
  bool alias(Instruction *SrcI, Instruction *DstI, DependencyType DepType);

  bool hasDep(sandboxir::Instruction *SrcI, sandboxir::Instruction *DstI);

  /// Go through all mem nodes in \p SrcScanRange and try to add dependencies to
  /// \p DstN.
  void scanAndAddDeps(DGNode &DstN, const Interval<MemDGNode> &SrcScanRange);

public:
  DependencyGraph(AAResults &AA)
      : BatchAA(std::make_unique<BatchAAResults>(AA)) {}

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
      if (DGNode::isMemDepNodeCandidate(I))
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
