//===- DependencyGraph.h ----------------------------------------*- C++ -*-===//
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
class SchedBundle;

/// SubclassIDs for isa/dyn_cast etc.
enum class DGNodeID {
  DGNode,
  MemDGNode,
};

class DGNode;
class MemDGNode;
class DependencyGraph;

/// While OpIt points to a Value that is not an Instruction keep incrementing
/// it. \Returns the first iterator that points to an Instruction, or end.
[[nodiscard]] static User::op_iterator skipNonInstr(User::op_iterator OpIt,
                                                    User::op_iterator OpItE) {
  while (OpIt != OpItE && !isa<Instruction>((*OpIt).get()))
    ++OpIt;
  return OpIt;
}

/// Iterate over both def-use and mem dependencies.
class PredIterator {
  User::op_iterator OpIt;
  User::op_iterator OpItE;
  DenseSet<MemDGNode *>::iterator MemIt;
  DGNode *N = nullptr;
  DependencyGraph *DAG = nullptr;

  PredIterator(const User::op_iterator &OpIt, const User::op_iterator &OpItE,
               const DenseSet<MemDGNode *>::iterator &MemIt, DGNode *N,
               DependencyGraph &DAG)
      : OpIt(OpIt), OpItE(OpItE), MemIt(MemIt), N(N), DAG(&DAG) {}
  PredIterator(const User::op_iterator &OpIt, const User::op_iterator &OpItE,
               DGNode *N, DependencyGraph &DAG)
      : OpIt(OpIt), OpItE(OpItE), N(N), DAG(&DAG) {}
  friend class DGNode;    // For constructor
  friend class MemDGNode; // For constructor

public:
  using difference_type = std::ptrdiff_t;
  using value_type = DGNode *;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::input_iterator_tag;
  value_type operator*();
  PredIterator &operator++();
  PredIterator operator++(int) {
    auto Copy = *this;
    ++(*this);
    return Copy;
  }
  bool operator==(const PredIterator &Other) const;
  bool operator!=(const PredIterator &Other) const { return !(*this == Other); }
};

/// A DependencyGraph Node that points to an Instruction and contains memory
/// dependency edges.
class DGNode {
protected:
  Instruction *I;
  // TODO: Use a PointerIntPair for SubclassID and I.
  /// For isa/dyn_cast etc.
  DGNodeID SubclassID;
  /// The number of unscheduled successors.
  unsigned UnscheduledSuccs = 0;
  /// This is true if this node has been scheduled.
  bool Scheduled = false;
  /// The scheduler bundle that this node belongs to.
  SchedBundle *SB = nullptr;

  void setSchedBundle(SchedBundle &SB) { this->SB = &SB; }
  void clearSchedBundle() { this->SB = nullptr; }
  friend class SchedBundle; // For setSchedBundle(), clearSchedBundle().

  DGNode(Instruction *I, DGNodeID ID) : I(I), SubclassID(ID) {}
  friend class MemDGNode;       // For constructor.
  friend class DependencyGraph; // For UnscheduledSuccs

public:
  DGNode(Instruction *I) : I(I), SubclassID(DGNodeID::DGNode) {
    assert(!isMemDepNodeCandidate(I) && "Expected Non-Mem instruction, ");
  }
  DGNode(const DGNode &Other) = delete;
  virtual ~DGNode();
  /// \Returns the number of unscheduled successors.
  unsigned getNumUnscheduledSuccs() const { return UnscheduledSuccs; }
  void decrUnscheduledSuccs() {
    assert(UnscheduledSuccs > 0 && "Counting error!");
    --UnscheduledSuccs;
  }
  /// \Returns true if all dependent successors have been scheduled.
  bool ready() const { return UnscheduledSuccs == 0; }
  /// \Returns true if this node has been scheduled.
  bool scheduled() const { return Scheduled; }
  void setScheduled(bool NewVal) { Scheduled = NewVal; }
  /// \Returns the scheduling bundle that this node belongs to, or nullptr.
  SchedBundle *getSchedBundle() const { return SB; }
  /// \Returns true if this is before \p Other in program order.
  bool comesBefore(const DGNode *Other) { return I->comesBefore(Other->I); }
  using iterator = PredIterator;
  virtual iterator preds_begin(DependencyGraph &DAG) {
    return PredIterator(skipNonInstr(I->op_begin(), I->op_end()), I->op_end(),
                        this, DAG);
  }
  virtual iterator preds_end(DependencyGraph &DAG) {
    return PredIterator(I->op_end(), I->op_end(), this, DAG);
  }
  iterator preds_begin(DependencyGraph &DAG) const {
    return const_cast<DGNode *>(this)->preds_begin(DAG);
  }
  iterator preds_end(DependencyGraph &DAG) const {
    return const_cast<DGNode *>(this)->preds_end(DAG);
  }
  /// \Returns a range of DAG predecessors nodes. If this is a MemDGNode then
  /// this will also include the memory dependency predecessors.
  /// Please note that this can include the same node more than once, if for
  /// example it's both a use-def predecessor and a mem dep predecessor.
  iterator_range<iterator> preds(DependencyGraph &DAG) const {
    return make_range(preds_begin(DAG), preds_end(DAG));
  }

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

#ifndef NDEBUG
  virtual void print(raw_ostream &OS, bool PrintDeps = true) const;
  friend raw_ostream &operator<<(raw_ostream &OS, DGNode &N) {
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
  /// Memory predecessors.
  DenseSet<MemDGNode *> MemPreds;
  friend class PredIterator; // For MemPreds.

  void setNextNode(MemDGNode *N) { NextMemN = N; }
  void setPrevNode(MemDGNode *N) { PrevMemN = N; }
  friend class DependencyGraph; // For setNextNode(), setPrevNode().
  void detachFromChain() {
    if (PrevMemN != nullptr)
      PrevMemN->NextMemN = NextMemN;
    if (NextMemN != nullptr)
      NextMemN->PrevMemN = PrevMemN;
    PrevMemN = nullptr;
    NextMemN = nullptr;
  }

public:
  MemDGNode(Instruction *I) : DGNode(I, DGNodeID::MemDGNode) {
    assert(isMemDepNodeCandidate(I) && "Expected Mem instruction!");
  }
  static bool classof(const DGNode *Other) {
    return Other->SubclassID == DGNodeID::MemDGNode;
  }
  iterator preds_begin(DependencyGraph &DAG) override {
    auto OpEndIt = I->op_end();
    return PredIterator(skipNonInstr(I->op_begin(), OpEndIt), OpEndIt,
                        MemPreds.begin(), this, DAG);
  }
  iterator preds_end(DependencyGraph &DAG) override {
    return PredIterator(I->op_end(), I->op_end(), MemPreds.end(), this, DAG);
  }
  /// \Returns the previous Mem DGNode in instruction order.
  MemDGNode *getPrevNode() const { return PrevMemN; }
  /// \Returns the next Mem DGNode in instruction order.
  MemDGNode *getNextNode() const { return NextMemN; }
  /// Adds the mem dependency edge PredN->this. This also increments the
  /// UnscheduledSuccs counter of the predecessor if this node has not been
  /// scheduled.
  void addMemPred(MemDGNode *PredN) {
    [[maybe_unused]] auto Inserted = MemPreds.insert(PredN).second;
    assert(Inserted && "PredN already exists!");
    if (!Scheduled) {
      ++PredN->UnscheduledSuccs;
    }
  }
  /// \Returns true if there is a memory dependency N->this.
  bool hasMemPred(DGNode *N) const {
    if (auto *MN = dyn_cast<MemDGNode>(N))
      return MemPreds.count(MN);
    return false;
  }
  /// \Returns all memory dependency predecessors. Used by tests.
  iterator_range<DenseSet<MemDGNode *>::const_iterator> memPreds() const {
    return make_range(MemPreds.begin(), MemPreds.end());
  }
#ifndef NDEBUG
  virtual void print(raw_ostream &OS, bool PrintDeps = true) const override;
#endif // NDEBUG
};

/// Convenience builders for a MemDGNode interval.
class MemDGNodeIntervalBuilder {
public:
  /// Scans the instruction chain in \p Intvl top-down, returning the top-most
  /// MemDGNode, or nullptr.
  static MemDGNode *getTopMemDGNode(const Interval<Instruction> &Intvl,
                                    const DependencyGraph &DAG);
  /// Scans the instruction chain in \p Intvl bottom-up, returning the
  /// bottom-most MemDGNode, or nullptr.
  static MemDGNode *getBotMemDGNode(const Interval<Instruction> &Intvl,
                                    const DependencyGraph &DAG);
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

  Context *Ctx = nullptr;
  std::optional<Context::CallbackID> CreateInstrCB;
  std::optional<Context::CallbackID> EraseInstrCB;
  std::optional<Context::CallbackID> MoveInstrCB;

  std::unique_ptr<BatchAAResults> BatchAA;

  enum class DependencyType {
    ReadAfterWrite,  ///> Memory dependency write -> read
    WriteAfterWrite, ///> Memory dependency write -> write
    WriteAfterRead,  ///> Memory dependency read -> write
    Control,         ///> Control-related dependency, like with PHI/Terminator
    Other,           ///> Currently used for stack related instrs
    None,            ///> No memory/other dependency
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
  void scanAndAddDeps(MemDGNode &DstN, const Interval<MemDGNode> &SrcScanRange);

  /// Sets the UnscheduledSuccs of all DGNodes in \p NewInterval based on
  /// def-use edges.
  void setDefUseUnscheduledSuccs(const Interval<Instruction> &NewInterval);

  /// Create DAG nodes for instrs in \p NewInterval and update the MemNode
  /// chain.
  void createNewNodes(const Interval<Instruction> &NewInterval);

  /// Helper for `notify*Instr()`. \Returns the first MemDGNode that comes
  /// before \p N, including or excluding \p N based on \p IncludingN, or
  /// nullptr if not found.
  MemDGNode *getMemDGNodeBefore(DGNode *N, bool IncludingN) const;
  /// Helper for `notifyMoveInstr()`. \Returns the first MemDGNode that comes
  /// after \p N, including or excluding \p N based on \p IncludingN, or nullptr
  /// if not found.
  MemDGNode *getMemDGNodeAfter(DGNode *N, bool IncludingN) const;

  /// Called by the callbacks when a new instruction \p I has been created.
  void notifyCreateInstr(Instruction *I);
  /// Called by the callbacks when instruction \p I is about to get
  /// deleted.
  void notifyEraseInstr(Instruction *I);
  /// Called by the callbacks when instruction \p I is about to be moved to
  /// \p To.
  void notifyMoveInstr(Instruction *I, const BBIterator &To);

public:
  /// This constructor also registers callbacks.
  DependencyGraph(AAResults &AA, Context &Ctx)
      : Ctx(&Ctx), BatchAA(std::make_unique<BatchAAResults>(AA)) {
    CreateInstrCB = Ctx.registerCreateInstrCallback(
        [this](Instruction *I) { notifyCreateInstr(I); });
    EraseInstrCB = Ctx.registerEraseInstrCallback(
        [this](Instruction *I) { notifyEraseInstr(I); });
    MoveInstrCB = Ctx.registerMoveInstrCallback(
        [this](Instruction *I, const BBIterator &To) {
          notifyMoveInstr(I, To);
        });
  }
  ~DependencyGraph() {
    if (CreateInstrCB)
      Ctx->unregisterCreateInstrCallback(*CreateInstrCB);
    if (EraseInstrCB)
      Ctx->unregisterEraseInstrCallback(*EraseInstrCB);
    if (MoveInstrCB)
      Ctx->unregisterMoveInstrCallback(*MoveInstrCB);
  }

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
  /// the range of instructions added to the DAG.
  Interval<Instruction> extend(ArrayRef<Instruction *> Instrs);
  /// \Returns the range of instructions included in the DAG.
  Interval<Instruction> getInterval() const { return DAGInterval; }
  void clear() {
    InstrToNodeMap.clear();
    DAGInterval = {};
  }
#ifndef NDEBUG
  void print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif // NDEBUG
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_DEPENDENCYGRAPH_H
