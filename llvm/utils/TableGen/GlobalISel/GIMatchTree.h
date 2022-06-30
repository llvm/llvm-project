//===- GIMatchTree.h - A decision tree to match GIMatchDag's --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_GIMATCHTREE_H
#define LLVM_UTILS_TABLEGEN_GIMATCHTREE_H

#include "GIMatchDag.h"
#include "llvm/ADT/BitVector.h"

namespace llvm {
class raw_ostream;

class GIMatchTreeBuilder;
class GIMatchTreePartitioner;

/// Describes the binding of a variable to the matched MIR
class GIMatchTreeVariableBinding {
  /// The name of the variable described by this binding.
  StringRef Name;
  // The matched instruction it is bound to. 
  unsigned InstrID;
  // The matched operand (if appropriate) it is bound to. 
  Optional<unsigned> OpIdx;

public:
  GIMatchTreeVariableBinding(StringRef Name, unsigned InstrID,
                             Optional<unsigned> OpIdx = None)
      : Name(Name), InstrID(InstrID), OpIdx(OpIdx) {}

  bool isInstr() const { return !OpIdx; }
  StringRef getName() const { return Name; }
  unsigned getInstrID() const { return InstrID; }
  unsigned getOpIdx() const {
    assert(OpIdx && "Is not an operand binding");
    return *OpIdx;
  }
};

/// Associates a matchable with a leaf of the decision tree.
class GIMatchTreeLeafInfo {
public:
  using const_var_binding_iterator =
      std::vector<GIMatchTreeVariableBinding>::const_iterator;
  using UntestedPredicatesTy = SmallVector<const GIMatchDagPredicate *, 1>;
  using const_untested_predicates_iterator = UntestedPredicatesTy::const_iterator;

protected:
  /// A name for the matchable. This is primarily for debugging.
  StringRef Name;
  /// Where rules have multiple roots, this is which root we're starting from.
  unsigned RootIdx;
  /// Opaque data the caller of the tree building code understands.
  void *Data;
  /// Has the decision tree covered every edge traversal? If it hasn't then this
  /// is an unrecoverable error indicating there's something wrong with the
  /// partitioners.
  bool IsFullyTraversed;
  /// Has the decision tree covered every predicate test? If it has, then
  /// subsequent matchables on the same leaf are unreachable. If it hasn't, the
  /// code that requested the GIMatchTree is responsible for finishing off any
  /// remaining predicates.
  bool IsFullyTested;
  /// The variable bindings associated with this leaf so far.
  std::vector<GIMatchTreeVariableBinding> VarBindings;
  /// Any predicates left untested by the time we reach this leaf.
  UntestedPredicatesTy UntestedPredicates;

public:
  GIMatchTreeLeafInfo() { llvm_unreachable("Cannot default-construct"); }
  GIMatchTreeLeafInfo(StringRef Name, unsigned RootIdx, void *Data)
      : Name(Name), RootIdx(RootIdx), Data(Data), IsFullyTraversed(false),
        IsFullyTested(false) {}

  StringRef getName() const { return Name; }
  unsigned getRootIdx() const { return RootIdx; }
  template <class Ty> Ty *getTargetData() const {
    return static_cast<Ty *>(Data);
  }
  bool isFullyTraversed() const { return IsFullyTraversed; }
  void setIsFullyTraversed(bool V) { IsFullyTraversed = V; }
  bool isFullyTested() const { return IsFullyTested; }
  void setIsFullyTested(bool V) { IsFullyTested = V; }

  void bindInstrVariable(StringRef Name, unsigned InstrID) {
    VarBindings.emplace_back(Name, InstrID);
  }
  void bindOperandVariable(StringRef Name, unsigned InstrID, unsigned OpIdx) {
    VarBindings.emplace_back(Name, InstrID, OpIdx);
  }

  const_var_binding_iterator var_bindings_begin() const {
    return VarBindings.begin();
  }
  const_var_binding_iterator var_bindings_end() const {
    return VarBindings.end();
  }
  iterator_range<const_var_binding_iterator> var_bindings() const {
    return make_range(VarBindings.begin(), VarBindings.end());
  }
  iterator_range<const_untested_predicates_iterator> untested_predicates() const {
    return make_range(UntestedPredicates.begin(), UntestedPredicates.end());
  }
  void addUntestedPredicate(const GIMatchDagPredicate *P) {
    UntestedPredicates.push_back(P);
  }
};

/// The nodes of a decision tree used to perform the match.
/// This will be used to generate the C++ code or state machine equivalent.
///
/// It should be noted that some nodes of this tree (most notably nodes handling
/// def -> use edges) will need to iterate over several possible matches. As
/// such, code generated from this will sometimes need to support backtracking.
class GIMatchTree {
  using LeafVector = std::vector<GIMatchTreeLeafInfo>;

  /// The partitioner that has been chosen for this node. This may be nullptr if
  /// a partitioner hasn't been chosen yet or if the node is a leaf.
  std::unique_ptr<GIMatchTreePartitioner> Partitioner;
  /// All the leaves that are possible for this node of the tree.
  /// Note: This should be emptied after the tree is built when there are
  /// children but this currently isn't done to aid debuggability of the DOT
  /// graph for the decision tree.
  LeafVector PossibleLeaves;
  /// The children of this node. The index into this array must match the index
  /// chosen by the partitioner.
  std::vector<GIMatchTree> Children;

  void writeDOTGraphNode(raw_ostream &OS) const;
  void writeDOTGraphEdges(raw_ostream &OS) const;

public:
  void writeDOTGraph(raw_ostream &OS) const;

  void setNumChildren(unsigned Num) { Children.resize(Num); }
  void addPossibleLeaf(const GIMatchTreeLeafInfo &V, bool IsFullyTraversed,
                       bool IsFullyTested) {
    PossibleLeaves.push_back(V);
    PossibleLeaves.back().setIsFullyTraversed(IsFullyTraversed);
    PossibleLeaves.back().setIsFullyTested(IsFullyTested);
  }
  void dropLeavesAfter(size_t Length) {
    if (PossibleLeaves.size() > Length)
      PossibleLeaves.resize(Length);
  }
  void setPartitioner(std::unique_ptr<GIMatchTreePartitioner> &&V) {
    Partitioner = std::move(V);
  }
  GIMatchTreePartitioner *getPartitioner() const { return Partitioner.get(); }

  std::vector<GIMatchTree>::iterator children_begin() {
    return Children.begin();
  }
  std::vector<GIMatchTree>::iterator children_end() { return Children.end(); }
  iterator_range<std::vector<GIMatchTree>::iterator> children() {
    return make_range(children_begin(), children_end());
  }
  std::vector<GIMatchTree>::const_iterator children_begin() const {
    return Children.begin();
  }
  std::vector<GIMatchTree>::const_iterator children_end() const {
    return Children.end();
  }
  iterator_range<std::vector<GIMatchTree>::const_iterator> children() const {
    return make_range(children_begin(), children_end());
  }

  LeafVector::const_iterator possible_leaves_begin() const {
    return PossibleLeaves.begin();
  }
  LeafVector::const_iterator possible_leaves_end() const {
    return PossibleLeaves.end();
  }
  iterator_range<LeafVector::const_iterator>
  possible_leaves() const {
    return make_range(possible_leaves_begin(), possible_leaves_end());
  }
  LeafVector::iterator possible_leaves_begin() {
    return PossibleLeaves.begin();
  }
  LeafVector::iterator possible_leaves_end() {
    return PossibleLeaves.end();
  }
  iterator_range<LeafVector::iterator> possible_leaves() {
    return make_range(possible_leaves_begin(), possible_leaves_end());
  }
};

/// Record information that is known about the instruction bound to this ID and
/// GIMatchDagInstrNode. Every rule gets its own set of
/// GIMatchTreeInstrInfo to bind the shared IDs to an instr node in its
/// DAG.
///
/// For example, if we know that there are 3 operands. We can record it here to
/// elide duplicate checks.
class GIMatchTreeInstrInfo {
  /// The instruction ID for the matched instruction.
  unsigned ID;
  /// The corresponding instruction node in the MatchDAG.
  const GIMatchDagInstr *InstrNode;

public:
  GIMatchTreeInstrInfo(unsigned ID, const GIMatchDagInstr *InstrNode)
      : ID(ID), InstrNode(InstrNode) {}

  unsigned getID() const { return ID; }
  const GIMatchDagInstr *getInstrNode() const { return InstrNode; }
};

/// Record information that is known about the operand bound to this ID, OpIdx,
/// and GIMatchDagInstrNode. Every rule gets its own set of
/// GIMatchTreeOperandInfo to bind the shared IDs to an operand of an
/// instr node from its DAG.
///
/// For example, if we know that there the operand is a register. We can record
/// it here to elide duplicate checks.
class GIMatchTreeOperandInfo {
  /// The corresponding instruction node in the MatchDAG that the operand
  /// belongs to.
  const GIMatchDagInstr *InstrNode;
  unsigned OpIdx;

public:
  GIMatchTreeOperandInfo(const GIMatchDagInstr *InstrNode, unsigned OpIdx)
      : InstrNode(InstrNode), OpIdx(OpIdx) {}

  const GIMatchDagInstr *getInstrNode() const { return InstrNode; }
  unsigned getOpIdx() const { return OpIdx; }
};

/// Represent a leaf of the match tree and any working data we need to build the
/// tree.
///
/// It's important to note that each rule can have multiple
/// GIMatchTreeBuilderLeafInfo's since the partitioners do not always partition
/// into mutually-exclusive partitions. For example:
///   R1: (FOO ..., ...)
///   R2: (oneof(FOO, BAR) ..., ...)
/// will partition by opcode into two partitions FOO=>[R1, R2], and BAR=>[R2]
///
/// As an optimization, all instructions, edges, and predicates in the DAGs are
/// numbered and tracked in BitVectors. As such, the GIMatchDAG must not be
/// modified once construction of the tree has begun.
class GIMatchTreeBuilderLeafInfo {
protected:
  GIMatchTreeBuilder &Builder;
  GIMatchTreeLeafInfo Info;
  const GIMatchDag &MatchDag;
  /// The association between GIMatchDagInstr* and GIMatchTreeInstrInfo.
  /// The primary reason for this members existence is to allow the use of
  /// InstrIDToInfo.lookup() since that requires that the value is
  /// default-constructible.
  DenseMap<const GIMatchDagInstr *, GIMatchTreeInstrInfo> InstrNodeToInfo;
  /// The instruction information for a given ID in the context of this
  /// particular leaf.
  DenseMap<unsigned, GIMatchTreeInstrInfo *> InstrIDToInfo;
  /// The operand information for a given ID and OpIdx in the context of this
  /// particular leaf.
  DenseMap<std::pair<unsigned, unsigned>, GIMatchTreeOperandInfo>
      OperandIDToInfo;

public:
  /// The remaining instrs/edges/predicates to visit
  BitVector RemainingInstrNodes;
  BitVector RemainingEdges;
  BitVector RemainingPredicates;

  // The remaining predicate dependencies for each predicate
  std::vector<BitVector> UnsatisfiedPredDepsForPred;

  /// The edges/predicates we can visit as a result of the declare*() calls we
  /// have already made. We don't need an instrs version since edges imply the
  /// instr.
  BitVector TraversableEdges;
  BitVector TestablePredicates;

  /// Map predicates from the DAG to their position in the DAG predicate
  /// iterators.
  DenseMap<GIMatchDagPredicate *, unsigned> PredicateIDs;
  /// Map predicate dependency edges from the DAG to their position in the DAG
  /// predicate dependency iterators.
  DenseMap<GIMatchDagPredicateDependencyEdge *, unsigned> PredicateDepIDs;

public:
  GIMatchTreeBuilderLeafInfo(GIMatchTreeBuilder &Builder, StringRef Name,
                             unsigned RootIdx, const GIMatchDag &MatchDag,
                             void *Data);

  StringRef getName() const { return Info.getName(); }
  GIMatchTreeLeafInfo &getInfo() { return Info; }
  const GIMatchTreeLeafInfo &getInfo() const { return Info; }
  const GIMatchDag &getMatchDag() const { return MatchDag; }
  unsigned getRootIdx() const { return Info.getRootIdx(); }

  /// Has this DAG been fully traversed. This must be true by the time the tree
  /// builder finishes.
  bool isFullyTraversed() const {
    // We don't need UnsatisfiedPredDepsForPred because RemainingPredicates
    // can't be all-zero without satisfying all the dependencies. The same is
    // almost true for Edges and Instrs but it's possible to have Instrs without
    // Edges.
    return RemainingInstrNodes.none() && RemainingEdges.none();
  }

  /// Has this DAG been fully tested. This hould be true by the time the tree
  /// builder finishes but clients can finish any untested predicates left over
  /// if it's not true.
  bool isFullyTested() const {
    // We don't need UnsatisfiedPredDepsForPred because RemainingPredicates
    // can't be all-zero without satisfying all the dependencies. The same is
    // almost true for Edges and Instrs but it's possible to have Instrs without
    // Edges.
    return RemainingInstrNodes.none() && RemainingEdges.none() &&
           RemainingPredicates.none();
  }

  const GIMatchDagInstr *getInstr(unsigned Idx) const {
    return *(MatchDag.instr_nodes_begin() + Idx);
  }
  const GIMatchDagEdge *getEdge(unsigned Idx) const {
    return *(MatchDag.edges_begin() + Idx);
  }
  GIMatchDagEdge *getEdge(unsigned Idx) {
    return *(MatchDag.edges_begin() + Idx);
  }
  const GIMatchDagPredicate *getPredicate(unsigned Idx) const {
    return *(MatchDag.predicates_begin() + Idx);
  }
  iterator_range<llvm::BitVector::const_set_bits_iterator>
  untested_instrs() const {
    return RemainingInstrNodes.set_bits();
  }
  iterator_range<llvm::BitVector::const_set_bits_iterator>
  untested_edges() const {
    return RemainingEdges.set_bits();
  }
  iterator_range<llvm::BitVector::const_set_bits_iterator>
  untested_predicates() const {
    return RemainingPredicates.set_bits();
  }

  /// Bind an instr node to the given ID and clear any blocking dependencies
  /// that were waiting for it.
  void declareInstr(const GIMatchDagInstr *Instr, unsigned ID);

  /// Bind an operand to the given ID and OpIdx and clear any blocking
  /// dependencies that were waiting for it.
  void declareOperand(unsigned InstrID, unsigned OpIdx);

  GIMatchTreeInstrInfo *getInstrInfo(unsigned ID) const {
    return InstrIDToInfo.lookup(ID);
  }

  void dump(raw_ostream &OS) const {
    OS << "Leaf " << getName() << " for root #" << getRootIdx() << "\n";
    MatchDag.print(OS);
    for (const auto &I : InstrIDToInfo)
      OS << "Declared Instr #" << I.first << "\n";
    for (const auto &I : OperandIDToInfo)
      OS << "Declared Instr #" << I.first.first << ", Op #" << I.first.second
         << "\n";
    OS << RemainingInstrNodes.count() << " untested instrs of "
       << RemainingInstrNodes.size() << "\n";
    OS << RemainingEdges.count() << " untested edges of "
       << RemainingEdges.size() << "\n";
    OS << RemainingPredicates.count() << " untested predicates of "
       << RemainingPredicates.size() << "\n";

    OS << TraversableEdges.count() << " edges could be traversed\n";
    OS << TestablePredicates.count() << " predicates could be tested\n";
  }
};

/// The tree builder has a fairly tough job. It's purpose is to merge all the
/// DAGs from the ruleset into a decision tree that walks all of them
/// simultaneously and identifies the rule that was matched. In addition to
/// that, it also needs to find the most efficient order to make decisions
/// without violating any dependencies and ensure that every DAG covers every
/// instr/edge/predicate.
class GIMatchTreeBuilder {
public:
  using LeafVec = std::vector<GIMatchTreeBuilderLeafInfo>;

protected:
  /// The leaves that the resulting decision tree will distinguish.
  LeafVec Leaves;
  /// The tree node being constructed.
  GIMatchTree *TreeNode;
  /// The builders for each subtree resulting from the current decision.
  std::vector<GIMatchTreeBuilder> SubtreeBuilders;
  /// The possible partitioners we could apply right now.
  std::vector<std::unique_ptr<GIMatchTreePartitioner>> Partitioners;
  /// The next instruction ID to allocate when requested by the chosen
  /// Partitioner.
  unsigned NextInstrID;

  /// Use any context we have stored to cull partitioners that only test things
  /// we already know. At the time of writing, there's no need to do anything
  /// here but it will become important once, for example, there is a
  /// num-operands and an opcode partitioner. This is because applying an opcode
  /// partitioner (usually) makes the number of operands known which makes
  /// additional checking pointless.
  void filterRedundantPartitioners();

  /// Evaluate the available partioners and select the best one at the moment.
  void evaluatePartitioners();

  /// Construct the current tree node.
  void runStep();

public:
  GIMatchTreeBuilder(unsigned NextInstrID) : NextInstrID(NextInstrID) {}
  GIMatchTreeBuilder(GIMatchTree *TreeNode, unsigned NextInstrID)
      : TreeNode(TreeNode), NextInstrID(NextInstrID) {}

  void addLeaf(StringRef Name, unsigned RootIdx, const GIMatchDag &MatchDag,
               void *Data) {
    Leaves.emplace_back(*this, Name, RootIdx, MatchDag, Data);
  }
  void addLeaf(const GIMatchTreeBuilderLeafInfo &L) { Leaves.push_back(L); }
  void addPartitioner(std::unique_ptr<GIMatchTreePartitioner> P) {
    Partitioners.push_back(std::move(P));
  }
  void addPartitionersForInstr(unsigned InstrIdx);
  void addPartitionersForOperand(unsigned InstrID, unsigned OpIdx);

  LeafVec &getPossibleLeaves() { return Leaves; }

  unsigned allocInstrID() { return NextInstrID++; }

  /// Construct the decision tree.
  std::unique_ptr<GIMatchTree> run();
};

/// Partitioners are the core of the tree builder and are unfortunately rather
/// tricky to write.
class GIMatchTreePartitioner {
protected:
  /// The partitions resulting from applying the partitioner to the possible
  /// leaves. The keys must be consecutive integers starting from 0. This can
  /// lead to some unfortunate situations where partitioners test a predicate
  /// and use 0 for success and 1 for failure if the ruleset encounters a
  /// success case first but is necessary to assign the partition to one of the
  /// tree nodes children. As a result, you usually need some kind of
  /// indirection to map the natural keys (e.g. ptrs/bools) to this linear
  /// sequence. The values are a bitvector indicating which leaves belong to
  /// this partition.
  DenseMap<unsigned, BitVector> Partitions;

public:
  virtual ~GIMatchTreePartitioner() {}
  virtual std::unique_ptr<GIMatchTreePartitioner> clone() const = 0;

  /// Determines which partitions the given leaves belong to. A leaf may belong
  /// to multiple partitions in which case it will be duplicated during
  /// applyForPartition().
  ///
  /// This function can be rather complicated. A few particular things to be
  /// aware of include:
  /// * One leaf can be assigned to multiple partitions when there's some
  ///   ambiguity.
  /// * Not all DAG's for the leaves may be able to perform the test. For
  ///   example, the opcode partitiioner must account for one DAG being a
  ///   superset of another such as [(ADD ..., ..., ...)], and [(MUL t, ...,
  ///   ...), (ADD ..., t, ...)]
  /// * Attaching meaning to a particular partition index will generally not
  ///   work due to the '0, 1, ..., n' requirement. You might encounter cases
  ///   where only partition 1 is seen, leaving a missing 0.
  /// * Finding a specific predicate such as the opcode predicate for a specific
  ///   instruction is non-trivial. It's often O(NumPredicates), leading to
  ///   O(NumPredicates*NumRules) when applied to the whole ruleset. The good
  ///   news there is that n is typically small thanks to predicate dependencies
  ///   limiting how many are testable at once. Also, with opcode and type
  ///   predicates being so frequent the value of m drops very fast too. It
  ///   wouldn't be terribly surprising to see a 10k ruleset drop down to an
  ///   average of 100 leaves per partition after a single opcode partitioner.
  /// * The same goes for finding specific edges. The need to traverse them in
  ///   dependency order dramatically limits the search space at any given
  ///   moment.
  /// * If you need to add a leaf to all partitions, make sure you don't forget
  ///   them when adding partitions later.
  virtual void repartition(GIMatchTreeBuilder::LeafVec &Leaves) = 0;

  /// Delegate the leaves for a given partition to the corresponding subbuilder,
  /// update any recorded context for this partition (e.g. allocate instr id's
  /// for instrs recorder by the current node), and clear any blocking
  /// dependencies this partitioner resolved.
  virtual void applyForPartition(unsigned PartitionIdx,
                                 GIMatchTreeBuilder &Builder,
                                 GIMatchTreeBuilder &SubBuilder) = 0;

  /// Return a BitVector indicating which leaves should be transferred to the
  /// specified partition. Note that the same leaf can be indicated for multiple
  /// partitions.
  BitVector getPossibleLeavesForPartition(unsigned Idx) {
    const auto &I = Partitions.find(Idx);
    assert(I != Partitions.end() && "Requested non-existant partition");
    return I->second;
  }

  size_t getNumPartitions() const { return Partitions.size(); }
  size_t getNumLeavesWithDupes() const {
    size_t S = 0;
    for (const auto &P : Partitions)
      S += P.second.size();
    return S;
  }

  /// Emit a brief description of the partitioner suitable for debug printing or
  /// use in a DOT graph.
  virtual void emitDescription(raw_ostream &OS) const = 0;
  /// Emit a label for the given partition suitable for debug printing or use in
  /// a DOT graph.
  virtual void emitPartitionName(raw_ostream &OS, unsigned Idx) const = 0;

  /// Emit a long description of how the partitioner partitions the leaves.
  virtual void emitPartitionResults(raw_ostream &OS) const = 0;

  /// Generate code to select between partitions based on the MIR being matched.
  /// This is typically a switch statement that picks a partition index.
  virtual void generatePartitionSelectorCode(raw_ostream &OS,
                                             StringRef Indent) const = 0;
};

/// Partition according to the opcode of the instruction.
///
/// Numbers CodeGenInstr ptrs for use as partition ID's. One special partition,
/// nullptr, represents the case where the instruction isn't known.
///
/// * If the opcode can be tested and is a single opcode, create the partition
///   for that opcode and assign the leaf to it. This partition no longer needs
///   to test the opcode, and many details about the instruction will usually
///   become known (e.g. number of operands for non-variadic instrs) via the
///   CodeGenInstr ptr.
/// * (not implemented yet) If the opcode can be tested and is a choice of
///   opcodes, then the leaf can be treated like the single-opcode case but must
///   be added to all relevant partitions and not quite as much becomes known as
///   a result. That said, multiple-choice opcodes are likely similar enough
///   (because if they aren't then handling them together makes little sense)
///   that plenty still becomes known. The main implementation issue with this
///   is having a description to represent the commonality between instructions.
/// * If the opcode is not tested, the leaf must be added to all partitions
///   including the wildcard nullptr partition. What becomes known as a result
///   varies between partitions.
/// * If the instruction to be tested is not declared then add the leaf to all
///   partitions. This occurs when we encounter one rule that is a superset of
///   the other and we are still matching the remainder of the superset. The
///   result is that the cases that don't match the superset will match the
///   subset rule, while the ones that do match the superset will match either
///   (which one is algorithm dependent but will usually be the superset).
class GIMatchTreeOpcodePartitioner : public GIMatchTreePartitioner {
  unsigned InstrID;
  DenseMap<const CodeGenInstruction *, unsigned> InstrToPartition;
  std::vector<const CodeGenInstruction *> PartitionToInstr;
  std::vector<BitVector> TestedPredicates;

public:
  GIMatchTreeOpcodePartitioner(unsigned InstrID) : InstrID(InstrID) {}

  std::unique_ptr<GIMatchTreePartitioner> clone() const override {
    return std::make_unique<GIMatchTreeOpcodePartitioner>(*this);
  }

  void emitDescription(raw_ostream &OS) const override {
    OS << "MI[" << InstrID << "].getOpcode()";
  }

  void emitPartitionName(raw_ostream &OS, unsigned Idx) const override;

  void repartition(GIMatchTreeBuilder::LeafVec &Leaves) override;
  void applyForPartition(unsigned Idx, GIMatchTreeBuilder &SubBuilder,
                         GIMatchTreeBuilder &Builder) override;

  void emitPartitionResults(raw_ostream &OS) const override;

  void generatePartitionSelectorCode(raw_ostream &OS,
                                     StringRef Indent) const override;
};

class GIMatchTreeVRegDefPartitioner : public GIMatchTreePartitioner {
  unsigned NewInstrID = -1;
  unsigned InstrID;
  unsigned OpIdx;
  std::vector<BitVector> TraversedEdges;
  DenseMap<unsigned, unsigned> ResultToPartition;
  BitVector PartitionToResult;

  void addToPartition(bool Result, unsigned LeafIdx);

public:
  GIMatchTreeVRegDefPartitioner(unsigned InstrID, unsigned OpIdx)
      : InstrID(InstrID), OpIdx(OpIdx) {}

  std::unique_ptr<GIMatchTreePartitioner> clone() const override {
    return std::make_unique<GIMatchTreeVRegDefPartitioner>(*this);
  }

  void emitDescription(raw_ostream &OS) const override {
    OS << "MI[" << NewInstrID << "] = getVRegDef(MI[" << InstrID
       << "].getOperand(" << OpIdx << "))";
  }

  void emitPartitionName(raw_ostream &OS, unsigned Idx) const override {
    bool Result = PartitionToResult[Idx];
    if (Result)
      OS << "true";
    else
      OS << "false";
  }

  void repartition(GIMatchTreeBuilder::LeafVec &Leaves) override;
  void applyForPartition(unsigned PartitionIdx, GIMatchTreeBuilder &Builder,
                         GIMatchTreeBuilder &SubBuilder) override;
  void emitPartitionResults(raw_ostream &OS) const override;

  void generatePartitionSelectorCode(raw_ostream &OS,
                                     StringRef Indent) const override;
};

} // end namespace llvm
#endif // ifndef LLVM_UTILS_TABLEGEN_GIMATCHTREE_H
