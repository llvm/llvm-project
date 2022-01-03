//===- GIMatchTree.cpp - A decision tree to match GIMatchDag's ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GIMatchTree.h"

#include "../CodeGenInstruction.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

#define DEBUG_TYPE "gimatchtree"

using namespace llvm;

void GIMatchTree::writeDOTGraph(raw_ostream &OS) const {
  OS << "digraph \"matchtree\" {\n";
  writeDOTGraphNode(OS);
  OS << "}\n";
}

void GIMatchTree::writeDOTGraphNode(raw_ostream &OS) const {
  OS << format("  Node%p", this) << " [shape=record,label=\"{";
  if (Partitioner) {
    Partitioner->emitDescription(OS);
    OS << "|" << Partitioner->getNumPartitions() << " partitions|";
  } else
    OS << "No partitioner|";
  bool IsFullyTraversed = true;
  bool IsFullyTested = true;
  StringRef Separator = "";
  for (const auto &Leaf : PossibleLeaves) {
    OS << Separator << Leaf.getName();
    Separator = ",";
    if (!Leaf.isFullyTraversed())
      IsFullyTraversed = false;
    if (!Leaf.isFullyTested())
      IsFullyTested = false;
  }
  if (!Partitioner && !IsFullyTraversed)
    OS << "|Not fully traversed";
  if (!Partitioner && !IsFullyTested) {
    OS << "|Not fully tested";
    if (IsFullyTraversed) {
      for (const GIMatchTreeLeafInfo &Leaf : PossibleLeaves) {
        if (Leaf.isFullyTested())
          continue;
        OS << "\\n" << Leaf.getName() << ": " << &Leaf;
        for (const GIMatchDagPredicate *P : Leaf.untested_predicates())
          OS << *P;
      }
    }
  }
  OS << "}\"";
  if (!Partitioner &&
      (!IsFullyTraversed || !IsFullyTested || PossibleLeaves.size() > 1))
    OS << ",color=red";
  OS << "]\n";
  for (const auto &C : Children)
    C.writeDOTGraphNode(OS);
  writeDOTGraphEdges(OS);
}

void GIMatchTree::writeDOTGraphEdges(raw_ostream &OS) const {
  for (const auto &Child : enumerate(Children)) {
    OS << format("  Node%p", this) << " -> " << format("Node%p", &Child.value())
       << " [label=\"#" << Child.index() << " ";
    Partitioner->emitPartitionName(OS, Child.index());
    OS << "\"]\n";
  }
}

GIMatchTreeBuilderLeafInfo::GIMatchTreeBuilderLeafInfo(
    GIMatchTreeBuilder &Builder, StringRef Name, unsigned RootIdx,
    const GIMatchDag &MatchDag, void *Data)
    : Builder(Builder), Info(Name, RootIdx, Data), MatchDag(MatchDag),
      InstrNodeToInfo(),
      RemainingInstrNodes(BitVector(MatchDag.getNumInstrNodes(), true)),
      RemainingEdges(BitVector(MatchDag.getNumEdges(), true)),
      RemainingPredicates(BitVector(MatchDag.getNumPredicates(), true)),
      TraversableEdges(MatchDag.getNumEdges()),
      TestablePredicates(MatchDag.getNumPredicates()) {
  // Number all the predicates in this DAG
  for (auto &P : enumerate(MatchDag.predicates())) {
    PredicateIDs.insert(std::make_pair(P.value(), P.index()));
  }

  // Number all the predicate dependencies in this DAG and set up a bitvector
  // for each predicate indicating the unsatisfied dependencies.
  for (auto &Dep : enumerate(MatchDag.predicate_edges())) {
    PredicateDepIDs.insert(std::make_pair(Dep.value(), Dep.index()));
  }
  UnsatisfiedPredDepsForPred.resize(MatchDag.getNumPredicates(),
                                    BitVector(PredicateDepIDs.size()));
  for (auto &Dep : enumerate(MatchDag.predicate_edges())) {
    unsigned ID = PredicateIDs.lookup(Dep.value()->getPredicate());
    UnsatisfiedPredDepsForPred[ID].set(Dep.index());
  }
}

void GIMatchTreeBuilderLeafInfo::declareInstr(const GIMatchDagInstr *Instr, unsigned ID) {
  // Record the assignment of this instr to the given ID.
  auto InfoI = InstrNodeToInfo.insert(std::make_pair(
      Instr, GIMatchTreeInstrInfo(ID, Instr)));
  InstrIDToInfo.insert(std::make_pair(ID, &InfoI.first->second));

  if (Instr == nullptr)
    return;

  if (!Instr->getUserAssignedName().empty())
    Info.bindInstrVariable(Instr->getUserAssignedName(), ID);
  for (const auto &VarBinding : Instr->user_assigned_operand_names())
    Info.bindOperandVariable(VarBinding.second, ID, VarBinding.first);

  // Clear the bit indicating we haven't visited this instr.
  const auto &NodeI = find(MatchDag.instr_nodes(), Instr);
  assert(NodeI != MatchDag.instr_nodes_end() && "Instr isn't in this DAG");
  unsigned InstrIdx = MatchDag.getInstrNodeIdx(NodeI);
  RemainingInstrNodes.reset(InstrIdx);

  // When we declare an instruction, we don't expose any traversable edges just
  // yet. A partitioner has to check they exist and are registers before they
  // are traversable.

  // When we declare an instruction, we potentially activate some predicates.
  // Mark the dependencies that are now satisfied as a result of this
  // instruction and mark any predicates whose dependencies are fully
  // satisfied.
  for (auto &Dep : enumerate(MatchDag.predicate_edges())) {
    if (Dep.value()->getRequiredMI() == Instr &&
        Dep.value()->getRequiredMO() == nullptr) {
      for (auto &DepsFor : enumerate(UnsatisfiedPredDepsForPred)) {
        DepsFor.value().reset(Dep.index());
        if (DepsFor.value().none())
          TestablePredicates.set(DepsFor.index());
      }
    }
  }
}

void GIMatchTreeBuilderLeafInfo::declareOperand(unsigned InstrID,
                                                unsigned OpIdx) {
  const GIMatchDagInstr *Instr = InstrIDToInfo.lookup(InstrID)->getInstrNode();

  OperandIDToInfo.insert(std::make_pair(
      std::make_pair(InstrID, OpIdx),
      GIMatchTreeOperandInfo(Instr, OpIdx)));

  // When an operand becomes reachable, we potentially activate some traversals.
  // Record the edges that can now be followed as a result of this
  // instruction.
  for (auto &E : enumerate(MatchDag.edges())) {
    if (E.value()->getFromMI() == Instr &&
        E.value()->getFromMO()->getIdx() == OpIdx) {
      TraversableEdges.set(E.index());
    }
  }

  // When an operand becomes reachable, we potentially activate some predicates.
  // Clear the dependencies that are now satisfied as a result of this
  // operand and activate any predicates whose dependencies are fully
  // satisfied.
  for (auto &Dep : enumerate(MatchDag.predicate_edges())) {
    if (Dep.value()->getRequiredMI() == Instr && Dep.value()->getRequiredMO() &&
        Dep.value()->getRequiredMO()->getIdx() == OpIdx) {
      for (auto &DepsFor : enumerate(UnsatisfiedPredDepsForPred)) {
        DepsFor.value().reset(Dep.index());
        if (DepsFor.value().none())
          TestablePredicates.set(DepsFor.index());
      }
    }
  }
}

void GIMatchTreeBuilder::addPartitionersForInstr(unsigned InstrIdx) {
  // Find the partitioners that can be used now that this node is
  // uncovered. Our choices are:
  // - Test the opcode
  addPartitioner(std::make_unique<GIMatchTreeOpcodePartitioner>(InstrIdx));
}

void GIMatchTreeBuilder::addPartitionersForOperand(unsigned InstrID,
                                                   unsigned OpIdx) {
  LLVM_DEBUG(dbgs() << "Add partitioners for Instrs[" << InstrID
                    << "].getOperand(" << OpIdx << ")\n");
  addPartitioner(
      std::make_unique<GIMatchTreeVRegDefPartitioner>(InstrID, OpIdx));
}

void GIMatchTreeBuilder::filterRedundantPartitioners() {
  // TODO: Filter partitioners for facts that are already known
  // - If we know the opcode, we can elide the num operand check so long as
  //   the instruction has a fixed number of operands.
  // - If we know an exact number of operands then we can elide further number
  //   of operand checks.
  // - If the current min number of operands exceeds the one we want to check
  //   then we can elide it.
}

void GIMatchTreeBuilder::evaluatePartitioners() {
  // Determine the partitioning the partitioner would produce
  for (auto &Partitioner : Partitioners) {
    LLVM_DEBUG(dbgs() << "    Weighing up ";
               Partitioner->emitDescription(dbgs()); dbgs() << "\n");
    Partitioner->repartition(Leaves);
    LLVM_DEBUG(Partitioner->emitPartitionResults(dbgs()));
  }
}

void GIMatchTreeBuilder::runStep() {
  LLVM_DEBUG(dbgs() << "Building match tree node for " << TreeNode << "\n");
  LLVM_DEBUG(dbgs() << "  Rules reachable at this node:\n");
  for (const auto &Leaf : Leaves) {
    LLVM_DEBUG(dbgs() << "    " << Leaf.getName() << " (" << &Leaf.getInfo() << "\n");
    TreeNode->addPossibleLeaf(Leaf.getInfo(), Leaf.isFullyTraversed(),
                              Leaf.isFullyTested());
  }

  LLVM_DEBUG(dbgs() << "  Partitioners available at this node:\n");
#ifndef NDEBUG
  for (const auto &Partitioner : Partitioners)
    LLVM_DEBUG(dbgs() << "    "; Partitioner->emitDescription(dbgs());
               dbgs() << "\n");
#endif // ifndef NDEBUG

  // Check for unreachable rules. Rules are unreachable if they are preceeded by
  // a fully tested rule.
  // Note: This is only true for the current algorithm, if we allow the
  //       algorithm to compare equally valid rules then they will become
  //       reachable.
  {
    auto FullyTestedLeafI = Leaves.end();
    for (auto LeafI = Leaves.begin(), LeafE = Leaves.end();
         LeafI != LeafE; ++LeafI) {
      if (LeafI->isFullyTraversed() && LeafI->isFullyTested())
        FullyTestedLeafI = LeafI;
      else if (FullyTestedLeafI != Leaves.end()) {
        PrintError("Leaf " + LeafI->getName() + " is unreachable");
        PrintNote("Leaf " + FullyTestedLeafI->getName() +
                  " will have already matched");
      }
    }
  }

  LLVM_DEBUG(dbgs() << "  Eliminating redundant partitioners:\n");
  filterRedundantPartitioners();
  LLVM_DEBUG(dbgs() << "  Partitioners remaining:\n");
#ifndef NDEBUG
  for (const auto &Partitioner : Partitioners)
    LLVM_DEBUG(dbgs() << "    "; Partitioner->emitDescription(dbgs());
               dbgs() << "\n");
#endif // ifndef NDEBUG

  if (Partitioners.empty()) {
    // Nothing left to do but check we really did identify a single rule.
    if (Leaves.size() > 1) {
      LLVM_DEBUG(dbgs() << "Leaf contains multiple rules, drop after the first "
                           "fully tested rule\n");
      auto FirstFullyTested =
          llvm::find_if(Leaves, [](const GIMatchTreeBuilderLeafInfo &X) {
            return X.isFullyTraversed() && X.isFullyTested() &&
                   !X.getMatchDag().hasPostMatchPredicate();
          });
      if (FirstFullyTested != Leaves.end())
        FirstFullyTested++;

#ifndef NDEBUG
      for (auto &Leaf : make_range(Leaves.begin(), FirstFullyTested))
        LLVM_DEBUG(dbgs() << "  Kept " << Leaf.getName() << "\n");
      for (const auto &Leaf : make_range(FirstFullyTested, Leaves.end()))
        LLVM_DEBUG(dbgs() << "  Dropped " << Leaf.getName() << "\n");
#endif // ifndef NDEBUG
      TreeNode->dropLeavesAfter(
          std::distance(Leaves.begin(), FirstFullyTested));
    }
    for (const auto &Leaf : Leaves) {
      if (!Leaf.isFullyTraversed()) {
        PrintError("Leaf " + Leaf.getName() + " is not fully traversed");
        PrintNote("This indicates a missing partitioner within tblgen");
        Leaf.dump(errs());
        for (unsigned InstrIdx : Leaf.untested_instrs())
          PrintNote("Instr " + llvm::to_string(*Leaf.getInstr(InstrIdx)));
        for (unsigned EdgeIdx : Leaf.untested_edges())
          PrintNote("Edge " + llvm::to_string(*Leaf.getEdge(EdgeIdx)));
      }
    }

    // Copy out information about untested predicates so the user of the tree
    // can deal with them.
    for (auto LeafPair : zip(Leaves, TreeNode->possible_leaves())) {
      const GIMatchTreeBuilderLeafInfo &BuilderLeaf = std::get<0>(LeafPair);
      GIMatchTreeLeafInfo &TreeLeaf = std::get<1>(LeafPair);
      if (!BuilderLeaf.isFullyTested())
        for (unsigned PredicateIdx : BuilderLeaf.untested_predicates())
          TreeLeaf.addUntestedPredicate(BuilderLeaf.getPredicate(PredicateIdx));
    }
    return;
  }

  LLVM_DEBUG(dbgs() << "  Weighing up partitioners:\n");
  evaluatePartitioners();

  // Select the best partitioner by its ability to partition
  // - Prefer partitioners that don't distinguish between partitions. This
  //   is to fail early on decisions that must go a single way.
  auto PartitionerI = std::max_element(
      Partitioners.begin(), Partitioners.end(),
      [](const std::unique_ptr<GIMatchTreePartitioner> &A,
         const std::unique_ptr<GIMatchTreePartitioner> &B) {
        // We generally want partitioners that subdivide the
        // ruleset as much as possible since these take fewer
        // checks to converge on a particular rule. However,
        // it's important to note that one leaf can end up in
        // multiple partitions if the check isn't mutually
        // exclusive (e.g. getVRegDef() vs isReg()).
        // We therefore minimize average leaves per partition.
        return (double)A->getNumLeavesWithDupes() / A->getNumPartitions() >
               (double)B->getNumLeavesWithDupes() / B->getNumPartitions();
      });

  // Select a partitioner and partition the ruleset
  // Note that it's possible for a single rule to end up in multiple
  // partitions. For example, an opcode test on a rule without an opcode
  // predicate will result in it being passed to all partitions.
  std::unique_ptr<GIMatchTreePartitioner> Partitioner = std::move(*PartitionerI);
  Partitioners.erase(PartitionerI);
  LLVM_DEBUG(dbgs() << "  Selected partitioner: ";
             Partitioner->emitDescription(dbgs()); dbgs() << "\n");

  assert(Partitioner->getNumPartitions() > 0 &&
         "Must always partition into at least one partition");

  TreeNode->setNumChildren(Partitioner->getNumPartitions());
  for (auto &C : enumerate(TreeNode->children())) {
    SubtreeBuilders.emplace_back(&C.value(), NextInstrID);
    Partitioner->applyForPartition(C.index(), *this, SubtreeBuilders.back());
  }

  TreeNode->setPartitioner(std::move(Partitioner));

  // Recurse into the subtree builders. Each one must get a copy of the
  // remaining partitioners as each path has to check everything.
  for (auto &SubtreeBuilder : SubtreeBuilders) {
    for (const auto &Partitioner : Partitioners)
      SubtreeBuilder.addPartitioner(Partitioner->clone());
    SubtreeBuilder.runStep();
  }
}

std::unique_ptr<GIMatchTree> GIMatchTreeBuilder::run() {
  unsigned NewInstrID = allocInstrID();
  // Start by recording the root instruction as instr #0 and set up the initial
  // partitioners.
  for (auto &Leaf : Leaves) {
    LLVM_DEBUG(Leaf.getMatchDag().writeDOTGraph(dbgs(), Leaf.getName()));
    GIMatchDagInstr *Root =
        *(Leaf.getMatchDag().roots().begin() + Leaf.getRootIdx());
    Leaf.declareInstr(Root, NewInstrID);
  }

  addPartitionersForInstr(NewInstrID);

  std::unique_ptr<GIMatchTree> TreeRoot = std::make_unique<GIMatchTree>();
  TreeNode = TreeRoot.get();
  runStep();

  return TreeRoot;
}

void GIMatchTreeOpcodePartitioner::emitPartitionName(raw_ostream &OS, unsigned Idx) const {
  if (PartitionToInstr[Idx] == nullptr) {
    OS << "* or nullptr";
    return;
  }
  OS << PartitionToInstr[Idx]->Namespace
     << "::" << PartitionToInstr[Idx]->TheDef->getName();
}

void GIMatchTreeOpcodePartitioner::repartition(
    GIMatchTreeBuilder::LeafVec &Leaves) {
  Partitions.clear();
  InstrToPartition.clear();
  PartitionToInstr.clear();
  TestedPredicates.clear();

  for (const auto &Leaf : enumerate(Leaves)) {
    bool AllOpcodes = true;
    GIMatchTreeInstrInfo *InstrInfo = Leaf.value().getInstrInfo(InstrID);
    BitVector TestedPredicatesForLeaf(
        Leaf.value().getMatchDag().getNumPredicates());

    // If the instruction isn't declared then we don't care about it. Ignore
    // it for now and add it to all partitions later once we know what
    // partitions we have.
    if (!InstrInfo) {
      LLVM_DEBUG(dbgs() << "      " << Leaf.value().getName()
                        << " doesn't care about Instr[" << InstrID << "]\n");
      assert(TestedPredicatesForLeaf.size() == Leaf.value().getMatchDag().getNumPredicates());
      TestedPredicates.push_back(TestedPredicatesForLeaf);
      continue;
    }

    // If the opcode is available to test then any opcode predicates will have
    // been enabled too.
    for (unsigned PIdx : Leaf.value().TestablePredicates.set_bits()) {
      const auto &P = Leaf.value().getPredicate(PIdx);
      SmallVector<const CodeGenInstruction *, 1> OpcodesForThisPredicate;
      if (const auto *OpcodeP = dyn_cast<const GIMatchDagOpcodePredicate>(P)) {
        // We've found _an_ opcode predicate, but we don't know if it's
        // checking this instruction yet.
        bool IsThisPredicate = false;
        for (const auto &PDep : Leaf.value().getMatchDag().predicate_edges()) {
          if (PDep->getRequiredMI() == InstrInfo->getInstrNode() &&
              PDep->getRequiredMO() == nullptr && PDep->getPredicate() == P) {
            IsThisPredicate = true;
            break;
          }
        }
        if (!IsThisPredicate)
          continue;

        // If we get here twice then we've somehow ended up with two opcode
        // predicates for one instruction in the same DAG. That should be
        // impossible.
        assert(AllOpcodes && "Conflicting opcode predicates");
        const CodeGenInstruction *Expected = OpcodeP->getInstr();
        OpcodesForThisPredicate.push_back(Expected);
      }

      if (const auto *OpcodeP =
              dyn_cast<const GIMatchDagOneOfOpcodesPredicate>(P)) {
        // We've found _an_ oneof(opcodes) predicate, but we don't know if it's
        // checking this instruction yet.
        bool IsThisPredicate = false;
        for (const auto &PDep : Leaf.value().getMatchDag().predicate_edges()) {
          if (PDep->getRequiredMI() == InstrInfo->getInstrNode() &&
              PDep->getRequiredMO() == nullptr && PDep->getPredicate() == P) {
            IsThisPredicate = true;
            break;
          }
        }
        if (!IsThisPredicate)
          continue;

        // If we get here twice then we've somehow ended up with two opcode
        // predicates for one instruction in the same DAG. That should be
        // impossible.
        assert(AllOpcodes && "Conflicting opcode predicates");
        append_range(OpcodesForThisPredicate, OpcodeP->getInstrs());
      }

      for (const CodeGenInstruction *Expected : OpcodesForThisPredicate) {
        // Mark this predicate as one we're testing.
        TestedPredicatesForLeaf.set(PIdx);

        // Partitions must be numbered 0, 1, .., N but instructions don't meet
        // that requirement. Assign a partition number to each opcode if we
        // lack one ...
        auto Partition = InstrToPartition.find(Expected);
        if (Partition == InstrToPartition.end()) {
          BitVector Contents(Leaves.size());
          Partition = InstrToPartition
                          .insert(std::make_pair(Expected, Partitions.size()))
                          .first;
          PartitionToInstr.push_back(Expected);
          Partitions.insert(std::make_pair(Partitions.size(), Contents));
        }
        // ... and mark this leaf as being in that partition.
        Partitions.find(Partition->second)->second.set(Leaf.index());
        AllOpcodes = false;
        LLVM_DEBUG(dbgs() << "      " << Leaf.value().getName()
                          << " is in partition " << Partition->second << "\n");
      }

      // TODO: This is where we would handle multiple choices of opcode
      //       the end result will be that this leaf ends up in multiple
      //       partitions similarly to AllOpcodes.
    }

    // If we never check the opcode, add it to every partition.
    if (AllOpcodes) {
      // Add a partition for the default case if we don't already have one.
      if (InstrToPartition.insert(std::make_pair(nullptr, 0)).second) {
        PartitionToInstr.push_back(nullptr);
        BitVector Contents(Leaves.size());
        Partitions.insert(std::make_pair(Partitions.size(), Contents));
      }
      LLVM_DEBUG(dbgs() << "      " << Leaf.value().getName()
                        << " is in all partitions (opcode not checked)\n");
      for (auto &Partition : Partitions)
        Partition.second.set(Leaf.index());
    }

    assert(TestedPredicatesForLeaf.size() == Leaf.value().getMatchDag().getNumPredicates());
    TestedPredicates.push_back(TestedPredicatesForLeaf);
  }

  if (Partitions.size() == 0) {
    // Add a partition for the default case if we don't already have one.
    if (InstrToPartition.insert(std::make_pair(nullptr, 0)).second) {
      PartitionToInstr.push_back(nullptr);
      BitVector Contents(Leaves.size());
      Partitions.insert(std::make_pair(Partitions.size(), Contents));
    }
  }

  // Add any leaves that don't care about this instruction to all partitions.
  for (const auto &Leaf : enumerate(Leaves)) {
    GIMatchTreeInstrInfo *InstrInfo = Leaf.value().getInstrInfo(InstrID);
    if (!InstrInfo) {
      // Add a partition for the default case if we don't already have one.
      if (InstrToPartition.insert(std::make_pair(nullptr, 0)).second) {
        PartitionToInstr.push_back(nullptr);
        BitVector Contents(Leaves.size());
        Partitions.insert(std::make_pair(Partitions.size(), Contents));
      }
      for (auto &Partition : Partitions)
        Partition.second.set(Leaf.index());
    }
  }

}

void GIMatchTreeOpcodePartitioner::applyForPartition(
    unsigned PartitionIdx, GIMatchTreeBuilder &Builder, GIMatchTreeBuilder &SubBuilder) {
  LLVM_DEBUG(dbgs() << "  Making partition " << PartitionIdx << "\n");
  const CodeGenInstruction *CGI = PartitionToInstr[PartitionIdx];

  BitVector PossibleLeaves = getPossibleLeavesForPartition(PartitionIdx);
  // Consume any predicates we handled.
  for (auto &EnumeratedLeaf : enumerate(Builder.getPossibleLeaves())) {
    if (!PossibleLeaves[EnumeratedLeaf.index()])
      continue;

    auto &Leaf = EnumeratedLeaf.value();
    const auto &TestedPredicatesForLeaf =
        TestedPredicates[EnumeratedLeaf.index()];

    for (unsigned PredIdx : TestedPredicatesForLeaf.set_bits()) {
      LLVM_DEBUG(dbgs() << "    " << Leaf.getName() << " tested predicate #"
                        << PredIdx << " of " << TestedPredicatesForLeaf.size()
                        << " " << *Leaf.getPredicate(PredIdx) << "\n");
      Leaf.RemainingPredicates.reset(PredIdx);
      Leaf.TestablePredicates.reset(PredIdx);
    }
    SubBuilder.addLeaf(Leaf);
  }

  // Nothing to do, we don't know anything about this instruction as a result
  // of this partitioner.
  if (CGI == nullptr)
    return;

  GIMatchTreeBuilder::LeafVec &NewLeaves = SubBuilder.getPossibleLeaves();
  // Find all the operands we know to exist and are referenced. This will
  // usually be all the referenced operands but there are some cases where
  // instructions are variadic. Such operands must be handled by partitioners
  // that check the number of operands.
  BitVector ReferencedOperands(1);
  for (auto &Leaf : NewLeaves) {
    GIMatchTreeInstrInfo *InstrInfo = Leaf.getInstrInfo(InstrID);
    // Skip any leaves that don't care about this instruction.
    if (!InstrInfo)
      continue;
    const GIMatchDagInstr *Instr = InstrInfo->getInstrNode();
    for (auto &E : enumerate(Leaf.getMatchDag().edges())) {
      if (E.value()->getFromMI() == Instr &&
          E.value()->getFromMO()->getIdx() < CGI->Operands.size()) {
        ReferencedOperands.resize(E.value()->getFromMO()->getIdx() + 1);
        ReferencedOperands.set(E.value()->getFromMO()->getIdx());
      }
    }
  }
  for (auto &Leaf : NewLeaves) {
    for (unsigned OpIdx : ReferencedOperands.set_bits()) {
      Leaf.declareOperand(InstrID, OpIdx);
    }
  }
  for (unsigned OpIdx : ReferencedOperands.set_bits()) {
    SubBuilder.addPartitionersForOperand(InstrID, OpIdx);
  }
}

void GIMatchTreeOpcodePartitioner::emitPartitionResults(
    raw_ostream &OS) const {
  OS << "Partitioning by opcode would produce " << Partitions.size()
     << " partitions\n";
  for (const auto &Partition : InstrToPartition) {
    if (Partition.first == nullptr)
      OS << "Default: ";
    else
      OS << Partition.first->TheDef->getName() << ": ";
    StringRef Separator = "";
    for (unsigned I : Partitions.find(Partition.second)->second.set_bits()) {
      OS << Separator << I;
      Separator = ", ";
    }
    OS << "\n";
  }
}

void GIMatchTreeOpcodePartitioner::generatePartitionSelectorCode(
    raw_ostream &OS, StringRef Indent) const {
  // Make sure not to emit empty switch or switch with just default
  if (PartitionToInstr.size() == 1 && PartitionToInstr[0] == nullptr) {
    OS << Indent << "Partition = 0;\n";
  } else if (PartitionToInstr.size()) {
    OS << Indent << "Partition = -1;\n"
       << Indent << "switch (MIs[" << InstrID << "]->getOpcode()) {\n";
    for (const auto &EnumInstr : enumerate(PartitionToInstr)) {
      if (EnumInstr.value() == nullptr)
        OS << Indent << "default:";
      else
        OS << Indent << "case " << EnumInstr.value()->Namespace
           << "::" << EnumInstr.value()->TheDef->getName() << ":";
      OS << " Partition = " << EnumInstr.index() << "; break;\n";
    }
    OS << Indent << "}\n";
  }
  OS << Indent
     << "// Default case but without conflicting with potential default case "
        "in selection.\n"
     << Indent << "if (Partition == -1) return false;\n";
}

void GIMatchTreeVRegDefPartitioner::addToPartition(bool Result,
                                                   unsigned LeafIdx) {
  auto I = ResultToPartition.find(Result);
  if (I == ResultToPartition.end()) {
    ResultToPartition.insert(std::make_pair(Result, PartitionToResult.size()));
    PartitionToResult.push_back(Result);
  }
  I = ResultToPartition.find(Result);
  auto P = Partitions.find(I->second);
  if (P == Partitions.end())
    P = Partitions.insert(std::make_pair(I->second, BitVector())).first;
  P->second.resize(LeafIdx + 1);
  P->second.set(LeafIdx);
}

void GIMatchTreeVRegDefPartitioner::repartition(
    GIMatchTreeBuilder::LeafVec &Leaves) {
  Partitions.clear();

  for (const auto &Leaf : enumerate(Leaves)) {
    GIMatchTreeInstrInfo *InstrInfo = Leaf.value().getInstrInfo(InstrID);
    BitVector TraversedEdgesForLeaf(Leaf.value().getMatchDag().getNumEdges());

    // If the instruction isn't declared then we don't care about it. Ignore
    // it for now and add it to all partitions later once we know what
    // partitions we have.
    if (!InstrInfo) {
      TraversedEdges.push_back(TraversedEdgesForLeaf);
      continue;
    }

    // If this node has an use -> def edge from this operand then this
    // instruction must be in partition 1 (isVRegDef()).
    bool WantsEdge = false;
    for (unsigned EIdx : Leaf.value().TraversableEdges.set_bits()) {
      const auto &E = Leaf.value().getEdge(EIdx);
      if (E->getFromMI() != InstrInfo->getInstrNode() ||
          E->getFromMO()->getIdx() != OpIdx || E->isDefToUse())
        continue;

      // We're looking at the right edge. This leaf wants a vreg def so we'll
      // put it in partition 1.
      addToPartition(true, Leaf.index());
      TraversedEdgesForLeaf.set(EIdx);
      WantsEdge = true;
    }

    bool isNotReg = false;
    if (!WantsEdge && isNotReg) {
      // If this leaf doesn't have an edge and we _don't_ want a register,
      // then add it to partition 0.
      addToPartition(false, Leaf.index());
    } else if (!WantsEdge) {
      // If this leaf doesn't have an edge and we don't know what we want,
      // then add it to partition 0 and 1.
      addToPartition(false, Leaf.index());
      addToPartition(true, Leaf.index());
    }

    TraversedEdges.push_back(TraversedEdgesForLeaf);
  }

  // Add any leaves that don't care about this instruction to all partitions.
  for (const auto &Leaf : enumerate(Leaves)) {
    GIMatchTreeInstrInfo *InstrInfo = Leaf.value().getInstrInfo(InstrID);
    if (!InstrInfo)
      for (auto &Partition : Partitions)
        Partition.second.set(Leaf.index());
  }
}

void GIMatchTreeVRegDefPartitioner::applyForPartition(
    unsigned PartitionIdx, GIMatchTreeBuilder &Builder,
    GIMatchTreeBuilder &SubBuilder) {
  BitVector PossibleLeaves = getPossibleLeavesForPartition(PartitionIdx);

  std::vector<BitVector> TraversedEdgesByNewLeaves;
  // Consume any edges we handled.
  for (auto &EnumeratedLeaf : enumerate(Builder.getPossibleLeaves())) {
    if (!PossibleLeaves[EnumeratedLeaf.index()])
      continue;

    auto &Leaf = EnumeratedLeaf.value();
    const auto &TraversedEdgesForLeaf = TraversedEdges[EnumeratedLeaf.index()];
    TraversedEdgesByNewLeaves.push_back(TraversedEdgesForLeaf);
    Leaf.RemainingEdges.reset(TraversedEdgesForLeaf);
    Leaf.TraversableEdges.reset(TraversedEdgesForLeaf);
    SubBuilder.addLeaf(Leaf);
  }

  // Nothing to do. The only thing we know is that it isn't a vreg-def.
  if (PartitionToResult[PartitionIdx] == false)
    return;

  NewInstrID = SubBuilder.allocInstrID();

  GIMatchTreeBuilder::LeafVec &NewLeaves = SubBuilder.getPossibleLeaves();
  for (const auto I : zip(NewLeaves, TraversedEdgesByNewLeaves)) {
    auto &Leaf = std::get<0>(I);
    auto &TraversedEdgesForLeaf = std::get<1>(I);
    GIMatchTreeInstrInfo *InstrInfo = Leaf.getInstrInfo(InstrID);
    // Skip any leaves that don't care about this instruction.
    if (!InstrInfo)
      continue;
    for (unsigned EIdx : TraversedEdgesForLeaf.set_bits()) {
      const GIMatchDagEdge *E = Leaf.getEdge(EIdx);
      Leaf.declareInstr(E->getToMI(), NewInstrID);
    }
  }
  SubBuilder.addPartitionersForInstr(NewInstrID);
}

void GIMatchTreeVRegDefPartitioner::emitPartitionResults(
    raw_ostream &OS) const {
  OS << "Partitioning by vreg-def would produce " << Partitions.size()
     << " partitions\n";
  for (const auto &Partition : Partitions) {
    OS << Partition.first << " (";
    emitPartitionName(OS, Partition.first);
    OS << "): ";
    StringRef Separator = "";
    for (unsigned I : Partition.second.set_bits()) {
      OS << Separator << I;
      Separator = ", ";
    }
    OS << "\n";
  }
}

void GIMatchTreeVRegDefPartitioner::generatePartitionSelectorCode(
    raw_ostream &OS, StringRef Indent) const {
  OS << Indent << "Partition = -1\n"
     << Indent << "if (MIs.size() <= NewInstrID) MIs.resize(NewInstrID + 1);\n"
     << Indent << "MIs[" << NewInstrID << "] = nullptr;\n"
     << Indent << "if (MIs[" << InstrID << "].getOperand(" << OpIdx
     << ").isReg()))\n"
     << Indent << "  MIs[" << NewInstrID << "] = MRI.getVRegDef(MIs[" << InstrID
     << "].getOperand(" << OpIdx << ").getReg()));\n";

  for (const auto &Pair : ResultToPartition)
    OS << Indent << "if (MIs[" << NewInstrID << "] "
       << (Pair.first ? "==" : "!=")
       << " nullptr) Partition = " << Pair.second << ";\n";

  OS << Indent << "if (Partition == -1) return false;\n";
}
