//===- DependencyGraph.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/DependencyGraph.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/SandboxIR/Utils.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Scheduler.h"

namespace llvm::sandboxir {

User::op_iterator PredIterator::skipBadIt(User::op_iterator OpIt,
                                          User::op_iterator OpItE,
                                          const DependencyGraph &DAG) {
  auto Skip = [&DAG](auto OpIt) {
    auto *I = dyn_cast<Instruction>((*OpIt).get());
    return I == nullptr || DAG.getNode(I) == nullptr;
  };
  while (OpIt != OpItE && Skip(OpIt))
    ++OpIt;
  return OpIt;
}

PredIterator::value_type PredIterator::operator*() {
  // If it's a DGNode then we dereference the operand iterator.
  if (!isa<MemDGNode>(N)) {
    assert(OpIt != OpItE && "Can't dereference end iterator!");
    return DAG->getNode(cast<Instruction>((Value *)*OpIt));
  }
  // It's a MemDGNode, so we check if we return either the use-def operand,
  // or a mem predecessor.
  if (OpIt != OpItE)
    return DAG->getNode(cast<Instruction>((Value *)*OpIt));
  // It's a MemDGNode with OpIt == end, so we need to use MemIt.
  assert(MemIt != cast<MemDGNode>(N)->MemPreds.end() &&
         "Cant' dereference end iterator!");
  return *MemIt;
}

PredIterator &PredIterator::operator++() {
  // If it's a DGNode then we increment the use-def iterator.
  if (!isa<MemDGNode>(N)) {
    assert(OpIt != OpItE && "Already at end!");
    ++OpIt;
    // Skip operands that are not instructions or are outside the DAG.
    OpIt = PredIterator::skipBadIt(OpIt, OpItE, *DAG);
    return *this;
  }
  // It's a MemDGNode, so if we are not at the end of the use-def iterator we
  // need to first increment that.
  if (OpIt != OpItE) {
    ++OpIt;
    // Skip operands that are not instructions or are outside the DAG.
    OpIt = PredIterator::skipBadIt(OpIt, OpItE, *DAG);
    return *this;
  }
  // It's a MemDGNode with OpIt == end, so we need to increment MemIt.
  assert(MemIt != cast<MemDGNode>(N)->MemPreds.end() && "Already at end!");
  ++MemIt;
  return *this;
}

bool PredIterator::operator==(const PredIterator &Other) const {
  assert(DAG == Other.DAG && "Iterators of different DAGs!");
  assert(N == Other.N && "Iterators of different nodes!");
  return OpIt == Other.OpIt && MemIt == Other.MemIt;
}

DGNode::~DGNode() {
  if (SB == nullptr)
    return;
  SB->eraseFromBundle(this);
}

#ifndef NDEBUG
void DGNode::print(raw_ostream &OS, bool PrintDeps) const {
  OS << *I << " USuccs:" << UnscheduledSuccs << " Sched:" << Scheduled << "\n";
}
void DGNode::dump() const { print(dbgs()); }
void MemDGNode::print(raw_ostream &OS, bool PrintDeps) const {
  DGNode::print(OS, false);
  if (PrintDeps) {
    // Print memory preds.
    static constexpr const unsigned Indent = 4;
    for (auto *Pred : MemPreds)
      OS.indent(Indent) << "<-" << *Pred->getInstruction() << "\n";
  }
}
#endif // NDEBUG

MemDGNode *
MemDGNodeIntervalBuilder::getTopMemDGNode(const Interval<Instruction> &Intvl,
                                          const DependencyGraph &DAG) {
  Instruction *I = Intvl.top();
  Instruction *BeforeI = Intvl.bottom();
  // Walk down the chain looking for a mem-dep candidate instruction.
  while (!DGNode::isMemDepNodeCandidate(I) && I != BeforeI)
    I = I->getNextNode();
  if (!DGNode::isMemDepNodeCandidate(I))
    return nullptr;
  return cast<MemDGNode>(DAG.getNode(I));
}

MemDGNode *
MemDGNodeIntervalBuilder::getBotMemDGNode(const Interval<Instruction> &Intvl,
                                          const DependencyGraph &DAG) {
  Instruction *I = Intvl.bottom();
  Instruction *AfterI = Intvl.top();
  // Walk up the chain looking for a mem-dep candidate instruction.
  while (!DGNode::isMemDepNodeCandidate(I) && I != AfterI)
    I = I->getPrevNode();
  if (!DGNode::isMemDepNodeCandidate(I))
    return nullptr;
  return cast<MemDGNode>(DAG.getNode(I));
}

Interval<MemDGNode>
MemDGNodeIntervalBuilder::make(const Interval<Instruction> &Instrs,
                               DependencyGraph &DAG) {
  auto *TopMemN = getTopMemDGNode(Instrs, DAG);
  // If we couldn't find a mem node in range TopN - BotN then it's empty.
  if (TopMemN == nullptr)
    return {};
  auto *BotMemN = getBotMemDGNode(Instrs, DAG);
  assert(BotMemN != nullptr && "TopMemN should be null too!");
  // Now that we have the mem-dep nodes, create and return the range.
  return Interval<MemDGNode>(TopMemN, BotMemN);
}

DependencyGraph::DependencyType
DependencyGraph::getRoughDepType(Instruction *FromI, Instruction *ToI) {
  // TODO: Perhaps compile-time improvement by skipping if neither is mem?
  if (FromI->mayWriteToMemory()) {
    if (ToI->mayReadFromMemory())
      return DependencyType::ReadAfterWrite;
    if (ToI->mayWriteToMemory())
      return DependencyType::WriteAfterWrite;
  } else if (FromI->mayReadFromMemory()) {
    if (ToI->mayWriteToMemory())
      return DependencyType::WriteAfterRead;
  }
  if (isa<sandboxir::PHINode>(FromI) || isa<sandboxir::PHINode>(ToI))
    return DependencyType::Control;
  if (ToI->isTerminator())
    return DependencyType::Control;
  if (DGNode::isStackSaveOrRestoreIntrinsic(FromI) ||
      DGNode::isStackSaveOrRestoreIntrinsic(ToI))
    return DependencyType::Other;
  return DependencyType::None;
}

static bool isOrdered(Instruction *I) {
  auto IsOrdered = [](Instruction *I) {
    if (auto *LI = dyn_cast<LoadInst>(I))
      return !LI->isUnordered();
    if (auto *SI = dyn_cast<StoreInst>(I))
      return !SI->isUnordered();
    if (DGNode::isFenceLike(I))
      return true;
    return false;
  };
  bool Is = IsOrdered(I);
  assert((!Is || DGNode::isMemDepCandidate(I)) &&
         "An ordered instruction must be a MemDepCandidate!");
  return Is;
}

bool DependencyGraph::alias(Instruction *SrcI, Instruction *DstI,
                            DependencyType DepType) {
  std::optional<MemoryLocation> DstLocOpt =
      Utils::memoryLocationGetOrNone(DstI);
  if (!DstLocOpt)
    return true;
  // Check aliasing.
  assert((SrcI->mayReadFromMemory() || SrcI->mayWriteToMemory()) &&
         "Expected a mem instr");
  // TODO: Check AABudget
  ModRefInfo SrcModRef =
      isOrdered(SrcI)
          ? ModRefInfo::ModRef
          : Utils::aliasAnalysisGetModRefInfo(*BatchAA, SrcI, *DstLocOpt);
  switch (DepType) {
  case DependencyType::ReadAfterWrite:
  case DependencyType::WriteAfterWrite:
    return isModSet(SrcModRef);
  case DependencyType::WriteAfterRead:
    return isRefSet(SrcModRef);
  default:
    llvm_unreachable("Expected only RAW, WAW and WAR!");
  }
}

bool DependencyGraph::hasDep(Instruction *SrcI, Instruction *DstI) {
  DependencyType RoughDepType = getRoughDepType(SrcI, DstI);
  switch (RoughDepType) {
  case DependencyType::ReadAfterWrite:
  case DependencyType::WriteAfterWrite:
  case DependencyType::WriteAfterRead:
    return alias(SrcI, DstI, RoughDepType);
  case DependencyType::Control:
    // Adding actual dep edges from PHIs/to terminator would just create too
    // many edges, which would be bad for compile-time.
    // So we ignore them in the DAG formation but handle them in the
    // scheduler, while sorting the ready list.
    return false;
  case DependencyType::Other:
    return true;
  case DependencyType::None:
    return false;
  }
  llvm_unreachable("Unknown DependencyType enum");
}

void DependencyGraph::scanAndAddDeps(MemDGNode &DstN,
                                     const Interval<MemDGNode> &SrcScanRange) {
  assert(isa<MemDGNode>(DstN) &&
         "DstN is the mem dep destination, so it must be mem");
  Instruction *DstI = DstN.getInstruction();
  // Walk up the instruction chain from ScanRange bottom to top, looking for
  // memory instrs that may alias.
  for (MemDGNode &SrcN : reverse(SrcScanRange)) {
    Instruction *SrcI = SrcN.getInstruction();
    if (hasDep(SrcI, DstI))
      DstN.addMemPred(&SrcN);
  }
}

void DependencyGraph::setDefUseUnscheduledSuccs(
    const Interval<Instruction> &NewInterval) {
  // +---+
  // |   |  Def
  // |   |   |
  // |   |   v
  // |   |  Use
  // +---+
  // Set the intra-interval counters in NewInterval.
  for (Instruction &I : NewInterval) {
    for (Value *Op : I.operands()) {
      auto *OpI = dyn_cast<Instruction>(Op);
      if (OpI == nullptr)
        continue;
      // TODO: For now don't cross BBs.
      if (OpI->getParent() != I.getParent())
        continue;
      if (!NewInterval.contains(OpI))
        continue;
      auto *OpN = getNode(OpI);
      if (OpN == nullptr)
        continue;
      ++OpN->UnscheduledSuccs;
    }
  }

  // Now handle the cross-interval edges.
  bool NewIsAbove = DAGInterval.empty() || NewInterval.comesBefore(DAGInterval);
  const auto &TopInterval = NewIsAbove ? NewInterval : DAGInterval;
  const auto &BotInterval = NewIsAbove ? DAGInterval : NewInterval;
  // +---+
  // |Top|
  // |   |  Def
  // +---+   |
  // |   |   v
  // |Bot|  Use
  // |   |
  // +---+
  // Walk over all instructions in "BotInterval" and update the counter
  // of operands that are in "TopInterval".
  for (Instruction &BotI : BotInterval) {
    auto *BotN = getNode(&BotI);
    // Skip scheduled nodes.
    if (BotN->scheduled())
      continue;
    for (Value *Op : BotI.operands()) {
      auto *OpI = dyn_cast<Instruction>(Op);
      if (OpI == nullptr)
        continue;
      auto *OpN = getNode(OpI);
      if (OpN == nullptr)
        continue;
      if (!TopInterval.contains(OpI))
        continue;
      ++OpN->UnscheduledSuccs;
    }
  }
}

void DependencyGraph::createNewNodes(const Interval<Instruction> &NewInterval) {
  // Create Nodes only for the new sections of the DAG.
  DGNode *LastN = getOrCreateNode(NewInterval.top());
  MemDGNode *LastMemN = dyn_cast<MemDGNode>(LastN);
  for (Instruction &I : drop_begin(NewInterval)) {
    auto *N = getOrCreateNode(&I);
    // Build the Mem node chain.
    if (auto *MemN = dyn_cast<MemDGNode>(N)) {
      MemN->setPrevNode(LastMemN);
      LastMemN = MemN;
    }
  }
  // Link new MemDGNode chain with the old one, if any.
  if (!DAGInterval.empty()) {
    bool NewIsAbove = NewInterval.comesBefore(DAGInterval);
    const auto &TopInterval = NewIsAbove ? NewInterval : DAGInterval;
    const auto &BotInterval = NewIsAbove ? DAGInterval : NewInterval;
    MemDGNode *LinkTopN =
        MemDGNodeIntervalBuilder::getBotMemDGNode(TopInterval, *this);
    MemDGNode *LinkBotN =
        MemDGNodeIntervalBuilder::getTopMemDGNode(BotInterval, *this);
    assert((LinkTopN == nullptr || LinkBotN == nullptr ||
            LinkTopN->comesBefore(LinkBotN)) &&
           "Wrong order!");
    if (LinkTopN != nullptr && LinkBotN != nullptr) {
      LinkTopN->setNextNode(LinkBotN);
    }
#ifndef NDEBUG
    // TODO: Remove this once we've done enough testing.
    // Check that the chain is well formed.
    auto UnionIntvl = DAGInterval.getUnionInterval(NewInterval);
    MemDGNode *ChainTopN =
        MemDGNodeIntervalBuilder::getTopMemDGNode(UnionIntvl, *this);
    MemDGNode *ChainBotN =
        MemDGNodeIntervalBuilder::getBotMemDGNode(UnionIntvl, *this);
    if (ChainTopN != nullptr && ChainBotN != nullptr) {
      for (auto *N = ChainTopN->getNextNode(), *LastN = ChainTopN; N != nullptr;
           LastN = N, N = N->getNextNode()) {
        assert(N == LastN->getNextNode() && "Bad chain!");
        assert(N->getPrevNode() == LastN && "Bad chain!");
      }
    }
#endif // NDEBUG
  }

  setDefUseUnscheduledSuccs(NewInterval);
}

MemDGNode *DependencyGraph::getMemDGNodeBefore(DGNode *N, bool IncludingN,
                                               MemDGNode *SkipN) const {
  auto *I = N->getInstruction();
  for (auto *PrevI = IncludingN ? I : I->getPrevNode(); PrevI != nullptr;
       PrevI = PrevI->getPrevNode()) {
    auto *PrevN = getNodeOrNull(PrevI);
    if (PrevN == nullptr)
      return nullptr;
    auto *PrevMemN = dyn_cast<MemDGNode>(PrevN);
    if (PrevMemN != nullptr && PrevMemN != SkipN)
      return PrevMemN;
  }
  return nullptr;
}

MemDGNode *DependencyGraph::getMemDGNodeAfter(DGNode *N, bool IncludingN,
                                              MemDGNode *SkipN) const {
  auto *I = N->getInstruction();
  for (auto *NextI = IncludingN ? I : I->getNextNode(); NextI != nullptr;
       NextI = NextI->getNextNode()) {
    auto *NextN = getNodeOrNull(NextI);
    if (NextN == nullptr)
      return nullptr;
    auto *NextMemN = dyn_cast<MemDGNode>(NextN);
    if (NextMemN != nullptr && NextMemN != SkipN)
      return NextMemN;
  }
  return nullptr;
}

void DependencyGraph::notifyCreateInstr(Instruction *I) {
  // Nothing to do if the node is not in the focus range of the DAG.
  if (!(DAGInterval.contains(I) || DAGInterval.touches(I)))
    return;
  // Include `I` into the interval.
  DAGInterval = DAGInterval.getUnionInterval({I, I});
  auto *N = getOrCreateNode(I);
  auto *MemN = dyn_cast<MemDGNode>(N);

  // Update the MemDGNode chain if this is a memory node.
  if (MemN != nullptr) {
    if (auto *PrevMemN = getMemDGNodeBefore(MemN, /*IncludingN=*/false)) {
      PrevMemN->NextMemN = MemN;
      MemN->PrevMemN = PrevMemN;
    }
    if (auto *NextMemN = getMemDGNodeAfter(MemN, /*IncludingN=*/false)) {
      NextMemN->PrevMemN = MemN;
      MemN->NextMemN = NextMemN;
    }

    // Add Mem dependencies.
    // 1. Scan for deps above `I` for deps to `I`: AboveN->MemN.
    if (DAGInterval.top()->comesBefore(I)) {
      Interval<Instruction> AboveIntvl(DAGInterval.top(), I->getPrevNode());
      auto SrcInterval = MemDGNodeIntervalBuilder::make(AboveIntvl, *this);
      scanAndAddDeps(*MemN, SrcInterval);
    }
    // 2. Scan for deps below `I` for deps from `I`: MemN->BelowN.
    if (I->comesBefore(DAGInterval.bottom())) {
      Interval<Instruction> BelowIntvl(I->getNextNode(), DAGInterval.bottom());
      for (MemDGNode &BelowN :
           MemDGNodeIntervalBuilder::make(BelowIntvl, *this))
        scanAndAddDeps(BelowN, Interval<MemDGNode>(MemN, MemN));
    }
  }
}

void DependencyGraph::notifyMoveInstr(Instruction *I, const BBIterator &To) {
  // NOTE: This function runs before `I` moves to its new destination.
  BasicBlock *BB = To.getNodeParent();
  assert(!(To != BB->end() && &*To == I->getNextNode()) &&
         !(To == BB->end() && std::next(I->getIterator()) == BB->end()) &&
         "Should not have been called if destination is same as origin.");

  // TODO: We can only handle fully internal movements within DAGInterval or at
  // the borders, i.e., right before the top or right after the bottom.
  assert(To.getNodeParent() == I->getParent() &&
         "TODO: We don't support movement across BBs!");
  assert(
      (To == std::next(DAGInterval.bottom()->getIterator()) ||
       (To != BB->end() && std::next(To) == DAGInterval.top()->getIterator()) ||
       (To != BB->end() && DAGInterval.contains(&*To))) &&
      "TODO: To should be either within the DAGInterval or right "
      "before/after it.");

  // Make a copy of the DAGInterval before we update it.
  auto OrigDAGInterval = DAGInterval;

  // Maintain the DAGInterval.
  DAGInterval.notifyMoveInstr(I, To);

  // TODO: Perhaps check if this is legal by checking the dependencies?

  // Update the MemDGNode chain to reflect the instr movement if necessary.
  DGNode *N = getNodeOrNull(I);
  if (N == nullptr)
    return;
  MemDGNode *MemN = dyn_cast<MemDGNode>(N);
  if (MemN == nullptr)
    return;

  // First safely detach it from the existing chain.
  MemN->detachFromChain();

  // Now insert it back into the chain at the new location.
  //
  // We won't always have a DGNode to insert before it. If `To` is BB->end() or
  // if it points to an instr after DAGInterval.bottom() then we will have to
  // find a node to insert *after*.
  //
  // BB:                              BB:
  //  I1                               I1 ^
  //  I2                               I2 | DAGInteval [I1 to I3]
  //  I3                               I3 V
  //  I4                               I4   <- `To` == right after DAGInterval
  //    <- `To` == BB->end()
  //
  if (To == BB->end() ||
      To == std::next(OrigDAGInterval.bottom()->getIterator())) {
    // If we don't have a node to insert before, find a node to insert after and
    // update the chain.
    DGNode *InsertAfterN = getNode(&*std::prev(To));
    MemN->setPrevNode(
        getMemDGNodeBefore(InsertAfterN, /*IncludingN=*/true, /*SkipN=*/MemN));
  } else {
    // We have a node to insert before, so update the chain.
    DGNode *BeforeToN = getNode(&*To);
    MemN->setPrevNode(
        getMemDGNodeBefore(BeforeToN, /*IncludingN=*/false, /*SkipN=*/MemN));
    MemN->setNextNode(
        getMemDGNodeAfter(BeforeToN, /*IncludingN=*/true, /*SkipN=*/MemN));
  }
}

void DependencyGraph::notifyEraseInstr(Instruction *I) {
  // Update the MemDGNode chain if this is a memory node.
  if (auto *MemN = dyn_cast_or_null<MemDGNode>(getNodeOrNull(I))) {
    auto *PrevMemN = getMemDGNodeBefore(MemN, /*IncludingN=*/false);
    auto *NextMemN = getMemDGNodeAfter(MemN, /*IncludingN=*/false);
    if (PrevMemN != nullptr)
      PrevMemN->NextMemN = NextMemN;
    if (NextMemN != nullptr)
      NextMemN->PrevMemN = PrevMemN;
  }

  InstrToNodeMap.erase(I);

  // TODO: Update the dependencies.
}

Interval<Instruction> DependencyGraph::extend(ArrayRef<Instruction *> Instrs) {
  if (Instrs.empty())
    return {};

  Interval<Instruction> InstrsInterval(Instrs);
  Interval<Instruction> Union = DAGInterval.getUnionInterval(InstrsInterval);
  auto NewInterval = Union.getSingleDiff(DAGInterval);
  if (NewInterval.empty())
    return {};

  createNewNodes(NewInterval);

  // Create the dependencies.
  //
  // 1. This is a new DAG, DAGInterval is empty. Fully scan the whole interval.
  // +---+       -             -
  // |   | SrcN  |             |
  // |   |  |    | SrcRange    |
  // |New|  v    |             | DstRange
  // |   | DstN  -             |
  // |   |                     |
  // +---+                     -
  // We are scanning for deps with destination in NewInterval and sources in
  // NewInterval until DstN, for each DstN.
  auto FullScan = [this](const Interval<Instruction> Intvl) {
    auto DstRange = MemDGNodeIntervalBuilder::make(Intvl, *this);
    if (!DstRange.empty()) {
      for (MemDGNode &DstN : drop_begin(DstRange)) {
        auto SrcRange = Interval<MemDGNode>(DstRange.top(), DstN.getPrevNode());
        scanAndAddDeps(DstN, SrcRange);
      }
    }
  };
  if (DAGInterval.empty()) {
    assert(NewInterval == InstrsInterval && "Expected empty DAGInterval!");
    FullScan(NewInterval);
  }
  // 2. The new section is below the old section.
  // +---+       -
  // |   |       |
  // |Old| SrcN  |
  // |   |  |    |
  // +---+  |    | SrcRange
  // +---+  |    |             -
  // |   |  |    |             |
  // |New|  v    |             | DstRange
  // |   | DstN  -             |
  // |   |                     |
  // +---+                     -
  // We are scanning for deps with destination in NewInterval because the deps
  // in DAGInterval have already been computed. We consider sources in the whole
  // range including both NewInterval and DAGInterval until DstN, for each DstN.
  else if (DAGInterval.bottom()->comesBefore(NewInterval.top())) {
    auto DstRange = MemDGNodeIntervalBuilder::make(NewInterval, *this);
    auto SrcRangeFull = MemDGNodeIntervalBuilder::make(
        DAGInterval.getUnionInterval(NewInterval), *this);
    for (MemDGNode &DstN : DstRange) {
      auto SrcRange =
          Interval<MemDGNode>(SrcRangeFull.top(), DstN.getPrevNode());
      scanAndAddDeps(DstN, SrcRange);
    }
  }
  // 3. The new section is above the old section.
  else if (NewInterval.bottom()->comesBefore(DAGInterval.top())) {
    // +---+       -             -
    // |   | SrcN  |             |
    // |New|  |    | SrcRange    | DstRange
    // |   |  v    |             |
    // |   | DstN  -             |
    // |   |                     |
    // +---+                     -
    // +---+
    // |Old|
    // |   |
    // +---+
    // When scanning for deps with destination in NewInterval we need to fully
    // scan the interval. This is the same as the scanning for a new DAG.
    FullScan(NewInterval);

    // +---+       -
    // |   |       |
    // |New| SrcN  | SrcRange
    // |   |  |    |
    // |   |  |    |
    // |   |  |    |
    // +---+  |    -
    // +---+  |                  -
    // |Old|  v                  | DstRange
    // |   | DstN                |
    // +---+                     -
    // When scanning for deps with destination in DAGInterval we need to
    // consider sources from the NewInterval only, because all intra-DAGInterval
    // dependencies have already been created.
    auto DstRangeOld = MemDGNodeIntervalBuilder::make(DAGInterval, *this);
    auto SrcRange = MemDGNodeIntervalBuilder::make(NewInterval, *this);
    for (MemDGNode &DstN : DstRangeOld)
      scanAndAddDeps(DstN, SrcRange);
  } else {
    llvm_unreachable("We don't expect extending in both directions!");
  }

  DAGInterval = Union;
  return NewInterval;
}

#ifndef NDEBUG
void DependencyGraph::print(raw_ostream &OS) const {
  // InstrToNodeMap is unordered so we need to create an ordered vector.
  SmallVector<DGNode *> Nodes;
  Nodes.reserve(InstrToNodeMap.size());
  for (const auto &Pair : InstrToNodeMap)
    Nodes.push_back(Pair.second.get());
  // Sort them based on which one comes first in the BB.
  sort(Nodes, [](DGNode *N1, DGNode *N2) {
    return N1->getInstruction()->comesBefore(N2->getInstruction());
  });
  for (auto *N : Nodes)
    N->print(OS, /*PrintDeps=*/true);
}

void DependencyGraph::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

} // namespace llvm::sandboxir
