//===- DependencyGraph.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/DependencyGraph.h"
#include "llvm/ADT/ArrayRef.h"

using namespace llvm::sandboxir;

#ifndef NDEBUG
void DGNode::print(raw_ostream &OS, bool PrintDeps) const {
  I->dumpOS(OS);
  if (PrintDeps) {
    OS << "\n";
    // Print memory preds.
    static constexpr const unsigned Indent = 4;
    for (auto *Pred : MemPreds) {
      OS.indent(Indent) << "<-";
      Pred->print(OS, false);
      OS << "\n";
    }
  }
}
void DGNode::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

Interval<MemDGNode>
MemDGNodeIntervalBuilder::make(const Interval<Instruction> &Instrs,
                               DependencyGraph &DAG) {
  // If top or bottom instructions are not mem-dep candidate nodes we need to
  // walk down/up the chain and find the mem-dep ones.
  Instruction *MemTopI = Instrs.top();
  Instruction *MemBotI = Instrs.bottom();
  while (!DGNode::isMemDepCandidate(MemTopI) && MemTopI != MemBotI)
    MemTopI = MemTopI->getNextNode();
  while (!DGNode::isMemDepCandidate(MemBotI) && MemBotI != MemTopI)
    MemBotI = MemBotI->getPrevNode();
  // If we couldn't find a mem node in range TopN - BotN then it's empty.
  if (!DGNode::isMemDepCandidate(MemTopI))
    return {};
  // Now that we have the mem-dep nodes, create and return the range.
  return Interval<MemDGNode>(cast<MemDGNode>(DAG.getNode(MemTopI)),
                             cast<MemDGNode>(DAG.getNode(MemBotI)));
}

Interval<Instruction> DependencyGraph::extend(ArrayRef<Instruction *> Instrs) {
  if (Instrs.empty())
    return {};
  // TODO: For now create a chain of dependencies.
  Interval<Instruction> Interval(Instrs);
  auto *TopI = Interval.top();
  auto *BotI = Interval.bottom();
  DGNode *LastN = getOrCreateNode(TopI);
  MemDGNode *LastMemN = dyn_cast<MemDGNode>(LastN);
  for (Instruction *I = TopI->getNextNode(), *E = BotI->getNextNode(); I != E;
       I = I->getNextNode()) {
    auto *N = getOrCreateNode(I);
    N->addMemPred(LastMemN);
    // Build the Mem node chain.
    if (auto *MemN = dyn_cast<MemDGNode>(N)) {
      MemN->setPrevNode(LastMemN);
      if (LastMemN != nullptr)
        LastMemN->setNextNode(MemN);
      LastMemN = MemN;
    }
    LastN = N;
  }
  return Interval;
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
