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

Interval<Instruction> DependencyGraph::extend(ArrayRef<Instruction *> Instrs) {
  if (Instrs.empty())
    return {};
  // TODO: For now create a chain of dependencies.
  Interval<Instruction> Interval(Instrs);
  auto *TopI = Interval.top();
  auto *BotI = Interval.bottom();
  DGNode *LastN = getOrCreateNode(TopI);
  for (Instruction *I = TopI->getNextNode(), *E = BotI->getNextNode(); I != E;
       I = I->getNextNode()) {
    auto *N = getOrCreateNode(I);
    N->addMemPred(LastN);
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
