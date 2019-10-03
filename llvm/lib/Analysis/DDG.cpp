//===- DDG.cpp - Data Dependence Graph -------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The implementation for the data dependence graph.
//===----------------------------------------------------------------------===//
#include "llvm/Analysis/DDG.h"
#include "llvm/Analysis/LoopInfo.h"

using namespace llvm;

#define DEBUG_TYPE "ddg"

template class llvm::DGEdge<DDGNode, DDGEdge>;
template class llvm::DGNode<DDGNode, DDGEdge>;
template class llvm::DirectedGraph<DDGNode, DDGEdge>;

//===--------------------------------------------------------------------===//
// DDGNode implementation
//===--------------------------------------------------------------------===//
DDGNode::~DDGNode() {}

bool DDGNode::collectInstructions(
    llvm::function_ref<bool(Instruction *)> const &Pred,
    InstructionListType &IList) const {
  assert(IList.empty() && "Expected the IList to be empty on entry.");
  if (isa<SimpleDDGNode>(this)) {
    for (auto *I : cast<const SimpleDDGNode>(this)->getInstructions())
      if (Pred(I))
        IList.push_back(I);
  } else
    llvm_unreachable("unimplemented type of node");
  return !IList.empty();
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const DDGNode::NodeKind K) {
  const char *Out;
  switch (K) {
  case DDGNode::NodeKind::SingleInstruction:
    Out = "single-instruction";
    break;
  case DDGNode::NodeKind::MultiInstruction:
    Out = "multi-instruction";
    break;
  case DDGNode::NodeKind::Root:
    Out = "root";
    break;
  case DDGNode::NodeKind::Unknown:
    Out = "??";
    break;
  }
  OS << Out;
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const DDGNode &N) {
  OS << "Node Address:" << &N << ":" << N.getKind() << "\n";
  if (isa<SimpleDDGNode>(N)) {
    OS << " Instructions:\n";
    for (auto *I : cast<const SimpleDDGNode>(N).getInstructions())
      OS.indent(2) << *I << "\n";
  } else if (!isa<RootDDGNode>(N))
    llvm_unreachable("unimplemented type of node");

  OS << (N.getEdges().empty() ? " Edges:none!\n" : " Edges:\n");
  for (auto &E : N.getEdges())
    OS.indent(2) << *E;
  return OS;
}

//===--------------------------------------------------------------------===//
// SimpleDDGNode implementation
//===--------------------------------------------------------------------===//

SimpleDDGNode::SimpleDDGNode(Instruction &I)
  : DDGNode(NodeKind::SingleInstruction), InstList() {
  assert(InstList.empty() && "Expected empty list.");
  InstList.push_back(&I);
}

SimpleDDGNode::SimpleDDGNode(const SimpleDDGNode &N)
    : DDGNode(N), InstList(N.InstList) {
  assert(((getKind() == NodeKind::SingleInstruction && InstList.size() == 1) ||
          (getKind() == NodeKind::MultiInstruction && InstList.size() > 1)) &&
         "constructing from invalid simple node.");
}

SimpleDDGNode::SimpleDDGNode(SimpleDDGNode &&N)
    : DDGNode(std::move(N)), InstList(std::move(N.InstList)) {
  assert(((getKind() == NodeKind::SingleInstruction && InstList.size() == 1) ||
          (getKind() == NodeKind::MultiInstruction && InstList.size() > 1)) &&
         "constructing from invalid simple node.");
}

SimpleDDGNode::~SimpleDDGNode() { InstList.clear(); }

//===--------------------------------------------------------------------===//
// DDGEdge implementation
//===--------------------------------------------------------------------===//

raw_ostream &llvm::operator<<(raw_ostream &OS, const DDGEdge::EdgeKind K) {
  const char *Out;
  switch (K) {
  case DDGEdge::EdgeKind::RegisterDefUse:
    Out = "def-use";
    break;
  case DDGEdge::EdgeKind::MemoryDependence:
    Out = "memory";
    break;
  case DDGEdge::EdgeKind::Rooted:
    Out = "rooted";
    break;
  case DDGEdge::EdgeKind::Unknown:
    Out = "??";
    break;
  }
  OS << Out;
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const DDGEdge &E) {
  OS << "[" << E.getKind() << "] to " << &E.getTargetNode() << "\n";
  return OS;
}

//===--------------------------------------------------------------------===//
// DataDependenceGraph implementation
//===--------------------------------------------------------------------===//
using BasicBlockListType = SmallVector<BasicBlock *, 8>;

DataDependenceGraph::DataDependenceGraph(Function &F, DependenceInfo &D)
    : DependenceGraphInfo(F.getName().str(), D) {
  BasicBlockListType BBList;
  for (auto &BB : F.getBasicBlockList())
    BBList.push_back(&BB);
  DDGBuilder(*this, D, BBList).populate();
}

DataDependenceGraph::DataDependenceGraph(const Loop &L, DependenceInfo &D)
    : DependenceGraphInfo(Twine(L.getHeader()->getParent()->getName() + "." +
                                L.getHeader()->getName())
                              .str(),
                          D) {
  BasicBlockListType BBList;
  for (BasicBlock *BB : L.blocks())
    BBList.push_back(BB);
  DDGBuilder(*this, D, BBList).populate();
}

DataDependenceGraph::~DataDependenceGraph() {
  for (auto *N : Nodes) {
    for (auto *E : *N)
      delete E;
    delete N;
  }
}

bool DataDependenceGraph::addNode(DDGNode &N) {
  if (!DDGBase::addNode(N))
    return false;

  // In general, if the root node is already created and linked, it is not safe
  // to add new nodes since they may be unreachable by the root.
  // TODO: Allow adding Pi-block nodes after root is created. Pi-blocks are an
  // exception because they represent components that are already reachable by
  // root.
  assert(!Root && "Root node is already added. No more nodes can be added.");
  if (isa<RootDDGNode>(N))
    Root = &N;

  return true;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const DataDependenceGraph &G) {
  for (auto *Node : G)
    OS << *Node << "\n";
  return OS;
}

//===--------------------------------------------------------------------===//
// DDG Analysis Passes
//===--------------------------------------------------------------------===//

/// DDG as a loop pass.
DDGAnalysis::Result DDGAnalysis::run(Loop &L, LoopAnalysisManager &AM,
                                     LoopStandardAnalysisResults &AR) {
  Function *F = L.getHeader()->getParent();
  DependenceInfo DI(F, &AR.AA, &AR.SE, &AR.LI);
  return std::make_unique<DataDependenceGraph>(L, DI);
}
AnalysisKey DDGAnalysis::Key;

PreservedAnalyses DDGAnalysisPrinterPass::run(Loop &L, LoopAnalysisManager &AM,
                                              LoopStandardAnalysisResults &AR,
                                              LPMUpdater &U) {
  OS << "'DDG' for loop '" << L.getHeader()->getName() << "':\n";
  OS << *AM.getResult<DDGAnalysis>(L, AR);
  return PreservedAnalyses::all();
}
