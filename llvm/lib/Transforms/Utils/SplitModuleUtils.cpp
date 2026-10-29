//===- SplitModuleUtils.cpp - Shared module splitting utilities ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SplitModuleUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <cassert>

using namespace llvm;
using namespace llvm::splitmodule;

SplitGraph::~SplitGraph() = default;

void SplitGraph::Node::visitAllDependencies(
    std::function<void(const Node &)> Visitor) const {
  const bool FollowIndirect = shouldFollowIndirectCalls();
  // FIXME: If this can access SplitGraph in the future, use a BitVector
  // instead.
  DenseSet<const Node *> Seen;
  SmallVector<const Node *, 8> WorkList({this});
  while (!WorkList.empty()) {
    const Node *CurN = WorkList.pop_back_val();
    if (auto [It, Inserted] = Seen.insert(CurN); !Inserted)
      continue;

    Visitor(*CurN);

    for (const Edge *E : CurN->outgoing_edges()) {
      if (!FollowIndirect && E->Kind == EdgeKind::IndirectCall)
        continue;
      WorkList.push_back(E->Dst);
    }
  }
}

#ifndef NDEBUG
bool SplitGraph::verifyGraph() const {
  unsigned ExpectedID = 0;
  // Exceptionally using a set here in case IDs are messed up.
  DenseSet<const Node *> SeenNodes;
  DenseSet<const Function *> SeenFunctionNodes;
  for (const Node *N : Nodes) {
    if (N->getID() != (ExpectedID++)) {
      errs() << "Node IDs are incorrect!\n";
      return false;
    }

    if (!SeenNodes.insert(N).second) {
      errs() << "Node seen more than once!\n";
      return false;
    }

    if (&getNode(N->getID()) != N) {
      errs() << "getNode doesn't return the right node\n";
      return false;
    }

    for (const Edge *E : N->IncomingEdges) {
      if (!E->Src || !E->Dst || (E->Dst != N) ||
          (find(E->Src->OutgoingEdges, E) == E->Src->OutgoingEdges.end())) {
        errs() << "ill-formed incoming edges\n";
        return false;
      }
    }

    for (const Edge *E : N->OutgoingEdges) {
      if (!E->Src || !E->Dst || (E->Src != N) ||
          (find(E->Dst->IncomingEdges, E) == E->Dst->IncomingEdges.end())) {
        errs() << "ill-formed outgoing edges\n";
        return false;
      }
    }

    const Function &Fn = N->getFunction();
    if (isEntryFunctionCC(Fn)) {
      if (N->hasAnyIncomingEdges()) {
        errs() << "Kernels cannot have incoming edges\n";
        return false;
      }
    }

    if (Fn.isDeclaration()) {
      errs() << "declarations shouldn't have nodes!\n";
      return false;
    }

    auto [It, Inserted] = SeenFunctionNodes.insert(&Fn);
    if (!Inserted) {
      errs() << "one function has multiple nodes!\n";
      return false;
    }
  }

  if (ExpectedID != Nodes.size()) {
    errs() << "Node IDs out of sync!\n";
    return false;
  }

  if (createNodesBitVector().size() != getNumNodes()) {
    errs() << "nodes bit vector doesn't have the right size!\n";
    return false;
  }

  // Check we respect the promise of Node::isKernel
  BitVector BV = createNodesBitVector();
  for (const Node *N : nodes()) {
    if (N->isGraphEntryPoint())
      N->getDependencies(BV);
  }

  // Ensure each function in the module has an associated node.
  for (const auto &Fn : M) {
    if (!Fn.isDeclaration()) {
      if (!SeenFunctionNodes.contains(&Fn)) {
        errs() << "Fn has no associated node in the graph!\n";
        return false;
      }
    }
  }

  if (!BV.all()) {
    errs() << "not all nodes are reachable through the graph's entry points!\n";
    return false;
  }

  return true;
}
#endif

CostType SplitGraph::calculateCost(const BitVector &BV) const {
  CostType Cost = 0;
  for (unsigned NodeID : BV.set_bits())
    Cost += getNode(NodeID).getIndividualCost();
  return Cost;
}

SplitGraph::Node &
SplitGraph::getNode(DenseMap<const GlobalValue *, Node *> &Cache,
                    const GlobalValue &GV) {
  auto &N = Cache[&GV];
  if (N)
    return *N;

  CostType Cost = 0;
  bool NonCopyable = false;
  bool EntryFnCC = false;
  if (const Function *Fn = dyn_cast<Function>(&GV)) {
    NonCopyable = isNonCopyable(*Fn);
    Cost = CostMap.at(Fn);
    EntryFnCC = isEntryFunctionCC(*Fn);
  }
  N = new (NodesPool.Allocate())
      Node(Nodes.size(), GV, Cost, NonCopyable, EntryFnCC);
  Nodes.push_back(N);
  assert(&getNode(N->getID()) == N);
  return *N;
}

const SplitGraph::Edge &SplitGraph::createEdge(Node &Src, Node &Dst,
                                               EdgeKind EK) {
  const Edge *E = new (EdgesPool.Allocate<Edge>(1)) Edge(&Src, &Dst, EK);
  Src.OutgoingEdges.push_back(E);
  Dst.IncomingEdges.push_back(E);
  return *E;
}

void SplitGraph::markAsGraphEntry(Node &N) { N.markAsGraphEntry(); }

FunctionCost llvm::splitmodule::calculateFunctionCosts(
    Module &M, function_ref<FunctionCost(Function &F)> CalculateFunctionCost,
    FunctionCostMap &FunctionCosts) {
  FunctionCost ModuleCost = 0;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    FunctionCost Cost = CalculateFunctionCost(F);
    assert(Cost != 0 && "function must have a non-zero cost");
    FunctionCosts[&F] = Cost;
    assert(static_cast<uint64_t>(ModuleCost) + static_cast<uint64_t>(Cost) >=
               static_cast<uint64_t>(ModuleCost) &&
           "Overflow!");
    ModuleCost += Cost;
  }
  return ModuleCost;
}

void llvm::splitmodule::materializePartitions(
    const Module &M, unsigned PartitionId,
    function_ref<bool(const GlobalValue *GV, unsigned PartitionId)>
        ShouldCloneDefinition,
    function_ref<void(std::unique_ptr<Module> MPart, unsigned PartitionId)>
        ModuleCallback,
    ModuleAsmPolicy AsmPolicy) {
  ValueToValueMapTy VMap;
  auto MPart = CloneModule(M, VMap, [&](const GlobalValue *GV) {
    return ShouldCloneDefinition(GV, PartitionId);
  });
  if (AsmPolicy == ModuleAsmPolicy::PreserveFirst && PartitionId != 0)
    MPart->removeModuleInlineAsm();
  ModuleCallback(std::move(MPart), PartitionId);
}
