//===- MaterializationUtils.cpp - Builds and manipulates coroutine frame
//-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains classes used to materialize insts after suspends points.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Coroutines/MaterializationUtils.h"
#include "CoroInternal.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/Transforms/Coroutines/SpillUtils.h"
#include <deque>

using namespace llvm;

using namespace coro;

// The "coro-suspend-crossing" flag is very noisy. There is another debug type,
// "coro-frame", which results in leaner debug spew.
#define DEBUG_TYPE "coro-suspend-crossing"

// Returns true if \p Root can be rematerialized in the resume function without
// introducing a new spill. This holds when a value is:
// 1. A constant: genuinely available at no cost, materialized as an immediate
//    in the resume function.
// 2. An argument: treated as available heuristically. An argument crossing the
//    suspend is actually spilled, but treating it as available preserves
//    beneficial shared-operand rematerialization.
// 3. A value that already crosses a suspend point for an independent use, i.e.,
//    it is spilled regardless.
// 4. A materializable value all of whose operands are themselves available.
static bool isAvailableAfterSuspend(
    Value *Root, const std::function<bool(Instruction &)> &Materializable,
    const SuspendCrossingInfo &Checker, SmallDenseMap<Value *, bool> &Memo) {
  SmallVector<Value *> Stack;
  // Materializable nodes whose operands have been scheduled but not yet folded.
  // Distinguishes a node being visited a second time (operands ready) from the
  // first visit, and lets an operand reached through a cycle resolve to false.
  SmallPtrSet<Value *, 16> Opened;
  Stack.push_back(Root);
  while (!Stack.empty()) {
    Value *V = Stack.back();
    if (Memo.contains(V)) {
      Stack.pop_back();
      continue;
    }

    // Leaves that resolve without inspecting operands. Non-instructions
    // (constants, arguments) are available; a value that already crosses a
    // suspend independently is spilled regardless, so referencing its slot is
    // available at no new cost.
    auto *I = dyn_cast<Instruction>(V);
    if (!I || Checker.isDefinitionAcrossSuspend(*I)) {
      Memo[V] = true;
      Stack.pop_back();
      continue;
    }
    if (!Materializable(*I)) {
      Memo[V] = false;
      Stack.pop_back();
      continue;
    }

    // Materializable: available iff all operands are available. On the first
    // visit, schedule the unresolved operands above V and revisit V once they
    // are folded.
    if (Opened.insert(V).second) {
      for (Use &U : I->operands())
        if (!Memo.contains(U.get()))
          Stack.push_back(U.get());
      continue;
    }

    // Second visit: operands are resolved, except any reached through a cycle
    // (only possible via non-materializable PHIs, which never open). An
    // unresolved operand is treated as not available, conservatively.
    Stack.pop_back();
    Memo[V] = all_of(I->operands(), [&](Use &U) {
      auto It = Memo.find(U.get());
      return It != Memo.end() && It->second;
    });
  }
  return Memo.lookup(Root);
}

// Returns true if every operand of \p I is available after the suspend. This is
// the profitability condition for rematerializing \p I: recomputing it then
// forces no new value into the coroutine frame. Note this deliberately does NOT
// consult isDefinitionAcrossSuspend(I) on I itself -- every remat candidate
// crosses a suspend by definition, so that would make the check a no-op.
static bool allOperandsAvailableAfterSuspend(
    Instruction &I, const std::function<bool(Instruction &)> &Materializable,
    const SuspendCrossingInfo &Checker, SmallDenseMap<Value *, bool> &Memo) {
  return all_of(I.operands(), [&](Use &U) {
    return isAvailableAfterSuspend(U.get(), Materializable, Checker, Memo);
  });
}

namespace {

// RematGraph is used to construct a DAG for rematerializable instructions
// When the constructor is invoked with a candidate instruction (which is
// materializable) it builds a DAG of materializable instructions from that
// point.
// Typically, for each instruction identified as re-materializable across a
// suspend point, a RematGraph will be created.
struct RematGraph {
  // Each RematNode in the graph contains the edges to instructions providing
  // operands in the current node.
  struct RematNode {
    Instruction *Node;
    SmallVector<RematNode *> Operands;
    RematNode() = default;
    RematNode(Instruction *V) : Node(V) {}
  };

  RematNode *EntryNode;
  using RematNodeMap =
      SmallMapVector<Instruction *, std::unique_ptr<RematNode>, 8>;
  RematNodeMap Remats;
  const std::function<bool(Instruction &)> &MaterializableCallback;
  SuspendCrossingInfo &Checker;
  // Shared across all RematGraphs and the candidate scan in
  // doRematerializations: isAvailableAfterSuspend is context-free and the IR is
  // not mutated until rematerialization, so results can be cached across uses.
  SmallDenseMap<Value *, bool> &AvailMemo;

  RematGraph(const std::function<bool(Instruction &)> &MaterializableCallback,
             Instruction *I, SuspendCrossingInfo &Checker,
             SmallDenseMap<Value *, bool> &AvailMemo)
      : MaterializableCallback(MaterializableCallback), Checker(Checker),
        AvailMemo(AvailMemo) {
    std::unique_ptr<RematNode> FirstNode = std::make_unique<RematNode>(I);
    EntryNode = FirstNode.get();
    std::deque<std::unique_ptr<RematNode>> WorkList;
    addNode(std::move(FirstNode), WorkList, cast<User>(I));
    while (WorkList.size()) {
      std::unique_ptr<RematNode> N = std::move(WorkList.front());
      WorkList.pop_front();
      addNode(std::move(N), WorkList, cast<User>(I));
    }
  }

  void addNode(std::unique_ptr<RematNode> NUPtr,
               std::deque<std::unique_ptr<RematNode>> &WorkList,
               User *FirstUse) {
    RematNode *N = NUPtr.get();
    auto [It, Inserted] = Remats.try_emplace(N->Node);
    if (!Inserted)
      return;

    // We haven't see this node yet - add to the list
    It->second = std::move(NUPtr);
    for (auto &Def : N->Node->operands()) {
      Instruction *D = dyn_cast<Instruction>(Def.get());
      if (!D || !MaterializableCallback(*D) ||
          !Checker.isDefinitionAcrossSuspend(*D, FirstUse) ||
          !allOperandsAvailableAfterSuspend(*D, MaterializableCallback, Checker,
                                            AvailMemo))
        continue;

      if (auto It = Remats.find(D); It != Remats.end()) {
        // Already have this in the graph
        N->Operands.push_back(It->second.get());
        continue;
      }

      bool NoMatch = true;
      for (auto &I : WorkList) {
        if (I->Node == D) {
          NoMatch = false;
          N->Operands.push_back(I.get());
          break;
        }
      }
      if (NoMatch) {
        // Create a new node
        std::unique_ptr<RematNode> ChildNode = std::make_unique<RematNode>(D);
        N->Operands.push_back(ChildNode.get());
        WorkList.push_back(std::move(ChildNode));
      }
    }
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  static void dumpBasicBlockLabel(const BasicBlock *BB,
                                  ModuleSlotTracker &MST) {
    if (BB->hasName()) {
      dbgs() << BB->getName();
      return;
    }

    dbgs() << MST.getLocalSlot(BB);
  }

  void dump() const {
    BasicBlock *BB = EntryNode->Node->getParent();
    Function *F = BB->getParent();

    ModuleSlotTracker MST(F->getParent());
    MST.incorporateFunction(*F);

    dbgs() << "Entry (";
    dumpBasicBlockLabel(BB, MST);
    dbgs() << ") : " << *EntryNode->Node << "\n";
    for (auto &E : Remats) {
      dbgs() << *(E.first) << "\n";
      for (RematNode *U : E.second->Operands)
        dbgs() << "  " << *U->Node << "\n";
    }
  }
#endif
};

} // namespace

template <> struct llvm::GraphTraits<RematGraph *> {
  using NodeRef = RematGraph::RematNode *;
  using ChildIteratorType = RematGraph::RematNode **;

  static NodeRef getEntryNode(RematGraph *G) { return G->EntryNode; }
  static ChildIteratorType child_begin(NodeRef N) {
    return N->Operands.begin();
  }
  static ChildIteratorType child_end(NodeRef N) { return N->Operands.end(); }
};

// For each instruction identified as materializable across the suspend point,
// and its associated DAG of other rematerializable instructions,
// recreate the DAG of instructions after the suspend point.
static void rewriteMaterializableInstructions(
    const SmallMapVector<Instruction *, std::unique_ptr<RematGraph>, 8>
        &AllRemats) {
  // This has to be done in 2 phases
  // Do the remats and record the required defs to be replaced in the
  // original use instructions
  // Once all the remats are complete, replace the uses in the final
  // instructions with the new defs
  typedef struct {
    Instruction *Use;
    Instruction *Def;
    Instruction *Remat;
  } ProcessNode;

  SmallVector<ProcessNode> FinalInstructionsToProcess;

  for (const auto &E : AllRemats) {
    Instruction *Use = E.first;
    Instruction *CurrentMaterialization = nullptr;
    RematGraph *RG = E.second.get();
    ReversePostOrderTraversal<RematGraph *> RPOT(RG);
    SmallVector<Instruction *> InstructionsToProcess;

    // If the target use is actually a suspend instruction then we have to
    // insert the remats into the end of the predecessor (there should only be
    // one). This is so that suspend blocks always have the suspend instruction
    // as the first instruction.
    BasicBlock::iterator InsertPoint = Use->getParent()->getFirstInsertionPt();
    if (isa<AnyCoroSuspendInst>(Use)) {
      BasicBlock *SuspendPredecessorBlock =
          Use->getParent()->getSinglePredecessor();
      assert(SuspendPredecessorBlock && "malformed coro suspend instruction");
      InsertPoint = SuspendPredecessorBlock->getTerminator()->getIterator();
    }

    // Note: skip the first instruction as this is the actual use that we're
    // rematerializing everything for.
    auto I = RPOT.begin();
    ++I;
    for (; I != RPOT.end(); ++I) {
      Instruction *D = (*I)->Node;
      CurrentMaterialization = D->clone();
      CurrentMaterialization->setName(D->getName());
      CurrentMaterialization->insertBefore(InsertPoint);
      InsertPoint = CurrentMaterialization->getIterator();

      // Replace all uses of Def in the instructions being added as part of this
      // rematerialization group
      for (auto &I : InstructionsToProcess)
        I->replaceUsesOfWith(D, CurrentMaterialization);

      // Don't replace the final use at this point as this can cause problems
      // for other materializations. Instead, for any final use that uses a
      // define that's being rematerialized, record the replace values
      for (unsigned i = 0, E = Use->getNumOperands(); i != E; ++i)
        if (Use->getOperand(i) == D) // Is this operand pointing to oldval?
          FinalInstructionsToProcess.push_back(
              {Use, D, CurrentMaterialization});

      InstructionsToProcess.push_back(CurrentMaterialization);
    }
  }

  // Finally, replace the uses with the defines that we've just rematerialized
  for (auto &R : FinalInstructionsToProcess) {
    if (auto *PN = dyn_cast<PHINode>(R.Use)) {
      assert(PN->getNumIncomingValues() == 1 && "unexpected number of incoming "
                                                "values in the PHINode");
      PN->replaceAllUsesWith(R.Remat);
      PN->eraseFromParent();
      continue;
    }
    R.Use->replaceUsesOfWith(R.Def, R.Remat);
  }
}

/// Default materializable callback
// Check for instructions that we can recreate on resume as opposed to spill
// the result into a coroutine frame.
bool llvm::coro::defaultMaterializable(Instruction &V) {
  if (isa<CastInst>(&V) || isa<GetElementPtrInst>(&V) ||
      isa<BinaryOperator>(&V) || isa<UnaryOperator>(&V) || isa<CmpInst>(&V) ||
      isa<SelectInst>(&V))
    return true;

  if (auto *II = dyn_cast<IntrinsicInst>(&V)) {
    switch (II->getIntrinsicID()) {
    // Floating-point unary math
    case Intrinsic::fabs:
    case Intrinsic::sqrt:
    case Intrinsic::sin:
    case Intrinsic::cos:
    case Intrinsic::exp2:
    case Intrinsic::log2:
    // Floating-point binary math
    case Intrinsic::minnum:
    case Intrinsic::maxnum:
    case Intrinsic::minimum:
    case Intrinsic::maximum:
    case Intrinsic::copysign:
    case Intrinsic::ldexp:
    // Floating-point conversion
    case Intrinsic::fptoui_sat:
    case Intrinsic::fptosi_sat:
    case Intrinsic::is_fpclass:
    // Rounding
    case Intrinsic::floor:
    case Intrinsic::ceil:
    case Intrinsic::trunc:
    case Intrinsic::rint:
    case Intrinsic::nearbyint:
    case Intrinsic::round:
    case Intrinsic::roundeven:
    // Integer bit ops
    case Intrinsic::ctpop:
    case Intrinsic::ctlz:
    case Intrinsic::cttz:
    case Intrinsic::bitreverse:
    // Integer saturating arithmetic
    case Intrinsic::sadd_sat:
    case Intrinsic::uadd_sat:
    case Intrinsic::ssub_sat:
    case Intrinsic::usub_sat:
    // Integer min and max
    case Intrinsic::smax:
    case Intrinsic::smin:
    case Intrinsic::umax:
    case Intrinsic::umin:
      return true;
    default:
      break;
    }
  }

  return false;
}

bool llvm::coro::isTriviallyMaterializable(Instruction &V) {
  return defaultMaterializable(V);
}

#ifndef NDEBUG
static void dumpRemats(
    StringRef Title,
    const SmallMapVector<Instruction *, std::unique_ptr<RematGraph>, 8> &RM) {
  dbgs() << "------------- " << Title << "--------------\n";
  for (const auto &E : RM) {
    E.second->dump();
    dbgs() << "--\n";
  }
}
#endif

void coro::doRematerializations(
    Function &F, SuspendCrossingInfo &Checker,
    std::function<bool(Instruction &)> IsMaterializable) {
  if (F.hasOptNone())
    return;

  coro::SpillInfo Spills;

  // See if there are materializable instructions across suspend points
  // We record these as the starting point to also identify materializable
  // defs of uses in these operations
  for (Instruction &I : instructions(F)) {
    if (!IsMaterializable(I))
      continue;
    for (User *U : I.users())
      if (Checker.isDefinitionAcrossSuspend(I, U))
        Spills[&I].push_back(cast<Instruction>(U));
  }

  // Process each of the identified rematerializable instructions
  // and add predecessor instructions that can also be rematerialized.
  // This is actually a graph of instructions since we could potentially
  // have multiple uses of a def in the set of predecessor instructions.
  // The approach here is to maintain a graph of instructions for each bottom
  // level instruction - where we have a unique set of instructions (nodes)
  // and edges between them. We then walk the graph in reverse post-dominator
  // order to insert them past the suspend point, but ensure that ordering is
  // correct. We also rely on CSE removing duplicate defs for remats of
  // different instructions with a def in common (rather than maintaining more
  // complex graphs for each suspend point)

  // Caches whether values are available after a suspend point.
  SmallDenseMap<Value *, bool> AvailMemo;

  // We can do this by adding new nodes to the list for each suspend
  // point. Then using standard GraphTraits to give a reverse post-order
  // traversal when we insert the nodes after the suspend
  SmallMapVector<Instruction *, std::unique_ptr<RematGraph>, 8> AllRemats;
  for (auto &E : Spills) {
    for (Instruction *U : E.second) {
      // Don't process a user twice (this can happen if the instruction uses
      // more than one rematerializable def)
      auto [It, Inserted] = AllRemats.try_emplace(U);
      if (!Inserted)
        continue;

      // Constructor creates the whole RematGraph for the given Use
      auto RematUPtr =
          std::make_unique<RematGraph>(IsMaterializable, U, Checker, AvailMemo);

      LLVM_DEBUG(dbgs() << "***** Next remat group *****\n";
                 ReversePostOrderTraversal<RematGraph *> RPOT(RematUPtr.get());
                 for (auto I = RPOT.begin(); I != RPOT.end();
                      ++I) { (*I)->Node->dump(); } dbgs()
                 << "\n";);

      It->second = std::move(RematUPtr);
    }
  }

  // Rewrite materializable instructions to be materialized at the use
  // point.
  LLVM_DEBUG(dumpRemats("Materializations", AllRemats));
  rewriteMaterializableInstructions(AllRemats);
}
