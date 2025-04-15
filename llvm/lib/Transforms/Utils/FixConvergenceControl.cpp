//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A utility function to fix convergence control tokens in the presence of
// irreducible control flow.
//
// When generating convergence control tokens for iteration statements, CodeGen
// ignores the presence of any jumps that may cause the resulting CFG to be
// irreducible. CodeGen optimistically emits a `loop` intrinsic in the header of
// every iteration statement, which is invalid if control can jump into that
// statement. We fix this in the LLVM IR after the whole function is fully
// generated. For each irreducible cycle discovered in the CFG, we replace the
// `loop` intrinsic in the header with suitable invocations of `anchor` instead.
//
// Separately, a "backwards goto" can create a cycle that encapsulates an
// iteration statement. As a result, the `loop` intrinsic in that iteration
// statement no longer uses a token from its immediate parent cycle in the CFG.
// This is also fixed by replacing the `loop` intrinsic with an `anchor`
// intrinsic.
//
// The overall result is to produce in implementation-defined convergence as a
// result of unstructured jumps as defined in the Clang spec for convergence.
//
// ===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/FixConvergenceControl.h"
#include "llvm/IR/CycleInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include <map>

#define DEBUG_TYPE "convergence-fixup"

using namespace llvm;

static void
initializeTokenSources(SmallVectorImpl<ConvergenceControlInst *> &Worklist,
                       Function *F) {
  for (Instruction &II : instructions(F)) {
    if (auto *CB = dyn_cast<ConvergenceControlInst>(&II)) {
      if (CB->isEntry() || CB->isAnchor())
        Worklist.push_back(CB);
    }
  }
}

namespace llvm {

void fixConvergenceControl(Function *F) {
  CycleInfo CI;
  CI.compute(*F);

  // F->dump();
  // CI.dump();

  enum DecisionTy { Delete, Replace };

  // Can't use a DenseMap because we later insert while iterating.
  std::map<CallBase *, DecisionTy> Decision;
  SmallPtrSet<CallBase *, 4> NonIntrinsicUsers;

  SmallVector<ConvergenceControlInst *> Worklist;
  initializeTokenSources(Worklist, F);

  while (!Worklist.empty()) {
    ConvergenceControlInst *CB = Worklist.pop_back_val();
    LLVM_DEBUG(llvm::dbgs() << "Visiting: " << *CB << "\n");
    Cycle *CurrentCycle = CI.getCycle(CB->getParent());

    for (Use &U : CB->uses()) {
      auto *UserCB = cast<CallBase>(U.getUser());
      if (auto *C = dyn_cast<ConvergenceControlInst>(UserCB)) {
        Worklist.push_back(C);
        continue;
      }
      Cycle *UserCycle = CI.getCycle(UserCB->getParent());
      // A non-intrinsic user cannot use a token defined outside its own cycle.
      if (UserCycle && !UserCycle->contains(CurrentCycle))
        NonIntrinsicUsers.insert(UserCB);
    }

    if (!CurrentCycle)
      continue;

    // A loop intrinsic is no longer useful in two cases:
    // 1. Its cycle became irreducible, or,
    // 2. The cycle appears `rotated` in the CFG and the call is not in the
    //    header. This happens when a loop statement is unreachable via
    //    sequential control flow, but is jumped into by a goto or switch.
    //
    // We will be visiting its users later.
    if (!CurrentCycle->isReducible() ||
        CurrentCycle->getHeader() != CB->getParent()) {
      LLVM_DEBUG(llvm::dbgs() << "  Delete.\n");
      Decision[CB] = Delete;
      continue;
    }

    // A token use is valid only if the def is with the immediate parent. It's
    // okay if the def is with a sibling, as long as the common parent is the
    // immediate parent.
    //
    // The def can end up outside the parent when a goto forms a reducible cycle
    // around a loop statement. Such a new reducible cycle does not itself have
    // a heart.
    ConvergenceControlInst *TokenUsed = CB->getConvergenceControlToken();
    Cycle *DefCycle = CI.getCycle(TokenUsed->getParent());
    assert(CurrentCycle == DefCycle || !CurrentCycle->contains(DefCycle));
    Cycle *Parent = CurrentCycle->getParentCycle();
    if (DefCycle != Parent && Parent && !Parent->contains(DefCycle)) {
      // Don't overwrite if previous decision was to delete.
      Decision.try_emplace(CB, Replace);
    }
  }

  SmallVector<CallBase *> ToDelete;
  // For deletion candidates, decide how to process each of the uses.
  for (auto [CB, D] : Decision) {
    if (D != Delete)
      continue;
    ToDelete.push_back(CB);

    for (Use &U : CB->uses()) {
      auto *ConvOp = cast<CallBase>(U.getUser());
      // Users that are calls to the loop intrinsic can no longer use this as
      // the parent token, so replace them with anchors.
      if (auto *Child = dyn_cast<ConvergenceControlInst>(ConvOp)) {
        // Don't overwrite if previous decision was to delete. Note that we are
        // inserting while iterating over the std::map. It is possible that the
        // newly inserted node is not visited, which is okay because we are only
        // iterating over candidates mapped to ``Delete``.
        Decision.try_emplace(Child, Replace);
        continue;
      }
      // Other convergent users should be made non-converent.
      NonIntrinsicUsers.insert(ConvOp);
    }
  }

  for (auto [CB, D] : Decision) {
    if (D != Replace)
      continue;
    Cycle *CurrentCycle = CI.getCycle(CB->getParent());
    assert(CurrentCycle && CurrentCycle->isReducible());
    LLVM_DEBUG(llvm::dbgs() << "  Replace with anchor: " << *CB << "\n");
    auto *Anchor = ConvergenceControlInst::CreateAnchor(*CB->getParent());
    CB->replaceAllUsesWith(Anchor);
    CB->eraseFromParent();
  }

  // Make all non-intrinsic users non-convergent. It would have been convenient
  // to just strip the token and the ``convergent`` attribute, but attributes
  // get checked on the callee too if they don't exist on the call. We could
  // have set the ``noconvergent`` attribute if it existed. For now,
  // equivalently, we replace the token with an anchor.
  for (CallBase *CB : NonIntrinsicUsers) {
    auto *Token = ConvergenceControlInst::CreateAnchor(*CB->getParent());
    CB = setConvergenceControlToken(CB, Token);
    LLVM_DEBUG(llvm::dbgs() << "  Make non-convergent: " << *CB << "\n");
  }

  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (unsigned I = 0, E = ToDelete.size(); I != E; ++I) {
      CallBase *CB = ToDelete[I];
      if (CB) {
        LLVM_DEBUG(llvm::dbgs() << "Try delete:\n" << *CB << "\n");
        if (!CB->use_empty()) {
          LLVM_DEBUG(llvm::dbgs() << " ... has pending use.\n");
          continue;
        }
        LLVM_DEBUG(llvm::dbgs() << " ... deleted.\n");
        CB->eraseFromParent();
        ToDelete[I] = nullptr;
        Changed = true;
      }
    }
  }
  LLVM_DEBUG(for (CallBase *CB : ToDelete) assert(!CB););

  // F->dump();
}

} // end namespace llvm
