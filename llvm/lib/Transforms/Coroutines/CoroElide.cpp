//===- CoroElide.cpp - Coroutine Frame Allocation Elision Pass ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This pass replaces dynamic allocation of coroutine frame with alloca and
// replaces calls to llvm.coro.resume and llvm.coro.destroy with direct calls
// to coroutine sub-functions.
//===----------------------------------------------------------------------===//

#include "CoroInternal.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "coro-elide"

//===----------------------------------------------------------------------===//
//                              Top Level Driver
//===----------------------------------------------------------------------===//

namespace {
struct CoroElide : FunctionPass {
  static char ID;
  CoroElide() : FunctionPass(ID) {}

  bool NeedsToRun = false;

  bool doInitialization(Module &M) override {
    NeedsToRun = coro::declaresIntrinsics(M, {"llvm.coro.begin"});
    return false;
  }

  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
};
}

// Go through the list of coro.subfn.addr intrinsics and replace them with the
// provided constant.
static void replaceWithConstant(Constant *Value,
                                SmallVectorImpl<CoroSubFnInst *> &Users) {
  if (Users.empty())
    return;

  // See if we need to bitcast the constant to match the type of the intrinsic
  // being replaced. Note: All coro.subfn.addr intrinsics return the same type,
  // so we only need to examine the type of the first one in the list.
  Type *IntrTy = Users.front()->getType();
  Type *ValueTy = Value->getType();
  if (ValueTy != IntrTy) {
    // May need to tweak the function type to match the type expected at the
    // use site.
    assert(ValueTy->isPointerTy() && IntrTy->isPointerTy());
    Value = ConstantExpr::getBitCast(Value, IntrTy);
  }

  // Now the value type matches the type of the intrinsic. Replace them all!
  for (CoroSubFnInst *I : Users)
    replaceAndRecursivelySimplify(I, Value);
}

// See if there are any coro.subfn.addr intrinsics directly referencing
// the coro.begin. If found, replace them with an appropriate coroutine
// subfunction associated with that coro.begin.
static bool replaceIndirectCalls(CoroBeginInst *CoroBegin) {
  SmallVector<CoroSubFnInst *, 8> ResumeAddr;
  SmallVector<CoroSubFnInst *, 8> DestroyAddr;

  for (User *U : CoroBegin->users()) {
    if (auto *II = dyn_cast<CoroSubFnInst>(U)) {
      switch (II->getIndex()) {
      case CoroSubFnInst::ResumeIndex:
        ResumeAddr.push_back(II);
        break;
      case CoroSubFnInst::DestroyIndex:
        DestroyAddr.push_back(II);
        break;
      default:
        llvm_unreachable("unexpected coro.subfn.addr constant");
      }
    }
  }
  if (ResumeAddr.empty() && DestroyAddr.empty())
    return false;

  // PostSplit coro.begin refers to an array of subfunctions in its Info
  // argument.
  ConstantArray *Resumers = CoroBegin->getInfo().Resumers;
  assert(Resumers && "PostSplit coro.begin Info argument must refer to an array"
                     "of coroutine subfunctions");
  auto *ResumeAddrConstant =
      ConstantExpr::getExtractValue(Resumers, CoroSubFnInst::ResumeIndex);
  auto *DestroyAddrConstant =
      ConstantExpr::getExtractValue(Resumers, CoroSubFnInst::DestroyIndex);

  replaceWithConstant(ResumeAddrConstant, ResumeAddr);
  replaceWithConstant(DestroyAddrConstant, DestroyAddr);
  return true;
}

// See if there are any coro.subfn.addr instructions referring to coro.devirt
// trigger, if so, replace them with a direct call to devirt trigger function.
static bool replaceDevirtTrigger(Function &F) {
  SmallVector<CoroSubFnInst *, 1> DevirtAddr;
  for (auto &I : instructions(F))
    if (auto *SubFn = dyn_cast<CoroSubFnInst>(&I))
      if (SubFn->getIndex() == CoroSubFnInst::RestartTrigger)
        DevirtAddr.push_back(SubFn);

  if (DevirtAddr.empty())
    return false;

  Module &M = *F.getParent();
  Function *DevirtFn = M.getFunction(CORO_DEVIRT_TRIGGER_FN);
  assert(DevirtFn && "coro.devirt.fn not found");
  replaceWithConstant(DevirtFn, DevirtAddr);

  return true;
}

bool CoroElide::runOnFunction(Function &F) {
  bool Changed = false;

  if (F.hasFnAttribute(CORO_PRESPLIT_ATTR))
    Changed = replaceDevirtTrigger(F);

  // Collect all PostSplit coro.begins.
  SmallVector<CoroBeginInst *, 4> CoroBegins;
  for (auto &I : instructions(F))
    if (auto *CB = dyn_cast<CoroBeginInst>(&I))
      if (CB->getInfo().isPostSplit())
        CoroBegins.push_back(CB);

  if (CoroBegins.empty())
    return Changed;

  for (auto *CB : CoroBegins)
    Changed |= replaceIndirectCalls(CB);

  return Changed;
}

char CoroElide::ID = 0;
INITIALIZE_PASS_BEGIN(
    CoroElide, "coro-elide",
    "Coroutine frame allocation elision and indirect calls replacement", false,
    false)
INITIALIZE_PASS_END(
    CoroElide, "coro-elide",
    "Coroutine frame allocation elision and indirect calls replacement", false,
    false)

Pass *llvm::createCoroElidePass() { return new CoroElide(); }
