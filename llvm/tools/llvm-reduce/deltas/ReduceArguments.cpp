//===- ReduceArguments.cpp - Specialized Delta Pass -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass in order
// to reduce uninteresting Arguments from declared and defined functions.
//
//===----------------------------------------------------------------------===//

#include "ReduceArguments.h"
#include "Utils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Operator.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <set>
#include <vector>

using namespace llvm;

static bool callingConvRequiresArgument(const Function &F,
                                        const Argument &Arg) {
  switch (F.getCallingConv()) {
  case CallingConv::X86_INTR:
    // If there are any arguments, the first one must by byval.
    return Arg.getArgNo() == 0 && F.arg_size() != 1;
  default:
    return false;
  }

  llvm_unreachable("covered calling conv switch");
}

/// Goes over OldF calls and replaces them with a call to NewF
static void replaceFunctionCalls(Function &OldF, Function &NewF,
                                 const std::set<int> &ArgIndexesToKeep) {
  LLVMContext &Ctx = OldF.getContext();

  const auto &Users = OldF.users();
  for (auto I = Users.begin(), E = Users.end(); I != E; )
    if (auto *CI = dyn_cast<CallInst>(*I++)) {
      // Skip uses in call instructions where OldF isn't the called function
      // (e.g. if OldF is an argument of the call).
      if (CI->getCalledFunction() != &OldF)
        continue;
      SmallVector<Value *, 8> Args;
      SmallVector<AttrBuilder, 8> ArgAttrs;

      for (auto ArgI = CI->arg_begin(), E = CI->arg_end(); ArgI != E; ++ArgI) {
        unsigned ArgIdx = ArgI - CI->arg_begin();
        if (ArgIndexesToKeep.count(ArgIdx)) {
          Args.push_back(*ArgI);
          ArgAttrs.emplace_back(Ctx, CI->getParamAttributes(ArgIdx));
        }
      }

      SmallVector<OperandBundleDef, 2> OpBundles;
      CI->getOperandBundlesAsDefs(OpBundles);

      CallInst *NewCI = CallInst::Create(&NewF, Args, OpBundles);
      NewCI->setCallingConv(CI->getCallingConv());

      AttrBuilder CallSiteAttrs(Ctx, CI->getAttributes().getFnAttrs());
      NewCI->setAttributes(
          AttributeList::get(Ctx, AttributeList::FunctionIndex, CallSiteAttrs));
      NewCI->addRetAttrs(AttrBuilder(Ctx, CI->getRetAttributes()));

      unsigned AttrIdx = 0;
      for (auto ArgI = NewCI->arg_begin(), E = NewCI->arg_end(); ArgI != E;
           ++ArgI, ++AttrIdx)
        NewCI->addParamAttrs(AttrIdx, ArgAttrs[AttrIdx]);

      if (auto *FPOp = dyn_cast<FPMathOperator>(NewCI))
        cast<Instruction>(FPOp)->setFastMathFlags(CI->getFastMathFlags());

      NewCI->copyMetadata(*CI);

      if (!CI->use_empty())
        CI->replaceAllUsesWith(NewCI);
      ReplaceInstWithInst(CI, NewCI);
    }
}

/// Returns whether or not this function should be considered a candidate for
/// argument removal. Currently, functions with no arguments and intrinsics are
/// not considered. Intrinsics aren't considered because their signatures are
/// fixed.
static bool shouldRemoveArguments(const Function &F) {
  return !F.arg_empty() && !F.isIntrinsic();
}

static bool allFuncUsersRewritable(const Function &F) {
  for (const Use &U : F.uses()) {
    const CallBase *CB = dyn_cast<CallBase>(U.getUser());
    if (!CB || !CB->isCallee(&U))
      continue;

    // TODO: Handle all CallBase cases.
    if (!isa<CallInst>(CB))
      return false;
  }

  return true;
}

/// Removes out-of-chunk arguments from functions, and modifies their calls
/// accordingly. It also removes allocations of out-of-chunk arguments.
void llvm::reduceArgumentsDeltaPass(Oracle &O, ReducerWorkItem &WorkItem) {
  Module &Program = WorkItem.getModule();
  std::vector<Argument *> InitArgsToKeep;
  std::vector<Function *> Funcs;

  // Get inside-chunk arguments, as well as their parent function
  for (auto &F : Program) {
    if (!shouldRemoveArguments(F))
      continue;
    if (!allFuncUsersRewritable(F))
      continue;
    Funcs.push_back(&F);
    for (auto &A : F.args()) {
      if (callingConvRequiresArgument(F, A) || O.shouldKeep())
        InitArgsToKeep.push_back(&A);
    }
  }

  // We create a vector first, then convert it to a set, so that we don't have
  // to pay the cost of rebalancing the set frequently if the order we insert
  // the elements doesn't match the order they should appear inside the set.
  std::set<Argument *> ArgsToKeep(InitArgsToKeep.begin(), InitArgsToKeep.end());

  for (auto *F : Funcs) {
    ValueToValueMapTy VMap;
    std::vector<WeakVH> InstToDelete;
    for (auto &A : F->args())
      if (!ArgsToKeep.count(&A)) {
        // By adding undesired arguments to the VMap, CloneFunction will remove
        // them from the resulting Function
        VMap[&A] = getDefaultValue(A.getType());
        for (auto *U : A.users())
          if (auto *I = dyn_cast<Instruction>(*&U))
            InstToDelete.push_back(I);
      }
    // Delete any (unique) instruction that uses the argument
    for (Value *V : InstToDelete) {
      if (!V)
        continue;
      auto *I = cast<Instruction>(V);
      I->replaceAllUsesWith(getDefaultValue(I->getType()));
      if (!I->isTerminator())
        I->eraseFromParent();
    }

    // No arguments to reduce
    if (VMap.empty())
      continue;

    std::set<int> ArgIndexesToKeep;
    for (const auto &[Index, Arg] : enumerate(F->args()))
      if (ArgsToKeep.count(&Arg))
        ArgIndexesToKeep.insert(Index);

    auto *ClonedFunc = CloneFunction(F, VMap);
    // In order to preserve function order, we move Clone after old Function
    ClonedFunc->takeName(F);
    ClonedFunc->removeFromParent();
    Program.getFunctionList().insertAfter(F->getIterator(), ClonedFunc);

    replaceFunctionCalls(*F, *ClonedFunc, ArgIndexesToKeep);
    F->replaceAllUsesWith(ClonedFunc);
    F->eraseFromParent();
  }
}
