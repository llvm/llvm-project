//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Try to reduce a function by inserting new return instructions. Try to insert
// an early return for each instruction value at that point. This requires
// mutating the return type, or finding instructions with a compatible type.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "llvm-reduce"

#include "ReduceValuesToReturn.h"

#include "Delta.h"
#include "Utils.h"
#include "llvm/IR/AttributeMask.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

/// Return true if it is legal to emit a copy of the function with a non-void
/// return type.
static bool canUseNonVoidReturnType(const Function &F) {
  // Functions with sret arguments must return void.
  return !F.hasStructRetAttr() &&
         CallingConv::supportsNonVoidReturnType(F.getCallingConv());
}

/// Return true if it's legal to replace a function return type to use \p Ty.
static bool isReallyValidReturnType(Type *Ty) {
  return FunctionType::isValidReturnType(Ty) && !Ty->isTokenTy() &&
         Ty->isFirstClassType();
}

/// Insert a ret inst after \p NewRetValue, which returns the value it produces.
static void rewriteFuncWithReturnType(Function &OldF, Value *NewRetValue) {
  Type *NewRetTy = NewRetValue->getType();
  FunctionType *OldFuncTy = OldF.getFunctionType();

  FunctionType *NewFuncTy =
      FunctionType::get(NewRetTy, OldFuncTy->params(), OldFuncTy->isVarArg());

  LLVMContext &Ctx = OldF.getContext();
  BasicBlock &EntryBB = OldF.getEntryBlock();
  Instruction *NewRetI = dyn_cast<Instruction>(NewRetValue);
  BasicBlock *NewRetBlock = NewRetI ? NewRetI->getParent() : &EntryBB;

  BasicBlock::iterator NewValIt =
      NewRetI ? NewRetI->getIterator() : EntryBB.end();

  Type *OldRetTy = OldFuncTy->getReturnType();

  // Hack up any return values in other blocks, we can't leave them as returning OldRetTy.
  if (OldRetTy != NewRetTy) {
    for (BasicBlock &OtherRetBB : OldF) {
      if (&OtherRetBB != NewRetBlock) {
        auto *OrigRI = dyn_cast<ReturnInst>(OtherRetBB.getTerminator());
        if (!OrigRI)
          continue;

        OrigRI->eraseFromParent();
        ReturnInst::Create(Ctx, getDefaultValue(NewRetTy), &OtherRetBB);
      }
    }
  }

  // Now prune any CFG edges we have to deal with.
  //
  // Use KeepOneInputPHIs in case the instruction we are using for the return is
  // that phi.
  // TODO: Could avoid this with fancier iterator management.
  for (BasicBlock *Succ : successors(NewRetBlock))
    Succ->removePredecessor(NewRetBlock, /*KeepOneInputPHIs=*/true);

  // Now delete the tail of this block, in reverse to delete uses before defs.
  for (Instruction &I : make_early_inc_range(
           make_range(NewRetBlock->rbegin(), NewValIt.getReverse()))) {
    Value *Replacement = getDefaultValue(I.getType());
    I.replaceAllUsesWith(Replacement);
    I.eraseFromParent();
  }

  ReturnInst::Create(Ctx, NewRetValue, NewRetBlock);

  // TODO: We may be eliminating blocks that were originally unreachable. We
  // probably ought to only be pruning blocks that became dead directly as a
  // result of our pruning here.
  EliminateUnreachableBlocks(OldF);

  // Drop the incompatible attributes before we copy over to the new function.
  if (OldRetTy != NewRetTy) {
    AttributeList AL = OldF.getAttributes();
    AttributeMask IncompatibleAttrs =
        AttributeFuncs::typeIncompatible(NewRetTy, AL.getRetAttrs());
    OldF.removeRetAttrs(IncompatibleAttrs);
  }

  // Now we need to remove any returned attributes from parameters.
  for (Argument &A : OldF.args())
    OldF.removeParamAttr(A.getArgNo(), Attribute::Returned);

  Function *NewF =
      Function::Create(NewFuncTy, OldF.getLinkage(), OldF.getAddressSpace(), "",
                       OldF.getParent());

  NewF->removeFromParent();
  OldF.getParent()->getFunctionList().insertAfter(OldF.getIterator(), NewF);
  NewF->takeName(&OldF);
  NewF->copyAttributesFrom(&OldF);

  // Adjust the callsite uses to the new return type. We pre-filtered cases
  // where the original call type was incorrectly non-void.
  for (User *U : make_early_inc_range(OldF.users())) {
    if (auto *CB = dyn_cast<CallBase>(U);
        CB && CB->getCalledOperand() == &OldF) {
      if (CB->getType()->isVoidTy()) {
        FunctionType *CallType = CB->getFunctionType();

        // The callsite may not match the new function type, in an undefined
        // behavior way. Only mutate the local return type.
        FunctionType *NewCallType = FunctionType::get(
            NewRetTy, CallType->params(), CallType->isVarArg());

        CB->mutateType(NewRetTy);
        CB->setCalledFunction(NewCallType, NewF);
      } else {
        assert(CB->getType() == NewRetTy &&
               "only handle exact return type match with non-void returns");
      }
    }
  }

  NewF->splice(NewF->begin(), &OldF);
  OldF.replaceAllUsesWith(NewF);

  // Preserve the parameters of OldF.
  for (auto Z : zip_first(OldF.args(), NewF->args())) {
    Argument &OldArg = std::get<0>(Z);
    Argument &NewArg = std::get<1>(Z);

    OldArg.replaceAllUsesWith(&NewArg);
    NewArg.takeName(&OldArg);
  }

  OldF.eraseFromParent();
}

// Check if all the callsites of the void function are void, or happen to
// incorrectly use the new return type.
//
// TODO: We could make better effort to handle call type mismatches.
static bool canReplaceFuncUsers(const Function &F, Type *NewRetTy) {
  for (const Use &U : F.uses()) {
    const CallBase *CB = dyn_cast<CallBase>(U.getUser());
    if (!CB)
      continue;

    // Normal pointer uses are trivially replacable.
    if (!CB->isCallee(&U))
      continue;

    // We can trivially replace the correct void call sites.
    if (CB->getType()->isVoidTy())
      continue;

    // We can trivially replace the call if the return type happened to match
    // the new return type.
    if (CB->getType() == NewRetTy)
      continue;

    // TODO: If all callsites have no uses, we could mutate the type of all the
    // callsites. This will complicate the visit and rewrite ordering though.
    LLVM_DEBUG(dbgs() << "Cannot replace used callsite with wrong type: " << *CB
                      << '\n');
    return false;
  }

  return true;
}

/// Return true if it's worthwhile replacing the non-void return value of \p BB
/// with \p Replacement
static bool shouldReplaceNonVoidReturnValue(const BasicBlock &BB,
                                            const Value *Replacement) {
  if (const auto *RI = dyn_cast<ReturnInst>(BB.getTerminator()))
    return RI->getReturnValue() != Replacement;
  return true;
}

static bool canHandleSuccessors(const BasicBlock &BB) {
  // TODO: Handle invoke and other exotic terminators
  if (!isa<ReturnInst, UnreachableInst, BranchInst, SwitchInst>(
          BB.getTerminator()))
    return false;

  for (const BasicBlock *Succ : successors(&BB)) {
    if (!Succ->canSplitPredecessors())
      return false;
  }

  return true;
}

static bool shouldForwardValueToReturn(const BasicBlock &BB, const Value *V,
                                       Type *RetTy) {
  if (!isReallyValidReturnType(V->getType()))
    return false;

  return (RetTy->isVoidTy() || shouldReplaceNonVoidReturnValue(BB, V)) &&
         canReplaceFuncUsers(*BB.getParent(), V->getType());
}

static bool tryForwardingInstructionsToReturn(
    Function &F, Oracle &O,
    std::vector<std::pair<Function *, Value *>> &FuncsToReplace) {

  // TODO: Should we try to expand returns to aggregate for function that
  // already have a return value?
  Type *RetTy = F.getReturnType();

  for (BasicBlock &BB : F) {
    if (!canHandleSuccessors(BB))
      continue;

    for (Instruction &I : BB) {
      if (shouldForwardValueToReturn(BB, &I, RetTy) && !O.shouldKeep()) {
        FuncsToReplace.emplace_back(&F, &I);
        return true;
      }
    }
  }

  return false;
}

static bool tryForwardingArgumentsToReturn(
    Function &F, Oracle &O,
    std::vector<std::pair<Function *, Value *>> &FuncsToReplace) {

  Type *RetTy = F.getReturnType();
  BasicBlock &EntryBB = F.getEntryBlock();

  for (Argument &A : F.args()) {
    if (shouldForwardValueToReturn(EntryBB, &A, RetTy) && !O.shouldKeep()) {
      FuncsToReplace.emplace_back(&F, &A);
      return true;
    }
  }

  return false;
}

void llvm::reduceArgumentsToReturnDeltaPass(Oracle &O,
                                            ReducerWorkItem &WorkItem) {
  Module &Program = WorkItem.getModule();

  // We're going to chaotically hack on the other users of the function in other
  // functions, so we need to collect a worklist of returns to replace.
  std::vector<std::pair<Function *, Value *>> FuncsToReplace;

  for (Function &F : Program.functions()) {
    if (!F.isDeclaration() && canUseNonVoidReturnType(F))
      tryForwardingArgumentsToReturn(F, O, FuncsToReplace);
  }

  for (auto [F, NewRetVal] : FuncsToReplace)
    rewriteFuncWithReturnType(*F, NewRetVal);
}

void llvm::reduceInstructionsToReturnDeltaPass(Oracle &O,
                                               ReducerWorkItem &WorkItem) {
  Module &Program = WorkItem.getModule();

  // We're going to chaotically hack on the other users of the function in other
  // functions, so we need to collect a worklist of returns to replace.
  std::vector<std::pair<Function *, Value *>> FuncsToReplace;

  for (Function &F : Program.functions()) {
    if (!F.isDeclaration() && canUseNonVoidReturnType(F))
      tryForwardingInstructionsToReturn(F, O, FuncsToReplace);
  }

  for (auto [F, NewRetVal] : FuncsToReplace)
    rewriteFuncWithReturnType(*F, NewRetVal);
}
