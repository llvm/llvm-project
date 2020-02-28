//===- CoroElide.cpp - Coroutine Frame Allocation Elision Pass ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Coroutines/CoroElide.h"
#include "CoroInternal.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "coro-elide"

namespace {
// Created on demand if the coro-elide pass has work to do.
struct Lowerer : coro::LowererBase {
  SmallVector<CoroIdInst *, 4> CoroIds;
  SmallVector<CoroBeginInst *, 1> CoroBegins;
  SmallVector<CoroAllocInst *, 1> CoroAllocs;
  SmallVector<CoroSubFnInst *, 4> ResumeAddr;
  DenseMap<CoroBeginInst *, SmallVector<CoroSubFnInst *, 4>> DestroyAddr;
  SmallVector<CoroFreeInst *, 1> CoroFrees;
  CoroSuspendInst *CoroFinalSuspend;

  Lowerer(Module &M) : LowererBase(M) {}

  void elideHeapAllocations(Function *F, Type *FrameTy, AAResults &AA);
  bool shouldElide(Function *F, DominatorTree &DT) const;
  void collectPostSplitCoroIds(Function *F);
  bool processCoroId(CoroIdInst *, AAResults &AA, DominatorTree &DT);
};
} // end anonymous namespace

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

// See if any operand of the call instruction references the coroutine frame.
static bool operandReferences(CallInst *CI, AllocaInst *Frame, AAResults &AA) {
  for (Value *Op : CI->operand_values())
    if (AA.alias(Op, Frame) != NoAlias)
      return true;
  return false;
}

// Look for any tail calls referencing the coroutine frame and remove tail
// attribute from them, since now coroutine frame resides on the stack and tail
// call implies that the function does not references anything on the stack.
static void removeTailCallAttribute(AllocaInst *Frame, AAResults &AA) {
  Function &F = *Frame->getFunction();
  for (Instruction &I : instructions(F))
    if (auto *Call = dyn_cast<CallInst>(&I))
      if (Call->isTailCall() && operandReferences(Call, Frame, AA)) {
        // FIXME: If we ever hit this check. Evaluate whether it is more
        // appropriate to retain musttail and allow the code to compile.
        if (Call->isMustTailCall())
          report_fatal_error("Call referring to the coroutine frame cannot be "
                             "marked as musttail");
        Call->setTailCall(false);
      }
}

// Given a resume function @f.resume(%f.frame* %frame), returns %f.frame type.
static Type *getFrameType(Function *Resume) {
  auto *ArgType = Resume->arg_begin()->getType();
  return cast<PointerType>(ArgType)->getElementType();
}

// Finds first non alloca instruction in the entry block of a function.
static Instruction *getFirstNonAllocaInTheEntryBlock(Function *F) {
  for (Instruction &I : F->getEntryBlock())
    if (!isa<AllocaInst>(&I))
      return &I;
  llvm_unreachable("no terminator in the entry block");
}

// To elide heap allocations we need to suppress code blocks guarded by
// llvm.coro.alloc and llvm.coro.free instructions.
void Lowerer::elideHeapAllocations(Function *F, Type *FrameTy, AAResults &AA) {
  LLVMContext &C = FrameTy->getContext();
  auto *InsertPt =
      getFirstNonAllocaInTheEntryBlock(CoroIds.front()->getFunction());

  // Replacing llvm.coro.alloc with false will suppress dynamic
  // allocation as it is expected for the frontend to generate the code that
  // looks like:
  //   id = coro.id(...)
  //   mem = coro.alloc(id) ? malloc(coro.size()) : 0;
  //   coro.begin(id, mem)
  auto *False = ConstantInt::getFalse(C);
  for (auto *CA : CoroAllocs) {
    CA->replaceAllUsesWith(False);
    CA->eraseFromParent();
  }

  // FIXME: Design how to transmit alignment information for every alloca that
  // is spilled into the coroutine frame and recreate the alignment information
  // here. Possibly we will need to do a mini SROA here and break the coroutine
  // frame into individual AllocaInst recreating the original alignment.
  const DataLayout &DL = F->getParent()->getDataLayout();
  auto *Frame = new AllocaInst(FrameTy, DL.getAllocaAddrSpace(), "", InsertPt);
  auto *FrameVoidPtr =
      new BitCastInst(Frame, Type::getInt8PtrTy(C), "vFrame", InsertPt);

  for (auto *CB : CoroBegins) {
    CB->replaceAllUsesWith(FrameVoidPtr);
    CB->eraseFromParent();
  }

  // Since now coroutine frame lives on the stack we need to make sure that
  // any tail call referencing it, must be made non-tail call.
  removeTailCallAttribute(Frame, AA);
}

bool Lowerer::shouldElide(Function *F, DominatorTree &DT) const {
  // If no CoroAllocs, we cannot suppress allocation, so elision is not
  // possible.
  if (CoroAllocs.empty())
    return false;

  // Check that for every coro.begin there is at least one coro.destroy directly
  // referencing the SSA value of that coro.begin along each
  // non-exceptional path.
  // If the value escaped, then coro.destroy would have been referencing a
  // memory location storing that value and not the virtual register.

  SmallPtrSet<Instruction *, 8> Terminators;
  bool HasMultiPred = false;
  // First gather all of the non-exceptional terminators for the function.
  // Consider the final coro.suspend as the real terminator when the current
  // function is a coroutine.
  if (CoroFinalSuspend) {
    // If block of final coro.suspend has more than one predecessor,
    // then there is one resume path and the others are exceptional paths,
    // consider these predecessors as terminators.
    BasicBlock *FinalBB = CoroFinalSuspend->getParent();
    if (FinalBB->hasNPredecessorsOrMore(2)) {
      HasMultiPred = true;
      for (auto *B : predecessors(FinalBB))
        Terminators.insert(B->getTerminator());
    } else
      Terminators.insert(CoroFinalSuspend);
  } else {
    for (BasicBlock &B : *F) {
      auto *TI = B.getTerminator();
      if (TI->getNumSuccessors() == 0 && !TI->isExceptionalTerminator() &&
          !isa<UnreachableInst>(TI))
        Terminators.insert(TI);
    }
  }

  // Filter out the coro.destroy that lie along exceptional paths.
  SmallPtrSet<CoroSubFnInst *, 4> DAs;
  SmallPtrSet<Instruction *, 2> TIs;
  SmallPtrSet<CoroBeginInst *, 8> ReferencedCoroBegins;
  for (auto &It : DestroyAddr) {
    for (CoroSubFnInst *DA : It.second) {
      for (Instruction *TI : Terminators) {
        if (DT.dominates(DA, TI)) {
          if (HasMultiPred)
            TIs.insert(TI);
          else
            DAs.insert(DA);
          break;
        }
      }
    }
    // If all the predecessors dominate coro.destroys that reference same
    // coro.begin, record the coro.begin
    if (TIs.size() == Terminators.size()) {
      ReferencedCoroBegins.insert(It.first);
      TIs.clear();
    }
  }

  // Find all the coro.begin referenced by coro.destroy along happy paths.
  for (CoroSubFnInst *DA : DAs) {
    if (auto *CB = dyn_cast<CoroBeginInst>(DA->getFrame()))
      ReferencedCoroBegins.insert(CB);
    else
      return false;
  }

  // If size of the set is the same as total number of coro.begin, that means we
  // found a coro.free or coro.destroy referencing each coro.begin, so we can
  // perform heap elision.
  return ReferencedCoroBegins.size() == CoroBegins.size();
}

void Lowerer::collectPostSplitCoroIds(Function *F) {
  CoroIds.clear();
  CoroFinalSuspend = nullptr;
  for (auto &I : instructions(F)) {
    if (auto *CII = dyn_cast<CoroIdInst>(&I))
      if (CII->getInfo().isPostSplit())
        // If it is the coroutine itself, don't touch it.
        if (CII->getCoroutine() != CII->getFunction())
          CoroIds.push_back(CII);

    if (auto *CSI = dyn_cast<CoroSuspendInst>(&I))
      if (CSI->isFinal()) {
        if (!CoroFinalSuspend)
          CoroFinalSuspend = CSI;
        else
          report_fatal_error("Only one suspend point can be marked as final");
      }
  }
}

bool Lowerer::processCoroId(CoroIdInst *CoroId, AAResults &AA,
                            DominatorTree &DT) {
  CoroBegins.clear();
  CoroAllocs.clear();
  CoroFrees.clear();
  ResumeAddr.clear();
  DestroyAddr.clear();

  // Collect all coro.begin and coro.allocs associated with this coro.id.
  for (User *U : CoroId->users()) {
    if (auto *CB = dyn_cast<CoroBeginInst>(U))
      CoroBegins.push_back(CB);
    else if (auto *CA = dyn_cast<CoroAllocInst>(U))
      CoroAllocs.push_back(CA);
    else if (auto *CF = dyn_cast<CoroFreeInst>(U))
      CoroFrees.push_back(CF);
  }

  // Collect all coro.subfn.addrs associated with coro.begin.
  // Note, we only devirtualize the calls if their coro.subfn.addr refers to
  // coro.begin directly. If we run into cases where this check is too
  // conservative, we can consider relaxing the check.
  for (CoroBeginInst *CB : CoroBegins) {
    for (User *U : CB->users())
      if (auto *II = dyn_cast<CoroSubFnInst>(U))
        switch (II->getIndex()) {
        case CoroSubFnInst::ResumeIndex:
          ResumeAddr.push_back(II);
          break;
        case CoroSubFnInst::DestroyIndex:
          DestroyAddr[CB].push_back(II);
          break;
        default:
          llvm_unreachable("unexpected coro.subfn.addr constant");
        }
  }

  // PostSplit coro.id refers to an array of subfunctions in its Info
  // argument.
  ConstantArray *Resumers = CoroId->getInfo().Resumers;
  assert(Resumers && "PostSplit coro.id Info argument must refer to an array"
                     "of coroutine subfunctions");
  auto *ResumeAddrConstant =
      ConstantExpr::getExtractValue(Resumers, CoroSubFnInst::ResumeIndex);

  replaceWithConstant(ResumeAddrConstant, ResumeAddr);

  bool ShouldElide = shouldElide(CoroId->getFunction(), DT);

  auto *DestroyAddrConstant = ConstantExpr::getExtractValue(
      Resumers,
      ShouldElide ? CoroSubFnInst::CleanupIndex : CoroSubFnInst::DestroyIndex);

  for (auto &It : DestroyAddr)
    replaceWithConstant(DestroyAddrConstant, It.second);

  if (ShouldElide) {
    auto *FrameTy = getFrameType(cast<Function>(ResumeAddrConstant));
    elideHeapAllocations(CoroId->getFunction(), FrameTy, AA);
    coro::replaceCoroFree(CoroId, /*Elide=*/true);
  }

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

static bool declaresCoroElideIntrinsics(Module &M) {
  return coro::declaresIntrinsics(M, {"llvm.coro.id"});
}

PreservedAnalyses CoroElidePass::run(Function &F, FunctionAnalysisManager &AM) {
  auto &M = *F.getParent();
  if (!declaresCoroElideIntrinsics(M))
    return PreservedAnalyses::all();

  Lowerer L(M);
  L.CoroIds.clear();
  L.collectPostSplitCoroIds(&F);
  // If we did not find any coro.id, there is nothing to do.
  if (L.CoroIds.empty())
    return PreservedAnalyses::all();

  AAResults &AA = AM.getResult<AAManager>(F);
  DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);

  bool Changed = false;
  for (auto *CII : L.CoroIds)
    Changed |= L.processCoroId(CII, AA, DT);

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

namespace {
struct CoroElideLegacy : FunctionPass {
  static char ID;
  CoroElideLegacy() : FunctionPass(ID) {
    initializeCoroElideLegacyPass(*PassRegistry::getPassRegistry());
  }

  std::unique_ptr<Lowerer> L;

  bool doInitialization(Module &M) override {
    if (declaresCoroElideIntrinsics(M))
      L = std::make_unique<Lowerer>(M);
    return false;
  }

  bool runOnFunction(Function &F) override {
    if (!L)
      return false;

    bool Changed = false;

    if (F.hasFnAttribute(CORO_PRESPLIT_ATTR))
      Changed = replaceDevirtTrigger(F);

    L->CoroIds.clear();
    L->collectPostSplitCoroIds(&F);
    // If we did not find any coro.id, there is nothing to do.
    if (L->CoroIds.empty())
      return Changed;

    AAResults &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();

    for (auto *CII : L->CoroIds)
      Changed |= L->processCoroId(CII, AA, DT);

    return Changed;
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
  }
  StringRef getPassName() const override { return "Coroutine Elision"; }
};
}

char CoroElideLegacy::ID = 0;
INITIALIZE_PASS_BEGIN(
    CoroElideLegacy, "coro-elide",
    "Coroutine frame allocation elision and indirect calls replacement", false,
    false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(
    CoroElideLegacy, "coro-elide",
    "Coroutine frame allocation elision and indirect calls replacement", false,
    false)

Pass *llvm::createCoroElideLegacyPass() { return new CoroElideLegacy(); }
