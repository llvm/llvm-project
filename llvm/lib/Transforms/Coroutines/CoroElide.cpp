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
#include "llvm/Support/ErrorHandling.h"

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
    AU.addRequired<AAResultsWrapperPass>();
    AU.setPreservesCFG();
  }
};
}

char CoroElide::ID = 0;
INITIALIZE_PASS_BEGIN(
    CoroElide, "coro-elide",
    "Coroutine frame allocation elision and indirect calls replacement", false,
    false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(
    CoroElide, "coro-elide",
    "Coroutine frame allocation elision and indirect calls replacement", false,
    false)

Pass *llvm::createCoroElidePass() { return new CoroElide(); }

//===----------------------------------------------------------------------===//
//                              Implementation
//===----------------------------------------------------------------------===//

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
  MemoryLocation Mem(Frame);
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
  auto *ArgType = Resume->getArgumentList().front().getType();
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
static void elideHeapAllocations(CoroBeginInst *CoroBegin, Type *FrameTy,
                                 CoroAllocInst *AllocInst, AAResults &AA) {
  LLVMContext &C = CoroBegin->getContext();
  auto *InsertPt = getFirstNonAllocaInTheEntryBlock(CoroBegin->getFunction());

  // FIXME: Design how to transmit alignment information for every alloca that
  // is spilled into the coroutine frame and recreate the alignment information
  // here. Possibly we will need to do a mini SROA here and break the coroutine
  // frame into individual AllocaInst recreating the original alignment.
  auto *Frame = new AllocaInst(FrameTy, "", InsertPt);
  auto *FrameVoidPtr =
      new BitCastInst(Frame, Type::getInt8PtrTy(C), "vFrame", InsertPt);

  // Replacing llvm.coro.alloc with non-null value will suppress dynamic
  // allocation as it is expected for the frontend to generate the code that
  // looks like:
  //   mem = coro.alloc();
  //   if (!mem) mem = malloc(coro.size());
  //   coro.begin(mem, ...)
  AllocInst->replaceAllUsesWith(FrameVoidPtr);
  AllocInst->eraseFromParent();

  // To suppress deallocation code, we replace all llvm.coro.free intrinsics
  // associated with this coro.begin with null constant.
  auto *NullPtr = ConstantPointerNull::get(Type::getInt8PtrTy(C));
  coro::replaceAllCoroFrees(CoroBegin, NullPtr);
  CoroBegin->lowerTo(FrameVoidPtr);

  // Since now coroutine frame lives on the stack we need to make sure that
  // any tail call referencing it, must be made non-tail call.
  removeTailCallAttribute(Frame, AA);
}

// See if there are any coro.subfn.addr intrinsics directly referencing
// the coro.begin. If found, replace them with an appropriate coroutine
// subfunction associated with that coro.begin.
static bool replaceIndirectCalls(CoroBeginInst *CoroBegin, AAResults &AA) {
  SmallVector<CoroSubFnInst *, 8> ResumeAddr;
  SmallVector<CoroSubFnInst *, 8> DestroyAddr;

  for (User *CF : CoroBegin->users()) {
    assert(isa<CoroFrameInst>(CF) &&
           "CoroBegin can be only used by coro.frame instructions");
    for (User *U : CF->users()) {
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
  replaceWithConstant(ResumeAddrConstant, ResumeAddr);

  if (DestroyAddr.empty())
    return true;

  auto *DestroyAddrConstant =
      ConstantExpr::getExtractValue(Resumers, CoroSubFnInst::DestroyIndex);
  replaceWithConstant(DestroyAddrConstant, DestroyAddr);

  // If llvm.coro.begin refers to llvm.coro.alloc, we can elide the allocation.
  if (auto *AllocInst = CoroBegin->getAlloc()) {
    // FIXME: The check above is overly lax. It only checks for whether we have
    // an ability to elide heap allocations, not whether it is safe to do so.
    // We need to do something like:
    // If for every exit from the function where coro.begin is
    // live, there is a coro.free or coro.destroy dominating that exit block,
    // then it is safe to elide heap allocation, since the lifetime of coroutine
    // is fully enclosed in its caller.
    auto *FrameTy = getFrameType(cast<Function>(ResumeAddrConstant));
    elideHeapAllocations(CoroBegin, FrameTy, AllocInst, AA);
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

  AAResults &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
  for (auto *CB : CoroBegins)
    Changed |= replaceIndirectCalls(CB, AA);

  return Changed;
}
