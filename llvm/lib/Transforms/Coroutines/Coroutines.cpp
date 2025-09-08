//===- Coroutines.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the common infrastructure for Coroutine Passes.
//
//===----------------------------------------------------------------------===//

#include "CoroInternal.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Coroutines/ABI.h"
#include "llvm/Transforms/Coroutines/CoroInstr.h"
#include "llvm/Transforms/Coroutines/CoroShape.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cassert>
#include <cstddef>
#include <utility>

using namespace llvm;

// Construct the lowerer base class and initialize its members.
coro::LowererBase::LowererBase(Module &M)
    : TheModule(M), Context(M.getContext()),
      Int8Ptr(PointerType::get(Context, 0)),
      ResumeFnType(FunctionType::get(Type::getVoidTy(Context), Int8Ptr,
                                     /*isVarArg=*/false)),
      NullPtr(ConstantPointerNull::get(Int8Ptr)) {}

// Creates a call to llvm.coro.subfn.addr to obtain a resume function address.
// It generates the following:
//
//    call ptr @llvm.coro.subfn.addr(ptr %Arg, i8 %index)

CallInst *coro::LowererBase::makeSubFnCall(Value *Arg, int Index,
                                           Instruction *InsertPt) {
  auto *IndexVal = ConstantInt::get(Type::getInt8Ty(Context), Index);
  auto *Fn =
      Intrinsic::getOrInsertDeclaration(&TheModule, Intrinsic::coro_subfn_addr);

  assert(Index >= CoroSubFnInst::IndexFirst &&
         Index < CoroSubFnInst::IndexLast &&
         "makeSubFnCall: Index value out of range");
  return CallInst::Create(Fn, {Arg, IndexVal}, "", InsertPt->getIterator());
}

// We can only efficiently check for non-overloaded intrinsics.
// The following intrinsics are absent for that reason:
// coro_align, coro_size, coro_suspend_async, coro_suspend_retcon
static Intrinsic::ID NonOverloadedCoroIntrinsics[] = {
    Intrinsic::coro_alloc,
    Intrinsic::coro_async_context_alloc,
    Intrinsic::coro_async_context_dealloc,
    Intrinsic::coro_async_resume,
    Intrinsic::coro_async_size_replace,
    Intrinsic::coro_await_suspend_bool,
    Intrinsic::coro_await_suspend_handle,
    Intrinsic::coro_await_suspend_void,
    Intrinsic::coro_begin,
    Intrinsic::coro_begin_custom_abi,
    Intrinsic::coro_destroy,
    Intrinsic::coro_done,
    Intrinsic::coro_end,
    Intrinsic::coro_end_async,
    Intrinsic::coro_frame,
    Intrinsic::coro_free,
    Intrinsic::coro_id,
    Intrinsic::coro_id_async,
    Intrinsic::coro_id_retcon,
    Intrinsic::coro_id_retcon_once,
    Intrinsic::coro_noop,
    Intrinsic::coro_prepare_async,
    Intrinsic::coro_prepare_retcon,
    Intrinsic::coro_promise,
    Intrinsic::coro_resume,
    Intrinsic::coro_save,
    Intrinsic::coro_subfn_addr,
    Intrinsic::coro_suspend,
};

bool coro::isSuspendBlock(BasicBlock *BB) {
  return isa<AnyCoroSuspendInst>(BB->front());
}

bool coro::declaresAnyIntrinsic(const Module &M) {
  return declaresIntrinsics(M, NonOverloadedCoroIntrinsics);
}

// Checks whether the module declares any of the listed intrinsics.
bool coro::declaresIntrinsics(const Module &M, ArrayRef<Intrinsic::ID> List) {
#ifndef NDEBUG
  for (Intrinsic::ID ID : List)
    assert(!Intrinsic::isOverloaded(ID) &&
           "Only non-overloaded intrinsics supported");
#endif

  for (Intrinsic::ID ID : List)
    if (Intrinsic::getDeclarationIfExists(&M, ID))
      return true;
  return false;
}

// Replace all coro.frees associated with the provided CoroId either with 'null'
// if Elide is true and with its frame parameter otherwise.
void coro::replaceCoroFree(CoroIdInst *CoroId, bool Elide) {
  SmallVector<CoroFreeInst *, 4> CoroFrees;
  for (User *U : CoroId->users())
    if (auto CF = dyn_cast<CoroFreeInst>(U))
      CoroFrees.push_back(CF);

  if (CoroFrees.empty())
    return;

  Value *Replacement =
      Elide
          ? ConstantPointerNull::get(PointerType::get(CoroId->getContext(), 0))
          : CoroFrees.front()->getFrame();

  for (CoroFreeInst *CF : CoroFrees) {
    CF->replaceAllUsesWith(Replacement);
    CF->eraseFromParent();
  }
}

void coro::suppressCoroAllocs(CoroIdInst *CoroId) {
  SmallVector<CoroAllocInst *, 4> CoroAllocs;
  for (User *U : CoroId->users())
    if (auto *CA = dyn_cast<CoroAllocInst>(U))
      CoroAllocs.push_back(CA);

  if (CoroAllocs.empty())
    return;

  coro::suppressCoroAllocs(CoroId->getContext(), CoroAllocs);
}

// Replacing llvm.coro.alloc with false will suppress dynamic
// allocation as it is expected for the frontend to generate the code that
// looks like:
//   id = coro.id(...)
//   mem = coro.alloc(id) ? malloc(coro.size()) : 0;
//   coro.begin(id, mem)
void coro::suppressCoroAllocs(LLVMContext &Context,
                              ArrayRef<CoroAllocInst *> CoroAllocs) {
  auto *False = ConstantInt::getFalse(Context);
  for (auto *CA : CoroAllocs) {
    CA->replaceAllUsesWith(False);
    CA->eraseFromParent();
  }
}

static CoroSaveInst *createCoroSave(CoroBeginInst *CoroBegin,
                                    CoroSuspendInst *SuspendInst) {
  Module *M = SuspendInst->getModule();
  auto *Fn = Intrinsic::getOrInsertDeclaration(M, Intrinsic::coro_save);
  auto *SaveInst = cast<CoroSaveInst>(
      CallInst::Create(Fn, CoroBegin, "", SuspendInst->getIterator()));
  assert(!SuspendInst->getCoroSave());
  SuspendInst->setArgOperand(0, SaveInst);
  return SaveInst;
}

// Collect "interesting" coroutine intrinsics.
void coro::Shape::analyze(Function &F,
                          SmallVectorImpl<CoroFrameInst *> &CoroFrames,
                          SmallVectorImpl<CoroSaveInst *> &UnusedCoroSaves,
                          CoroPromiseInst *&CoroPromise) {
  clear();

  bool HasFinalSuspend = false;
  bool HasUnwindCoroEnd = false;
  size_t FinalSuspendIndex = 0;

  for (Instruction &I : instructions(F)) {
    // FIXME: coro_await_suspend_* are not proper `IntrinisicInst`s
    // because they might be invoked
    if (auto AWS = dyn_cast<CoroAwaitSuspendInst>(&I)) {
      CoroAwaitSuspends.push_back(AWS);
    } else if (auto II = dyn_cast<IntrinsicInst>(&I)) {
      switch (II->getIntrinsicID()) {
      default:
        continue;
      case Intrinsic::coro_size:
        CoroSizes.push_back(cast<CoroSizeInst>(II));
        break;
      case Intrinsic::coro_align:
        CoroAligns.push_back(cast<CoroAlignInst>(II));
        break;
      case Intrinsic::coro_frame:
        CoroFrames.push_back(cast<CoroFrameInst>(II));
        break;
      case Intrinsic::coro_save:
        // After optimizations, coro_suspends using this coro_save might have
        // been removed, remember orphaned coro_saves to remove them later.
        if (II->use_empty())
          UnusedCoroSaves.push_back(cast<CoroSaveInst>(II));
        break;
      case Intrinsic::coro_suspend_async: {
        auto *Suspend = cast<CoroSuspendAsyncInst>(II);
        Suspend->checkWellFormed();
        CoroSuspends.push_back(Suspend);
        break;
      }
      case Intrinsic::coro_suspend_retcon: {
        auto Suspend = cast<CoroSuspendRetconInst>(II);
        CoroSuspends.push_back(Suspend);
        break;
      }
      case Intrinsic::coro_suspend: {
        auto Suspend = cast<CoroSuspendInst>(II);
        CoroSuspends.push_back(Suspend);
        if (Suspend->isFinal()) {
          if (HasFinalSuspend)
            report_fatal_error(
              "Only one suspend point can be marked as final");
          HasFinalSuspend = true;
          FinalSuspendIndex = CoroSuspends.size() - 1;
        }
        break;
      }
      case Intrinsic::coro_begin:
      case Intrinsic::coro_begin_custom_abi: {
        auto CB = cast<CoroBeginInst>(II);

        // Ignore coro id's that aren't pre-split.
        auto Id = dyn_cast<CoroIdInst>(CB->getId());
        if (Id && !Id->getInfo().isPreSplit())
          break;

        if (CoroBegin)
          report_fatal_error(
                "coroutine should have exactly one defining @llvm.coro.begin");
        CB->addRetAttr(Attribute::NonNull);
        CB->addRetAttr(Attribute::NoAlias);
        CB->removeFnAttr(Attribute::NoDuplicate);
        CoroBegin = CB;
        break;
      }
      case Intrinsic::coro_end_async:
      case Intrinsic::coro_end:
        CoroEnds.push_back(cast<AnyCoroEndInst>(II));
        if (auto *AsyncEnd = dyn_cast<CoroAsyncEndInst>(II)) {
          AsyncEnd->checkWellFormed();
        }

        if (CoroEnds.back()->isUnwind())
          HasUnwindCoroEnd = true;

        if (CoroEnds.back()->isFallthrough() && isa<CoroEndInst>(II)) {
          // Make sure that the fallthrough coro.end is the first element in the
          // CoroEnds vector.
          // Note: I don't think this is neccessary anymore.
          if (CoroEnds.size() > 1) {
            if (CoroEnds.front()->isFallthrough())
              report_fatal_error(
                  "Only one coro.end can be marked as fallthrough");
            std::swap(CoroEnds.front(), CoroEnds.back());
          }
        }
        break;
      case Intrinsic::coro_promise:
        assert(CoroPromise == nullptr &&
               "CoroEarly must ensure coro.promise unique");
        CoroPromise = cast<CoroPromiseInst>(II);
        break;
      }
    }
  }

  // If there is no CoroBegin then this is not a coroutine.
  if (!CoroBegin)
    return;

  // Determination of ABI and initializing lowering info
  auto Id = CoroBegin->getId();
  switch (auto IntrID = Id->getIntrinsicID()) {
  case Intrinsic::coro_id: {
    ABI = coro::ABI::Switch;
    SwitchLowering.HasFinalSuspend = HasFinalSuspend;
    SwitchLowering.HasUnwindCoroEnd = HasUnwindCoroEnd;

    auto SwitchId = getSwitchCoroId();
    SwitchLowering.ResumeSwitch = nullptr;
    SwitchLowering.PromiseAlloca = SwitchId->getPromise();
    SwitchLowering.ResumeEntryBlock = nullptr;

    // Move final suspend to the last element in the CoroSuspends vector.
    if (SwitchLowering.HasFinalSuspend &&
        FinalSuspendIndex != CoroSuspends.size() - 1)
      std::swap(CoroSuspends[FinalSuspendIndex], CoroSuspends.back());
    break;
  }
  case Intrinsic::coro_id_async: {
    ABI = coro::ABI::Async;
    auto *AsyncId = getAsyncCoroId();
    AsyncId->checkWellFormed();
    AsyncLowering.Context = AsyncId->getStorage();
    AsyncLowering.ContextArgNo = AsyncId->getStorageArgumentIndex();
    AsyncLowering.ContextHeaderSize = AsyncId->getStorageSize();
    AsyncLowering.ContextAlignment = AsyncId->getStorageAlignment().value();
    AsyncLowering.AsyncFuncPointer = AsyncId->getAsyncFunctionPointer();
    AsyncLowering.AsyncCC = F.getCallingConv();
    break;
  }
  case Intrinsic::coro_id_retcon:
  case Intrinsic::coro_id_retcon_once: {
    ABI = IntrID == Intrinsic::coro_id_retcon ? coro::ABI::Retcon
                                              : coro::ABI::RetconOnce;
    auto ContinuationId = getRetconCoroId();
    ContinuationId->checkWellFormed();
    auto Prototype = ContinuationId->getPrototype();
    RetconLowering.ResumePrototype = Prototype;
    RetconLowering.Alloc = ContinuationId->getAllocFunction();
    RetconLowering.Dealloc = ContinuationId->getDeallocFunction();
    RetconLowering.ReturnBlock = nullptr;
    RetconLowering.IsFrameInlineInStorage = false;
    break;
  }
  default:
    llvm_unreachable("coro.begin is not dependent on a coro.id call");
  }
}

// If for some reason, we were not able to find coro.begin, bailout.
void coro::Shape::invalidateCoroutine(
    Function &F, SmallVectorImpl<CoroFrameInst *> &CoroFrames) {
  assert(!CoroBegin);
  {
    // Replace coro.frame which are supposed to be lowered to the result of
    // coro.begin with poison.
    auto *Poison = PoisonValue::get(PointerType::get(F.getContext(), 0));
    for (CoroFrameInst *CF : CoroFrames) {
      CF->replaceAllUsesWith(Poison);
      CF->eraseFromParent();
    }
    CoroFrames.clear();

    // Replace all coro.suspend with poison and remove related coro.saves if
    // present.
    for (AnyCoroSuspendInst *CS : CoroSuspends) {
      CS->replaceAllUsesWith(PoisonValue::get(CS->getType()));
      CS->eraseFromParent();
      if (auto *CoroSave = CS->getCoroSave())
        CoroSave->eraseFromParent();
    }
    CoroSuspends.clear();

    // Replace all coro.ends with unreachable instruction.
    for (AnyCoroEndInst *CE : CoroEnds)
      changeToUnreachable(CE);
  }
}

void coro::SwitchABI::init() {
  assert(Shape.ABI == coro::ABI::Switch);
  {
    for (auto *AnySuspend : Shape.CoroSuspends) {
      auto Suspend = dyn_cast<CoroSuspendInst>(AnySuspend);
      if (!Suspend) {
#ifndef NDEBUG
        AnySuspend->dump();
#endif
        report_fatal_error("coro.id must be paired with coro.suspend");
      }

      if (!Suspend->getCoroSave())
        createCoroSave(Shape.CoroBegin, Suspend);
    }
  }
}

void coro::AsyncABI::init() { assert(Shape.ABI == coro::ABI::Async); }

void coro::AnyRetconABI::init() {
  assert(Shape.ABI == coro::ABI::Retcon || Shape.ABI == coro::ABI::RetconOnce);
  {
    // Determine the result value types, and make sure they match up with
    // the values passed to the suspends.
    auto ResultTys = Shape.getRetconResultTypes();
    auto ResumeTys = Shape.getRetconResumeTypes();

    for (auto *AnySuspend : Shape.CoroSuspends) {
      auto Suspend = dyn_cast<CoroSuspendRetconInst>(AnySuspend);
      if (!Suspend) {
#ifndef NDEBUG
        AnySuspend->dump();
#endif
        report_fatal_error("coro.id.retcon.* must be paired with "
                           "coro.suspend.retcon");
      }

      // Check that the argument types of the suspend match the results.
      auto SI = Suspend->value_begin(), SE = Suspend->value_end();
      auto RI = ResultTys.begin(), RE = ResultTys.end();
      for (; SI != SE && RI != RE; ++SI, ++RI) {
        auto SrcTy = (*SI)->getType();
        if (SrcTy != *RI) {
          // The optimizer likes to eliminate bitcasts leading into variadic
          // calls, but that messes with our invariants.  Re-insert the
          // bitcast and ignore this type mismatch.
          if (CastInst::isBitCastable(SrcTy, *RI)) {
            auto BCI = new BitCastInst(*SI, *RI, "", Suspend->getIterator());
            SI->set(BCI);
            continue;
          }

#ifndef NDEBUG
          Suspend->dump();
          Shape.RetconLowering.ResumePrototype->getFunctionType()->dump();
#endif
          report_fatal_error("argument to coro.suspend.retcon does not "
                             "match corresponding prototype function result");
        }
      }
      if (SI != SE || RI != RE) {
#ifndef NDEBUG
        Suspend->dump();
        Shape.RetconLowering.ResumePrototype->getFunctionType()->dump();
#endif
        report_fatal_error("wrong number of arguments to coro.suspend.retcon");
      }

      // Check that the result type of the suspend matches the resume types.
      Type *SResultTy = Suspend->getType();
      ArrayRef<Type *> SuspendResultTys;
      if (SResultTy->isVoidTy()) {
        // leave as empty array
      } else if (auto SResultStructTy = dyn_cast<StructType>(SResultTy)) {
        SuspendResultTys = SResultStructTy->elements();
      } else {
        // forms an ArrayRef using SResultTy, be careful
        SuspendResultTys = SResultTy;
      }
      if (SuspendResultTys.size() != ResumeTys.size()) {
#ifndef NDEBUG
        Suspend->dump();
        Shape.RetconLowering.ResumePrototype->getFunctionType()->dump();
#endif
        report_fatal_error("wrong number of results from coro.suspend.retcon");
      }
      for (size_t I = 0, E = ResumeTys.size(); I != E; ++I) {
        if (SuspendResultTys[I] != ResumeTys[I]) {
#ifndef NDEBUG
          Suspend->dump();
          Shape.RetconLowering.ResumePrototype->getFunctionType()->dump();
#endif
          report_fatal_error("result from coro.suspend.retcon does not "
                             "match corresponding prototype function param");
        }
      }
    }
  }
}

void coro::Shape::cleanCoroutine(
    SmallVectorImpl<CoroFrameInst *> &CoroFrames,
    SmallVectorImpl<CoroSaveInst *> &UnusedCoroSaves, CoroPromiseInst *PI) {
  // The coro.frame intrinsic is always lowered to the result of coro.begin.
  for (CoroFrameInst *CF : CoroFrames) {
    CF->replaceAllUsesWith(CoroBegin);
    CF->eraseFromParent();
  }
  CoroFrames.clear();

  // Remove orphaned coro.saves.
  for (CoroSaveInst *CoroSave : UnusedCoroSaves)
    CoroSave->eraseFromParent();
  UnusedCoroSaves.clear();

  if (PI) {
    PI->replaceAllUsesWith(PI->isFromPromise()
                               ? cast<Value>(CoroBegin)
                               : cast<Value>(getPromiseAlloca()));
    PI->eraseFromParent();
  }
}

static void propagateCallAttrsFromCallee(CallInst *Call, Function *Callee) {
  Call->setCallingConv(Callee->getCallingConv());
  // TODO: attributes?
}

static void addCallToCallGraph(CallGraph *CG, CallInst *Call, Function *Callee){
  if (CG)
    (*CG)[Call->getFunction()]->addCalledFunction(Call, (*CG)[Callee]);
}

Value *coro::Shape::emitAlloc(IRBuilder<> &Builder, Value *Size,
                              CallGraph *CG) const {
  switch (ABI) {
  case coro::ABI::Switch:
    llvm_unreachable("can't allocate memory in coro switch-lowering");

  case coro::ABI::Retcon:
  case coro::ABI::RetconOnce: {
    auto Alloc = RetconLowering.Alloc;
    Size = Builder.CreateIntCast(Size,
                                 Alloc->getFunctionType()->getParamType(0),
                                 /*is signed*/ false);
    auto *Call = Builder.CreateCall(Alloc, Size);
    propagateCallAttrsFromCallee(Call, Alloc);
    addCallToCallGraph(CG, Call, Alloc);
    return Call;
  }
  case coro::ABI::Async:
    llvm_unreachable("can't allocate memory in coro async-lowering");
  }
  llvm_unreachable("Unknown coro::ABI enum");
}

void coro::Shape::emitDealloc(IRBuilder<> &Builder, Value *Ptr,
                              CallGraph *CG) const {
  switch (ABI) {
  case coro::ABI::Switch:
    llvm_unreachable("can't allocate memory in coro switch-lowering");

  case coro::ABI::Retcon:
  case coro::ABI::RetconOnce: {
    auto Dealloc = RetconLowering.Dealloc;
    Ptr = Builder.CreateBitCast(Ptr,
                                Dealloc->getFunctionType()->getParamType(0));
    auto *Call = Builder.CreateCall(Dealloc, Ptr);
    propagateCallAttrsFromCallee(Call, Dealloc);
    addCallToCallGraph(CG, Call, Dealloc);
    return;
  }
  case coro::ABI::Async:
    llvm_unreachable("can't allocate memory in coro async-lowering");
  }
  llvm_unreachable("Unknown coro::ABI enum");
}

[[noreturn]] static void fail(const Instruction *I, const char *Reason,
                              Value *V) {
#ifndef NDEBUG
  I->dump();
  if (V) {
    errs() << "  Value: ";
    V->printAsOperand(llvm::errs());
    errs() << '\n';
  }
#endif
  report_fatal_error(Reason);
}

/// Check that the given value is a well-formed prototype for the
/// llvm.coro.id.retcon.* intrinsics.
static void checkWFRetconPrototype(const AnyCoroIdRetconInst *I, Value *V) {
  auto F = dyn_cast<Function>(V->stripPointerCasts());
  if (!F)
    fail(I, "llvm.coro.id.retcon.* prototype not a Function", V);

  auto FT = F->getFunctionType();

  if (isa<CoroIdRetconInst>(I)) {
    bool ResultOkay;
    if (FT->getReturnType()->isPointerTy()) {
      ResultOkay = true;
    } else if (auto SRetTy = dyn_cast<StructType>(FT->getReturnType())) {
      ResultOkay = (!SRetTy->isOpaque() &&
                    SRetTy->getNumElements() > 0 &&
                    SRetTy->getElementType(0)->isPointerTy());
    } else {
      ResultOkay = false;
    }
    if (!ResultOkay)
      fail(I, "llvm.coro.id.retcon prototype must return pointer as first "
              "result", F);

    if (FT->getReturnType() !=
          I->getFunction()->getFunctionType()->getReturnType())
      fail(I, "llvm.coro.id.retcon prototype return type must be same as"
              "current function return type", F);
  } else {
    // No meaningful validation to do here for llvm.coro.id.unique.once.
  }

  if (FT->getNumParams() == 0 || !FT->getParamType(0)->isPointerTy())
    fail(I, "llvm.coro.id.retcon.* prototype must take pointer as "
            "its first parameter", F);
}

/// Check that the given value is a well-formed allocator.
static void checkWFAlloc(const Instruction *I, Value *V) {
  auto F = dyn_cast<Function>(V->stripPointerCasts());
  if (!F)
    fail(I, "llvm.coro.* allocator not a Function", V);

  auto FT = F->getFunctionType();
  if (!FT->getReturnType()->isPointerTy())
    fail(I, "llvm.coro.* allocator must return a pointer", F);

  if (FT->getNumParams() != 1 ||
      !FT->getParamType(0)->isIntegerTy())
    fail(I, "llvm.coro.* allocator must take integer as only param", F);
}

/// Check that the given value is a well-formed deallocator.
static void checkWFDealloc(const Instruction *I, Value *V) {
  auto F = dyn_cast<Function>(V->stripPointerCasts());
  if (!F)
    fail(I, "llvm.coro.* deallocator not a Function", V);

  auto FT = F->getFunctionType();
  if (!FT->getReturnType()->isVoidTy())
    fail(I, "llvm.coro.* deallocator must return void", F);

  if (FT->getNumParams() != 1 ||
      !FT->getParamType(0)->isPointerTy())
    fail(I, "llvm.coro.* deallocator must take pointer as only param", F);
}

static void checkConstantInt(const Instruction *I, Value *V,
                             const char *Reason) {
  if (!isa<ConstantInt>(V)) {
    fail(I, Reason, V);
  }
}

void AnyCoroIdRetconInst::checkWellFormed() const {
  checkConstantInt(this, getArgOperand(SizeArg),
                   "size argument to coro.id.retcon.* must be constant");
  checkConstantInt(this, getArgOperand(AlignArg),
                   "alignment argument to coro.id.retcon.* must be constant");
  checkWFRetconPrototype(this, getArgOperand(PrototypeArg));
  checkWFAlloc(this, getArgOperand(AllocArg));
  checkWFDealloc(this, getArgOperand(DeallocArg));
}

static void checkAsyncFuncPointer(const Instruction *I, Value *V) {
  auto *AsyncFuncPtrAddr = dyn_cast<GlobalVariable>(V->stripPointerCasts());
  if (!AsyncFuncPtrAddr)
    fail(I, "llvm.coro.id.async async function pointer not a global", V);
}

void CoroIdAsyncInst::checkWellFormed() const {
  checkConstantInt(this, getArgOperand(SizeArg),
                   "size argument to coro.id.async must be constant");
  checkConstantInt(this, getArgOperand(AlignArg),
                   "alignment argument to coro.id.async must be constant");
  checkConstantInt(this, getArgOperand(StorageArg),
                   "storage argument offset to coro.id.async must be constant");
  checkAsyncFuncPointer(this, getArgOperand(AsyncFuncPtrArg));
}

static void checkAsyncContextProjectFunction(const Instruction *I,
                                             Function *F) {
  auto *FunTy = cast<FunctionType>(F->getValueType());
  if (!FunTy->getReturnType()->isPointerTy())
    fail(I,
         "llvm.coro.suspend.async resume function projection function must "
         "return a ptr type",
         F);
  if (FunTy->getNumParams() != 1 || !FunTy->getParamType(0)->isPointerTy())
    fail(I,
         "llvm.coro.suspend.async resume function projection function must "
         "take one ptr type as parameter",
         F);
}

void CoroSuspendAsyncInst::checkWellFormed() const {
  checkAsyncContextProjectFunction(this, getAsyncContextProjectionFunction());
}

void CoroAsyncEndInst::checkWellFormed() const {
  auto *MustTailCallFunc = getMustTailCallFunction();
  if (!MustTailCallFunc)
    return;
  auto *FnTy = MustTailCallFunc->getFunctionType();
  if (FnTy->getNumParams() != (arg_size() - 3))
    fail(this,
         "llvm.coro.end.async must tail call function argument type must "
         "match the tail arguments",
         MustTailCallFunc);
}
