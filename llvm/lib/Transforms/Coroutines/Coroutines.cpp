//===-- Coroutines.cpp ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file implements the common infrastructure for Coroutine Passes.
//===----------------------------------------------------------------------===//

#include "CoroInternal.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

using namespace llvm;

void llvm::initializeCoroutines(PassRegistry &Registry) {
  initializeCoroEarlyPass(Registry);
  initializeCoroSplitPass(Registry);
  initializeCoroElidePass(Registry);
  initializeCoroCleanupPass(Registry);
}

static void addCoroutineOpt0Passes(const PassManagerBuilder &Builder,
                                   legacy::PassManagerBase &PM) {
  PM.add(createCoroSplitPass());
  PM.add(createCoroElidePass());

  PM.add(createBarrierNoopPass());
  PM.add(createCoroCleanupPass());
}

static void addCoroutineEarlyPasses(const PassManagerBuilder &Builder,
                                    legacy::PassManagerBase &PM) {
  PM.add(createCoroEarlyPass());
}

static void addCoroutineScalarOptimizerPasses(const PassManagerBuilder &Builder,
                                              legacy::PassManagerBase &PM) {
  PM.add(createCoroElidePass());
}

static void addCoroutineSCCPasses(const PassManagerBuilder &Builder,
                                  legacy::PassManagerBase &PM) {
  PM.add(createCoroSplitPass());
}

static void addCoroutineOptimizerLastPasses(const PassManagerBuilder &Builder,
                                            legacy::PassManagerBase &PM) {
  PM.add(createCoroCleanupPass());
}

void llvm::addCoroutinePassesToExtensionPoints(PassManagerBuilder &Builder) {
  Builder.addExtension(PassManagerBuilder::EP_EarlyAsPossible,
                       addCoroutineEarlyPasses);
  Builder.addExtension(PassManagerBuilder::EP_EnabledOnOptLevel0,
                       addCoroutineOpt0Passes);
  Builder.addExtension(PassManagerBuilder::EP_CGSCCOptimizerLate,
                       addCoroutineSCCPasses);
  Builder.addExtension(PassManagerBuilder::EP_ScalarOptimizerLate,
                       addCoroutineScalarOptimizerPasses);
  Builder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                       addCoroutineOptimizerLastPasses);
}

// Construct the lowerer base class and initialize its members.
coro::LowererBase::LowererBase(Module &M)
    : TheModule(M), Context(M.getContext()),
      ResumeFnType(FunctionType::get(Type::getVoidTy(Context),
                                     Type::getInt8PtrTy(Context),
                                     /*isVarArg=*/false)) {}

// Creates a sequence of instructions to obtain a resume function address using
// llvm.coro.subfn.addr. It generates the following sequence:
//
//    call i8* @llvm.coro.subfn.addr(i8* %Arg, i8 %index)
//    bitcast i8* %2 to void(i8*)*

Value *coro::LowererBase::makeSubFnCall(Value *Arg, int Index,
                                        Instruction *InsertPt) {
  auto *IndexVal = ConstantInt::get(Type::getInt8Ty(Context), Index);
  auto *Fn = Intrinsic::getDeclaration(&TheModule, Intrinsic::coro_subfn_addr);

  assert(Index >= CoroSubFnInst::IndexFirst &&
         Index < CoroSubFnInst::IndexLast &&
         "makeSubFnCall: Index value out of range");
  auto *Call = CallInst::Create(Fn, {Arg, IndexVal}, "", InsertPt);

  auto *Bitcast =
      new BitCastInst(Call, ResumeFnType->getPointerTo(), "", InsertPt);
  return Bitcast;
}

#ifndef NDEBUG
static bool isCoroutineIntrinsicName(StringRef Name) {
  // NOTE: Must be sorted!
  static const char *const CoroIntrinsics[] = {
      "llvm.coro.alloc",   "llvm.coro.begin", "llvm.coro.destroy",
      "llvm.coro.done",    "llvm.coro.end",   "llvm.coro.frame",
      "llvm.coro.free",    "llvm.coro.param", "llvm.coro.promise",
      "llvm.coro.resume",  "llvm.coro.save",  "llvm.coro.size",
      "llvm.coro.suspend",
  };
  return Intrinsic::lookupLLVMIntrinsicByName(CoroIntrinsics, Name) != -1;
}
#endif

// Verifies if a module has named values listed. Also, in debug mode verifies
// that names are intrinsic names.
bool coro::declaresIntrinsics(Module &M,
                              std::initializer_list<StringRef> List) {

  for (StringRef Name : List) {
    assert(isCoroutineIntrinsicName(Name) && "not a coroutine intrinsic");
    if (M.getNamedValue(Name))
      return true;
  }

  return false;
}
