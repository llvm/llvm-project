//===-- CopyProf.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements the LLVM IR instrumentation passes for CopyProf.
/// It adds enter/exit callbacks to C++ special member functions, and
/// instruments store instructions.
///
/// The basic idea of the CopyProf algorithm works like this:
/// An object copy Y is made from original object X. The shadow memory
/// corresponding to (and owned by) Y is marked as "copied". Any subsequent
/// memory store to the memory corresponding to Y marks the shadow memory as
/// "modified". When Y is destroyed and all of its corresponding shadow memory
/// is marked as "copied", the object is reported as an unnecessary copy.
///
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/CopyProf.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Transforms/Utils/Instrumentation.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <array>
#include <cstddef>
#include <cstdint>

// TODO: Convert CopyProfPass and CopyProfStoresPass to module passes so that
// the runtime callbacks can be cached, thus avoiding repetitive symbol table
// lookups.

using namespace llvm;

// Names for the module c'tor to initialize the runtime, and the runtime
// initialization function itself.
constexpr StringRef CopyProfModuleCtorName = "copyprof.module_ctor";
constexpr StringRef CopyProfInitName = "__copyprof_init";

// Runtime callback function names.
constexpr StringRef CopyProfCtorEnterCallbackName =
    "__copyprof_ctor_enter_callback";
constexpr StringRef CopyProfCtorExitCallbackName =
    "__copyprof_ctor_exit_callback";
constexpr StringRef CopyProfCopyCtorEnterCallbackName =
    "__copyprof_copy_ctor_enter_callback";
constexpr StringRef CopyProfCopyCtorExitCallbackName =
    "__copyprof_copy_ctor_exit_callback";
constexpr StringRef CopyProfCopyAssignOpEnterCallbackName =
    "__copyprof_copy_assign_op_enter_callback";
constexpr StringRef CopyProfCopyAssignOpExitCallbackName =
    "__copyprof_copy_assign_op_exit_callback";
constexpr StringRef CopyProfDtorEnterCallbackName =
    "__copyprof_dtor_enter_callback";
constexpr StringRef CopyProfDtorExitCallbackName =
    "__copyprof_dtor_exit_callback";
constexpr StringRef CopyProfStoreCallbackName = "__copyprof_store_callback";

// Attribute strings used by the frontend to mark special member functions.
constexpr StringRef CopyProfCtorAttr = "copyprof-ctor";
constexpr StringRef CopyProfCopyCtorAttr = "copyprof-copy-ctor";
constexpr StringRef CopyProfCopyAssignAttr = "copyprof-copy-assign-op";
constexpr StringRef CopyProfDtorAttr = "copyprof-dtor";

static bool insertModuleCtor(Module &M) {
  bool Modified = false;
  getOrCreateSanitizerCtorAndInitFunctions(
      M, CopyProfModuleCtorName, CopyProfInitName,
      /*InitArgTypes=*/{},
      /*InitArgs=*/{}, [&](Function *Ctor, FunctionCallee) {
        // Mark the ctor so it's never instrumented itself.
        Ctor->addFnAttr(Attribute::DisableSanitizerInstrumentation);
        appendToGlobalCtors(M, Ctor, 0);
        Modified = true;
      });
  return Modified;
}

static bool isCopyProfCandidate(const Function &F) {
  // Must not instrument functions that are explicitly disallowed for
  // instrumentation, or naked functions.
  if (F.isDeclaration() ||
      F.hasFnAttribute(Attribute::DisableSanitizerInstrumentation) ||
      F.hasFnAttribute(Attribute::Naked))
    return false;

  // Don't instrument a function at all if it's ending with a tail call.
  // Alternatively, the exit callback could be placed before the tail call, but
  // that would risk missing observable side-effects needed by CopyProf to infer
  // memory ownership (potentially leading to flase positive reports).
  // Skipping this function favors false negatives over false positives.
  for (const BasicBlock &BB : F)
    if (BB.getTerminatingMustTailCall())
      return false;

  return F.hasFnAttribute(CopyProfCtorAttr) ||
         F.hasFnAttribute(CopyProfCopyCtorAttr) ||
         F.hasFnAttribute(CopyProfCopyAssignAttr) ||
         F.hasFnAttribute(CopyProfDtorAttr);
}

static bool isCopyProfStoresCandidate(const Function &F) {
  return !F.isDeclaration() &&
         !F.hasFnAttribute(Attribute::DisableSanitizerInstrumentation) &&
         !F.hasFnAttribute(Attribute::Naked);
}

// Returns the object size in bytes that was stored in the given function
// attribute during parsing in the frontend.
static size_t getAttrValueAsInt(const Function &F, StringRef Attr) {
  size_t IntValue = 0;
  if (!to_integer<size_t>(F.getFnAttribute(Attr).getValueAsString(), IntValue,
                          /*Base=*/10)) {
    report_fatal_error(formatv("Unable to parse integer value from function "
                               "attribute value in '{0}': {1}:{2}",
                               F.getName(), Attr,
                               F.getFnAttribute(Attr).getValueAsString()));
  }
  return IntValue;
}

namespace {

// Instruments special member functions to call into the CopyProf runtime.
class CopyProf {
public:
  explicit CopyProf(Module &M);
  bool instrumentFunction(Function &F);

private:
  void insertCallback(Function &F, size_t ObjSize, unsigned NumArgs,
                      FunctionCallee Callback, FunctionCallee ExitCallback);

  LLVMContext *Ctx;
  Type *IntPtrTy;
  FunctionCallee CtorEnterCallback;
  FunctionCallee CtorExitCallback;
  FunctionCallee CopyCtorEnterCallback;
  FunctionCallee CopyCtorExitCallback;
  FunctionCallee CopyAssignOpEnterCallback;
  FunctionCallee CopyAssignOpExitCallback;
  FunctionCallee DtorEnterCallback;
  FunctionCallee DtorExitCallback;
};

// Late-stage pass that instruments store instructions after all optimizations
// have run (to avoid instrumenting stores that would be eliminated).
class CopyProfStores {
public:
  explicit CopyProfStores(Module &M);
  bool instrumentFunction(Function &F);

private:
  Type *IntPtrTy;
  FunctionCallee StoreCallback;
};

} // namespace

CopyProf::CopyProf(Module &M) {
  Ctx = &M.getContext();
  IRBuilder<> IRB(*Ctx);
  IntPtrTy = IRB.getIntPtrTy(M.getDataLayout());
  Type *PtrTy = IRB.getPtrTy();
  Type *VoidTy = IRB.getVoidTy();
  // CopyProf callbacks never throw exceptions.
  AttributeList Attr;
  Attr = Attr.addFnAttribute(*Ctx, Attribute::NoUnwind);
  CtorEnterCallback = M.getOrInsertFunction(CopyProfCtorEnterCallbackName, Attr,
                                            VoidTy, PtrTy, IntPtrTy);
  CtorExitCallback = M.getOrInsertFunction(CopyProfCtorExitCallbackName, Attr,
                                           VoidTy, PtrTy, IntPtrTy);
  CopyCtorEnterCallback = M.getOrInsertFunction(
      CopyProfCopyCtorEnterCallbackName, Attr, VoidTy, PtrTy, PtrTy, IntPtrTy);
  CopyCtorExitCallback = M.getOrInsertFunction(
      CopyProfCopyCtorExitCallbackName, Attr, VoidTy, PtrTy, PtrTy, IntPtrTy);
  CopyAssignOpEnterCallback =
      M.getOrInsertFunction(CopyProfCopyAssignOpEnterCallbackName, Attr, VoidTy,
                            PtrTy, PtrTy, IntPtrTy);
  CopyAssignOpExitCallback =
      M.getOrInsertFunction(CopyProfCopyAssignOpExitCallbackName, Attr, VoidTy,
                            PtrTy, PtrTy, IntPtrTy);
  DtorEnterCallback = M.getOrInsertFunction(CopyProfDtorEnterCallbackName, Attr,
                                            VoidTy, PtrTy, IntPtrTy);
  DtorExitCallback = M.getOrInsertFunction(CopyProfDtorExitCallbackName, Attr,
                                           VoidTy, PtrTy, IntPtrTy);
}

bool CopyProf::instrumentFunction(Function &F) {
  bool Modified = true;
  if (F.hasFnAttribute(CopyProfCtorAttr))
    insertCallback(F, getAttrValueAsInt(F, CopyProfCtorAttr), /*NumArgs=*/1,
                   CtorEnterCallback, CtorExitCallback);
  else if (F.hasFnAttribute(CopyProfCopyCtorAttr))
    insertCallback(F, getAttrValueAsInt(F, CopyProfCopyCtorAttr), /*NumArgs=*/2,
                   CopyCtorEnterCallback, CopyCtorExitCallback);
  else if (F.hasFnAttribute(CopyProfCopyAssignAttr))
    insertCallback(F, getAttrValueAsInt(F, CopyProfCopyAssignAttr),
                   /*NumArgs=*/2, CopyAssignOpEnterCallback,
                   CopyAssignOpExitCallback);
  else if (F.hasFnAttribute(CopyProfDtorAttr))
    insertCallback(F, getAttrValueAsInt(F, CopyProfDtorAttr), /*NumArgs=*/1,
                   DtorEnterCallback, DtorExitCallback);
  else
    Modified = false;

  return Modified;
}

void CopyProf::insertCallback(Function &F, size_t ObjSize, unsigned NumArgs,
                              FunctionCallee EntryCallback,
                              FunctionCallee ExitCallback) {
  auto InsertCallback = [IntPtrTy = IntPtrTy, ObjSize,
                         NumArgs](Function &F, InstrumentationIRBuilder &&IRB,
                                  FunctionCallee Callback) {
    SmallVector<Value *, 3> Args;
    // `this` is always the first argument to a special member function, but
    // copy c'tor / copy assignment operator will have the other `this` ptr
    // passed as their second argument.
    assert(NumArgs == 1 || NumArgs == 2);
    for (unsigned I = 0; I < NumArgs; ++I)
      Args.push_back(F.getArg(I));
    // The last argument to the callback is the static size of the object
    // pointed at by `this`.
    Args.push_back(ConstantInt::get(IntPtrTy, ObjSize));
    IRB.CreateCall(Callback, Args);
  };

  InsertCallback(
      F,
      InstrumentationIRBuilder{&F.getEntryBlock(),
                               F.getEntryBlock().getFirstNonPHIOrDbgOrAlloca()},
      EntryCallback);
  for (BasicBlock &BB : F) {
    Instruction *Term = BB.getTerminator();
    if (isa<ReturnInst>(Term) || isa<ResumeInst>(Term))
      InsertCallback(F, InstrumentationIRBuilder{Term}, ExitCallback);
  }
}

CopyProfStores::CopyProfStores(Module &M) {
  LLVMContext &Ctx = M.getContext();
  IRBuilder<> IRB(Ctx);
  IntPtrTy = IRB.getIntPtrTy(M.getDataLayout());
  Type *PtrTy = IRB.getPtrTy();
  Type *VoidTy = IRB.getVoidTy();
  // CopyProf callbacks never throw exceptions.
  AttributeList Attr;
  Attr = Attr.addFnAttribute(Ctx, Attribute::NoUnwind);
  StoreCallback = M.getOrInsertFunction(CopyProfStoreCallbackName, Attr, VoidTy,
                                        PtrTy, IntPtrTy);
}

bool CopyProfStores::instrumentFunction(Function &F) {
  // TODO: handle all types of memory stores (memory intrinsics, masked store
  // intrinsics, AtomicRMW, and AtomicCmpXchg).
  const DataLayout &DL = F.getParent()->getDataLayout();
  SmallVector<StoreInst *, 16> ToInstrument;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *SI = dyn_cast<StoreInst>(&I);
          SI != nullptr && SI->getPointerAddressSpace() == 0 &&
          !SI->hasMetadata(LLVMContext::MD_nosanitize) &&
          // Scalable vector stores have no compile-time-constant size so skip
          // them.
          !DL.getTypeStoreSize(SI->getValueOperand()->getType()).isScalable())
        ToInstrument.push_back(SI);
    }
  }
  if (ToInstrument.empty())
    return false;

  for (StoreInst *SI : ToInstrument) {
    uint64_t StoredSize =
        DL.getTypeStoreSize(SI->getValueOperand()->getType()).getFixedValue();
    InstrumentationIRBuilder IRB(SI);
    std::array<Value *, 2> Args = {SI->getPointerOperand(),
                                   ConstantInt::get(IntPtrTy, StoredSize)};
    IRB.CreateCall(StoreCallback, Args);
  }
  return true;
}

PreservedAnalyses CopyProfPass::run(Function &F, FunctionAnalysisManager &) {
  if (!isCopyProfCandidate(F))
    return PreservedAnalyses::all();
  CopyProf Impl(*F.getParent());
  if (!Impl.instrumentFunction(F))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

PreservedAnalyses ModuleCopyProfPass::run(Module &M, ModuleAnalysisManager &) {
  return insertModuleCtor(M) ? PreservedAnalyses::none()
                             : PreservedAnalyses::all();
}

PreservedAnalyses CopyProfStoresPass::run(Function &F,
                                          FunctionAnalysisManager &) {
  if (!isCopyProfStoresCandidate(F))
    return PreservedAnalyses::all();
  CopyProfStores Impl(*F.getParent());
  if (!Impl.instrumentFunction(F))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
