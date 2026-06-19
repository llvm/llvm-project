//===-- EJitWrapperGen.cpp - EmbeddedJIT Wrapper Code Generation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  PASS3: Insert wrapper prologue in every ejit_entry function. Uses the
//  single-function mixed scheme: wraps the original body in a fallback
//  block with a JIT dispatch path. No separate wrapper function.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/EmbeddedJIT/EJitPasses.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <map>
#include <string>

using namespace llvm;
using namespace llvm::ejit;

#define DEBUG_TYPE "ejit-wrapper-gen"

extern cl::opt<bool> EnableEJitGlobalCtors;

static cl::opt<bool> EJitNoInlineEntry(
    "ejit-noinline-entry", cl::init(true), cl::Hidden,
    cl::desc("Add noinline attribute to ejit_entry functions to prevent the "
             "CGSCC inliner from duplicating the JIT wrapper into callers"));

// Temporary migration switch while the taskpool path is being stabilized.
// The default preserves the mature synchronous ABI. Remove this option and
// make the asynchronous path unconditional once taskpool rollout is complete.
static cl::opt<bool> EJitAsyncWrapper(
    "ejit-wrapper-async", cl::init(false), cl::Hidden,
    cl::desc("Generate taskpool-based asynchronous EJIT wrappers instead of "
             "the synchronous ejit_compile_or_get wrapper"));

namespace {

struct PeriodArrIndInfo {
  std::string PeriodName;
  unsigned ArgIndex;
  uint32_t DimType;
};

static SmallVector<PeriodArrIndInfo, 4> getPeriodArrIndInfo(const Function &F) {
  SmallVector<PeriodArrIndInfo, 4> Result;
  MDNode *MD = F.getMetadata(MD_EJIT_METADATA);
  if (!MD)
    return Result;

  for (const MDOperand &Op : MD->operands()) {
    auto *Sub = dyn_cast<MDNode>(Op.get());
    if (!Sub || Sub->getNumOperands() < 3)
      continue;
    if (auto *Tag = dyn_cast<MDString>(Sub->getOperand(0))) {
      if (Tag->getString() == TAG_EJIT_PERIOD_ARR_IND) {
        auto *PN = dyn_cast<MDString>(Sub->getOperand(1));
        auto *IdxC = dyn_cast<ConstantAsMetadata>(Sub->getOperand(2));
        if (PN && IdxC)
          if (auto *CI = dyn_cast<ConstantInt>(IdxC->getValue()))
            Result.push_back({PN->getString().str(),
                              static_cast<unsigned>(CI->getZExtValue()), 0});
      }
    }
  }
  return Result;
}

// Per-lifecycle i32 global holding the dimType slot. Internal linkage so the
// same-named global in another module stays independent (each is filled with
// the same registry-assigned slot at registration). Initialized to the
// "unassigned" sentinel so a missing registration cleanly disables the path.
static GlobalVariable *getOrCreateDimTypeGlobal(Module &M,
                                                StringRef PeriodName) {
  std::string GVName = ("__ejit_dimtype_" + PeriodName).str();
  if (auto *Existing = M.getGlobalVariable(GVName))
    return Existing;
  auto *I32Ty = Type::getInt32Ty(M.getContext());
  return new GlobalVariable(
      M, I32Ty, /*isConstant=*/false, GlobalValue::InternalLinkage,
      ConstantInt::get(I32Ty, kEJitInvalidDimType), GVName);
}

// Emit registration that fills each per-lifecycle dimType global with the slot
// the process-global EJitLifecycleRegistry assigns by name: ejit_register_
// lifecycle() calls in ejit_auto_register (constructor path) plus a static
// __ejit_registry_lifecycle[] table (bare-metal / test fallback). Mirrors the
// period pass. Idempotent: skips if the static table already exists.
static void
emitLifecycleRegistration(Module &M,
                          const std::map<std::string, GlobalVariable *> &LCs) {
  if (LCs.empty() || M.getGlobalVariable("__ejit_registry_lifecycle"))
    return;
  LLVMContext &Ctx = M.getContext();
  auto *PtrTy = PointerType::getUnqual(Ctx);
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  // void ejit_register_lifecycle(const char *name, uint32_t *slotOut)
  M.getOrInsertFunction(
      FN_REGISTER_LIFECYCLE,
      FunctionType::get(Type::getVoidTy(Ctx), {PtrTy, PtrTy}, false));

  Function *AutoReg = M.getFunction(FN_AUTO_REGISTER);
  bool CreatedAutoReg = false;
  if (!AutoReg) {
    auto *AutoRegTy = FunctionType::get(Type::getVoidTy(Ctx), false);
    AutoReg = Function::Create(AutoRegTy, GlobalValue::InternalLinkage,
                               FN_AUTO_REGISTER, &M);
    BasicBlock::Create(Ctx, "entry", AutoReg);
    ReturnInst::Create(Ctx, &AutoReg->getEntryBlock());
    CreatedAutoReg = true;
  }
  Instruction *Ret = AutoReg->getEntryBlock().getTerminator();
  FunctionCallee FnRegLc = M.getFunction(FN_REGISTER_LIFECYCLE);
  for (auto &KV : LCs) {
    IRBuilder<> Builder(Ret);
    Value *Name = Builder.CreateGlobalString(KV.first);
    Builder.CreateCall(FnRegLc,
                       {Name, Builder.CreateBitCast(KV.second, PtrTy)});
  }

  // Only register the constructor when WE created ejit_auto_register: if PASS2
  // (period registration) already created and appended it, reusing it here and
  // appending again would run the whole constructor twice.
  if (EnableEJitGlobalCtors && CreatedAutoReg)
    appendToGlobalCtors(M, AutoReg, EJIT_CTOR_PRIORITY);

  // Static registry table for bare-metal / testing fallback.
  StructType *EntryTy = StructType::get(
      Ctx, {I32Ty, PtrTy, PtrTy, PtrTy, I64Ty}, /*isPacked=*/false);
  auto makeStrGV = [&](const std::string &S) -> Constant * {
    Constant *Str = ConstantDataArray::getString(Ctx, S, true);
    auto *GV =
        new GlobalVariable(M, Str->getType(), true, GlobalValue::PrivateLinkage,
                           Str, ".ejit.str.");
    return ConstantExpr::getBitCast(GV, PtrTy);
  };
  SmallVector<Constant *, 16> Entries;
  for (auto &KV : LCs) {
    Entries.push_back(ConstantStruct::get(
        EntryTy, {ConstantInt::get(I32Ty, 5), // EJIT_REG_LIFECYCLE
                  makeStrGV(KV.first), ConstantPointerNull::get(PtrTy),
                  ConstantExpr::getBitCast(KV.second, PtrTy),
                  ConstantInt::get(I64Ty, 0)}));
  }
  Entries.push_back(ConstantStruct::get(
      EntryTy,
      {ConstantInt::get(I32Ty, 4), // EJIT_REG_NONE (sentinel)
       ConstantPointerNull::get(PtrTy), ConstantPointerNull::get(PtrTy),
       ConstantPointerNull::get(PtrTy), ConstantInt::get(I64Ty, 0)}));
  ArrayType *ArrayTy = ArrayType::get(EntryTy, Entries.size());
  (void)new GlobalVariable(
      M, ArrayTy, /*isConstant=*/true, GlobalValue::ExternalLinkage,
      ConstantArray::get(ArrayTy, Entries), "__ejit_registry_lifecycle");
}

// Per-function i32 global holding the dense funcIndex. Internal linkage so the
// same-named global in another module stays independent (each is filled with
// the same registry-assigned index at registration). Initialized to the
// "unregistered" sentinel so a missing/overflowing registration cleanly falls
// back without entering the taskpool.
static GlobalVariable *getOrCreateFuncIndexGlobal(Module &M,
                                                  StringRef FuncName) {
  std::string GVName = ("__ejit_funcidx_" + FuncName).str();
  if (auto *Existing = M.getGlobalVariable(GVName))
    return Existing;
  auto *I32Ty = Type::getInt32Ty(M.getContext());
  return new GlobalVariable(
      M, I32Ty, /*isConstant=*/false, GlobalValue::InternalLinkage,
      ConstantInt::get(I32Ty, kEJitInvalidFuncIndex), GVName);
}

// Emit registration that fills each per-function dense-funcIndex global with
// the index the process-global EJitFuncRegistry assigns by name: ejit_register_
// funcindex() calls in ejit_auto_register (constructor path) plus a static
// __ejit_registry_funcindex[] table (bare-metal / test fallback). Mirrors the
// lifecycle registration. Idempotent: skips if the static table already exists.
static void
emitFuncIndexRegistration(Module &M,
                          const std::map<std::string, GlobalVariable *> &Fns) {
  if (Fns.empty() || M.getGlobalVariable("__ejit_registry_funcindex"))
    return;
  LLVMContext &Ctx = M.getContext();
  auto *PtrTy = PointerType::getUnqual(Ctx);
  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  // void ejit_register_funcindex(const char *name, uint32_t *slotOut)
  M.getOrInsertFunction(
      FN_REGISTER_FUNCINDEX,
      FunctionType::get(Type::getVoidTy(Ctx), {PtrTy, PtrTy}, false));

  Function *AutoReg = M.getFunction(FN_AUTO_REGISTER);
  bool CreatedAutoReg = false;
  if (!AutoReg) {
    auto *AutoRegTy = FunctionType::get(Type::getVoidTy(Ctx), false);
    AutoReg = Function::Create(AutoRegTy, GlobalValue::InternalLinkage,
                               FN_AUTO_REGISTER, &M);
    BasicBlock::Create(Ctx, "entry", AutoReg);
    ReturnInst::Create(Ctx, &AutoReg->getEntryBlock());
    CreatedAutoReg = true;
  }
  Instruction *Ret = AutoReg->getEntryBlock().getTerminator();
  FunctionCallee FnReg = M.getFunction(FN_REGISTER_FUNCINDEX);
  for (auto &KV : Fns) {
    IRBuilder<> Builder(Ret);
    Value *Name = Builder.CreateGlobalString(KV.first);
    Builder.CreateCall(FnReg, {Name, Builder.CreateBitCast(KV.second, PtrTy)});
  }

  // Only register the constructor when WE created ejit_auto_register (else
  // PASS2 / lifecycle emission already appended it).
  if (EnableEJitGlobalCtors && CreatedAutoReg)
    appendToGlobalCtors(M, AutoReg, EJIT_CTOR_PRIORITY);

  // Static registry table for bare-metal / testing fallback.
  StructType *EntryTy = StructType::get(
      Ctx, {I32Ty, PtrTy, PtrTy, PtrTy, I64Ty}, /*isPacked=*/false);
  auto makeStrGV = [&](const std::string &S) -> Constant * {
    Constant *Str = ConstantDataArray::getString(Ctx, S, true);
    auto *GV =
        new GlobalVariable(M, Str->getType(), true, GlobalValue::PrivateLinkage,
                           Str, ".ejit.str.");
    return ConstantExpr::getBitCast(GV, PtrTy);
  };
  SmallVector<Constant *, 16> Entries;
  for (auto &KV : Fns) {
    Entries.push_back(ConstantStruct::get(
        EntryTy, {ConstantInt::get(I32Ty, 6), // EJIT_REG_FUNCINDEX
                  makeStrGV(KV.first), ConstantPointerNull::get(PtrTy),
                  ConstantExpr::getBitCast(KV.second, PtrTy),
                  ConstantInt::get(I64Ty, 0)}));
  }
  Entries.push_back(ConstantStruct::get(
      EntryTy,
      {ConstantInt::get(I32Ty, 4), // EJIT_REG_NONE (sentinel)
       ConstantPointerNull::get(PtrTy), ConstantPointerNull::get(PtrTy),
       ConstantPointerNull::get(PtrTy), ConstantInt::get(I64Ty, 0)}));
  ArrayType *ArrayTy = ArrayType::get(EntryTy, Entries.size());
  (void)new GlobalVariable(
      M, ArrayTy, /*isConstant=*/true, GlobalValue::ExternalLinkage,
      ConstantArray::get(ArrayTy, Entries), "__ejit_registry_funcindex");
}

} // anonymous namespace

PreservedAnalyses EJitWrapperGenPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  LLVMContext &Ctx = M.getContext();
  auto *PtrTy = PointerType::getUnqual(Ctx);

  SmallVector<Function *, 4> EntryFuncs;
  for (Function &F : M.functions()) {
    MDNode *MD = F.getMetadata(MD_EJIT_METADATA);
    if (hasMDStringEntry(MD, TAG_EJIT_ENTRY) && !F.isDeclaration())
      EntryFuncs.push_back(&F);
  }

  if (EntryFuncs.empty()) {
    return PreservedAnalyses::all();
  }

  auto *I32Ty = Type::getInt32Ty(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);
  if (EJitAsyncWrapper) {
    // ejit_taskpool_compile_or_get(i32 funcIndex, ptr dims, i32 numDims,
    //                              ptr outFn, ptr outBucket)
    // ejit_taskpool_release_read(i32 bucketIndex)
    M.getOrInsertFunction(
        FN_TASKPOOL_COMPILE_OR_GET,
        FunctionType::get(I32Ty, {I32Ty, PtrTy, I32Ty, PtrTy, PtrTy}, false));
    M.getOrInsertFunction(
        FN_TASKPOOL_RELEASE_READ,
        FunctionType::get(Type::getVoidTy(Ctx), {I32Ty}, false));
  } else {
    // Mature synchronous path used during taskpool rollout.
    M.getOrInsertFunction(FN_COMPILE_OR_GET,
                          FunctionType::get(PtrTy, {I64Ty, PtrTy}, false));
  }

  auto isAlreadyWrapped = [](Function &F) -> bool {
    if (!F.getEntryBlock().getName().starts_with("jit_entry"))
      return false;
    // The wrapper's jit_entry loads this function's @__ejit_funcidx_<name>
    // global; that load uniquely identifies an already-wrapped function.
    for (Instruction &I : F.getEntryBlock())
      if (auto *LI = dyn_cast<LoadInst>(&I))
        if (auto *GV = dyn_cast<GlobalVariable>(LI->getPointerOperand()))
          if (GV->getName().starts_with("__ejit_funcidx_"))
            return true;
    return false;
  };

  // Idempotency: if a previous PASS3 run already wrapped every entry function
  // (EJitAotModulePass may invoke PASS3 several times), there is nothing to do
  // — and re-emitting the module-level lifecycle registration would duplicate
  // it.
  if (llvm::all_of(EntryFuncs,
                   [&](Function *F) { return isAlreadyWrapped(*F); }))
    return PreservedAnalyses::all();

  // Cross-module-stable dimType: gather the distinct lifecycle (period) names
  // this module references and give each a per-lifecycle i32 global seeded with
  // the "unassigned" sentinel. The slot is assigned ONCE, by name, in the
  // process-global EJitLifecycleRegistry at registration time and written into
  // this global; the wrapper LOADS it instead of baking a per-module sorted
  // guess, so two modules sharing a lifecycle observe the same slot and two
  // different lifecycles never collide (EJitLifecycleRegistry.h). funcIndex is
  // assigned the same way by the process-global EJitFuncRegistry (below).
  std::map<std::string, GlobalVariable *> DimTypeGlobals;
  for (Function *F : EntryFuncs)
    for (auto &PI : getPeriodArrIndInfo(*F))
      if (!PI.PeriodName.empty())
        DimTypeGlobals.emplace(PI.PeriodName, nullptr);
  if (EJitAsyncWrapper && DimTypeGlobals.size() > kEJitMaxDimTypes) {
    Ctx.emitError("ejit-wrapper-gen: module references " +
                  Twine(DimTypeGlobals.size()) +
                  " distinct lifecycle dimensions but at most " +
                  Twine(kEJitMaxDimTypes) + " are supported (spec §5.1)");
    return PreservedAnalyses::all();
  }
  if (EJitAsyncWrapper) {
    for (auto &KV : DimTypeGlobals)
      KV.second = getOrCreateDimTypeGlobal(M, KV.first);
    emitLifecycleRegistration(M, DimTypeGlobals);
  }

  // Explicit, registration-time dense funcIndex: give each entry function a
  // per-function i32 global seeded with kEJitInvalidFuncIndex. The dense index
  // is assigned ONCE, by name, in the process-global EJitFuncRegistry and
  // backfilled into this global; the wrapper LOADS it and falls back WITHOUT
  // entering the taskpool while it is still invalid (unregistered / capacity
  // exhausted). The loader keys its table by the SAME registry index, so no two
  // functions can alias one slot (EJitFuncRegistry.h).
  std::map<std::string, GlobalVariable *> FuncIndexGlobals;
  for (Function *F : EntryFuncs)
    FuncIndexGlobals.emplace(F->getName().str(),
                             getOrCreateFuncIndexGlobal(M, F->getName()));
  emitFuncIndexRegistration(M, FuncIndexGlobals);

  LLVM_DEBUG(dbgs() << "ejit-wrapper-gen: " << EntryFuncs.size()
                    << " entry function(s)\n");
  bool Changed = false;
  for (Function *F : EntryFuncs) {
    LLVM_DEBUG(dbgs() << "ejit-wrapper-gen: wrapping " << F->getName() << "\n");
    // Idempotency guard: skip functions already wrapped by an earlier pass run.
    // PASS3 may be invoked multiple times via EJitAotModulePass (e.g. O1+O2
    // pipelines), and re-wrapping produces broken PHI nodes referencing stale
    // predecessor blocks.
    if (isAlreadyWrapped(*F))
      continue;

    // Prevent the CGSCC inliner from inlining the wrapped function into
    // callers. Each call site would duplicate the JIT dispatch logic
    // (cacheKey computation, ejit_compile_or_get call, indirect call) and
    // the inliner may produce
    // inconsistent AOT fallback code depending on call-site context.
    if (EJitNoInlineEntry)
      F->addFnAttr(Attribute::NoInline);

    auto PeriodInds = getPeriodArrIndInfo(*F);
    unsigned DimCount = PeriodInds.size();

    if (DimCount > 4) {
      F->getContext().emitError("ejit-wrapper-gen: more than 4 "
                                "ejit_period_arr_ind dimensions are not "
                                "supported");
      continue;
    }

    // Validate the metadata: every dim must name a non-empty lifecycle that has
    // a per-lifecycle dimType global (created above for every distinct name the
    // module references), no two dims may name the SAME lifecycle (a duplicated
    // dimension — distinct names are guaranteed distinct slots at runtime), and
    // arg indices/types must be in range. The dimType slot itself is resolved
    // at runtime via the global, never baked here.
    bool Invalid = false;
    SmallVector<StringRef, 4> SeenNames;
    unsigned ArgCount = F->arg_size();
    for (unsigned I = 0; I < DimCount; ++I) {
      auto GIt = DimTypeGlobals.find(PeriodInds[I].PeriodName);
      if (PeriodInds[I].PeriodName.empty() || GIt == DimTypeGlobals.end()) {
        F->getContext().emitError("ejit-wrapper-gen: invalid period name in "
                                  "ejit_period_arr_ind: " +
                                  PeriodInds[I].PeriodName);
        Invalid = true;
        break;
      }
      if (llvm::is_contained(SeenNames, StringRef(PeriodInds[I].PeriodName))) {
        F->getContext().emitError("ejit-wrapper-gen: duplicated lifecycle "
                                  "dimension in ejit_period_arr_ind metadata");
        Invalid = true;
        break;
      }
      SeenNames.push_back(PeriodInds[I].PeriodName);

      if (PeriodInds[I].ArgIndex >= ArgCount) {
        F->getContext().emitError("ejit-wrapper-gen: ejit_period_arr_ind "
                                  "argument index out of range");
        Invalid = true;
        break;
      }
      Value *ArgVal = F->getArg(PeriodInds[I].ArgIndex);
      if (!ArgVal->getType()->isIntegerTy()) {
        F->getContext().emitError("ejit-wrapper-gen: ejit_period_arr_ind "
                                  "argument must be an integer type");
        Invalid = true;
        break;
      }
    }

    if (Invalid)
      continue;

    // Save original entry block
    BasicBlock &OrigEntry = F->getEntryBlock();

    // Create four new blocks: jit_entry (funcIndex guard), jit_call (taskpool
    // request), jit_fallback (AOT body) and jit_dispatch (run JIT code).
    auto *JitEntry = BasicBlock::Create(Ctx, "jit_entry", F, &OrigEntry);
    auto *JitCall = BasicBlock::Create(Ctx, "jit_call", F);
    auto *JitFallback = BasicBlock::Create(Ctx, "jit_fallback", F);
    auto *JitDispatch = BasicBlock::Create(Ctx, "jit_dispatch", F);

    // Splice all instructions from OrigEntry to jit_fallback
    JitFallback->splice(JitFallback->end(), &OrigEntry, OrigEntry.begin(),
                        OrigEntry.end());

    // Fix PHI nodes in successor blocks that reference the original entry
    // block. Optimization passes may have created PHI nodes pointing at the
    // entry block, and after we replace it with jit_fallback those PHI entries
    // must be updated.
    OrigEntry.replaceAllUsesWith(JitFallback);

    // Delete the now-empty original entry block
    OrigEntry.eraseFromParent();

    // jit_entry: load the registration-backfilled dense funcIndex. While it is
    // invalid, branch straight to the AOT fallback without entering either
    // compile path.
    IRBuilder<> Builder(JitEntry);
    auto *DimPairTy = StructType::get(I32Ty, I32Ty);
    Value *DimsAlloca = nullptr;
    Value *OutFnAlloca = nullptr;
    Value *OutBucketAlloca = nullptr;
    if (EJitAsyncWrapper) {
      DimsAlloca = Builder.CreateAlloca(ArrayType::get(DimPairTy, 4), nullptr,
                                        "ejit_dims");
      OutFnAlloca = Builder.CreateAlloca(PtrTy, nullptr, "ejit_out_fn");
      OutBucketAlloca = Builder.CreateAlloca(I32Ty, nullptr, "ejit_out_bucket");
    }
    Value *FuncIdx = Builder.CreateLoad(
        I32Ty, FuncIndexGlobals[F->getName().str()], "ejit_funcidx");
    Value *IdxValid = Builder.CreateICmpNE(
        FuncIdx, ConstantInt::get(I32Ty, kEJitInvalidFuncIndex), "ejit_idx_ok");
    Builder.CreateCondBr(IdxValid, JitCall, JitFallback);

    // jit_call: issue either the temporary synchronous ABI or the taskpool ABI.
    Builder.SetInsertPoint(JitCall);
    Value *OutFn = nullptr;
    if (EJitAsyncWrapper) {
      for (unsigned I = 0; I < DimCount; ++I) {
        Value *Idxs[] = {ConstantInt::get(I32Ty, 0),
                         ConstantInt::get(I32Ty, I)};
        Value *PairPtr = Builder.CreateInBoundsGEP(ArrayType::get(DimPairTy, 4),
                                                   DimsAlloca, Idxs);
        Value *DimTypePtr =
            Builder.CreateStructGEP(DimPairTy, PairPtr, 0, "dim_type_ptr");
        Value *InstancePtr =
            Builder.CreateStructGEP(DimPairTy, PairPtr, 1, "instance_ptr");
        Value *DimTypeVal = Builder.CreateLoad(
            I32Ty, DimTypeGlobals[PeriodInds[I].PeriodName], "ejit_dimtype");
        Builder.CreateStore(DimTypeVal, DimTypePtr);

        Value *ArgVal = F->getArg(PeriodInds[I].ArgIndex);
        unsigned BW = cast<IntegerType>(ArgVal->getType())->getBitWidth();
        Value *InstanceId = ArgVal;
        if (BW > 32)
          InstanceId = Builder.CreateTrunc(ArgVal, I32Ty);
        else if (BW < 32)
          InstanceId = Builder.CreateZExt(ArgVal, I32Ty);
        Builder.CreateStore(InstanceId, InstancePtr);
      }

      Value *DimsPtr = Builder.CreatePointerCast(DimsAlloca, PtrTy);
      Value *Status = Builder.CreateCall(
          M.getFunction(FN_TASKPOOL_COMPILE_OR_GET),
          {FuncIdx, DimsPtr, ConstantInt::get(I32Ty, DimCount),
           Builder.CreatePointerCast(OutFnAlloca, PtrTy),
           Builder.CreatePointerCast(OutBucketAlloca, PtrTy)});
      OutFn = Builder.CreateLoad(PtrTy, OutFnAlloca, "ejit_fn");
      Value *HitStatus =
          Builder.CreateICmpEQ(Status, ConstantInt::get(I32Ty, 0));
      Builder.CreateCondBr(
          Builder.CreateAnd(HitStatus, Builder.CreateIsNotNull(OutFn)),
          JitDispatch, JitFallback);
    } else {
      // Keep the established 64-bit synchronous cache-key layout, replacing
      // only the old name hash with the runtime-assigned dense funcIndex.
      Value *Key = Builder.CreateShl(Builder.CreateZExt(FuncIdx, I64Ty), 32);
      for (unsigned I = 0; I < DimCount; ++I) {
        Value *ArgVal = F->getArg(PeriodInds[I].ArgIndex);
        Value *Dim = Builder.CreateZExt(
            Builder.CreateTrunc(ArgVal, Type::getInt8Ty(Ctx)), I64Ty);
        if (I > 0)
          Dim = Builder.CreateShl(Dim, I * 8);
        Key = Builder.CreateOr(Key, Dim);
      }
      OutFn =
          Builder.CreateCall(M.getFunction(FN_COMPILE_OR_GET),
                             {Key, ConstantPointerNull::get(PtrTy)}, "ejit_fn");
      Builder.CreateCondBr(Builder.CreateIsNotNull(OutFn), JitDispatch,
                           JitFallback);
    }

    // jit_dispatch: cast function pointer and call
    Builder.SetInsertPoint(JitDispatch);

    // Build argument list for indirect call
    SmallVector<Value *, 8> Args;
    for (auto &Arg : F->args())
      Args.push_back(&Arg);

    if (F->getReturnType()->isVoidTy()) {
      Builder.CreateCall(F->getFunctionType(), OutFn, Args);
      if (EJitAsyncWrapper) {
        Value *Bucket = Builder.CreateLoad(I32Ty, OutBucketAlloca);
        Builder.CreateCall(M.getFunction(FN_TASKPOOL_RELEASE_READ), {Bucket});
      }
      Builder.CreateRetVoid();
    } else {
      Value *RetVal = Builder.CreateCall(F->getFunctionType(), OutFn, Args);
      if (EJitAsyncWrapper) {
        Value *Bucket = Builder.CreateLoad(I32Ty, OutBucketAlloca);
        Builder.CreateCall(M.getFunction(FN_TASKPOOL_RELEASE_READ), {Bucket});
      }
      Builder.CreateRet(RetVal);
    }

    Changed = true;
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
