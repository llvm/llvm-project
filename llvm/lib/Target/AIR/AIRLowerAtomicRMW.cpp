//===- AIRLowerAtomicRMW.cpp - Lower atomicrmw to AIR intrinsics --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRLowerAtomicRMW.h"
#include "AIR.h"
#include "AIRAddressSpaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "air-lower-atomic-rmw"

// Lower LLVM atomics to AIR's atomic functions, in the form of MSL 4.0
// (AIR 2.8), which floor's metal_atomic.hpp declares:
//
//   T    air.atomic.<space>.<op>.<sign>T(ptr, T val, i32 order, i32 scope,
//                                         i1 volatile)
//   i32  air.atomic.<space>.cmpxchg.weak.i32(ptr, ptr expected, i32 desired,
//                                             i32 success, i32 failure,
//                                             i32 scope, i1 volatile)
//   void air.atomic.fence(i32 flags, i32 order, i32 scope)
//
// <space> is "global" (device memory) or "local" (threadgroup memory). <sign>
// is "s." or "u." for integer operations whose result depends on the sign
// (min, max) and for add, sub, and, or, xor (which do not care, and use
// "s."); it is empty for xchg and for float operations. Before MSL 4.1 the
// operations themselves are relaxed: an ordered LLVM atomic becomes a relaxed
// one between a release fence and an acquire fence, as Metal.jl does.
//
// cmpxchg.weak returns the value it found, and on an Apple M2 Max never
// failed spuriously; a strong LLVM cmpxchg is lowered to one call and
// succeeds when the value found is the expected one.

namespace {
// MSL memory_order and thread_scope, and the mem_flags of fences.
enum : unsigned { OrderRelaxed = 0, OrderSeqCst = 5 };
enum : unsigned { ScopeThreadgroup = 1, ScopeDevice = 2 };
enum : unsigned { MemDevice = 1, MemThreadgroup = 2 };
} // namespace

static std::string buildAtomicName(bool IsGlobal, AtomicRMWInst::BinOp Op,
                                   bool IsFloat) {
  StringRef OpName, Sign;
  switch (Op) {
  case AtomicRMWInst::Xchg:
    OpName = "xchg";
    break;
  case AtomicRMWInst::Add:
  case AtomicRMWInst::FAdd:
    OpName = "add", Sign = "s.";
    break;
  case AtomicRMWInst::Sub:
  case AtomicRMWInst::FSub:
    OpName = "sub", Sign = "s.";
    break;
  case AtomicRMWInst::Max:
    OpName = "max", Sign = "s.";
    break;
  case AtomicRMWInst::Min:
    OpName = "min", Sign = "s.";
    break;
  case AtomicRMWInst::UMax:
    OpName = "max", Sign = "u.";
    break;
  case AtomicRMWInst::UMin:
    OpName = "min", Sign = "u.";
    break;
  case AtomicRMWInst::And:
    OpName = "and", Sign = "s.";
    break;
  case AtomicRMWInst::Or:
    OpName = "or", Sign = "s.";
    break;
  case AtomicRMWInst::Xor:
    OpName = "xor", Sign = "s.";
    break;
  default:
    return {};
  }
  if (IsFloat) {
    if (OpName != "add" && OpName != "sub" && OpName != "xchg")
      return {};
    Sign = "";
  }
  return ("air.atomic." + Twine(IsGlobal ? "global" : "local") + "." + OpName +
          "." + Sign + (IsFloat ? "f32" : "i32"))
      .str();
}

/// The MSL thread_scope of an atomic on address space AS with LLVM scope SSID.
static unsigned getScope(LLVMContext &Ctx, unsigned AS, SyncScope::ID SSID) {
  if (AS == metal::AS::Threadgroup)
    return ScopeThreadgroup;
  SmallVector<StringRef, 4> Names;
  Ctx.getSyncScopeNames(Names);
  if (SSID < Names.size() && Names[SSID] == "workgroup")
    return ScopeThreadgroup;
  return ScopeDevice;
}

/// The MSL memory_order of an LLVM ordering.
static unsigned mslOrder(AtomicOrdering O) {
  switch (O) {
  case AtomicOrdering::Acquire:
    return 2;
  case AtomicOrdering::Release:
    return 3;
  case AtomicOrdering::AcquireRelease:
    return 4;
  case AtomicOrdering::SequentiallyConsistent:
    return OrderSeqCst;
  default:
    return OrderRelaxed;
  }
}

/// air::AtomicOrderingMD on the AIR atomic \p Call.
static void addOrderingHint(CallInst *Call, ArrayRef<AtomicOrdering> Orders) {
  LLVMContext &Ctx = Call->getContext();
  SmallVector<Metadata *, 2> Ops;
  for (AtomicOrdering O : Orders)
    Ops.push_back(ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt32Ty(Ctx), mslOrder(O))));
  Call->setMetadata(air::AtomicOrderingMD, MDNode::get(Ctx, Ops));
}

static bool isOrdered(AtomicOrdering O) {
  return O != AtomicOrdering::NotAtomic && O != AtomicOrdering::Unordered &&
         O != AtomicOrdering::Monotonic;
}

/// air.atomic.fence for an ordered operation of scope Scope.
static void emitFence(IRBuilder<> &B, Module &M, unsigned Scope) {
  Type *I32 = B.getInt32Ty();
  FunctionCallee F = M.getOrInsertFunction(
      "air.atomic.fence", FunctionType::get(B.getVoidTy(), {I32, I32, I32},
                                            false));
  unsigned Flags =
      Scope == ScopeThreadgroup ? MemThreadgroup : MemDevice | MemThreadgroup;
  B.CreateCall(F, {B.getInt32(Flags), B.getInt32(OrderSeqCst),
                   B.getInt32(Scope)});
}

static bool lowerAtomicRMW(Module &M, bool OrderingHints) {
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I1 = Type::getInt1Ty(Ctx);
  bool Changed = false;

  SmallVector<Instruction *, 16> Atomics;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (isa<AtomicRMWInst, AtomicCmpXchgInst, FenceInst>(I))
          Atomics.push_back(&I);

  for (Instruction *I : Atomics) {
    IRBuilder<> B(I);

    if (auto *Fence = dyn_cast<FenceInst>(I)) {
      // A fence within one thread orders nothing for other threads.
      if (Fence->getSyncScopeID() != SyncScope::SingleThread)
        emitFence(B, M, getScope(Ctx, /*AS=*/0, Fence->getSyncScopeID()));
      Fence->eraseFromParent();
      Changed = true;
      continue;
    }

    if (auto *CX = dyn_cast<AtomicCmpXchgInst>(I)) {
      Value *Ptr = CX->getPointerOperand();
      unsigned AS = Ptr->getType()->getPointerAddressSpace();
      if (!CX->getCompareOperand()->getType()->isIntegerTy(32))
        continue; // The writer reports what is left.
      unsigned Scope = getScope(Ctx, AS, CX->getSyncScopeID());
      bool Ordered = isOrdered(CX->getSuccessOrdering());
      // The expected value lives in a private alloca, in the entry block.
      IRBuilder<> EB(&*CX->getFunction()->getEntryBlock().getFirstInsertionPt());
      AllocaInst *Expected = EB.CreateAlloca(I32, nullptr, "cmpxchg.expected");
      if (Ordered)
        emitFence(B, M, Scope);
      B.CreateStore(CX->getCompareOperand(), Expected);
      std::string Name =
          ("air.atomic." +
           Twine(AS == metal::AS::Threadgroup ? "local" : "global") +
           ".cmpxchg.weak.i32")
              .str();
      FunctionCallee F = M.getOrInsertFunction(
          Name, FunctionType::get(I32,
                                  {Ptr->getType(), Expected->getType(), I32,
                                   I32, I32, I32, I1},
                                  false));
      CallInst *Old = B.CreateCall(
          F, {Ptr, Expected, CX->getNewValOperand(), B.getInt32(OrderRelaxed),
              B.getInt32(OrderRelaxed), B.getInt32(Scope), B.getTrue()});
      if (Ordered) {
        emitFence(B, M, Scope);
        if (OrderingHints)
          addOrderingHint(Old, {CX->getSuccessOrdering(),
                                CX->getFailureOrdering()});
      }
      Value *Success = B.CreateICmpEQ(Old, CX->getCompareOperand());
      Value *Result = B.CreateInsertValue(PoisonValue::get(CX->getType()), Old, 0);
      Result = B.CreateInsertValue(Result, Success, 1);
      CX->replaceAllUsesWith(Result);
      CX->eraseFromParent();
      Changed = true;
      continue;
    }

    auto *RMW = cast<AtomicRMWInst>(I);
    unsigned AddrSpace =
        RMW->getPointerOperand()->getType()->getPointerAddressSpace();
    bool IsGlobal = AddrSpace != metal::AS::Threadgroup;

    Type *ValTy = RMW->getValOperand()->getType();
    bool IsFloat = ValTy->isFloatTy();
    if (!(ValTy->isFloatTy() || ValTy->isIntegerTy(32)))
      continue;
    std::string Name = buildAtomicName(IsGlobal, RMW->getOperation(), IsFloat);
    if (Name.empty())
      continue;

    unsigned Scope = getScope(Ctx, AddrSpace, RMW->getSyncScopeID());
    bool Ordered = isOrdered(RMW->getOrdering());
    auto *PtrTy = PointerType::get(Ctx, AddrSpace);
    FunctionType *FTy =
        FunctionType::get(ValTy, {PtrTy, ValTy, I32, I32, I1}, false);
    FunctionCallee FC = M.getOrInsertFunction(Name, FTy);

    if (Ordered)
      emitFence(B, M, Scope);
    CallInst *Result = B.CreateCall(
        FC, {RMW->getPointerOperand(), RMW->getValOperand(),
             B.getInt32(OrderRelaxed), B.getInt32(Scope), B.getTrue()},
        RMW->getName());
    if (Ordered) {
      emitFence(B, M, Scope);
      if (OrderingHints)
        addOrderingHint(Result, RMW->getOrdering());
    }
    RMW->replaceAllUsesWith(Result);
    RMW->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

PreservedAnalyses AIRLowerAtomicRMWPass::run(Module &M,
                                               ModuleAnalysisManager &AM) {
  return lowerAtomicRMW(M, /*OrderingHints=*/false) ? PreservedAnalyses::none()
                           : PreservedAnalyses::all();
}

bool AIRLowerAtomicRMWLegacy::runOnModule(Module &M) {
  return lowerAtomicRMW(M, OrderingHints);
}

char AIRLowerAtomicRMWLegacy::ID = 0;

INITIALIZE_PASS(AIRLowerAtomicRMWLegacy, DEBUG_TYPE, "AIR Lower AtomicRMW",
                false, false)

ModulePass *llvm::createAIRLowerAtomicRMWLegacyPass(bool OrderingHints) {
  return new AIRLowerAtomicRMWLegacy(OrderingHints);
}
