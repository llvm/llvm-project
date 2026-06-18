//===- AIRDeviceLoadsVolatile.cpp - Mark loop device loads volatile -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRDeviceLoadsVolatile.h"
#include "AIR.h"
#include "AIRAddressSpaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "air-device-loads-volatile"

// AIR device address space.

static bool isDeviceLoad(const Instruction *I) {
  if (auto *LI = dyn_cast<LoadInst>(I))
    return LI->getPointerAddressSpace() == metal::AS::Device;
  return false;
}

static bool isDeviceStore(const Instruction *I) {
  if (auto *SI = dyn_cast<StoreInst>(I))
    return SI->getPointerAddressSpace() == metal::AS::Device;
  return false;
}

// Collect the pointer operand of every cmpxchg-named call in F.
static void collectCASPtrs(Function &F,
                           SmallPtrSetImpl<const Value *> &CasPtrs) {
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (const Function *Callee = CI->getCalledFunction())
          if (Callee->getName().contains("cmpxchg"))
            if (CI->arg_size() >= 1)
              CasPtrs.insert(CI->getArgOperand(0));
}

// True if Ptr transparently derives from any value in CasPtrs via
// GEP / bitcast / addrspacecast / ptrtoint+inttoptr / unique select+phi.
// Mirrors the reach walker in AIRAliasAnnotate.cpp.
static bool reachesCASPtr(const Value *V,
                          const SmallPtrSetImpl<const Value *> &CasPtrs,
                          SmallPtrSetImpl<const Value *> &Seen) {
  if (!Seen.insert(V).second)
    return false;
  if (CasPtrs.count(V))
    return true;
  if (auto *GEP = dyn_cast<GEPOperator>(V))
    return reachesCASPtr(GEP->getPointerOperand(), CasPtrs, Seen);
  if (auto *BC = dyn_cast<BitCastOperator>(V))
    return reachesCASPtr(BC->getOperand(0), CasPtrs, Seen);
  if (auto *ASC = dyn_cast<AddrSpaceCastOperator>(V))
    return reachesCASPtr(ASC->getPointerOperand(), CasPtrs, Seen);
  if (auto *I2P = dyn_cast<IntToPtrInst>(V))
    if (auto *P2I = dyn_cast<PtrToIntInst>(I2P->getOperand(0)))
      return reachesCASPtr(P2I->getPointerOperand(), CasPtrs, Seen);
  if (auto *Sel = dyn_cast<SelectInst>(V))
    return reachesCASPtr(Sel->getTrueValue(), CasPtrs, Seen) ||
           reachesCASPtr(Sel->getFalseValue(), CasPtrs, Seen);
  if (auto *PN = dyn_cast<PHINode>(V)) {
    for (unsigned I = 0; I < PN->getNumIncomingValues(); I++)
      if (reachesCASPtr(PN->getIncomingValue(I), CasPtrs, Seen))
        return true;
  }
  return false;
}

static bool reachesCASPtr(const Value *V,
                          const SmallPtrSetImpl<const Value *> &CasPtrs) {
  SmallPtrSet<const Value *, 16> Seen;
  return reachesCASPtr(V, CasPtrs, Seen);
}

static Value *accessPtr(Instruction *I) {
  if (auto *LI = dyn_cast<LoadInst>(I))
    return LI->getPointerOperand();
  if (auto *SI = dyn_cast<StoreInst>(I))
    return SI->getPointerOperand();
  return nullptr;
}

static bool derivesFromKernelArg(const Value *V) {
  while (true) {
    if (isa<Argument>(V))
      return true;
    if (auto *GEP = dyn_cast<GEPOperator>(V)) {
      V = GEP->getPointerOperand();
      continue;
    }
    if (auto *BC = dyn_cast<BitCastOperator>(V)) {
      V = BC->getOperand(0);
      continue;
    }
    if (auto *ASC = dyn_cast<AddrSpaceCastOperator>(V)) {
      V = ASC->getPointerOperand();
      continue;
    }
    return false;
  }
}

static std::string coherentTypeSuffix(Type *Ty) {
  auto scalarSuffix = [](Type *T) -> std::string {
    if (T->isFloatTy())
      return "f32";
    if (T->isHalfTy())
      return "f16";
    if (auto *IT = dyn_cast<IntegerType>(T)) {
      unsigned BW = IT->getBitWidth();
      if (BW == 8 || BW == 16 || BW == 32 || BW == 64)
        return "i" + std::to_string(BW);
    }
    return std::string();
  };
  if (auto *VT = dyn_cast<FixedVectorType>(Ty)) {
    std::string Elem = scalarSuffix(VT->getElementType());
    if (Elem.empty())
      return std::string();
    return "v" + std::to_string(VT->getNumElements()) + Elem;
  }
  return scalarSuffix(Ty);
}

static bool makeCoherent(Instruction *I) {
  Module *M = I->getModule();
  LLVMContext &Ctx = M->getContext();
  PointerType *PtrTy = PointerType::get(Ctx, metal::AS::Device);

  if (auto *LI = dyn_cast<LoadInst>(I)) {
    Type *Ty = LI->getType();
    std::string Suf = coherentTypeSuffix(Ty);
    if (Suf.empty()) {
      LI->setVolatile(true);
      return true;
    }
    std::string Name = "air.load.device_coherent." + Suf + ".p1" + Suf;
    FunctionType *FTy = FunctionType::get(Ty, {PtrTy}, false);
    FunctionCallee FC = M->getOrInsertFunction(Name, FTy);
    IRBuilder<> B(LI);
    CallInst *Call = B.CreateCall(FC, {LI->getPointerOperand()});
    LI->replaceAllUsesWith(Call);
    LI->eraseFromParent();
    return true;
  }
  if (auto *SI = dyn_cast<StoreInst>(I)) {
    Type *Ty = SI->getValueOperand()->getType();
    std::string Suf = coherentTypeSuffix(Ty);
    if (Suf.empty()) {
      SI->setVolatile(true);
      return true;
    }
    std::string Name = "air.store.device_coherent." + Suf + ".p1" + Suf;
    FunctionType *FTy =
        FunctionType::get(Type::getVoidTy(Ctx), {Ty, PtrTy}, false);
    FunctionCallee FC = M->getOrInsertFunction(Name, FTy);
    IRBuilder<> B(SI);
    B.CreateCall(FC, {SI->getValueOperand(), SI->getPointerOperand()});
    SI->eraseFromParent();
    return true;
  }
  return false;
}

static bool deviceLoadsVolatile(Module &M) {
  bool Changed = false;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    // CAS atomics: lock-word accesses (pointer derives from a cmpxchg arg) only
    // need volatile. For a true scalar spinlock (a CAS whose pointer is a bare
    // kernel argument), the lock-guarded foreign buffer also needs to bypass the
    // per-threadgroup cache: rewrite those argument-derived accesses to the
    // device_coherent load/store intrinsics (volatile alone does not flush the
    // AGX cache, so cross-threadgroup fences lose the critical section). A
    // per-element CAS / RMW CAS-loop is not a spinlock and is left untouched.
    SmallPtrSet<const Value *, 4> CasPtrs;
    collectCASPtrs(F, CasPtrs);
    if (!CasPtrs.empty()) {
      bool HasScalarLock = false;
      for (const Value *P : CasPtrs)
        if (isa<Argument>(P))
          HasScalarLock = true;

      SmallVector<Instruction *, 16> ToCoherent;
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          Value *Ptr = accessPtr(&I);
          if (!Ptr || !(isDeviceLoad(&I) || isDeviceStore(&I)))
            continue;
          if (reachesCASPtr(Ptr, CasPtrs)) {
            if (auto *LI = dyn_cast<LoadInst>(&I))
              LI->setVolatile(true);
            else
              cast<StoreInst>(&I)->setVolatile(true);
            Changed = true;
          } else if (HasScalarLock && derivesFromKernelArg(Ptr)) {
            ToCoherent.push_back(&I);
          }
        }
      }
      for (Instruction *I : ToCoherent)
        Changed |= makeCoherent(I);
      continue;
    }

    DominatorTree DT(F);
    LoopInfo LI(DT);

    for (Loop *L : LI.getLoopsInPreorder()) {
      SmallPtrSet<Value *, 8> StoredPtrs;
      for (BasicBlock *BB : L->blocks())
        for (Instruction &I : *BB)
          if (isDeviceStore(&I))
            StoredPtrs.insert(cast<StoreInst>(&I)->getPointerOperand());
      if (StoredPtrs.empty())
        continue;

      for (BasicBlock *BB : L->blocks()) {
        for (Instruction &I : *BB) {
          auto *LdI = dyn_cast<LoadInst>(&I);
          if (LdI && isDeviceLoad(&I) && !LdI->isVolatile() &&
              StoredPtrs.count(LdI->getPointerOperand())) {
            LdI->setVolatile(true);
            Changed = true;
          }
        }
      }
    }
  }

  return Changed;
}

PreservedAnalyses AIRDeviceLoadsVolatilePass::run(Module &M,
                                                    ModuleAnalysisManager &AM) {
  return deviceLoadsVolatile(M) ? PreservedAnalyses::none()
                                : PreservedAnalyses::all();
}

bool AIRDeviceLoadsVolatileLegacy::runOnModule(Module &M) {
  return deviceLoadsVolatile(M);
}

char AIRDeviceLoadsVolatileLegacy::ID = 0;

INITIALIZE_PASS(AIRDeviceLoadsVolatileLegacy, DEBUG_TYPE,
                "AIR Device Loads Volatile", false, false)

ModulePass *llvm::createAIRDeviceLoadsVolatileLegacyPass() {
  return new AIRDeviceLoadsVolatileLegacy();
}
