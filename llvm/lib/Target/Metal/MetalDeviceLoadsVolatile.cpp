//===- MetalDeviceLoadsVolatile.cpp - Mark loop device loads volatile -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MetalDeviceLoadsVolatile.h"
#include "Metal.h"
#include "MetalAddressSpaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "metal-device-loads-volatile"

// Metal device address space.

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
// Mirrors the reach walker in MetalAliasAnnotate.cpp.
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

static bool deviceLoadsVolatile(Module &M) {
  bool Changed = false;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    // CAS atomics: mark device loads/stores volatile only if their pointer
    // derives from a cmpxchg call's pointer operand. Loads inside the CAS
    // critical section that touch unrelated buffers need no volatile.
    SmallPtrSet<const Value *, 4> CasPtrs;
    collectCASPtrs(F, CasPtrs);
    if (!CasPtrs.empty()) {
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          if (auto *LI = dyn_cast<LoadInst>(&I)) {
            if (isDeviceLoad(&I) && !LI->isVolatile() &&
                reachesCASPtr(LI->getPointerOperand(), CasPtrs)) {
              LI->setVolatile(true);
              Changed = true;
            }
          } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
            if (isDeviceStore(&I) && !SI->isVolatile() &&
                reachesCASPtr(SI->getPointerOperand(), CasPtrs)) {
              SI->setVolatile(true);
              Changed = true;
            }
          }
        }
      }
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

PreservedAnalyses MetalDeviceLoadsVolatilePass::run(Module &M,
                                                    ModuleAnalysisManager &AM) {
  return deviceLoadsVolatile(M) ? PreservedAnalyses::none()
                                : PreservedAnalyses::all();
}

bool MetalDeviceLoadsVolatileLegacy::runOnModule(Module &M) {
  return deviceLoadsVolatile(M);
}

char MetalDeviceLoadsVolatileLegacy::ID = 0;

INITIALIZE_PASS(MetalDeviceLoadsVolatileLegacy, DEBUG_TYPE,
                "Metal Device Loads Volatile", false, false)

ModulePass *llvm::createMetalDeviceLoadsVolatileLegacyPass() {
  return new MetalDeviceLoadsVolatileLegacy();
}
