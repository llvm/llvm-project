//===-- NVPTXLowerAlloca.cpp - Make alloca to use local memory =====--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// For generic alloca instructions, create an equivalent alloca in local
// address space and rewrite uses that can directly use local memory. For
// example,
//
//   %A = alloca i32
//   store i32 0, i32* %A ; emits st.u32
//
// will be transformed to
//
//   %A = alloca i32, addrspace(5)
//   store i32 0, i32 addrspace(5)* %A ; emits st.local.u32
//
// Uses that require a generic pointer use a single addrspacecast from the
// local alloca.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/NVPTXBaseInfo.h"
#include "NVPTX.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace {
class NVPTXLowerAlloca : public FunctionPass {
  bool runOnFunction(Function &F) override;

public:
  static char ID; // Pass identification, replacement for typeid
  NVPTXLowerAlloca() : FunctionPass(ID) {}
  StringRef getPassName() const override {
    return "convert address space of alloca'ed memory to local";
  }
};
} // namespace

char NVPTXLowerAlloca::ID = 1;

INITIALIZE_PASS(NVPTXLowerAlloca, "nvptx-lower-alloca", "Lower Alloca", false,
                false)

static Value *getOrCreateGenericPtr(Value *LocalPtr,
                                    DenseMap<Value *, Value *> &GenericPtrs) {
  auto It = GenericPtrs.find(LocalPtr);
  if (It != GenericPtrs.end())
    return It->second;

  auto *LocalInst = cast<Instruction>(LocalPtr);
  auto *GenericPtr = new AddrSpaceCastInst(
      LocalPtr, PointerType::get(LocalPtr->getContext(), ADDRESS_SPACE_GENERIC),
      "");
  GenericPtr->insertAfter(LocalInst->getIterator());
  GenericPtrs[LocalPtr] = GenericPtr;
  return GenericPtr;
}

static void updateMemIntrinsicDeclaration(MemIntrinsic *MI) {
  SmallVector<Type *, 3> Tys;
  if (auto *MTI = dyn_cast<MemTransferInst>(MI)) {
    Tys.push_back(MTI->getRawDest()->getType());
    Tys.push_back(MTI->getRawSource()->getType());
    Tys.push_back(MTI->getLength()->getType());
  } else {
    auto *MSI = cast<MemSetInst>(MI);
    Tys.push_back(MSI->getRawDest()->getType());
    Tys.push_back(MSI->getLength()->getType());
  }

  Function *Decl = Intrinsic::getOrInsertDeclaration(MI->getModule(),
                                                     MI->getIntrinsicID(), Tys);
  MI->setCalledFunction(Decl);
}

static void convertPointerUsersToLocal(Value *OldPtr, Value *LocalPtr,
                                       DenseMap<Value *, Value *> &GenericPtrs,
                                       SmallPtrSetImpl<Value *> &Visited) {
  if (!Visited.insert(OldPtr).second)
    return;

  // Debug records aren't on the use-list visited below, so retarget them here;
  // otherwise the variable's location is lost when the old alloca is erased.
  SmallVector<DbgVariableRecord *, 2> DbgUsers;
  findDbgUsers(OldPtr, DbgUsers);
  for (DbgVariableRecord *DVR : DbgUsers)
    DVR->replaceVariableLocationOp(OldPtr, LocalPtr);

  for (Use &U : llvm::make_early_inc_range(OldPtr->uses())) {
    auto *UserInst = dyn_cast<Instruction>(U.getUser());
    if (!UserInst)
      continue;

    if (auto *LI = dyn_cast<LoadInst>(UserInst);
        LI && LI->getPointerOperand() == OldPtr && !LI->isVolatile()) {
      U.set(LocalPtr);
      continue;
    }

    if (auto *SI = dyn_cast<StoreInst>(UserInst);
        SI && SI->getPointerOperand() == OldPtr && !SI->isVolatile()) {
      U.set(LocalPtr);
      continue;
    }

    if (auto *GEP = dyn_cast<GetElementPtrInst>(UserInst);
        GEP && GEP->getPointerOperand() == OldPtr) {
      SmallVector<Value *, 4> Indices(GEP->idx_begin(), GEP->idx_end());
      auto *NewGEP =
          GetElementPtrInst::Create(GEP->getSourceElementType(), LocalPtr,
                                    Indices, "", GEP->getIterator());
      NewGEP->setNoWrapFlags(GEP->getNoWrapFlags());
      NewGEP->copyMetadata(*GEP);
      NewGEP->setDebugLoc(GEP->getDebugLoc());
      convertPointerUsersToLocal(GEP, NewGEP, GenericPtrs, Visited);
      if (GEP->use_empty())
        GEP->eraseFromParent();
      continue;
    }

    if (auto *BC = dyn_cast<BitCastInst>(UserInst);
        BC && BC->getOperand(0) == OldPtr && BC->getType()->isPointerTy()) {
      convertPointerUsersToLocal(BC, LocalPtr, GenericPtrs, Visited);
      if (BC->use_empty())
        BC->eraseFromParent();
      continue;
    }

    if (auto *ASC = dyn_cast<AddrSpaceCastInst>(UserInst);
        ASC && ASC->getOperand(0) == OldPtr &&
        ASC->getDestAddressSpace() == ADDRESS_SPACE_LOCAL) {
      ASC->replaceAllUsesWith(LocalPtr);
      ASC->eraseFromParent();
      continue;
    }

    if (auto *II = dyn_cast<IntrinsicInst>(UserInst);
        II &&
        (II->getIntrinsicID() == Intrinsic::lifetime_start ||
         II->getIntrinsicID() == Intrinsic::lifetime_end) &&
        isa<AllocaInst>(LocalPtr)) {
      // Lifetime markers must reference an alloca directly, so retarget them
      // to the local alloca and update the overloaded declaration.
      U.set(LocalPtr);
      Function *Decl = Intrinsic::getOrInsertDeclaration(
          II->getModule(), II->getIntrinsicID(), {LocalPtr->getType()});
      II->setCalledFunction(Decl);
      continue;
    }

    if (auto *MI = dyn_cast<MemIntrinsic>(UserInst)) {
      if (auto *MTI = dyn_cast<MemTransferInst>(MI)) {
        bool Changed = false;
        if (MTI->getRawDest() == OldPtr) {
          MTI->getRawDestUse().set(LocalPtr);
          Changed = true;
        }
        if (MTI->getRawSource() == OldPtr) {
          MTI->getRawSourceUse().set(LocalPtr);
          Changed = true;
        }
        if (Changed) {
          updateMemIntrinsicDeclaration(MI);
          continue;
        }
      } else if (auto *MSI = dyn_cast<MemSetInst>(MI);
                 MSI && MSI->getRawDest() == OldPtr) {
        MSI->getRawDestUse().set(LocalPtr);
        updateMemIntrinsicDeclaration(MI);
        continue;
      }
    }

    U.set(getOrCreateGenericPtr(LocalPtr, GenericPtrs));
  }
}

// =============================================================================
// Main function for this pass.
// =============================================================================
bool NVPTXLowerAlloca::runOnFunction(Function &F) {
  // Mandatory lowering: later stack lowering relies on local allocas, so run
  // even for optnone functions (skipFunction is intentionally not called).
  bool Changed = false;
  SmallVector<AllocaInst *, 8> Allocas;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *AI = dyn_cast<AllocaInst>(&I))
        Allocas.push_back(AI);
    }
  }

  for (AllocaInst *allocaInst : Allocas) {
    Changed = true;

    unsigned AllocAddrSpace = allocaInst->getAddressSpace();
    assert((AllocAddrSpace == ADDRESS_SPACE_GENERIC ||
            AllocAddrSpace == ADDRESS_SPACE_LOCAL) &&
           "AllocaInst can only be in Generic or Local address space for "
           "NVPTX.");

    Instruction *AllocaInLocalAS = allocaInst;

    // We need to make sure that LLVM has info that alloca needs to go to
    // ADDRESS_SPACE_LOCAL for InferAddressSpace pass.
    //
    // For allocas in ADDRESS_SPACE_GENERIC, create an equivalent local alloca
    // and rewrite localizable users to use it directly. Users that still need a
    // generic pointer use a single addrspacecast from the local alloca.
    //
    // For allocas already in ADDRESS_SPACE_LOCAL, we just need
    // addrspacecast to ADDRESS_SPACE_GENERIC.
    if (AllocAddrSpace == ADDRESS_SPACE_GENERIC) {
      auto *AllocaInLocalAS =
          new AllocaInst(allocaInst->getAllocatedType(), ADDRESS_SPACE_LOCAL,
                         allocaInst->getArraySize(), allocaInst->getAlign(), "",
                         allocaInst->getIterator());
      AllocaInLocalAS->takeName(allocaInst);
      AllocaInLocalAS->setDebugLoc(allocaInst->getDebugLoc());
      AllocaInLocalAS->copyMetadata(*allocaInst);
      AllocaInLocalAS->setUsedWithInAlloca(allocaInst->isUsedWithInAlloca());
      AllocaInLocalAS->setSwiftError(allocaInst->isSwiftError());
      DenseMap<Value *, Value *> GenericPtrs;
      SmallPtrSet<Value *, 8> Visited;
      convertPointerUsersToLocal(allocaInst, AllocaInLocalAS, GenericPtrs,
                                 Visited);
      assert(allocaInst->use_empty() &&
             "All generic alloca uses should have been rewritten.");
      allocaInst->eraseFromParent();
      continue;
    }

    auto AllocaInGenericAS = new AddrSpaceCastInst(
        AllocaInLocalAS,
        PointerType::get(allocaInst->getContext(), ADDRESS_SPACE_GENERIC), "");
    AllocaInGenericAS->insertAfter(AllocaInLocalAS->getIterator());

    for (Use &AllocaUse : llvm::make_early_inc_range(allocaInst->uses())) {
      // Check Load, Store, GEP, and BitCast Uses on alloca and make them
      // use the converted generic address, in order to expose non-generic
      // addrspacecast to NVPTXInferAddressSpaces. For other types
      // of instructions this is unnecessary and may introduce redundant
      // address cast.
      auto LI = dyn_cast<LoadInst>(AllocaUse.getUser());
      if (LI && LI->getPointerOperand() == allocaInst && !LI->isVolatile()) {
        LI->setOperand(LI->getPointerOperandIndex(), AllocaInGenericAS);
        continue;
      }
      auto SI = dyn_cast<StoreInst>(AllocaUse.getUser());
      if (SI && SI->getPointerOperand() == allocaInst && !SI->isVolatile()) {
        SI->setOperand(SI->getPointerOperandIndex(), AllocaInGenericAS);
        continue;
      }
      auto GI = dyn_cast<GetElementPtrInst>(AllocaUse.getUser());
      if (GI && GI->getPointerOperand() == allocaInst) {
        GI->setOperand(GI->getPointerOperandIndex(), AllocaInGenericAS);
        continue;
      }
      auto BI = dyn_cast<BitCastInst>(AllocaUse.getUser());
      if (BI && BI->getOperand(0) == allocaInst) {
        BI->setOperand(0, AllocaInGenericAS);
        continue;
      }
    }
  }
  return Changed;
}

FunctionPass *llvm::createNVPTXLowerAllocaPass() {
  return new NVPTXLowerAlloca();
}
