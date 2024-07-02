//===- AMDGPUMarkPromotableLaneShared.cpp - mark lane-shared promotable -- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass looks over all the lane-shared global variables, and mark those
/// that are suitable for shared-vgpr indexing access.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-mark-promotable-lane-shared"

namespace {
class AMDGPUMarkPromotableLaneShared {
  const DataLayout *DL = nullptr;

public:
  AMDGPUMarkPromotableLaneShared() {}

  bool runOnFunction(Function &F);

private:
  bool checkPromotable(GlobalVariable &GV);
};

bool AMDGPUMarkPromotableLaneShared::runOnFunction(Function &F) {
  auto M = F.getParent();
  DL = &(M->getDataLayout());
  SmallVector<Constant *> LaneSharedGlobals;
  for (auto &GV : M->globals()) {
    if (GV.getAddressSpace() == AMDGPUAS::LANE_SHARED &&
        !GV.hasAttribute("lane-shared-in-vgpr") &&
        !GV.hasAttribute("lane-shared-in-mem"))
      LaneSharedGlobals.push_back(&GV);
  }
  if (LaneSharedGlobals.empty())
    return false;

  bool Changed = false;
  for (auto *GVC : LaneSharedGlobals) {
    GlobalVariable &GV = *cast<GlobalVariable>(GVC);
    if (checkPromotable(GV)) {
      GV.addAttribute("lane-shared-in-vgpr");
      Changed = true;
    } else
      GV.addAttribute("lane-shared-in-mem");
  }
  return Changed;
}

static void collectUses(GlobalVariable &GV, SmallVectorImpl<Use *> &Uses) {
  SmallVector<Instruction *> WorkList;
  for (auto &U : GV.uses()) {
    Uses.push_back(&U);

    if (Instruction *I = dyn_cast<GetElementPtrInst>(U.getUser()))
      WorkList.push_back(I);
  }
  while (!WorkList.empty()) {
    auto *Cur = WorkList.pop_back_val();
    for (auto &U : Cur->uses()) {
      Uses.push_back(&U);

      if (Instruction *I = dyn_cast<GetElementPtrInst>(U.getUser()))
        WorkList.push_back(I);
    }
  }
}

static bool allPtrInputsInSameClass(GlobalVariable *GV, Instruction *Inst) {
  unsigned i = isa<SelectInst>(Inst) ? 1 : 0;
  for (; i < Inst->getNumOperands(); ++i) {
    Value *Op = Inst->getOperand(i);

    if (isa<ConstantPointerNull>(Op))
      continue;

    Value *Obj = getUnderlyingObject(Op);
    if (!isa<GlobalVariable>(Obj))
      return false;

    // TODO-GFX13: if pointers are derived from two different
    // global lane-shared, it should still work. The
    // important part is both must be promotable into vgpr at
    // the end. It will require one more iteration of processing
    if (Obj != GV) {
      LLVM_DEBUG(
          dbgs()
          << "Found a select/phi with ptrs derived from two different GVs\n");
      return false;
    }
  }
  return true;
}

// Checks if the instruction I is a memset user of the global variable that we
// can deal with. Currently, only non-volatile memsets that affect the whole
// global variable are handled.
static bool isSupportedMemset(MemSetInst *I, const GlobalVariable &GV,
                              const DataLayout &DL) {
  using namespace PatternMatch;
  // For now we only care about non-volatile memsets that affect the whole
  // type (start at index 0 and fill the whole global variable).
  const unsigned Size = DL.getTypeStoreSize(GV.getType());
  return I->getOperand(0) == &GV &&
         match(I->getOperand(2), m_SpecificInt(Size)) && !I->isVolatile();
}

// Check if a lane-shared global variable can be stored in VGPRs, and
// accumulate a list of use-insts that need to be marked or transformed.
bool AMDGPUMarkPromotableLaneShared::checkPromotable(GlobalVariable &GV) {

  const auto RejectUser = [&](Instruction *Inst, Twine Msg) {
    LLVM_DEBUG(dbgs() << "  Cannot promote lane-shared to vgpr: " << Msg << "\n"
                      << "    " << *Inst << "\n");
    return false;
  };

  SmallVector<Use *, 8> Uses;
  collectUses(GV, Uses);

  for (auto *U : Uses) {
    Instruction *Inst = dyn_cast<Instruction>(U->getUser());
    if (!Inst)
      continue;

    if (Value *Ptr = getLoadStorePointerOperand(Inst)) {
      // This is a store of the pointer, not to the pointer.
      if (isa<StoreInst>(Inst) &&
          U->getOperandNo() != StoreInst::getPointerOperandIndex())
        return RejectUser(Inst, "pointer is being stored");

      // Check that this is a simple access of a vector element.
      bool IsSimple = isa<LoadInst>(Inst) ? cast<LoadInst>(Inst)->isSimple()
                                          : cast<StoreInst>(Inst)->isSimple();
      if (!IsSimple)
        return RejectUser(Inst, "not a simple load or store");

      auto Align = isa<LoadInst>(Inst) ? cast<LoadInst>(Inst)->getAlign()
                                       : cast<StoreInst>(Inst)->getAlign();
      if (Align < 4u)
        return RejectUser(Inst, "address is less than dword-aligned");

      Type *AccessTy = getLoadStoreType(Inst);
      auto DataSize = DL->getTypeAllocSize(AccessTy);
      if (DataSize % 4)
        return RejectUser(Inst, "data-size is not supported");

      continue;
    }

    if (auto *GEP = dyn_cast<GetElementPtrInst>(Inst)) {
      continue;
    }

    if (auto *Phi = dyn_cast<PHINode>(Inst)) {
      if (allPtrInputsInSameClass(&GV, Inst)) {
        continue;
      }
      return RejectUser(Inst, "phi on ptrs from two different GVs");
    }
    if (auto *Phi = dyn_cast<SelectInst>(Inst)) {
      if (allPtrInputsInSameClass(&GV, Inst)) {
        continue;
      }
      return RejectUser(Inst, "select on ptrs from two different GVs");
    }

    if (MemSetInst *MSI = dyn_cast<MemSetInst>(Inst);
        MSI && isSupportedMemset(MSI, GV, *DL)) {
      continue;
    } else
      return RejectUser(Inst, "cannot handle partial memset inst yet");

    if (MemTransferInst *TransferInst = dyn_cast<MemTransferInst>(Inst))
      return RejectUser(Inst, "cannot handle mem transfer inst yet");

    if (auto *Intr = dyn_cast<IntrinsicInst>(Inst)) {
      if (Intr->getIntrinsicID() == Intrinsic::objectsize) {
        continue;
      }
    }

    // Ignore assume-like intrinsics and comparisons used in assumes.
    if (isAssumeLikeIntrinsic(Inst)) {
      assert(Inst->use_empty() &&
             "does not expect assume-like intrinsic with any user");
      continue;
    }

    if (isa<ICmpInst>(Inst)) {
      if (!all_of(Inst->users(), [](User *U) {
            return isAssumeLikeIntrinsic(cast<Instruction>(U));
          }))
        return RejectUser(Inst, "used in icmp with non-assume-like uses");
      continue;
    }

    return RejectUser(Inst, "unhandled global-variable user");
  }
  return true;
}

class AMDGPUMarkPromotableLaneSharedLegacy : public FunctionPass {
public:
  static char ID;

  AMDGPUMarkPromotableLaneSharedLegacy() : FunctionPass(ID) {
    initializeAMDGPUMarkPromotableLaneSharedLegacyPass(
        *PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {}

  bool runOnFunction(Function &F) override {
    return AMDGPUMarkPromotableLaneShared().runOnFunction(F);
  }
};

} // namespace

char AMDGPUMarkPromotableLaneSharedLegacy::ID = 0;

char &llvm::AMDGPUMarkPromotableLaneSharedLegacyPassID =
    AMDGPUMarkPromotableLaneSharedLegacy::ID;

INITIALIZE_PASS_BEGIN(AMDGPUMarkPromotableLaneSharedLegacy, DEBUG_TYPE,
                      "Mark promotable lane-shared", false, false)
INITIALIZE_PASS_END(AMDGPUMarkPromotableLaneSharedLegacy, DEBUG_TYPE,
                    "Mark promotable lane-shared", false, false)

FunctionPass *llvm::createAMDGPUMarkPromotableLaneSharedLegacyPass() {
  return new AMDGPUMarkPromotableLaneSharedLegacy();
}

PreservedAnalyses
AMDGPUMarkPromotableLaneSharedPass::run(Function &F,
                                        FunctionAnalysisManager &) {
  return AMDGPUMarkPromotableLaneShared().runOnFunction(F)
             ? PreservedAnalyses::none()
             : PreservedAnalyses::all();
}
