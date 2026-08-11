//===-- NVPTXMarkKernelPtrsGlobal.cpp - Mark kernel pointers as global ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// For CUDA kernels, pointers loaded from byval parameters are known to be in
// global address space. This pass inserts addrspacecast pairs to make that
// explicit, enabling later address-space inference to propagate the global AS.
// It also handles the pattern where a pointer is loaded as an integer and then
// converted via inttoptr.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "NVVMProperties.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/NVPTXAddrSpace.h"

using namespace llvm;
using namespace NVPTXAS;

static void markPointerAsAS(Value *Ptr, unsigned AS) {
  if (Ptr->getType()->getPointerAddressSpace() != ADDRESS_SPACE_GENERIC)
    return;

  BasicBlock::iterator InsertPt;
  if (auto *Arg = dyn_cast<Argument>(Ptr)) {
    InsertPt = Arg->getParent()->getEntryBlock().begin();
  } else {
    InsertPt = ++cast<Instruction>(Ptr)->getIterator();
    assert(InsertPt != InsertPt->getParent()->end() &&
           "We don't call this function with Ptr being a terminator.");
  }

  Instruction *PtrInGlobal = new AddrSpaceCastInst(
      Ptr, PointerType::get(Ptr->getContext(), AS), Ptr->getName(), InsertPt);
  Value *PtrInGeneric = new AddrSpaceCastInst(PtrInGlobal, Ptr->getType(),
                                              Ptr->getName(), InsertPt);
  Ptr->replaceAllUsesWith(PtrInGeneric);
  PtrInGlobal->setOperand(0, Ptr);
}

static void markPointerAsGlobal(Value *Ptr) {
  markPointerAsAS(Ptr, ADDRESS_SPACE_GLOBAL);
}

static void handleIntToPtr(Value &V) {
  if (!all_of(V.users(), [](User *U) { return isa<IntToPtrInst>(U); }))
    return;

  SmallVector<User *, 16> UsersToUpdate(V.users());
  for (User *U : UsersToUpdate)
    markPointerAsGlobal(U);
}

static bool markKernelPtrsGlobal(Function &F) {
  if (!isKernelFunction(F))
    return false;

  // Copying of byval aggregates + SROA may result in pointers being loaded as
  // integers, followed by inttoptr. We mark those as global too, but only if
  // the loaded integer is used exclusively for conversion to a pointer.
  for (auto &I : instructions(F)) {
    auto *LI = dyn_cast<LoadInst>(&I);
    if (!LI)
      continue;

    if (LI->getType()->isPointerTy() || LI->getType()->isIntegerTy()) {
      Value *UO = getUnderlyingObject(LI->getPointerOperand());
      if (auto *Arg = dyn_cast<Argument>(UO)) {
        if (Arg->hasByValAttr()) {
          if (LI->getType()->isPointerTy())
            markPointerAsGlobal(LI);
          else
            handleIntToPtr(*LI);
        }
      }
    }
  }

  for (Argument &Arg : F.args())
    if (Arg.getType()->isIntegerTy())
      handleIntToPtr(Arg);

  return true;
}

namespace {

class NVPTXMarkKernelPtrsGlobalLegacyPass : public FunctionPass {
public:
  static char ID;
  NVPTXMarkKernelPtrsGlobalLegacyPass() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override;
};

} // namespace

INITIALIZE_PASS(NVPTXMarkKernelPtrsGlobalLegacyPass,
                "nvptx-mark-kernel-ptrs-global",
                "NVPTX Mark Kernel Pointers Global", false, false)

bool NVPTXMarkKernelPtrsGlobalLegacyPass::runOnFunction(Function &F) {
  return markKernelPtrsGlobal(F);
}

char NVPTXMarkKernelPtrsGlobalLegacyPass::ID = 0;

FunctionPass *llvm::createNVPTXMarkKernelPtrsGlobalPass() {
  return new NVPTXMarkKernelPtrsGlobalLegacyPass();
}

PreservedAnalyses
NVPTXMarkKernelPtrsGlobalPass::run(Function &F, FunctionAnalysisManager &) {
  return markKernelPtrsGlobal(F) ? PreservedAnalyses::none()
                                 : PreservedAnalyses::all();
}
