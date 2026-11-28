//===- SPIRVLowerResourceNames.cpp - Lower resource names to metadata -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

using namespace llvm;

#define DEBUG_TYPE "spirv-lower-resource-names"

static void promoteNameAllocas(ArrayRef<CallInst *> Intrinsics) {
  MapVector<Function *, SmallVector<AllocaInst *, 4>> Allocas;
  SmallPtrSet<AllocaInst *, 8> Seen;
  for (CallInst *CI : Intrinsics) {
    auto *LI = dyn_cast<LoadInst>(CI->getArgOperand(4));
    if (!LI)
      continue;
    auto *AI = dyn_cast<AllocaInst>(LI->getPointerOperand());
    if (AI && !Seen.insert(AI).second)
      continue;
    if (AI && AI->isStaticAlloca() && isAllocaPromotable(AI))
      Allocas[CI->getFunction()].push_back(AI);
    else
      reportFatalInternalError("unresolvable SPIR-V resource name");
  }
  for (auto &[F, Names] : Allocas) {
    DominatorTree DT(*F);
    PromoteMemToReg(Names, DT);
  }
}

static bool lowerResourceNames(Module &M) {
  SmallVector<CallInst *, 8> Intrinsics;
  for (Function &F : M) {
    switch (F.getIntrinsicID()) {
    case Intrinsic::spv_resource_handlefrombinding:
    case Intrinsic::spv_resource_handlefromimplicitbinding:
      for (User *U : F.users()) {
        if (auto *CI = dyn_cast<CallInst>(U))
          Intrinsics.push_back(CI);
      }
      break;
    default:
      break;
    }
  }
  if (Intrinsics.empty())
    return false;

  promoteNameAllocas(Intrinsics);

  SmallPtrSet<GlobalVariable *, 8> Names;
  for (CallInst *CI : Intrinsics) {
    Function *F = CI->getCalledFunction();
    Intrinsic::ID MdID =
        F->getIntrinsicID() == Intrinsic::spv_resource_handlefrombinding
            ? Intrinsic::spv_resource_handlefrombinding_mdname
            : Intrinsic::spv_resource_handlefromimplicitbinding_mdname;
    Function *MdFn =
        Intrinsic::getOrInsertDeclaration(&M, MdID, {F->getReturnType()});
    Value *NamePtr = CI->getArgOperand(4);
    StringRef Name;
    if (!isa<ConstantPointerNull>(NamePtr) &&
        !getConstantStringInfo(NamePtr, Name))
      reportFatalUsageError("expected a constant resource name");
    if (auto *GV = dyn_cast<GlobalVariable>(getUnderlyingObject(NamePtr)))
      Names.insert(GV);
    LLVMContext &Ctx = M.getContext();
    auto *NameMD = MDNode::get(Ctx, MDString::get(Ctx, Name));
    CI->setCalledFunction(MdFn);
    CI->setArgOperand(4, MetadataAsValue::get(Ctx, NameMD));
    CI->setAttributes(CI->getAttributes().removeParamAttributes(Ctx, 4));
  }
  for (GlobalVariable *GV : Names) {
    GV->removeDeadConstantUsers();
    if (GV->hasLocalLinkage() && GV->use_empty())
      GV->eraseFromParent();
  }
  return true;
}

PreservedAnalyses SPIRVLowerResourceNamesPass::run(Module &M,
                                                   ModuleAnalysisManager &AM) {
  return lowerResourceNames(M) ? PreservedAnalyses::none()
                               : PreservedAnalyses::all();
}

namespace {
class SPIRVLowerResourceNamesLegacy : public ModulePass {
public:
  static char ID;
  SPIRVLowerResourceNamesLegacy() : ModulePass(ID) {}
  bool runOnModule(Module &M) override { return lowerResourceNames(M); }
};
char SPIRVLowerResourceNamesLegacy::ID = 0;
} // namespace

INITIALIZE_PASS(SPIRVLowerResourceNamesLegacy, DEBUG_TYPE,
                "SPIRV Lower Resource Names", false, false)

ModulePass *llvm::createSPIRVLowerResourceNamesLegacyPass() {
  return new SPIRVLowerResourceNamesLegacy();
}
