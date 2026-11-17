//===- SPIRVLowerResourceNames.cpp - Lower resource names to metadata
//-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "spirv-lower-resource-names"

static bool lowerResourceNames(Module &M) {
  SmallPtrSet<GlobalVariable *, 8> Names;
  bool Changed = false;
  for (Function &F : make_early_inc_range(M)) {
    Intrinsic::ID MdID;
    switch (F.getIntrinsicID()) {
    case Intrinsic::spv_resource_handlefrombinding:
      MdID = Intrinsic::spv_resource_handlefrombinding_mdname;
      break;
    case Intrinsic::spv_resource_handlefromimplicitbinding:
      MdID = Intrinsic::spv_resource_handlefromimplicitbinding_mdname;
      break;
    default:
      continue;
    }
    if (F.use_empty())
      continue;
    Function *MdFn =
        Intrinsic::getOrInsertDeclaration(&M, MdID, {F.getReturnType()});
    for (User *U : make_early_inc_range(F.users())) {
      auto *CI = cast<CallInst>(U);
      Value *NamePtr = CI->getArgOperand(4);
      StringRef Name;
      if (!isa<ConstantPointerNull>(NamePtr) &&
          !getConstantStringInfo(NamePtr, Name))
        report_fatal_error("expected a constant resource name");
      if (auto *GV = dyn_cast<GlobalVariable>(getUnderlyingObject(NamePtr)))
        Names.insert(GV);
      LLVMContext &Ctx = M.getContext();
      auto *NameMD = MDNode::get(Ctx, MDString::get(Ctx, Name));
      CI->setCalledFunction(MdFn);
      CI->setArgOperand(4, MetadataAsValue::get(Ctx, NameMD));
      CI->setAttributes(CI->getAttributes().removeParamAttributes(Ctx, 4));
      Changed = true;
    }
    F.eraseFromParent();
  }
  for (GlobalVariable *GV : Names) {
    GV->removeDeadConstantUsers();
    if (GV->hasLocalLinkage() && GV->use_empty())
      GV->eraseFromParent();
  }
  return Changed;
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
