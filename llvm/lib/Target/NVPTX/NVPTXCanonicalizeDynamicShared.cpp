//===-- NVPTXCanonicalizeDynamicShared.cpp - Merge dynamic shared decls ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Every unsized `.extern .shared` array names the start of the same dynamic
// shared memory allocation, so `extern __shared__ int a[]` and
// `extern __shared__ float b[]` are the same address. LLVM sees two distinct
// global variables, which alias analysis treats as disjoint objects, and can
// then reorder or forward a store through @a past a load through @b.
//
// This pass gives all such declarations one underlying object. It redirects
// their uses to a single declaration that carries the strongest alignment and
// the debug info of each source variable, and erases the others unless a used
// list names them.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

using namespace llvm;

static bool isDynamicSharedDecl(const GlobalVariable &GV) {
  const auto *AT = dyn_cast<ArrayType>(GV.getValueType());
  return GV.isDeclaration() && GV.hasExternalLinkage() && AT &&
         AT->getNumElements() == 0 &&
         GV.getAddressSpace() == NVPTX::AddressSpace::Shared;
}

static void restoreUsedList(GlobalVariable *List,
                            ArrayRef<GlobalValue *> Values) {
  if (!List)
    return;
  auto *AT = cast<ArrayType>(List->getValueType());
  SmallVector<Constant *, 8> Init;
  for (GlobalValue *V : Values)
    Init.push_back(ConstantExpr::getPointerBitCastOrAddrSpaceCast(
        V, AT->getElementType()));
  List->setInitializer(ConstantArray::get(AT, Init));
}

static bool canonicalizeDynamicShared(Module &M) {
  SmallVector<GlobalVariable *, 8> DynSharedDecls;
  for (GlobalVariable &GV : M.globals())
    if (isDynamicSharedDecl(GV))
      DynSharedDecls.push_back(&GV);
  if (DynSharedDecls.size() < 2)
    return false;

  // Detach the used lists so that their entries neither count as uses nor get
  // redirected, and rebuild them unchanged at the end.
  SmallVector<GlobalValue *, 8> Used, CompilerUsed;
  GlobalVariable *UsedList = collectUsedGlobalVariables(M, Used, false);
  GlobalVariable *CompilerUsedList =
      collectUsedGlobalVariables(M, CompilerUsed, true);
  if (UsedList)
    UsedList->setInitializer(nullptr);
  if (CompilerUsedList)
    CompilerUsedList->setInitializer(nullptr);
  SmallPtrSet<GlobalValue *, 8> InUsedList(llvm::from_range, Used);
  InUsedList.insert_range(CompilerUsed);

  const DataLayout &DL = M.getDataLayout();
  GlobalVariable *Canonical = nullptr;
  Align MaxAlign;
  bool Changed = false;
  for (GlobalVariable *GV : DynSharedDecls) {
    GV->removeDeadConstantUsers();
    if (GV->use_empty())
      continue;
    MaxAlign = std::max(MaxAlign, DL.getValueOrABITypeAlignment(
                                      GV->getAlign(), GV->getValueType()));
    if (!Canonical) {
      Canonical = GV;
      continue;
    }

    // A write through any name may change what the others read.
    if (!GV->isConstant())
      Canonical->setConstant(false);

    // Each source variable keeps its name and type at the common address.
    SmallVector<DIGlobalVariableExpression *, 4> GVEs;
    GV->getDebugInfo(GVEs);
    for (DIGlobalVariableExpression *GVE : GVEs)
      Canonical->addDebugInfo(GVE);
    GV->eraseMetadata(LLVMContext::MD_dbg);

    GV->replaceAllUsesWith(Canonical);
    if (!InUsedList.contains(GV))
      GV->eraseFromParent();
    Changed = true;
  }
  if (Changed)
    Canonical->setAlignment(MaxAlign);

  restoreUsedList(UsedList, Used);
  restoreUsedList(CompilerUsedList, CompilerUsed);
  return Changed;
}

namespace {
class NVPTXCanonicalizeDynamicSharedLegacyPass : public ModulePass {
public:
  static char ID;
  NVPTXCanonicalizeDynamicSharedLegacyPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override { return canonicalizeDynamicShared(M); }
};
} // namespace

char NVPTXCanonicalizeDynamicSharedLegacyPass::ID = 0;

INITIALIZE_PASS(NVPTXCanonicalizeDynamicSharedLegacyPass,
                "nvptx-canonicalize-dynamic-shared",
                "Merge NVPTX dynamic shared memory declarations", false, false)

ModulePass *llvm::createNVPTXCanonicalizeDynamicSharedLegacyPass() {
  return new NVPTXCanonicalizeDynamicSharedLegacyPass();
}

PreservedAnalyses
NVPTXCanonicalizeDynamicSharedPass::run(Module &M, ModuleAnalysisManager &) {
  if (!canonicalizeDynamicShared(M))
    return PreservedAnalyses::all();
  return PreservedAnalyses::none().preserveSet<CFGAnalyses>();
}
