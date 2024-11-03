//===- DXILResourceAccess.cpp - Resource access via load/store ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILResourceAccess.h"
#include "DirectX.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "dxil-resource-access"

using namespace llvm;

static void replaceTypedBufferAccess(IntrinsicInst *II,
                                     dxil::ResourceInfo &RI) {
  const DataLayout &DL = II->getDataLayout();

  auto *HandleType = cast<TargetExtType>(II->getOperand(0)->getType());
  assert(HandleType->getName() == "dx.TypedBuffer" &&
         "Unexpected typed buffer type");
  Type *ContainedType = HandleType->getTypeParameter(0);
  Type *ScalarType = ContainedType->getScalarType();
  uint64_t ScalarSize = DL.getTypeSizeInBits(ScalarType) / 8;
  int NumElements = ContainedType->getNumContainedTypes();
  if (!NumElements)
    NumElements = 1;

  // Process users keeping track of indexing accumulated from GEPs.
  struct AccessAndIndex {
    User *Access;
    Value *Index;
  };
  SmallVector<AccessAndIndex> Worklist;
  for (User *U : II->users())
    Worklist.push_back({U, nullptr});

  SmallVector<Instruction *> DeadInsts;
  while (!Worklist.empty()) {
    AccessAndIndex Current = Worklist.back();
    Worklist.pop_back();

    if (auto *GEP = dyn_cast<GetElementPtrInst>(Current.Access)) {
      IRBuilder<> Builder(GEP);

      Value *Index;
      APInt ConstantOffset(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
      if (GEP->accumulateConstantOffset(DL, ConstantOffset)) {
        APInt Scaled = ConstantOffset.udiv(ScalarSize);
        Index = ConstantInt::get(Builder.getInt32Ty(), Scaled);
      } else {
        auto IndexIt = GEP->idx_begin();
        assert(cast<ConstantInt>(IndexIt)->getZExtValue() == 0 &&
               "GEP is not indexing through pointer");
        ++IndexIt;
        Index = *IndexIt;
        assert(++IndexIt == GEP->idx_end() && "Too many indices in GEP");
      }

      for (User *U : GEP->users())
        Worklist.push_back({U, Index});
      DeadInsts.push_back(GEP);

    } else if (auto *SI = dyn_cast<StoreInst>(Current.Access)) {
      assert(SI->getValueOperand() != II && "Pointer escaped!");
      IRBuilder<> Builder(SI);

      Value *V = SI->getValueOperand();
      if (V->getType() == ContainedType) {
        // V is already the right type.
      } else if (V->getType() == ScalarType) {
        // We're storing a scalar, so we need to load the current value and only
        // replace the relevant part.
        auto *Load = Builder.CreateIntrinsic(
            ContainedType, Intrinsic::dx_typedBufferLoad,
            {II->getOperand(0), II->getOperand(1)});
        // If we have an offset from seeing a GEP earlier, use it.
        Value *IndexOp = Current.Index
                             ? Current.Index
                             : ConstantInt::get(Builder.getInt32Ty(), 0);
        V = Builder.CreateInsertElement(Load, V, IndexOp);
      } else {
        llvm_unreachable("Store to typed resource has invalid type");
      }

      auto *Inst = Builder.CreateIntrinsic(
          Builder.getVoidTy(), Intrinsic::dx_typedBufferStore,
          {II->getOperand(0), II->getOperand(1), V});
      SI->replaceAllUsesWith(Inst);
      DeadInsts.push_back(SI);

    } else if (auto *LI = dyn_cast<LoadInst>(Current.Access)) {
      IRBuilder<> Builder(LI);
      Value *V =
          Builder.CreateIntrinsic(ContainedType, Intrinsic::dx_typedBufferLoad,
                                  {II->getOperand(0), II->getOperand(1)});
      if (Current.Index)
        V = Builder.CreateExtractElement(V, Current.Index);

      LI->replaceAllUsesWith(V);
      DeadInsts.push_back(LI);

    } else
      llvm_unreachable("Unhandled instruction - pointer escaped?");
  }

  // Traverse the now-dead instructions in RPO and remove them.
  for (Instruction *Dead : llvm::reverse(DeadInsts))
    Dead->eraseFromParent();
  II->eraseFromParent();
}

static bool transformResourcePointers(Function &F, DXILResourceMap &DRM) {
  // TODO: Should we have a more efficient way to find resources used in a
  // particular function?
  SmallVector<std::pair<IntrinsicInst *, dxil::ResourceInfo &>> Resources;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        auto It = DRM.find(CI);
        if (It == DRM.end())
          continue;
        for (User *U : CI->users())
          if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(U))
            if (II->getIntrinsicID() == Intrinsic::dx_resource_getpointer)
              Resources.emplace_back(II, *It);
      }

  for (const auto &[II, RI] : Resources) {
    if (RI.isTyped())
      replaceTypedBufferAccess(II, RI);

    // TODO: handle other resource types. We should probably have an
    // `unreachable` here once we've added support for all of them.
  }

  return false;
}

PreservedAnalyses DXILResourceAccess::run(Function &F,
                                          FunctionAnalysisManager &FAM) {
  auto &MAMProxy = FAM.getResult<ModuleAnalysisManagerFunctionProxy>(F);
  DXILResourceMap *DRM =
      MAMProxy.getCachedResult<DXILResourceAnalysis>(*F.getParent());
  assert(DRM && "DXILResourceAnalysis must be available");

  bool MadeChanges = transformResourcePointers(F, *DRM);
  if (!MadeChanges)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<DXILResourceAnalysis>();
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}

namespace {
class DXILResourceAccessLegacy : public FunctionPass {
public:
  bool runOnFunction(Function &F) override {
    DXILResourceMap &DRM =
        getAnalysis<DXILResourceWrapperPass>().getResourceMap();

    return transformResourcePointers(F, DRM);
  }
  StringRef getPassName() const override { return "DXIL Resource Access"; }
  DXILResourceAccessLegacy() : FunctionPass(ID) {}

  static char ID; // Pass identification.
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<DXILResourceWrapperPass>();
    AU.addPreserved<DXILResourceWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
  }
};
char DXILResourceAccessLegacy::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS_BEGIN(DXILResourceAccessLegacy, DEBUG_TYPE,
                      "DXIL Resource Access", false, false)
INITIALIZE_PASS_DEPENDENCY(DXILResourceWrapperPass)
INITIALIZE_PASS_END(DXILResourceAccessLegacy, DEBUG_TYPE,
                    "DXIL Resource Access", false, false)

FunctionPass *llvm::createDXILResourceAccessLegacyPass() {
  return new DXILResourceAccessLegacy();
}
