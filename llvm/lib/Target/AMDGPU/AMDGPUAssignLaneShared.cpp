#if LLPC_BUILD_NPI
//===----- AMDGPUAssignLaneShared.cpp - assign lane-shared offsets ------- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass looks over all the lane-shared global variables, and give them
/// either VGPR assignment or scratch assignment.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUMemoryUtils.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;
using namespace AMDGPU;

#define DEBUG_TYPE "amdgpu-assign-laneshared"

namespace {
class AMDGPUAssignLaneShared {
  unsigned MaxLaneSharedVGPRs;

public:
  AMDGPUAssignLaneShared(unsigned MaxLSVGPRs) {
    MaxLaneSharedVGPRs = MaxLSVGPRs;
  }

  bool runOnModule(Module &M);
};

static void
recordLaneSharedAbsoluteAddress(Module *M, GlobalVariable *GV, uint32_t Offset,
                                DenseSet<Value *> *PtrSet = nullptr) {
  // Use the 28th bit to indicate VGPR.
  uint32_t Address = PtrSet ? (Offset | (1 << 28)) : Offset;
  // Write the specified address into metadata where it can be retrieved by
  // the assembler. Format is a half open range, [Address Address+1)
  LLVMContext &Ctx = M->getContext();
  auto *IntTy = M->getDataLayout().getIntPtrType(Ctx, AMDGPUAS::LANE_SHARED);
  auto *MinC = ConstantAsMetadata::get(ConstantInt::get(IntTy, Address));
  auto *MaxC = ConstantAsMetadata::get(ConstantInt::get(IntTy, Address + 1));
  GV->setMetadata(LLVMContext::MD_absolute_symbol,
                  MDNode::get(Ctx, {MinC, MaxC}));
  if (PtrSet) {
    // Set the metadata for all the pointers to this GV
    // to facilitate the VGPR-promotion during Instruction selection.
    MDNode *EmptyMD = MDNode::get(Ctx, {});
    for (Value *Ptr : *PtrSet) {
      if (auto *Inst = dyn_cast<Instruction>(Ptr))
        Inst->setMetadata("lane-shared-in-vgpr", EmptyMD);
    }
  }
}

static Function *findUseFunction(User *U) {
  if (auto *I = dyn_cast<Instruction>(U))
    return I->getFunction();
  if (auto *C = dyn_cast<ConstantExpr>(U)) {
    for (User *UC : C->users())
      if (Function *F = findUseFunction(UC))
        return F;
  }
  return nullptr;
}

bool AMDGPUAssignLaneShared::runOnModule(Module &M) {
  // Collect all lane-shared global variables.
  SmallVector<GlobalVariable *> LaneSharedGlobals;
  for (auto &GV : M.globals()) {
    if (GV.getAddressSpace() != AMDGPUAS::LANE_SHARED)
      continue;
    if (!GV.isAbsoluteSymbolRef())
      LaneSharedGlobals.push_back(&GV);
    else if (!LaneSharedGlobals.empty())
      report_fatal_error(
          "Module cannot have a mix of absolute and non-absolute "
          "lane-shared global variables");
  }
  if (LaneSharedGlobals.empty())
    return false;

  DenseSet<Function *> WavegroupKernels;
  for (auto &F : M.functions()) {
    if (isKernel(F.getCallingConv()) && getWavegroupEnable(F))
      WavegroupKernels.insert(&F);
  }
  if (WavegroupKernels.empty())
    report_fatal_error(
        "Module has lane-shared variables but no wavegroup kernel");

  // Find the direct usage of lane-shared variables.
  FunctionVariableMap Func2GVs;
  for (auto *GV : LaneSharedGlobals) {
    for (User *V : GV->users()) {
      if (Function *F = findUseFunction(V)) {
        if (isKernel(F->getCallingConv())) {
          if (getWavegroupEnable(*F)) {
            Func2GVs[F].insert(GV);
          } else
            report_fatal_error("Lane-shared variable used in non-wavegroup "
                               "kernel");
        } else
          Func2GVs[F].insert(GV);
      }
    }
  }

  // Collect variables that are used by functions whose address has escaped.
  DenseSet<GlobalVariable *> FuzzyUsedGVs;
  for (auto &K : Func2GVs) {
    Function *F = K.first;
    if (F->hasAddressTaken(nullptr,
                           /* IgnoreCallbackUses */ false,
                           /* IgnoreAssumeLikeCalls */ false,
                           /* IgnoreLLVMUsed */ true,
                           /* IgnoreArcAttachedCall */ false)) {
      set_union(FuzzyUsedGVs, K.second);
    }
  }

  CallGraph CG = CallGraph(M);
  auto FunctionMakesUnknownCall = [&](const Function *F) -> bool {
    assert(!F->isDeclaration());
    for (const CallGraphNode::CallRecord &R : *CG[F]) {
      if (!R.second->getFunction())
        return true;
    }
    return false;
  };
  // If the function makes any unknown call, assume the worst case that it can
  // access all variables accessed by functions whose address escaped.
  for (auto &K : Func2GVs) {
    Function *F = K.first;
    if (FunctionMakesUnknownCall(F))
      set_union(K.second, FuzzyUsedGVs);
  }
  // For each GV, find the set of wavegroup kernel that uses it directly or
  // indirectly.
  VariableFunctionMap GV2Kernels;
  for (Function *Kernel : WavegroupKernels) {
    DenseSet<Function *> Seen; // catches cycles
    SmallVector<Function *, 4> WorkList = {Kernel};
    while (!WorkList.empty()) {
      Function *F = WorkList.pop_back_val();
      for (GlobalVariable *GV : Func2GVs[F]) {
        GV2Kernels[GV].insert(Kernel);
      }

      for (const CallGraphNode::CallRecord &R : *CG[F]) {
        Function *Ith = R.second->getFunction();
        if (Ith) {
          if (!Seen.contains(Ith)) {
            Seen.insert(Ith);
            WorkList.push_back(Ith);
          }
        }
      }
    }
  }
  // Find lane-shared GVs that can be promoted into VGPRs.
  SmallVector<GlobalVariable *> GVsInVGPR;
  SmallVector<GlobalVariable *> GVsInOverflow;
  SmallVector<GlobalVariable *> GVsInScratch;
  DenseMap<GlobalVariable *, DenseSet<Value *>> GVPtrSets;
  for (auto *GV : LaneSharedGlobals) {
    DenseSet<Value *> Pointers;
    if (MaxLaneSharedVGPRs > 0 &&
        IsPromotableToVGPR(*GV, M.getDataLayout(), Pointers)) {
      GVsInVGPR.push_back(GV);
      GVPtrSets[GV] = std::move(Pointers);
    } else {
      GVsInScratch.push_back(GV);
    }
  }
  // GV that gets used in multiple kernels should get a module absolute address
  // either in VGPRs or in scratch. GV that gets used in only one kernel then
  // gets a kernel relative address on top of the module absolute address.
  unsigned LaneSharedVGPRSize = 0;
  auto DL = M.getDataLayout();
  for (auto *GV : GVsInVGPR) {
    if (GV2Kernels.find(GV) == GV2Kernels.end())
      continue;
    if (GV2Kernels[GV].size() <= 1)
      continue;
    unsigned GVBytes = DL.getTypeAllocSize(GV->getValueType());
    if (LaneSharedVGPRSize + GVBytes > MaxLaneSharedVGPRs * 4) {
      GVsInOverflow.push_back(GV);
      continue;
    }
    recordLaneSharedAbsoluteAddress(&M, GV, LaneSharedVGPRSize, &GVPtrSets[GV]);

    LaneSharedVGPRSize = alignTo(LaneSharedVGPRSize + GVBytes, 4u);
  }
  DenseMap<Function *, unsigned> Kernel2Offset;
  for (auto *GV : GVsInVGPR) {
    if (GV2Kernels.find(GV) == GV2Kernels.end())
      continue;
    if (GV2Kernels[GV].size() != 1)
      continue;
    Function *Kernel = *GV2Kernels[GV].begin();
    if (Kernel2Offset.find(Kernel) == Kernel2Offset.end()) {
      Kernel2Offset[Kernel] = LaneSharedVGPRSize;
    }
    unsigned GVBytes = DL.getTypeAllocSize(GV->getValueType());
    if (Kernel2Offset[Kernel] + GVBytes > MaxLaneSharedVGPRs * 4) {
      GVsInOverflow.push_back(GV);
      continue;
    }
    recordLaneSharedAbsoluteAddress(&M, GV, Kernel2Offset[Kernel],
                                    &GVPtrSets[GV]);

    Kernel2Offset[Kernel] = alignTo(Kernel2Offset[Kernel] + GVBytes, 4u);
  }

  GVsInOverflow.insert(GVsInOverflow.end(), GVsInScratch.begin(),
                       GVsInScratch.end());
  // Perform the similar assignment for GVsInOverflow to scratch.
  unsigned LaneSharedScratchSize = 0;
  Kernel2Offset.clear();
  for (auto *GV : GVsInOverflow) {
    if (GV2Kernels.find(GV) == GV2Kernels.end())
      continue;
    if (GV2Kernels[GV].size() <= 1)
      continue;
    unsigned GVBytes = DL.getTypeAllocSize(GV->getValueType());
    recordLaneSharedAbsoluteAddress(&M, GV, LaneSharedScratchSize);

    Align Alignment =
        DL.getValueOrABITypeAlignment(GV->getAlign(), GV->getValueType());
    LaneSharedScratchSize = alignTo(LaneSharedScratchSize, Alignment);
    LaneSharedScratchSize = LaneSharedScratchSize + GVBytes;
  }
  for (auto *GV : GVsInOverflow) {
    if (GV2Kernels.find(GV) == GV2Kernels.end())
      continue;
    if (GV2Kernels[GV].size() != 1)
      continue;
    Function *Kernel = *GV2Kernels[GV].begin();
    if (Kernel2Offset.find(Kernel) == Kernel2Offset.end()) {
      Kernel2Offset[Kernel] = LaneSharedScratchSize;
    }
    unsigned GVBytes = DL.getTypeAllocSize(GV->getValueType());
    recordLaneSharedAbsoluteAddress(&M, GV, Kernel2Offset[Kernel]);

    Align Alignment =
        DL.getValueOrABITypeAlignment(GV->getAlign(), GV->getValueType());
    Kernel2Offset[Kernel] = alignTo(Kernel2Offset[Kernel], Alignment);
    Kernel2Offset[Kernel] = Kernel2Offset[Kernel] + GVBytes;
  }

  return true;
}

class AMDGPUAssignLaneSharedLegacy : public ModulePass {
public:
  unsigned MaxLaneSharedVGPRs;
  static char ID;

  AMDGPUAssignLaneSharedLegacy(unsigned MaxLS)
      : ModulePass(ID), MaxLaneSharedVGPRs(MaxLS) {
    initializeAMDGPUAssignLaneSharedLegacyPass(
        *PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {}

  bool runOnModule(Module &M) override {
    return AMDGPUAssignLaneShared(MaxLaneSharedVGPRs).runOnModule(M);
  }
};

} // namespace

char AMDGPUAssignLaneSharedLegacy::ID = 0;

char &llvm::AMDGPUAssignLaneSharedLegacyPassID =
    AMDGPUAssignLaneSharedLegacy::ID;

INITIALIZE_PASS_BEGIN(AMDGPUAssignLaneSharedLegacy, DEBUG_TYPE,
                      "Assign scratch or vgpr offset for laneshared", false,
                      false)
INITIALIZE_PASS_END(AMDGPUAssignLaneSharedLegacy, DEBUG_TYPE,
                    "Assign scratch or vgpr offset for laneshared", false,
                    false)

ModulePass *
llvm::createAMDGPUAssignLaneSharedLegacyPass(unsigned MaxLaneSharedVGPRs) {
  return new AMDGPUAssignLaneSharedLegacy(MaxLaneSharedVGPRs);
}

PreservedAnalyses AMDGPUAssignLaneSharedPass::run(Module &M,
                                                  ModuleAnalysisManager &) {
  return AMDGPUAssignLaneShared(MaxLaneSharedVGPRs).runOnModule(M)
             ? PreservedAnalyses::none()
             : PreservedAnalyses::all();
}
#endif /* LLPC_BUILD_NPI */
