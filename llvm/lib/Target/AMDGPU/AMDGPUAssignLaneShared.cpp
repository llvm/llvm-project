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
#include "llvm/IR/IntrinsicsAMDGPU.h"
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

// Assign absolute addresses to lane-shared GVs.
//
// This function is used both for GVs assigned to scratch and for GVs assigned
// to VGPRs. In the latter case, all optional arguments must be specified.
static void assignAbsoluteAddresses(
    SmallVectorImpl<GlobalVariable *> &GVs,
    const VariableFunctionMap &GV2Kernels, uint32_t EndAddress = 1u << 28,
    SmallVectorImpl<GlobalVariable *> *GVsInOverflow = nullptr,
    DenseMap<GlobalVariable *, DenseSet<Value *>> *VGPRPtrSets = nullptr) {
  if (GVs.empty())
    return;

  Module &M = *GVs[0]->getParent();
  LLVMContext &Ctx = M.getContext();
  auto DL = M.getDataLayout();
  bool IsVGPRs = VGPRPtrSets != nullptr;
  auto *IntTy = M.getDataLayout().getIntPtrType(Ctx, AMDGPUAS::LANE_SHARED);

  // Sort the GVs by the number of kernels that use them, so that we assign
  // the GVs that are used by more kernels first. This helps pack lane-shared
  // variables that are used by multiple kernels, since we must assign them a
  // module absolute address.
  llvm::stable_sort(GVs, [&](GlobalVariable *A, GlobalVariable *B) {
    return GV2Kernels.find(A)->second.size() >
           GV2Kernels.find(B)->second.size();
  });

  DenseMap<Function *, uint32_t> Kernel2Offset;
  for (auto *GV : GVs) {
    auto Kernels = GV2Kernels.find(GV);
    assert(Kernels != GV2Kernels.end());

    // Determine the lowest address that we can guarantee does not conflict with
    // already assigned addresses for all relevant kernels.
    uint32_t Address = 0;
    for (auto *Kernel : Kernels->second)
      Address = std::max(Address, Kernel2Offset[Kernel]);

    uint32_t Align = std::max<uint32_t>(GV->getAlignment(), 4u);
    Address = alignTo(Address, Align);

    // Determine the size of the variable.
    uint32_t GVBytes = DL.getTypeAllocSize(GV->getValueType());
    if (Address + GVBytes > EndAddress) {
      if (!GVsInOverflow) {
        report_fatal_error("Lane-shared variable exceeds the maximum address "
                           "space size");
      }
      GVsInOverflow->push_back(GV);
      continue;
    }

    // Update the per-kernel offsets.
    for (auto *Kernel : Kernels->second)
      Kernel2Offset[Kernel] = Address + GVBytes;

    // Write the specified address into metadata where it can be retrieved by
    // the assembler. The metadata represents the half-open range of possible
    // symbol values, i.e. [Address, Address + 1).
    //
    // Use the 28th bit to indicate VGPR.
    if (IsVGPRs)
      Address |= (1 << 28);

    auto *MinC = ConstantAsMetadata::get(ConstantInt::get(IntTy, Address));
    auto *MaxC = ConstantAsMetadata::get(ConstantInt::get(IntTy, Address + 1));
    GV->setMetadata(LLVMContext::MD_absolute_symbol,
                    MDNode::get(Ctx, {MinC, MaxC}));

    if (IsVGPRs) {
      // Set the metadata for all the pointers to this GV
      // to facilitate the VGPR-promotion during Instruction selection.
      MDNode *EmptyMD = MDNode::get(Ctx, {});
      auto PtrSetsIt = VGPRPtrSets->find(GV);
      assert(PtrSetsIt != VGPRPtrSets->end());
      for (Value *Ptr : PtrSetsIt->second) {
        if (auto *Inst = dyn_cast<Instruction>(Ptr))
          Inst->setMetadata("laneshared-in-vgpr", EmptyMD);
      }
    }
  }

  // Record per-kernel allocation bound as metadata.
  unsigned MDKindID = Ctx.getMDKindID(IsVGPRs ? "laneshared-vgpr-size"
                                              : "laneshared-scratch-size");

  for (auto K : Kernel2Offset) {
    Function *Kernel = K.first;
    unsigned Offset = alignTo(K.second, 4);
    Kernel->setMetadata(
        MDKindID, MDNode::get(M.getContext(),
                              {ConstantAsMetadata::get(ConstantInt::get(
                                  Type::getInt32Ty(M.getContext()), Offset))}));
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
  // Validate spatial cluster kernels.
  for (auto& F : M.functions()) {
    if (getSpatialClusterEnable(F)) {
      if (!getWavegroupEnable(F))
        report_fatal_error("Spatial cluster kernel is not wavegroup kernel");
      AMDGPU::ClusterDimsAttr ClusterDims = AMDGPU::ClusterDimsAttr::get(F);
      if (!ClusterDims.isFixedDims())
        report_fatal_error("Spatial cluster kernel has non fixed cluster dims");
      auto& Dims = ClusterDims.getDims();
      if (Dims[1] != 1 || Dims[2] != 1)
        report_fatal_error("Spatial cluster kernel is not 1D");
    }
  }
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
    if (F.isDeclaration())
      continue;
    if (isKernel(F.getCallingConv()) && getWavegroupEnable(F) &&
         !getWavegroupRankFunction(F))
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
  for (Function &F : M) {
    if (!F.isDeclaration() && FunctionMakesUnknownCall(&F))
      set_union(Func2GVs[&F], FuzzyUsedGVs);
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
          if (Ith->isIntrinsic()) {
            if (Ith->getIntrinsicID() == Intrinsic::amdgcn_wavegroup_rank) {
              assert(R.first);
              if (CallInst *CI = dyn_cast<CallInst>(*R.first)) {
                if (auto *Callee = dyn_cast<Function>(CI->getArgOperand(1));
                    Callee && !Seen.contains(Callee)) {
                  Seen.insert(Callee);
                  WorkList.push_back(Callee);
                }
              }
            }
          } else if (!Seen.contains(Ith)) {
            Seen.insert(Ith);
            WorkList.push_back(Ith);
          }
        }
      }
    }
  }

  // Filter out any lane-shared GVs that are never used.
  llvm::erase_if(LaneSharedGlobals, [&](GlobalVariable *GV) {
    return GV2Kernels.find(GV) == GV2Kernels.end();
  });
  if (LaneSharedGlobals.empty())
    return false;

  // Find lane-shared GVs that can be promoted into VGPRs.
  SmallVector<GlobalVariable *> GVsInVGPR;
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

  // Assign VGPRs to GVs and record related metadata. This also records GVs that
  // overflow the available space and must be assigned to scratch.
  assignAbsoluteAddresses(GVsInVGPR, GV2Kernels, MaxLaneSharedVGPRs * 4,
                          &GVsInScratch, &GVPtrSets);

  // Assign remaining GVs to scratch.
  assignAbsoluteAddresses(GVsInScratch, GV2Kernels);

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
