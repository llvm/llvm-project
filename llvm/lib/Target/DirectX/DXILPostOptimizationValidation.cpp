//===- DXILPostOptimizationValidation.cpp - Opt DXIL validation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILPostOptimizationValidation.h"
#include "DXILRootSignature.h"
#include "DXILShaderFlags.h"
#include "DirectX.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "dxil-post-optimization-validation"

using namespace llvm;
using namespace llvm::dxil;

namespace {

static void reportInvalidDirection(Module &M, DXILResourceMap &DRM) {
  for (const auto &UAV : DRM.uavs()) {
    if (UAV.CounterDirection != ResourceCounterDirection::Invalid)
      continue;

    CallInst *ResourceHandle = nullptr;
    for (CallInst *MaybeHandle : DRM.calls()) {
      if (*DRM.find(MaybeHandle) == UAV) {
        ResourceHandle = MaybeHandle;
        break;
      }
    }

    StringRef Message = "RWStructuredBuffers may increment or decrement their "
                        "counters, but not both.";
    for (const auto &U : ResourceHandle->users()) {
      const CallInst *CI = dyn_cast<CallInst>(U);
      if (!CI && CI->getIntrinsicID() != Intrinsic::dx_resource_updatecounter)
        continue;

      M.getContext().diagnose(DiagnosticInfoGenericWithLoc(
          Message, *CI->getFunction(), CI->getDebugLoc()));
    }
  }
}

static void reportOverlappingError(Module &M, ResourceInfo R1,
                                   ResourceInfo R2) {
  SmallString<128> Message;
  raw_svector_ostream OS(Message);
  OS << "resource " << R1.getName() << " at register "
     << R1.getBinding().LowerBound << " overlaps with resource " << R2.getName()
     << " at register " << R2.getBinding().LowerBound << " in space "
     << R2.getBinding().Space;
  M.getContext().diagnose(DiagnosticInfoGeneric(Message));
}

static void reportOverlappingBinding(Module &M, DXILResourceMap &DRM) {
  if (DRM.empty())
    return;

  for (const auto &ResList :
       {DRM.srvs(), DRM.uavs(), DRM.cbuffers(), DRM.samplers()}) {
    if (ResList.empty())
      continue;
    const ResourceInfo *PrevRI = &*ResList.begin();
    for (auto *I = ResList.begin() + 1; I != ResList.end(); ++I) {
      const ResourceInfo *CurrentRI = &*I;
      const ResourceInfo *RI = CurrentRI;
      while (RI != ResList.end() &&
             PrevRI->getBinding().overlapsWith(RI->getBinding())) {
        reportOverlappingError(M, *PrevRI, *RI);
        RI++;
      }
      PrevRI = CurrentRI;
    }
  }
}
  uint64_t combine_uint32_to_uint64(uint32_t high, uint32_t low) {
      return (static_cast<uint64_t>(high) << 32) | low;
  }
static void reportErrors(Module &M, DXILResourceMap &DRM,
                         DXILResourceBindingInfo &DRBI,
                         RootSignatureBindingInfo &RSBI,
                         dxil::ModuleMetadataInfo &MMI) {
  if (DRM.hasInvalidCounterDirection())
    reportInvalidDirection(M, DRM);

  if (DRBI.hasOverlappingBinding())
    reportOverlappingBinding(M, DRM);

  assert(!DRBI.hasImplicitBinding() && "implicit bindings should be handled in "
                                       "DXILResourceImplicitBinding pass");
  // Assuming this is used to validate only the root signature assigned to the
  // entry function.
  //Start test stuff
  if(MMI.EntryPropertyVec.size() == 0)
    return;

  std::optional<mcdxbc::RootSignatureDesc> RootSigDesc =
      RSBI.getDescForFunction(MMI.EntryPropertyVec[0].Entry);
  if (!RootSigDesc)
    return;

  using MapT = llvm::IntervalMap<uint64_t, llvm::dxil::ResourceInfo::ResourceBinding, sizeof(llvm::dxil::ResourceInfo::ResourceBinding), llvm::IntervalMapInfo<uint64_t>>;
  MapT::Allocator Allocator;
  MapT BindingsMap(Allocator);
  auto RSD = *RootSigDesc;
   for (size_t I = 0; I < RSD.ParametersContainer.size(); I++) {
    const auto &[Type, Loc] =
        RootSigDesc->ParametersContainer.getTypeAndLocForParameter(I);
    switch (Type) {
    case llvm::to_underlying(dxbc::RootParameterType::CBV):{
      dxbc::RTS0::v2::RootDescriptor Desc =
          RootSigDesc->ParametersContainer.getRootDescriptor(Loc);

      llvm::dxil::ResourceInfo::ResourceBinding Binding;
      Binding.LowerBound = Desc.ShaderRegister;
      Binding.Space = Desc.RegisterSpace;
      Binding.Size = 1;

      BindingsMap.insert(combine_uint32_to_uint64(Binding.Space, Binding.LowerBound), combine_uint32_to_uint64(Binding.Space, Binding.LowerBound + Binding.Size -1), Binding);
      break;
    }
    // case llvm::to_underlying(dxbc::RootParameterType::DescriptorTable):{
    //   mcdxbc::DescriptorTable Table =
    //       RootSigDesc->ParametersContainer.getDescriptorTable(Loc);
    //   for (const dxbc::RTS0::v2::DescriptorRange &Range : Table){
    //     Range.
    //   }
      
    //   break;
    // }
    }

  }

  for(const auto &CBuf : DRM.cbuffers()) {
    auto Binding = CBuf.getBinding();
    if(!BindingsMap.overlaps(combine_uint32_to_uint64(Binding.Space, Binding.LowerBound), combine_uint32_to_uint64(Binding.Space, Binding.LowerBound + Binding.Size -1)))
      auto X = 1;
  }
}
} // namespace

PreservedAnalyses
DXILPostOptimizationValidation::run(Module &M, ModuleAnalysisManager &MAM) {
  DXILResourceMap &DRM = MAM.getResult<DXILResourceAnalysis>(M);
  DXILResourceBindingInfo &DRBI = MAM.getResult<DXILResourceBindingAnalysis>(M);
  RootSignatureBindingInfo &RSBI = MAM.getResult<RootSignatureAnalysis>(M);
  ModuleMetadataInfo &MMI = MAM.getResult<DXILMetadataAnalysis>(M);

  reportErrors(M, DRM, DRBI, RSBI, MMI);
  return PreservedAnalyses::all();
}

namespace {
class DXILPostOptimizationValidationLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override {
    DXILResourceMap &DRM =
        getAnalysis<DXILResourceWrapperPass>().getResourceMap();
    DXILResourceBindingInfo &DRBI =
        getAnalysis<DXILResourceBindingWrapperPass>().getBindingInfo();

    RootSignatureBindingInfo& RSBI =
        getAnalysis<RootSignatureAnalysisWrapper>().getRSInfo();
    dxil::ModuleMetadataInfo &MMI =
        getAnalysis<DXILMetadataAnalysisWrapperPass>().getModuleMetadata();

    reportErrors(M, DRM, DRBI, RSBI, MMI);
    return false;
  }
  StringRef getPassName() const override {
    return "DXIL Post Optimization Validation";
  }
  DXILPostOptimizationValidationLegacy() : ModulePass(ID) {}

  static char ID; // Pass identification.
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<DXILResourceWrapperPass>();
    AU.addRequired<DXILResourceBindingWrapperPass>();
    AU.addRequired<RootSignatureAnalysisWrapper>();
    AU.addRequired<DXILMetadataAnalysisWrapperPass>();
    AU.addPreserved<DXILResourceWrapperPass>();
    AU.addPreserved<DXILResourceBindingWrapperPass>();
    AU.addPreserved<DXILMetadataAnalysisWrapperPass>();
    AU.addPreserved<ShaderFlagsAnalysisWrapper>();
    AU.addPreserved<RootSignatureAnalysisWrapper>();
  }
};
char DXILPostOptimizationValidationLegacy::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS_BEGIN(DXILPostOptimizationValidationLegacy, DEBUG_TYPE,
                      "DXIL Post Optimization Validation", false, false)
INITIALIZE_PASS_DEPENDENCY(DXILResourceBindingWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DXILResourceTypeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DXILResourceWrapperPass)
INITIALIZE_PASS_END(DXILPostOptimizationValidationLegacy, DEBUG_TYPE,
                    "DXIL Post Optimization Validation", false, false)

ModulePass *llvm::createDXILPostOptimizationValidationLegacyPass() {
  return new DXILPostOptimizationValidationLegacy();
}
