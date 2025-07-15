//===- DXILPostOptimizationValidation.cpp - Opt DXIL validation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILPostOptimizationValidation.h"
#include "DXILShaderFlags.h"
#include "DirectX.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
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

static void reportRegNotBound(Module &M, Twine Type,
                              ResourceInfo::ResourceBinding Binding) {
  SmallString<128> Message;
  raw_svector_ostream OS(Message);
  OS << "register " << Type << " (space=" << Binding.Space
     << ", register=" << Binding.LowerBound << ")"
     << " is not defined in Root Signature";
  M.getContext().diagnose(DiagnosticInfoGeneric(Message));
}

static dxbc::ShaderVisibility
tripleToVisibility(llvm::Triple::EnvironmentType ET) {
  assert((ET == Triple::Pixel || ET == Triple::Vertex ||
          ET == Triple::Geometry || ET == Triple::Hull ||
          ET == Triple::Domain || ET == Triple::Mesh ||
          ET == Triple::Compute) &&
         "Invalid Triple to shader stage conversion");

  switch (ET) {
  case Triple::Pixel:
    return dxbc::ShaderVisibility::Pixel;
  case Triple::Vertex:
    return dxbc::ShaderVisibility::Vertex;
  case Triple::Geometry:
    return dxbc::ShaderVisibility::Geometry;
  case Triple::Hull:
    return dxbc::ShaderVisibility::Hull;
  case Triple::Domain:
    return dxbc::ShaderVisibility::Domain;
  case Triple::Mesh:
    return dxbc::ShaderVisibility::Mesh;
  case Triple::Compute:
    return dxbc::ShaderVisibility::All;
  default:
    llvm_unreachable("Invalid triple to shader stage conversion");
  }
}

static uint32_t parameterToRangeType(uint32_t Type) {
  switch (Type) {
  case llvm::to_underlying(dxbc::RootParameterType::CBV):
    return llvm::to_underlying(dxbc::DescriptorRangeType::CBV);
  case llvm::to_underlying(dxbc::RootParameterType::SRV):
    return llvm::to_underlying(dxbc::DescriptorRangeType::SRV);
  case llvm::to_underlying(dxbc::RootParameterType::UAV):
    return llvm::to_underlying(dxbc::DescriptorRangeType::UAV);
  default:
    llvm_unreachable("Root Parameter Type has no Range Type equivalent");
  }
}

static RootSignatureBindingValidation
initRSBindingValidation(const mcdxbc::RootSignatureDesc &RSD,
                        dxbc::ShaderVisibility Visibility) {

  RootSignatureBindingValidation Validation;

  for (size_t I = 0; I < RSD.ParametersContainer.size(); I++) {
    const auto &[Type, Loc] =
        RSD.ParametersContainer.getTypeAndLocForParameter(I);

    const auto &Header = RSD.ParametersContainer.getHeader(I);
    if (Header.ShaderVisibility !=
            llvm::to_underlying(dxbc::ShaderVisibility::All) &&
        Header.ShaderVisibility != llvm::to_underlying(Visibility))
      continue;

    switch (Type) {
    case llvm::to_underlying(dxbc::RootParameterType::Constants32Bit): {
      dxbc::RTS0::v1::RootConstants Const =
          RSD.ParametersContainer.getConstant(Loc);

      llvm::dxil::ResourceInfo::ResourceBinding Binding;
      Binding.LowerBound = Const.ShaderRegister;
      Binding.Space = Const.RegisterSpace;
      Binding.Size = 1;

      // Root Constants Bind to CBuffers
      Validation.addBinding(llvm::to_underlying(dxbc::DescriptorRangeType::CBV),
                            Binding);

      break;
    }

    case llvm::to_underlying(dxbc::RootParameterType::SRV):
    case llvm::to_underlying(dxbc::RootParameterType::UAV):
    case llvm::to_underlying(dxbc::RootParameterType::CBV): {
      dxbc::RTS0::v2::RootDescriptor Desc =
          RSD.ParametersContainer.getRootDescriptor(Loc);

      llvm::dxil::ResourceInfo::ResourceBinding Binding;
      Binding.LowerBound = Desc.ShaderRegister;
      Binding.Space = Desc.RegisterSpace;
      Binding.Size = 1;

      Validation.addBinding(parameterToRangeType(Type), Binding);
      break;
    }
    case llvm::to_underlying(dxbc::RootParameterType::DescriptorTable): {
      const mcdxbc::DescriptorTable &Table =
          RSD.ParametersContainer.getDescriptorTable(Loc);

      for (const dxbc::RTS0::v2::DescriptorRange &Range : Table.Ranges) {
        llvm::dxil::ResourceInfo::ResourceBinding Binding;
        Binding.LowerBound = Range.BaseShaderRegister;
        Binding.Space = Range.RegisterSpace;
        Binding.Size = Range.NumDescriptors;
        Validation.addBinding(Range.RangeType, Binding);
      }
      break;
    }
    }
  }

  return Validation;
}

std::optional<mcdxbc::RootSignatureDesc>
getRootSignature(RootSignatureBindingInfo &RSBI,
                 dxil::ModuleMetadataInfo &MMI) {
  if (MMI.EntryPropertyVec.size() == 0)
    return std::nullopt;
  std::optional<mcdxbc::RootSignatureDesc> RootSigDesc =
      RSBI.getDescForFunction(MMI.EntryPropertyVec[0].Entry);
  if (!RootSigDesc)
    return std::nullopt;
  return RootSigDesc;
}

static void reportUnboundRegisters(
    Module &M,
    const llvm::ArrayRef<llvm::dxil::ResourceInfo::ResourceBinding> &Bindings,
    const iterator_range<SmallVectorImpl<dxil::ResourceInfo>::iterator>
        &Resources) {
  for (auto Res = Resources.begin(), End = Resources.end(); Res != End; Res++) {
    bool Bound = false;
    ResourceInfo::ResourceBinding ResBinding = Res->getBinding();
    for (const auto &Binding : Bindings) {
      if (ResBinding.Space == Binding.Space &&
          ResBinding.LowerBound >= Binding.LowerBound &&
          ResBinding.LowerBound + ResBinding.Size - 1 <
              Binding.LowerBound + Binding.Size) {
        Bound = true;
        break;
      }
    }
    if (!Bound) {
      reportRegNotBound(M, Res->getName(), Res->getBinding());
    }
  }
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

  if (auto RSD = getRootSignature(RSBI, MMI)) {

    RootSignatureBindingValidation Validation =
        initRSBindingValidation(*RSD, tripleToVisibility(MMI.ShaderProfile));

    reportUnboundRegisters(
        M, Validation.getBindingsOfType(dxbc::DescriptorRangeType::CBV),
        DRM.cbuffers());
    reportUnboundRegisters(
        M, Validation.getBindingsOfType(dxbc::DescriptorRangeType::UAV),
        DRM.uavs());
    reportUnboundRegisters(
        M, Validation.getBindingsOfType(dxbc::DescriptorRangeType::Sampler),
        DRM.samplers());
    reportUnboundRegisters(
        M, Validation.getBindingsOfType(dxbc::DescriptorRangeType::SRV),
        DRM.srvs());
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
    RootSignatureBindingInfo &RSBI =
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
    AU.addRequired<DXILMetadataAnalysisWrapperPass>();
    AU.addRequired<RootSignatureAnalysisWrapper>();
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
INITIALIZE_PASS_DEPENDENCY(DXILMetadataAnalysisWrapperPass)
INITIALIZE_PASS_DEPENDENCY(RootSignatureAnalysisWrapper)
INITIALIZE_PASS_END(DXILPostOptimizationValidationLegacy, DEBUG_TYPE,
                    "DXIL Post Optimization Validation", false, false)

ModulePass *llvm::createDXILPostOptimizationValidationLegacyPass() {
  return new DXILPostOptimizationValidationLegacy();
}
