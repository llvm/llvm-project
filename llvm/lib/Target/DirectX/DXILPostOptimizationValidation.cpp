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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/Frontend/HLSL/RootSignatureValidations.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "dxil-post-optimization-validation"

using namespace llvm;
using namespace llvm::dxil;

namespace {
static const char *ResourceClassToString(llvm::dxil::ResourceClass Class) {
  switch (Class) {
  case ResourceClass::SRV:
    return "SRV";
  case ResourceClass::UAV:
    return "UAV";
  case ResourceClass::CBuffer:
    return "CBuffer";
  case ResourceClass::Sampler:
    return "Sampler";
  }
}

static ResourceClass RangeToResourceClass(uint32_t RangeType) {
  using namespace dxbc;
  switch (static_cast<DescriptorRangeType>(RangeType)) {
  case DescriptorRangeType::SRV:
    return ResourceClass::SRV;
  case DescriptorRangeType::UAV:
    return ResourceClass::UAV;
  case DescriptorRangeType::CBV:
    return ResourceClass::CBuffer;
  case DescriptorRangeType::Sampler:
    return ResourceClass::Sampler;
  }
}

ResourceClass ParameterToResourceClass(uint32_t Type) {
  using namespace dxbc;
  switch (Type) {
  case llvm::to_underlying(RootParameterType::Constants32Bit):
    return ResourceClass::CBuffer;
  case llvm::to_underlying(RootParameterType::SRV):
    return ResourceClass::SRV;
  case llvm::to_underlying(RootParameterType::UAV):
    return ResourceClass::UAV;
  case llvm::to_underlying(RootParameterType::CBV):
    return ResourceClass::CBuffer;
  default:
    llvm_unreachable("Unknown RootParameterType");
  }
}

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

static void
reportInvalidHandleTyBoundInRs(Module &M, Twine Type,
                               ResourceInfo::ResourceBinding Binding) {
  SmallString<128> Message;
  raw_svector_ostream OS(Message);
  OS << "resource " << Type << " at register (space=" << Binding.Space
     << ", register=" << Binding.LowerBound << ")"
     << " is bound to a texture or typed buffer.";
  M.getContext().diagnose(DiagnosticInfoGeneric(Message));
}

static void reportRegNotBound(Module &M,
                              llvm::hlsl::rootsig::RangeInfo Unbound) {
  SmallString<128> Message;
  raw_svector_ostream OS(Message);
  OS << "register " << ResourceClassToString(Unbound.Class)
     << " (space=" << Unbound.Space << ", register=" << Unbound.LowerBound
     << ")"
     << " does not have a binding in the Root Signature";
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

static hlsl::rootsig::RootSignatureBindingValidation
initRSBindingValidation(const mcdxbc::RootSignatureDesc &RSD,
                        dxbc::ShaderVisibility Visibility) {

  hlsl::rootsig::RootSignatureBindingValidation Validation;

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

      hlsl::rootsig::RangeInfo Binding;
      Binding.LowerBound = Const.ShaderRegister;
      Binding.Space = Const.RegisterSpace;
      Binding.UpperBound = Binding.LowerBound;

      // Root Constants Bind to CBuffers
      Validation.addBinding(ResourceClass::CBuffer, Binding);

      break;
    }

    case llvm::to_underlying(dxbc::RootParameterType::SRV):
    case llvm::to_underlying(dxbc::RootParameterType::UAV):
    case llvm::to_underlying(dxbc::RootParameterType::CBV): {
      dxbc::RTS0::v2::RootDescriptor Desc =
          RSD.ParametersContainer.getRootDescriptor(Loc);

      hlsl::rootsig::RangeInfo Binding;
      Binding.LowerBound = Desc.ShaderRegister;
      Binding.Space = Desc.RegisterSpace;
      Binding.UpperBound = Binding.LowerBound;

      Validation.addBinding(ParameterToResourceClass(Type), Binding);
      break;
    }
    case llvm::to_underlying(dxbc::RootParameterType::DescriptorTable): {
      const mcdxbc::DescriptorTable &Table =
          RSD.ParametersContainer.getDescriptorTable(Loc);

      for (const dxbc::RTS0::v2::DescriptorRange &Range : Table.Ranges) {
        hlsl::rootsig::RangeInfo Binding;
        Binding.LowerBound = Range.BaseShaderRegister;
        Binding.Space = Range.RegisterSpace;
        Binding.UpperBound = Binding.LowerBound + Range.NumDescriptors - 1;
        Validation.addBinding(RangeToResourceClass(Range.RangeType), Binding);
      }
      break;
    }
    }
  }

  return Validation;
}

static SmallVector<ResourceInfo::ResourceBinding>
getRootDescriptorsBindingInfo(const mcdxbc::RootSignatureDesc &RSD,
                              dxbc::ShaderVisibility Visibility) {

  SmallVector<ResourceInfo::ResourceBinding> RDs;

  for (size_t I = 0; I < RSD.ParametersContainer.size(); I++) {
    const auto &[Type, Loc] =
        RSD.ParametersContainer.getTypeAndLocForParameter(I);

    const auto &Header = RSD.ParametersContainer.getHeader(I);
    if (Header.ShaderVisibility !=
            llvm::to_underlying(dxbc::ShaderVisibility::All) &&
        Header.ShaderVisibility != llvm::to_underlying(Visibility))
      continue;

    switch (Type) {

    case llvm::to_underlying(dxbc::RootParameterType::SRV):
    case llvm::to_underlying(dxbc::RootParameterType::UAV):
    case llvm::to_underlying(dxbc::RootParameterType::CBV): {
      dxbc::RTS0::v2::RootDescriptor Desc =
          RSD.ParametersContainer.getRootDescriptor(Loc);

      ResourceInfo::ResourceBinding Binding;
      Binding.LowerBound = Desc.ShaderRegister;
      Binding.Space = Desc.RegisterSpace;
      Binding.Size = 1;

      RDs.push_back(Binding);
      break;
    }
    }
  }

  return RDs;
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

static void reportInvalidHandleTy(
    Module &M, const llvm::ArrayRef<dxil::ResourceInfo::ResourceBinding> &RDs,
    const iterator_range<SmallVectorImpl<dxil::ResourceInfo>::iterator>
        &Resources) {
  for (auto Res = Resources.begin(), End = Resources.end(); Res != End; Res++) {
    llvm::dxil::ResourceInfo::ResourceBinding Binding = Res->getBinding();
    for (const auto &RD : RDs) {
      if (Binding.overlapsWith(RD)) {
        TargetExtType *Handle = Res->getHandleTy();
        auto *TypedBuffer = dyn_cast_or_null<TypedBufferExtType>(Handle);
        auto *Texture = dyn_cast_or_null<TextureExtType>(Handle);

        if (TypedBuffer != nullptr || Texture != nullptr)
          reportInvalidHandleTyBoundInRs(M, Res->getName(), Res->getBinding());
      }
    }
  }
}

static void reportUnboundRegisters(
    Module &M,
    const llvm::hlsl::rootsig::RootSignatureBindingValidation &Validation,
    ResourceClass Class,
    const iterator_range<SmallVectorImpl<dxil::ResourceInfo>::iterator>
        &Resources) {
  SmallVector<hlsl::rootsig::RangeInfo> Ranges;
  for (auto Res = Resources.begin(), End = Resources.end(); Res != End; Res++) {
    ResourceInfo::ResourceBinding ResBinding = Res->getBinding();
    hlsl::rootsig::RangeInfo Range;
    Range.Space = ResBinding.Space;
    Range.LowerBound = ResBinding.LowerBound;
    Range.UpperBound = Range.LowerBound + ResBinding.Size - 1;
    Range.Class = Class;
    Ranges.push_back(Range);
  }

  SmallVector<hlsl::rootsig::RangeInfo> Unbounds =
      hlsl::rootsig::findUnboundRanges(Ranges,
                                       Validation.getBindingsOfType(Class));
  for (const auto &Unbound : Unbounds)
    reportRegNotBound(M, Unbound);
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
    dxbc::ShaderVisibility Visibility = tripleToVisibility(MMI.ShaderProfile);
    llvm::hlsl::rootsig::RootSignatureBindingValidation Validation =
        initRSBindingValidation(*RSD, Visibility);

    reportUnboundRegisters(M, Validation, ResourceClass::CBuffer,
                           DRM.cbuffers());
    reportUnboundRegisters(M, Validation, ResourceClass::UAV, DRM.uavs());
    reportUnboundRegisters(M, Validation, ResourceClass::Sampler,
                           DRM.samplers());
    reportUnboundRegisters(M, Validation, ResourceClass::SRV, DRM.srvs());

    SmallVector<ResourceInfo::ResourceBinding> RDs =
        getRootDescriptorsBindingInfo(*RSD, Visibility);

    reportInvalidHandleTy(M, RDs, DRM.cbuffers());
    reportInvalidHandleTy(M, RDs, DRM.srvs());
    reportInvalidHandleTy(M, RDs, DRM.uavs());
    reportInvalidHandleTy(M, RDs, DRM.samplers());
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
