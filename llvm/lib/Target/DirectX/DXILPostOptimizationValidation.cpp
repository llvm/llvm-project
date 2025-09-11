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
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/DXILABI.h"

#define DEBUG_TYPE "dxil-post-optimization-validation"

using namespace llvm;
using namespace llvm::dxil;

static ResourceClass toResourceClass(dxbc::RootParameterType Type) {
  using namespace dxbc;
  switch (Type) {
  case RootParameterType::Constants32Bit:
    return ResourceClass::CBuffer;
  case RootParameterType::SRV:
    return ResourceClass::SRV;
  case RootParameterType::UAV:
    return ResourceClass::UAV;
  case RootParameterType::CBV:
    return ResourceClass::CBuffer;
  case dxbc::RootParameterType::DescriptorTable:
    llvm_unreachable("DescriptorTable is not convertible to ResourceClass");
  }
  llvm_unreachable("Unknown RootParameterType");
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
  [[maybe_unused]] bool ErrorFound = false;
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
        ErrorFound = true;
        RI++;
      }
      PrevRI = CurrentRI;
    }
  }
  assert(ErrorFound && "this function should be called only when if "
                       "DXILResourceBindingInfo::hasOverlapingBinding() is "
                       "true, yet no overlapping binding was found");
}

static void reportInvalidHandleTyError(Module &M, ResourceClass RC,
                                       ResourceInfo::ResourceBinding Binding) {
  SmallString<160> Message;
  raw_svector_ostream OS(Message);
  StringRef RCName = getResourceClassName(RC);
  OS << RCName << " at register " << Binding.LowerBound << " and space "
     << Binding.Space << " is bound to a texture or typed buffer. " << RCName
     << " root descriptors can only be Raw or Structured buffers.";
  M.getContext().diagnose(DiagnosticInfoGeneric(Message));
}

static void reportOverlappingRegisters(Module &M, const llvm::hlsl::Binding &R1,
                                       const llvm::hlsl::Binding &R2) {
  SmallString<128> Message;

  raw_svector_ostream OS(Message);
  OS << "resource " << getResourceClassName(R1.RC) << " (space=" << R1.Space
     << ", registers=[" << R1.LowerBound << ", " << R1.UpperBound
     << "]) overlaps with resource " << getResourceClassName(R2.RC)
     << " (space=" << R2.Space << ", registers=[" << R2.LowerBound << ", "
     << R2.UpperBound << "])";
  M.getContext().diagnose(DiagnosticInfoGeneric(Message));
}

static void
reportRegNotBound(Module &M, ResourceClass Class,
                  const llvm::dxil::ResourceInfo::ResourceBinding &Unbound) {
  SmallString<128> Message;
  raw_svector_ostream OS(Message);
  OS << getResourceClassName(Class) << " register " << Unbound.LowerBound
     << " in space " << Unbound.Space
     << " does not have a binding in the Root Signature";
  M.getContext().diagnose(DiagnosticInfoGeneric(Message));
}

static dxbc::ShaderVisibility
tripleToVisibility(llvm::Triple::EnvironmentType ET) {
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

static void validateRootSignature(Module &M,
                                  const mcdxbc::RootSignatureDesc &RSD,
                                  dxil::ModuleMetadataInfo &MMI,
                                  DXILResourceMap &DRM,
                                  DXILResourceTypeMap &DRTM) {

  hlsl::BindingInfoBuilder Builder;
  dxbc::ShaderVisibility Visibility = tripleToVisibility(MMI.ShaderProfile);

  for (const mcdxbc::RootParameterInfo &ParamInfo : RSD.ParametersContainer) {
    dxbc::ShaderVisibility ParamVisibility =
        dxbc::ShaderVisibility(ParamInfo.Visibility);
    if (ParamVisibility != dxbc::ShaderVisibility::All &&
        ParamVisibility != Visibility)
      continue;
    dxbc::RootParameterType ParamType = dxbc::RootParameterType(ParamInfo.Type);
    switch (ParamType) {
    case dxbc::RootParameterType::Constants32Bit: {
      mcdxbc::RootConstants Const =
          RSD.ParametersContainer.getConstant(ParamInfo.Location);
      Builder.trackBinding(dxil::ResourceClass::CBuffer, Const.RegisterSpace,
                           Const.ShaderRegister, Const.ShaderRegister,
                           &ParamInfo);
      break;
    }

    case dxbc::RootParameterType::SRV:
    case dxbc::RootParameterType::UAV:
    case dxbc::RootParameterType::CBV: {
      mcdxbc::RootDescriptor Desc =
          RSD.ParametersContainer.getRootDescriptor(ParamInfo.Location);
      Builder.trackBinding(toResourceClass(ParamInfo.Type), Desc.RegisterSpace,
                           Desc.ShaderRegister, Desc.ShaderRegister,
                           &ParamInfo);

      break;
    }
    case dxbc::RootParameterType::DescriptorTable: {
      const mcdxbc::DescriptorTable &Table =
          RSD.ParametersContainer.getDescriptorTable(ParamInfo.Location);

      for (const mcdxbc::DescriptorRange &Range : Table.Ranges) {
        uint32_t UpperBound =
            Range.NumDescriptors == ~0U
                ? Range.BaseShaderRegister
                : Range.BaseShaderRegister + Range.NumDescriptors - 1;
        Builder.trackBinding(Range.RangeType, Range.RegisterSpace,
                             Range.BaseShaderRegister, UpperBound, &ParamInfo);
      }
      break;
    }
    }
  }

  for (const mcdxbc::StaticSampler &S : RSD.StaticSamplers)
    Builder.trackBinding(dxil::ResourceClass::Sampler, S.RegisterSpace,
                         S.ShaderRegister, S.ShaderRegister, &S);

  Builder.calculateBindingInfo(
      [&M](const llvm::hlsl::BindingInfoBuilder &Builder,
           const llvm::hlsl::Binding &ReportedBinding) {
        const llvm::hlsl::Binding &Overlaping =
            Builder.findOverlapping(ReportedBinding);
        reportOverlappingRegisters(M, ReportedBinding, Overlaping);
      });
  const hlsl::BoundRegs &BoundRegs = Builder.takeBoundRegs();
  for (const ResourceInfo &RI : DRM) {
    const ResourceInfo::ResourceBinding &Binding = RI.getBinding();
    const dxil::ResourceTypeInfo &RTI = DRTM[RI.getHandleTy()];
    dxil::ResourceClass RC = RTI.getResourceClass();
    dxil::ResourceKind RK = RTI.getResourceKind();

    const llvm::hlsl::Binding *Reg =
        BoundRegs.findBoundReg(RC, Binding.Space, Binding.LowerBound,
                               Binding.LowerBound + Binding.Size - 1);

    if (Reg != nullptr) {
      const auto *ParamInfo =
          static_cast<const mcdxbc::RootParameterInfo *>(Reg->Cookie);

      if (RC != ResourceClass::SRV && RC != ResourceClass::UAV)
        continue;

      if (ParamInfo->Type == dxbc::RootParameterType::DescriptorTable)
        continue;

      if (RK != ResourceKind::RawBuffer && RK != ResourceKind::StructuredBuffer)
        reportInvalidHandleTyError(M, RC, Binding);
    } else {
      reportRegNotBound(M, RC, Binding);
    }
  }
}

static mcdxbc::RootSignatureDesc *
getRootSignature(RootSignatureBindingInfo &RSBI,
                 dxil::ModuleMetadataInfo &MMI) {
  if (MMI.EntryPropertyVec.size() == 0)
    return nullptr;
  return RSBI.getDescForFunction(MMI.EntryPropertyVec[0].Entry);
}

static void reportErrors(Module &M, DXILResourceMap &DRM,
                         DXILResourceBindingInfo &DRBI,
                         RootSignatureBindingInfo &RSBI,
                         dxil::ModuleMetadataInfo &MMI,
                         DXILResourceTypeMap &DRTM) {
  if (DRM.hasInvalidCounterDirection())
    reportInvalidDirection(M, DRM);

  if (DRBI.hasOverlappingBinding())
    reportOverlappingBinding(M, DRM);

  assert(!DRBI.hasImplicitBinding() && "implicit bindings should be handled in "
                                       "DXILResourceImplicitBinding pass");

  if (mcdxbc::RootSignatureDesc *RSD = getRootSignature(RSBI, MMI))
    validateRootSignature(M, *RSD, MMI, DRM, DRTM);
}

PreservedAnalyses
DXILPostOptimizationValidation::run(Module &M, ModuleAnalysisManager &MAM) {
  DXILResourceMap &DRM = MAM.getResult<DXILResourceAnalysis>(M);
  DXILResourceBindingInfo &DRBI = MAM.getResult<DXILResourceBindingAnalysis>(M);
  RootSignatureBindingInfo &RSBI = MAM.getResult<RootSignatureAnalysis>(M);
  ModuleMetadataInfo &MMI = MAM.getResult<DXILMetadataAnalysis>(M);
  DXILResourceTypeMap &DRTM = MAM.getResult<DXILResourceTypeAnalysis>(M);

  reportErrors(M, DRM, DRBI, RSBI, MMI, DRTM);
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
    DXILResourceTypeMap &DRTM =
        getAnalysis<DXILResourceTypeWrapperPass>().getResourceTypeMap();

    reportErrors(M, DRM, DRBI, RSBI, MMI, DRTM);
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
    AU.addRequired<DXILResourceTypeWrapperPass>();
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
INITIALIZE_PASS_DEPENDENCY(DXILResourceTypeWrapperPass)
INITIALIZE_PASS_END(DXILPostOptimizationValidationLegacy, DEBUG_TYPE,
                    "DXIL Post Optimization Validation", false, false)

ModulePass *llvm::createDXILPostOptimizationValidationLegacyPass() {
  return new DXILPostOptimizationValidationLegacy();
}
