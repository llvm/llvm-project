//===- DXContainerGlobals.cpp - DXContainer global generator pass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DXContainerGlobalsPass implementation.
//
//===----------------------------------------------------------------------===//

#include "DXILShaderFlags.h"
#include "DXILRootSignature.h"
#include "DirectX.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/DXContainerPSVInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/MD5.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <optional>

using namespace llvm;
using namespace llvm::dxil;
using namespace llvm::mcdxbc;

namespace {
class DXContainerGlobals : public llvm::ModulePass {

  GlobalVariable *buildContainerGlobal(Module &M, Constant *Content,
                                       StringRef Name, StringRef SectionName);
  GlobalVariable *getFeatureFlags(Module &M);
  GlobalVariable *computeShaderHash(Module &M);
  GlobalVariable *buildSignature(Module &M, Signature &Sig, StringRef Name,
                                 StringRef SectionName);
  void addSignature(Module &M, SmallVector<GlobalValue *> &Globals);
  void addRootSignature(Module &M, SmallVector<GlobalValue *> &Globals);
  void addResourcesForPSV(Module &M, PSVRuntimeInfo &PSV);
  void addPipelineStateValidationInfo(Module &M,
                                      SmallVector<GlobalValue *> &Globals);

public:
  static char ID; // Pass identification, replacement for typeid
  DXContainerGlobals() : ModulePass(ID) {
    initializeDXContainerGlobalsPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "DXContainer Global Emitter";
  }

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<ShaderFlagsAnalysisWrapper>();
    AU.addRequired<RootSignatureAnalysisWrapper>();
    AU.addRequired<DXILMetadataAnalysisWrapperPass>();
    AU.addRequired<DXILResourceTypeWrapperPass>();
    AU.addRequired<DXILResourceBindingWrapperPass>();
  }
};

} // namespace

bool DXContainerGlobals::runOnModule(Module &M) {
  llvm::SmallVector<GlobalValue *> Globals;
  Globals.push_back(getFeatureFlags(M));
  Globals.push_back(computeShaderHash(M));
  addSignature(M, Globals);
  addRootSignature(M, Globals);
  addPipelineStateValidationInfo(M, Globals);
  appendToCompilerUsed(M, Globals);
  return true;
}

GlobalVariable *DXContainerGlobals::getFeatureFlags(Module &M) {
  uint64_t CombinedFeatureFlags = getAnalysis<ShaderFlagsAnalysisWrapper>()
                                      .getShaderFlags()
                                      .getCombinedFlags()
                                      .getFeatureFlags();

  Constant *FeatureFlagsConstant =
      ConstantInt::get(M.getContext(), APInt(64, CombinedFeatureFlags));
  return buildContainerGlobal(M, FeatureFlagsConstant, "dx.sfi0", "SFI0");
}

GlobalVariable *DXContainerGlobals::computeShaderHash(Module &M) {
  auto *DXILConstant =
      cast<ConstantDataArray>(M.getNamedGlobal("dx.dxil")->getInitializer());
  MD5 Digest;
  Digest.update(DXILConstant->getRawDataValues());
  MD5::MD5Result Result = Digest.final();

  dxbc::ShaderHash HashData = {0, {0}};
  // The Hash's IncludesSource flag gets set whenever the hashed shader includes
  // debug information.
  if (M.debug_compile_units_begin() != M.debug_compile_units_end())
    HashData.Flags = static_cast<uint32_t>(dxbc::HashFlags::IncludesSource);

  memcpy(reinterpret_cast<void *>(&HashData.Digest), Result.data(), 16);
  if (sys::IsBigEndianHost)
    HashData.swapBytes();
  StringRef Data(reinterpret_cast<char *>(&HashData), sizeof(dxbc::ShaderHash));

  Constant *ModuleConstant =
      ConstantDataArray::get(M.getContext(), arrayRefFromStringRef(Data));
  return buildContainerGlobal(M, ModuleConstant, "dx.hash", "HASH");
}

GlobalVariable *DXContainerGlobals::buildContainerGlobal(
    Module &M, Constant *Content, StringRef Name, StringRef SectionName) {
  auto *GV = new llvm::GlobalVariable(
      M, Content->getType(), true, GlobalValue::PrivateLinkage, Content, Name);
  GV->setSection(SectionName);
  GV->setAlignment(Align(4));
  return GV;
}

GlobalVariable *DXContainerGlobals::buildSignature(Module &M, Signature &Sig,
                                                   StringRef Name,
                                                   StringRef SectionName) {
  SmallString<256> Data;
  raw_svector_ostream OS(Data);
  Sig.write(OS);
  Constant *Constant =
      ConstantDataArray::getString(M.getContext(), Data, /*AddNull*/ false);
  return buildContainerGlobal(M, Constant, Name, SectionName);
}

void DXContainerGlobals::addSignature(Module &M,
                                      SmallVector<GlobalValue *> &Globals) {
  // FIXME: support graphics shader.
  //  see issue https://github.com/llvm/llvm-project/issues/90504.

  Signature InputSig;
  Globals.emplace_back(buildSignature(M, InputSig, "dx.isg1", "ISG1"));

  Signature OutputSig;
  Globals.emplace_back(buildSignature(M, OutputSig, "dx.osg1", "OSG1"));
}

void DXContainerGlobals::addRootSignature(Module &M,
                                          SmallVector<GlobalValue *> &Globals) {

  std::optional<ModuleRootSignature> MRS =
      getAnalysis<RootSignatureAnalysisWrapper>()
          .getRootSignature();
  if (!MRS.has_value())
    return;

  SmallString<256> Data;
  raw_svector_ostream OS(Data);
  MRS->write(OS);

  Constant *Constant =
      ConstantDataArray::getString(M.getContext(), Data, /*AddNull*/ false);
  Globals.emplace_back(buildContainerGlobal(M, Constant, "dx.rts0", "RTS0"));
}

void DXContainerGlobals::addResourcesForPSV(Module &M, PSVRuntimeInfo &PSV) {
  const DXILBindingMap &DBM =
      getAnalysis<DXILResourceBindingWrapperPass>().getBindingMap();
  DXILResourceTypeMap &DRTM =
      getAnalysis<DXILResourceTypeWrapperPass>().getResourceTypeMap();

  for (const dxil::ResourceBindingInfo &RBI : DBM) {
    const dxil::ResourceBindingInfo::ResourceBinding &Binding =
        RBI.getBinding();
    dxbc::PSV::v2::ResourceBindInfo BindInfo;
    BindInfo.LowerBound = Binding.LowerBound;
    BindInfo.UpperBound = Binding.LowerBound + Binding.Size - 1;
    BindInfo.Space = Binding.Space;

    dxil::ResourceTypeInfo &TypeInfo = DRTM[RBI.getHandleTy()];
    dxbc::PSV::ResourceType ResType = dxbc::PSV::ResourceType::Invalid;
    bool IsUAV = TypeInfo.getResourceClass() == dxil::ResourceClass::UAV;
    switch (TypeInfo.getResourceKind()) {
    case dxil::ResourceKind::Sampler:
      ResType = dxbc::PSV::ResourceType::Sampler;
      break;
    case dxil::ResourceKind::CBuffer:
      ResType = dxbc::PSV::ResourceType::CBV;
      break;
    case dxil::ResourceKind::StructuredBuffer:
      ResType = IsUAV ? dxbc::PSV::ResourceType::UAVStructured
                      : dxbc::PSV::ResourceType::SRVStructured;
      if (IsUAV && TypeInfo.getUAV().HasCounter)
        ResType = dxbc::PSV::ResourceType::UAVStructuredWithCounter;
      break;
    case dxil::ResourceKind::RTAccelerationStructure:
      ResType = dxbc::PSV::ResourceType::SRVRaw;
      break;
    case dxil::ResourceKind::RawBuffer:
      ResType = IsUAV ? dxbc::PSV::ResourceType::UAVRaw
                      : dxbc::PSV::ResourceType::SRVRaw;
      break;
    default:
      ResType = IsUAV ? dxbc::PSV::ResourceType::UAVTyped
                      : dxbc::PSV::ResourceType::SRVTyped;
      break;
    }
    BindInfo.Type = ResType;

    BindInfo.Kind =
        static_cast<dxbc::PSV::ResourceKind>(TypeInfo.getResourceKind());
    // TODO: Add support for dxbc::PSV::ResourceFlag::UsedByAtomic64, tracking
    // with https://github.com/llvm/llvm-project/issues/104392
    BindInfo.Flags.Flags = 0u;

    PSV.Resources.emplace_back(BindInfo);
  }
}

void DXContainerGlobals::addPipelineStateValidationInfo(
    Module &M, SmallVector<GlobalValue *> &Globals) {
  SmallString<256> Data;
  raw_svector_ostream OS(Data);
  PSVRuntimeInfo PSV;
  PSV.BaseData.MinimumWaveLaneCount = 0;
  PSV.BaseData.MaximumWaveLaneCount = std::numeric_limits<uint32_t>::max();

  dxil::ModuleMetadataInfo &MMI =
      getAnalysis<DXILMetadataAnalysisWrapperPass>().getModuleMetadata();
  assert(MMI.EntryPropertyVec.size() == 1 ||
         MMI.ShaderProfile == Triple::Library);
  PSV.BaseData.ShaderStage =
      static_cast<uint8_t>(MMI.ShaderProfile - Triple::Pixel);

  addResourcesForPSV(M, PSV);

  // Hardcoded values here to unblock loading the shader into D3D.
  //
  // TODO: Lots more stuff to do here!
  //
  // See issue https://github.com/llvm/llvm-project/issues/96674.
  switch (MMI.ShaderProfile) {
  case Triple::Compute:
    PSV.BaseData.NumThreadsX = MMI.EntryPropertyVec[0].NumThreadsX;
    PSV.BaseData.NumThreadsY = MMI.EntryPropertyVec[0].NumThreadsY;
    PSV.BaseData.NumThreadsZ = MMI.EntryPropertyVec[0].NumThreadsZ;
    break;
  default:
    break;
  }

  if (MMI.ShaderProfile != Triple::Library)
    PSV.EntryName = MMI.EntryPropertyVec[0].Entry->getName();

  PSV.finalize(MMI.ShaderProfile);
  PSV.write(OS);
  Constant *Constant =
      ConstantDataArray::getString(M.getContext(), Data, /*AddNull*/ false);
  Globals.emplace_back(buildContainerGlobal(M, Constant, "dx.psv0", "PSV0"));
}

char DXContainerGlobals::ID = 0;
INITIALIZE_PASS_BEGIN(DXContainerGlobals, "dxil-globals",
                      "DXContainer Global Emitter", false, true)
INITIALIZE_PASS_DEPENDENCY(ShaderFlagsAnalysisWrapper)
INITIALIZE_PASS_DEPENDENCY(DXILMetadataAnalysisWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DXILResourceTypeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DXILResourceBindingWrapperPass)
INITIALIZE_PASS_END(DXContainerGlobals, "dxil-globals",
                    "DXContainer Global Emitter", false, true)

ModulePass *llvm::createDXContainerGlobalsPass() {
  return new DXContainerGlobals();
}
