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

#include "DXILRootSignature.h"
#include "DXILShaderFlags.h"
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
#include "llvm/MC/DXContainerInfo.h"
#include "llvm/MC/DXContainerPSVInfo.h"
#include "llvm/MC/MCDXContainerWriter.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <cstdint>

using namespace llvm;
using namespace llvm::dxil;
using namespace llvm::mcdxbc;

static cl::opt<bool> ShaderHashDependsOnSource(
    "dx-Zss", cl::desc("Compute Shader Hash considering source information"));
extern cl::opt<std::string> PdbDebugPath;
extern cl::opt<bool> SourceInDebugModule;

namespace {
class DXContainerGlobals : public llvm::ModulePass {

  GlobalVariable *buildContainerGlobal(Module &M, Constant *Content,
                                       StringRef Name, StringRef SectionName);
  void addSection(Module &M, SmallVector<GlobalValue *> &Globals,
                  StringRef SectionData, StringRef MetadataName,
                  StringRef SectionName);
  GlobalVariable *getFeatureFlags(Module &M);
  void computeShaderHashAndDebugName(Module &M,
                                     SmallVector<GlobalValue *> &Globals);
  GlobalVariable *buildSignature(Module &M, Signature &Sig, StringRef Name,
                                 StringRef SectionName);
  void addSignature(Module &M, SmallVector<GlobalValue *> &Globals);
  void addRootSignature(Module &M, SmallVector<GlobalValue *> &Globals);
  void addResourcesForPSV(Module &M, PSVRuntimeInfo &PSV);
  void addPipelineStateValidationInfo(Module &M,
                                      SmallVector<GlobalValue *> &Globals);
  void addCompilerVersion(Module &M, SmallVector<GlobalValue *> &Globals);
  void addSourceInfo(Module &M, SmallVector<GlobalValue *> &Globals);

public:
  static char ID; // Pass identification, replacement for typeid
  DXContainerGlobals() : ModulePass(ID) {}

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
    AU.addRequired<DXILResourceWrapperPass>();
  }
};

} // namespace

bool DXContainerGlobals::runOnModule(Module &M) {
  llvm::SmallVector<GlobalValue *> Globals;
  Globals.push_back(getFeatureFlags(M));
  computeShaderHashAndDebugName(M, Globals);
  addSignature(M, Globals);
  addRootSignature(M, Globals);
  addPipelineStateValidationInfo(M, Globals);
  addCompilerVersion(M, Globals);
  addSourceInfo(M, Globals);
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

void DXContainerGlobals::addSection(Module &M,
                                    SmallVector<GlobalValue *> &Globals,
                                    StringRef SectionData,
                                    StringRef MetadataName,
                                    StringRef SectionName) {
  Constant *SectionConstant = ConstantDataArray::getString(
      M.getContext(), SectionData, /*AddNull*/ false);
  Globals.emplace_back(
      buildContainerGlobal(M, SectionConstant, MetadataName, SectionName));
}

void DXContainerGlobals::computeShaderHashAndDebugName(
    Module &M, SmallVector<GlobalValue *> &Globals) {
  ConstantDataArray *DXILConstant;
  MD5 Digest;
  dxbc::ShaderHash HashData = {0, {0}};

  if (ShaderHashDependsOnSource) {
    if (auto *ILDB = M.getNamedGlobal("dx.ildb")) {
      DXILConstant = cast<ConstantDataArray>(ILDB->getInitializer());
      HashData.Flags = static_cast<uint32_t>(dxbc::HashFlags::IncludesSource);
    } else {
      reportFatalUsageError("/Zss requires debug info (/Zi or /Zs)");
    }
  } else {
    DXILConstant =
        cast<ConstantDataArray>(M.getNamedGlobal("dx.dxil")->getInitializer());
  }

  Digest.update(DXILConstant->getRawDataValues());
  MD5::MD5Result MD5 = Digest.final();

  memcpy(reinterpret_cast<void *>(&HashData.Digest), MD5.data(), 16);
  if (sys::IsBigEndianHost)
    HashData.swapBytes();
  StringRef Data(reinterpret_cast<char *>(&HashData), sizeof(dxbc::ShaderHash));

  Constant *ModuleConstant =
      ConstantDataArray::get(M.getContext(), arrayRefFromStringRef(Data));
  Globals.emplace_back(
      buildContainerGlobal(M, ModuleConstant, "dx.hash", "HASH"));

  if (M.debug_compile_units().empty())
    return;

  SmallString<40> DebugNameStr;
  Digest.stringifyResult(MD5, DebugNameStr);
  DebugNameStr += ".pdb";
  if (!PdbDebugPath.empty()) {
    StringRef DebugFile = PdbDebugPath.getValue();
    SmallString<256> AbsoluteDebugName;
    if (sys::path::is_separator(DebugFile.back())) {
      // If /Fd was specified as a directory, put the MD5.pdb file there.
      AbsoluteDebugName = DebugFile;
      sys::path::append(AbsoluteDebugName, DebugNameStr);
    } else {
      // Otherwise, use /Fd value as a user-provided PDB file name.
      DebugNameStr = DebugFile;
      AbsoluteDebugName = DebugNameStr;
    }

    // Pass PDB name to DXContainerPDBPass via PDBNAME section.
    addSection(M, Globals, AbsoluteDebugName, "dx.pdb.name",
               PdbFileNameSectionName);
    // Pass module hash to DXContainerPDBPass.
    Globals.emplace_back(buildContainerGlobal(
        M, ConstantDataArray::get(M.getContext(), ArrayRef(HashData.Digest)),
        "dx.pdb.hash", ModuleHashSectionName));
  }

  // Emit ILDN part in debug info mode.
  mcdxbc::DebugName DebugName;
  DebugName.setFilename(DebugNameStr);
  SmallString<64> ILDNData;
  raw_svector_ostream OS(ILDNData);
  DebugName.write(OS);
  addSection(M, Globals, ILDNData, "dx.ildn", "ILDN");
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

  dxil::ModuleMetadataInfo &MMI =
      getAnalysis<DXILMetadataAnalysisWrapperPass>().getModuleMetadata();

  // Root Signature in Library don't compile to DXContainer.
  if (MMI.ShaderProfile == llvm::Triple::Library)
    return;

  auto &RSA = getAnalysis<RootSignatureAnalysisWrapper>().getRSInfo();
  const Function *EntryFunction = nullptr;

  if (MMI.ShaderProfile != llvm::Triple::RootSignature) {
    assert(MMI.EntryPropertyVec.size() == 1);
    EntryFunction = MMI.EntryPropertyVec[0].Entry;
  }

  const mcdxbc::RootSignatureDesc *RS = RSA.getDescForFunction(EntryFunction);
  if (!RS)
    return;

  SmallString<256> Data;
  raw_svector_ostream OS(Data);

  RS->write(OS);

  addSection(M, Globals, Data, "dx.rts0", "RTS0");
}

void DXContainerGlobals::addResourcesForPSV(Module &M, PSVRuntimeInfo &PSV) {
  const DXILResourceMap &DRM =
      getAnalysis<DXILResourceWrapperPass>().getResourceMap();
  DXILResourceTypeMap &DRTM =
      getAnalysis<DXILResourceTypeWrapperPass>().getResourceTypeMap();

  auto MakeBinding =
      [](const dxil::ResourceInfo::ResourceBinding &Binding,
         const dxbc::PSV::ResourceType Type, const dxil::ResourceKind Kind,
         const dxbc::PSV::ResourceFlags Flags = dxbc::PSV::ResourceFlags()) {
        dxbc::PSV::v2::ResourceBindInfo BindInfo;
        BindInfo.Type = Type;
        BindInfo.LowerBound = Binding.LowerBound;
        assert(
            (Binding.Size == 0 ||
             (uint64_t)Binding.LowerBound + Binding.Size - 1 <= UINT32_MAX) &&
            "Resource range is too large");
        BindInfo.UpperBound = (Binding.Size == 0)
                                  ? UINT32_MAX
                                  : Binding.LowerBound + Binding.Size - 1;
        BindInfo.Space = Binding.Space;
        BindInfo.Kind = static_cast<dxbc::PSV::ResourceKind>(Kind);
        BindInfo.Flags = Flags;
        return BindInfo;
      };

  for (const dxil::ResourceInfo &RI : DRM.cbuffers()) {
    const dxil::ResourceInfo::ResourceBinding &Binding = RI.getBinding();
    PSV.Resources.push_back(MakeBinding(Binding, dxbc::PSV::ResourceType::CBV,
                                        dxil::ResourceKind::CBuffer));
  }
  for (const dxil::ResourceInfo &RI : DRM.samplers()) {
    const dxil::ResourceInfo::ResourceBinding &Binding = RI.getBinding();
    PSV.Resources.push_back(MakeBinding(Binding,
                                        dxbc::PSV::ResourceType::Sampler,
                                        dxil::ResourceKind::Sampler));
  }
  for (const dxil::ResourceInfo &RI : DRM.srvs()) {
    const dxil::ResourceInfo::ResourceBinding &Binding = RI.getBinding();

    dxil::ResourceTypeInfo &TypeInfo = DRTM[RI.getHandleTy()];
    dxbc::PSV::ResourceType ResType;
    if (TypeInfo.isStruct())
      ResType = dxbc::PSV::ResourceType::SRVStructured;
    else if (TypeInfo.isTyped())
      ResType = dxbc::PSV::ResourceType::SRVTyped;
    else
      ResType = dxbc::PSV::ResourceType::SRVRaw;

    PSV.Resources.push_back(
        MakeBinding(Binding, ResType, TypeInfo.getResourceKind()));
  }
  for (const dxil::ResourceInfo &RI : DRM.uavs()) {
    const dxil::ResourceInfo::ResourceBinding &Binding = RI.getBinding();

    dxil::ResourceTypeInfo &TypeInfo = DRTM[RI.getHandleTy()];
    dxbc::PSV::ResourceType ResType;
    if (RI.hasCounter())
      ResType = dxbc::PSV::ResourceType::UAVStructuredWithCounter;
    else if (TypeInfo.isStruct())
      ResType = dxbc::PSV::ResourceType::UAVStructured;
    else if (TypeInfo.isTyped())
      ResType = dxbc::PSV::ResourceType::UAVTyped;
    else
      ResType = dxbc::PSV::ResourceType::UAVRaw;

    dxbc::PSV::ResourceFlags Flags;
    // TODO: Add support for dxbc::PSV::ResourceFlag::UsedByAtomic64, tracking
    // with https://github.com/llvm/llvm-project/issues/104392
    Flags.Flags = 0u;

    PSV.Resources.push_back(
        MakeBinding(Binding, ResType, TypeInfo.getResourceKind(), Flags));
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
         MMI.ShaderProfile == Triple::Library ||
         MMI.ShaderProfile == Triple::RootSignature);
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
    if (MMI.EntryPropertyVec[0].WaveSizeMin) {
      PSV.BaseData.MinimumWaveLaneCount = MMI.EntryPropertyVec[0].WaveSizeMin;
      PSV.BaseData.MaximumWaveLaneCount =
          MMI.EntryPropertyVec[0].WaveSizeMax
              ? MMI.EntryPropertyVec[0].WaveSizeMax
              : MMI.EntryPropertyVec[0].WaveSizeMin;
    }
    break;
  default:
    break;
  }

  if (MMI.ShaderProfile != Triple::Library &&
      MMI.ShaderProfile != Triple::RootSignature)
    PSV.EntryName = MMI.EntryPropertyVec[0].Entry->getName();

  PSV.finalize(MMI.ShaderProfile);
  PSV.write(OS);
  addSection(M, Globals, Data, "dx.psv0", "PSV0");
}

void DXContainerGlobals::addCompilerVersion(
    Module &M, SmallVector<GlobalValue *> &Globals) {
  if (M.debug_compile_units().empty())
    return;

  SmallString<256> Data;
  raw_svector_ostream OS(Data);
  mcdxbc::CompilerVersion CompilerVersion;
  CompilerVersion.write(OS);
  addSection(M, Globals, Data, "dx.vers", "VERS");
}

void DXContainerGlobals::addSourceInfo(Module &M,
                                       SmallVector<GlobalValue *> &Globals) {
  dxil::ModuleMetadataInfo &MMI =
      getAnalysis<DXILMetadataAnalysisWrapperPass>().getModuleMetadata();

  if (!MMI.SourceInfo || SourceInDebugModule)
    return;

  MMI.SourceInfo->computeEntries();
  MMI.SourceInfo->finalize();
  SmallString<256> Data;
  raw_svector_ostream OS(Data);
  MMI.SourceInfo->write(OS);
  addSection(M, Globals, Data, "dx.srci", "SRCI");
}

char DXContainerGlobals::ID = 0;
INITIALIZE_PASS_BEGIN(DXContainerGlobals, "dxil-globals",
                      "DXContainer Global Emitter", false, true)
INITIALIZE_PASS_DEPENDENCY(ShaderFlagsAnalysisWrapper)
INITIALIZE_PASS_DEPENDENCY(DXILMetadataAnalysisWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DXILResourceTypeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DXILResourceWrapperPass)
INITIALIZE_PASS_END(DXContainerGlobals, "dxil-globals",
                    "DXContainer Global Emitter", false, true)

ModulePass *llvm::createDXContainerGlobalsPass() {
  return new DXContainerGlobals();
}
