//===- DXILContainerGlobals.cpp - DXContainer global generator pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DXILContainerGlobalsPass implementation.
//
//===----------------------------------------------------------------------===//

#include "DXILRootSignature.h"
#include "DXILShaderFlags.h"
#include "DXILSignatureAnalysis.h"
#include "DirectX.h"
#include "MCTargetDesc/DirectXMCOptions.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/DXContainerInfo.h"
#include "llvm/MC/DXContainerPSVInfo.h"
#include "llvm/MC/MCDXContainerWriter.h"
#include "llvm/Pass.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <cstdint>

using namespace llvm;
using namespace llvm::dxil;
using namespace llvm::mcdxbc;

namespace {
class DXILContainerGlobalsLegacy : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  DXILContainerGlobalsLegacy() : ModulePass(ID) {}

  StringRef getPassName() const override {
    return "DXContainer Global Emitter";
  }

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<ShaderFlagsAnalysisWrapper>();
    AU.addRequired<RootSignatureAnalysisWrapper>();
    AU.addRequired<SignatureAnalysisWrapper>();
    AU.addRequired<DXILMetadataAnalysisWrapperPass>();
    AU.addRequired<DXILResourceTypeWrapperPass>();
    AU.addRequired<DXILResourceWrapperPass>();
  }
};

} // namespace

static GlobalVariable *buildContainerGlobal(Module &M, Constant *Content,
                                            StringRef Name,
                                            StringRef SectionName) {
  auto *GV = new llvm::GlobalVariable(
      M, Content->getType(), true, GlobalValue::PrivateLinkage, Content, Name);
  GV->setSection(SectionName);
  GV->setAlignment(Align(4));
  return GV;
}

static GlobalVariable *
getFeatureFlags(Module &M,
                function_ref<const ModuleShaderFlags &()> GetShaderFlags) {
  uint64_t CombinedFeatureFlags =
      GetShaderFlags().getCombinedFlags().getFeatureFlags();

  Constant *FeatureFlagsConstant =
      ConstantInt::get(M.getContext(), APInt(64, CombinedFeatureFlags));
  return buildContainerGlobal(M, FeatureFlagsConstant, "dx.sfi0", "SFI0");
}

static void addSection(Module &M, SmallVector<GlobalValue *> &Globals,
                       StringRef SectionData, StringRef MetadataName,
                       StringRef SectionName) {
  Constant *SectionConstant = ConstantDataArray::getString(
      M.getContext(), SectionData, /*AddNull*/ false);
  Globals.emplace_back(
      buildContainerGlobal(M, SectionConstant, MetadataName, SectionName));
}

static void computeShaderHashAndDebugName(Module &M,
                                          SmallVector<GlobalValue *> &Globals) {
  ConstantDataArray *DXILConstant;
  MD5 Digest;
  dxbc::ShaderHash HashData = {0, {0}};
  const auto &Opts = M.getContext().getLibraryOptions().get<DirectXMCOptions>();

  if (Opts.Zss) {
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
  if (!Opts.pdb_path.empty() || Opts.pdb_in_private) {
    if (!Opts.pdb_path.empty()) {
      StringRef DebugFile = Opts.pdb_path;
      SmallString<256> AbsoluteDebugName;
      if (sys::path::is_separator(DebugFile.back())) {
        // If PDB output path was specified as a directory, put the MD5.pdb file
        // there.
        AbsoluteDebugName = DebugFile;
        sys::path::append(AbsoluteDebugName, DebugNameStr);
      } else {
        // Otherwise, use PDB output path as a user-provided PDB file name.
        DebugNameStr = DebugFile;
        AbsoluteDebugName = DebugNameStr;
      }

      // Pass PDB name to DXContainerPDBPass via PDBNAME section.
      addSection(M, Globals, AbsoluteDebugName, "dx.pdb.name",
                 PdbFileNameSectionName);
    }

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

static GlobalVariable *buildSignature(Module &M, Signature &Sig, StringRef Name,
                                      StringRef SectionName,
                                      const dxil::ModuleMetadataInfo &MMI) {
  SmallString<256> Data;
  raw_svector_ostream OS(Data);
  VersionTuple V = MMI.ValidatorVersion;
  bool Legacy = !V.empty() && V != VersionTuple(0, 0) && V < VersionTuple(1, 7);
  Sig.write(OS, /*PadToDword=*/!Legacy);
  Constant *Constant =
      ConstantDataArray::getString(M.getContext(), Data, /*AddNull*/ false);
  return buildContainerGlobal(M, Constant, Name, SectionName);
}

static void
addSignature(Module &M, SmallVector<GlobalValue *> &Globals,
             function_ref<dxil::ModuleMetadataInfo &()> GetModuleMetadataInfo,
             function_ref<const ModuleSignatureInfo &()> GetSignatureInfo) {
  const ModuleMetadataInfo &MMI = GetModuleMetadataInfo();
  const ModuleSignatureInfo &Signatures = GetSignatureInfo();
  Signature InputSig, OutputSig;
  if (MMI.ShaderProfile != Triple::Library && MMI.EntryPropertyVec.size() == 1)
    if (const auto *Sig = Signatures.get(MMI.EntryPropertyVec[0].Entry))
      Sig->buildSignatures(InputSig, OutputSig, MMI.ValidatorVersion);

  Globals.emplace_back(buildSignature(M, InputSig, "dx.isg1", "ISG1", MMI));
  Globals.emplace_back(buildSignature(M, OutputSig, "dx.osg1", "OSG1", MMI));
}

static void addRootSignature(
    Module &M, SmallVector<GlobalValue *> &Globals,
    function_ref<dxil::ModuleMetadataInfo &()> GetModuleMetadataInfo,
    function_ref<RootSignatureBindingInfo &()> GetRootSignatureBindingInfo) {

  dxil::ModuleMetadataInfo &MMI = GetModuleMetadataInfo();

  // Root Signature in Library don't compile to DXContainer.
  if (MMI.ShaderProfile == llvm::Triple::Library)
    return;

  RootSignatureBindingInfo &RSA = GetRootSignatureBindingInfo();
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

static void
addResourcesForPSV(Module &M, PSVRuntimeInfo &PSV,
                   function_ref<DXILResourceMap &()> GetResourceMap,
                   function_ref<DXILResourceTypeMap &()> GetResourceTypeMap) {
  const DXILResourceMap &DRM = GetResourceMap();
  DXILResourceTypeMap &DRTM = GetResourceTypeMap();

  auto MakeBinding = [](const dxil::ResourceInfo::ResourceBinding &Binding,
                        const dxbc::PSV::ResourceType Type,
                        const dxil::ResourceKind Kind,
                        const dxbc::PSV::ResourceFlags Flags =
                            dxbc::PSV::ResourceFlags()) {
    dxbc::PSV::v2::ResourceBindInfo BindInfo;
    BindInfo.Type = Type;
    BindInfo.LowerBound = Binding.LowerBound;
    assert((Binding.Size == 0 ||
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
    if (!RI.hasBinding())
      continue;
    const dxil::ResourceInfo::ResourceBinding &Binding = RI.getBinding();
    PSV.Resources.push_back(MakeBinding(Binding, dxbc::PSV::ResourceType::CBV,
                                        dxil::ResourceKind::CBuffer));
  }
  for (const dxil::ResourceInfo &RI : DRM.samplers()) {
    if (!RI.hasBinding())
      continue;
    const dxil::ResourceInfo::ResourceBinding &Binding = RI.getBinding();
    PSV.Resources.push_back(MakeBinding(Binding,
                                        dxbc::PSV::ResourceType::Sampler,
                                        dxil::ResourceKind::Sampler));
  }
  for (const dxil::ResourceInfo &RI : DRM.srvs()) {
    if (!RI.hasBinding())
      continue;
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
    if (!RI.hasBinding())
      continue;
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
    Flags.Bits.UsedByAtomic64 = RI.HasAtomic64Use;

    PSV.Resources.push_back(
        MakeBinding(Binding, ResType, TypeInfo.getResourceKind(), Flags));
  }
}

static void addPipelineStateValidationInfo(
    Module &M, SmallVector<GlobalValue *> &Globals,
    function_ref<dxil::ModuleMetadataInfo &()> GetModuleMetadataInfo,
    function_ref<DXILResourceMap &()> GetResourceMap,
    function_ref<DXILResourceTypeMap &()> GetResourceTypeMap,
    function_ref<const ModuleShaderFlags &()> GetModuleShaderFlags,
    function_ref<const ModuleSignatureInfo &()> GetModuleSignatureInfo) {
  SmallString<256> Data;
  raw_svector_ostream OS(Data);
  PSVRuntimeInfo PSV;
  PSV.BaseData.MinimumWaveLaneCount = 0;
  PSV.BaseData.MaximumWaveLaneCount = std::numeric_limits<uint32_t>::max();

  dxil::ModuleMetadataInfo &MMI = GetModuleMetadataInfo();
  assert(MMI.EntryPropertyVec.size() == 1 ||
         MMI.ShaderProfile == Triple::Library ||
         MMI.ShaderProfile == Triple::RootSignature);
  PSV.BaseData.ShaderStage =
      static_cast<uint8_t>(MMI.ShaderProfile - Triple::Pixel);

  addResourcesForPSV(M, PSV, GetResourceMap, GetResourceTypeMap);
  unsigned PSVVersion = 3;
  if (MMI.ShaderProfile != Triple::Library &&
      MMI.EntryPropertyVec.size() == 1) {
    const Function *Entry = MMI.EntryPropertyVec[0].Entry;
    bool UsesViewID = GetModuleShaderFlags().getFunctionFlags(Entry).ViewID;
    PSV.BaseData.UsesViewID = UsesViewID;
    const ModuleSignatureInfo &Signatures = GetModuleSignatureInfo();
    if (const auto *Sig = Signatures.get(Entry)) {
      Sig->updatePSV(PSV, MMI.ValidatorVersion, UsesViewID);
      // Preserve the existing empty-signature path, and select an ABI version
      // compatible with the validator for graphics signature data.
      VersionTuple V = MMI.ValidatorVersion;
      if ((!Sig->Inputs.empty() || !Sig->Outputs.empty()) && !V.empty() &&
          V != VersionTuple(0, 0))
        PSVVersion = V < VersionTuple(1, 1)   ? 0
                     : V < VersionTuple(1, 6) ? 1
                     : V < VersionTuple(1, 8) ? 2
                                              : 3;
    }
  }

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

  PSV.finalize(MMI.ShaderProfile, PSVVersion);
  PSV.write(OS, PSVVersion);
  addSection(M, Globals, Data, "dx.psv0", "PSV0");
}

static void addCompilerVersion(Module &M, SmallVector<GlobalValue *> &Globals) {
  if (M.debug_compile_units().empty())
    return;

  SmallString<256> Data;
  raw_svector_ostream OS(Data);
  mcdxbc::CompilerVersion CompilerVersion;
  CompilerVersion.write(OS);
  addSection(M, Globals, Data, "dx.vers", "VERS");
}

static void addSourceInfo(
    Module &M, SmallVector<GlobalValue *> &Globals,
    function_ref<dxil::ModuleMetadataInfo &()> GetModuleMetadataInfo) {
  dxil::ModuleMetadataInfo &MMI = GetModuleMetadataInfo();

  const auto &Opts = M.getContext().getLibraryOptions().get<DirectXMCOptions>();

  if (!MMI.SourceInfo || Opts.source_in_debug_module)
    return;

  MMI.SourceInfo->computeEntries();
  MMI.SourceInfo->finalize();
  SmallString<256> Data;
  raw_svector_ostream OS(Data);
  MMI.SourceInfo->write(OS);
  addSection(M, Globals, Data, "dx.srci", "SRCI");
}

static bool runContainerGlobalsOnModule(
    Module &M, function_ref<dxil::ModuleMetadataInfo &()> GetModuleMetadataInfo,
    function_ref<DXILResourceMap &()> GetResourceMap,
    function_ref<DXILResourceTypeMap &()> GetResourceTypeMap,
    function_ref<const ModuleShaderFlags &()> GetModuleShaderFlags,
    function_ref<const ModuleSignatureInfo &()> GetModuleSignatureInfo,
    function_ref<RootSignatureBindingInfo &()> GetRootSignatureBindingInfo) {
  llvm::SmallVector<GlobalValue *> Globals;
  Globals.push_back(getFeatureFlags(M, GetModuleShaderFlags));
  computeShaderHashAndDebugName(M, Globals);
  addSignature(M, Globals, GetModuleMetadataInfo, GetModuleSignatureInfo);
  addRootSignature(M, Globals, GetModuleMetadataInfo,
                   GetRootSignatureBindingInfo);
  addPipelineStateValidationInfo(M, Globals, GetModuleMetadataInfo,
                                 GetResourceMap, GetResourceTypeMap,
                                 GetModuleShaderFlags, GetModuleSignatureInfo);
  addCompilerVersion(M, Globals);
  addSourceInfo(M, Globals, GetModuleMetadataInfo);
  appendToCompilerUsed(M, Globals);
  return true;
}

bool DXILContainerGlobalsLegacy::runOnModule(Module &M) {
  return runContainerGlobalsOnModule(
      M,
      [&]() -> ModuleMetadataInfo & {
        return getAnalysis<DXILMetadataAnalysisWrapperPass>()
            .getModuleMetadata();
      },
      [&]() -> DXILResourceMap & {
        return getAnalysis<DXILResourceWrapperPass>().getResourceMap();
      },
      [&]() -> DXILResourceTypeMap & {
        return getAnalysis<DXILResourceTypeWrapperPass>().getResourceTypeMap();
      },
      [&]() -> const ModuleShaderFlags & {
        return getAnalysis<ShaderFlagsAnalysisWrapper>().getShaderFlags();
      },
      [&]() -> const ModuleSignatureInfo & {
        return getAnalysis<SignatureAnalysisWrapper>().getSignatureInfo();
      },
      [&]() -> RootSignatureBindingInfo & {
        return getAnalysis<RootSignatureAnalysisWrapper>().getRSInfo();
      });
}

PreservedAnalyses DXILContainerGlobalsPass::run(Module &M,
                                                ModuleAnalysisManager &MAM) {
  runContainerGlobalsOnModule(
      M,
      [&]() -> ModuleMetadataInfo & {
        return MAM.getResult<DXILMetadataAnalysis>(M);
      },
      [&]() -> DXILResourceMap & {
        return MAM.getResult<DXILResourceAnalysis>(M);
      },
      [&]() -> DXILResourceTypeMap & {
        return MAM.getResult<DXILResourceTypeAnalysis>(M);
      },
      [&]() -> const ModuleShaderFlags & {
        return MAM.getResult<ShaderFlagsAnalysis>(M);
      },
      [&]() -> const ModuleSignatureInfo & {
        return MAM.getResult<SignatureAnalysis>(M);
      },
      [&]() -> RootSignatureBindingInfo & {
        return MAM.getResult<RootSignatureAnalysis>(M);
      });
  return PreservedAnalyses::all();
}

char DXILContainerGlobalsLegacy::ID = 0;
INITIALIZE_PASS_BEGIN(DXILContainerGlobalsLegacy, "dxil-container-globals",
                      "DXContainer Global Emitter", false, true)
INITIALIZE_PASS_DEPENDENCY(ShaderFlagsAnalysisWrapper)
INITIALIZE_PASS_DEPENDENCY(SignatureAnalysisWrapper)
INITIALIZE_PASS_DEPENDENCY(DXILMetadataAnalysisWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DXILResourceTypeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DXILResourceWrapperPass)
INITIALIZE_PASS_END(DXILContainerGlobalsLegacy, "dxil-container-globals",
                    "DXContainer Global Emitter", false, true)

ModulePass *llvm::createDXILContainerGlobalsLegacyPass() {
  return new DXILContainerGlobalsLegacy();
}
