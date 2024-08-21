//===- DXILTranslateMetadata.cpp - Pass to emit DXIL metadata -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILTranslateMetadata.h"
#include "DXILResource.h"
#include "DXILResourceAnalysis.h"
#include "DXILShaderFlags.h"
#include "DirectX.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/TargetParser/Triple.h"
#include <cstdint>

using namespace llvm;
using namespace llvm::dxil;

static void emitResourceMetadata(Module &M, const DXILResourceMap &DRM,
                                 const dxil::Resources &MDResources) {
  LLVMContext &Context = M.getContext();

  SmallVector<Metadata *> SRVs, UAVs, CBufs, Smps;
  for (const ResourceInfo &RI : DRM.srvs())
    SRVs.push_back(RI.getAsMetadata(Context));
  for (const ResourceInfo &RI : DRM.uavs())
    UAVs.push_back(RI.getAsMetadata(Context));
  for (const ResourceInfo &RI : DRM.cbuffers())
    CBufs.push_back(RI.getAsMetadata(Context));
  for (const ResourceInfo &RI : DRM.samplers())
    Smps.push_back(RI.getAsMetadata(Context));

  Metadata *SRVMD = SRVs.empty() ? nullptr : MDNode::get(Context, SRVs);
  Metadata *UAVMD = UAVs.empty() ? nullptr : MDNode::get(Context, UAVs);
  Metadata *CBufMD = CBufs.empty() ? nullptr : MDNode::get(Context, CBufs);
  Metadata *SmpMD = Smps.empty() ? nullptr : MDNode::get(Context, Smps);
  bool HasResources = !DRM.empty();

  if (MDResources.hasUAVs()) {
    assert(!UAVMD && "Old and new UAV representations can't coexist");
    UAVMD = MDResources.writeUAVs(M);
    HasResources = true;
  }

  if (MDResources.hasCBuffers()) {
    assert(!CBufMD && "Old and new cbuffer representations can't coexist");
    CBufMD = MDResources.writeCBuffers(M);
    HasResources = true;
  }

  if (!HasResources)
    return;

  NamedMDNode *ResourceMD = M.getOrInsertNamedMetadata("dx.resources");
  ResourceMD->addOperand(
      MDNode::get(M.getContext(), {SRVMD, UAVMD, CBufMD, SmpMD}));
}

static StringRef getShortShaderStage(Triple::EnvironmentType Env) {
  switch (Env) {
  case Triple::Pixel:
    return "ps";
  case Triple::Vertex:
    return "vs";
  case Triple::Geometry:
    return "gs";
  case Triple::Hull:
    return "hs";
  case Triple::Domain:
    return "ds";
  case Triple::Compute:
    return "cs";
  case Triple::Library:
    return "lib";
  case Triple::Mesh:
    return "ms";
  case Triple::Amplification:
    return "as";
  default:
    break;
  }
  llvm_unreachable("Unsupported environment for DXIL generation.");
  return "";
}

static uint32_t getShaderStage(Triple::EnvironmentType Env) {
  return (uint32_t)Env - (uint32_t)llvm::Triple::Pixel;
}

struct ShaderEntryMDInfo : EntryProperties {

  enum EntryPropsTag {
    ShaderFlagsTag = 0,
    GSStateTag,
    DSStateTag,
    HSStateTag,
    NumThreadsTag,
    AutoBindingSpaceTag,
    RayPayloadSizeTag,
    RayAttribSizeTag,
    ShaderKindTag,
    MSStateTag,
    ASStateTag,
    WaveSizeTag,
    EntryRootSigTag,
  };

  ShaderEntryMDInfo(EntryProperties &EP, LLVMContext &C,
                    Triple::EnvironmentType SP, MDTuple *MDR = nullptr,
                    uint64_t ShaderFlags = 0)
      : EntryProperties(EP), Ctx(C), EntryShaderFlags(ShaderFlags),
        MDResources(MDR), ShaderProfile(SP) {};

  MDTuple *getAsMetadata() {
    MDTuple *Properties = constructEntryPropMetadata();
    // FIXME: Add support to construct Signatures
    // See https://github.com/llvm/llvm-project/issues/57928
    MDTuple *Signatures = nullptr;
    return constructEntryMetadata(Signatures, MDResources, Properties);
  }

private:
  LLVMContext &Ctx;
  // Shader Flags for the Entry - from ShadeFLagsAnalysis pass
  uint64_t EntryShaderFlags{0};
  MDTuple *MDResources{nullptr};
  Triple::EnvironmentType ShaderProfile{
      Triple::EnvironmentType::UnknownEnvironment};
  // Each entry point metadata record specifies:
  //  * reference to the entry point function global symbol
  //  * unmangled name
  //  * list of signatures
  //  * list of resources
  //  * list of tag-value pairs of shader capabilities and other properties

  MDTuple *constructEntryMetadata(MDTuple *Signatures, MDTuple *Resources,
                                  MDTuple *Properties) {
    Metadata *MDVals[5];
    MDVals[0] =
        Entry ? ValueAsMetadata::get(const_cast<Function *>(Entry)) : nullptr;
    MDVals[1] = MDString::get(Ctx, Entry ? Entry->getName() : "");
    MDVals[2] = Signatures;
    MDVals[3] = Resources;
    MDVals[4] = Properties;
    return MDNode::get(Ctx, MDVals);
  }

  SmallVector<Metadata *> getTagValueAsMetadata(EntryPropsTag Tag,
                                                uint64_t Value) {
    SmallVector<Metadata *> MDVals;
    MDVals.emplace_back(
        ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), Tag)));
    switch (Tag) {
    case ShaderFlagsTag:
      MDVals.emplace_back(ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt64Ty(Ctx), Value)));
      break;
    case ShaderKindTag:
      MDVals.emplace_back(ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt32Ty(Ctx), Value)));
      break;
    default:
      assert(false && "NYI: Unhandled entry property tag");
    }
    return MDVals;
  }

  MDTuple *constructEntryPropMetadata() {
    SmallVector<Metadata *> MDVals;
    if (EntryShaderFlags != 0)
      MDVals.append(getTagValueAsMetadata(ShaderFlagsTag, EntryShaderFlags));

    if (Entry != nullptr) {
      // FIXME: support more props.
      // See https://github.com/llvm/llvm-project/issues/57948.
      // Add shader kind for lib entries.
      if (ShaderProfile == Triple::EnvironmentType::Library &&
          ShaderStage != Triple::EnvironmentType::Library)
        MDVals.append(
            getTagValueAsMetadata(ShaderKindTag, getShaderStage(ShaderStage)));

      if (ShaderStage == Triple::EnvironmentType::Compute) {
        MDVals.emplace_back(ConstantAsMetadata::get(
            ConstantInt::get(Type::getInt32Ty(Ctx), NumThreadsTag)));
        std::vector<Metadata *> NumThreadVals;
        NumThreadVals.emplace_back(ConstantAsMetadata::get(
            ConstantInt::get(Type::getInt32Ty(Ctx), NumThreadsX)));
        NumThreadVals.emplace_back(ConstantAsMetadata::get(
            ConstantInt::get(Type::getInt32Ty(Ctx), NumThreadsY)));
        NumThreadVals.emplace_back(ConstantAsMetadata::get(
            ConstantInt::get(Type::getInt32Ty(Ctx), NumThreadsZ)));
        MDVals.emplace_back(MDNode::get(Ctx, NumThreadVals));
      }
    }
    if (MDVals.empty())
      return nullptr;
    return MDNode::get(Ctx, MDVals);
  }
};

static void createEntryMD(Module &M, const uint64_t ShaderFlags,
                          const dxil::ModuleMetadataInfo &MDAnalysisInfo) {
  auto &Ctx = M.getContext();
  // FIXME: generate metadata for resource.
  MDTuple *MDResources = nullptr;
  if (auto *NamedResources = M.getNamedMetadata("dx.resources"))
    MDResources = dyn_cast<MDTuple>(NamedResources->getOperand(0));

  std::vector<MDNode *> EntryFnMDNodes;
  switch (MDAnalysisInfo.ShaderProfile) {
  case Triple::EnvironmentType::Library: {
    // Library has an entry metadata with resource table metadata and all other
    // MDNodes as null.
    EntryProperties EP{};
    // FIXME: ShaderFlagsAnalysis pass needs to collect and provide ShaderFlags
    // for each entry function. Currently, ShaderFlags value provided by
    // ShaderFlagsAnalysis pass is created by walking *all* the function
    // instructions of the module. Is it is correct to use this value for
    // metadata of the empty library entry?
    ShaderEntryMDInfo EmptyFunEntryProps(EP, Ctx, MDAnalysisInfo.ShaderProfile,
                                         MDResources, ShaderFlags);
    MDTuple *EmptyMDT = EmptyFunEntryProps.getAsMetadata();
    EntryFnMDNodes.emplace_back(EmptyMDT);

    for (auto EntryProp : MDAnalysisInfo.EntryPropertyVec) {
      // FIXME: ShaderFlagsAnalysis pass needs to collect and provide
      // ShaderFlags for each entry function. For now, assume shader flags value
      // of entry functions being compiled for lib_* shader profile viz.,
      // EntryPro.Entry is 0.
      ShaderEntryMDInfo SEP(EntryProp, Ctx, MDAnalysisInfo.ShaderProfile,
                            nullptr, 0);
      MDTuple *EmptyMDT = SEP.getAsMetadata();
      EntryFnMDNodes.emplace_back(EmptyMDT);
    }
  } break;
  case Triple::EnvironmentType::Compute: {
    size_t NumEntries = MDAnalysisInfo.EntryPropertyVec.size();
    if (NumEntries > 0) {
      assert(NumEntries == 1 &&
             "Compute shader: One and only one entry expected");
      EntryProperties EntryProp = MDAnalysisInfo.EntryPropertyVec[0];
      // ShaderFlagsAnalysis pass needs to collect and provide ShaderFlags for
      // each entry function. Currently, even though the ShaderFlags value
      // provided by ShaderFlagsAnalysis pass is created by walking all the
      // function instructions of the module, it is sufficient to since there is
      // only one entry function in the module.
      ShaderEntryMDInfo SEP(EntryProp, Ctx, MDAnalysisInfo.ShaderProfile,
                            MDResources, ShaderFlags);
      MDTuple *EmptyMDT = SEP.getAsMetadata();
      EntryFnMDNodes.emplace_back(EmptyMDT);
    }
    break;
  }
  case Triple::EnvironmentType::Amplification:
  case Triple::EnvironmentType::Mesh:
  case Triple::EnvironmentType::Vertex:
  case Triple::EnvironmentType::Hull:
  case Triple::EnvironmentType::Domain:
  case Triple::EnvironmentType::Geometry:
  case Triple::EnvironmentType::Pixel: {
    size_t NumEntries = MDAnalysisInfo.EntryPropertyVec.size();
    if (NumEntries > 0) {
      assert(NumEntries == 1 && "non-lib profiles should only have one entry");
      EntryProperties EntryProp = MDAnalysisInfo.EntryPropertyVec[0];
      // ShaderFlagsAnalysis pass needs to collect and provide ShaderFlags for
      // each entry function. Currently, even though the ShaderFlags value
      // provided by ShaderFlagsAnalysis pass is created by walking all the
      // function instructions of the module, it is sufficient to since there is
      // only one entry function in the module.
      ShaderEntryMDInfo SEP(EntryProp, Ctx, MDAnalysisInfo.ShaderProfile,
                            MDResources, ShaderFlags);
      MDTuple *EmptyMDT = SEP.getAsMetadata();
      EntryFnMDNodes.emplace_back(EmptyMDT);
    }
  } break;
  default:
    assert(0 && "invalid profile");
    break;
  }

  NamedMDNode *EntryPointsNamedMD =
      M.getOrInsertNamedMetadata("dx.entryPoints");
  for (auto *Entry : EntryFnMDNodes)
    EntryPointsNamedMD->addOperand(Entry);
}

static void translateMetadata(Module &M, const DXILResourceMap &DRM,
                              const dxil::Resources &MDResources,
                              const ComputedShaderFlags &ShaderFlags,
                              const dxil::ModuleMetadataInfo &MDAnalysisInfo) {
  LLVMContext &Ctx = M.getContext();
  IRBuilder<> IRB(Ctx);
  if (MDAnalysisInfo.ValidatorVersion.empty()) {
    // Module has no metadata node signifying valid validator version.
    // Create metadata dx.valver node with version value of 1.0
    const VersionTuple DefaultValidatorVer{1, 0};
    Metadata *MDVals[2];
    MDVals[0] =
        ConstantAsMetadata::get(IRB.getInt32(DefaultValidatorVer.getMajor()));
    MDVals[1] = ConstantAsMetadata::get(
        IRB.getInt32(DefaultValidatorVer.getMinor().value_or(0)));
    NamedMDNode *ValVerNode = M.getOrInsertNamedMetadata("dx.valver");
    ValVerNode->addOperand(MDNode::get(Ctx, MDVals));
  }

  Metadata *SMVals[3];
  VersionTuple SM = MDAnalysisInfo.ShaderModelVersion;
  SMVals[0] =
      MDString::get(Ctx, getShortShaderStage(MDAnalysisInfo.ShaderProfile));
  SMVals[1] = ConstantAsMetadata::get(IRB.getInt32(SM.getMajor()));
  SMVals[2] = ConstantAsMetadata::get(IRB.getInt32(SM.getMinor().value_or(0)));
  NamedMDNode *SMMDNode = M.getOrInsertNamedMetadata("dx.shaderModel");
  SMMDNode->addOperand(MDNode::get(Ctx, SMVals));

  VersionTuple DXILVer = MDAnalysisInfo.DXILVersion;
  Metadata *DXILVals[2];
  DXILVals[0] = ConstantAsMetadata::get(IRB.getInt32(DXILVer.getMajor()));
  DXILVals[1] =
      ConstantAsMetadata::get(IRB.getInt32(DXILVer.getMinor().value_or(0)));
  NamedMDNode *DXILVerMDNode = M.getOrInsertNamedMetadata("dx.version");
  DXILVerMDNode->addOperand(MDNode::get(Ctx, DXILVals));

  emitResourceMetadata(M, DRM, MDResources);

  createEntryMD(M, static_cast<uint64_t>(ShaderFlags), MDAnalysisInfo);
}

PreservedAnalyses DXILTranslateMetadata::run(Module &M,
                                             ModuleAnalysisManager &MAM) {
  const DXILResourceMap &DRM = MAM.getResult<DXILResourceAnalysis>(M);
  const dxil::Resources &MDResources = MAM.getResult<DXILResourceMDAnalysis>(M);
  const ComputedShaderFlags &ShaderFlags =
      MAM.getResult<ShaderFlagsAnalysis>(M);
  const dxil::ModuleMetadataInfo MetadataInfo =
      MAM.getResult<DXILMetadataAnalysis>(M);

  translateMetadata(M, DRM, MDResources, ShaderFlags, MetadataInfo);

  return PreservedAnalyses::all();
}

namespace {
class DXILTranslateMetadataLegacy : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DXILTranslateMetadataLegacy() : ModulePass(ID) {}

  StringRef getPassName() const override { return "DXIL Translate Metadata"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<DXILResourceWrapperPass>();
    AU.addRequired<DXILResourceMDWrapper>();
    AU.addRequired<ShaderFlagsAnalysisWrapper>();
    AU.addRequired<DXILMetadataAnalysisWrapperPass>();
  }

  bool runOnModule(Module &M) override {
    const DXILResourceMap &DRM =
        getAnalysis<DXILResourceWrapperPass>().getResourceMap();
    const dxil::Resources &MDResources =
        getAnalysis<DXILResourceMDWrapper>().getDXILResource();
    const ComputedShaderFlags &ShaderFlags =
        getAnalysis<ShaderFlagsAnalysisWrapper>().getShaderFlags();
    dxil::ModuleMetadataInfo MetadataInfo =
        getAnalysis<DXILMetadataAnalysisWrapperPass>().getModuleMetadata();

    translateMetadata(M, DRM, MDResources, ShaderFlags, MetadataInfo);
    return true;
  }
};

} // namespace

char DXILTranslateMetadataLegacy::ID = 0;

ModulePass *llvm::createDXILTranslateMetadataLegacyPass() {
  return new DXILTranslateMetadataLegacy();
}

INITIALIZE_PASS_BEGIN(DXILTranslateMetadataLegacy, "dxil-translate-metadata",
                      "DXIL Translate Metadata", false, false)
INITIALIZE_PASS_DEPENDENCY(DXILResourceWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DXILResourceMDWrapper)
INITIALIZE_PASS_DEPENDENCY(ShaderFlagsAnalysisWrapper)
INITIALIZE_PASS_DEPENDENCY(DXILMetadataAnalysisWrapperPass)
INITIALIZE_PASS_END(DXILTranslateMetadataLegacy, "dxil-translate-metadata",
                    "DXIL Translate Metadata", false, false)
