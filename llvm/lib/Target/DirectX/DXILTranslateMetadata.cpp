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
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/TargetParser/Triple.h"
#include <cstdint>

using namespace llvm;
using namespace llvm::dxil;

namespace {
/// A simple Wrapper DiagnosticInfo that generates Module-level diagnostic
/// for TranslateMetadata pass
class DiagnosticInfoTranslateMD : public DiagnosticInfo {
private:
  const Twine &Msg;
  const Module &Mod;

public:
  /// \p M is the module for which the diagnostic is being emitted. \p Msg is
  /// the message to show. Note that this class does not copy this message, so
  /// this reference must be valid for the whole life time of the diagnostic.
  DiagnosticInfoTranslateMD(const Module &M, const Twine &Msg,
                            DiagnosticSeverity Severity = DS_Error)
      : DiagnosticInfo(DK_Unsupported, Severity), Msg(Msg), Mod(M) {}

  void print(DiagnosticPrinter &DP) const override {
    DP << Mod.getName() << ": " << Msg << '\n';
  }
};

enum class EntryPropsTag {
  ShaderFlags = 0,
  GSState,
  DSState,
  HSState,
  NumThreads,
  AutoBindingSpace,
  RayPayloadSize,
  RayAttribSize,
  ShaderKind,
  MSState,
  ASStateTag,
  WaveSize,
  EntryRootSig,
};

} // namespace

static NamedMDNode *emitResourceMetadata(Module &M, const DXILResourceMap &DRM,
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
    return nullptr;

  NamedMDNode *ResourceMD = M.getOrInsertNamedMetadata("dx.resources");
  ResourceMD->addOperand(
      MDNode::get(M.getContext(), {SRVMD, UAVMD, CBufMD, SmpMD}));

  return ResourceMD;
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
}

static uint32_t getShaderStage(Triple::EnvironmentType Env) {
  return (uint32_t)Env - (uint32_t)llvm::Triple::Pixel;
}

static SmallVector<Metadata *>
getTagValueAsMetadata(EntryPropsTag Tag, uint64_t Value, LLVMContext &Ctx) {
  SmallVector<Metadata *> MDVals;
  MDVals.emplace_back(ConstantAsMetadata::get(
      ConstantInt::get(Type::getInt32Ty(Ctx), static_cast<int>(Tag))));
  switch (Tag) {
  case EntryPropsTag::ShaderFlags:
    MDVals.emplace_back(ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt64Ty(Ctx), Value)));
    break;
  case EntryPropsTag::ShaderKind:
    MDVals.emplace_back(ConstantAsMetadata::get(
        ConstantInt::get(Type::getInt32Ty(Ctx), Value)));
    break;
  case EntryPropsTag::GSState:
  case EntryPropsTag::DSState:
  case EntryPropsTag::HSState:
  case EntryPropsTag::NumThreads:
  case EntryPropsTag::AutoBindingSpace:
  case EntryPropsTag::RayPayloadSize:
  case EntryPropsTag::RayAttribSize:
  case EntryPropsTag::MSState:
  case EntryPropsTag::ASStateTag:
  case EntryPropsTag::WaveSize:
  case EntryPropsTag::EntryRootSig:
    llvm_unreachable("NYI: Unhandled entry property tag");
  }
  return MDVals;
}

static MDTuple *
getEntryPropAsMetadata(const EntryProperties &EP, uint64_t EntryShaderFlags,
                       const Triple::EnvironmentType ShaderProfile) {
  SmallVector<Metadata *> MDVals;
  LLVMContext &Ctx = EP.Entry->getContext();
  if (EntryShaderFlags != 0)
    MDVals.append(getTagValueAsMetadata(EntryPropsTag::ShaderFlags,
                                        EntryShaderFlags, Ctx));

  if (EP.Entry != nullptr) {
    // FIXME: support more props.
    // See https://github.com/llvm/llvm-project/issues/57948.
    // Add shader kind for lib entries.
    if (ShaderProfile == Triple::EnvironmentType::Library &&
        EP.ShaderStage != Triple::EnvironmentType::Library)
      MDVals.append(getTagValueAsMetadata(EntryPropsTag::ShaderKind,
                                          getShaderStage(EP.ShaderStage), Ctx));

    if (EP.ShaderStage == Triple::EnvironmentType::Compute) {
      MDVals.emplace_back(ConstantAsMetadata::get(ConstantInt::get(
          Type::getInt32Ty(Ctx), static_cast<int>(EntryPropsTag::NumThreads))));
      Metadata *NumThreadVals[] = {ConstantAsMetadata::get(ConstantInt::get(
                                       Type::getInt32Ty(Ctx), EP.NumThreadsX)),
                                   ConstantAsMetadata::get(ConstantInt::get(
                                       Type::getInt32Ty(Ctx), EP.NumThreadsY)),
                                   ConstantAsMetadata::get(ConstantInt::get(
                                       Type::getInt32Ty(Ctx), EP.NumThreadsZ))};
      MDVals.emplace_back(MDNode::get(Ctx, NumThreadVals));
    }
  }
  if (MDVals.empty())
    return nullptr;
  return MDNode::get(Ctx, MDVals);
}

MDTuple *constructEntryMetadata(const Function *EntryFn, MDTuple *Signatures,
                                MDNode *Resources, MDTuple *Properties,
                                LLVMContext &Ctx) {
  // Each entry point metadata record specifies:
  //  * reference to the entry point function global symbol
  //  * unmangled name
  //  * list of signatures
  //  * list of resources
  //  * list of tag-value pairs of shader capabilities and other properties
  Metadata *MDVals[5];
  MDVals[0] =
      EntryFn ? ValueAsMetadata::get(const_cast<Function *>(EntryFn)) : nullptr;
  MDVals[1] = MDString::get(Ctx, EntryFn ? EntryFn->getName() : "");
  MDVals[2] = Signatures;
  MDVals[3] = Resources;
  MDVals[4] = Properties;
  return MDNode::get(Ctx, MDVals);
}

static MDTuple *emitEntryMD(const EntryProperties &EP, MDTuple *Signatures,
                            MDNode *MDResources,
                            const uint64_t EntryShaderFlags,
                            const Triple::EnvironmentType ShaderProfile) {
  MDTuple *Properties =
      getEntryPropAsMetadata(EP, EntryShaderFlags, ShaderProfile);
  return constructEntryMetadata(EP.Entry, Signatures, MDResources, Properties,
                                EP.Entry->getContext());
}

static void emitValidatorVersionMD(Module &M, const ModuleMetadataInfo &MMDI) {
  if (MMDI.ValidatorVersion.empty())
    return;

  LLVMContext &Ctx = M.getContext();
  IRBuilder<> IRB(Ctx);
  Metadata *MDVals[2];
  MDVals[0] =
      ConstantAsMetadata::get(IRB.getInt32(MMDI.ValidatorVersion.getMajor()));
  MDVals[1] = ConstantAsMetadata::get(
      IRB.getInt32(MMDI.ValidatorVersion.getMinor().value_or(0)));
  NamedMDNode *ValVerNode = M.getOrInsertNamedMetadata("dx.valver");
  // Set validator version obtained from DXIL Metadata Analysis pass
  ValVerNode->clearOperands();
  ValVerNode->addOperand(MDNode::get(Ctx, MDVals));
}

static void emitShaderModelVersionMD(Module &M,
                                     const ModuleMetadataInfo &MMDI) {
  LLVMContext &Ctx = M.getContext();
  IRBuilder<> IRB(Ctx);
  Metadata *SMVals[3];
  VersionTuple SM = MMDI.ShaderModelVersion;
  SMVals[0] = MDString::get(Ctx, getShortShaderStage(MMDI.ShaderProfile));
  SMVals[1] = ConstantAsMetadata::get(IRB.getInt32(SM.getMajor()));
  SMVals[2] = ConstantAsMetadata::get(IRB.getInt32(SM.getMinor().value_or(0)));
  NamedMDNode *SMMDNode = M.getOrInsertNamedMetadata("dx.shaderModel");
  SMMDNode->addOperand(MDNode::get(Ctx, SMVals));
}

static void emitDXILVersionTupleMD(Module &M, const ModuleMetadataInfo &MMDI) {
  LLVMContext &Ctx = M.getContext();
  IRBuilder<> IRB(Ctx);
  VersionTuple DXILVer = MMDI.DXILVersion;
  Metadata *DXILVals[2];
  DXILVals[0] = ConstantAsMetadata::get(IRB.getInt32(DXILVer.getMajor()));
  DXILVals[1] =
      ConstantAsMetadata::get(IRB.getInt32(DXILVer.getMinor().value_or(0)));
  NamedMDNode *DXILVerMDNode = M.getOrInsertNamedMetadata("dx.version");
  DXILVerMDNode->addOperand(MDNode::get(Ctx, DXILVals));
}

static MDTuple *emitTopLevelLibraryNode(Module &M, MDNode *RMD,
                                        uint64_t ShaderFlags) {
  LLVMContext &Ctx = M.getContext();
  MDTuple *Properties = nullptr;
  if (ShaderFlags != 0) {
    SmallVector<Metadata *> MDVals;
    // FIXME: ShaderFlagsAnalysis pass needs to collect and provide
    // ShaderFlags for each entry function. Currently, ShaderFlags value
    // provided by ShaderFlagsAnalysis pass is created by walking *all* the
    // function instructions of the module. Is it is correct to use this value
    // for metadata of the empty library entry?
    MDVals.append(
        getTagValueAsMetadata(EntryPropsTag::ShaderFlags, ShaderFlags, Ctx));
    Properties = MDNode::get(Ctx, MDVals);
  }
  // Library has an entry metadata with resource table metadata and all other
  // MDNodes as null.
  return constructEntryMetadata(nullptr, nullptr, RMD, Properties, Ctx);
}

static void translateMetadata(Module &M, const DXILResourceMap &DRM,
                              const Resources &MDResources,
                              const ComputedShaderFlags &ShaderFlags,
                              const ModuleMetadataInfo &MMDI) {
  LLVMContext &Ctx = M.getContext();
  IRBuilder<> IRB(Ctx);
  SmallVector<MDNode *> EntryFnMDNodes;

  emitValidatorVersionMD(M, MMDI);
  emitShaderModelVersionMD(M, MMDI);
  emitDXILVersionTupleMD(M, MMDI);
  NamedMDNode *NamedResourceMD = emitResourceMetadata(M, DRM, MDResources);
  auto *ResourceMD =
      (NamedResourceMD != nullptr) ? NamedResourceMD->getOperand(0) : nullptr;
  // FIXME: Add support to construct Signatures
  // See https://github.com/llvm/llvm-project/issues/57928
  MDTuple *Signatures = nullptr;

  if (MMDI.ShaderProfile == Triple::EnvironmentType::Library)
    EntryFnMDNodes.emplace_back(
        emitTopLevelLibraryNode(M, ResourceMD, ShaderFlags));
  else if (MMDI.EntryPropertyVec.size() > 1) {
    M.getContext().diagnose(DiagnosticInfoTranslateMD(
        M, "Non-library shader: One and only one entry expected"));
  }

  for (const EntryProperties &EntryProp : MMDI.EntryPropertyVec) {
    // FIXME: ShaderFlagsAnalysis pass needs to collect and provide
    // ShaderFlags for each entry function. For now, assume shader flags value
    // of entry functions being compiled for lib_* shader profile viz.,
    // EntryPro.Entry is 0.
    uint64_t EntryShaderFlags =
        (MMDI.ShaderProfile == Triple::EnvironmentType::Library) ? 0
                                                                 : ShaderFlags;
    if (MMDI.ShaderProfile != Triple::EnvironmentType::Library) {
      if (EntryProp.ShaderStage != MMDI.ShaderProfile) {
        M.getContext().diagnose(DiagnosticInfoTranslateMD(
            M,
            "Shader stage '" +
                Twine(getShortShaderStage(EntryProp.ShaderStage) +
                      "' for entry '" + Twine(EntryProp.Entry->getName()) +
                      "' different from specified target profile '" +
                      Twine(Triple::getEnvironmentTypeName(MMDI.ShaderProfile) +
                            "'"))));
      }
    }
    EntryFnMDNodes.emplace_back(emitEntryMD(EntryProp, Signatures, ResourceMD,
                                            EntryShaderFlags,
                                            MMDI.ShaderProfile));
  }

  NamedMDNode *EntryPointsNamedMD =
      M.getOrInsertNamedMetadata("dx.entryPoints");
  for (auto *Entry : EntryFnMDNodes)
    EntryPointsNamedMD->addOperand(Entry);
}

PreservedAnalyses DXILTranslateMetadata::run(Module &M,
                                             ModuleAnalysisManager &MAM) {
  const DXILResourceMap &DRM = MAM.getResult<DXILResourceAnalysis>(M);
  const dxil::Resources &MDResources = MAM.getResult<DXILResourceMDAnalysis>(M);
  const ComputedShaderFlags &ShaderFlags =
      MAM.getResult<ShaderFlagsAnalysis>(M);
  const dxil::ModuleMetadataInfo MMDI = MAM.getResult<DXILMetadataAnalysis>(M);

  translateMetadata(M, DRM, MDResources, ShaderFlags, MMDI);

  return PreservedAnalyses::all();
}

namespace {
class DXILTranslateMetadataLegacy : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DXILTranslateMetadataLegacy() : ModulePass(ID) {}

  StringRef getPassName() const override { return "DXIL Translate Metadata"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DXILResourceWrapperPass>();
    AU.addRequired<DXILResourceMDWrapper>();
    AU.addRequired<ShaderFlagsAnalysisWrapper>();
    AU.addRequired<DXILMetadataAnalysisWrapperPass>();
    AU.addPreserved<DXILResourceWrapperPass>();
    AU.addPreserved<DXILResourceMDWrapper>();
    AU.addPreserved<DXILMetadataAnalysisWrapperPass>();
  }

  bool runOnModule(Module &M) override {
    const DXILResourceMap &DRM =
        getAnalysis<DXILResourceWrapperPass>().getResourceMap();
    const dxil::Resources &MDResources =
        getAnalysis<DXILResourceMDWrapper>().getDXILResource();
    const ComputedShaderFlags &ShaderFlags =
        getAnalysis<ShaderFlagsAnalysisWrapper>().getShaderFlags();
    dxil::ModuleMetadataInfo MMDI =
        getAnalysis<DXILMetadataAnalysisWrapperPass>().getModuleMetadata();

    translateMetadata(M, DRM, MDResources, ShaderFlags, MMDI);
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
