//===- DXILTranslateMetadata.cpp - Pass to emit DXIL metadata -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILTranslateMetadata.h"
#include "DXILRootSignature.h"
#include "DXILShaderFlags.h"
#include "DirectX.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
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

/// A simple wrapper of DiagnosticInfo that generates module-level diagnostic
/// for the DXILValidateMetadata pass
class DiagnosticInfoValidateMD : public DiagnosticInfo {
private:
  const Twine &Msg;
  const Module &Mod;

public:
  /// \p M is the module for which the diagnostic is being emitted. \p Msg is
  /// the message to show. Note that this class does not copy this message, so
  /// this reference must be valid for the whole life time of the diagnostic.
  DiagnosticInfoValidateMD(const Module &M,
                           const Twine &Msg LLVM_LIFETIME_BOUND,
                           DiagnosticSeverity Severity = DS_Error)
      : DiagnosticInfo(DK_Unsupported, Severity), Msg(Msg), Mod(M) {}

  void print(DiagnosticPrinter &DP) const override {
    DP << Mod.getName() << ": " << Msg << '\n';
  }
};

static bool reportError(Module &M, Twine Message,
                        DiagnosticSeverity Severity = DS_Error) {
  M.getContext().diagnose(DiagnosticInfoValidateMD(M, Message, Severity));
  return true;
}

static bool reportLoopError(Module &M, Twine Message,
                            DiagnosticSeverity Severity = DS_Error) {
  return reportError(M, Twine("Invalid \"llvm.loop\" metadata: ") + Message,
                     Severity);
}

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

static NamedMDNode *emitResourceMetadata(Module &M, DXILResourceMap &DRM,
                                         DXILResourceTypeMap &DRTM) {
  LLVMContext &Context = M.getContext();

  for (ResourceInfo &RI : DRM)
    if (!RI.hasSymbol())
      RI.createSymbol(M,
                      DRTM[RI.getHandleTy()].createElementStruct(RI.getName()));

  SmallVector<Metadata *> SRVs, UAVs, CBufs, Smps;
  for (const ResourceInfo &RI : DRM.srvs())
    SRVs.push_back(RI.getAsMetadata(M, DRTM[RI.getHandleTy()]));
  for (const ResourceInfo &RI : DRM.uavs())
    UAVs.push_back(RI.getAsMetadata(M, DRTM[RI.getHandleTy()]));
  for (const ResourceInfo &RI : DRM.cbuffers())
    CBufs.push_back(RI.getAsMetadata(M, DRTM[RI.getHandleTy()]));
  for (const ResourceInfo &RI : DRM.samplers())
    Smps.push_back(RI.getAsMetadata(M, DRTM[RI.getHandleTy()]));

  Metadata *SRVMD = SRVs.empty() ? nullptr : MDNode::get(Context, SRVs);
  Metadata *UAVMD = UAVs.empty() ? nullptr : MDNode::get(Context, UAVs);
  Metadata *CBufMD = CBufs.empty() ? nullptr : MDNode::get(Context, CBufs);
  Metadata *SmpMD = Smps.empty() ? nullptr : MDNode::get(Context, Smps);

  if (DRM.empty())
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
  case Triple::RootSignature:
    return "rootsig";
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

static MDTuple *constructEntryMetadata(const Function *EntryFn,
                                       MDTuple *Signatures, MDNode *Resources,
                                       MDTuple *Properties, LLVMContext &Ctx) {
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
    MDVals.append(
        getTagValueAsMetadata(EntryPropsTag::ShaderFlags, ShaderFlags, Ctx));
    Properties = MDNode::get(Ctx, MDVals);
  }
  // Library has an entry metadata with resource table metadata and all other
  // MDNodes as null.
  return constructEntryMetadata(nullptr, nullptr, RMD, Properties, Ctx);
}

static void translateBranchMetadata(Module &M, Instruction *BBTerminatorInst) {
  MDNode *HlslControlFlowMD =
      BBTerminatorInst->getMetadata("hlsl.controlflow.hint");

  if (!HlslControlFlowMD)
    return;

  assert(HlslControlFlowMD->getNumOperands() == 2 &&
         "invalid operands for hlsl.controlflow.hint");

  MDBuilder MDHelper(M.getContext());

  llvm::Metadata *HintsStr = MDHelper.createString("dx.controlflow.hints");
  llvm::Metadata *HintsValue = MDHelper.createConstant(
      mdconst::extract<ConstantInt>(HlslControlFlowMD->getOperand(1)));

  MDNode *MDNode = llvm::MDNode::get(M.getContext(), {HintsStr, HintsValue});

  BBTerminatorInst->setMetadata("dx.controlflow.hints", MDNode);
  BBTerminatorInst->setMetadata("hlsl.controlflow.hint", nullptr);
}

static void translateLoopMetadata(Module &M, MDNode *LoopMD) {
  // DXIL only accepts the following loop hints:
  //   llvm.loop.unroll.disable, llvm.loop.unroll.full, llvm.loop.unroll.count
  std::array<StringLiteral, 3> ValidHintNames = {"llvm.loop.unroll.count",
                                                 "llvm.loop.unroll.disable",
                                                 "llvm.loop.unroll.full"};

  // llvm.loop metadata must have its first operand be a self-reference, so we
  // require at least 1 operand.
  //
  // It only makes sense to specify up to 1 of the hints on a branch, so we can
  // have at most 2 operands.

  if (LoopMD->getNumOperands() != 1 && LoopMD->getNumOperands() != 2) {
    reportLoopError(M, "Requires exactly 1 or 2 operands");
    return;
  }

  if (LoopMD != LoopMD->getOperand(0)) {
    reportLoopError(M, "First operand must be a self-reference");
    return;
  }

  // A node only containing a self-reference is a valid use to denote a loop
  if (LoopMD->getNumOperands() == 1)
    return;

  LoopMD = dyn_cast<MDNode>(LoopMD->getOperand(1));
  if (!LoopMD) {
    reportLoopError(M, "Second operand must be a metadata node");
    return;
  }

  if (LoopMD->getNumOperands() != 1 && LoopMD->getNumOperands() != 2) {
    reportLoopError(M, "Requires exactly 1 or 2 operands");
    return;
  }

  // It is valid to have a chain of self-referential loop metadata nodes so if
  // we have another self-reference, recurse.
  //
  // Eg:
  // !0 = !{!0, !1}
  // !1 = !{!1, !2}
  // !2 = !{"llvm.loop.unroll.disable"}
  if (LoopMD == LoopMD->getOperand(0))
    return translateLoopMetadata(M, LoopMD);

  // Otherwise, we are at our base hint metadata node
  auto *HintStr = dyn_cast<MDString>(LoopMD->getOperand(0));
  if (!HintStr || !llvm::is_contained(ValidHintNames, HintStr->getString())) {
    reportLoopError(M,
                    "First operand must be a valid \"llvm.loop.unroll\" hint");
    return;
  }

  // Ensure count node is a constant integer value
  auto ValidCountNode = [](MDNode *HintMD) -> bool {
    if (HintMD->getNumOperands() == 2)
      if (auto *CountMD = dyn_cast<ConstantAsMetadata>(HintMD->getOperand(1)))
        if (isa<ConstantInt>(CountMD->getValue()))
          return true;
    return false;
  };

  if (HintStr->getString() == "llvm.loop.unroll.count") {
    if (!ValidCountNode(LoopMD)) {
      reportLoopError(M, "Second operand of \"llvm.loop.unroll.count\" "
                         "must be a constant integer");
      return;
    }
  } else if (LoopMD->getNumOperands() != 1) {
    reportLoopError(M, "Can't have a second operand");
    return;
  }
}

using InstructionMDList = std::array<unsigned, 7>;

static InstructionMDList getCompatibleInstructionMDs(llvm::Module &M) {
  return {
      M.getMDKindID("dx.nonuniform"),    M.getMDKindID("dx.controlflow.hints"),
      M.getMDKindID("dx.precise"),       llvm::LLVMContext::MD_range,
      llvm::LLVMContext::MD_alias_scope, llvm::LLVMContext::MD_noalias,
      M.getMDKindID("llvm.loop")};
}

static void translateInstructionMetadata(Module &M) {
  // construct allowlist of valid metadata node kinds
  InstructionMDList DXILCompatibleMDs = getCompatibleInstructionMDs(M);
  unsigned char MDLoopKind = M.getContext().getMDKindID("llvm.loop");

  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      // This needs to be done first so that "hlsl.controlflow.hints" isn't
      // removed in the whitelist below
      if (auto *I = BB.getTerminator())
        translateBranchMetadata(M, I);

      for (auto &I : make_early_inc_range(BB)) {
        if (isa<BranchInst>(I)) {
          if (MDNode *LoopMD = I.getMetadata(MDLoopKind))
            translateLoopMetadata(M, LoopMD);
        }
        I.dropUnknownNonDebugMetadata(DXILCompatibleMDs);
      }
    }
  }
}

static void cleanModuleFlags(Module &M) {
  NamedMDNode *MDFlags = M.getModuleFlagsMetadata();
  if (!MDFlags)
    return;

  SmallVector<llvm::Module::ModuleFlagEntry> FlagEntries;
  M.getModuleFlagsMetadata(FlagEntries);
  bool Updated = false;
  for (auto &Flag : FlagEntries) {
    // llvm 3.7 only supports behavior up to AppendUnique.
    if (Flag.Behavior <= Module::ModFlagBehavior::AppendUnique)
      continue;
    Flag.Behavior = Module::ModFlagBehavior::Warning;
    Updated = true;
  }

  if (!Updated)
    return;

  MDFlags->eraseFromParent();

  for (auto &Flag : FlagEntries)
    M.addModuleFlag(Flag.Behavior, Flag.Key->getString(), Flag.Val);
}

static void translateGlobalMetadata(Module &M, DXILResourceMap &DRM,
                                    DXILResourceTypeMap &DRTM,
                                    const ModuleShaderFlags &ShaderFlags,
                                    const ModuleMetadataInfo &MMDI) {
  LLVMContext &Ctx = M.getContext();
  IRBuilder<> IRB(Ctx);
  SmallVector<MDNode *> EntryFnMDNodes;

  emitValidatorVersionMD(M, MMDI);
  emitShaderModelVersionMD(M, MMDI);
  emitDXILVersionTupleMD(M, MMDI);
  NamedMDNode *NamedResourceMD = emitResourceMetadata(M, DRM, DRTM);
  auto *ResourceMD =
      (NamedResourceMD != nullptr) ? NamedResourceMD->getOperand(0) : nullptr;
  // FIXME: Add support to construct Signatures
  // See https://github.com/llvm/llvm-project/issues/57928
  MDTuple *Signatures = nullptr;

  if (MMDI.ShaderProfile == Triple::EnvironmentType::Library) {
    // Get the combined shader flag mask of all functions in the library to be
    // used as shader flags mask value associated with top-level library entry
    // metadata.
    uint64_t CombinedMask = ShaderFlags.getCombinedFlags();
    EntryFnMDNodes.emplace_back(
        emitTopLevelLibraryNode(M, ResourceMD, CombinedMask));
  } else if (1 < MMDI.EntryPropertyVec.size())
    reportError(M, "Non-library shader: One and only one entry expected");

  for (const EntryProperties &EntryProp : MMDI.EntryPropertyVec) {
    uint64_t EntryShaderFlags = 0;
    if (MMDI.ShaderProfile != Triple::EnvironmentType::Library) {
      EntryShaderFlags = ShaderFlags.getFunctionFlags(EntryProp.Entry);
      if (EntryProp.ShaderStage != MMDI.ShaderProfile)
        reportError(
            M,
            "Shader stage '" +
                Twine(Twine(getShortShaderStage(EntryProp.ShaderStage)) +
                      "' for entry '" + Twine(EntryProp.Entry->getName()) +
                      "' different from specified target profile '" +
                      Twine(Triple::getEnvironmentTypeName(MMDI.ShaderProfile) +
                            "'")));
    }

    EntryFnMDNodes.emplace_back(emitEntryMD(EntryProp, Signatures, ResourceMD,
                                            EntryShaderFlags,
                                            MMDI.ShaderProfile));
  }

  NamedMDNode *EntryPointsNamedMD =
      M.getOrInsertNamedMetadata("dx.entryPoints");
  for (auto *Entry : EntryFnMDNodes)
    EntryPointsNamedMD->addOperand(Entry);

  cleanModuleFlags(M);

  // dx.rootsignatures will have been parsed from its metadata form as its
  // binary form as part of the RootSignatureAnalysisWrapper, so safely
  // remove it as it is not recognized in DXIL
  if (NamedMDNode *RootSignature = M.getNamedMetadata("dx.rootsignatures"))
    RootSignature->eraseFromParent();

  // llvm.errno.tbaa was recently added but is not supported in LLVM 3.7 and
  // causes all tests using the DXIL Validator to fail.
  //
  // This is a temporary fix and should be replaced with a allowlist once
  // we have determined all metadata that the DXIL Validator allows
  if (NamedMDNode *ErrNo = M.getNamedMetadata("llvm.errno.tbaa"))
    ErrNo->eraseFromParent();
}

PreservedAnalyses DXILTranslateMetadata::run(Module &M,
                                             ModuleAnalysisManager &MAM) {
  DXILResourceMap &DRM = MAM.getResult<DXILResourceAnalysis>(M);
  DXILResourceTypeMap &DRTM = MAM.getResult<DXILResourceTypeAnalysis>(M);
  const ModuleShaderFlags &ShaderFlags = MAM.getResult<ShaderFlagsAnalysis>(M);
  const dxil::ModuleMetadataInfo MMDI = MAM.getResult<DXILMetadataAnalysis>(M);

  translateGlobalMetadata(M, DRM, DRTM, ShaderFlags, MMDI);
  translateInstructionMetadata(M);

  return PreservedAnalyses::all();
}

void DXILTranslateMetadataLegacy::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DXILResourceTypeWrapperPass>();
  AU.addRequired<DXILResourceWrapperPass>();
  AU.addRequired<ShaderFlagsAnalysisWrapper>();
  AU.addRequired<DXILMetadataAnalysisWrapperPass>();
  AU.addRequired<RootSignatureAnalysisWrapper>();

  AU.addPreserved<DXILMetadataAnalysisWrapperPass>();
  AU.addPreserved<DXILResourceBindingWrapperPass>();
  AU.addPreserved<DXILResourceWrapperPass>();
  AU.addPreserved<RootSignatureAnalysisWrapper>();
  AU.addPreserved<ShaderFlagsAnalysisWrapper>();
}

bool DXILTranslateMetadataLegacy::runOnModule(Module &M) {
  DXILResourceMap &DRM =
      getAnalysis<DXILResourceWrapperPass>().getResourceMap();
  DXILResourceTypeMap &DRTM =
      getAnalysis<DXILResourceTypeWrapperPass>().getResourceTypeMap();
  const ModuleShaderFlags &ShaderFlags =
      getAnalysis<ShaderFlagsAnalysisWrapper>().getShaderFlags();
  dxil::ModuleMetadataInfo MMDI =
      getAnalysis<DXILMetadataAnalysisWrapperPass>().getModuleMetadata();

  translateGlobalMetadata(M, DRM, DRTM, ShaderFlags, MMDI);
  translateInstructionMetadata(M);
  return true;
}

char DXILTranslateMetadataLegacy::ID = 0;

ModulePass *llvm::createDXILTranslateMetadataLegacyPass() {
  return new DXILTranslateMetadataLegacy();
}

INITIALIZE_PASS_BEGIN(DXILTranslateMetadataLegacy, "dxil-translate-metadata",
                      "DXIL Translate Metadata", false, false)
INITIALIZE_PASS_DEPENDENCY(DXILResourceWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ShaderFlagsAnalysisWrapper)
INITIALIZE_PASS_DEPENDENCY(RootSignatureAnalysisWrapper)
INITIALIZE_PASS_DEPENDENCY(DXILMetadataAnalysisWrapperPass)
INITIALIZE_PASS_END(DXILTranslateMetadataLegacy, "dxil-translate-metadata",
                    "DXIL Translate Metadata", false, false)
