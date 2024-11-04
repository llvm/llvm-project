//===- DXILTranslateMetadata.cpp - Pass to emit DXIL metadata -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILTranslateMetadata.h"
#include "DXILMetadata.h"
#include "DXILResource.h"
#include "DXILResourceAnalysis.h"
#include "DXILShaderFlags.h"
#include "DirectX.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/TargetParser/Triple.h"

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

static void translateMetadata(Module &M, const DXILResourceMap &DRM,
                              const dxil::Resources &MDResources,
                              const ComputedShaderFlags &ShaderFlags) {
  dxil::ValidatorVersionMD ValVerMD(M);
  if (ValVerMD.isEmpty())
    ValVerMD.update(VersionTuple(1, 0));
  dxil::createShaderModelMD(M);
  dxil::createDXILVersionMD(M);

  emitResourceMetadata(M, DRM, MDResources);

  dxil::createEntryMD(M, static_cast<uint64_t>(ShaderFlags));
}

PreservedAnalyses DXILTranslateMetadata::run(Module &M,
                                             ModuleAnalysisManager &MAM) {
  const DXILResourceMap &DRM = MAM.getResult<DXILResourceAnalysis>(M);
  const dxil::Resources &MDResources = MAM.getResult<DXILResourceMDAnalysis>(M);
  const ComputedShaderFlags &ShaderFlags =
      MAM.getResult<ShaderFlagsAnalysis>(M);

  translateMetadata(M, DRM, MDResources, ShaderFlags);

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
  }

  bool runOnModule(Module &M) override {
    const DXILResourceMap &DRM =
        getAnalysis<DXILResourceWrapperPass>().getResourceMap();
    const dxil::Resources &MDResources =
        getAnalysis<DXILResourceMDWrapper>().getDXILResource();
    const ComputedShaderFlags &ShaderFlags =
        getAnalysis<ShaderFlagsAnalysisWrapper>().getShaderFlags();

    translateMetadata(M, DRM, MDResources, ShaderFlags);
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
INITIALIZE_PASS_END(DXILTranslateMetadataLegacy, "dxil-translate-metadata",
                    "DXIL Translate Metadata", false, false)
