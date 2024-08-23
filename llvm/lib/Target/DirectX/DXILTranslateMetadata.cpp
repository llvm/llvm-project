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
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace llvm::dxil;

static void translateMetadata(Module &M, const dxil::Resources &MDResources,
                              const ComputedShaderFlags &ShaderFlags) {
  dxil::ValidatorVersionMD ValVerMD(M);
  if (ValVerMD.isEmpty())
    ValVerMD.update(VersionTuple(1, 0));
  dxil::createShaderModelMD(M);
  dxil::createDXILVersionMD(M);

  MDResources.write(M);

  dxil::createEntryMD(M, static_cast<uint64_t>(ShaderFlags));
}

PreservedAnalyses DXILTranslateMetadata::run(Module &M,
                                             ModuleAnalysisManager &MAM) {
  const dxil::Resources &MDResources = MAM.getResult<DXILResourceMDAnalysis>(M);
  const ComputedShaderFlags &ShaderFlags =
      MAM.getResult<ShaderFlagsAnalysis>(M);

  translateMetadata(M, MDResources, ShaderFlags);

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
    AU.addRequired<DXILResourceMDWrapper>();
    AU.addRequired<ShaderFlagsAnalysisWrapper>();
  }

  bool runOnModule(Module &M) override {
    const dxil::Resources &MDResources =
        getAnalysis<DXILResourceMDWrapper>().getDXILResource();
    const ComputedShaderFlags &ShaderFlags =
        getAnalysis<ShaderFlagsAnalysisWrapper>().getShaderFlags();

    translateMetadata(M, MDResources, ShaderFlags);
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
INITIALIZE_PASS_DEPENDENCY(DXILResourceMDWrapper)
INITIALIZE_PASS_DEPENDENCY(ShaderFlagsAnalysisWrapper)
INITIALIZE_PASS_END(DXILTranslateMetadataLegacy, "dxil-translate-metadata",
                    "DXIL Translate Metadata", false, false)
