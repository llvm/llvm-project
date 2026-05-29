//===- MetalEmbedderPass.cpp - Embed serialized Metal AIR -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Module pass that serializes the (already AIR-conformant) module into a
// monolithic .metallib byte blob and embeds it as a private global with
// section name ".metallib". The MetalLib MC ObjectWriter then writes those
// bytes verbatim to the -filetype=obj output. Mirrors DXIL's EmbedDXILPass.
//
//===----------------------------------------------------------------------===//

#include "MetalEmbedderPass.h"
#include "MetallibWriter.h"
#include "PointeeTypeMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <cstdint>
#include <vector>

using namespace llvm;

static void embedMetallibImpl(Module &M) {
  metal::PointeeTypeMap PTM = metal::buildPointeeTypeMap(M);
  std::vector<uint8_t> Bytes = metal::serializeMetallib(M, PTM);

  ArrayRef<uint8_t> Ref(Bytes.data(), Bytes.size());
  Constant *Init = ConstantDataArray::get(M.getContext(), Ref);

  auto *GV =
      new GlobalVariable(M, Init->getType(), /*isConstant=*/true,
                         GlobalValue::PrivateLinkage, Init, "metal.metallib");
  GV->setSection(".metallib");
  GV->setAlignment(Align(4));
  appendToCompilerUsed(M, {GV});
}

PreservedAnalyses MetalEmbedderPass::run(Module &M, ModuleAnalysisManager &AM) {
  embedMetallibImpl(M);
  return PreservedAnalyses::none();
}

namespace {

class MetalEmbedderLegacyPass : public ModulePass {
public:
  static char ID;
  MetalEmbedderLegacyPass() : ModulePass(ID) {
    initializeMetalEmbedderLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "Metal AIR Embedder"; }

  bool runOnModule(Module &M) override {
    embedMetallibImpl(M);
    return true;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};

} // namespace

char MetalEmbedderLegacyPass::ID = 0;
INITIALIZE_PASS(MetalEmbedderLegacyPass, "metal-embed", "Embed Metal AIR",
                false, true)

ModulePass *llvm::createMetalEmbedderPass() {
  return new MetalEmbedderLegacyPass();
}
