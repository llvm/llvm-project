//===- AIREmbedderPass.cpp - Embed serialized AIR -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Module pass that serializes the (already AIR-conformant) module into a
// monolithic .metallib byte blob and embeds it as a private global with
// section name ".metallib". The AIRLib MC ObjectWriter then writes those
// bytes verbatim to the -filetype=obj output. Mirrors DXIL's EmbedDXILPass.
//
//===----------------------------------------------------------------------===//

#include "AIREmbedderPass.h"
#include "AIRLibWriter.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/GlobalVariable.h"
#include "BitcodeEmitter.h"
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

static void emitTGBytesRemark(Module &M) {
  Function *F = nullptr;
  for (Function &Fn : M)
    if (!Fn.isDeclaration()) {
      F = &Fn;
      break;
    }
  if (!F || F->empty())
    return;
  const DataLayout &DL = M.getDataLayout();
  uint64_t Total = 0;
  for (const GlobalVariable &GV : M.globals()) {
    if (GV.getAddressSpace() != 3 || GV.isDeclaration())
      continue;
    uint64_t AlignB = GV.getAlign().value_or(Align(1)).value();
    if (Total % AlignB)
      Total += AlignB - (Total % AlignB);
    Total += DL.getTypeAllocSize(GV.getValueType());
  }
  OptimizationRemark R("metal-tg", "TGBytes", DebugLoc(), &F->getEntryBlock());
  R << "threadgroup memory total "
    << DiagnosticInfoOptimizationBase::Argument("TGBytes", Total);
  F->getContext().diagnose(R);
}

static void embedAIRLibImpl(Module &M) {
  metal::lowerConstantExprs(M);
  metal::PointeeTypeMap PTM = metal::buildPointeeTypeMap(M);
  emitTGBytesRemark(M);
  std::vector<uint8_t> Bytes = metal::serializeAIRLib(M, PTM);

  ArrayRef<uint8_t> Ref(Bytes.data(), Bytes.size());
  Constant *Init = ConstantDataArray::get(M.getContext(), Ref);

  auto *GV =
      new GlobalVariable(M, Init->getType(), /*isConstant=*/true,
                         GlobalValue::PrivateLinkage, Init, "metal.metallib");
  GV->setSection(".metallib");
  GV->setAlignment(Align(4));
  appendToCompilerUsed(M, {GV});
}

PreservedAnalyses AIREmbedderPass::run(Module &M, ModuleAnalysisManager &AM) {
  embedAIRLibImpl(M);
  return PreservedAnalyses::none();
}

namespace {

class AIREmbedderLegacyPass : public ModulePass {
public:
  static char ID;
  AIREmbedderLegacyPass() : ModulePass(ID) {
    initializeAIREmbedderLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "AIR Embedder"; }

  bool runOnModule(Module &M) override {
    embedAIRLibImpl(M);
    return true;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};

} // namespace

char AIREmbedderLegacyPass::ID = 0;
INITIALIZE_PASS(AIREmbedderLegacyPass, "air-embed", "Embed AIR",
                false, true)

ModulePass *llvm::createAIREmbedderPass() {
  return new AIREmbedderLegacyPass();
}
