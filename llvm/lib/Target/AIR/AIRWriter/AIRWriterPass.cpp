//===- AIRWriterPass.cpp - Emit a .metallib container -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIRWriterPass.h"
#include "AIRLibWriter.h"
#include "BitcodeEmitter.h"
#include "PointeeTypeMap.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static void writeAIRLibImpl(Module &M, raw_pwrite_stream &OS) {
  // Reconstruct typed-pointer info into a side table (AIR v1 bitcode needs
  // typed pointers; the module itself stays opaque).
  metal::lowerConstantExprs(M);
  metal::PointeeTypeMap PTM = metal::buildPointeeTypeMap(M);
  metal::writeAIRLib(M, PTM, OS);
}

PreservedAnalyses AIRWriterPass::run(Module &M, ModuleAnalysisManager &AM) {
  writeAIRLibImpl(M, OS);
  return PreservedAnalyses::all();
}

namespace {

class AIRWriterLegacyPass : public ModulePass {
  raw_pwrite_stream &OS;

public:
  static char ID;
  explicit AIRWriterLegacyPass(raw_pwrite_stream &Out)
      : ModulePass(ID), OS(Out) {}

  StringRef getPassName() const override { return "AIR metallib writer"; }

  bool runOnModule(Module &M) override {
    writeAIRLibImpl(M, OS);
    return false;
  }
};

} // namespace

char AIRWriterLegacyPass::ID = 0;

ModulePass *llvm::createAIRWriterPass(raw_pwrite_stream &Out) {
  return new AIRWriterLegacyPass(Out);
}
