//===- MetalWriterPass.cpp - Emit a .metallib container -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MetalWriterPass.h"
#include "MetallibWriter.h"
#include "PointeeTypeMap.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static void writeMetallibImpl(Module &M, raw_pwrite_stream &OS) {
  // Reconstruct typed-pointer info into a side table (AIR v1 bitcode needs
  // typed pointers; the module itself stays opaque).
  metal::PointeeTypeMap PTM = metal::buildPointeeTypeMap(M);
  metal::writeMetallib(M, PTM, OS);
}

PreservedAnalyses MetalWriterPass::run(Module &M, ModuleAnalysisManager &AM) {
  writeMetallibImpl(M, OS);
  return PreservedAnalyses::all();
}

namespace {

class MetalWriterLegacyPass : public ModulePass {
  raw_pwrite_stream &OS;

public:
  static char ID;
  explicit MetalWriterLegacyPass(raw_pwrite_stream &Out)
      : ModulePass(ID), OS(Out) {}

  StringRef getPassName() const override { return "Metal metallib writer"; }

  bool runOnModule(Module &M) override {
    writeMetallibImpl(M, OS);
    return false;
  }
};

} // namespace

char MetalWriterLegacyPass::ID = 0;

ModulePass *llvm::createMetalWriterPass(raw_pwrite_stream &Out) {
  return new MetalWriterLegacyPass(Out);
}
