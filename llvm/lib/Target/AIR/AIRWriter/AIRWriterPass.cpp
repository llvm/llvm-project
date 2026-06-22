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
#include "ConstantExprLower.h"
#include "PointeeTypeMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

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
  uint64_t SmemHigh = 0, SmemAlign = 0;
  for (const GlobalVariable &GV : M.globals()) {
    if (GV.getAddressSpace() != 3 || GV.isDeclaration())
      continue;
    StringRef Name = GV.getName();
    if (Name.starts_with("global_smem")) {
      uint64_t Off = 0;
      if (size_t P = Name.find("__off"); P != StringRef::npos)
        Name.substr(P + 5).consumeInteger(10, Off);
      SmemHigh = std::max(SmemHigh, Off + DL.getTypeAllocSize(GV.getValueType()));
      SmemAlign = std::max(SmemAlign, GV.getAlign().value_or(Align(1)).value());
      continue;
    }
    uint64_t AlignB = GV.getAlign().value_or(Align(1)).value();
    if (Total % AlignB)
      Total += AlignB - (Total % AlignB);
    Total += DL.getTypeAllocSize(GV.getValueType());
  }
  if (SmemHigh) {
    if (Total % SmemAlign)
      Total += SmemAlign - (Total % SmemAlign);
    Total += SmemHigh;
  }
  OptimizationRemark R("metal-tg", "TGBytes", DebugLoc(), &F->getEntryBlock());
  R << "threadgroup memory total "
    << DiagnosticInfoOptimizationBase::Argument("TGBytes", Total);
  F->getContext().diagnose(R);
}

static void writeAIRLibImpl(Module &M, raw_pwrite_stream &OS) {
  // Reconstruct typed-pointer info into a side table (AIR v1 bitcode needs
  // typed pointers; the module itself stays opaque).
  metal::lowerConstantExprs(M);
  metal::PointeeTypeMap PTM = metal::buildPointeeTypeMap(M);
  emitTGBytesRemark(M);
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
