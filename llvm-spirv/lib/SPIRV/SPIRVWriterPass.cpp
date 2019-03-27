//===- SPIRVWriterPass.cpp - SPIRV writing pass -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// SPIRVWriterPass implementation.
//
//===----------------------------------------------------------------------===//

#include "SPIRVWriterPass.h"
#include "LLVMSPIRVLib.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
using namespace llvm;

PreservedAnalyses SPIRVWriterPass::run(Module &M) {
  // FIXME: at the moment LLVM/SPIR-V translation errors are ignored.
  std::string Err;
  writeSpirv(&M, OS, Err);
  return PreservedAnalyses::all();
}

namespace {
class WriteSPIRVPass : public ModulePass {
  std::ostream &OS; // std::ostream to print on
public:
  static char ID; // Pass identification, replacement for typeid
  explicit WriteSPIRVPass(std::ostream &O) : ModulePass(ID), OS(O) {}

  StringRef getPassName() const override { return "SPIRV Writer"; }

  bool runOnModule(Module &M) override {
    // FIXME: at the moment LLVM/SPIR-V translation errors are ignored.
    std::string Err;
    writeSpirv(&M, OS, Err);
    return false;
  }
};
} // namespace

char WriteSPIRVPass::ID = 0;

ModulePass *llvm::createSPIRVWriterPass(std::ostream &Str) {
  return new WriteSPIRVPass(Str);
}
