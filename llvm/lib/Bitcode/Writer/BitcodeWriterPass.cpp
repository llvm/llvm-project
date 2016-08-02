//===- BitcodeWriterPass.cpp - Bitcode writing pass -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// BitcodeWriterPass implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
using namespace llvm;

PreservedAnalyses BitcodeWriterPass::run(Module &M) {
  WriteBitcodeToFile(&M, OS, ShouldPreserveUseListOrder, EmitFunctionSummary);
  return PreservedAnalyses::all();
}

namespace {
  class WriteBitcodePass : public ModulePass {
    raw_ostream &OS; // raw_ostream to print on
    bool ShouldPreserveUseListOrder;
    bool EmitFunctionSummary;

  public:
    static char ID; // Pass identification, replacement for typeid
    explicit WriteBitcodePass(raw_ostream &o, bool ShouldPreserveUseListOrder,
                              bool EmitFunctionSummary)
        : ModulePass(ID), OS(o),
          ShouldPreserveUseListOrder(ShouldPreserveUseListOrder),
          EmitFunctionSummary(EmitFunctionSummary) {}

    const char *getPassName() const override { return "Bitcode Writer"; }

    bool runOnModule(Module &M) override {
      WriteBitcodeToFile(&M, OS, ShouldPreserveUseListOrder,
                         EmitFunctionSummary);
      return false;
    }
  };
}

char WriteBitcodePass::ID = 0;

ModulePass *llvm::createBitcodeWriterPass(raw_ostream &Str,
                                          bool ShouldPreserveUseListOrder,
                                          bool EmitFunctionSummary) {
  return new WriteBitcodePass(Str, ShouldPreserveUseListOrder,
                              EmitFunctionSummary);
}
