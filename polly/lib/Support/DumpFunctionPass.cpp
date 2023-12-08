//===------ DumpFunctionPass.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Write a function to a file.
//
//===----------------------------------------------------------------------===//

#include "polly/Support/DumpFunctionPass.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/Utils/Cloning.h"

#define DEBUG_TYPE "polly-dump-func"

using namespace llvm;
using namespace polly;

namespace {

static void runDumpFunction(llvm::Function &F, StringRef Suffix) {
  StringRef FName = F.getName();
  Module *M = F.getParent();

  StringRef ModuleName = M->getName();
  StringRef Stem = sys::path::stem(ModuleName);
  std::string Dumpfile = (Twine(Stem) + "-" + FName + Suffix + ".ll").str();
  LLVM_DEBUG(dbgs() << "Dumping function '" << FName << "' to '" << Dumpfile
                    << "'...\n");

  ValueToValueMapTy VMap;
  auto ShouldCloneDefinition = [&F](const GlobalValue *GV) -> bool {
    return GV == &F;
  };
  std::unique_ptr<Module> CM = CloneModule(*M, VMap, ShouldCloneDefinition);
  Function *NewF = cast<Function>(VMap.lookup(&F));
  assert(NewF && "Expected selected function to be cloned");

  LLVM_DEBUG(dbgs() << "Global DCE...\n");

  // Stop F itself from being pruned
  GlobalValue::LinkageTypes OrigLinkage = NewF->getLinkage();
  NewF->setLinkage(GlobalValue::ExternalLinkage);

  {
    ModuleAnalysisManager MAM;
    ModulePassManager MPM;

    PassInstrumentationCallbacks PIC;
    MAM.registerPass([&] { return PassInstrumentationAnalysis(&PIC); });

    MPM.addPass(GlobalDCEPass());
    MPM.addPass(StripDeadPrototypesPass());
    MPM.run(*CM, MAM);
  }

  // Restore old linkage
  NewF->setLinkage(OrigLinkage);

  LLVM_DEBUG(dbgs() << "Write to file '" << Dumpfile << "'...\n");

  std::unique_ptr<ToolOutputFile> Out;
  std::error_code EC;
  Out.reset(new ToolOutputFile(Dumpfile, EC, sys::fs::OF_None));
  if (EC) {
    errs() << EC.message() << '\n';
    return;
  }

  CM->print(Out->os(), nullptr);
  Out->keep();
  LLVM_DEBUG(dbgs() << "Dump file " << Dumpfile << " written successfully\n");
}

class DumpFunctionWrapperPass final : public FunctionPass {
private:
  DumpFunctionWrapperPass(const DumpFunctionWrapperPass &) = delete;
  const DumpFunctionWrapperPass &
  operator=(const DumpFunctionWrapperPass &) = delete;

  std::string Suffix;

public:
  static char ID;

  explicit DumpFunctionWrapperPass() : FunctionPass(ID), Suffix("-dump") {}

  explicit DumpFunctionWrapperPass(std::string Suffix)
      : FunctionPass(ID), Suffix(std::move(Suffix)) {}

  /// @name FunctionPass interface
  //@{
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool runOnFunction(llvm::Function &F) override {
    runDumpFunction(F, Suffix);
    return false;
  }
  //@}
};

char DumpFunctionWrapperPass::ID;
} // namespace

FunctionPass *polly::createDumpFunctionWrapperPass(std::string Suffix) {
  return new DumpFunctionWrapperPass(std::move(Suffix));
}

llvm::PreservedAnalyses DumpFunctionPass::run(Function &F,
                                              FunctionAnalysisManager &AM) {
  runDumpFunction(F, Suffix);
  return PreservedAnalyses::all();
}

INITIALIZE_PASS_BEGIN(DumpFunctionWrapperPass, "polly-dump-function",
                      "Polly - Dump Function", false, false)
INITIALIZE_PASS_END(DumpFunctionWrapperPass, "polly-dump-function",
                    "Polly - Dump Function", false, false)
