//===------ DumpModulePass.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Write a module to a file.
//
//===----------------------------------------------------------------------===//

#include "polly/Support/DumpModulePass.h"

#include "polly/Options.h"
#include "llvm/IR/LegacyPassManagers.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include <string.h>
#define DEBUG_TYPE "polly-dump-module"

using namespace llvm;
using namespace polly;

namespace {

class DumpModule : public ModulePass {
private:
  DumpModule(const DumpModule &) = delete;
  const DumpModule &operator=(const DumpModule &) = delete;

  std::string Filename;
  bool IsSuffix;

public:
  static char ID;

  /// This constructor is used e.g. if using opt -polly-dump-module.
  ///
  /// Provide a default suffix to not overwrite the original file.
  explicit DumpModule() : ModulePass(ID), Filename("-dump"), IsSuffix(true) {}

  explicit DumpModule(llvm::StringRef Filename, bool IsSuffix)
      : ModulePass(ID), Filename(Filename), IsSuffix(IsSuffix) {}

  /// @name ModulePass interface
  //@{
  virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  virtual bool runOnModule(llvm::Module &M) override {
    std::string Dumpfile;
    if (IsSuffix) {
      auto ModuleName = M.getName();
      auto Stem = sys::path::stem(ModuleName);
      Dumpfile = (Twine(Stem) + Filename + ".ll").str();
    } else {
      Dumpfile = Filename;
    }
    LLVM_DEBUG(dbgs() << "Dumping module to " << Dumpfile << '\n');

    std::unique_ptr<ToolOutputFile> Out;
    std::error_code EC;
    Out.reset(new ToolOutputFile(Dumpfile, EC, sys::fs::F_None));
    if (EC) {
      errs() << EC.message() << '\n';
      return false;
    }

    M.print(Out->os(), nullptr);
    Out->keep();

    return false;
  }
  //@}
};

char DumpModule::ID;
} // namespace

ModulePass *polly::createDumpModulePass(llvm::StringRef Filename,
                                        bool IsSuffix) {
  return new DumpModule(Filename, IsSuffix);
}

INITIALIZE_PASS_BEGIN(DumpModule, "polly-dump-module", "Polly - Dump Module",
                      false, false)
INITIALIZE_PASS_END(DumpModule, "polly-dump-module", "Polly - Dump Module",
                    false, false)
