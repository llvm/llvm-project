//===- MetaRenamer.cpp - Rename everything with metasyntatic names --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass renames everything with metasyntatic names. The intent is to use
// this pass after bugpoint reduction to conceal the nature of the original
// program.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/MetaRenamer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/TypeFinder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils.h"

using namespace llvm;

static const char *const metaNames[] = {
  // See http://en.wikipedia.org/wiki/Metasyntactic_variable
  "foo", "bar", "baz", "quux", "barney", "snork", "zot", "blam", "hoge",
  "wibble", "wobble", "widget", "wombat", "ham", "eggs", "pluto", "spam"
};

namespace {
// This PRNG is from the ISO C spec. It is intentionally simple and
// unsuitable for cryptographic use. We're just looking for enough
// variety to surprise and delight users.
struct PRNG {
  unsigned long next;

  void srand(unsigned int seed) { next = seed; }

  int rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next / 65536) % 32768;
  }
};

struct Renamer {
  Renamer(unsigned int seed) { prng.srand(seed); }

  const char *newName() {
    return metaNames[prng.rand() % array_lengthof(metaNames)];
  }

  PRNG prng;
};

void MetaRename(Function &F) {
  for (Argument &Arg : F.args())
    if (!Arg.getType()->isVoidTy())
      Arg.setName("arg");

  for (auto &BB : F) {
    BB.setName("bb");

    for (auto &I : BB)
      if (!I.getType()->isVoidTy())
        I.setName("tmp");
  }
}

void MetaRename(Module &M,
                function_ref<TargetLibraryInfo &(Function &)> GetTLI) {
  // Seed our PRNG with simple additive sum of ModuleID. We're looking to
  // simply avoid always having the same function names, and we need to
  // remain deterministic.
  unsigned int randSeed = 0;
  for (auto C : M.getModuleIdentifier())
    randSeed += C;

  Renamer renamer(randSeed);

  // Rename all aliases
  for (auto AI = M.alias_begin(), AE = M.alias_end(); AI != AE; ++AI) {
    StringRef Name = AI->getName();
    if (Name.startswith("llvm.") || (!Name.empty() && Name[0] == 1))
      continue;

    AI->setName("alias");
  }

  // Rename all global variables
  for (GlobalVariable &GV : M.globals()) {
    StringRef Name = GV.getName();
    if (Name.startswith("llvm.") || (!Name.empty() && Name[0] == 1))
      continue;

    GV.setName("global");
  }

  // Rename all struct types
  TypeFinder StructTypes;
  StructTypes.run(M, true);
  for (StructType *STy : StructTypes) {
    if (STy->isLiteral() || STy->getName().empty())
      continue;

    SmallString<128> NameStorage;
    STy->setName(
        (Twine("struct.") + renamer.newName()).toStringRef(NameStorage));
  }

  // Rename all functions
  for (auto &F : M) {
    StringRef Name = F.getName();
    LibFunc Tmp;
    // Leave library functions alone because their presence or absence could
    // affect the behavior of other passes.
    if (Name.startswith("llvm.") || (!Name.empty() && Name[0] == 1) ||
        GetTLI(F).getLibFunc(F, Tmp))
      continue;

    // Leave @main alone. The output of -metarenamer might be passed to
    // lli for execution and the latter needs a main entry point.
    if (Name != "main")
      F.setName(renamer.newName());

    MetaRename(F);
  }
}

struct MetaRenamer : public ModulePass {
  // Pass identification, replacement for typeid
  static char ID;

  MetaRenamer() : ModulePass(ID) {
    initializeMetaRenamerPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.setPreservesAll();
  }

  bool runOnModule(Module &M) override {
    auto GetTLI = [this](Function &F) -> TargetLibraryInfo & {
      return this->getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    };
    MetaRename(M, GetTLI);
    return true;
  }
};

} // end anonymous namespace

char MetaRenamer::ID = 0;

INITIALIZE_PASS_BEGIN(MetaRenamer, "metarenamer",
                      "Assign new names to everything", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(MetaRenamer, "metarenamer",
                    "Assign new names to everything", false, false)

//===----------------------------------------------------------------------===//
//
// MetaRenamer - Rename everything with metasyntactic names.
//
ModulePass *llvm::createMetaRenamerPass() {
  return new MetaRenamer();
}

PreservedAnalyses MetaRenamerPass::run(Module &M, ModuleAnalysisManager &AM) {
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto GetTLI = [&FAM](Function &F) -> TargetLibraryInfo & {
    return FAM.getResult<TargetLibraryAnalysis>(F);
  };
  MetaRename(M, GetTLI);

  return PreservedAnalyses::all();
}
