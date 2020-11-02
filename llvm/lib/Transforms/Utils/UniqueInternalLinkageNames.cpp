//===- UniqueInternalLinkageNames.cpp - Unique Internal Linkage Sym Names -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements unique naming of internal linkage symbols with option
// -funique-internal-linkage-symbols.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/UniqueInternalLinkageNames.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/MD5.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

static bool uniqueifyInternalLinkageNames(Module &M) {
  llvm::MD5 Md5;
  Md5.update(M.getSourceFileName());
  llvm::MD5::MD5Result R;
  Md5.final(R);
  SmallString<32> Str;
  llvm::MD5::stringifyResult(R, Str);
  // Prepend "__uniq" before the hash for tools like profilers to understand that
  // this symbol is of internal linkage type.
  std::string ModuleNameHash = (Twine(".__uniq.") + Twine(Str)).str();
  bool Changed = false;

  // Append the module hash to all internal linkage functions.
  for (auto &F : M) {
    if (F.hasInternalLinkage()) {
      F.setName(F.getName() + ModuleNameHash);
      Changed = true;
    }
  }

  // Append the module hash to all internal linkage globals.
  for (auto &GV : M.globals()) {
    if (GV.hasInternalLinkage()) {
      GV.setName(GV.getName() + ModuleNameHash);
      Changed = true;
    }
  }
  return Changed;
}

namespace {

// Legacy pass that provides a name to every anon globals.
class UniqueInternalLinkageNamesLegacyPass : public ModulePass {

public:
  /// Pass identification, replacement for typeid
  static char ID;

  /// Specify pass name for debug output
  StringRef getPassName() const override {
    return "Unique Internal Linkage Names";
  }

  explicit UniqueInternalLinkageNamesLegacyPass() : ModulePass(ID) {
    initializeUniqueInternalLinkageNamesLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    return uniqueifyInternalLinkageNames(M);
  }
};

char UniqueInternalLinkageNamesLegacyPass::ID = 0;
} // anonymous namespace

PreservedAnalyses
UniqueInternalLinkageNamesPass::run(Module &M, ModuleAnalysisManager &AM) {
  if (!uniqueifyInternalLinkageNames(M))
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

INITIALIZE_PASS_BEGIN(UniqueInternalLinkageNamesLegacyPass,
                      "unique-internal-linkage-names",
                      "Uniqueify internal linkage names", false, false)
INITIALIZE_PASS_END(UniqueInternalLinkageNamesLegacyPass,
                    "unique-internal-linkage-names",
                    "Uniqueify Internal linkage names", false, false)

namespace llvm {
ModulePass *createUniqueInternalLinkageNamesPass() {
  return new UniqueInternalLinkageNamesLegacyPass();
}
} // namespace llvm
