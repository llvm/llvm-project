//===- NameAnonFunctions.cpp - ThinLTO Summary-based Function Import ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements naming anonymous function to make sure they can be
// refered to by ThinLTO.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/NameAnonFunctions.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MD5.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

namespace {
// Compute a "unique" hash for the module based on the name of the public
// functions.
class ModuleHasher {
  Module &TheModule;
  std::string TheHash;

public:
  ModuleHasher(Module &M) : TheModule(M) {}

  /// Return the lazily computed hash.
  std::string &get() {
    if (!TheHash.empty())
      // Cache hit :)
      return TheHash;

    MD5 Hasher;
    for (auto &F : TheModule) {
      if (F.isDeclaration() || F.hasLocalLinkage() || !F.hasName())
        continue;
      auto Name = F.getName();
      Hasher.update(Name);
    }
    for (auto &GV : TheModule.globals()) {
      if (GV.isDeclaration() || GV.hasLocalLinkage() || !GV.hasName())
        continue;
      auto Name = GV.getName();
      Hasher.update(Name);
    }

    // Now return the result.
    MD5::MD5Result Hash;
    Hasher.final(Hash);
    SmallString<32> Result;
    MD5::stringifyResult(Hash, Result);
    TheHash = Result.str();
    return TheHash;
  }
};
} // end anonymous namespace

// Rename all the anon functions in the module
bool llvm::nameUnamedFunctions(Module &M) {
  bool Changed = false;
  ModuleHasher ModuleHash(M);
  int count = 0;
  for (auto &F : M) {
    if (F.hasName())
      continue;
    F.setName(Twine("anon.") + ModuleHash.get() + "." + Twine(count++));
    Changed = true;
  }
  return Changed;
}

namespace {

// Legacy pass that provides a name to every anon function.
class NameAnonFunctionLegacyPass : public ModulePass {

public:
  /// Pass identification, replacement for typeid
  static char ID;

  /// Specify pass name for debug output
  const char *getPassName() const override { return "Name Anon Functions"; }

  explicit NameAnonFunctionLegacyPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override { return nameUnamedFunctions(M); }
};
char NameAnonFunctionLegacyPass::ID = 0;

} // anonymous namespace

PreservedAnalyses NameAnonFunctionPass::run(Module &M,
                                            ModuleAnalysisManager &AM) {
  if (!nameUnamedFunctions(M))
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}

INITIALIZE_PASS_BEGIN(NameAnonFunctionLegacyPass, "name-anon-functions",
                      "Provide a name to nameless functions", false, false)
INITIALIZE_PASS_END(NameAnonFunctionLegacyPass, "name-anon-functions",
                    "Provide a name to nameless functions", false, false)

namespace llvm {
ModulePass *createNameAnonFunctionPass() {
  return new NameAnonFunctionLegacyPass();
}
}
