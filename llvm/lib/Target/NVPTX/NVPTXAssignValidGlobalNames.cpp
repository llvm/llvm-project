//===-- NVPTXAssignValidGlobalNames.cpp - Assign valid names to globals ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Clean up the names of global variables in the module to not contain symbols
// that are invalid in PTX.
//
// Currently NVPTX, like other backends, relies on generic symbol name
// sanitizing done by MC. However, the ptxas assembler is more stringent and
// disallows some additional characters in symbol names. This pass makes sure
// such names do not reach MC at all.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "NVPTXUtilities.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"

using namespace llvm;

/// Give \p GV a name that is a valid PTX identifier, returning whether it had
/// to be renamed. Invalid names are rare, so checking first lets the pass
/// report accurately that it left the module alone.
///
/// Note: this does not create collisions - if setName is asked to set the name
/// to something that already exists, it adds a proper postfix to avoid
/// collisions.
static bool assignValidName(GlobalValue &GV) {
  std::string ValidName = NVPTX::getValidPTXIdentifier(GV.getName());
  if (ValidName == GV.getName())
    return false;
  GV.setName(ValidName);
  return true;
}

static bool assignValidGlobalNames(Module &M) {
  bool Changed = false;
  for (GlobalVariable &GV : M.globals()) {
    // We are only allowed to rename symbols that are not externally linked by
    // name
    // - local symbols, as all references will be renamed
    // - .extern .shared symbols, as they're the same regardless of name
    if (GV.hasLocalLinkage() ||
        (GV.hasExternalLinkage() &&
         GV.getAddressSpace() == NVPTX::AddressSpace::Shared))
      Changed |= assignValidName(GV);
  }

  // Do the same for local functions.
  for (Function &F : M.functions())
    if (F.hasLocalLinkage())
      Changed |= assignValidName(F);

  return Changed;
}

namespace {
/// NVPTXAssignValidGlobalNamesLegacyPass
class NVPTXAssignValidGlobalNamesLegacyPass : public ModulePass {
public:
  static char ID;
  NVPTXAssignValidGlobalNamesLegacyPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override { return assignValidGlobalNames(M); }
};
} // namespace

char NVPTXAssignValidGlobalNamesLegacyPass::ID = 0;

INITIALIZE_PASS(NVPTXAssignValidGlobalNamesLegacyPass,
                "nvptx-assign-valid-global-names",
                "Assign valid PTX names to globals", false, false)

ModulePass *llvm::createNVPTXAssignValidGlobalNamesLegacyPass() {
  return new NVPTXAssignValidGlobalNamesLegacyPass();
}

PreservedAnalyses
NVPTXAssignValidGlobalNamesPass::run(Module &M, ModuleAnalysisManager &MAM) {
  if (!assignValidGlobalNames(M))
    return PreservedAnalyses::all();
  return PreservedAnalyses::none().preserveSet<CFGAnalyses>();
}
