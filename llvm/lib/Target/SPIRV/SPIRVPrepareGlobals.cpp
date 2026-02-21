//===-- SPIRVPrepareGlobals.cpp - Prepare IR SPIRV globals ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The pass:
//   - transforms IR globals that cannot be trivially mapped to SPIRV into
//     something that is trival to lower;
//   - for AMDGCN flavoured SPIRV, it assigns unique IDs to the specialisation
//     constants associated with feature predicates, which were inserted by the
//     FE when expanding calls to __builtin_amdgcn_processor_is or
//     __builtin_amdgcn_is_invocable
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "SPIRVUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"

#include <climits>
#include <string>

#define DEBUG_TYPE "spirv-prepare-globals"

using namespace llvm;

namespace {

struct SPIRVPrepareGlobals : public ModulePass {
  static char ID;
  SPIRVPrepareGlobals() : ModulePass(ID) {}

  StringRef getPassName() const override {
    return "SPIRV prepare global variables";
  }

  bool runOnModule(Module &M) override;
};

// The backend does not support GlobalAlias. Replace aliases with their aliasees
// when possible and remove them from the module.
bool tryReplaceAliasWithAliasee(GlobalAlias &GA) {
  // According to the lang ref, aliases cannot be replaced if either the alias
  // or the aliasee are interposable. We only replace in the case that both
  // are not interposable.
  if (GA.isInterposable()) {
    LLVM_DEBUG(dbgs() << "Skipping interposable alias: " << GA.getName()
                      << "\n");
    return false;
  }

  auto *AO = dyn_cast<GlobalObject>(GA.getAliasee());
  if (!AO) {
    LLVM_DEBUG(dbgs() << "Skipping alias whose aliasee is not a GlobalObject: "
                      << GA.getName() << "\n");
    return false;
  }

  if (AO->isInterposable()) {
    LLVM_DEBUG(dbgs() << "Skipping interposable aliasee: " << AO->getName()
                      << "\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "Replacing alias " << GA.getName()
                    << " with aliasee: " << AO->getName() << "\n");

  GA.replaceAllUsesWith(AO);
  if (GA.isDiscardableIfUnused()) {
    GA.eraseFromParent();
  }

  return true;
}

bool tryAssignPredicateSpecConstIDs(Module &M, Function *F) {
  StringMap<unsigned> IDs;

  // Replace placeholder Specialisation Constant IDs with unique IDs associated
  // with the predicate being evaluated, which is encoded in the call name.
  for (auto &&U : F->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;
    auto *Arg0 = dyn_cast<ConstantInt>(CI->getArgOperand(0));
    if (!Arg0)
      continue;

    unsigned ID = Arg0->getZExtValue();

    if (ID != UINT32_MAX)
      continue;

    assert(CI->getMetadata("llvm.amdgcn.feature.predicate") &&
           "Feature predicates must be encoded into metadata!");
    auto *P = cast<MDString>(
        CI->getMetadata("llvm.amdgcn.feature.predicate")->getOperand(0));
    ID = IDs.try_emplace(P->getString(), IDs.size()).first->second;

    CI->setArgOperand(0, ConstantInt::get(CI->getArgOperand(0)->getType(), ID));
  }

  if (IDs.empty())
    return false;

  // Store the predicate -> ID mapping as a fixed format string
  // (predicate ID\0...), for later use during SPIR-V consumption.
  std::string Tmp;
  for (auto &&[Predicate, SpecID] : IDs)
    Tmp.append(Predicate).append(" ").append(utostr(SpecID)).push_back('\0');

  Constant *PredSpecIDStr =
      ConstantDataArray::getString(M.getContext(), Tmp, false);

  new GlobalVariable(M, PredSpecIDStr->getType(), true,
                     GlobalVariable::LinkageTypes::ExternalLinkage,
                     PredSpecIDStr, "llvm.amdgcn.feature.predicate.ids");

  return true;
}

bool SPIRVPrepareGlobals::runOnModule(Module &M) {
  bool Changed = false;

  for (GlobalAlias &GA : make_early_inc_range(M.aliases())) {
    Changed |= tryReplaceAliasWithAliasee(GA);
  }

  if (M.getTargetTriple().getVendor() != Triple::AMD)
    return Changed;

  // TODO: Currently, for AMDGCN flavoured SPIR-V, the symbol can only be
  //       inserted via feature predicate use, but in the future this will need
  //       revisiting if we start making more liberal use of the intrinsic.
  if (Function *F = M.getFunction("_Z20__spirv_SpecConstantib"))
    Changed |= tryAssignPredicateSpecConstIDs(M, F);

  return Changed;
}
char SPIRVPrepareGlobals::ID = 0;

} // namespace

INITIALIZE_PASS(SPIRVPrepareGlobals, "prepare-globals",
                "SPIRV prepare global variables", false, false)

namespace llvm {
ModulePass *createSPIRVPrepareGlobalsPass() {
  return new SPIRVPrepareGlobals();
}
} // namespace llvm
