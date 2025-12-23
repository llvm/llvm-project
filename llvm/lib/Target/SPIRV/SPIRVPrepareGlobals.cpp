//===-- SPIRVPrepareGlobals.cpp - Prepare IR SPIRV globals ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The pass transforms IR globals that cannot be trivially mapped to SPIRV
// into something that is trival to lower.
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "SPIRVUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"

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

bool tryExtendLLVMBitcodeMarker(GlobalVariable &Bitcode) {
  assert(Bitcode.getName() == "llvm.embedded.module");

  ArrayType *AT = cast<ArrayType>(Bitcode.getValueType());
  if (AT->getNumElements() != 0)
    return false;

  ArrayType *AT1 = ArrayType::get(AT->getElementType(), 1);
  Constant *OneEltInit = Constant::getNullValue(AT1);
  Bitcode.replaceInitializer(OneEltInit);
  return true;
}

// In HIP, dynamic LDS variables are represented using 0-element global arrays
// in the __shared__ language address-space.
//
//  extern __shared__ int LDS[];
//
// These are not representable in SPIRV directly.
// To represent them, for AMD, we use an array with UINT32_MAX-elements.
// These are reverse translated to 0-element arrays.
bool tryExtendDynamicLDSGlobal(GlobalVariable &GV) {
  constexpr unsigned WorkgroupAS =
      storageClassToAddressSpace(SPIRV::StorageClass::Workgroup);
  const bool IsWorkgroupExternal =
      GV.hasExternalLinkage() && GV.getAddressSpace() == WorkgroupAS;
  if (!IsWorkgroupExternal)
    return false;

  const ArrayType *AT = dyn_cast<ArrayType>(GV.getValueType());
  if (!AT || AT->getNumElements() != 0)
    return false;

  constexpr auto UInt32Max = std::numeric_limits<uint32_t>::max();
  ArrayType *NewAT = ArrayType::get(AT->getElementType(), UInt32Max);
  GlobalVariable *NewGV = new GlobalVariable(
      *GV.getParent(), NewAT, GV.isConstant(), GV.getLinkage(), nullptr, "",
      &GV, GV.getThreadLocalMode(), WorkgroupAS, GV.isExternallyInitialized());
  NewGV->takeName(&GV);
  GV.replaceAllUsesWith(NewGV);
  GV.eraseFromParent();

  return true;
}

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

bool SPIRVPrepareGlobals::runOnModule(Module &M) {
  bool Changed = false;

  for (GlobalAlias &GA : make_early_inc_range(M.aliases())) {
    Changed |= tryReplaceAliasWithAliasee(GA);
  }

  const bool IsAMD = M.getTargetTriple().getVendor() == Triple::AMD;
  if (!IsAMD)
    return Changed;

  if (GlobalVariable *Bitcode = M.getNamedGlobal("llvm.embedded.module"))
    Changed |= tryExtendLLVMBitcodeMarker(*Bitcode);

  for (GlobalVariable &GV : make_early_inc_range(M.globals()))
    Changed |= tryExtendDynamicLDSGlobal(GV);

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
