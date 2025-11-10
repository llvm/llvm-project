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

bool SPIRVPrepareGlobals::runOnModule(Module &M) {
  const bool IsAMD = M.getTargetTriple().getVendor() == Triple::AMD;
  if (!IsAMD)
    return false;

  bool Changed = false;
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
