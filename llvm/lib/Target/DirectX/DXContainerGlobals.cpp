//===- DXContainerGlobals.cpp - DXContainer global generator pass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DXContainerGlobalsPass implementation.
//
//===----------------------------------------------------------------------===//

#include "DXILShaderFlags.h"
#include "DirectX.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/MD5.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;
using namespace llvm::dxil;

namespace {
class DXContainerGlobals : public llvm::ModulePass {

  GlobalVariable *getShaderFlags(Module &M);
  GlobalVariable *computeShaderHash(Module &M);

public:
  static char ID; // Pass identification, replacement for typeid
  DXContainerGlobals() : ModulePass(ID) {
    initializeDXContainerGlobalsPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "DXContainer Global Emitter";
  }

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<ShaderFlagsAnalysisWrapper>();
  }
};

} // namespace

bool DXContainerGlobals::runOnModule(Module &M) {
  llvm::SmallVector<GlobalValue *> Globals;
  Globals.push_back(getShaderFlags(M));
  Globals.push_back(computeShaderHash(M));

  appendToCompilerUsed(M, Globals);
  return true;
}

GlobalVariable *DXContainerGlobals::getShaderFlags(Module &M) {
  const uint64_t Flags =
      (uint64_t)(getAnalysis<ShaderFlagsAnalysisWrapper>().getShaderFlags());

  Constant *FlagsConstant = ConstantInt::get(M.getContext(), APInt(64, Flags));
  auto *GV = new llvm::GlobalVariable(M, FlagsConstant->getType(), true,
                                      GlobalValue::PrivateLinkage,
                                      FlagsConstant, "dx.sfi0");
  GV->setSection("SFI0");
  GV->setAlignment(Align(4));
  return GV;
}

GlobalVariable *DXContainerGlobals::computeShaderHash(Module &M) {
  auto *DXILConstant =
      cast<ConstantDataArray>(M.getNamedGlobal("dx.dxil")->getInitializer());
  MD5 Digest;
  Digest.update(DXILConstant->getRawDataValues());
  MD5::MD5Result Result = Digest.final();

  dxbc::ShaderHash HashData = {0, {0}};
  // The Hash's IncludesSource flag gets set whenever the hashed shader includes
  // debug information.
  if (M.debug_compile_units_begin() != M.debug_compile_units_end())
    HashData.Flags = static_cast<uint32_t>(dxbc::HashFlags::IncludesSource);

  memcpy(reinterpret_cast<void *>(&HashData.Digest), Result.data(), 16);
  if (sys::IsBigEndianHost)
    HashData.swapBytes();
  StringRef Data(reinterpret_cast<char *>(&HashData), sizeof(dxbc::ShaderHash));

  Constant *ModuleConstant =
      ConstantDataArray::get(M.getContext(), arrayRefFromStringRef(Data));
  auto *GV = new llvm::GlobalVariable(M, ModuleConstant->getType(), true,
                                      GlobalValue::PrivateLinkage,
                                      ModuleConstant, "dx.hash");
  GV->setSection("HASH");
  GV->setAlignment(Align(4));
  return GV;
}

char DXContainerGlobals::ID = 0;
INITIALIZE_PASS_BEGIN(DXContainerGlobals, "dxil-globals",
                      "DXContainer Global Emitter", false, true)
INITIALIZE_PASS_DEPENDENCY(ShaderFlagsAnalysisWrapper)
INITIALIZE_PASS_END(DXContainerGlobals, "dxil-globals",
                    "DXContainer Global Emitter", false, true)

ModulePass *llvm::createDXContainerGlobalsPass() {
  return new DXContainerGlobals();
}
