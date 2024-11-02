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
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;
using namespace llvm::dxil;

namespace {
class DXContainerGlobals : public llvm::ModulePass {

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
  const uint64_t Flags =
      (uint64_t)(getAnalysis<ShaderFlagsAnalysisWrapper>().getShaderFlags());

  Constant *FlagsConstant = ConstantInt::get(M.getContext(), APInt(64, Flags));
  auto *GV = new llvm::GlobalVariable(M, FlagsConstant->getType(), true,
                                      GlobalValue::PrivateLinkage,
                                      FlagsConstant, "dx.sfi0");
  GV->setSection("SFI0");
  GV->setAlignment(Align(4));
  appendToCompilerUsed(M, {GV});
  return true;
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
