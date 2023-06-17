//===-- NVPTXCtorDtorLowering.cpp - Handle global ctors and dtors --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This pass creates a unified init and fini kernel with the required metadata
//===----------------------------------------------------------------------===//

#include "NVPTXCtorDtorLowering.h"
#include "NVPTX.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "nvptx-lower-ctor-dtor"

static cl::opt<std::string>
    GlobalStr("nvptx-lower-global-ctor-dtor-id",
              cl::desc("Override unique ID of ctor/dtor globals."),
              cl::init(""), cl::Hidden);

namespace {

static std::string getHash(StringRef Str) {
  llvm::MD5 Hasher;
  llvm::MD5::MD5Result Hash;
  Hasher.update(Str);
  Hasher.final(Hash);
  return llvm::utohexstr(Hash.low(), /*LowerCase=*/true);
}

static bool createInitOrFiniGlobls(Module &M, StringRef GlobalName,
                                   bool IsCtor) {
  GlobalVariable *GV = M.getGlobalVariable(GlobalName);
  if (!GV || !GV->hasInitializer())
    return false;
  ConstantArray *GA = dyn_cast<ConstantArray>(GV->getInitializer());
  if (!GA || GA->getNumOperands() == 0)
    return false;

  // NVPTX has no way to emit variables at specific sections or support for
  // the traditional constructor sections. Instead, we emit mangled global
  // names so the runtime can build the list manually.
  for (Value *V : GA->operands()) {
    auto *CS = cast<ConstantStruct>(V);
    auto *F = cast<Constant>(CS->getOperand(1));
    uint64_t Priority = cast<ConstantInt>(CS->getOperand(0))->getSExtValue();
    std::string PriorityStr = "." + std::to_string(Priority);
    // We append a semi-unique hash and the priority to the global name.
    std::string GlobalID =
        !GlobalStr.empty() ? GlobalStr : getHash(M.getSourceFileName());
    std::string NameStr =
        ((IsCtor ? "__init_array_object_" : "__fini_array_object_") +
         F->getName() + "_" + GlobalID + "_" + std::to_string(Priority))
            .str();
    // PTX does not support exported names with '.' in them.
    llvm::transform(NameStr, NameStr.begin(),
                    [](char c) { return c == '.' ? '_' : c; });

    auto *GV = new GlobalVariable(M, F->getType(), /*IsConstant=*/true,
                                  GlobalValue::ExternalLinkage, F, NameStr,
                                  nullptr, GlobalValue::NotThreadLocal,
                                  /*AddressSpace=*/4);
    // This isn't respected by Nvidia, simply put here for clarity.
    GV->setSection(IsCtor ? ".init_array" + PriorityStr
                          : ".fini_array" + PriorityStr);
    GV->setVisibility(GlobalVariable::ProtectedVisibility);
    appendToUsed(M, {GV});
  }

  GV->eraseFromParent();
  return true;
}

static bool lowerCtorsAndDtors(Module &M) {
  bool Modified = false;
  Modified |= createInitOrFiniGlobls(M, "llvm.global_ctors", /*IsCtor =*/true);
  Modified |= createInitOrFiniGlobls(M, "llvm.global_dtors", /*IsCtor =*/false);
  return Modified;
}

class NVPTXCtorDtorLoweringLegacy final : public ModulePass {
public:
  static char ID;
  NVPTXCtorDtorLoweringLegacy() : ModulePass(ID) {}
  bool runOnModule(Module &M) override { return lowerCtorsAndDtors(M); }
};

} // End anonymous namespace

PreservedAnalyses NVPTXCtorDtorLoweringPass::run(Module &M,
                                                 ModuleAnalysisManager &AM) {
  return lowerCtorsAndDtors(M) ? PreservedAnalyses::none()
                               : PreservedAnalyses::all();
}

char NVPTXCtorDtorLoweringLegacy::ID = 0;
char &llvm::NVPTXCtorDtorLoweringLegacyPassID = NVPTXCtorDtorLoweringLegacy::ID;
INITIALIZE_PASS(NVPTXCtorDtorLoweringLegacy, DEBUG_TYPE,
                "Lower ctors and dtors for NVPTX", false, false)

ModulePass *llvm::createNVPTXCtorDtorLoweringLegacyPass() {
  return new NVPTXCtorDtorLoweringLegacy();
}
