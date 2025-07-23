//===- DXILWriterPass.cpp - Bitcode writing pass --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DXILWriterPass implementation.
//
//===----------------------------------------------------------------------===//

#include "DXILWriterPass.h"
#include "DXILBitcodeWriter.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;
using namespace llvm::dxil;

namespace {
class WriteDXILPass : public llvm::ModulePass {
  raw_ostream &OS; // raw_ostream to print on

public:
  static char ID; // Pass identification, replacement for typeid
  WriteDXILPass() : ModulePass(ID), OS(dbgs()) {
    initializeWriteDXILPassPass(*PassRegistry::getPassRegistry());
  }

  explicit WriteDXILPass(raw_ostream &o) : ModulePass(ID), OS(o) {
    initializeWriteDXILPassPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "Bitcode Writer"; }

  bool runOnModule(Module &M) override {
    WriteDXILToFile(M, OS);
    return false;
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};

static void legalizeLifetimeIntrinsics(Module &M) {
  for (Function &F : M) {
    Intrinsic::ID IID = F.getIntrinsicID();
    if (IID != Intrinsic::lifetime_start && IID != Intrinsic::lifetime_end)
      continue;

    // Lifetime intrinsics in LLVM 3.7 do not have the memory FnAttr
    F.removeFnAttr(Attribute::Memory);

    // Lifetime intrinsics in LLVM 3.7 do not have mangled names
    F.setName(Intrinsic::getBaseName(IID));

    // LLVM 3.7 Lifetime intrinics require an i8* operand, so we insert bitcasts
    // to ensure that is the case
    for (auto *User : make_early_inc_range(F.users())) {
      CallInst *CI = dyn_cast<CallInst>(User);
      assert(CI && "Expected user of a lifetime intrinsic function to be a "
                   "lifetime intrinsic call");
      Value *PtrOperand = CI->getArgOperand(1);
      PointerType *PtrTy = cast<PointerType>(PtrOperand->getType());
      Value *NoOpBitCast = CastInst::Create(Instruction::BitCast, PtrOperand,
                                            PtrTy, "", CI->getIterator());
      CI->setArgOperand(1, NoOpBitCast);
    }
  }
}

static void removeLifetimeIntrinsics(Module &M) {
  for (Function &F : make_early_inc_range(M)) {
    if (Intrinsic::ID IID = F.getIntrinsicID();
        IID != Intrinsic::lifetime_start && IID != Intrinsic::lifetime_end)
      continue;

    for (User *U : make_early_inc_range(F.users())) {
      LifetimeIntrinsic *LI = dyn_cast<LifetimeIntrinsic>(U);
      assert(LI && "Expected user of lifetime intrinsic function to be "
                   "a LifetimeIntrinsic instruction");
      BitCastInst *BCI = dyn_cast<BitCastInst>(LI->getArgOperand(1));
      assert(BCI && "Expected pointer operand of LifetimeIntrinsic to be a "
                    "BitCastInst");
      LI->eraseFromParent();
      BCI->eraseFromParent();
    }
    F.eraseFromParent();
  }
}

class EmbedDXILPass : public llvm::ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  EmbedDXILPass() : ModulePass(ID) {
    initializeEmbedDXILPassPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "DXIL Embedder"; }

  bool runOnModule(Module &M) override {
    std::string Data;
    llvm::raw_string_ostream OS(Data);

    Triple OriginalTriple = M.getTargetTriple();
    // Set to DXIL triple when write to bitcode.
    // Only the output bitcode need to be DXIL triple.
    M.setTargetTriple(Triple("dxil-ms-dx"));

    // Perform late legalization of lifetime intrinsics that would otherwise
    // fail the Module Verifier if performed in an earlier pass
    legalizeLifetimeIntrinsics(M);

    WriteDXILToFile(M, OS);

    // We no longer need lifetime intrinsics after bitcode serialization, so we
    // simply remove them to keep the Module Verifier happy after our
    // not-so-legal legalizations
    removeLifetimeIntrinsics(M);

    // Recover triple.
    M.setTargetTriple(OriginalTriple);

    Constant *ModuleConstant =
        ConstantDataArray::get(M.getContext(), arrayRefFromStringRef(Data));
    auto *GV = new llvm::GlobalVariable(M, ModuleConstant->getType(), true,
                                        GlobalValue::PrivateLinkage,
                                        ModuleConstant, "dx.dxil");
    GV->setSection("DXIL");
    GV->setAlignment(Align(4));
    appendToCompilerUsed(M, {GV});
    return true;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};
} // namespace

char WriteDXILPass::ID = 0;
INITIALIZE_PASS_BEGIN(WriteDXILPass, "dxil-write-bitcode", "Write Bitcode",
                      false, true)
INITIALIZE_PASS_DEPENDENCY(ModuleSummaryIndexWrapperPass)
INITIALIZE_PASS_END(WriteDXILPass, "dxil-write-bitcode", "Write Bitcode", false,
                    true)

ModulePass *llvm::createDXILWriterPass(raw_ostream &Str) {
  return new WriteDXILPass(Str);
}

char EmbedDXILPass::ID = 0;
INITIALIZE_PASS(EmbedDXILPass, "dxil-embed", "Embed DXIL", false, true)

ModulePass *llvm::createDXILEmbedderPass() { return new EmbedDXILPass(); }
