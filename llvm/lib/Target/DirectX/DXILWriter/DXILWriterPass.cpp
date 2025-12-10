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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
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
  LLVMContext &Ctx = M.getContext();
  Type *I64Ty = IntegerType::get(Ctx, 64);
  Type *PtrTy = PointerType::get(Ctx, 0);
  Intrinsic::ID LifetimeIIDs[2] = {Intrinsic::lifetime_start,
                                   Intrinsic::lifetime_end};
  for (Intrinsic::ID &IID : LifetimeIIDs) {
    Function *F = M.getFunction(Intrinsic::getName(IID, {PtrTy}, &M));
    if (!F)
      continue;

    // Get or insert an LLVM 3.7-compliant lifetime intrinsic function of the
    // form `void @llvm.lifetime.[start/end](i64, ptr)` with the NoUnwind
    // attribute
    AttributeList Attr;
    Attr = Attr.addFnAttribute(Ctx, Attribute::NoUnwind);
    FunctionCallee LifetimeCallee = M.getOrInsertFunction(
        Intrinsic::getBaseName(IID), Attr, Type::getVoidTy(Ctx), I64Ty, PtrTy);

    // Replace all calls to lifetime intrinsics with calls to the
    // LLVM 3.7-compliant version of the lifetime intrinsic
    for (User *U : make_early_inc_range(F->users())) {
      CallInst *CI = dyn_cast<CallInst>(U);
      assert(CI &&
             "Expected user of a lifetime intrinsic function to be a CallInst");

      // LLVM 3.7 lifetime intrinics require an i8* operand, so we insert
      // a bitcast to ensure that is the case
      Value *PtrOperand = CI->getArgOperand(0);
      PointerType *PtrOpPtrTy = cast<PointerType>(PtrOperand->getType());
      Value *NoOpBitCast = CastInst::Create(Instruction::BitCast, PtrOperand,
                                            PtrOpPtrTy, "", CI->getIterator());

      // LLVM 3.7 lifetime intrinsics have an explicit size operand, whose value
      // we can obtain from the pointer operand which must be an AllocaInst (as
      // of https://github.com/llvm/llvm-project/pull/149310)
      AllocaInst *AI = dyn_cast<AllocaInst>(PtrOperand);
      assert(AI &&
             "The pointer operand of a lifetime intrinsic call must be an "
             "AllocaInst");
      std::optional<TypeSize> AllocSize =
          AI->getAllocationSize(CI->getDataLayout());
      assert(AllocSize.has_value() &&
             "Expected the allocation size of AllocaInst to be known");
      CallInst *NewCI = CallInst::Create(
          LifetimeCallee,
          {ConstantInt::get(I64Ty, AllocSize.value().getFixedValue()),
           NoOpBitCast},
          "", CI->getIterator());
      for (Attribute ParamAttr : CI->getParamAttributes(0))
        NewCI->addParamAttr(1, ParamAttr);

      CI->eraseFromParent();
    }

    F->eraseFromParent();
  }
}

static void removeLifetimeIntrinsics(Module &M) {
  Intrinsic::ID LifetimeIIDs[2] = {Intrinsic::lifetime_start,
                                   Intrinsic::lifetime_end};
  for (Intrinsic::ID &IID : LifetimeIIDs) {
    Function *F = M.getFunction(Intrinsic::getBaseName(IID));
    if (!F)
      continue;

    for (User *U : make_early_inc_range(F->users())) {
      CallInst *CI = dyn_cast<CallInst>(U);
      assert(CI && "Expected user of lifetime function to be a CallInst");
      BitCastInst *BCI = dyn_cast<BitCastInst>(CI->getArgOperand(1));
      assert(BCI && "Expected pointer operand of CallInst to be a BitCastInst");
      CI->eraseFromParent();
      BCI->eraseFromParent();
    }
    F->eraseFromParent();
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

    // Perform late legalization of lifetime intrinsics that would otherwise
    // fail the Module Verifier if performed in an earlier pass
    legalizeLifetimeIntrinsics(M);

    WriteDXILToFile(M, OS);

    // We no longer need lifetime intrinsics after bitcode serialization, so we
    // simply remove them to keep the Module Verifier happy after our
    // not-so-legal legalizations
    removeLifetimeIntrinsics(M);

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
