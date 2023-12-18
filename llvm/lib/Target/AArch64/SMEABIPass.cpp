//===--------- SMEABI - SME  ABI-------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements parts of the the SME ABI, such as:
// * Using the lazy-save mechanism before enabling the use of ZA.
// * Setting up the lazy-save mechanism around invokes.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "Utils/AArch64BaseInfo.h"
#include "Utils/AArch64SMEAttributes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-sme-abi"

namespace {
struct SMEABI : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  SMEABI() : FunctionPass(ID) {
    initializeSMEABIPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

private:
  bool updateNewZAFunctions(Module *M, Function *F, IRBuilder<> &Builder);
};
} // end anonymous namespace

char SMEABI::ID = 0;
static const char *name = "SME ABI Pass";
INITIALIZE_PASS_BEGIN(SMEABI, DEBUG_TYPE, name, false, false)
INITIALIZE_PASS_END(SMEABI, DEBUG_TYPE, name, false, false)

FunctionPass *llvm::createSMEABIPass() { return new SMEABI(); }

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

// Utility function to emit a call to __arm_tpidr2_save and clear TPIDR2_EL0.
void emitTPIDR2Save(Module *M, IRBuilder<> &Builder) {
  auto *TPIDR2SaveTy =
      FunctionType::get(Builder.getVoidTy(), {}, /*IsVarArgs=*/false);
  auto Attrs =
      AttributeList()
          .addFnAttribute(M->getContext(), "aarch64_pstate_sm_compatible")
          .addFnAttribute(M->getContext(), "aarch64_pstate_za_preserved");
  FunctionCallee Callee =
      M->getOrInsertFunction("__arm_tpidr2_save", TPIDR2SaveTy, Attrs);
  CallInst *Call = Builder.CreateCall(Callee);
  Call->setCallingConv(
      CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X0);

  // A save to TPIDR2 should be followed by clearing TPIDR2_EL0.
  Function *WriteIntr =
      Intrinsic::getDeclaration(M, Intrinsic::aarch64_sme_set_tpidr2);
  Builder.CreateCall(WriteIntr->getFunctionType(), WriteIntr,
                     Builder.getInt64(0));
}

/// This function generates code to commit a lazy save at the beginning of a
/// function marked with `aarch64_pstate_za_new`. If the value read from
/// TPIDR2_EL0 is not null on entry to the function then the lazy-saving scheme
/// is active and we should call __arm_tpidr2_save to commit the lazy save.
/// Additionally, PSTATE.ZA should be enabled at the beginning of the function
/// and disabled before returning.
bool SMEABI::updateNewZAFunctions(Module *M, Function *F,
                                  IRBuilder<> &Builder) {
  LLVMContext &Context = F->getContext();
  BasicBlock *OrigBB = &F->getEntryBlock();

  // Create the new blocks for reading TPIDR2_EL0 & enabling ZA state.
  auto *SaveBB = OrigBB->splitBasicBlock(OrigBB->begin(), "save.za", true);
  auto *PreludeBB = BasicBlock::Create(Context, "prelude", F, SaveBB);

  // Read TPIDR2_EL0 in PreludeBB & branch to SaveBB if not 0.
  Builder.SetInsertPoint(PreludeBB);
  Function *TPIDR2Intr =
      Intrinsic::getDeclaration(M, Intrinsic::aarch64_sme_get_tpidr2);
  auto *TPIDR2 = Builder.CreateCall(TPIDR2Intr->getFunctionType(), TPIDR2Intr,
                                    {}, "tpidr2");
  auto *Cmp =
      Builder.CreateCmp(ICmpInst::ICMP_NE, TPIDR2, Builder.getInt64(0), "cmp");
  Builder.CreateCondBr(Cmp, SaveBB, OrigBB);

  // Create a call __arm_tpidr2_save, which commits the lazy save.
  Builder.SetInsertPoint(&SaveBB->back());
  emitTPIDR2Save(M, Builder);

  // Enable pstate.za at the start of the function.
  Builder.SetInsertPoint(&OrigBB->front());
  Function *EnableZAIntr =
      Intrinsic::getDeclaration(M, Intrinsic::aarch64_sme_za_enable);
  Builder.CreateCall(EnableZAIntr->getFunctionType(), EnableZAIntr);

  // ZA state must be zeroed upon entry to a function with NewZA
  Function *ZeroIntr =
      Intrinsic::getDeclaration(M, Intrinsic::aarch64_sme_zero);
  Builder.CreateCall(ZeroIntr->getFunctionType(), ZeroIntr,
                     Builder.getInt32(0xff));

  // Before returning, disable pstate.za
  for (BasicBlock &BB : *F) {
    Instruction *T = BB.getTerminator();
    if (!T || !isa<ReturnInst>(T))
      continue;
    Builder.SetInsertPoint(T);
    Function *DisableZAIntr =
        Intrinsic::getDeclaration(M, Intrinsic::aarch64_sme_za_disable);
    Builder.CreateCall(DisableZAIntr->getFunctionType(), DisableZAIntr);
  }

  F->addFnAttr("aarch64_expanded_pstate_za");
  return true;
}

bool SMEABI::runOnFunction(Function &F) {
  Module *M = F.getParent();
  LLVMContext &Context = F.getContext();
  IRBuilder<> Builder(Context);

  if (F.isDeclaration() || F.hasFnAttribute("aarch64_expanded_pstate_za"))
    return false;

  bool Changed = false;
  SMEAttrs FnAttrs(F);
  if (FnAttrs.hasNewZABody())
    Changed |= updateNewZAFunctions(M, &F, Builder);

  return Changed;
}
