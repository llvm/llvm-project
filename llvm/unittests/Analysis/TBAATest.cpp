//===--- TBAATest.cpp - Mixed TBAA unit tests -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysisEvaluator.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

class OldTBAATest : public testing::Test {
protected:
  OldTBAATest() : M("MixedTBAATest", C), MD(C) {}

  LLVMContext C;
  Module M;
  MDBuilder MD;
};

TEST_F(OldTBAATest, checkVerifierBehavior) {
  // C++ unit test case to avoid going through the auto upgrade logic.

  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C), {});
  auto *F = cast<Function>(M.getOrInsertFunction("f", FTy));
  auto *BB = BasicBlock::Create(C, "entry", F);
  auto *IntType = Type::getInt32Ty(C);
  auto *PtrType = Type::getInt32PtrTy(C);
  auto *SI = new StoreInst(ConstantInt::get(IntType, 42),
                           ConstantPointerNull::get(PtrType), BB);
  ReturnInst::Create(C, nullptr, BB);

  auto *RootMD = MD.createTBAARoot("Simple C/C++ TBAA");
  auto *MD1 = MD.createTBAANode("omnipotent char", RootMD);
  auto *MD2 = MD.createTBAANode("int", MD1);
  SI->setMetadata(LLVMContext::MD_tbaa, MD2);

  SmallVector<char, 0> ErrorMsg;
  raw_svector_ostream Outs(ErrorMsg);

  StringRef ExpectedFailureMsg(
      "Old-style TBAA is no longer allowed, use struct-path TBAA instead");

  EXPECT_TRUE(verifyFunction(*F, &Outs));
  EXPECT_TRUE(StringRef(ErrorMsg.begin(), ErrorMsg.size())
                  .startswith(ExpectedFailureMsg));
}

} // end anonymous namspace
} // end llvm namespace
