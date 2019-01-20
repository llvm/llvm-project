//===--- TBAATest.cpp - Mixed TBAA unit tests -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasAnalysisEvaluator.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

class TBAATest : public testing::Test {
protected:
  TBAATest() : M("TBAATest", C), MD(C) {}

  LLVMContext C;
  Module M;
  MDBuilder MD;
};

static StoreInst *getFunctionWithSingleStore(Module *M, StringRef Name) {
  auto &C = M->getContext();
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C), {});
  auto *F = cast<Function>(M->getOrInsertFunction(Name, FTy));
  auto *BB = BasicBlock::Create(C, "entry", F);
  auto *IntType = Type::getInt32Ty(C);
  auto *PtrType = Type::getInt32PtrTy(C);
  auto *SI = new StoreInst(ConstantInt::get(IntType, 42),
                           ConstantPointerNull::get(PtrType), BB);
  ReturnInst::Create(C, nullptr, BB);

  return SI;
}

TEST_F(TBAATest, checkVerifierBehaviorForOldTBAA) {
  auto *SI = getFunctionWithSingleStore(&M, "f1");
  auto *F = SI->getFunction();

  // C++ unit test case to avoid going through the auto upgrade logic.
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

TEST_F(TBAATest, checkTBAAMerging) {
  auto *SI = getFunctionWithSingleStore(&M, "f2");
  auto *F = SI->getFunction();

  auto *RootMD = MD.createTBAARoot("tbaa-root");
  auto *MD1 = MD.createTBAANode("scalar-a", RootMD);
  auto *StructTag1 = MD.createTBAAStructTagNode(MD1, MD1, 0);
  auto *MD2 = MD.createTBAANode("scalar-b", RootMD);
  auto *StructTag2 = MD.createTBAAStructTagNode(MD2, MD2, 0);

  auto *GenericMD = MDNode::getMostGenericTBAA(StructTag1, StructTag2);

  EXPECT_EQ(GenericMD, nullptr);

  // Despite GenericMD being nullptr, we expect the setMetadata call to be well
  // defined and produce a well-formed function.
  SI->setMetadata(LLVMContext::MD_tbaa, GenericMD);

  EXPECT_TRUE(!verifyFunction(*F));
}

} // end anonymous namspace
} // end llvm namespace
