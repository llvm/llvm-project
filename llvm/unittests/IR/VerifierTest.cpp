//===- llvm/unittest/IR/VerifierTest.cpp - Verifier unit tests ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Verifier.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

TEST(VerifierTest, Branch_i1) {
  LLVMContext &C = getGlobalContext();
  Module M("M", C);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg=*/false);
  Function *F = cast<Function>(M.getOrInsertFunction("foo", FTy));
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  BasicBlock *Exit = BasicBlock::Create(C, "exit", F);
  ReturnInst::Create(C, Exit);

  // To avoid triggering an assertion in BranchInst::Create, we first create
  // a branch with an 'i1' condition ...

  Constant *False = ConstantInt::getFalse(C);
  BranchInst *BI = BranchInst::Create(Exit, Exit, False, Entry);

  // ... then use setOperand to redirect it to a value of different type.

  Constant *Zero32 = ConstantInt::get(IntegerType::get(C, 32), 0);
  BI->setOperand(0, Zero32);

  EXPECT_TRUE(verifyFunction(*F));
}

TEST(VerifierTest, InvalidRetAttribute) {
  LLVMContext &C = getGlobalContext();
  Module M("M", C);
  FunctionType *FTy = FunctionType::get(Type::getInt32Ty(C), /*isVarArg=*/false);
  Function *F = cast<Function>(M.getOrInsertFunction("foo", FTy));
  AttributeSet AS = F->getAttributes();
  F->setAttributes(AS.addAttribute(C, AttributeSet::ReturnIndex,
                                   Attribute::UWTable));

  std::string Error;
  raw_string_ostream ErrorOS(Error);
  EXPECT_TRUE(verifyModule(M, &ErrorOS));
  EXPECT_TRUE(StringRef(ErrorOS.str()).startswith(
      "Attribute 'uwtable' only applies to functions!"));
}

}
}
