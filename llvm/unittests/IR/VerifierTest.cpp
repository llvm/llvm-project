//===- llvm/unittest/IR/VerifierTest.cpp - Verifier unit tests --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Verifier.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
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

TEST(VerifierTest, CrossModuleRef) {
  LLVMContext &C = getGlobalContext();
  Module M1("M1", C);
  Module M2("M2", C);
  Module M3("M3", C);
  FunctionType *FTy = FunctionType::get(Type::getInt32Ty(C), /*isVarArg=*/false);
  Function *F1 = cast<Function>(M1.getOrInsertFunction("foo1", FTy));
  Function *F2 = cast<Function>(M2.getOrInsertFunction("foo2", FTy));
  Function *F3 = cast<Function>(M3.getOrInsertFunction("foo3", FTy));

  BasicBlock *Entry1 = BasicBlock::Create(C, "entry", F1);
  BasicBlock *Entry3 = BasicBlock::Create(C, "entry", F3);

  // BAD: Referencing function in another module
  CallInst::Create(F2,"call",Entry1);

  // BAD: Referencing personality routine in another module
  F3->setPersonalityFn(F2);

  // Fill in the body
  Constant *ConstZero = ConstantInt::get(Type::getInt32Ty(C), 0);
  ReturnInst::Create(C, ConstZero, Entry1);
  ReturnInst::Create(C, ConstZero, Entry3);

  std::string Error;
  raw_string_ostream ErrorOS(Error);
  EXPECT_TRUE(verifyModule(M2, &ErrorOS));
  EXPECT_TRUE(StringRef(ErrorOS.str())
                  .equals("Global is used by function in a different module\n"
                          "i32 ()* @foo2\n"
                          "; ModuleID = 'M2'\n"
                          "i32 ()* @foo3\n"
                          "; ModuleID = 'M3'\n"
                          "Global is referenced in a different module!\n"
                          "i32 ()* @foo2\n"
                          "; ModuleID = 'M2'\n"
                          "  %call = call i32 @foo2()\n"
                          "i32 ()* @foo1\n"
                          "; ModuleID = 'M1'\n"));

  Error.clear();
  EXPECT_TRUE(verifyModule(M1, &ErrorOS));
  EXPECT_TRUE(StringRef(ErrorOS.str()).equals(
      "Referencing function in another module!\n"
      "  %call = call i32 @foo2()\n"
      "; ModuleID = 'M1'\n"
      "i32 ()* @foo2\n"
      "; ModuleID = 'M2'\n"));

  Error.clear();
  EXPECT_TRUE(verifyModule(M3, &ErrorOS));
  EXPECT_TRUE(StringRef(ErrorOS.str()).startswith(
      "Referencing personality function in another module!"));

  // Erase bad methods to avoid triggering an assertion failure on destruction
  F1->eraseFromParent();
  F3->eraseFromParent();
}

TEST(VerifierTest, CrossModuleMetadataRef) {
  LLVMContext &C = getGlobalContext();
  Module M1("M1", C);
  Module M2("M2", C);
  GlobalVariable *newGV =
      new GlobalVariable(M1, Type::getInt8Ty(C), false,
                         GlobalVariable::ExternalLinkage, nullptr,
                         "Some Global");

  DIBuilder dbuilder(M2);
  auto CU = dbuilder.createCompileUnit(dwarf::DW_LANG_Julia, "test.jl", ".",
                                       "unittest", false, "", 0);
  auto File = dbuilder.createFile("test.jl", ".");
  auto Ty = dbuilder.createBasicType("Int8", 8, 8, dwarf::DW_ATE_signed);
  dbuilder.createGlobalVariable(CU, "_SOME_GLOBAL", "_SOME_GLOBAL", File, 1, Ty,
                                false, newGV);
  dbuilder.finalize();

  std::string Error;
  raw_string_ostream ErrorOS(Error);
  EXPECT_TRUE(verifyModule(M2, &ErrorOS));
  EXPECT_TRUE(StringRef(ErrorOS.str())
                  .startswith("Referencing global in another module!"));
}
} // end anonymous namespace
} // end namespace llvm
