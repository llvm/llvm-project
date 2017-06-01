//===- llvm/unittest/IR/VerifierTest.cpp - Verifier unit tests --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

TEST(VerifierTest, Branch_i1) {
  LLVMContext C;
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
  LLVMContext C;
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
  LLVMContext C;
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

TEST(VerifierTest, InvalidVariableLinkage) {
  LLVMContext C;
  Module M("M", C);
  new GlobalVariable(M, Type::getInt8Ty(C), false,
                     GlobalValue::LinkOnceODRLinkage, nullptr, "Some Global");
  std::string Error;
  raw_string_ostream ErrorOS(Error);
  EXPECT_TRUE(verifyModule(M, &ErrorOS));
  EXPECT_TRUE(
      StringRef(ErrorOS.str()).startswith("Global is external, but doesn't "
                                          "have external or weak linkage!"));
}

TEST(VerifierTest, InvalidFunctionLinkage) {
  LLVMContext C;
  Module M("M", C);

  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg=*/false);
  Function::Create(FTy, GlobalValue::LinkOnceODRLinkage, "foo", &M);
  std::string Error;
  raw_string_ostream ErrorOS(Error);
  EXPECT_TRUE(verifyModule(M, &ErrorOS));
  EXPECT_TRUE(
      StringRef(ErrorOS.str()).startswith("Global is external, but doesn't "
                                          "have external or weak linkage!"));
}

TEST(VerifierTest, StripInvalidDebugInfo) {
  {
    LLVMContext C;
    Module M("M", C);
    DIBuilder DIB(M);
    DIB.createCompileUnit(dwarf::DW_LANG_C89, DIB.createFile("broken.c", "/"),
                          "unittest", false, "", 0);
    DIB.finalize();
    EXPECT_FALSE(verifyModule(M));

    // Now break it by inserting non-CU node to the list of CUs.
    auto *File = DIB.createFile("not-a-CU.f", ".");
    NamedMDNode *NMD = M.getOrInsertNamedMetadata("llvm.dbg.cu");
    NMD->addOperand(File);
    EXPECT_TRUE(verifyModule(M));

    ModulePassManager MPM(true);
    MPM.addPass(VerifierPass(false));
    ModuleAnalysisManager MAM(true);
    MAM.registerPass([&] { return VerifierAnalysis(); });
    MPM.run(M, MAM);
    EXPECT_FALSE(verifyModule(M));
  }
  {
    LLVMContext C;
    Module M("M", C);
    DIBuilder DIB(M);
    auto *CU = DIB.createCompileUnit(dwarf::DW_LANG_C89,
                                     DIB.createFile("broken.c", "/"),
                                     "unittest", false, "", 0);
    new GlobalVariable(M, Type::getInt8Ty(C), false,
                       GlobalValue::ExternalLinkage, nullptr, "g");

    auto *F = cast<Function>(M.getOrInsertFunction(
        "f", FunctionType::get(Type::getVoidTy(C), false)));
    IRBuilder<> Builder(BasicBlock::Create(C, "", F));
    Builder.CreateUnreachable();
    F->setSubprogram(DIB.createFunction(CU, "f", "f",
                                        DIB.createFile("broken.c", "/"), 1,
                                        nullptr, true, true, 1));
    DIB.finalize();
    EXPECT_FALSE(verifyModule(M));

    // Now break it by not listing the CU at all.
    M.eraseNamedMetadata(M.getOrInsertNamedMetadata("llvm.dbg.cu"));
    EXPECT_TRUE(verifyModule(M));

    ModulePassManager MPM(true);
    MPM.addPass(VerifierPass(false));
    ModuleAnalysisManager MAM(true);
    MAM.registerPass([&] { return VerifierAnalysis(); });
    MPM.run(M, MAM);
    EXPECT_FALSE(verifyModule(M));
  }
}

TEST(VerifierTest, StripInvalidDebugInfoLegacy) {
  LLVMContext C;
  Module M("M", C);
  DIBuilder DIB(M);
  DIB.createCompileUnit(dwarf::DW_LANG_C89, DIB.createFile("broken.c", "/"),
                        "unittest", false, "", 0);
  DIB.finalize();
  EXPECT_FALSE(verifyModule(M));

  // Now break it.
  auto *File = DIB.createFile("not-a-CU.f", ".");
  NamedMDNode *NMD = M.getOrInsertNamedMetadata("llvm.dbg.cu");
  NMD->addOperand(File);
  EXPECT_TRUE(verifyModule(M));

  legacy::PassManager Passes;
  Passes.add(createVerifierPass(false));
  Passes.run(M);
  EXPECT_FALSE(verifyModule(M));
}

} // end anonymous namespace
} // end namespace llvm
