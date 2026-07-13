//===- llvm/unittest/IR/VerifierTest.cpp - Verifier unit tests --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Verifier.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "gtest/gtest.h"

using namespace llvm;
namespace {

TEST(VerifierTest, Branch_i1) {
  LLVMContext C;
  Module M("M", C);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg=*/false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "foo", M);
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  BasicBlock *Exit = BasicBlock::Create(C, "exit", F);
  ReturnInst::Create(C, Exit);

  // To avoid triggering an assertion in CondBrInst::Create, we first create
  // a branch with an 'i1' condition ...

  Constant *False = ConstantInt::getFalse(C);
  CondBrInst *BI = CondBrInst::Create(False, Exit, Exit, Entry);

  // ... then use setOperand to redirect it to a value of different type.

  Constant *Zero32 = ConstantInt::get(IntegerType::get(C, 32), 0);
  BI->setOperand(0, Zero32);

  EXPECT_TRUE(verifyFunction(*F));
}

TEST(VerifierTest, Freeze) {
  LLVMContext C;
  Module M("M", C);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg=*/false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "foo", M);
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  ReturnInst *RI = ReturnInst::Create(C, Entry);

  IntegerType *ITy = IntegerType::get(C, 32);
  ConstantInt *CI = ConstantInt::get(ITy, 0);

  // Valid type : freeze(<2 x i32>)
  Constant *CV = ConstantVector::getSplat(ElementCount::getFixed(2), CI);
  FreezeInst *FI_vec = new FreezeInst(CV);
  FI_vec->insertBefore(RI->getIterator());

  EXPECT_FALSE(verifyFunction(*F));

  FI_vec->eraseFromParent();

  // Valid type : freeze(float)
  Constant *CFP = ConstantFP::get(Type::getDoubleTy(C), 0.0);
  FreezeInst *FI_dbl = new FreezeInst(CFP);
  FI_dbl->insertBefore(RI->getIterator());

  EXPECT_FALSE(verifyFunction(*F));

  FI_dbl->eraseFromParent();

  // Valid type : freeze(ptr)
  PointerType *PT = PointerType::get(C, 0);
  ConstantPointerNull *CPN = ConstantPointerNull::get(PT);
  FreezeInst *FI_ptr = new FreezeInst(CPN);
  FI_ptr->insertBefore(RI->getIterator());

  EXPECT_FALSE(verifyFunction(*F));

  FI_ptr->eraseFromParent();

  // Valid type : freeze(int)
  FreezeInst *FI = new FreezeInst(CI);
  FI->insertBefore(RI->getIterator());

  EXPECT_FALSE(verifyFunction(*F));

  FI->eraseFromParent();
}

TEST(VerifierTest, InvalidRetAttribute) {
  LLVMContext C;
  Module M("M", C);
  FunctionType *FTy = FunctionType::get(Type::getInt32Ty(C), /*isVarArg=*/false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "foo", M);
  AttributeList AS = F->getAttributes();
  F->setAttributes(AS.addRetAttribute(
      C, Attribute::getWithUWTableKind(C, UWTableKind::Default)));

  std::string Error;
  raw_string_ostream ErrorOS(Error);
  EXPECT_TRUE(verifyModule(M, &ErrorOS));
  EXPECT_TRUE(StringRef(Error).starts_with(
      "Attribute 'uwtable' does not apply to function return values"));
}

/// Test the verifier rejects invalid nofpclass values that the assembler may
/// also choose to reject.
TEST(VerifierTest, InvalidNoFPClassAttribute) {
  LLVMContext C;

  const unsigned InvalidMasks[] = {0, fcAllFlags + 1};

  for (unsigned InvalidMask : InvalidMasks) {
    Module M("M", C);
    FunctionType *FTy =
        FunctionType::get(Type::getFloatTy(C), /*isVarArg=*/false);
    Function *F = Function::Create(FTy, Function::ExternalLinkage, "foo", M);
    AttributeList AS = F->getAttributes();

    // Don't use getWithNoFPClass to avoid using out of bounds enum values here.
    F->setAttributes(AS.addRetAttribute(
        C, Attribute::get(C, Attribute::NoFPClass, InvalidMask)));

    std::string Error;
    raw_string_ostream ErrorOS(Error);
    EXPECT_TRUE(verifyModule(M, &ErrorOS));

    StringRef ErrMsg(Error);

    if (InvalidMask == 0) {
      EXPECT_TRUE(ErrMsg.starts_with(
          "Attribute 'nofpclass' must have at least one test bit set"))
          << ErrMsg;
    } else {
      EXPECT_TRUE(ErrMsg.starts_with("Invalid value for 'nofpclass' test mask"))
          << ErrMsg;
    }
  }
}

TEST(VerifierTest, CrossModuleRef) {
  LLVMContext C;
  Module M1("M1", C);
  Module M2("M2", C);
  Module M3("M3", C);
  FunctionType *FTy = FunctionType::get(Type::getInt32Ty(C), /*isVarArg=*/false);
  Function *F1 = Function::Create(FTy, Function::ExternalLinkage, "foo1", M1);
  Function *F2 = Function::Create(FTy, Function::ExternalLinkage, "foo2", M2);
  Function *F3 = Function::Create(FTy, Function::ExternalLinkage, "foo3", M3);

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
  EXPECT_TRUE(Error == "Global is referenced in a different module!\n"
                       "ptr @foo2\n"
                       "; ModuleID = 'M2'\n"
                       "  %call = call i32 @foo2()\n"
                       "ptr @foo1\n"
                       "; ModuleID = 'M1'\n"
                       "Global is used by function in a different module\n"
                       "ptr @foo2\n"
                       "; ModuleID = 'M2'\n"
                       "ptr @foo3\n"
                       "; ModuleID = 'M3'\n");

  Error.clear();
  EXPECT_TRUE(verifyModule(M1, &ErrorOS));
  EXPECT_TRUE(StringRef(Error) == "Referencing function in another module!\n"
                                  "  %call = call i32 @foo2()\n"
                                  "; ModuleID = 'M1'\n"
                                  "ptr @foo2\n"
                                  "; ModuleID = 'M2'\n");

  Error.clear();
  EXPECT_TRUE(verifyModule(M3, &ErrorOS));
  EXPECT_TRUE(StringRef(Error).starts_with(
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
  EXPECT_TRUE(StringRef(Error).starts_with("Global is external, but doesn't "
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
  EXPECT_TRUE(StringRef(Error).starts_with("Global is external, but doesn't "
                                           "have external or weak linkage!"));
}

TEST(VerifierTest, DetectInvalidDebugInfo) {
  {
    LLVMContext C;
    Module M("M", C);
    DIBuilder DIB(M);
    DIB.createCompileUnit(DISourceLanguageName(dwarf::DW_LANG_C89),
                          DIB.createFile("broken.c", "/"), "unittest", false,
                          "", 0);
    DIB.finalize();
    EXPECT_FALSE(verifyModule(M));

    // Now break it by inserting non-CU node to the list of CUs.
    auto *File = DIB.createFile("not-a-CU.f", ".");
    NamedMDNode *NMD = M.getOrInsertNamedMetadata("llvm.dbg.cu");
    NMD->addOperand(File);
    EXPECT_TRUE(verifyModule(M));
  }
  {
    // A DICompileUnit whose dialect is outside the defined enumeration is
    // rejected by the verifier. The textual IR parser, the bitcode reader,
    // and the C API all guard against this, so this path is only reachable
    // via programmatic IR construction.
    LLVMContext C;
    Module M("M", C);
    DIBuilder DIB(M);
    const uint16_t OutOfRangeDialect = dwarf::DW_LLVM_LANG_DIALECT_max + 1;
    DIB.createCompileUnit(
        DISourceLanguageName(dwarf::DW_LANG_C89, OutOfRangeDialect),
        DIB.createFile("broken.c", "/"), "unittest", false, "", 0);
    DIB.finalize();

    std::string Error;
    raw_string_ostream ErrorOS(Error);
    EXPECT_TRUE(verifyModule(M, &ErrorOS));
    EXPECT_TRUE(StringRef(Error).contains("invalid language dialect")) << Error;
  }
  {
    LLVMContext C;
    Module M("M", C);
    DIBuilder DIB(M);
    auto *CU = DIB.createCompileUnit(DISourceLanguageName(dwarf::DW_LANG_C89),
                                     DIB.createFile("broken.c", "/"),
                                     "unittest", false, "", 0);
    new GlobalVariable(M, Type::getInt8Ty(C), false,
                       GlobalValue::ExternalLinkage, nullptr, "g");

    auto *F = Function::Create(FunctionType::get(Type::getVoidTy(C), false),
                               Function::ExternalLinkage, "f", M);
    IRBuilder<> Builder(BasicBlock::Create(C, "", F));
    Builder.CreateUnreachable();
    auto *SPType = DIB.createSubroutineType(DIB.getOrCreateTypeArray({}));
    F->setSubprogram(DIB.createFunction(
        CU, "f", "f", DIB.createFile("broken.c", "/"), 1, SPType, 1,
        DINode::FlagZero,
        DISubprogram::SPFlagLocalToUnit | DISubprogram::SPFlagDefinition));
    DIB.finalize();
    EXPECT_FALSE(verifyModule(M));

    // Now break it by not listing the CU at all.
    M.eraseNamedMetadata(M.getOrInsertNamedMetadata("llvm.dbg.cu"));
    EXPECT_TRUE(verifyModule(M));
  }
}

TEST(VerifierTest, MDNodeWrongContext) {
  LLVMContext C1, C2;
  auto *Node = MDNode::get(C1, {});

  Module M("M", C2);
  auto *NamedNode = M.getOrInsertNamedMetadata("test");
  NamedNode->addOperand(Node);

  std::string Error;
  raw_string_ostream ErrorOS(Error);
  EXPECT_TRUE(verifyModule(M, &ErrorOS));
  EXPECT_TRUE(StringRef(Error).starts_with(
      "MDNode context does not match Module context!"));
}

TEST(VerifierTest, AttributesWrongContext) {
  LLVMContext C1, C2;
  Module M1("M", C1);
  FunctionType *FTy1 =
      FunctionType::get(Type::getVoidTy(C1), /*isVarArg=*/false);
  Function *F1 = Function::Create(FTy1, Function::ExternalLinkage, "foo", M1);
  F1->setDoesNotReturn();

  Module M2("M", C2);
  FunctionType *FTy2 =
      FunctionType::get(Type::getVoidTy(C2), /*isVarArg=*/false);
  Function *F2 = Function::Create(FTy2, Function::ExternalLinkage, "foo", M2);
  F2->copyAttributesFrom(F1);

  EXPECT_TRUE(verifyFunction(*F2));
}

TEST(VerifierTest, SwitchInst) {
  LLVMContext C;
  Module M("M", C);
  IntegerType *Int32Ty = Type::getInt32Ty(C);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C), {Int32Ty, Int32Ty},
                                        /*isVarArg=*/false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "foo", M);
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  BasicBlock *Default = BasicBlock::Create(C, "default", F);
  BasicBlock *OnOne = BasicBlock::Create(C, "on_one", F);
  BasicBlock *OnTwo = BasicBlock::Create(C, "on_two", F);

  BasicBlock *Exit = BasicBlock::Create(C, "exit", F);

  UncondBrInst::Create(Exit, Default);
  UncondBrInst::Create(Exit, OnTwo);
  UncondBrInst::Create(Exit, OnOne);
  ReturnInst::Create(C, Exit);

  Value *Cond = F->getArg(0);
  SwitchInst *Switch = SwitchInst::Create(Cond, Default, 2, Entry);
  Switch->addCase(ConstantInt::get(Int32Ty, 1), OnOne);
  Switch->addCase(ConstantInt::get(Int32Ty, 2), OnTwo);

  EXPECT_FALSE(verifyFunction(*F));
}

TEST(VerifierTest, CrossFunctionRef) {
  LLVMContext C;
  Module M("M", C);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg=*/false);
  Function *F1 = Function::Create(FTy, Function::ExternalLinkage, "foo1", M);
  Function *F2 = Function::Create(FTy, Function::ExternalLinkage, "foo2", M);
  BasicBlock *Entry1 = BasicBlock::Create(C, "entry", F1);
  BasicBlock *Entry2 = BasicBlock::Create(C, "entry", F2);
  Type *I32 = Type::getInt32Ty(C);

  Value *Alloca = new AllocaInst(I32, 0, "alloca", Entry1);
  ReturnInst::Create(C, Entry1);

  Instruction *Store = new StoreInst(ConstantInt::get(I32, 0), Alloca, Entry2);
  ReturnInst::Create(C, Entry2);

  std::string Error;
  raw_string_ostream ErrorOS(Error);
  EXPECT_TRUE(verifyModule(M, &ErrorOS));
  EXPECT_TRUE(StringRef(Error).starts_with(
      "Referring to an instruction in another function!"));

  // Explicitly erase the store to avoid a use-after-free when the module is
  // destroyed.
  Store->eraseFromParent();
}

TEST(VerifierTest, AtomicRMW) {
  LLVMContext C;
  Module M("M", C);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg=*/false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "foo", M);
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  Value *Ptr = PoisonValue::get(PointerType::get(C, 0));

  Type *FPTy = Type::getFloatTy(C);
  Constant *CF = ConstantFP::getZero(FPTy);

  // Invalid scalable type : atomicrmw (<vscale x 2 x float>)
  Constant *CV = ConstantVector::getSplat(ElementCount::getScalable(2), CF);
  new AtomicRMWInst(AtomicRMWInst::FAdd, Ptr, CV, Align(8),
                    AtomicOrdering::SequentiallyConsistent, SyncScope::System,
                    /*Elementwise=*/false, Entry);
  ReturnInst::Create(C, Entry);

  std::string Error;
  raw_string_ostream ErrorOS(Error);
  EXPECT_TRUE(verifyFunction(*F, &ErrorOS));
  EXPECT_TRUE(StringRef(Error).starts_with(
      "atomicrmw fadd operand must have floating-point or "
      "fixed vector of floating-point type!"))
      << Error;
}

TEST(VerifierTest, AtomicRMWElementwiseScalar) {
  LLVMContext C;
  Module M("M", C);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg=*/false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "foo", M);
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  Value *Ptr = PoisonValue::get(PointerType::get(C, 0));

  Type *I32Ty = Type::getInt32Ty(C);
  Constant *CI = ConstantInt::get(I32Ty, 0);

  new AtomicRMWInst(AtomicRMWInst::Add, Ptr, CI, Align(4),
                    AtomicOrdering::Monotonic, SyncScope::System,
                    /*Elementwise=*/true, Entry);
  ReturnInst::Create(C, Entry);

  std::string Error;
  raw_string_ostream ErrorOS(Error);
  EXPECT_TRUE(verifyFunction(*F, &ErrorOS));
  EXPECT_TRUE(StringRef(Error).starts_with(
      "atomicrmw elementwise operand must have fixed vector type!"))
      << Error;
}

TEST(VerifierTest, AtomicRMWElementwiseIntOpOnFPVector) {
  LLVMContext C;
  Module M("M", C);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg=*/false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "foo", M);
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  Value *Ptr = PoisonValue::get(PointerType::get(C, 0));

  Type *FPTy = Type::getFloatTy(C);
  Constant *CV = ConstantVector::getSplat(ElementCount::getFixed(4),
                                          ConstantFP::getZero(FPTy));

  new AtomicRMWInst(AtomicRMWInst::Add, Ptr, CV, Align(16),
                    AtomicOrdering::Monotonic, SyncScope::System,
                    /*Elementwise=*/true, Entry);
  ReturnInst::Create(C, Entry);

  std::string Error;
  raw_string_ostream ErrorOS(Error);
  EXPECT_TRUE(verifyFunction(*F, &ErrorOS));
  EXPECT_TRUE(
      StringRef(Error).starts_with("atomicrmw add operand must have integer"))
      << Error;
}

TEST(VerifierTest, AtomicRMWElementwiseOddSizedVector) {
  LLVMContext C;
  Module M("M", C);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg=*/false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "foo", M);
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  Value *Ptr = PoisonValue::get(PointerType::get(C, 0));

  Type *I32Ty = Type::getInt32Ty(C);
  Constant *CV = ConstantVector::getSplat(ElementCount::getFixed(5),
                                          ConstantInt::get(I32Ty, 0));

  new AtomicRMWInst(AtomicRMWInst::Add, Ptr, CV, Align(4),
                    AtomicOrdering::Monotonic, SyncScope::System,
                    /*Elementwise=*/true, Entry);
  ReturnInst::Create(C, Entry);

  std::string Error;
  raw_string_ostream ErrorOS(Error);
  EXPECT_TRUE(verifyFunction(*F, &ErrorOS));
  EXPECT_TRUE(StringRef(Error).starts_with(
      "atomic memory access' operand must have a power-of-two size"))
      << Error;
}

TEST(VerifierTest, AtomicRMWElementwiseFPOpOnIntVector) {
  LLVMContext C;
  Module M("M", C);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg=*/false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "foo", M);
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  Value *Ptr = PoisonValue::get(PointerType::get(C, 0));

  Type *I32Ty = Type::getInt32Ty(C);
  Constant *CV = ConstantVector::getSplat(ElementCount::getFixed(4),
                                          ConstantInt::get(I32Ty, 0));

  new AtomicRMWInst(AtomicRMWInst::FAdd, Ptr, CV, Align(16),
                    AtomicOrdering::Monotonic, SyncScope::System,
                    /*Elementwise=*/true, Entry);
  ReturnInst::Create(C, Entry);

  std::string Error;
  raw_string_ostream ErrorOS(Error);
  EXPECT_TRUE(verifyFunction(*F, &ErrorOS));
  EXPECT_TRUE(StringRef(Error).starts_with(
      "atomicrmw fadd operand must have floating-point or "
      "fixed vector of floating-point type!"))
      << Error;
}

TEST(VerifierTest, AtomicRMWIntVector) {
  LLVMContext C;
  Module M("M", C);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg=*/false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "foo", M);
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  Value *Ptr = PoisonValue::get(PointerType::get(C, 0));

  Type *IntTy = Type::getInt16Ty(C);
  Constant *CI = ConstantInt::get(IntTy, 0);

  // Invalid scalable type : atomicrmw (<vscale x 2 x i16>)
  Constant *CV = ConstantVector::getSplat(ElementCount::getScalable(2), CI);
  new AtomicRMWInst(AtomicRMWInst::Add, Ptr, CV, Align(8),
                    AtomicOrdering::SequentiallyConsistent, SyncScope::System,
                    /*Elementwise=*/false, Entry);
  ReturnInst::Create(C, Entry);

  std::string Error;
  raw_string_ostream ErrorOS(Error);
  EXPECT_TRUE(verifyFunction(*F, &ErrorOS));
  EXPECT_TRUE(
      StringRef(Error).starts_with("atomicrmw add operand must have integer or "
                                   "fixed vector of integer type!"))
      << Error;
}

TEST(VerifierTest, GetElementPtrInst) {
  LLVMContext C;
  Module M("M", C);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg=*/false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "foo", M);
  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  ReturnInst *RI = ReturnInst::Create(C, Entry);

  FixedVectorType *V2P1Ty = FixedVectorType::get(PointerType::get(C, 1), 2);
  FixedVectorType *V2P2Ty = FixedVectorType::get(PointerType::get(C, 2), 2);

  Instruction *GEPVec = GetElementPtrInst::Create(
      Type::getInt8Ty(C), ConstantAggregateZero::get(V2P1Ty),
      {ConstantVector::getSplat(ElementCount::getFixed(2),
                                ConstantInt::get(Type::getInt64Ty(C), 0))},
      Entry);

  GEPVec->insertBefore(RI->getIterator());

  // Break the address space of the source value
  GEPVec->getOperandUse(0).set(ConstantAggregateZero::get(V2P2Ty));

  std::string Error;
  raw_string_ostream ErrorOS(Error);
  EXPECT_TRUE(verifyFunction(*F, &ErrorOS));
  EXPECT_TRUE(
      StringRef(Error).starts_with("GEP address space doesn't match type"))
      << Error;
}

TEST(VerifierTest, DeeplyNested) {
  LLVMContext Ctx;
  Module M("M", Ctx);

  // Construct an extremely deeply nested metadata node that should cause
  // a stack overflow on most platforms if recursion through the entire
  // chain is performed.
  Metadata *CurrentMetadataNode =
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), 0));
  for (int i = 0; i < 100000; ++i) {
    CurrentMetadataNode = MDTuple::get(Ctx, {CurrentMetadataNode});
  }

  NamedMDNode *NamedMetadataNode = M.getOrInsertNamedMetadata("foo");
  NamedMetadataNode->addOperand(cast<MDNode>(CurrentMetadataNode));

  std::string Error;
  raw_string_ostream ErrorOS(Error);
  EXPECT_FALSE(verifyModule(M, &ErrorOS));
}

TEST(VerifierTest, IntrinsicRetInvalidStruct) {
  LLVMContext Ctx;

  // Create 2 invalid struct types for @llvm.nvvm.elect.sync intrinsic.
  Type *I32Ty = Type::getInt32Ty(Ctx);
  Type *I1Ty = Type::getInt1Ty(Ctx);

  StructType *NonLiteral = StructType::create(Ctx, {I32Ty, I1Ty}, "st");
  StructType *LiteralPacked =
      StructType::get(Ctx, {I32Ty, I1Ty}, /*isPacked=*/true);
  for (StructType *STy : {NonLiteral, LiteralPacked}) {
    Module M("M", Ctx);
    FunctionType *IntrFTy = FunctionType::get(STy, I32Ty, /*isVarArg=*/false);
    Function *Intr = Function::Create(IntrFTy, Function::ExternalLinkage,
                                      "llvm.nvvm.elect.sync", M);

    FunctionType *FTy =
        FunctionType::get(Type::getVoidTy(Ctx), /*isVarArg=*/false);
    Function *F = Function::Create(FTy, Function::ExternalLinkage, "foo", M);
    BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", F);

    Constant *Zero = ConstantInt::get(I32Ty, 0);
    CallInst::Create(Intr, Zero, /*Bundles=*/{}, "ci", Entry);
    ReturnInst::Create(Ctx, Entry);

    std::string Error;
    raw_string_ostream ErrorOS(Error);
    EXPECT_TRUE(verifyModule(M, &ErrorOS));

    EXPECT_TRUE(StringRef(Error).starts_with(
        "intrinsic return type expected literal non-packed struct with 2 "
        "elements, but got"))
        << Error;
  }
}

} // end anonymous namespace
