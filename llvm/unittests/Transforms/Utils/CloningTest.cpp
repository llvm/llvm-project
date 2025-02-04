//===- Cloning.cpp - Unit tests for the Cloner ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class CloneInstruction : public ::testing::Test {
protected:
  void SetUp() override { V = nullptr; }

  template <typename T>
  T *clone(T *V1) {
    Value *V2 = V1->clone();
    Orig.insert(V1);
    Clones.insert(V2);
    return cast<T>(V2);
  }

  void eraseClones() {
    for (Value *V : Clones)
      V->deleteValue();
    Clones.clear();
  }

  void TearDown() override {
    eraseClones();
    for (Value *V : Orig)
      V->deleteValue();
    Orig.clear();
    if (V)
      V->deleteValue();
  }

  SmallPtrSet<Value *, 4> Orig;   // Erase on exit
  SmallPtrSet<Value *, 4> Clones; // Erase in eraseClones

  LLVMContext context;
  Value *V;
};

TEST_F(CloneInstruction, OverflowBits) {
  V = new Argument(Type::getInt32Ty(context));

  BinaryOperator *Add = BinaryOperator::Create(Instruction::Add, V, V);
  BinaryOperator *Sub = BinaryOperator::Create(Instruction::Sub, V, V);
  BinaryOperator *Mul = BinaryOperator::Create(Instruction::Mul, V, V);

  BinaryOperator *AddClone = this->clone(Add);
  BinaryOperator *SubClone = this->clone(Sub);
  BinaryOperator *MulClone = this->clone(Mul);

  EXPECT_FALSE(AddClone->hasNoUnsignedWrap());
  EXPECT_FALSE(AddClone->hasNoSignedWrap());
  EXPECT_FALSE(SubClone->hasNoUnsignedWrap());
  EXPECT_FALSE(SubClone->hasNoSignedWrap());
  EXPECT_FALSE(MulClone->hasNoUnsignedWrap());
  EXPECT_FALSE(MulClone->hasNoSignedWrap());

  eraseClones();

  Add->setHasNoUnsignedWrap();
  Sub->setHasNoUnsignedWrap();
  Mul->setHasNoUnsignedWrap();

  AddClone = this->clone(Add);
  SubClone = this->clone(Sub);
  MulClone = this->clone(Mul);

  EXPECT_TRUE(AddClone->hasNoUnsignedWrap());
  EXPECT_FALSE(AddClone->hasNoSignedWrap());
  EXPECT_TRUE(SubClone->hasNoUnsignedWrap());
  EXPECT_FALSE(SubClone->hasNoSignedWrap());
  EXPECT_TRUE(MulClone->hasNoUnsignedWrap());
  EXPECT_FALSE(MulClone->hasNoSignedWrap());

  eraseClones();

  Add->setHasNoSignedWrap();
  Sub->setHasNoSignedWrap();
  Mul->setHasNoSignedWrap();

  AddClone = this->clone(Add);
  SubClone = this->clone(Sub);
  MulClone = this->clone(Mul);

  EXPECT_TRUE(AddClone->hasNoUnsignedWrap());
  EXPECT_TRUE(AddClone->hasNoSignedWrap());
  EXPECT_TRUE(SubClone->hasNoUnsignedWrap());
  EXPECT_TRUE(SubClone->hasNoSignedWrap());
  EXPECT_TRUE(MulClone->hasNoUnsignedWrap());
  EXPECT_TRUE(MulClone->hasNoSignedWrap());

  eraseClones();

  Add->setHasNoUnsignedWrap(false);
  Sub->setHasNoUnsignedWrap(false);
  Mul->setHasNoUnsignedWrap(false);

  AddClone = this->clone(Add);
  SubClone = this->clone(Sub);
  MulClone = this->clone(Mul);

  EXPECT_FALSE(AddClone->hasNoUnsignedWrap());
  EXPECT_TRUE(AddClone->hasNoSignedWrap());
  EXPECT_FALSE(SubClone->hasNoUnsignedWrap());
  EXPECT_TRUE(SubClone->hasNoSignedWrap());
  EXPECT_FALSE(MulClone->hasNoUnsignedWrap());
  EXPECT_TRUE(MulClone->hasNoSignedWrap());
}

TEST_F(CloneInstruction, Inbounds) {
  V = new Argument(PointerType::get(context, 0));

  Constant *Z = Constant::getNullValue(Type::getInt32Ty(context));
  std::vector<Value *> ops;
  ops.push_back(Z);
  GetElementPtrInst *GEP =
      GetElementPtrInst::Create(Type::getInt32Ty(context), V, ops);
  EXPECT_FALSE(this->clone(GEP)->isInBounds());

  GEP->setIsInBounds();
  EXPECT_TRUE(this->clone(GEP)->isInBounds());
}

TEST_F(CloneInstruction, Exact) {
  V = new Argument(Type::getInt32Ty(context));

  BinaryOperator *SDiv = BinaryOperator::Create(Instruction::SDiv, V, V);
  EXPECT_FALSE(this->clone(SDiv)->isExact());

  SDiv->setIsExact(true);
  EXPECT_TRUE(this->clone(SDiv)->isExact());
}

TEST_F(CloneInstruction, Attributes) {
  Type *ArgTy1[] = {PointerType::get(context, 0)};
  FunctionType *FT1 =
      FunctionType::get(Type::getVoidTy(context), ArgTy1, false);

  Function *F1 = Function::Create(FT1, Function::ExternalLinkage);
  BasicBlock *BB = BasicBlock::Create(context, "", F1);
  IRBuilder<> Builder(BB);
  Builder.CreateRetVoid();

  Function *F2 = Function::Create(FT1, Function::ExternalLinkage);

  Argument *A = &*F1->arg_begin();
  A->addAttr(
      Attribute::getWithCaptureInfo(A->getContext(), CaptureInfo::none()));

  SmallVector<ReturnInst*, 4> Returns;
  ValueToValueMapTy VMap;
  VMap[A] = UndefValue::get(A->getType());

  CloneFunctionInto(F2, F1, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns);
  EXPECT_FALSE(F2->arg_begin()->hasNoCaptureAttr());

  delete F1;
  delete F2;
}

TEST_F(CloneInstruction, CallingConvention) {
  Type *ArgTy1[] = {PointerType::get(context, 0)};
  FunctionType *FT1 =
      FunctionType::get(Type::getVoidTy(context), ArgTy1, false);

  Function *F1 = Function::Create(FT1, Function::ExternalLinkage);
  F1->setCallingConv(CallingConv::Cold);
  BasicBlock *BB = BasicBlock::Create(context, "", F1);
  IRBuilder<> Builder(BB);
  Builder.CreateRetVoid();

  Function *F2 = Function::Create(FT1, Function::ExternalLinkage);

  SmallVector<ReturnInst*, 4> Returns;
  ValueToValueMapTy VMap;
  VMap[&*F1->arg_begin()] = &*F2->arg_begin();

  CloneFunctionInto(F2, F1, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns);
  EXPECT_EQ(CallingConv::Cold, F2->getCallingConv());

  delete F1;
  delete F2;
}

TEST_F(CloneInstruction, DuplicateInstructionsToSplit) {
  Type *ArgTy1[] = {PointerType::get(context, 0)};
  FunctionType *FT = FunctionType::get(Type::getVoidTy(context), ArgTy1, false);
  V = new Argument(Type::getInt32Ty(context));

  Function *F = Function::Create(FT, Function::ExternalLinkage);

  BasicBlock *BB1 = BasicBlock::Create(context, "", F);
  IRBuilder<> Builder1(BB1);

  BasicBlock *BB2 = BasicBlock::Create(context, "", F);
  IRBuilder<> Builder2(BB2);

  Builder1.CreateBr(BB2);

  Instruction *AddInst = cast<Instruction>(Builder2.CreateAdd(V, V));
  Instruction *MulInst = cast<Instruction>(Builder2.CreateMul(AddInst, V));
  Instruction *SubInst = cast<Instruction>(Builder2.CreateSub(MulInst, V));
  Builder2.CreateRetVoid();

  // Dummy DTU.
  ValueToValueMapTy Mapping;
  DomTreeUpdater DTU(DomTreeUpdater::UpdateStrategy::Lazy);
  auto Split =
      DuplicateInstructionsInSplitBetween(BB2, BB1, SubInst, Mapping, DTU);

  EXPECT_TRUE(Split);
  EXPECT_EQ(Mapping.size(), 2u);
  EXPECT_TRUE(Mapping.find(AddInst) != Mapping.end());
  EXPECT_TRUE(Mapping.find(MulInst) != Mapping.end());

  auto AddSplit = dyn_cast<Instruction>(Mapping[AddInst]);
  EXPECT_TRUE(AddSplit);
  EXPECT_EQ(AddSplit->getOperand(0), V);
  EXPECT_EQ(AddSplit->getOperand(1), V);
  EXPECT_EQ(AddSplit->getParent(), Split);

  auto MulSplit = dyn_cast<Instruction>(Mapping[MulInst]);
  EXPECT_TRUE(MulSplit);
  EXPECT_EQ(MulSplit->getOperand(0), AddSplit);
  EXPECT_EQ(MulSplit->getOperand(1), V);
  EXPECT_EQ(MulSplit->getParent(), Split);

  EXPECT_EQ(AddSplit->getNextNode(), MulSplit);
  EXPECT_EQ(MulSplit->getNextNode(), Split->getTerminator());

  delete F;
}

TEST_F(CloneInstruction, DuplicateInstructionsToSplitBlocksEq1) {
  Type *ArgTy1[] = {PointerType::get(context, 0)};
  FunctionType *FT = FunctionType::get(Type::getVoidTy(context), ArgTy1, false);
  V = new Argument(Type::getInt32Ty(context));

  Function *F = Function::Create(FT, Function::ExternalLinkage);

  BasicBlock *BB1 = BasicBlock::Create(context, "", F);
  IRBuilder<> Builder1(BB1);

  BasicBlock *BB2 = BasicBlock::Create(context, "", F);
  IRBuilder<> Builder2(BB2);

  Builder1.CreateBr(BB2);

  Instruction *AddInst = cast<Instruction>(Builder2.CreateAdd(V, V));
  Instruction *MulInst = cast<Instruction>(Builder2.CreateMul(AddInst, V));
  Instruction *SubInst = cast<Instruction>(Builder2.CreateSub(MulInst, V));
  Builder2.CreateBr(BB2);

  // Dummy DTU.
  DomTreeUpdater DTU(DomTreeUpdater::UpdateStrategy::Lazy);
  ValueToValueMapTy Mapping;
  auto Split = DuplicateInstructionsInSplitBetween(
      BB2, BB2, BB2->getTerminator(), Mapping, DTU);

  EXPECT_TRUE(Split);
  EXPECT_EQ(Mapping.size(), 3u);
  EXPECT_TRUE(Mapping.find(AddInst) != Mapping.end());
  EXPECT_TRUE(Mapping.find(MulInst) != Mapping.end());
  EXPECT_TRUE(Mapping.find(SubInst) != Mapping.end());

  auto AddSplit = dyn_cast<Instruction>(Mapping[AddInst]);
  EXPECT_TRUE(AddSplit);
  EXPECT_EQ(AddSplit->getOperand(0), V);
  EXPECT_EQ(AddSplit->getOperand(1), V);
  EXPECT_EQ(AddSplit->getParent(), Split);

  auto MulSplit = dyn_cast<Instruction>(Mapping[MulInst]);
  EXPECT_TRUE(MulSplit);
  EXPECT_EQ(MulSplit->getOperand(0), AddSplit);
  EXPECT_EQ(MulSplit->getOperand(1), V);
  EXPECT_EQ(MulSplit->getParent(), Split);

  auto SubSplit = dyn_cast<Instruction>(Mapping[SubInst]);
  EXPECT_EQ(MulSplit->getNextNode(), SubSplit);
  EXPECT_EQ(SubSplit->getNextNode(), Split->getTerminator());
  EXPECT_EQ(Split->getSingleSuccessor(), BB2);
  EXPECT_EQ(BB2->getSingleSuccessor(), Split);

  delete F;
}

TEST_F(CloneInstruction, DuplicateInstructionsToSplitBlocksEq2) {
  Type *ArgTy1[] = {PointerType::get(context, 0)};
  FunctionType *FT = FunctionType::get(Type::getVoidTy(context), ArgTy1, false);
  V = new Argument(Type::getInt32Ty(context));

  Function *F = Function::Create(FT, Function::ExternalLinkage);

  BasicBlock *BB1 = BasicBlock::Create(context, "", F);
  IRBuilder<> Builder1(BB1);

  BasicBlock *BB2 = BasicBlock::Create(context, "", F);
  IRBuilder<> Builder2(BB2);

  Builder1.CreateBr(BB2);

  Instruction *AddInst = cast<Instruction>(Builder2.CreateAdd(V, V));
  Instruction *MulInst = cast<Instruction>(Builder2.CreateMul(AddInst, V));
  Instruction *SubInst = cast<Instruction>(Builder2.CreateSub(MulInst, V));
  Builder2.CreateBr(BB2);

  // Dummy DTU.
  DomTreeUpdater DTU(DomTreeUpdater::UpdateStrategy::Lazy);
  ValueToValueMapTy Mapping;
  auto Split =
      DuplicateInstructionsInSplitBetween(BB2, BB2, SubInst, Mapping, DTU);

  EXPECT_TRUE(Split);
  EXPECT_EQ(Mapping.size(), 2u);
  EXPECT_TRUE(Mapping.find(AddInst) != Mapping.end());
  EXPECT_TRUE(Mapping.find(MulInst) != Mapping.end());

  auto AddSplit = dyn_cast<Instruction>(Mapping[AddInst]);
  EXPECT_TRUE(AddSplit);
  EXPECT_EQ(AddSplit->getOperand(0), V);
  EXPECT_EQ(AddSplit->getOperand(1), V);
  EXPECT_EQ(AddSplit->getParent(), Split);

  auto MulSplit = dyn_cast<Instruction>(Mapping[MulInst]);
  EXPECT_TRUE(MulSplit);
  EXPECT_EQ(MulSplit->getOperand(0), AddSplit);
  EXPECT_EQ(MulSplit->getOperand(1), V);
  EXPECT_EQ(MulSplit->getParent(), Split);
  EXPECT_EQ(MulSplit->getNextNode(), Split->getTerminator());
  EXPECT_EQ(Split->getSingleSuccessor(), BB2);
  EXPECT_EQ(BB2->getSingleSuccessor(), Split);

  delete F;
}

static void runWithLoopInfoAndDominatorTree(
    Module &M, StringRef FuncName,
    function_ref<void(Function &F, LoopInfo &LI, DominatorTree &DT)> Test) {
  auto *F = M.getFunction(FuncName);
  ASSERT_NE(F, nullptr) << "Could not find " << FuncName;

  DominatorTree DT(*F);
  LoopInfo LI(DT);

  Test(*F, LI, DT);
}

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("CloneLoop", errs());
  return Mod;
}

TEST(CloneLoop, CloneLoopNest) {
  // Parse the module.
  LLVMContext Context;

  std::unique_ptr<Module> M = parseIR(
    Context,
    R"(define void @foo(i32* %A, i32 %ub) {
entry:
  %guardcmp = icmp slt i32 0, %ub
  br i1 %guardcmp, label %for.outer.preheader, label %for.end
for.outer.preheader:
  br label %for.outer
for.outer:
  %j = phi i32 [ 0, %for.outer.preheader ], [ %inc.outer, %for.outer.latch ]
  br i1 %guardcmp, label %for.inner.preheader, label %for.outer.latch
for.inner.preheader:
  br label %for.inner
for.inner:
  %i = phi i32 [ 0, %for.inner.preheader ], [ %inc, %for.inner ]
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom
  store i32 %i, i32* %arrayidx, align 4
  %inc = add nsw i32 %i, 1
  %cmp = icmp slt i32 %inc, %ub
  br i1 %cmp, label %for.inner, label %for.inner.exit
for.inner.exit:
  br label %for.outer.latch
for.outer.latch:
  %inc.outer = add nsw i32 %j, 1
  %cmp.outer = icmp slt i32 %inc.outer, %ub
  br i1 %cmp.outer, label %for.outer, label %for.outer.exit
for.outer.exit:
  br label %for.end
for.end:
  ret void
})"
    );

  runWithLoopInfoAndDominatorTree(
      *M, "foo", [&](Function &F, LoopInfo &LI, DominatorTree &DT) {
        Function::iterator FI = F.begin();
        // First basic block is entry - skip it.
        BasicBlock *Preheader = &*(++FI);
        BasicBlock *Header = &*(++FI);
        assert(Header->getName() == "for.outer");
        Loop *L = LI.getLoopFor(Header);
        EXPECT_NE(L, nullptr);
        EXPECT_EQ(Header, L->getHeader());
        EXPECT_EQ(Preheader, L->getLoopPreheader());

        ValueToValueMapTy VMap;
        SmallVector<BasicBlock *, 4> ClonedLoopBlocks;
        Loop *NewLoop = cloneLoopWithPreheader(Preheader, Preheader, L, VMap,
                                               "", &LI, &DT, ClonedLoopBlocks);
        EXPECT_NE(NewLoop, nullptr);
        EXPECT_EQ(NewLoop->getSubLoops().size(), 1u);
        Loop::block_iterator BI = NewLoop->block_begin();
        EXPECT_TRUE((*BI)->getName().starts_with("for.outer"));
        EXPECT_TRUE((*(++BI))->getName().starts_with("for.inner.preheader"));
        EXPECT_TRUE((*(++BI))->getName().starts_with("for.inner"));
        EXPECT_TRUE((*(++BI))->getName().starts_with("for.inner.exit"));
        EXPECT_TRUE((*(++BI))->getName().starts_with("for.outer.latch"));
      });
}

class CloneFunc : public ::testing::Test {
protected:
  void SetUp() override {
    SetupModule();
    CreateOldFunc();
    CreateNewFunc();
    SetupFinder();
  }

  void TearDown() override { delete Finder; }

  void SetupModule() {
    M = new Module("", C);
  }

  void CreateOldFunc() {
    FunctionType* FuncType = FunctionType::get(Type::getVoidTy(C), false);
    OldFunc = Function::Create(FuncType, GlobalValue::PrivateLinkage, "f", M);
    CreateOldFunctionBodyAndDI();
  }

  void CreateOldFunctionBodyAndDI() {
    DIBuilder DBuilder(*M);
    IRBuilder<> IBuilder(C);

    // Function DI
    auto *File = DBuilder.createFile("filename.c", "/file/dir/");
    DITypeRefArray ParamTypes = DBuilder.getOrCreateTypeArray({});
    DISubroutineType *FuncType =
        DBuilder.createSubroutineType(ParamTypes);
    auto *CU = DBuilder.createCompileUnit(dwarf::DW_LANG_C99,
                                          DBuilder.createFile("filename.c",
                                                              "/file/dir"),
                                          "CloneFunc", false, "", 0);

    auto *Subprogram = DBuilder.createFunction(
        CU, "f", "f", File, 4, FuncType, 3, DINode::FlagZero,
        DISubprogram::SPFlagLocalToUnit | DISubprogram::SPFlagDefinition);
    OldFunc->setSubprogram(Subprogram);

    // Function body
    BasicBlock* Entry = BasicBlock::Create(C, "", OldFunc);
    IBuilder.SetInsertPoint(Entry);
    DebugLoc Loc = DILocation::get(Subprogram->getContext(), 3, 2, Subprogram);
    IBuilder.SetCurrentDebugLocation(Loc);
    AllocaInst* Alloca = IBuilder.CreateAlloca(IntegerType::getInt32Ty(C));
    IBuilder.SetCurrentDebugLocation(
        DILocation::get(Subprogram->getContext(), 4, 2, Subprogram));
    Value* AllocaContent = IBuilder.getInt32(1);
    Instruction* Store = IBuilder.CreateStore(AllocaContent, Alloca);
    IBuilder.SetCurrentDebugLocation(
        DILocation::get(Subprogram->getContext(), 5, 2, Subprogram));

    // Create a local variable around the alloca
    auto *IntType = DBuilder.createBasicType("int", 32, dwarf::DW_ATE_signed);
    auto *E = DBuilder.createExpression();
    auto *Variable =
        DBuilder.createAutoVariable(Subprogram, "x", File, 5, IntType, true);
    auto *DL = DILocation::get(Subprogram->getContext(), 5, 0, Subprogram);
    DBuilder.insertDeclare(Alloca, Variable, E, DL, Store);
    DBuilder.insertDbgValueIntrinsic(AllocaContent, Variable, E, DL, Entry);
    // Also create an inlined variable.
    // Create a distinct struct type that we should not duplicate during
    // cloning).
    auto *StructType = DICompositeType::getDistinct(
        C, dwarf::DW_TAG_structure_type, "some_struct", nullptr, 0, nullptr,
        nullptr, 32, 32, 0, DINode::FlagZero, nullptr, 0, nullptr, nullptr);
    auto *InlinedSP = DBuilder.createFunction(
        CU, "inlined", "inlined", File, 8, FuncType, 9, DINode::FlagZero,
        DISubprogram::SPFlagLocalToUnit | DISubprogram::SPFlagDefinition);
    auto *InlinedVar =
        DBuilder.createAutoVariable(InlinedSP, "inlined", File, 5, StructType, true);
    auto *Scope = DBuilder.createLexicalBlock(
        DBuilder.createLexicalBlockFile(InlinedSP, File), File, 1, 1);
    auto InlinedDL = DILocation::get(
        Subprogram->getContext(), 9, 4, Scope,
        DILocation::get(Subprogram->getContext(), 5, 2, Subprogram));
    IBuilder.SetCurrentDebugLocation(InlinedDL);
    DBuilder.insertDeclare(Alloca, InlinedVar, E, InlinedDL, Store);
    IBuilder.CreateStore(IBuilder.getInt32(2), Alloca);
    // Finalize the debug info.
    DBuilder.finalize();
    IBuilder.CreateRetVoid();

    // Create another, empty, compile unit.
    DIBuilder DBuilder2(*M);
    DBuilder2.createCompileUnit(dwarf::DW_LANG_C99,
                                DBuilder.createFile("extra.c", "/file/dir"),
                                "CloneFunc", false, "", 0);
    DBuilder2.finalize();
  }

  void CreateNewFunc() {
    ValueToValueMapTy VMap;
    NewFunc = CloneFunction(OldFunc, VMap, nullptr);
  }

  void SetupFinder() {
    Finder = new DebugInfoFinder();
    Finder->processModule(*M);
  }

  LLVMContext C;
  Function* OldFunc;
  Function* NewFunc;
  Module* M;
  DebugInfoFinder* Finder;
};

// Test that a new, distinct function was created.
TEST_F(CloneFunc, NewFunctionCreated) {
  EXPECT_NE(OldFunc, NewFunc);
}

// Test that a new subprogram entry was added and is pointing to the new
// function, while the original subprogram still points to the old one.
TEST_F(CloneFunc, Subprogram) {
  EXPECT_FALSE(verifyModule(*M, &errs()));
  EXPECT_EQ(3U, Finder->subprogram_count());
  EXPECT_NE(NewFunc->getSubprogram(), OldFunc->getSubprogram());
}

// Test that instructions in the old function still belong to it in the
// metadata, while instruction in the new function belong to the new one.
TEST_F(CloneFunc, InstructionOwnership) {
  EXPECT_FALSE(verifyModule(*M));

  inst_iterator OldIter = inst_begin(OldFunc);
  inst_iterator OldEnd = inst_end(OldFunc);
  inst_iterator NewIter = inst_begin(NewFunc);
  inst_iterator NewEnd = inst_end(NewFunc);
  while (OldIter != OldEnd && NewIter != NewEnd) {
    Instruction& OldI = *OldIter;
    Instruction& NewI = *NewIter;
    EXPECT_NE(&OldI, &NewI);

    EXPECT_EQ(OldI.hasMetadata(), NewI.hasMetadata());
    if (OldI.hasMetadata()) {
      const DebugLoc& OldDL = OldI.getDebugLoc();
      const DebugLoc& NewDL = NewI.getDebugLoc();

      // Verify that the debug location data is the same
      EXPECT_EQ(OldDL.getLine(), NewDL.getLine());
      EXPECT_EQ(OldDL.getCol(), NewDL.getCol());

      // But that they belong to different functions
      auto *OldSubprogram = cast<DISubprogram>(OldDL.getInlinedAtScope());
      auto *NewSubprogram = cast<DISubprogram>(NewDL.getInlinedAtScope());
      EXPECT_EQ(OldFunc->getSubprogram(), OldSubprogram);
      EXPECT_EQ(NewFunc->getSubprogram(), NewSubprogram);
    }

    ++OldIter;
    ++NewIter;
  }
  EXPECT_EQ(OldEnd, OldIter);
  EXPECT_EQ(NewEnd, NewIter);
}

// Test that the arguments for debug intrinsics in the new function were
// properly cloned
TEST_F(CloneFunc, DebugIntrinsics) {
  EXPECT_FALSE(verifyModule(*M));

  inst_iterator OldIter = inst_begin(OldFunc);
  inst_iterator OldEnd = inst_end(OldFunc);
  inst_iterator NewIter = inst_begin(NewFunc);
  inst_iterator NewEnd = inst_end(NewFunc);
  while (OldIter != OldEnd && NewIter != NewEnd) {
    Instruction& OldI = *OldIter;
    Instruction& NewI = *NewIter;
    if (DbgDeclareInst* OldIntrin = dyn_cast<DbgDeclareInst>(&OldI)) {
      DbgDeclareInst* NewIntrin = dyn_cast<DbgDeclareInst>(&NewI);
      EXPECT_TRUE(NewIntrin);

      // Old address must belong to the old function
      EXPECT_EQ(OldFunc, cast<AllocaInst>(OldIntrin->getAddress())->
                         getParent()->getParent());
      // New address must belong to the new function
      EXPECT_EQ(NewFunc, cast<AllocaInst>(NewIntrin->getAddress())->
                         getParent()->getParent());

      if (OldIntrin->getDebugLoc()->getInlinedAt()) {
        // Inlined variable should refer to the same DILocalVariable as in the
        // Old Function
        EXPECT_EQ(OldIntrin->getVariable(), NewIntrin->getVariable());
      } else {
        // Old variable must belong to the old function.
        EXPECT_EQ(OldFunc->getSubprogram(),
                  cast<DISubprogram>(OldIntrin->getVariable()->getScope()));
        // New variable must belong to the new function.
        EXPECT_EQ(NewFunc->getSubprogram(),
                  cast<DISubprogram>(NewIntrin->getVariable()->getScope()));
      }
    } else if (DbgValueInst* OldIntrin = dyn_cast<DbgValueInst>(&OldI)) {
      DbgValueInst* NewIntrin = dyn_cast<DbgValueInst>(&NewI);
      EXPECT_TRUE(NewIntrin);

      if (!OldIntrin->getDebugLoc()->getInlinedAt()) {
        // Old variable must belong to the old function.
        EXPECT_EQ(OldFunc->getSubprogram(),
                  cast<DISubprogram>(OldIntrin->getVariable()->getScope()));
        // New variable must belong to the new function.
        EXPECT_EQ(NewFunc->getSubprogram(),
                  cast<DISubprogram>(NewIntrin->getVariable()->getScope()));
      }
    }

    ++OldIter;
    ++NewIter;
  }
}

static int GetDICompileUnitCount(const Module& M) {
  if (const auto* LLVM_DBG_CU = M.getNamedMetadata("llvm.dbg.cu")) {
    return LLVM_DBG_CU->getNumOperands();
  }
  return 0;
}

static bool haveCompileUnitsInCommon(const Module &LHS, const Module &RHS) {
  const NamedMDNode *LHSCUs = LHS.getNamedMetadata("llvm.dbg.cu");
  if (!LHSCUs)
    return false;

  const NamedMDNode *RHSCUs = RHS.getNamedMetadata("llvm.dbg.cu");
  if (!RHSCUs)
    return false;

  SmallPtrSet<const MDNode *, 8> Found;
  for (int I = 0, E = LHSCUs->getNumOperands(); I != E; ++I)
    if (const MDNode *N = LHSCUs->getOperand(I))
      Found.insert(N);

  for (int I = 0, E = RHSCUs->getNumOperands(); I != E; ++I)
    if (const MDNode *N = RHSCUs->getOperand(I))
      if (Found.count(N))
        return true;

  return false;
}

TEST(CloneFunction, CloneEmptyFunction) {
  StringRef ImplAssembly = R"(
    define void @foo() {
      ret void
    }
    declare void @bar()
  )";

  LLVMContext Context;
  SMDiagnostic Error;

  auto ImplModule = parseAssemblyString(ImplAssembly, Error, Context);
  EXPECT_TRUE(ImplModule != nullptr);
  auto *ImplFunction = ImplModule->getFunction("foo");
  EXPECT_TRUE(ImplFunction != nullptr);
  auto *DeclFunction = ImplModule->getFunction("bar");
  EXPECT_TRUE(DeclFunction != nullptr);

  ValueToValueMapTy VMap;
  SmallVector<ReturnInst *, 8> Returns;
  ClonedCodeInfo CCI;
  CloneFunctionInto(ImplFunction, DeclFunction, VMap,
                    CloneFunctionChangeType::GlobalChanges, Returns, "", &CCI);

  EXPECT_FALSE(verifyModule(*ImplModule, &errs()));
  EXPECT_FALSE(CCI.ContainsCalls);
  EXPECT_FALSE(CCI.ContainsDynamicAllocas);
}

TEST(CloneFunction, CloneFunctionWithInalloca) {
  StringRef ImplAssembly = R"(
    declare void @a(i32* inalloca(i32))
    define void @foo() {
      %a = alloca inalloca i32
      call void @a(i32* inalloca(i32) %a)
      ret void
    }
    declare void @bar()
  )";

  LLVMContext Context;
  SMDiagnostic Error;

  auto ImplModule = parseAssemblyString(ImplAssembly, Error, Context);
  EXPECT_TRUE(ImplModule != nullptr);
  auto *ImplFunction = ImplModule->getFunction("foo");
  EXPECT_TRUE(ImplFunction != nullptr);
  auto *DeclFunction = ImplModule->getFunction("bar");
  EXPECT_TRUE(DeclFunction != nullptr);

  ValueToValueMapTy VMap;
  SmallVector<ReturnInst *, 8> Returns;
  ClonedCodeInfo CCI;
  CloneFunctionInto(DeclFunction, ImplFunction, VMap,
                    CloneFunctionChangeType::GlobalChanges, Returns, "", &CCI);

  EXPECT_FALSE(verifyModule(*ImplModule, &errs()));
  EXPECT_TRUE(CCI.ContainsCalls);
  EXPECT_TRUE(CCI.ContainsDynamicAllocas);
}

TEST(CloneFunction, CloneFunctionWithSubprograms) {
  // Tests that the debug info is duplicated correctly when a DISubprogram
  // happens to be one of the operands of the DISubprogram that is being cloned.
  // In general, operands of "test" that are distinct should be duplicated,
  // but in this case "my_operator" should not be duplicated. If it is
  // duplicated, the metadata in the llvm.dbg.declare could end up with
  // different duplicates.
  StringRef ImplAssembly = R"(
    declare void @llvm.dbg.declare(metadata, metadata, metadata)

    define void @test() !dbg !5 {
      call void @llvm.dbg.declare(metadata i8* undef, metadata !4, metadata !DIExpression()), !dbg !6
      ret void
    }

    declare void @cloned()

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!2}
    !0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
    !1 = !DIFile(filename: "test.cpp",  directory: "")
    !2 = !{i32 1, !"Debug Info Version", i32 3}
    !3 = distinct !DISubprogram(name: "my_operator", scope: !1, unit: !0, retainedNodes: !{!4})
    !4 = !DILocalVariable(name: "awaitables", scope: !3)
    !5 = distinct !DISubprogram(name: "test", scope: !3, unit: !0)
    !6 = !DILocation(line: 55, column: 15, scope: !3, inlinedAt: !7)
    !7 = distinct !DILocation(line: 73, column: 14, scope: !5)
  )";

  LLVMContext Context;
  SMDiagnostic Error;

  auto ImplModule = parseAssemblyString(ImplAssembly, Error, Context);
  EXPECT_TRUE(ImplModule != nullptr);
  auto *OldFunc = ImplModule->getFunction("test");
  EXPECT_TRUE(OldFunc != nullptr);
  auto *NewFunc = ImplModule->getFunction("cloned");
  EXPECT_TRUE(NewFunc != nullptr);

  ValueToValueMapTy VMap;
  SmallVector<ReturnInst *, 8> Returns;
  ClonedCodeInfo CCI;
  CloneFunctionInto(NewFunc, OldFunc, VMap,
                    CloneFunctionChangeType::GlobalChanges, Returns, "", &CCI);

  // This fails if the scopes in the llvm.dbg.declare variable and location
  // aren't the same.
  EXPECT_FALSE(verifyModule(*ImplModule, &errs()));
}

TEST(CloneFunction, CloneFunctionWithInlinedSubprograms) {
  StringRef ImplAssembly = R"(
    declare void @llvm.dbg.declare(metadata, metadata, metadata)

    define void @test() !dbg !3 {
      call void @llvm.dbg.declare(metadata i8* undef, metadata !5, metadata !DIExpression()), !dbg !7
      ret void
    }

    declare void @cloned()

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!2}
    !0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
    !1 = !DIFile(filename: "test.cpp",  directory: "")
    !2 = !{i32 1, !"Debug Info Version", i32 3}
    !3 = distinct !DISubprogram(name: "test", scope: !0, unit: !0)
    !4 = distinct !DISubprogram(name: "inlined", scope: !0, unit: !0, retainedNodes: !{!5})
    !5 = !DILocalVariable(name: "awaitables", scope: !4)
    !6 = distinct !DILexicalBlock(scope: !4, file: !1, line: 1)
    !7 = !DILocation(line: 1, scope: !6, inlinedAt: !8)
    !8 = !DILocation(line: 10, scope: !3)
  )";

  LLVMContext Context;
  SMDiagnostic Error;

  auto ImplModule = parseAssemblyString(ImplAssembly, Error, Context);
  EXPECT_TRUE(ImplModule != nullptr);
  auto *Func = ImplModule->getFunction("test");
  EXPECT_TRUE(Func != nullptr);
  auto *ClonedFunc = ImplModule->getFunction("cloned");
  EXPECT_TRUE(ClonedFunc != nullptr);

  ValueToValueMapTy VMap;
  SmallVector<ReturnInst *, 8> Returns;
  ClonedCodeInfo CCI;
  CloneFunctionInto(ClonedFunc, Func, VMap,
                    CloneFunctionChangeType::GlobalChanges, Returns, "", &CCI);

  EXPECT_FALSE(verifyModule(*ImplModule, &errs()));

  // Check that DILexicalBlock of inlined function was not cloned.
  auto DbgDeclareI = Func->begin()->begin()->getDbgRecordRange().begin();
  auto ClonedDbgDeclareI =
      ClonedFunc->begin()->begin()->getDbgRecordRange().begin();
  const DebugLoc &DbgLoc = DbgDeclareI->getDebugLoc();
  const DebugLoc &ClonedDbgLoc = ClonedDbgDeclareI->getDebugLoc();
  EXPECT_NE(DbgLoc.get(), ClonedDbgLoc.get());
  EXPECT_EQ(cast<DILexicalBlock>(DbgLoc.getScope()),
            cast<DILexicalBlock>(ClonedDbgLoc.getScope()));
}

TEST(CloneFunction, CloneFunctionToDifferentModule) {
  StringRef ImplAssembly = R"(
    define void @foo() {
      ret void, !dbg !5
    }

    !llvm.module.flags = !{!0}
    !llvm.dbg.cu = !{!2, !6}
    !0 = !{i32 1, !"Debug Info Version", i32 3}
    !1 = distinct !DISubprogram(unit: !2)
    !2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3)
    !3 = !DIFile(filename: "foo.c", directory: "/tmp")
    !4 = distinct !DISubprogram(unit: !2)
    !5 = !DILocation(line: 4, scope: !1)
    !6 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3)
  )";
  StringRef DeclAssembly = R"(
    declare void @foo()
  )";

  LLVMContext Context;
  SMDiagnostic Error;

  auto ImplModule = parseAssemblyString(ImplAssembly, Error, Context);
  EXPECT_TRUE(ImplModule != nullptr);
  // DICompileUnits: !2, !6. Only !2 is reachable from @foo().
  EXPECT_TRUE(GetDICompileUnitCount(*ImplModule) == 2);
  auto* ImplFunction = ImplModule->getFunction("foo");
  EXPECT_TRUE(ImplFunction != nullptr);

  auto DeclModule = parseAssemblyString(DeclAssembly, Error, Context);
  EXPECT_TRUE(DeclModule != nullptr);
  // No DICompileUnits defined here.
  EXPECT_TRUE(GetDICompileUnitCount(*DeclModule) == 0);
  auto* DeclFunction = DeclModule->getFunction("foo");
  EXPECT_TRUE(DeclFunction != nullptr);

  ValueToValueMapTy VMap;
  VMap[ImplFunction] = DeclFunction;
  // No args to map
  SmallVector<ReturnInst*, 8> Returns;
  CloneFunctionInto(DeclFunction, ImplFunction, VMap,
                    CloneFunctionChangeType::DifferentModule, Returns);

  EXPECT_FALSE(verifyModule(*ImplModule, &errs()));
  EXPECT_FALSE(verifyModule(*DeclModule, &errs()));
  // DICompileUnit !2 shall be cloned into DeclModule.
  EXPECT_TRUE(GetDICompileUnitCount(*DeclModule) == 1);
  EXPECT_FALSE(haveCompileUnitsInCommon(*ImplModule, *DeclModule));
}

class CloneModule : public ::testing::Test {
protected:
  void SetUp() override {
    SetupModule();
    CreateOldModule();
    CreateNewModule();
  }

  void SetupModule() { OldM = new Module("", C); }

  void CreateOldModule() {
    auto *CD = OldM->getOrInsertComdat("comdat");
    CD->setSelectionKind(Comdat::ExactMatch);

    auto GV = new GlobalVariable(
        *OldM, Type::getInt32Ty(C), false, GlobalValue::ExternalLinkage,
        ConstantInt::get(Type::getInt32Ty(C), 1), "gv");
    GV->addMetadata(LLVMContext::MD_type, *MDNode::get(C, {}));
    GV->setComdat(CD);

    // Add ifuncs
    {
      const unsigned AddrSpace = 123;
      auto *FuncPtrTy = PointerType::get(C, AddrSpace);
      auto *FuncTy = FunctionType::get(FuncPtrTy, false);

      auto *ResolverF = Function::Create(FuncTy, GlobalValue::PrivateLinkage,
                                         AddrSpace, "resolver", OldM);
      BasicBlock *ResolverBody = BasicBlock::Create(C, "", ResolverF);
      ReturnInst::Create(C, ConstantPointerNull::get(FuncPtrTy), ResolverBody);

      GlobalIFunc *GI = GlobalIFunc::create(FuncTy, AddrSpace,
                                            GlobalValue::LinkOnceODRLinkage,
                                            "an_ifunc", ResolverF, OldM);
      GI->setVisibility(GlobalValue::ProtectedVisibility);
    }

    {
      // Add an empty compile unit first that isn't otherwise referenced, to
      // confirm that compile units get cloned in the correct order.
      DIBuilder EmptyBuilder(*OldM);
      auto *File = EmptyBuilder.createFile("empty.c", "/file/dir/");
      (void)EmptyBuilder.createCompileUnit(dwarf::DW_LANG_C99, File,
                                           "EmptyUnit", false, "", 0);
      EmptyBuilder.finalize();
    }

    DIBuilder DBuilder(*OldM);
    IRBuilder<> IBuilder(C);

    auto *FuncType = FunctionType::get(Type::getVoidTy(C), false);
    auto *PersFn = Function::Create(FuncType, GlobalValue::ExternalLinkage,
                                    "persfn", OldM);
    auto *F =
        Function::Create(FuncType, GlobalValue::PrivateLinkage, "f", OldM);
    F->setPersonalityFn(PersFn);
    F->setComdat(CD);

    // Create debug info
    auto *File = DBuilder.createFile("filename.c", "/file/dir/");
    DITypeRefArray ParamTypes = DBuilder.getOrCreateTypeArray({});
    DISubroutineType *DFuncType = DBuilder.createSubroutineType(ParamTypes);
    auto *CU = DBuilder.createCompileUnit(dwarf::DW_LANG_C99,
                                          DBuilder.createFile("filename.c",
                                                              "/file/dir"),
                                          "CloneModule", false, "", 0);
    // Function DI
    auto *Subprogram = DBuilder.createFunction(
        CU, "f", "f", File, 4, DFuncType, 3, DINode::FlagZero,
        DISubprogram::SPFlagLocalToUnit | DISubprogram::SPFlagDefinition);
    F->setSubprogram(Subprogram);

    // Create and assign DIGlobalVariableExpression to gv
    auto GVExpression = DBuilder.createGlobalVariableExpression(
        Subprogram, "gv", "gv", File, 1, DBuilder.createNullPtrType(), false);
    GV->addDebugInfo(GVExpression);

    // DIGlobalVariableExpression not attached to any global variable
    auto Expr = DBuilder.createExpression(
        ArrayRef<uint64_t>{dwarf::DW_OP_constu, 42U, dwarf::DW_OP_stack_value});

    DBuilder.createGlobalVariableExpression(
        Subprogram, "unattached", "unattached", File, 1,
        DBuilder.createNullPtrType(), false, true, Expr);

    auto *Entry = BasicBlock::Create(C, "", F);
    IBuilder.SetInsertPoint(Entry);
    IBuilder.CreateRetVoid();

    auto *G =
        Function::Create(FuncType, GlobalValue::ExternalLinkage, "g", OldM);
    G->addMetadata(LLVMContext::MD_type, *MDNode::get(C, {}));

    // Finalize the debug info
    DBuilder.finalize();
  }

  void CreateNewModule() { NewM = llvm::CloneModule(*OldM).release(); }

  LLVMContext C;
  Module *OldM;
  Module *NewM;
};

TEST_F(CloneModule, Verify) {
  // Confirm the old module is (still) valid.
  EXPECT_FALSE(verifyModule(*OldM, &errs()));

  // Check the new module.
  EXPECT_FALSE(verifyModule(*NewM, &errs()));
}

TEST_F(CloneModule, OldModuleUnchanged) {
  DebugInfoFinder Finder;
  Finder.processModule(*OldM);
  EXPECT_EQ(1U, Finder.subprogram_count());
}

TEST_F(CloneModule, Subprogram) {
  Function *NewF = NewM->getFunction("f");
  DISubprogram *SP = NewF->getSubprogram();
  EXPECT_TRUE(SP != nullptr);
  EXPECT_EQ(SP->getName(), "f");
  EXPECT_EQ(SP->getFile()->getFilename(), "filename.c");
  EXPECT_EQ(SP->getLine(), (unsigned)4);
}

TEST_F(CloneModule, FunctionDeclarationMetadata) {
  Function *NewF = NewM->getFunction("g");
  EXPECT_NE(nullptr, NewF->getMetadata(LLVMContext::MD_type));
}

TEST_F(CloneModule, GlobalMetadata) {
  GlobalVariable *NewGV = NewM->getGlobalVariable("gv");
  EXPECT_NE(nullptr, NewGV->getMetadata(LLVMContext::MD_type));
}

TEST_F(CloneModule, GlobalDebugInfo) {
  GlobalVariable *NewGV = NewM->getGlobalVariable("gv");
  EXPECT_TRUE(NewGV != nullptr);

  // Find debug info expression assigned to global
  SmallVector<DIGlobalVariableExpression *, 1> GVs;
  NewGV->getDebugInfo(GVs);
  EXPECT_EQ(GVs.size(), 1U);

  DIGlobalVariableExpression *GVExpr = GVs[0];
  DIGlobalVariable *GV = GVExpr->getVariable();
  EXPECT_TRUE(GV != nullptr);

  EXPECT_EQ(GV->getName(), "gv");
  EXPECT_EQ(GV->getLine(), 1U);

  // Assert that the scope of the debug info attached to
  // global variable matches the cloned function.
  DISubprogram *SP = NewM->getFunction("f")->getSubprogram();
  EXPECT_TRUE(SP != nullptr);
  EXPECT_EQ(GV->getScope(), SP);
}

TEST_F(CloneModule, CompileUnit) {
  // Find DICompileUnit listed in llvm.dbg.cu
  auto *NMD = NewM->getNamedMetadata("llvm.dbg.cu");
  EXPECT_TRUE(NMD != nullptr);
  EXPECT_EQ(NMD->getNumOperands(), 2U);
  EXPECT_FALSE(haveCompileUnitsInCommon(*OldM, *NewM));

  // Check that the empty CU is first, even though it's not referenced except
  // from named metadata.
  DICompileUnit *EmptyCU = dyn_cast<llvm::DICompileUnit>(NMD->getOperand(0));
  EXPECT_TRUE(EmptyCU != nullptr);
  EXPECT_EQ("EmptyUnit", EmptyCU->getProducer());

  // Get the interesting CU.
  DICompileUnit *CU = dyn_cast<llvm::DICompileUnit>(NMD->getOperand(1));
  EXPECT_TRUE(CU != nullptr);
  EXPECT_EQ("CloneModule", CU->getProducer());

  // Assert this CU is consistent with the cloned function debug info
  DISubprogram *SP = NewM->getFunction("f")->getSubprogram();
  EXPECT_TRUE(SP != nullptr);
  EXPECT_EQ(SP->getUnit(), CU);

  // Check globals listed in CU have the correct scope
  DIGlobalVariableExpressionArray GlobalArray = CU->getGlobalVariables();
  EXPECT_EQ(GlobalArray.size(), 2U);
  for (DIGlobalVariableExpression *GVExpr : GlobalArray) {
    DIGlobalVariable *GV = GVExpr->getVariable();
    EXPECT_EQ(GV->getScope(), SP);
  }
}

TEST_F(CloneModule, Comdat) {
  GlobalVariable *NewGV = NewM->getGlobalVariable("gv");
  auto *CD = NewGV->getComdat();
  ASSERT_NE(nullptr, CD);
  EXPECT_EQ("comdat", CD->getName());
  EXPECT_EQ(Comdat::ExactMatch, CD->getSelectionKind());

  Function *NewF = NewM->getFunction("f");
  EXPECT_EQ(CD, NewF->getComdat());
}

TEST_F(CloneModule, IFunc) {
  ASSERT_EQ(1u, NewM->ifunc_size());

  const GlobalIFunc &IFunc = *NewM->ifunc_begin();
  EXPECT_EQ("an_ifunc", IFunc.getName());
  EXPECT_EQ(GlobalValue::LinkOnceODRLinkage, IFunc.getLinkage());
  EXPECT_EQ(GlobalValue::ProtectedVisibility, IFunc.getVisibility());
  EXPECT_EQ(123u, IFunc.getAddressSpace());

  const Function *Resolver = IFunc.getResolverFunction();
  ASSERT_NE(nullptr, Resolver);
  EXPECT_EQ("resolver", Resolver->getName());
  EXPECT_EQ(GlobalValue::PrivateLinkage, Resolver->getLinkage());
}

TEST_F(CloneModule, CloneDbgLabel) {
  LLVMContext Context;

  std::unique_ptr<Module> M = parseIR(Context,
                                      R"M(
define void @noop(ptr nocapture noundef writeonly align 4 %dst) local_unnamed_addr !dbg !3 {
entry:
  %call = tail call spir_func i64 @foo(i32 noundef 0)
    #dbg_label(!11, !12)
  store i64 %call, ptr %dst, align 4
  ret void
}

declare i64 @foo(i32 noundef) local_unnamed_addr

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 19.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "<stdin>", directory: "foo")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "noop", scope: !4, file: !4, line: 17, type: !5, scopeLine: 17, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
!4 = !DIFile(filename: "file", directory: "foo")
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{}
!11 = !DILabel(scope: !3, name: "foo", file: !4, line: 23)
!12 = !DILocation(line: 23, scope: !3)
)M");

  ASSERT_FALSE(verifyModule(*M, &errs()));
  auto NewM = llvm::CloneModule(*M);
  EXPECT_FALSE(verifyModule(*NewM, &errs()));
}
} // namespace
