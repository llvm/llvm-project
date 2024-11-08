//===- RandomIRBuilderTest.cpp - Tests for injector strategy --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/FuzzMutate/RandomIRBuilder.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/AsmParser/SlotMapping.h"
#include "llvm/FuzzMutate/IRMutator.h"
#include "llvm/FuzzMutate/OpDescriptor.h"
#include "llvm/FuzzMutate/Operations.h"
#include "llvm/FuzzMutate/Random.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace llvm;

static constexpr int Seed = 5;

namespace {

std::unique_ptr<Module> parseAssembly(const char *Assembly,
                                      LLVMContext &Context) {

  SMDiagnostic Error;
  std::unique_ptr<Module> M = parseAssemblyString(Assembly, Error, Context);

  std::string ErrMsg;
  raw_string_ostream OS(ErrMsg);
  Error.print("", OS);

  assert(M && !verifyModule(*M, &errs()));
  return M;
}

TEST(RandomIRBuilderTest, ShuffleVectorIncorrectOperands) {
  // Test that we don't create load instruction as a source for the shuffle
  // vector operation.

  LLVMContext Ctx;
  const char *Source =
      "define <2 x i32> @test(<2 x i1> %cond, <2 x i32> %a) {\n"
      "  %A = alloca <2 x i32>\n"
      "  %I = insertelement <2 x i32> %a, i32 1, i32 1\n"
      "  ret <2 x i32> undef\n"
      "}";
  auto M = parseAssembly(Source, Ctx);

  fuzzerop::OpDescriptor Descr = fuzzerop::shuffleVectorDescriptor(1);

  // Empty known types since we ShuffleVector descriptor doesn't care about them
  RandomIRBuilder IB(Seed, {});

  // Get first basic block of the first function
  Function &F = *M->begin();
  BasicBlock &BB = *F.begin();

  SmallVector<Instruction *, 32> Insts;
  for (auto I = BB.getFirstInsertionPt(), E = BB.end(); I != E; ++I)
    Insts.push_back(&*I);

  // Pick first and second sources
  SmallVector<Value *, 2> Srcs;
  ASSERT_TRUE(Descr.SourcePreds[0].matches(Srcs, Insts[1]));
  Srcs.push_back(Insts[1]);
  ASSERT_TRUE(Descr.SourcePreds[1].matches(Srcs, Insts[1]));
  Srcs.push_back(Insts[1]);

  // Create new source. Check that it always matches with the descriptor.
  // Run some iterations to account for random decisions.
  for (int i = 0; i < 10; ++i) {
    Value *LastSrc = IB.newSource(BB, Insts, Srcs, Descr.SourcePreds[2]);
    ASSERT_TRUE(Descr.SourcePreds[2].matches(Srcs, LastSrc));
  }
}

TEST(RandomIRBuilderTest, InsertValueIndexes) {
  // Check that we will generate correct indexes for the insertvalue operation

  LLVMContext Ctx;
  const char *Source = "%T = type {i8, i32, i64}\n"
                       "define void @test() {\n"
                       "  %A = alloca %T\n"
                       "  %L = load %T, ptr %A"
                       "  ret void\n"
                       "}";
  auto M = parseAssembly(Source, Ctx);

  fuzzerop::OpDescriptor IVDescr = fuzzerop::insertValueDescriptor(1);

  std::array<Type *, 3> Types = {Type::getInt8Ty(Ctx), Type::getInt32Ty(Ctx),
                                 Type::getInt64Ty(Ctx)};
  RandomIRBuilder IB(Seed, Types);

  // Get first basic block of the first function
  Function &F = *M->begin();
  BasicBlock &BB = *F.begin();

  // Pick first source
  Instruction *Src = &*std::next(BB.begin());

  SmallVector<Value *, 2> Srcs(2);
  ASSERT_TRUE(IVDescr.SourcePreds[0].matches({}, Src));
  Srcs[0] = Src;

  // Generate constants for each of the types and check that we pick correct
  // index for the given type
  for (auto *T : Types) {
    // Loop to account for possible random decisions
    for (int i = 0; i < 10; ++i) {
      // Create value we want to insert. Only it's type matters.
      Srcs[1] = ConstantInt::get(T, 5);

      // Try to pick correct index
      Value *Src =
          IB.findOrCreateSource(BB, &*BB.begin(), Srcs, IVDescr.SourcePreds[2]);
      ASSERT_TRUE(IVDescr.SourcePreds[2].matches(Srcs, Src));
    }
  }
}

TEST(RandomIRBuilderTest, ShuffleVectorSink) {
  // Check that we will never use shuffle vector mask as a sink from the
  // unrelated operation.

  LLVMContext Ctx;
  const char *SourceCode =
      "define void @test(<4 x i32> %a) {\n"
      "  %S1 = shufflevector <4 x i32> %a, <4 x i32> %a, <4 x i32> undef\n"
      "  %S2 = shufflevector <4 x i32> %a, <4 x i32> %a, <4 x i32> undef\n"
      "  ret void\n"
      "}";
  auto M = parseAssembly(SourceCode, Ctx);

  fuzzerop::OpDescriptor IVDescr = fuzzerop::insertValueDescriptor(1);

  RandomIRBuilder IB(Seed, {});

  // Get first basic block of the first function
  Function &F = *M->begin();
  BasicBlock &BB = *F.begin();

  // Source is %S1
  Instruction *Source = &*BB.begin();
  // Sink is %S2
  SmallVector<Instruction *, 1> Sinks = {&*std::next(BB.begin())};

  // Loop to account for random decisions
  for (int i = 0; i < 10; ++i) {
    // Try to connect S1 to S2. We should always create new sink.
    IB.connectToSink(BB, Sinks, Source);
    ASSERT_TRUE(!verifyModule(*M, &errs()));
  }
}

TEST(RandomIRBuilderTest, InsertValueArray) {
  // Check that we can generate insertvalue for the vector operations

  LLVMContext Ctx;
  const char *SourceCode = "define void @test() {\n"
                           "  %A = alloca [8 x i32]\n"
                           "  %L = load [8 x i32], ptr %A"
                           "  ret void\n"
                           "}";
  auto M = parseAssembly(SourceCode, Ctx);

  fuzzerop::OpDescriptor Descr = fuzzerop::insertValueDescriptor(1);

  std::array<Type *, 3> Types = {Type::getInt8Ty(Ctx), Type::getInt32Ty(Ctx),
                                 Type::getInt64Ty(Ctx)};
  RandomIRBuilder IB(Seed, Types);

  // Get first basic block of the first function
  Function &F = *M->begin();
  BasicBlock &BB = *F.begin();

  // Pick first source
  Instruction *Source = &*std::next(BB.begin());
  ASSERT_TRUE(Descr.SourcePreds[0].matches({}, Source));

  SmallVector<Value *, 2> Srcs(2);

  // Check that we can always pick the last two operands.
  for (int i = 0; i < 10; ++i) {
    Srcs[0] = Source;
    Srcs[1] = IB.findOrCreateSource(BB, {Source}, Srcs, Descr.SourcePreds[1]);
    IB.findOrCreateSource(BB, {}, Srcs, Descr.SourcePreds[2]);
  }
}

TEST(RandomIRBuilderTest, Invokes) {
  // Check that we never generate load or store after invoke instruction

  LLVMContext Ctx;
  const char *SourceCode =
      "declare ptr @f()"
      "declare i32 @personality_function()"
      "define ptr @test() personality ptr @personality_function {\n"
      "entry:\n"
      "  %val = invoke ptr @f()\n"
      "          to label %normal unwind label %exceptional\n"
      "normal:\n"
      "  ret ptr %val\n"
      "exceptional:\n"
      "  %landing_pad4 = landingpad token cleanup\n"
      "  ret ptr undef\n"
      "}";
  auto M = parseAssembly(SourceCode, Ctx);

  std::array<Type *, 1> Types = {Type::getInt8Ty(Ctx)};
  RandomIRBuilder IB(Seed, Types);

  // Get first basic block of the test function
  Function &F = *M->getFunction("test");
  BasicBlock &BB = *F.begin();

  Instruction *Invoke = &*BB.begin();

  // Find source but never insert new load after invoke
  for (int i = 0; i < 10; ++i) {
    (void)IB.findOrCreateSource(BB, {Invoke}, {}, fuzzerop::anyIntType());
    ASSERT_TRUE(!verifyModule(*M, &errs()));
  }
}

TEST(RandomIRBuilderTest, SwiftError) {
  // Check that we never pick swifterror value as a source for operation
  // other than load, store and call.

  LLVMContext Ctx;
  const char *SourceCode = "declare void @use(ptr swifterror %err)"
                           "define void @test() {\n"
                           "entry:\n"
                           "  %err = alloca swifterror ptr, align 8\n"
                           "  call void @use(ptr swifterror %err)\n"
                           "  ret void\n"
                           "}";
  auto M = parseAssembly(SourceCode, Ctx);

  std::array<Type *, 1> Types = {Type::getInt8Ty(Ctx)};
  RandomIRBuilder IB(Seed, Types);

  // Get first basic block of the test function
  Function &F = *M->getFunction("test");
  BasicBlock &BB = *F.begin();
  Instruction *Alloca = &*BB.begin();

  fuzzerop::OpDescriptor Descr = fuzzerop::gepDescriptor(1);

  for (int i = 0; i < 10; ++i) {
    Value *V = IB.findOrCreateSource(BB, {Alloca}, {}, Descr.SourcePreds[0]);
    ASSERT_FALSE(isa<AllocaInst>(V));
  }
}

TEST(RandomIRBuilderTest, dontConnectToSwitch) {
  // Check that we never put anything into switch's case branch
  // If we accidently put a variable, the module is invalid.
  LLVMContext Ctx;
  const char *SourceCode = "\n\
    define void @test(i1 %C1, i1 %C2, i32 %I, i32 %J) { \n\
    Entry:  \n\
      %I.1 = add i32 %I, 42 \n\
      %J.1 = add i32 %J, 42 \n\
      %IJ = add i32 %I, %J \n\
      switch i32 %I, label %Default [ \n\
        i32 1, label %OnOne  \n\
      ] \n\
    Default:  \n\
      %CIEqJ = icmp eq i32 %I.1, %J.1 \n\
      %CISltJ = icmp slt i32 %I.1, %J.1 \n\
      %CAnd = and i1 %C1, %C2 \n\
      br i1 %CIEqJ, label %Default, label %Exit \n\
    OnOne:  \n\
      br i1 %C1, label %OnOne, label %Exit \n\
    Exit:  \n\
      ret void \n\
    }";

  std::array<Type *, 2> Types = {Type::getInt32Ty(Ctx), Type::getInt1Ty(Ctx)};
  RandomIRBuilder IB(Seed, Types);
  for (int i = 0; i < 20; i++) {
    std::unique_ptr<Module> M = parseAssembly(SourceCode, Ctx);
    Function &F = *M->getFunction("test");
    auto RS = makeSampler(IB.Rand, make_pointer_range(F));
    BasicBlock *BB = RS.getSelection();
    SmallVector<Instruction *, 32> Insts;
    for (auto I = BB->getFirstInsertionPt(), E = BB->end(); I != E; ++I)
      Insts.push_back(&*I);
    if (Insts.size() < 2)
      continue;
    // Choose an instruction and connect to later operations.
    size_t IP = uniform<size_t>(IB.Rand, 1, Insts.size() - 1);
    Instruction *Inst = Insts[IP - 1];
    auto ConnectAfter = ArrayRef(Insts).slice(IP);
    IB.connectToSink(*BB, ConnectAfter, Inst);
    ASSERT_FALSE(verifyModule(*M, &errs()));
  }
}

TEST(RandomIRBuilderTest, createStackMemory) {
  LLVMContext Ctx;
  const char *SourceCode = "\n\
    define void @test(i1 %C1, i1 %C2, i32 %I, i32 %J) { \n\
    Entry:  \n\
      ret void \n\
    }";
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Constant *Int32_1 = ConstantInt::get(Int32Ty, APInt(32, 1));
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Constant *Int64_42 = ConstantInt::get(Int64Ty, APInt(64, 42));
  Type *DoubleTy = Type::getDoubleTy(Ctx);
  Constant *Double_0 =
      ConstantFP::get(Ctx, APFloat::getZero(DoubleTy->getFltSemantics()));
  std::array<Type *, 8> Types = {
      Int32Ty,
      Int64Ty,
      DoubleTy,
      PointerType::get(Ctx, 0),
      PointerType::get(Int32Ty, 0),
      VectorType::get(Int32Ty, 4, false),
      StructType::create({Int32Ty, DoubleTy, Int64Ty}),
      ArrayType::get(Int64Ty, 4),
  };
  std::array<Value *, 8> Inits = {
      Int32_1,
      Int64_42,
      Double_0,
      UndefValue::get(Types[3]),
      UndefValue::get(Types[4]),
      ConstantVector::get({Int32_1, Int32_1, Int32_1, Int32_1}),
      ConstantStruct::get(cast<StructType>(Types[6]),
                          {Int32_1, Double_0, Int64_42}),
      ConstantArray::get(cast<ArrayType>(Types[7]),
                         {Int64_42, Int64_42, Int64_42, Int64_42}),
  };
  ASSERT_EQ(Types.size(), Inits.size());
  unsigned NumTests = Types.size();
  RandomIRBuilder IB(Seed, Types);
  auto CreateStackMemoryAndVerify = [&Ctx, &SourceCode, &IB](Type *Ty,
                                                             Value *Init) {
    std::unique_ptr<Module> M = parseAssembly(SourceCode, Ctx);
    Function &F = *M->getFunction("test");
    // Create stack memory without initializer.
    IB.createStackMemory(&F, Ty, nullptr);
    // Create stack memory with initializer.
    IB.createStackMemory(&F, Ty, Init);
    EXPECT_FALSE(verifyModule(*M, &errs()));
  };
  for (unsigned i = 0; i < NumTests; i++) {
    CreateStackMemoryAndVerify(Types[i], Inits[i]);
  }
}

TEST(RandomIRBuilderTest, findOrCreateGlobalVariable) {
  LLVMContext Ctx;
  const char *SourceCode = "\n\
    @G0 = external global i16 \n\
    @G1 = global i32 1 \n\
  ";
  std::array<Type *, 3> Types = {Type::getInt16Ty(Ctx), Type::getInt32Ty(Ctx),
                                 Type::getInt64Ty(Ctx)};
  RandomIRBuilder IB(Seed, Types);

  // Find external global
  std::unique_ptr<Module> M0 = parseAssembly(SourceCode, Ctx);
  Type *ExternalTy = M0->globals().begin()->getValueType();
  ASSERT_TRUE(ExternalTy->isIntegerTy(16));
  IB.findOrCreateGlobalVariable(&*M0, {}, fuzzerop::onlyType(Types[0]));
  ASSERT_FALSE(verifyModule(*M0, &errs()));
  unsigned NumGV0 = M0->getNumNamedValues();
  auto [GV0, DidCreate0] =
      IB.findOrCreateGlobalVariable(&*M0, {}, fuzzerop::onlyType(Types[0]));
  ASSERT_FALSE(verifyModule(*M0, &errs()));
  ASSERT_EQ(M0->getNumNamedValues(), NumGV0 + DidCreate0);

  // Find existing global
  std::unique_ptr<Module> M1 = parseAssembly(SourceCode, Ctx);
  IB.findOrCreateGlobalVariable(&*M1, {}, fuzzerop::onlyType(Types[1]));
  ASSERT_FALSE(verifyModule(*M1, &errs()));
  unsigned NumGV1 = M1->getNumNamedValues();
  auto [GV1, DidCreate1] =
      IB.findOrCreateGlobalVariable(&*M1, {}, fuzzerop::onlyType(Types[1]));
  ASSERT_FALSE(verifyModule(*M1, &errs()));
  ASSERT_EQ(M1->getNumNamedValues(), NumGV1 + DidCreate1);

  // Create new global
  std::unique_ptr<Module> M2 = parseAssembly(SourceCode, Ctx);
  auto [GV2, DidCreate2] =
      IB.findOrCreateGlobalVariable(&*M2, {}, fuzzerop::onlyType(Types[2]));
  ASSERT_FALSE(verifyModule(*M2, &errs()));
  ASSERT_TRUE(DidCreate2);
}

/// Checks if the source and sink we find for an instruction has correct
/// domination relation.
TEST(RandomIRBuilderTest, findSourceAndSink) {
  const char *Source = "\n\
        define i64 @test(i1 %0, i1 %1, i1 %2, i32 %3, i32 %4) { \n\
        Entry:  \n\
          %A = alloca i32, i32 8, align 4 \n\
          %E.1 = and i32 %3, %4 \n\
          %E.2 = add i32 %4 , 1 \n\
          %A.GEP.1 = getelementptr i32, ptr %A, i32 0 \n\
          %A.GEP.2 = getelementptr i32, ptr %A.GEP.1, i32 1 \n\
          %L.2 = load i32, ptr %A.GEP.2 \n\
          %L.1 = load i32, ptr %A.GEP.1 \n\
          %E.3 = sub i32 %E.2, %L.1 \n\
          %Cond.1 = icmp eq i32 %E.3, %E.2 \n\
          %Cond.2 = and i1 %0, %1 \n\
          %Cond = or i1 %Cond.1, %Cond.2 \n\
          br i1 %Cond, label %BB0, label %BB1  \n\
        BB0:  \n\
          %Add = add i32 %L.1, %L.2 \n\
          %Sub = sub i32 %L.1, %L.2 \n\
          %Sub.1 = sub i32 %Sub, 12 \n\
          %Cast.1 = bitcast i32 %4 to float \n\
          %Add.2 = add i32 %3, 1 \n\
          %Cast.2 = bitcast i32 %Add.2 to float \n\
          %FAdd = fadd float %Cast.1, %Cast.2 \n\
          %Add.3 = add i32 %L.2, %L.1 \n\
          %Cast.3 = bitcast float %FAdd to i32 \n\
          %Sub.2 = sub i32 %Cast.3, %Sub.1 \n\
          %SExt = sext i32 %Cast.3 to i64 \n\
          %A.GEP.3 = getelementptr i64, ptr %A, i32 1 \n\
          store i64 %SExt, ptr %A.GEP.3 \n\
          br label %Exit  \n\
        BB1:  \n\
          %PHI.1 = phi i32 [0, %Entry] \n\
          %SExt.1 = sext i1 %Cond.2 to i32 \n\
          %SExt.2 = sext i1 %Cond.1 to i32 \n\
          %E.164 = zext i32 %E.1 to i64 \n\
          %E.264 = zext i32 %E.2 to i64 \n\
          %E.1264 = mul i64 %E.164, %E.264 \n\
          %E.12 = trunc i64 %E.1264 to i32 \n\
          %A.GEP.4 = getelementptr i32, ptr %A, i32 2 \n\
          %A.GEP.5 = getelementptr i32, ptr %A.GEP.4, i32 2 \n\
          store i32 %E.12, ptr %A.GEP.5 \n\
          br label %Exit  \n\
        Exit:  \n\
          %PHI.2 = phi i32 [%Add, %BB0], [%E.3, %BB1] \n\
          %PHI.3 = phi i64 [%SExt, %BB0], [%E.1264, %BB1] \n\
          %ZExt = zext i32 %PHI.2 to i64 \n\
          %Add.5 = add i64 %PHI.3, 3 \n\
          ret i64 %Add.5  \n\
      }";
  LLVMContext Ctx;
  std::array<Type *, 3> Types = {Type::getInt1Ty(Ctx), Type::getInt32Ty(Ctx),
                                 Type::getInt64Ty(Ctx)};
  std::mt19937 mt(Seed);
  std::uniform_int_distribution<int> RandInt(INT_MIN, INT_MAX);

  // Get a random instruction, try to find source and sink, make sure it is
  // dominated.
  for (int i = 0; i < 100; i++) {
    RandomIRBuilder IB(RandInt(mt), Types);
    std::unique_ptr<Module> M = parseAssembly(Source, Ctx);
    Function &F = *M->getFunction("test");
    DominatorTree DT(F);
    BasicBlock *BB = makeSampler(IB.Rand, make_pointer_range(F)).getSelection();
    SmallVector<Instruction *, 32> Insts;
    for (auto I = BB->getFirstInsertionPt(), E = BB->end(); I != E; ++I)
      Insts.push_back(&*I);
    // Choose an insertion point for our new instruction.
    size_t IP = uniform<size_t>(IB.Rand, 1, Insts.size() - 2);

    auto InstsBefore = ArrayRef(Insts).slice(0, IP);
    auto InstsAfter = ArrayRef(Insts).slice(IP);
    Value *Src = IB.findOrCreateSource(
        *BB, InstsBefore, {}, fuzzerop::onlyType(Types[i % Types.size()]));
    ASSERT_TRUE(DT.dominates(Src, Insts[IP + 1]));
    Instruction *Sink = IB.connectToSink(*BB, InstsAfter, Insts[IP - 1]);
    if (!DT.dominates(Insts[IP - 1], Sink)) {
      errs() << *Insts[IP - 1] << "\n" << *Sink << "\n ";
    }
    ASSERT_TRUE(DT.dominates(Insts[IP - 1], Sink));
  }
}
TEST(RandomIRBuilderTest, sinkToIntrinsic) {
  const char *Source = "\n\
        declare double @llvm.sqrt.f64(double %Val)  \n\
        declare void   @llvm.ubsantrap(i8 immarg) cold noreturn nounwind  \n\
        \n\
        define double @test(double %0, double %1, i64 %2, i64 %3, i64 %4, i8 %5) {  \n\
        Entry:   \n\
            %sqrt = call double @llvm.sqrt.f64(double %0)  \n\
            call void @llvm.ubsantrap(i8 1)  \n\
            ret double %sqrt \n\
        }";
  LLVMContext Ctx;
  std::array<Type *, 3> Types = {Type::getInt8Ty(Ctx), Type::getInt64Ty(Ctx),
                                 Type::getDoubleTy(Ctx)};
  std::mt19937 mt(Seed);
  std::uniform_int_distribution<int> RandInt(INT_MIN, INT_MAX);

  RandomIRBuilder IB(RandInt(mt), Types);
  std::unique_ptr<Module> M = parseAssembly(Source, Ctx);
  Function &F = *M->getFunction("test");
  BasicBlock &BB = F.getEntryBlock();
  bool Modified = false;

  Instruction *I = &*BB.begin();
  for (int i = 0; i < 20; i++) {
    Value *OldOperand = I->getOperand(0);
    Value *Src = F.getArg(1);
    IB.connectToSink(BB, {I}, Src);
    Value *NewOperand = I->getOperand(0);
    Modified |= (OldOperand != NewOperand);
    ASSERT_FALSE(verifyModule(*M, &errs()));
  }
  ASSERT_TRUE(Modified);

  Modified = false;
  I = I->getNextNonDebugInstruction();
  for (int i = 0; i < 20; i++) {
    Value *OldOperand = I->getOperand(0);
    Value *Src = F.getArg(5);
    IB.connectToSink(BB, {I}, Src);
    Value *NewOperand = I->getOperand(0);
    Modified |= (OldOperand != NewOperand);
    ASSERT_FALSE(verifyModule(*M, &errs()));
  }
  ASSERT_FALSE(Modified);
}

TEST(RandomIRBuilderTest, DoNotCallPointerWhenSink) {
  const char *Source = "\n\
        declare void @g()  \n\
        define void @f(ptr %ptr) {  \n\
        Entry:   \n\
            call void @g()  \n\
            ret void \n\
        }";
  LLVMContext Ctx;
  std::mt19937 mt(Seed);
  std::uniform_int_distribution<int> RandInt(INT_MIN, INT_MAX);

  RandomIRBuilder IB(RandInt(mt), {});
  std::unique_ptr<Module> M = parseAssembly(Source, Ctx);
  Function &F = *M->getFunction("f");
  BasicBlock &BB = F.getEntryBlock();
  bool Modified = false;

  Instruction *I = &*BB.begin();
  for (int i = 0; i < 20; i++) {
    Value *OldOperand = I->getOperand(0);
    Value *Src = F.getArg(0);
    IB.connectToSink(BB, {I}, Src);
    Value *NewOperand = I->getOperand(0);
    Modified |= (OldOperand != NewOperand);
    ASSERT_FALSE(verifyModule(*M, &errs()));
  }
  ASSERT_FALSE(Modified);
}

TEST(RandomIRBuilderTest, SrcAndSinkWOrphanBlock) {
  const char *Source = "\n\
        define i1 @test(i1 %Bool, i32 %Int, i64 %Long) {   \n\
        Entry:    \n\
            %Eq0 = icmp eq i64 %Long, 0 \n\
            br i1 %Eq0, label %True, label %False \n\
        True: \n\
            %Or = or i1 %Bool, %Eq0 \n\
            ret i1 %Or \n\
        False: \n\
            %And = and i1 %Bool, %Eq0 \n\
            ret i1 %And \n\
        Orphan_1:  \n\
            %NotBool = sub i1 1, %Bool \n\
            ret i1 %NotBool \n\
        Orphan_2:  \n\
            %Le42 = icmp sle i32 %Int, 42 \n\
            ret i1 %Le42 \n\
        }";
  LLVMContext Ctx;
  std::mt19937 mt(Seed);
  std::uniform_int_distribution<int> RandInt(INT_MIN, INT_MAX);
  std::array<Type *, 3> IntTys(
      {Type::getInt64Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt1Ty(Ctx)});
  std::vector<Value *> Constants;
  for (Type *IntTy : IntTys) {
    for (size_t v : {1, 42}) {
      Constants.push_back(ConstantInt::get(IntTy, v));
    }
  }
  for (int i = 0; i < 10; i++) {
    RandomIRBuilder IB(RandInt(mt), IntTys);
    std::unique_ptr<Module> M = parseAssembly(Source, Ctx);
    Function &F = *M->getFunction("test");
    for (BasicBlock &BB : F) {
      SmallVector<Instruction *, 4> Insts;
      for (Instruction &I : BB) {
        Insts.push_back(&I);
      }
      for (int j = 0; j < 10; j++) {
        IB.findOrCreateSource(BB, Insts);
      }
      for (Value *V : Constants) {
        IB.connectToSink(BB, Insts, V);
      }
    }
  }
}
} // namespace
