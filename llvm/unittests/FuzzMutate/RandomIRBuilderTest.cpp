//===- RandomIRBuilderTest.cpp - Tests for injector strategy --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/FuzzMutate/RandomIRBuilder.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/AsmParser/SlotMapping.h"
#include "llvm/FuzzMutate/IRMutator.h"
#include "llvm/FuzzMutate/OpDescriptor.h"
#include "llvm/FuzzMutate/Operations.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace llvm;

static constexpr int Seed = 5;

namespace {

std::unique_ptr<Module> parseAssembly(
    const char *Assembly, LLVMContext &Context) {

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
  const char *Source =
      "%T = type {i8, i32, i64}\n"
      "define void @test() {\n"
      "  %A = alloca %T\n"
      "  %L = load %T, %T* %A"
      "  ret void\n"
      "}";
  auto M = parseAssembly(Source, Ctx);

  fuzzerop::OpDescriptor IVDescr = fuzzerop::insertValueDescriptor(1);

  std::vector<Type *> Types =
      {Type::getInt8Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx)};
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
  for (auto *T: Types) {
    // Loop to account for possible random decisions
    for (int i = 0; i < 10; ++i) {
      // Create value we want to insert. Only it's type matters.
      Srcs[1] = ConstantInt::get(T, 5);

      // Try to pick correct index
      Value *Src = IB.findOrCreateSource(
          BB, &*BB.begin(), Srcs, IVDescr.SourcePreds[2]);
      ASSERT_TRUE(IVDescr.SourcePreds[2].matches(Srcs, Src));
    }
  }
}

TEST(RandomIRBuilderTest, ShuffleVectorSink) {
  // Check that we will never use shuffle vector mask as a sink form the
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
  const char *SourceCode =
      "define void @test() {\n"
      "  %A = alloca [8 x i32]\n"
      "  %L = load [8 x i32], [8 x i32]* %A"
      "  ret void\n"
      "}";
  auto M = parseAssembly(SourceCode, Ctx);

  fuzzerop::OpDescriptor Descr = fuzzerop::insertValueDescriptor(1);

  std::vector<Type *> Types =
      {Type::getInt8Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt64Ty(Ctx)};
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
      "declare i32* @f()"
      "declare i32 @personality_function()"
      "define i32* @test() personality i32 ()* @personality_function {\n"
      "entry:\n"
      "  %val = invoke i32* @f()\n"
      "          to label %normal unwind label %exceptional\n"
      "normal:\n"
      "  ret i32* %val\n"
      "exceptional:\n"
      "  %landing_pad4 = landingpad token cleanup\n"
      "  ret i32* undef\n"
      "}";
  auto M = parseAssembly(SourceCode, Ctx);


  std::vector<Type *> Types = {Type::getInt8Ty(Ctx)};
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

TEST(RandomIRBuilderTest, FirstClassTypes) {
  // Check that we never insert new source as a load from non first class
  // or unsized type.

  LLVMContext Ctx;
  const char *SourceCode = "%Opaque = type opaque\n"
                           "define void @test(i8* %ptr) {\n"
                           "entry:\n"
                           "  %tmp = bitcast i8* %ptr to i32* (i32*)*\n"
                           "  %tmp1 = bitcast i8* %ptr to %Opaque*\n"
                           "  ret void\n"
                           "}";
  auto M = parseAssembly(SourceCode, Ctx);

  std::vector<Type *> Types = {Type::getInt8Ty(Ctx)};
  RandomIRBuilder IB(Seed, Types);

  Function &F = *M->getFunction("test");
  BasicBlock &BB = *F.begin();
  // Non first class type
  Instruction *FuncPtr = &*BB.begin();
  // Unsized type
  Instruction *OpaquePtr = &*std::next(BB.begin());

  for (int i = 0; i < 10; ++i) {
    Value *V = IB.findOrCreateSource(BB, {FuncPtr, OpaquePtr});
    ASSERT_FALSE(isa<LoadInst>(V));
  }
}

}
