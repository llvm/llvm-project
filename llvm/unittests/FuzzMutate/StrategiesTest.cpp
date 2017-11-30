//===- InjectorIRStrategyTest.cpp - Tests for injector strategy -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/AsmParser/SlotMapping.h"
#include "llvm/FuzzMutate/IRMutator.h"
#include "llvm/FuzzMutate/Operations.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace llvm;

static constexpr int Seed = 5;

namespace {

std::unique_ptr<IRMutator> createInjectorMutator() {
  std::vector<TypeGetter> Types{
      Type::getInt1Ty,  Type::getInt8Ty,  Type::getInt16Ty, Type::getInt32Ty,
      Type::getInt64Ty, Type::getFloatTy, Type::getDoubleTy};

  std::vector<std::unique_ptr<IRMutationStrategy>> Strategies;
  Strategies.push_back(
      llvm::make_unique<InjectorIRStrategy>(
          InjectorIRStrategy::getDefaultOps()));

  return llvm::make_unique<IRMutator>(std::move(Types), std::move(Strategies));
}

std::unique_ptr<IRMutator> createDeleterMutator() {
  std::vector<TypeGetter> Types{
      Type::getInt1Ty,  Type::getInt8Ty,  Type::getInt16Ty, Type::getInt32Ty,
      Type::getInt64Ty, Type::getFloatTy, Type::getDoubleTy};

  std::vector<std::unique_ptr<IRMutationStrategy>> Strategies;
  Strategies.push_back(llvm::make_unique<InstDeleterIRStrategy>());

  return llvm::make_unique<IRMutator>(std::move(Types), std::move(Strategies));
}

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

TEST(InjectorIRStrategyTest, EmptyModule) {
  // Test that we can inject into empty module

  LLVMContext Ctx;
  auto M = llvm::make_unique<Module>("M", Ctx);
  ASSERT_TRUE(M && !verifyModule(*M, &errs()));

  auto Mutator = createInjectorMutator();
  ASSERT_TRUE(Mutator);

  Mutator->mutateModule(*M, Seed, 1, 1);
  EXPECT_TRUE(!verifyModule(*M, &errs()));
}

TEST(InstDeleterIRStrategyTest, EmptyFunction) {
  // Test that we don't crash even if we can't remove from one of the functions.

  LLVMContext Ctx;
  StringRef Source = ""
      "define <8 x i32> @func1() {\n"
        "ret <8 x i32> undef\n"
      "}\n"
      "\n"
      "define i32 @func2() {\n"
        "%A9 = alloca i32\n"
        "%L6 = load i32, i32* %A9\n"
        "ret i32 %L6\n"
      "}\n";

  auto Mutator = createDeleterMutator();
  ASSERT_TRUE(Mutator);

  // We need to choose 'func1' in order for the crash to appear.
  // Loop 10 times and assume we are lucky.
  for (int i = 0; i < 10; ++i) {
    auto M = parseAssembly(Source.data(), Ctx);
    ASSERT_TRUE(M && !verifyModule(*M, &errs()));

    Mutator->mutateModule(*M, Seed, Source.size(), Source.size() + 100);
    EXPECT_TRUE(!verifyModule(*M, &errs()));
  }
}

}
