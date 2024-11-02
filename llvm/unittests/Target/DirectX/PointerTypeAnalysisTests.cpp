//===- llvm/unittests/Target/DirectX/PointerTypeAnalysisTests.cpp ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PointerTypeAnalysis.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/TypedPointerType.h"
#include "llvm/Support/SourceMgr.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::Contains;
using ::testing::Pair;

using namespace llvm;
using namespace llvm::dxil;

template <typename T> struct IsA {
  friend bool operator==(const Value *V, const IsA &) { return isa<T>(V); }
};

TEST(PointerTypeAnalysis, DigressToi8) {
  StringRef Assembly = R"(
    define i64 @test(ptr %p) {
      store i32 0, ptr %p
      %v = load i64, ptr %p
      ret i64 %v
    }
  )";

  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyString(Assembly, Error, Context);
  ASSERT_TRUE(M) << "Bad assembly?";

  PointerTypeMap Map = PointerTypeAnalysis::run(*M);
  ASSERT_EQ(Map.size(), 2u);
  Type *I8Ptr = TypedPointerType::get(Type::getInt8Ty(Context), 0);
  Type *FnTy = FunctionType::get(Type::getInt64Ty(Context), {I8Ptr}, false);

  EXPECT_THAT(Map,
              Contains(Pair(IsA<Function>(), TypedPointerType::get(FnTy, 0))));
  EXPECT_THAT(Map, Contains(Pair(IsA<Argument>(), I8Ptr)));
}

TEST(PointerTypeAnalysis, DiscoverStore) {
  StringRef Assembly = R"(
    define i32 @test(ptr %p) {
      store i32 0, ptr %p
      ret i32 0
    }
  )";

  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyString(Assembly, Error, Context);
  ASSERT_TRUE(M) << "Bad assembly?";

  PointerTypeMap Map = PointerTypeAnalysis::run(*M);
  ASSERT_EQ(Map.size(), 2u);
  Type *I32Ptr = TypedPointerType::get(Type::getInt32Ty(Context), 0);
  Type *FnTy = FunctionType::get(Type::getInt32Ty(Context), {I32Ptr}, false);

  EXPECT_THAT(Map,
              Contains(Pair(IsA<Function>(), TypedPointerType::get(FnTy, 0))));
  EXPECT_THAT(Map, Contains(Pair(IsA<Argument>(), I32Ptr)));
}

TEST(PointerTypeAnalysis, DiscoverLoad) {
  StringRef Assembly = R"(
    define i32 @test(ptr %p) {
      %v = load i32, ptr %p
      ret i32 %v
    }
  )";

  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyString(Assembly, Error, Context);
  ASSERT_TRUE(M) << "Bad assembly?";

  PointerTypeMap Map = PointerTypeAnalysis::run(*M);
  ASSERT_EQ(Map.size(), 2u);
  Type *I32Ptr = TypedPointerType::get(Type::getInt32Ty(Context), 0);
  Type *FnTy = FunctionType::get(Type::getInt32Ty(Context), {I32Ptr}, false);

  EXPECT_THAT(Map,
              Contains(Pair(IsA<Function>(), TypedPointerType::get(FnTy, 0))));
  EXPECT_THAT(Map, Contains(Pair(IsA<Argument>(), I32Ptr)));
}

TEST(PointerTypeAnalysis, DiscoverGEP) {
  StringRef Assembly = R"(
    define ptr @test(ptr %p) {
      %p2 = getelementptr i64, ptr %p, i64 1
      ret ptr %p2
    }
  )";

  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyString(Assembly, Error, Context);
  ASSERT_TRUE(M) << "Bad assembly?";

  PointerTypeMap Map = PointerTypeAnalysis::run(*M);
  ASSERT_EQ(Map.size(), 3u);

  Type *I64Ptr = TypedPointerType::get(Type::getInt64Ty(Context), 0);
  Type *FnTy = FunctionType::get(I64Ptr, {I64Ptr}, false);

  EXPECT_THAT(Map,
              Contains(Pair(IsA<Function>(), TypedPointerType::get(FnTy, 0))));
  EXPECT_THAT(Map, Contains(Pair(IsA<Argument>(), I64Ptr)));
  EXPECT_THAT(Map, Contains(Pair(IsA<GetElementPtrInst>(), I64Ptr)));
}

TEST(PointerTypeAnalysis, TraceIndirect) {
  StringRef Assembly = R"(
    define i64 @test(ptr %p) {
      %p2 = load ptr, ptr %p
      %v = load i64, ptr %p2
      ret i64 %v
    }
  )";

  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyString(Assembly, Error, Context);
  ASSERT_TRUE(M) << "Bad assembly?";

  PointerTypeMap Map = PointerTypeAnalysis::run(*M);
  ASSERT_EQ(Map.size(), 3u);

  Type *I64Ptr = TypedPointerType::get(Type::getInt64Ty(Context), 0);
  Type *I64PtrPtr = TypedPointerType::get(I64Ptr, 0);
  Type *FnTy = FunctionType::get(Type::getInt64Ty(Context), {I64PtrPtr}, false);

  EXPECT_THAT(Map,
              Contains(Pair(IsA<Function>(), TypedPointerType::get(FnTy, 0))));
  EXPECT_THAT(Map, Contains(Pair(IsA<Argument>(), I64PtrPtr)));
  EXPECT_THAT(Map, Contains(Pair(IsA<LoadInst>(), I64Ptr)));
}

TEST(PointerTypeAnalysis, WithNoOpCasts) {
  StringRef Assembly = R"(
    define i64 @test(ptr %p) {
      %1 = bitcast ptr %p to ptr
      %2 = bitcast ptr %p to ptr
      store i32 0, ptr %1, align 4
      %3 = load i64, ptr %2, align 8
      ret i64 %3
    }
  )";

  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyString(Assembly, Error, Context);
  ASSERT_TRUE(M) << "Bad assembly?";

  PointerTypeMap Map = PointerTypeAnalysis::run(*M);
  ASSERT_EQ(Map.size(), 4u);

  Type *I8Ptr = TypedPointerType::get(Type::getInt8Ty(Context), 0);
  Type *I32Ptr = TypedPointerType::get(Type::getInt32Ty(Context), 0);
  Type *I64Ptr = TypedPointerType::get(Type::getInt64Ty(Context), 0);
  Type *FnTy = FunctionType::get(Type::getInt64Ty(Context), {I8Ptr}, false);

  EXPECT_THAT(Map,
              Contains(Pair(IsA<Function>(), TypedPointerType::get(FnTy, 0))));
  EXPECT_THAT(Map, Contains(Pair(IsA<Argument>(), I8Ptr)));
  EXPECT_THAT(Map, Contains(Pair(IsA<BitCastInst>(), I64Ptr)));
  EXPECT_THAT(Map, Contains(Pair(IsA<BitCastInst>(), I32Ptr)));
}
