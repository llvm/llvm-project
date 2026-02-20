//===- SPIRVTypeAnalysisTests.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/SPIRVTypeAnalysis.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/TypedPointerType.h"
#include "llvm/Support/SourceMgr.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>

using ::testing::Contains;
using ::testing::Pair;

using namespace llvm;
using namespace llvm::SPIRV;

template <typename T> struct IsA {
  friend bool operator==(const Value *V, const IsA &) { return isa<T>(V); }
};

class SPIRVTypeAnalysisTest : public testing::Test {
protected:
  void SetUp() override {
    // Required for tests.
    FAM.registerPass([&] { return PassInstrumentationAnalysis(); });
    MAM.registerPass([&] { return PassInstrumentationAnalysis(); });

    MAM.registerPass([&] { return SPIRVTypeAnalysis(); });
  }

  void TearDown() override { M.reset(); }

  SPIRVTypeAnalysis::Result &runAnalysis(StringRef Assembly) {
    assert(M == nullptr &&
           "Calling runAnalysis multiple times is unsafe. See getAnalysis().");

    SMDiagnostic Error;
    M = parseAssemblyString(Assembly, Error, Context);
    if (!M) {
      std::cerr << Error.getMessage().str() << std::endl;
      std::cerr << "> " << Error.getLineContents().str() << std::endl;
    }
    assert(M && "Bad assembly. Bad test?");

    ModulePassManager MPM;
    MPM.run(*M, MAM);

    // Setup helper types.
    IntTy = IntegerType::get(M->getContext(), 32);
    FloatTy = Type::getFloatTy(M->getContext());

    return MAM.getResult<SPIRVTypeAnalysis>(*M);
  }

  const Value *getValue(StringRef Name) {
    assert(M != nullptr && "Has runAnalysis been called before?");

    llvm::GlobalValue *GV = M->getNamedValue(Name);
    if (GV)
      return GV;

    for (auto &F : *M) {
      for (Argument &A : F.args())
        if (A.getName() == Name)
          return &A;

      for (auto &BB : F)
        for (auto &V : BB)
          if (Name == V.getName())
            return &V;
    }
    ADD_FAILURE() << "Error: Could not locate requested variable. Bad test?";
    return nullptr;
  }

  StructType *getStructType(StringRef Name) {
    for (StructType *ST : M->getIdentifiedStructTypes()) {
      if (ST->getName() == Name)
        return ST;
    }

    ADD_FAILURE() << "Error: Could not locate requested struct type. Bad test?";
    return nullptr;
  }

protected:
  LLVMContext Context;
  FunctionAnalysisManager FAM;
  ModuleAnalysisManager MAM;
  std::unique_ptr<Module> M;

  // Helper types for writting tests.
  Type *IntTy = nullptr;
  Type *FloatTy = nullptr;
};

TEST_F(SPIRVTypeAnalysisTest, ScalarAlloca) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %test = alloca i32
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("test")),
            TypedPointerType::get(IntTy, /* AS= */ 0));
}

TEST_F(SPIRVTypeAnalysisTest, ScalarAllocaFloat) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %test = alloca float
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("test")),
            TypedPointerType::get(FloatTy, /* AS= */ 0));
}

TEST_F(SPIRVTypeAnalysisTest, AllocaArray) {
  StringRef Assembly = R"(
    define void @foo() {
      %test = alloca [5 x i32]
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("test")), ArrayType::get(IntTy, 5));
}

TEST_F(SPIRVTypeAnalysisTest, AllocaArrayArray) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %test = alloca [5 x [10 x i32]]
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(
      TI.getType(getValue("test")), ArrayType::get(ArrayType::get(IntTy, 10), 5));
}

TEST_F(SPIRVTypeAnalysisTest, AllocaVector) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %test = alloca <4 x i32>
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("test")),
            TypedPointerType::get(VectorType::get(IntTy, 4, false), 0));
}

TEST_F(SPIRVTypeAnalysisTest, AllocaArrayVector) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %test = alloca [5 x <4 x i32>]
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("test")), ArrayType::get(VectorType::get(IntTy, 4, false), 5));
}

TEST_F(SPIRVTypeAnalysisTest, AllocaLoad) {
  StringRef Assembly = R"(
    define void @foo() {
      %test = alloca [5 x <4 x i32>]
      %v = load <4 x i32>, ptr %test
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  auto VT = VectorType::get(IntegerType::get(M->getContext(), 32), 4,
                            /* scalable= */ false);
  EXPECT_EQ(TI.getType(getValue("test")), ArrayType::get(VT, 5));
  EXPECT_EQ(TI.getType(getValue("v")), VT);
}

TEST_F(SPIRVTypeAnalysisTest, AllocaPtrIndirectDeduction) {
  StringRef Assembly = R"(
    define void @foo() {
      %a = alloca ptr
      %b = load ptr, ptr %a
      %c = load i32, ptr %b
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("a")), TypedPointerType::get(TypedPointerType::get(IntTy, /* AS= */ 0), /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("b")), TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("c")), IntTy);
}

TEST_F(SPIRVTypeAnalysisTest, AllocaPtrNestedIndirectDeduction) {
  StringRef Assembly = R"(
    define void @foo() {
      %v = alloca ptr
      %l1 = load ptr, ptr %v
      %l2 = load ptr, ptr %l1
      %l3 = load ptr, ptr %l2
      %l4 = load i32, ptr %l3
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);

  Type *LoadType = IntTy;
  EXPECT_EQ(TI.getType(getValue("l4")), LoadType);
  LoadType = TypedPointerType::get(LoadType, /* AS= */ 0);
  EXPECT_EQ(TI.getType(getValue("l3")), LoadType);
  LoadType = TypedPointerType::get(LoadType, /* AS= */ 0);
  EXPECT_EQ(TI.getType(getValue("l2")), LoadType);
  LoadType = TypedPointerType::get(LoadType, /* AS= */ 0);
  EXPECT_EQ(TI.getType(getValue("l1")), LoadType);
  LoadType = TypedPointerType::get(LoadType, /* AS= */ 0);
  EXPECT_EQ(TI.getType(getValue("v")), LoadType);
}

TEST_F(SPIRVTypeAnalysisTest, AllocaLoadArrayPtr) {
  StringRef Assembly = R"(
    define void @foo() {
      ; %ptr_array = alloca [5 x *i32]
      %ptr_array = alloca [5 x ptr]
      %l1 = load ptr, ptr %ptr_array
      %l2 = load i32, ptr %l1
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);

  EXPECT_EQ(TI.getType(getValue("ptr_array")),
            ArrayType::get(TypedPointerType::get(IntTy, /* AS= */ 0), 5));
  EXPECT_EQ(TI.getType(getValue("l1")), TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("l2")), IntTy);
}

TEST_F(SPIRVTypeAnalysisTest, AllocaStruct) {
  StringRef Assembly = R"(
    define void @foo() {
      %a = alloca {i32, i32}
      %b = load i32, ptr %a
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);

  EXPECT_EQ(TI.getType(getValue("a")), TypedPointerType::get(StructType::get(IntTy, IntTy), /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("b")), IntTy);
}

TEST_F(SPIRVTypeAnalysisTest, AllocaStructPartiallyOpaque) {
  StringRef Assembly = R"(
    define void @foo() {
      %a = alloca {ptr, i32}
      %b = load ptr, ptr %a
      %c = load i32, ptr %b
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);

  EXPECT_EQ(TI.getType(getValue("a")), TypedPointerType::get(StructType::get(TypedPointerType::get(IntTy, /* AS= */ 0), IntTy), /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("b")), TypedPointerType::get(IntTy, /* AS= */ 0));
}

TEST_F(SPIRVTypeAnalysisTest, AllocaNamedStruct) {
  StringRef Assembly = R"(
    %st = type { i32 }

    define void @foo() {
      %var = alloca %st
      %l = load %st, ptr %var
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  auto ST = getStructType("st");
  EXPECT_EQ(TI.getType(getValue("var")),
            TypedPointerType::get(ST, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("l")), ST);
}

TEST_F(SPIRVTypeAnalysisTest, AllocaStructDeducedFromLoad) {
  StringRef Assembly = R"(
    %st = type { i32 }

    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %var = alloca ptr
      %l1 = load ptr, ptr %var
      %l2 = load %st, ptr %l1
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);

  Type *LoadType = getStructType("st");
  EXPECT_EQ(TI.getType(getValue("l2")), LoadType);
  LoadType = TypedPointerType::get(LoadType, /* AS= */ 0);
  EXPECT_EQ(TI.getType(getValue("l1")), LoadType);
  LoadType = TypedPointerType::get(LoadType, /* AS= */ 0);
  EXPECT_EQ(TI.getType(getValue("var")), LoadType);
}

TEST_F(SPIRVTypeAnalysisTest, StructMemberDirectLoad) {
  StringRef Assembly = R"(
    %st = type { i32 }

    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %var = alloca %st
      %l2 = load i32, ptr %var
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);

  Type *ST = getStructType("st");
  EXPECT_EQ(TI.getType(getValue("l2")), IntTy);
  EXPECT_EQ(TI.getType(getValue("var")),
            TypedPointerType::get(ST, /* AS= */ 0));
}

TEST_F(SPIRVTypeAnalysisTest, StructMemberConflictingDeductionFromLoadA) {
  StringRef Assembly = R"(
    %st = type { i32 }

    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      ; %var = alloca *%st
      %var = alloca ptr

      %l1 = load ptr, ptr %var

      %l2 = load %st, ptr %l1
      %l3 = load i32, ptr %l1

      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);

  Type *ST = getStructType("st");
  EXPECT_EQ(TI.getType(getValue("l3")), IntTy);
  EXPECT_EQ(TI.getType(getValue("l2")), ST);
  EXPECT_EQ(TI.getType(getValue("l1")), TypedPointerType::get(ST, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("var")),
            TypedPointerType::get(TypedPointerType::get(ST, /* AS= */ 0), 0));
}

TEST_F(SPIRVTypeAnalysisTest, StructMemberConflictingDeductionFromLoadB) {
  StringRef Assembly = R"(
    %st = type { i32 }

    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      ; %var = alloca *%st
      %var = alloca ptr

      %l1 = load ptr, ptr %var

      %l3 = load i32, ptr %l1
      %l2 = load %st, ptr %l1

      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);

  Type *ST = getStructType("st");
  EXPECT_EQ(TI.getType(getValue("l3")), IntTy);
  EXPECT_EQ(TI.getType(getValue("l2")), ST);
  EXPECT_EQ(TI.getType(getValue("l1")), TypedPointerType::get(ST, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("var")),
            TypedPointerType::get(TypedPointerType::get(ST, /* AS= */ 0), 0));
}

TEST_F(SPIRVTypeAnalysisTest, StructMemberConflictingTree) {
  StringRef Assembly = R"(
    %st = type { i32 }

    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      ; %var = alloca *%st
      %var = alloca ptr

      %l_0 = load ptr, ptr %var
      %l_1 = load ptr, ptr %var

      %l_00 = load %st, ptr %l_0
      %l_01 = load %st, ptr %l_1

      %l_10 = load i32, ptr %l_0
      %l_11 = load i32, ptr %l_1

      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);

  Type *ST = getStructType("st");
  EXPECT_EQ(TI.getType(getValue("l_10")), IntTy);
  EXPECT_EQ(TI.getType(getValue("l_11")), IntTy);

  EXPECT_EQ(TI.getType(getValue("l_00")), ST);
  EXPECT_EQ(TI.getType(getValue("l_01")), ST);

  EXPECT_EQ(TI.getType(getValue("l_0")),
            TypedPointerType::get(ST, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("l_1")),
            TypedPointerType::get(ST, /* AS= */ 0));

  EXPECT_EQ(TI.getType(getValue("var")),
            TypedPointerType::get(TypedPointerType::get(ST, /* AS= */ 0), 0));
}

TEST_F(SPIRVTypeAnalysisTest, ArrayElementConflict) {
  StringRef Assembly = R"(
    %st = type { i32 }

    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      ; %var = alloca *[5 x %st]
      %var = alloca ptr

      %l1 = load ptr, ptr %var

      %l2 = load %st, ptr %l1
      %l3 = load i32, ptr %l1
      %l4 = load [5 x %st], ptr %l1

      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);

  Type *ST = getStructType("st");
  EXPECT_EQ(TI.getType(getValue("l4")), ArrayType::get(ST, 5));
  EXPECT_EQ(TI.getType(getValue("l3")), IntTy);
  EXPECT_EQ(TI.getType(getValue("l2")), ST);
  EXPECT_EQ(TI.getType(getValue("l1")),
            TypedPointerType::get(ArrayType::get(ST, 5), /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("var")),
            TypedPointerType::get(
                TypedPointerType::get(ArrayType::get(ST, 5), /* AS= */ 0), 0));
}

TEST_F(SPIRVTypeAnalysisTest, VectorElementConflict) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      ; %var = alloca *<4 x i32>
      %var = alloca ptr

      %l1 = load ptr, ptr %var

      %l3 = load i32, ptr %l1
      %l4 = load <4 x i32>, ptr %l1

      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);

  auto *VT = VectorType::get(IntTy, 4, /* scalable= */ false);

  EXPECT_EQ(TI.getType(getValue("l4")), VT);
  EXPECT_EQ(TI.getType(getValue("l3")), IntTy);
  EXPECT_EQ(TI.getType(getValue("l1")), TypedPointerType::get(VT, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("var")),
            TypedPointerType::get(TypedPointerType::get(VT, /* AS= */ 0), 0));
}

TEST_F(SPIRVTypeAnalysisTest, MissingInformationOnLoad) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %var = alloca ptr
      %l1 = load ptr, ptr %var
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);

  EXPECT_EQ(TI.getType(getValue("l1")),
            PointerType::get(M->getContext(), /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("var")),
            PointerType::get(M->getContext(), /* AS= */ 0));
}

TEST_F(SPIRVTypeAnalysisTest, MissingInformationAlloca) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %var = alloca ptr
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);

  EXPECT_EQ(TI.getType(getValue("var")),
            PointerType::get(M->getContext(), /* AS= */ 0));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFromGep) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %var = alloca ptr
      %ptr = getelementptr [5 x i32], ptr %var, i64 0, i64 0
      %val = load i32, ptr %ptr
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  Type *AT = ArrayType::get(IntTy, 5);
  EXPECT_EQ(TI.getType(getValue("var")),
            TypedPointerType::get(AT, /* AS= */ 0));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFromGepOpaqueBaseType) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      ; *[5 x *[2 x i32] ]
      ; *[5 x ptr ]
      ;   ptr
      %var = alloca ptr
      %ptr1 = getelementptr ptr, ptr %var, i64 0
      %ptr2 = getelementptr ptr, ptr %ptr1, i64 0
      %val = load i32, ptr %ptr2
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  Type *T = IntTy;
  EXPECT_EQ(TI.getType(getValue("val")), T);
  T = TypedPointerType::get(T, /* AS= */ 0);
  EXPECT_EQ(TI.getType(getValue("ptr2")), T);
  T = TypedPointerType::get(T, /* AS= */ 0);
  EXPECT_EQ(TI.getType(getValue("ptr1")), T);
  T = TypedPointerType::get(T, /* AS= */ 0);
  EXPECT_EQ(TI.getType(getValue("var")), T);
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFromGepPartialOpaqueBaseType) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      ; *[5 x *[2 x i32] ]
      ; *[5 x ptr ]
      ;   ptr
      %var = alloca ptr
      %ptr1 = getelementptr [5 x ptr], ptr %var, i64 0, i64 0
      %ptr2 = getelementptr ptr, ptr %ptr1, i64 0
      %val = load i32, ptr %ptr2
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  Type *T = IntTy;
  EXPECT_EQ(TI.getType(getValue("val")), T);
  T = TypedPointerType::get(T, /* AS= */ 0);
  EXPECT_EQ(TI.getType(getValue("ptr2")), T);
  T = TypedPointerType::get(T, /* AS= */ 0);
  EXPECT_EQ(TI.getType(getValue("ptr1")), T);
  T = TypedPointerType::get(ArrayType::get(T, 5), 0);
  EXPECT_EQ(TI.getType(getValue("var")), T);
}

TEST_F(SPIRVTypeAnalysisTest, DeduceParamFromLoad) {
  StringRef Assembly = R"(
    define void @foo(ptr %input) {
      %a = load i32, ptr %input
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("input")), TypedPointerType::get(IntTy, 0));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFromStore) {
  StringRef Assembly = R"(
    define void @foo(ptr %input) {
      store i32 0, ptr %input
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("input")), TypedPointerType::get(IntTy, 0));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFromStoreStructConflict) {
  StringRef Assembly = R"(
    %st = type { i32 }

    define void @foo(ptr %a, ptr %b) {
      %s = load %st, ptr %a
      store i32 0, ptr %b
      store %st %s, ptr %b
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  auto ST = getStructType("st");
  EXPECT_EQ(TI.getType(getValue("a")), TypedPointerType::get(ST, 0));
  EXPECT_EQ(TI.getType(getValue("b")), TypedPointerType::get(ST, 0));
  EXPECT_EQ(TI.getType(getValue("s")), ST);
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFromStoreStructInline) {
  StringRef Assembly = R"(
    %st = type { i32 }

    define void @foo(ptr %a) {
      store i32 0, ptr %a
      store %st { i32 0 }, ptr %a
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  auto ST = getStructType("st");
  EXPECT_EQ(TI.getType(getValue("a")), TypedPointerType::get(ST, 0));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFromStoreArray) {
  StringRef Assembly = R"(
    define void @foo(ptr %a) {
      store i32 0, ptr %a
      store [2 x i32] [ i32 0, i32 1 ], ptr %a
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("a")),
            TypedPointerType::get(ArrayType::get(IntTy, 2), 0));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFromCall) {
  StringRef Assembly = R"(
    define ptr @foo(ptr %par) {
      ret ptr %par
    }

    define void @bar() {
      %var = alloca i32
      %res = call ptr @foo(ptr %var)
      store i32 0, ptr %res
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("var")), TypedPointerType::get(IntTy, 0));
  EXPECT_EQ(TI.getType(getValue("res")), TypedPointerType::get(IntTy, 0));
  EXPECT_EQ(TI.getType(getValue("par")), TypedPointerType::get(IntTy, 0));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFromCallLackInfo) {
  StringRef Assembly = R"(
    define ptr @foo(ptr %par) {
      ret ptr %par
    }

    define void @bar() {
      %var = alloca ptr
      %res = call ptr @foo(ptr %var)
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("var")),
            PointerType::get(M->getContext(), /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("res")),
            PointerType::get(M->getContext(), /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("par")),
            PointerType::get(M->getContext(), /* AS= */ 0));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFromCallPartial) {
  StringRef Assembly = R"(
    define ptr @foo(ptr %fpar1, ptr %fpar2) {
      ret ptr %fpar2
    }

    define void @bar(ptr %bpar1, ptr %bpar2) {
      %res = call ptr @foo(ptr %bpar1, ptr %bpar2)
      store i32 0, ptr %res
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("fpar1")),
            PointerType::get(M->getContext(), /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("bpar1")),
            PointerType::get(M->getContext(), /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("res")), TypedPointerType::get(IntTy, 0));
  EXPECT_EQ(TI.getType(getValue("fpar2")), TypedPointerType::get(IntTy, 0));
  EXPECT_EQ(TI.getType(getValue("bpar2")), TypedPointerType::get(IntTy, 0));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFromCallPartialWithLoad) {
  StringRef Assembly = R"(
    define ptr @foo(ptr %fpar1, ptr %fpar2) {
      %val = load i32, ptr %fpar2
      ret ptr %fpar2
    }

    define void @bar(ptr %bpar1, ptr %bpar2) {
      %res = call ptr @foo(ptr %bpar1, ptr %bpar2)
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("val")), IntTy);
  EXPECT_EQ(TI.getType(getValue("fpar1")),
            PointerType::get(M->getContext(), /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("bpar1")),
            PointerType::get(M->getContext(), /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("res")), TypedPointerType::get(IntTy, 0));
  EXPECT_EQ(TI.getType(getValue("fpar2")), TypedPointerType::get(IntTy, 0));
  EXPECT_EQ(TI.getType(getValue("bpar2")), TypedPointerType::get(IntTy, 0));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceRecursiveFromStore) {
  StringRef Assembly = R"(
    define ptr @foo(ptr %par) {
      store i32 0, ptr %par
      %tmp = call ptr @foo(ptr %par)
      ret ptr %par
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("tmp")), TypedPointerType::get(IntTy, 0));
  EXPECT_EQ(TI.getType(getValue("par")), TypedPointerType::get(IntTy, 0));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceRecursiveFromLoad) {
  StringRef Assembly = R"(
    define ptr @foo(ptr %par) {
      %val = load i32, ptr %par
      %tmp = call ptr @foo(ptr %par)
      ret ptr %par
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("val")), IntTy);
  EXPECT_EQ(TI.getType(getValue("tmp")), TypedPointerType::get(IntTy, 0));
  EXPECT_EQ(TI.getType(getValue("par")), TypedPointerType::get(IntTy, 0));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceRecursiveFromLoadingReturn) {
  StringRef Assembly = R"(
    define ptr @foo(ptr %par) {
      %tmp = call ptr @foo(ptr %par)
      %val = load i32, ptr %tmp
      ret ptr %par
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("val")), IntTy);
  EXPECT_EQ(TI.getType(getValue("tmp")), TypedPointerType::get(IntTy, 0));
  EXPECT_EQ(TI.getType(getValue("par")), TypedPointerType::get(IntTy, 0));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceGlobalValue) {
  StringRef Assembly = R"(
    @global = external addrspace(1) global ptr addrspace(1)

    define i32 @foo() {
      %val = load i32, ptr addrspace(1) @global
      ret i32 %val
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("val")), IntTy);
  EXPECT_EQ(TI.getType(getValue("global")), TypedPointerType::get(IntTy, 1));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceGlobalValueIndirect) {
  StringRef Assembly = R"(
    @gptr = external addrspace(1) global ptr addrspace(1)
    @gptrptr = addrspace(1) global ptr addrspace(1) @gptr

    define i32 @foo() {
      %ptr = load ptr addrspace(1), ptr addrspace(1) @gptrptr
      %val = load i32, ptr addrspace(1) %ptr
      ret i32 %val
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("val")), IntTy);
  EXPECT_EQ(TI.getType(getValue("gptr")), TypedPointerType::get(IntTy, 1));
  EXPECT_EQ(TI.getType(getValue("gptrptr")), TypedPointerType::get(TypedPointerType::get(IntTy, 1), 1));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceGlobalValueIndirectWithoutInstruction) {
  StringRef Assembly = R"(
    @gi32 = addrspace(1) global i32 0
    @gptr = addrspace(1) global ptr addrspace(1) @gi32
    @gptrptr = addrspace(1) global ptr addrspace(1) @gptr

    define i32 @foo() {
      ret i32 0
    }
  )";

  auto TI = runAnalysis(Assembly);
  Type *T = TypedPointerType::get(IntTy, 1);
  EXPECT_EQ(TI.getType(getValue("gi32")), T);
  T = TypedPointerType::get(T, 1);
  EXPECT_EQ(TI.getType(getValue("gptr")), T);
  T = TypedPointerType::get(T, 1);
  EXPECT_EQ(TI.getType(getValue("gptrptr")), T);
}

TEST_F(SPIRVTypeAnalysisTest, DeduceGlobalValueFromCall) {
  StringRef Assembly = R"(
    @gptr = external addrspace(1) global ptr addrspace(1)

    define i32 @foo(ptr addrspace(1) %arg) {
      %foo_tmp = load i32, ptr addrspace(1) %arg
      ret i32 %foo_tmp
    }

    define i32 @bar() {
      %bar_tmp = call i32 @foo(ptr addrspace(1) @gptr)
      ret i32 %bar_tmp
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("foo_tmp")), IntTy);
  EXPECT_EQ(TI.getType(getValue("bar_tmp")), IntTy);
  EXPECT_EQ(TI.getType(getValue("gptr")), TypedPointerType::get(IntTy, 1));
  EXPECT_EQ(TI.getType(getValue("arg")), TypedPointerType::get(IntTy, 1));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceValueFromBitcast) {
  StringRef Assembly = R"(
    define void @foo(ptr %ptr) {
      %val = load i32, ptr %ptr
      %dst = bitcast i32 %val to float
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("val")), IntTy);
  EXPECT_EQ(TI.getType(getValue("dst")), FloatTy);
  EXPECT_EQ(TI.getType(getValue("ptr")), TypedPointerType::get(IntTy, 0));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFromIntrinsicType) {
  StringRef Assembly = R"(
    define void @foo(<4 x float> noundef %vec) {
      %len = call float @llvm.spv.length.f32(<4 x float> %vec)
      ret void
    }

    declare half @llvm.spv.length.f32(<4 x float>)
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("vec")), VectorType::get(FloatTy, 4, false));
  EXPECT_EQ(TI.getType(getValue("len")), FloatTy);
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFromCompareXchg) {
  StringRef Assembly = R"(
    define void @foo(ptr %ptr, i32 %comparator) {
      %res = cmpxchg ptr %ptr, i32 %comparator, i32 123 seq_cst acquire, align 4
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("ptr")), TypedPointerType::get(IntTy, 0));
  EXPECT_EQ(TI.getType(getValue("comparator")), IntTy);
  EXPECT_EQ(TI.getType(getValue("res")), StructType::get(IntTy, IntegerType::get(M->getContext(), 1)));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFromCompareXchgNoInformation) {
  StringRef Assembly = R"(
    define void @foo(ptr %ptr, ptr %comparator) {
      %new_value = load ptr, ptr null
      %res = cmpxchg ptr %ptr, ptr %comparator, ptr %new_value seq_cst acquire, align 4
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  Type *PT = PointerType::get(M->getContext(), /* AS= */ 0);
  EXPECT_EQ(TI.getType(getValue("ptr")), PT);
  EXPECT_EQ(TI.getType(getValue("comparator")), PT);
  EXPECT_EQ(TI.getType(getValue("res")), StructType::get(PT, IntegerType::get(M->getContext(), 1)));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFromCompareXchgLateInfo) {
  StringRef Assembly = R"(
    define void @foo(ptr %ptr, ptr %comparator) {
      %new_value = load ptr, ptr null
      %res = cmpxchg ptr %ptr, ptr %comparator, ptr %new_value seq_cst acquire, align 4
      %tmp = load ptr, ptr %ptr
      %val = load i32, ptr %tmp
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("ptr")), TypedPointerType::get(TypedPointerType::get(IntTy, 0), 0));
  EXPECT_EQ(TI.getType(getValue("comparator")), TypedPointerType::get(TypedPointerType::get(IntTy, 0), 0));
  EXPECT_EQ(TI.getType(getValue("new_value")), TypedPointerType::get(TypedPointerType::get(IntTy, 0), 0));
  EXPECT_EQ(TI.getType(getValue("tmp")), TypedPointerType::get(IntTy, 0));
  EXPECT_EQ(TI.getType(getValue("val")), IntTy);
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFromExtractValue) {
  StringRef Assembly = R"(
    define void @foo(ptr %ptr, ptr %comparator) {
      %new_value = load ptr, ptr null
      %res = cmpxchg ptr %ptr, ptr %comparator, ptr %new_value seq_cst acquire, align 4
      %a = extractvalue { ptr, i1 } %res, 0
      %b = load ptr, ptr %a
      %c = load i32, ptr %b
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("ptr")), TypedPointerType::get(TypedPointerType::get(TypedPointerType::get(IntTy, 0), 0), 0));
  EXPECT_EQ(TI.getType(getValue("comparator")), TypedPointerType::get(TypedPointerType::get(IntTy, 0), 0));
  EXPECT_EQ(TI.getType(getValue("new_value")), TypedPointerType::get(TypedPointerType::get(IntTy, 0), 0));
  EXPECT_EQ(TI.getType(getValue("a")), TypedPointerType::get(TypedPointerType::get(IntTy, 0), 0));
  EXPECT_EQ(TI.getType(getValue("b")), TypedPointerType::get(IntTy, 0));
  EXPECT_EQ(TI.getType(getValue("c")), IntTy);
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFromInsertValue) {
  StringRef Assembly = R"(
    define void @foo({ i32, i32 } %a) {
      %b = insertvalue { i32, i32 } %a, i32 0, 0
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("a")), StructType::get(IntTy, IntTy));
  EXPECT_EQ(TI.getType(getValue("b")), StructType::get(IntTy, IntTy));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFromASCastNoInfo) {
  StringRef Assembly = R"(
    define void @foo(ptr %ptr) {
      %out = addrspacecast ptr %ptr to ptr addrspace(10)
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("ptr")), PointerType::get(M->getContext(), /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("out")), PointerType::get(M->getContext(), /* AS= */ 10));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFromASCastFromAbove) {
  StringRef Assembly = R"(
    define void @foo(ptr %ptr) {
      %val = load i32, ptr %ptr
      %out = addrspacecast ptr %ptr to ptr addrspace(10)
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("ptr")), TypedPointerType::get(IntTy, 0));
  EXPECT_EQ(TI.getType(getValue("val")), IntTy);
  EXPECT_EQ(TI.getType(getValue("out")), TypedPointerType::get(IntTy, 10));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFromASCastFromBelow) {
  StringRef Assembly = R"(
    define void @foo(ptr %ptr) {
      %out = addrspacecast ptr %ptr to ptr addrspace(10)
      %val = load i32, ptr addrspace(10) %out
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("ptr")), TypedPointerType::get(IntTy, 0));
  EXPECT_EQ(TI.getType(getValue("val")), IntTy);
  EXPECT_EQ(TI.getType(getValue("out")), TypedPointerType::get(IntTy, 10));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceInsertElementVec) {
  StringRef Assembly = R"(
    define void @foo(<2 x i32> %a) {
      %vec = insertelement <2 x i32> %a, i32 1, i32 0
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("vec")), VectorType::get(IntTy, 2, false));
  EXPECT_EQ(TI.getType(getValue("a")), VectorType::get(IntTy, 2, false));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceExtractElementInst) {
  StringRef Assembly = R"(
    define void @foo(<2 x i32> %a) {
      %scal = extractelement <2 x i32> %a, i32 0
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("a")), VectorType::get(IntTy, 2, false));
  EXPECT_EQ(TI.getType(getValue("scal")), IntTy);
}

TEST_F(SPIRVTypeAnalysisTest, DeduceSelectTypeScalar) {
  StringRef Assembly = R"(
    define void @foo(i32 %a, i32 %b) {
      %c = select i1 0, i32 %a, i32 %b
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("a")), IntTy);
  EXPECT_EQ(TI.getType(getValue("b")), IntTy);
  EXPECT_EQ(TI.getType(getValue("c")), IntTy);
}

TEST_F(SPIRVTypeAnalysisTest, DeduceTypeFromAboveLeft) {
  StringRef Assembly = R"(
    define void @foo(ptr %a, ptr %b) {
      %d = load i32, ptr %a
      %c = select i1 0, ptr %a, ptr %b
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("d")), IntTy);
  EXPECT_EQ(TI.getType(getValue("a")), TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("b")), TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("c")), TypedPointerType::get(IntTy, /* AS= */ 0));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceTypeFromAboveRight) {
  StringRef Assembly = R"(
    define void @foo(ptr %a, ptr %b) {
      %d = load i32, ptr %b
      %c = select i1 0, ptr %a, ptr %b
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("d")), IntTy);
  EXPECT_EQ(TI.getType(getValue("a")), TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("b")), TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("c")), TypedPointerType::get(IntTy, /* AS= */ 0));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceTypeFromBelow) {
  StringRef Assembly = R"(
    define void @foo(ptr %a, ptr %b) {
      %c = select i1 0, ptr %a, ptr %b
      %d = load i32, ptr %c
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("d")), IntTy);
  EXPECT_EQ(TI.getType(getValue("a")), TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("b")), TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("c")), TypedPointerType::get(IntTy, /* AS= */ 0));
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFromBinaryOp) {
  StringRef Assembly = R"(
    define void @foo(i32 %a, i32 %b) {
      %add = add i32 %a, %b
      %sub = sub i32 %a, %b
      %mul = mul i32 %a, %b
      %sdiv = sdiv i32 %a, %b
      %shl = shl i32 %a, %b
      %or = or i32 %a, %b
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("a")), IntTy);
  EXPECT_EQ(TI.getType(getValue("b")), IntTy);
  EXPECT_EQ(TI.getType(getValue("add")), IntTy);
  EXPECT_EQ(TI.getType(getValue("sub")), IntTy);
  EXPECT_EQ(TI.getType(getValue("mul")), IntTy);
  EXPECT_EQ(TI.getType(getValue("sdiv")), IntTy);
  EXPECT_EQ(TI.getType(getValue("shl")), IntTy);
  EXPECT_EQ(TI.getType(getValue("or")), IntTy);
}

TEST_F(SPIRVTypeAnalysisTest, DeducePhiNoInfo) {
  StringRef Assembly = R"(
    define void @foo(ptr %a, ptr %b) {
      br i1 0, label %lhs, label %rhs
    lhs:
      br label %merge;
    rhs:
      br label %merge;
    merge:
      %c = phi ptr [ %a, %lhs ], [ %b, %rhs ]
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("a")), PointerType::get(M->getContext(), /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("b")), PointerType::get(M->getContext(), /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("c")), PointerType::get(M->getContext(), /* AS= */ 0));
}

TEST_F(SPIRVTypeAnalysisTest, DeducePhiScalar) {
  StringRef Assembly = R"(
    define void @foo(i32 %a, i32 %b) {
      br i1 0, label %lhs, label %rhs
    lhs:
      br label %merge;
    rhs:
      br label %merge;
    merge:
      %c = phi i32 [ %a, %lhs ], [ %b, %rhs ]
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("a")), IntTy);
  EXPECT_EQ(TI.getType(getValue("b")), IntTy);
  EXPECT_EQ(TI.getType(getValue("c")), IntTy);
}

TEST_F(SPIRVTypeAnalysisTest, DeducePhiAbove) {
  StringRef Assembly = R"(
    define void @foo(ptr %a, ptr %b) {
      br i1 0, label %lhs, label %rhs
    lhs:
      %val = load i32, ptr %a
      br label %merge;
    rhs:
      br label %merge;
    merge:
      %c = phi ptr [ %a, %lhs ], [ %b, %rhs ]
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("a")), TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("b")), TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("c")), TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("val")), IntTy);
}

TEST_F(SPIRVTypeAnalysisTest, DeducePhiAboveRhs) {
  StringRef Assembly = R"(
    define void @foo(ptr %a, ptr %b) {
      br i1 0, label %lhs, label %rhs
    lhs:
      br label %merge;
    rhs:
      %val = load i32, ptr %b
      br label %merge;
    merge:
      %c = phi ptr [ %a, %lhs ], [ %b, %rhs ]
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("a")), TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("b")), TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("c")), TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("val")), IntTy);
}

TEST_F(SPIRVTypeAnalysisTest, DeducePhiBelow) {
  StringRef Assembly = R"(
    define void @foo(ptr %a, ptr %b) {
      br i1 0, label %lhs, label %rhs
    lhs:
      br label %merge;
    rhs:
      br label %merge;
    merge:
      %c = phi ptr [ %a, %lhs ], [ %b, %rhs ]
      %val = load i32, ptr %c
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("a")), TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("b")), TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("c")), TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("val")), IntTy);
}

TEST_F(SPIRVTypeAnalysisTest, DeducePhiBelowPtrCastConflict) {
  StringRef Assembly = R"(
    %st = type { i32 }

    define void @foo(ptr %a, ptr %b) {
      br i1 0, label %lhs, label %rhs
    lhs:
      %val1 = load %st, ptr %a
      br label %merge;
    rhs:
      br label %merge;
    merge:
      %c = phi ptr [ %a, %lhs ], [ %b, %rhs ]
      %val2 = load i32, ptr %c
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  auto ST = getStructType("st");
  EXPECT_EQ(TI.getType(getValue("a")), TypedPointerType::get(ST, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("b")), TypedPointerType::get(ST, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("c")), TypedPointerType::get(ST, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("val1")), ST);
  EXPECT_EQ(TI.getType(getValue("val2")), IntTy);
}

TEST_F(SPIRVTypeAnalysisTest, DeducePhiBelowPtrCastConflictRHS) {
  StringRef Assembly = R"(
    %st = type { i32 }

    define void @foo(ptr %a, ptr %b) {
      br i1 0, label %lhs, label %rhs
    lhs:
      br label %merge;
    rhs:
      %val1 = load %st, ptr %b
      br label %merge;
    merge:
      %c = phi ptr [ %a, %lhs ], [ %b, %rhs ]
      %val2 = load i32, ptr %c
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  auto ST = getStructType("st");
  EXPECT_EQ(TI.getType(getValue("a")), TypedPointerType::get(ST, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("b")), TypedPointerType::get(ST, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("c")), TypedPointerType::get(ST, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("val1")), ST);
  EXPECT_EQ(TI.getType(getValue("val2")), IntTy);
}

TEST_F(SPIRVTypeAnalysisTest, DeducePhiBelowPtrCastConflictBelow) {
  StringRef Assembly = R"(
    %st = type { i32 }

    define void @foo(ptr %a, ptr %b) {
      %val1 = load i32, ptr %a
      br i1 0, label %lhs, label %rhs
    lhs:
      br label %merge;
    rhs:
      br label %merge;
    merge:
      %c = phi ptr [ %a, %lhs ], [ %b, %rhs ]
      %val2 = load %st, ptr %c
     ret void
   }
 )";

   auto TI = runAnalysis(Assembly);
  auto ST = getStructType("st");
  EXPECT_EQ(TI.getType(getValue("a")), TypedPointerType::get(ST, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("b")), TypedPointerType::get(ST, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("c")), TypedPointerType::get(ST, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("val1")), IntTy);
  EXPECT_EQ(TI.getType(getValue("val2")), ST);
}

TEST_F(SPIRVTypeAnalysisTest, DeduceAtomicRMW) {
  StringRef Assembly = R"(
    define void @foo(ptr %a) {
      %old = atomicrmw xchg ptr %a, i32 0 acq_rel, align 4
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("a")), TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("old")), IntTy);
}

TEST_F(SPIRVTypeAnalysisTest, DeduceAtomicRMWAddressSpace) {
  StringRef Assembly = R"(
    define void @foo(ptr addrspace(2) %a) {
      %old = atomicrmw xchg ptr addrspace(2) %a, i32 0 acq_rel, align 4
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("a")), TypedPointerType::get(IntTy, /* AS= */ 2));
  EXPECT_EQ(TI.getType(getValue("old")), IntTy);
}

TEST_F(SPIRVTypeAnalysisTest, DeduceSupportsFence) {
  StringRef Assembly = R"(
    define void @foo() {
      fence acquire
      ret void
    }
  )";

  runAnalysis(Assembly);
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFreeze) {
  StringRef Assembly = R"(
    define void @foo(ptr %a) {
      %b = freeze ptr %a
      %c = load i32, ptr %b
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("a")), TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("b")), TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("c")), IntTy);
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFreezeAddressSpace) {
  StringRef Assembly = R"(
    define void @foo(ptr addrspace(2) %a) {
      %b = freeze ptr addrspace(2) %a
      %c = load i32, ptr addrspace(2) %b
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("a")), TypedPointerType::get(IntTy, /* AS= */ 2));
  EXPECT_EQ(TI.getType(getValue("b")), TypedPointerType::get(IntTy, /* AS= */ 2));
  EXPECT_EQ(TI.getType(getValue("c")), IntTy);
}

TEST_F(SPIRVTypeAnalysisTest, DeduceFreezeTopDown) {
  StringRef Assembly = R"(
    define void @foo(ptr addrspace(2) %a) {
      %b = load i32, ptr addrspace(2) %a
      %c = freeze ptr addrspace(2) %a
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("a")), TypedPointerType::get(IntTy, /* AS= */ 2));
  EXPECT_EQ(TI.getType(getValue("b")), IntTy);
  EXPECT_EQ(TI.getType(getValue("c")), TypedPointerType::get(IntTy, /* AS= */ 2));
}
