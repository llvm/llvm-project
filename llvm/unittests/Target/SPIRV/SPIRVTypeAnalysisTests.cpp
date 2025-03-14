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
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %test = alloca [5 x i32]
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("test")),
            TypedPointerType::get(ArrayType::get(IntTy, 5), 0));
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
      TI.getType(getValue("test")),
      TypedPointerType::get(ArrayType::get(ArrayType::get(IntTy, 10), 5), 0));
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
  EXPECT_EQ(TI.getType(getValue("test")),
            TypedPointerType::get(
                ArrayType::get(VectorType::get(IntTy, 4, false), 5), 0));
}

TEST_F(SPIRVTypeAnalysisTest, AllocaLoad) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %test = alloca [5 x <4 x i32>]
      %v = load <4 x i32>, ptr %test
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  auto VT = VectorType::get(IntegerType::get(M->getContext(), 32), 4,
                            /* scalable= */ false);
  EXPECT_EQ(TI.getType(getValue("test")),
            TypedPointerType::get(ArrayType::get(VT, 5), 0));
  EXPECT_EQ(TI.getType(getValue("v")), VT);
}

TEST_F(SPIRVTypeAnalysisTest, AllocaPtrIndirectDeduction) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %v = alloca ptr
      %l = load i32, ptr %v
      ret void
    }
  )";

  auto TI = runAnalysis(Assembly);
  EXPECT_EQ(TI.getType(getValue("v")),
            TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("l")), IntTy);
}

TEST_F(SPIRVTypeAnalysisTest, AllocaPtrNestedIndirectDeduction) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
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
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
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
  EXPECT_EQ(TI.getType(getValue("l1")),
            TypedPointerType::get(IntTy, /* AS= */ 0));
  EXPECT_EQ(TI.getType(getValue("l2")), IntTy);
}

TEST_F(SPIRVTypeAnalysisTest, AllocaStruct) {
  StringRef Assembly = R"(
    %st = type { i32 }

    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
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
