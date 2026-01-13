//===- llvm/unittests/Target/DirectX/ResourceBindingAnalysisTests.cpp -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/DXILABI.h"
#include "gtest/gtest.h"
#include <cstdint>

using namespace llvm;
using namespace llvm::dxil;

namespace {
class ResourceBindingAnalysisTest : public testing::Test {
protected:
  PassBuilder *PB;
  ModuleAnalysisManager *MAM;
  LLVMContext *Context;

  virtual void SetUp() {
    PB = new PassBuilder();
    MAM = new ModuleAnalysisManager();
    Context = new LLVMContext();
    PB->registerModuleAnalyses(*MAM);
    MAM->registerPass([&] { return DXILResourceBindingAnalysis(); });
  }

  std::unique_ptr<Module> parseAsm(StringRef Asm) {
    SMDiagnostic Error;
    std::unique_ptr<Module> M = parseAssemblyString(Asm, Error, *Context);
    EXPECT_TRUE(M) << "Bad assembly?: " << Error.getMessage();
    return M;
  }

  virtual void TearDown() {
    delete PB;
    delete MAM;
    delete Context;
  }
};

TEST_F(ResourceBindingAnalysisTest, TestTrivialCase) {
  // RWBuffer<float> Buf : register(u5);
  StringRef Assembly = R"(
define void @main() {
entry:
  %handle = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 5, i32 1, i32 0, ptr null)
  ret void
}
  )";

  auto M = parseAsm(Assembly);

  DXILResourceBindingInfo &DRBI =
      MAM->getResult<DXILResourceBindingAnalysis>(*M);

  EXPECT_EQ(false, DRBI.hasImplicitBinding());
  EXPECT_EQ(false, DRBI.hasOverlappingBinding());
}

TEST_F(ResourceBindingAnalysisTest, TestOverlap) {
  // StructuredBuffer<float> A[]  : register(t0, space2);
  // StructuredBuffer<float> B    : register(t4, space2); /* overlapping */
  StringRef Assembly = R"(
define void @main() {
entry:
  %handleA = call target("dx.RawBuffer", float, 0, 0) @llvm.dx.resource.handlefrombinding(i32 2, i32 0, i32 -1, i32 100, ptr null)
  %handleB = call target("dx.RawBuffer", float, 0, 0) @llvm.dx.resource.handlefrombinding(i32 2, i32 4, i32 1, i32 0, ptr null)
  ret void
}
  )";

  auto M = parseAsm(Assembly);

  DXILResourceBindingInfo &DRBI =
      MAM->getResult<DXILResourceBindingAnalysis>(*M);

  EXPECT_EQ(false, DRBI.hasImplicitBinding());
  EXPECT_EQ(true, DRBI.hasOverlappingBinding());
}

TEST_F(ResourceBindingAnalysisTest, TestExactOverlap) {
  // StructuredBuffer<float> A  : register(t5);
  // StructuredBuffer<float> B  : register(t5);
  StringRef Assembly = R"(
@A.str = private unnamed_addr constant [2 x i8] c"A\00", align 1
@B.str = private unnamed_addr constant [2 x i8] c"B\00", align 1
define void @main() {
entry:
  %handleA = call target("dx.RawBuffer", float, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 5, i32 1, i32 0, ptr  @A.str)
  %handleB = call target("dx.RawBuffer", float, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 5, i32 1, i32 0, ptr  @B.str)
  ret void
}
  )";

  auto M = parseAsm(Assembly);

  DXILResourceBindingInfo &DRBI =
      MAM->getResult<DXILResourceBindingAnalysis>(*M);

  EXPECT_EQ(false, DRBI.hasImplicitBinding());
  EXPECT_EQ(true, DRBI.hasOverlappingBinding());
}

TEST_F(ResourceBindingAnalysisTest, TestImplicitFlag) {
  // RWBuffer<float> A : register(u5, space100);
  // RWBuffer<float> B;
  StringRef Assembly = R"(
define void @main() {
entry:
  %handleA = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding(i32 100, i32 5, i32 1, i32 0, ptr null)
  %handleB = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefromimplicitbinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  ret void
}
  )";

  auto M = parseAsm(Assembly);

  DXILResourceBindingInfo &DRBI =
      MAM->getResult<DXILResourceBindingAnalysis>(*M);
  EXPECT_TRUE(DRBI.hasImplicitBinding());
  EXPECT_FALSE(DRBI.hasOverlappingBinding());
}

} // namespace
