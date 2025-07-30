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

  void checkExpectedSpaceAndFreeRanges(
      DXILResourceBindingInfo::RegisterSpace &RegSpace, uint32_t ExpSpace,
      ArrayRef<uint32_t> ExpValues) {
    EXPECT_EQ(RegSpace.Space, ExpSpace);
    EXPECT_EQ(RegSpace.FreeRanges.size() * 2, ExpValues.size());
    unsigned I = 0;
    for (auto &R : RegSpace.FreeRanges) {
      EXPECT_EQ(R.LowerBound, ExpValues[I]);
      EXPECT_EQ(R.UpperBound, ExpValues[I + 1]);
      I += 2;
    }
  }
};

TEST_F(ResourceBindingAnalysisTest, TestTrivialCase) {
  // RWBuffer<float> Buf : register(u5);
  StringRef Assembly = R"(
define void @main() {
entry:
  %handle = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 5, i32 1, i32 0, i1 false, ptr null)
  ret void
}
  )";

  auto M = parseAsm(Assembly);

  DXILResourceBindingInfo &DRBI =
      MAM->getResult<DXILResourceBindingAnalysis>(*M);

  EXPECT_EQ(false, DRBI.hasImplicitBinding());
  EXPECT_EQ(false, DRBI.hasOverlappingBinding());

  // check that UAV has exactly one gap
  DXILResourceBindingInfo::BindingSpaces &UAVSpaces =
      DRBI.getBindingSpaces(ResourceClass::UAV);
  EXPECT_EQ(UAVSpaces.RC, ResourceClass::UAV);
  EXPECT_EQ(UAVSpaces.Spaces.size(), 1u);
  checkExpectedSpaceAndFreeRanges(UAVSpaces.Spaces[0], 0,
                                  {0, 4, 6, UINT32_MAX});

  // check that other kinds of register spaces are all available
  for (auto RC :
       {ResourceClass::SRV, ResourceClass::CBuffer, ResourceClass::Sampler}) {
    DXILResourceBindingInfo::BindingSpaces &Spaces = DRBI.getBindingSpaces(RC);
    EXPECT_EQ(Spaces.RC, RC);
    EXPECT_EQ(Spaces.Spaces.size(), 0u);
  }
}

TEST_F(ResourceBindingAnalysisTest, TestManyBindings) {
  // cbuffer CB                 : register(b3) { int a; }
  // RWBuffer<float4> A[5]      : register(u10, space20);
  // StructuredBuffer<int> B    : register(t5);
  // RWBuffer<float> C          : register(u5);
  // StructuredBuffer<int> D[5] : register(t0);
  // RWBuffer<float> E[2]       : register(u2);
  // SamplerState S1            : register(s5, space2);
  // SamplerState S2            : register(s4, space2);
  StringRef Assembly = R"(
%__cblayout_CB = type <{ i32 }>
define void @main() {
entry:
  %handleCB = call target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 4, 0)) @llvm.dx.resource.handlefrombinding(i32 0, i32 3, i32 1, i32 0, i1 false, ptr null)
  %handleA = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding(i32 20, i32 10, i32 5, i32 0, i1 false, ptr null)
  %handleB = call target("dx.RawBuffer", i32, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 5, i32 1, i32 0, i1 false, ptr null)
  %handleC = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 5, i32 1, i32 0, i1 false, ptr null)
  %handleD = call target("dx.RawBuffer", i32, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 5, i32 4, i1 false, ptr null)
  %handleE = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 2, i32 2, i32 0, i1 false, ptr null)
  %handleS1 = call target("dx.Sampler", 0) @llvm.dx.resource.handlefrombinding(i32 2, i32 5, i32 1, i32 0, i1 false, ptr null)
  %handleS2 = call target("dx.Sampler", 0) @llvm.dx.resource.handlefrombinding(i32 2, i32 4, i32 1, i32 0, i1 false, ptr null)
  ; duplicate binding for the same resource
  %handleD2 = call target("dx.RawBuffer", i32, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 5, i32 4, i1 false, ptr null)
  ret void
}
  )";

  auto M = parseAsm(Assembly);

  DXILResourceBindingInfo &DRBI =
      MAM->getResult<DXILResourceBindingAnalysis>(*M);

  EXPECT_EQ(false, DRBI.hasImplicitBinding());
  EXPECT_EQ(false, DRBI.hasOverlappingBinding());

  DXILResourceBindingInfo::BindingSpaces &SRVSpaces =
      DRBI.getBindingSpaces(ResourceClass::SRV);
  EXPECT_EQ(SRVSpaces.RC, ResourceClass::SRV);
  EXPECT_EQ(SRVSpaces.Spaces.size(), 1u);
  // verify that consecutive bindings are merged
  // (SRVSpaces has only one free space range {6, UINT32_MAX}).
  checkExpectedSpaceAndFreeRanges(SRVSpaces.Spaces[0], 0, {6, UINT32_MAX});

  DXILResourceBindingInfo::BindingSpaces &UAVSpaces =
      DRBI.getBindingSpaces(ResourceClass::UAV);
  EXPECT_EQ(UAVSpaces.RC, ResourceClass::UAV);
  EXPECT_EQ(UAVSpaces.Spaces.size(), 2u);
  checkExpectedSpaceAndFreeRanges(UAVSpaces.Spaces[0], 0,
                                  {0, 1, 4, 4, 6, UINT32_MAX});
  checkExpectedSpaceAndFreeRanges(UAVSpaces.Spaces[1], 20,
                                  {0, 9, 15, UINT32_MAX});

  DXILResourceBindingInfo::BindingSpaces &CBufferSpaces =
      DRBI.getBindingSpaces(ResourceClass::CBuffer);
  EXPECT_EQ(CBufferSpaces.RC, ResourceClass::CBuffer);
  EXPECT_EQ(CBufferSpaces.Spaces.size(), 1u);
  checkExpectedSpaceAndFreeRanges(CBufferSpaces.Spaces[0], 0,
                                  {0, 2, 4, UINT32_MAX});

  DXILResourceBindingInfo::BindingSpaces &SamplerSpaces =
      DRBI.getBindingSpaces(ResourceClass::Sampler);
  EXPECT_EQ(SamplerSpaces.RC, ResourceClass::Sampler);
  EXPECT_EQ(SamplerSpaces.Spaces.size(), 1u);
  checkExpectedSpaceAndFreeRanges(SamplerSpaces.Spaces[0], 2,
                                  {0, 3, 6, UINT32_MAX});
}

TEST_F(ResourceBindingAnalysisTest, TestUnboundedAndOverlap) {
  // StructuredBuffer<float> A[]  : register(t5);
  // StructuredBuffer<float> B[3] : register(t0);
  // StructuredBuffer<float> C[]  : register(t0, space2);
  // StructuredBuffer<float> D    : register(t4, space2); /* overlapping */
  StringRef Assembly = R"(
define void @main() {
entry:
  %handleA = call target("dx.RawBuffer", float, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 5, i32 -1, i32 10, i1 false, ptr null)
  %handleB = call target("dx.RawBuffer", float, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 3, i32 0, i1 false, ptr null)
  %handleC = call target("dx.RawBuffer", float, 0, 0) @llvm.dx.resource.handlefrombinding(i32 2, i32 0, i32 -1, i32 100, i1 false, ptr null)
  %handleD = call target("dx.RawBuffer", float, 0, 0) @llvm.dx.resource.handlefrombinding(i32 2, i32 4, i32 1, i32 0, i1 false, ptr null)
  ret void
}
  )";

  auto M = parseAsm(Assembly);

  DXILResourceBindingInfo &DRBI =
      MAM->getResult<DXILResourceBindingAnalysis>(*M);

  EXPECT_EQ(false, DRBI.hasImplicitBinding());
  EXPECT_EQ(true, DRBI.hasOverlappingBinding());

  DXILResourceBindingInfo::BindingSpaces &SRVSpaces =
      DRBI.getBindingSpaces(ResourceClass::SRV);
  EXPECT_EQ(SRVSpaces.RC, ResourceClass::SRV);
  EXPECT_EQ(SRVSpaces.Spaces.size(), 2u);
  checkExpectedSpaceAndFreeRanges(SRVSpaces.Spaces[0], 0, {3, 4});
  checkExpectedSpaceAndFreeRanges(SRVSpaces.Spaces[1], 2, {});
}

TEST_F(ResourceBindingAnalysisTest, TestExactOverlap) {
  // StructuredBuffer<float> A  : register(t5);
  // StructuredBuffer<float> B  : register(t5);
  StringRef Assembly = R"(
@A.str = private unnamed_addr constant [2 x i8] c"A\00", align 1
@B.str = private unnamed_addr constant [2 x i8] c"B\00", align 1
define void @main() {
entry:
  %handleA = call target("dx.RawBuffer", float, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 5, i32 1, i32 0, i1 false, ptr @A.str)
  %handleB = call target("dx.RawBuffer", float, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 5, i32 1, i32 0, i1 false, ptr @B.str)
  ret void
}
  )";

  auto M = parseAsm(Assembly);

  DXILResourceBindingInfo &DRBI =
      MAM->getResult<DXILResourceBindingAnalysis>(*M);

  EXPECT_EQ(false, DRBI.hasImplicitBinding());
  EXPECT_EQ(true, DRBI.hasOverlappingBinding());

  DXILResourceBindingInfo::BindingSpaces &SRVSpaces =
      DRBI.getBindingSpaces(ResourceClass::SRV);
  EXPECT_EQ(SRVSpaces.RC, ResourceClass::SRV);
  EXPECT_EQ(SRVSpaces.Spaces.size(), 1u);
  checkExpectedSpaceAndFreeRanges(SRVSpaces.Spaces[0], 0,
                                  {0, 4, 6, UINT32_MAX});
}

TEST_F(ResourceBindingAnalysisTest, TestEndOfRange) {
  // RWBuffer<float> A     : register(u4294967295);  /* UINT32_MAX */
  // RWBuffer<float> B[10] : register(u4294967286, space1);
  //                         /* range (UINT32_MAX - 9, UINT32_MAX )*/
  // RWBuffer<float> C[10] : register(u2147483647, space2);
  //                         /* range (INT32_MAX, INT32_MAX + 9) */
  StringRef Assembly = R"(
%__cblayout_CB = type <{ i32 }>
define void @main() {
entry:
  %handleA = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 -1, i32 1, i32 0, i1 false, ptr null)
  %handleB = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding(i32 1, i32 -10, i32 10, i32 50, i1 false, ptr null)
  %handleC = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding(i32 2, i32 2147483647, i32 10, i32 100, i1 false, ptr null)
  ret void
}
  )";

  auto M = parseAsm(Assembly);

  DXILResourceBindingInfo &DRBI =
      MAM->getResult<DXILResourceBindingAnalysis>(*M);

  EXPECT_EQ(false, DRBI.hasImplicitBinding());
  EXPECT_EQ(false, DRBI.hasOverlappingBinding());

  DXILResourceBindingInfo::BindingSpaces &UAVSpaces =
      DRBI.getBindingSpaces(ResourceClass::UAV);
  EXPECT_EQ(UAVSpaces.RC, ResourceClass::UAV);
  EXPECT_EQ(UAVSpaces.Spaces.size(), 3u);
  checkExpectedSpaceAndFreeRanges(UAVSpaces.Spaces[0], 0, {0, UINT32_MAX - 1});
  checkExpectedSpaceAndFreeRanges(UAVSpaces.Spaces[1], 1, {0, UINT32_MAX - 10});
  checkExpectedSpaceAndFreeRanges(
      UAVSpaces.Spaces[2], 2,
      {0, (uint32_t)INT32_MAX - 1, (uint32_t)INT32_MAX + 10, UINT32_MAX});
}

TEST_F(ResourceBindingAnalysisTest, TestImplicitFlag) {
  // RWBuffer<float> A : register(u5, space100);
  // RWBuffer<float> B;
  StringRef Assembly = R"(
define void @main() {
entry:
  %handleA = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding(i32 100, i32 5, i32 1, i32 0, i1 false, ptr null)
  %handleB = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefromimplicitbinding(i32 0, i32 0, i32 1, i32 0, i1 false, ptr null)
  ret void
}
  )";

  auto M = parseAsm(Assembly);

  DXILResourceBindingInfo &DRBI =
      MAM->getResult<DXILResourceBindingAnalysis>(*M);
  EXPECT_EQ(true, DRBI.hasImplicitBinding());
  EXPECT_EQ(false, DRBI.hasOverlappingBinding());

  DXILResourceBindingInfo::BindingSpaces &UAVSpaces =
      DRBI.getBindingSpaces(ResourceClass::UAV);
  EXPECT_EQ(UAVSpaces.RC, ResourceClass::UAV);
  EXPECT_EQ(UAVSpaces.Spaces.size(), 1u);
  checkExpectedSpaceAndFreeRanges(UAVSpaces.Spaces[0], 100,
                                  {0, 4, 6, UINT32_MAX});
}

} // namespace
