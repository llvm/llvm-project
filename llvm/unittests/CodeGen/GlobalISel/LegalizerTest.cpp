//===- LegalizerTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GISelMITest.h"
#include "llvm/CodeGen/GlobalISel/Legalizer.h"

using namespace LegalizeActions;
using namespace LegalizeMutations;
using namespace LegalityPredicates;

namespace {

::testing::AssertionResult isNullMIPtr(const MachineInstr *MI) {
  if (MI == nullptr)
    return ::testing::AssertionSuccess();
  std::string MIBuffer;
  raw_string_ostream MISStream(MIBuffer);
  MI->print(MISStream, /*IsStandalone=*/true, /*SkipOpers=*/false,
            /*SkipDebugLoc=*/false, /*AddNewLine=*/false);
  return ::testing::AssertionFailure()
         << "unable to legalize instruction: " << MISStream.str();
}

TEST_F(GISelMITest, BasicLegalizerTest) {
  StringRef MIRString = R"(
    %vptr:_(p0) = COPY $x4
    %v:_(<2 x s8>) = G_LOAD %vptr:_(p0) :: (load 2, align 1)
    $h4 = COPY %v:_(<2 x s8>)
  )";
  setUp(MIRString.rtrim(' '));
  if (!TM)
    return;

  DefineLegalizerInfo(ALegalizer, {
    auto p0 = LLT::pointer(0, 64);
    auto v2s8 = LLT::vector(2, 8);
    auto v2s16 = LLT::vector(2, 16);
    getActionDefinitionsBuilder(G_LOAD)
        .legalForTypesWithMemDesc({{s16, p0, 8, 8}})
        .scalarize(0)
        .clampScalar(0, s16, s16);
    getActionDefinitionsBuilder(G_PTR_ADD).legalFor({{p0, s64}});
    getActionDefinitionsBuilder(G_CONSTANT).legalFor({s64});
    getActionDefinitionsBuilder(G_BUILD_VECTOR)
        .legalFor({{v2s16, s16}})
        .clampScalar(1, s16, s16);
    getActionDefinitionsBuilder(G_BUILD_VECTOR_TRUNC).legalFor({{v2s8, s16}});
    getActionDefinitionsBuilder(G_ANYEXT).legalFor({{s32, s16}});
  });

  ALegalizerInfo LI(MF->getSubtarget());

  Legalizer::MFResult Result =
      Legalizer::legalizeMachineFunction(*MF, LI, {}, B);

  EXPECT_TRUE(isNullMIPtr(Result.FailedOn));
  EXPECT_TRUE(Result.Changed);

  StringRef CheckString = R"(
    CHECK:      %vptr:_(p0) = COPY $x4
    CHECK-NEXT: [[LOAD_0:%[0-9]+]]:_(s16) = G_LOAD %vptr:_(p0) :: (load 1)
    CHECK-NEXT: [[OFFSET_1:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
    CHECK-NEXT: [[VPTR_1:%[0-9]+]]:_(p0) = G_PTR_ADD %vptr:_, [[OFFSET_1]]:_(s64)
    CHECK-NEXT: [[LOAD_1:%[0-9]+]]:_(s16) = G_LOAD [[VPTR_1]]:_(p0) :: (load 1)
    CHECK-NEXT: [[V0:%[0-9]+]]:_(s16) = COPY [[LOAD_0]]:_(s16)
    CHECK-NEXT: [[V1:%[0-9]+]]:_(s16) = COPY [[LOAD_1]]:_(s16)
    CHECK-NEXT: %v:_(<2 x s8>) = G_BUILD_VECTOR_TRUNC [[V0]]:_(s16), [[V1]]:_(s16)
    CHECK-NEXT: $h4 = COPY %v:_(<2 x s8>)
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckString)) << *MF;
}

} // namespace
