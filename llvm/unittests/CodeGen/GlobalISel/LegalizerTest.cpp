//===- LegalizerTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "GISelMITest.h"
#include "llvm/CodeGen/GlobalISel/LostDebugLocObserver.h"

#define DEBUG_TYPE "legalizer-test"

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

DefineLegalizerInfo(ALegalizer, {
  auto p0 = LLT::pointer(0, 64);
  auto v2s8 = LLT::vector(2, 8);
  auto v2s16 = LLT::vector(2, 16);
  getActionDefinitionsBuilder(G_LOAD)
      .legalForTypesWithMemDesc({{s16, p0, 8, 8}})
      .scalarize(0)
      .clampScalar(0, s16, s16);
  getActionDefinitionsBuilder(G_PTR_ADD).legalFor({{p0, s64}});
  getActionDefinitionsBuilder(G_CONSTANT).legalFor({s32, s64});
  getActionDefinitionsBuilder(G_BUILD_VECTOR)
      .legalFor({{v2s16, s16}})
      .clampScalar(1, s16, s16);
  getActionDefinitionsBuilder(G_BUILD_VECTOR_TRUNC).legalFor({{v2s8, s16}});
  getActionDefinitionsBuilder(G_ANYEXT).legalFor({{s32, s16}});
  getActionDefinitionsBuilder(G_ZEXT).legalFor({{s32, s16}});
  getActionDefinitionsBuilder(G_SEXT).legalFor({{s32, s16}});
  getActionDefinitionsBuilder(G_AND).legalFor({s32});
  getActionDefinitionsBuilder(G_SEXT_INREG).lower();
  getActionDefinitionsBuilder(G_ASHR).legalFor({{s32, s32}});
  getActionDefinitionsBuilder(G_SHL).legalFor({{s32, s32}});
})

TEST_F(AArch64GISelMITest, BasicLegalizerTest) {
  StringRef MIRString = R"(
    %vptr:_(p0) = COPY $x4
    %v:_(<2 x s8>) = G_LOAD %vptr:_(p0) :: (load 2, align 1)
    $h4 = COPY %v:_(<2 x s8>)
  )";
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString.rtrim(' '));
  if (!TM)
    return;

  ALegalizerInfo LI(MF->getSubtarget());
  LostDebugLocObserver LocObserver(DEBUG_TYPE);

  Legalizer::MFResult Result = Legalizer::legalizeMachineFunction(
      *MF, LI, {&LocObserver}, LocObserver, B);

  EXPECT_TRUE(isNullMIPtr(Result.FailedOn));
  EXPECT_TRUE(Result.Changed);

  StringRef CheckString = R"(
    CHECK:      %vptr:_(p0) = COPY $x4
    CHECK-NEXT: [[LOAD_0:%[0-9]+]]:_(s16) = G_LOAD %vptr:_(p0) :: (load 1)
    CHECK-NEXT: [[OFFSET_1:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
    CHECK-NEXT: [[VPTR_1:%[0-9]+]]:_(p0) = G_PTR_ADD %vptr:_, [[OFFSET_1]]:_(s64)
    CHECK-NEXT: [[LOAD_1:%[0-9]+]]:_(s16) = G_LOAD [[VPTR_1]]:_(p0) :: (load 1 from unknown-address + 1)
    CHECK-NEXT: [[V0:%[0-9]+]]:_(s16) = COPY [[LOAD_0]]:_(s16)
    CHECK-NEXT: [[V1:%[0-9]+]]:_(s16) = COPY [[LOAD_1]]:_(s16)
    CHECK-NEXT: %v:_(<2 x s8>) = G_BUILD_VECTOR_TRUNC [[V0]]:_(s16), [[V1]]:_(s16)
    CHECK-NEXT: $h4 = COPY %v:_(<2 x s8>)
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckString)) << *MF;
}

// Making sure the legalization finishes successfully w/o failure to combine
// away all the legalization artifacts regardless of the order of their
// creation.
TEST_F(AArch64GISelMITest, UnorderedArtifactCombiningTest) {
  StringRef MIRString = R"(
    %vptr:_(p0) = COPY $x4
    %v:_(<2 x s8>) = G_LOAD %vptr:_(p0) :: (load 2, align 1)
    %v0:_(s8), %v1:_(s8) = G_UNMERGE_VALUES %v:_(<2 x s8>)
    %v0_ext:_(s16) = G_ANYEXT %v0:_(s8)
    $h4 = COPY %v0_ext:_(s16)
  )";
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString.rtrim(' '));
  if (!TM)
    return;

  ALegalizerInfo LI(MF->getSubtarget());
  LostDebugLocObserver LocObserver(DEBUG_TYPE);

  // The events here unfold as follows:
  // 1. First, the function is scanned pre-forming the worklist of artifacts:
  //
  //      UNMERGE  (1): pushed into the worklist first, will be processed last.
  //         |
  //       ANYEXT  (2)
  //
  // 2. Second, the load is scalarized, and then its destination is widened,
  //    forming the following chain of legalization artifacts:
  //
  //        TRUNC  (4): created last, will be processed first.
  //         |
  // BUILD_VECTOR  (3)
  //         |
  //      UNMERGE  (1): pushed into the worklist first, will be processed last.
  //         |
  //       ANYEXT  (2)
  //
  // 3. Third, the artifacts are attempted to be combined in pairs, looking
  //    through the def-use chain from the roots towards the leafs, visiting the
  //    roots in order they happen to be in the worklist:
  //      (4) - (trunc):                  can not be combined;
  //      (3) - (build_vector (trunc)):   can not be combined;
  //      (2) - (anyext (unmerge)):       can not be combined;
  //      (1) - (unmerge (build_vector)): combined and eliminated;
  //
  //    leaving the function in the following state:
  //
  //        TRUNC  (1): moved to non-artifact instructions worklist first.
  //         |
  //       ANYEXT  (2): also moved to non-artifact instructions worklist.
  //
  //    Every other instruction is successfully legalized in full.
  //    If combining (unmerge (build_vector)) does not re-insert every artifact
  //    that had its def-use chain modified (shortened) into the artifact
  //    worklist (here it's just ANYEXT), the process moves on onto the next
  //    outer loop iteration of the top-level legalization algorithm here, w/o
  //    performing all the artifact combines possible. Let's consider this
  //    scenario first:
  //  4.A. Neither TRUNC, nor ANYEXT can be legalized in isolation, both of them
  //       get moved to the retry worklist, but no additional artifacts were
  //       created in the process, thus algorithm concludes no progress could be
  //       made, and fails.
  //  4.B. If, however, combining (unmerge (build_vector)) had re-inserted
  //       ANYEXT into the worklist (as ANYEXT's source changes, not by value,
  //       but by implementation), (anyext (trunc)) combine happens next, which
  //       fully eliminates all the artifacts and legalization succeeds.
  //
  //  We're looking into making sure that (4.B) happens here, not (4.A). Note
  //  that in that case the first scan through the artifacts worklist, while not
  //  being done in any guaranteed order, only needs to find the innermost
  //  pair(s) of artifacts that could be immediately combined out. After that
  //  the process follows def-use chains, making them shorter at each step, thus
  //  combining everything that can be combined in O(n) time.
  Legalizer::MFResult Result = Legalizer::legalizeMachineFunction(
      *MF, LI, {&LocObserver}, LocObserver, B);

  EXPECT_TRUE(isNullMIPtr(Result.FailedOn));
  EXPECT_TRUE(Result.Changed);

  StringRef CheckString = R"(
    CHECK:      %vptr:_(p0) = COPY $x4
    CHECK-NEXT: [[LOAD_0:%[0-9]+]]:_(s16) = G_LOAD %vptr:_(p0) :: (load 1)
    CHECK:      %v0_ext:_(s16) = COPY [[LOAD_0]]:_(s16)
    CHECK-NEXT: $h4 = COPY %v0_ext:_(s16)
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckString)) << *MF;
}

TEST_F(AArch64GISelMITest, UnorderedArtifactCombiningManyCopiesTest) {
  StringRef MIRString = R"(
    %vptr:_(p0) = COPY $x4
    %v:_(<2 x s8>) = G_LOAD %vptr:_(p0) :: (load 2, align 1)
    %vc0:_(<2 x s8>) = COPY %v:_(<2 x s8>)
    %vc1:_(<2 x s8>) = COPY %v:_(<2 x s8>)
    %vc00:_(s8), %vc01:_(s8) = G_UNMERGE_VALUES %vc0:_(<2 x s8>)
    %vc10:_(s8), %vc11:_(s8) = G_UNMERGE_VALUES %vc1:_(<2 x s8>)
    %v0t:_(s8) = COPY %vc00:_(s8)
    %v0:_(s8) = COPY %v0t:_(s8)
    %v1t:_(s8) = COPY %vc11:_(s8)
    %v1:_(s8) = COPY %v1t:_(s8)
    %v0_zext:_(s32) = G_ZEXT %v0:_(s8)
    %v1_sext:_(s32) = G_SEXT %v1:_(s8)
    $w4 = COPY %v0_zext:_(s32)
    $w5 = COPY %v1_sext:_(s32)
  )";
  LLVMTargetMachine *TM = createTargetMachineAndModule(MIRString.rtrim(' '));
  if (!TM)
    return;

  ALegalizerInfo LI(MF->getSubtarget());
  LostDebugLocObserver LocObserver(DEBUG_TYPE);

  Legalizer::MFResult Result = Legalizer::legalizeMachineFunction(
      *MF, LI, {&LocObserver}, LocObserver, B);

  EXPECT_TRUE(isNullMIPtr(Result.FailedOn));
  EXPECT_TRUE(Result.Changed);

  StringRef CheckString = R"(
    CHECK:      %vptr:_(p0) = COPY $x4
    CHECK-NEXT: [[LOAD_0:%[0-9]+]]:_(s16) = G_LOAD %vptr:_(p0) :: (load 1)
    CHECK-NEXT: [[OFFSET_1:%[0-9]+]]:_(s64) = G_CONSTANT i64 1
    CHECK-NEXT: [[VPTR_1:%[0-9]+]]:_(p0) = G_PTR_ADD %vptr:_, [[OFFSET_1]]:_(s64)
    CHECK-NEXT: [[LOAD_1:%[0-9]+]]:_(s16) = G_LOAD [[VPTR_1]]:_(p0) :: (load 1 from unknown-address + 1)
    CHECK-NEXT: [[FF_MASK:%[0-9]+]]:_(s32) = G_CONSTANT i32 255
    CHECK-NEXT: [[V0_EXT:%[0-9]+]]:_(s32) = G_ANYEXT [[LOAD_0]]:_(s16)
    CHECK-NEXT: %v0_zext:_(s32) = G_AND [[V0_EXT]]:_, [[FF_MASK]]:_
    CHECK-NEXT: [[V1_EXT:%[0-9]+]]:_(s32) = G_ANYEXT [[LOAD_1]]:_(s16)
    CHECK-NEXT: [[SHAMNT:%[0-9]+]]:_(s32) = G_CONSTANT i32 24
    CHECK-NEXT: [[V1_SHL:%[0-9]+]]:_(s32) = G_SHL [[V1_EXT]]:_, [[SHAMNT]]:_(s32)
    CHECK-NEXT: %v1_sext:_(s32) = G_ASHR [[V1_SHL]]:_, [[SHAMNT]]:_(s32)
    CHECK-NEXT: $w4 = COPY %v0_zext:_(s32)
    CHECK-NEXT: $w5 = COPY %v1_sext:_(s32)
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckString)) << *MF;
}

} // namespace
