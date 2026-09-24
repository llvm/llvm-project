//===- LoopRotationUtilsTest.cpp - Unit tests for LoopRotation utility ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LoopRotationUtils.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("LoopRotationUtilsTest", errs());
  return Mod;
}

/// This test contains multi-deopt-exits pattern that might allow loop rotation
/// to trigger multiple times if multiple rotations are enabled.
/// At least one rotation should be performed, no matter what loop rotation settings are.
TEST(LoopRotate, MultiDeoptExit) {
  LLVMContext C;

  std::unique_ptr<Module> M = parseIR(
    C,
    R"(
declare i32 @llvm.experimental.deoptimize.i32(...)

define i32 @test(ptr nonnull %a, i64 %x) {
entry:
  br label %for.cond1

for.cond1:
  %idx = phi i64 [ 0, %entry ], [ %idx.next, %for.tail ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %for.tail ]
  %a.idx = getelementptr inbounds i32, ptr %a, i64 %idx
  %val.a.idx = load i32, ptr %a.idx, align 4
  %zero.check = icmp eq i32 %val.a.idx, 0
  br i1 %zero.check, label %deopt.exit, label %for.cond2

for.cond2:
  %for.check = icmp ult i64 %idx, %x
  br i1 %for.check, label %for.body, label %return

for.body:
  br label %for.tail

for.tail:
  %sum.next = add i32 %sum, %val.a.idx
  %idx.next = add nuw nsw i64 %idx, 1
  br label %for.cond1

return:
  ret i32 %sum

deopt.exit:
  %deopt.val = call i32(...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 %val.a.idx) ]
  ret i32 %deopt.val
})"
    );

  auto *F = M->getFunction("test");
  DominatorTree DT(*F);
  LoopInfo LI(DT);
  AssumptionCache AC(*F);
  TargetTransformInfo TTI(M->getDataLayout());
  TargetLibraryInfoImpl TLII(M->getTargetTriple());
  TargetLibraryInfo TLI(TLII);
  ScalarEvolution SE(*F, TLI, AC, DT, LI);
  SimplifyQuery SQ(M->getDataLayout());

  Loop *L = *LI.begin();

  bool ret = LoopRotation(L, &LI, &TTI,
                          &AC, &DT,
                          &SE, nullptr,
                          SQ, true, -1, false);
  EXPECT_TRUE(ret);
}

/// Checking a special case of multi-deopt exit loop that can not perform
/// required amount of rotations due to the desired header containing
/// non-duplicatable code.
/// Similar to MultiDeoptExit test this one should do at least one rotation and
/// pass no matter what loop rotation settings are.
TEST(LoopRotate, MultiDeoptExit_Nondup) {
  LLVMContext C;

  std::unique_ptr<Module> M = parseIR(
    C,
    R"(
; Rotation should be done once, attempted twice.
; Second time fails due to non-duplicatable header.

declare i32 @llvm.experimental.deoptimize.i32(...)

declare void @nondup()

define i32 @test_nondup(ptr nonnull %a, i64 %x) {
entry:
  br label %for.cond1

for.cond1:
  %idx = phi i64 [ 0, %entry ], [ %idx.next, %for.tail ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %for.tail ]
  %a.idx = getelementptr inbounds i32, ptr %a, i64 %idx
  %val.a.idx = load i32, ptr %a.idx, align 4
  %zero.check = icmp eq i32 %val.a.idx, 0
  br i1 %zero.check, label %deopt.exit, label %for.cond2

for.cond2:
  call void @nondup() noduplicate
  %for.check = icmp ult i64 %idx, %x
  br i1 %for.check, label %for.body, label %return

for.body:
  br label %for.tail

for.tail:
  %sum.next = add i32 %sum, %val.a.idx
  %idx.next = add nuw nsw i64 %idx, 1
  br label %for.cond1

return:
  ret i32 %sum

deopt.exit:
  %deopt.val = call i32(...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 %val.a.idx) ]
  ret i32 %deopt.val
})"
    );

  auto *F = M->getFunction("test_nondup");
  DominatorTree DT(*F);
  LoopInfo LI(DT);
  AssumptionCache AC(*F);
  TargetTransformInfo TTI(M->getDataLayout());
  TargetLibraryInfoImpl TLII(M->getTargetTriple());
  TargetLibraryInfo TLI(TLII);
  ScalarEvolution SE(*F, TLI, AC, DT, LI);
  SimplifyQuery SQ(M->getDataLayout());

  Loop *L = *LI.begin();

  bool ret = LoopRotation(L, &LI, &TTI,
                          &AC, &DT,
                          &SE, nullptr,
                          SQ, true, -1, false);
  /// LoopRotation should properly report "true" as we still perform the first rotation
  /// so we do change the IR.
  EXPECT_TRUE(ret);
}

TEST(LoopRotate, PreserveMappedWaveCounts) {
  for (uint64_t HeaderCount : {30u, 31u}) {
    SCOPED_TRACE(HeaderCount);
    LLVMContext C;
    std::unique_ptr<Module> M = parseIR(C, R"(
define void @test(ptr %out, i32 %start) {
entry:
  br label %header
header:
  %i = phi i32 [ %start, %entry ], [ %inc, %body ]
  %cond = icmp ult i32 %i, 4
  br i1 %cond, label %body, label %exit
exit:
  ret void
body:
  store volatile i32 %i, ptr %out
  %inc = add i32 %i, 1
  br label %header
}
)");
    ASSERT_TRUE(M);
    Function &F = *M->getFunction("test");
    if (HeaderCount == 30)
      F.getArg(1)->replaceAllUsesWith(ConstantInt::get(Type::getInt32Ty(C), 0));
    setBlockWaveCounts(F, {6, HeaderCount, 6, 24});

    // An unused historical identity makes this a valid partial profile.
    MDNode *Table = F.getMetadata(LLVMContext::MD_wave_profile);
    SmallVector<Metadata *> Ops(Table->op_begin(), Table->op_end());
    Ops.push_back(
        ConstantAsMetadata::get(ConstantInt::get(Type::getInt64Ty(C), 0)));
    F.setMetadata(LLVMContext::MD_wave_profile, MDNode::getDistinct(C, Ops));

    SmallVector<uint64_t> Counts;
    BitVector Valid;
    EXPECT_FALSE(extractBlockWaveCounts(F, Counts, &Valid));

    DominatorTree DT(F);
    LoopInfo LI(DT);
    AssumptionCache AC(F);
    TargetTransformInfo TTI(M->getDataLayout());
    TargetLibraryInfoImpl TLII(M->getTargetTriple());
    TargetLibraryInfo TLI(TLII);
    ScalarEvolution SE(F, TLI, AC, DT, LI);
    SimplifyQuery SQ(M->getDataLayout());
    ASSERT_TRUE(LoopRotation(*LI.begin(), &LI, &TTI, &AC, &DT, &SE, nullptr, SQ,
                             true, -1, false));

    uint64_t EntryCount = 0;
    ASSERT_TRUE(extractMappedBlockWaveCounts(F, Counts, Valid, EntryCount));
    EXPECT_EQ(EntryCount, 6u);
    EXPECT_GE(F.getMetadata(LLVMContext::MD_wave_profile)->getNumOperands(),
              7u);
    for (auto [Index, BB] : enumerate(F)) {
      if (BB.getName() == "entry" || BB.getName() == "exit") {
        EXPECT_TRUE(Valid[Index]);
        EXPECT_EQ(Counts[Index], 6u);
      } else if (BB.getName() == "body") {
        EXPECT_TRUE(Valid[Index]);
        EXPECT_EQ(Counts[Index], 24u);
      } else {
        EXPECT_FALSE(Valid[Index]);
      }
    }
  }
}
