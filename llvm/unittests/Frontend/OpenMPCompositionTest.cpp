//===- llvm/unittests/Frontend/OpenMPCompositionTest.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::omp;

TEST(Composition, GetLeafConstructs) {
  ArrayRef<Directive> L1 = getLeafConstructs(OMPD_loop);
  ASSERT_EQ(L1, (ArrayRef<Directive>{}));
  ArrayRef<Directive> L2 = getLeafConstructs(OMPD_parallel_for);
  ASSERT_EQ(L2, (ArrayRef<Directive>{OMPD_parallel, OMPD_for}));
  ArrayRef<Directive> L3 = getLeafConstructs(OMPD_parallel_for_simd);
  ASSERT_EQ(L3, (ArrayRef<Directive>{OMPD_parallel, OMPD_for, OMPD_simd}));
}

TEST(Composition, GetLeafConstructsOrSelf) {
  ArrayRef<Directive> L1 = getLeafConstructsOrSelf(OMPD_loop);
  ASSERT_EQ(L1, (ArrayRef<Directive>{OMPD_loop}));
  ArrayRef<Directive> L2 = getLeafConstructsOrSelf(OMPD_parallel_for);
  ASSERT_EQ(L2, (ArrayRef<Directive>{OMPD_parallel, OMPD_for}));
  ArrayRef<Directive> L3 = getLeafConstructsOrSelf(OMPD_parallel_for_simd);
  ASSERT_EQ(L3, (ArrayRef<Directive>{OMPD_parallel, OMPD_for, OMPD_simd}));
}

TEST(Composition, GetCompoundConstruct) {
  Directive C1 =
      getCompoundConstruct({OMPD_target, OMPD_teams, OMPD_distribute});
  ASSERT_EQ(C1, OMPD_target_teams_distribute);
  Directive C2 = getCompoundConstruct({OMPD_target});
  ASSERT_EQ(C2, OMPD_target);
  Directive C3 = getCompoundConstruct({OMPD_target, OMPD_masked});
  ASSERT_EQ(C3, OMPD_unknown);
  Directive C4 = getCompoundConstruct({OMPD_target, OMPD_teams_distribute});
  ASSERT_EQ(C4, OMPD_target_teams_distribute);
  Directive C5 = getCompoundConstruct({});
  ASSERT_EQ(C5, OMPD_unknown);
  Directive C6 = getCompoundConstruct({OMPD_parallel_for, OMPD_simd});
  ASSERT_EQ(C6, OMPD_parallel_for_simd);
  Directive C7 = getCompoundConstruct({OMPD_do, OMPD_simd});
  ASSERT_EQ(C7, OMPD_do_simd); // Make sure it's not OMPD_end_do_simd
}

TEST(Composition, GetLeafOrCompositeConstructs) {
  SmallVector<Directive> Out1;
  auto Ret1 = getLeafOrCompositeConstructs(
      OMPD_target_teams_distribute_parallel_for, Out1);
  ASSERT_EQ(Ret1, ArrayRef<Directive>(Out1));
  ASSERT_EQ((ArrayRef<Directive>(Out1)),
            (ArrayRef<Directive>{OMPD_target, OMPD_teams,
                                 OMPD_distribute_parallel_for}));

  SmallVector<Directive> Out2;
  auto Ret2 =
      getLeafOrCompositeConstructs(OMPD_parallel_masked_taskloop_simd, Out2);
  ASSERT_EQ(Ret2, ArrayRef<Directive>(Out2));
  ASSERT_EQ(
      (ArrayRef<Directive>(Out2)),
      (ArrayRef<Directive>{OMPD_parallel, OMPD_masked, OMPD_taskloop_simd}));

  SmallVector<Directive> Out3;
  auto Ret3 =
      getLeafOrCompositeConstructs(OMPD_distribute_parallel_do_simd, Out3);
  ASSERT_EQ(Ret3, ArrayRef<Directive>(Out3));
  ASSERT_EQ((ArrayRef<Directive>(Out3)),
            (ArrayRef<Directive>{OMPD_distribute_parallel_do_simd}));

  SmallVector<Directive> Out4;
  auto Ret4 = getLeafOrCompositeConstructs(OMPD_target_parallel_loop, Out4);
  ASSERT_EQ(Ret4, ArrayRef<Directive>(Out4));
  ASSERT_EQ((ArrayRef<Directive>(Out4)),
            (ArrayRef<Directive>{OMPD_target, OMPD_parallel, OMPD_loop}));
}

TEST(Composition, IsLeafConstruct) {
  ASSERT_TRUE(isLeafConstruct(OMPD_loop));
  ASSERT_TRUE(isLeafConstruct(OMPD_teams));
  ASSERT_FALSE(isLeafConstruct(OMPD_for_simd));
  ASSERT_FALSE(isLeafConstruct(OMPD_distribute_simd));
  ASSERT_FALSE(isLeafConstruct(OMPD_parallel_for));
}

TEST(Composition, IsCompositeConstruct) {
  ASSERT_TRUE(isCompositeConstruct(OMPD_distribute_simd));
  ASSERT_FALSE(isCompositeConstruct(OMPD_for));
  ASSERT_TRUE(isCompositeConstruct(OMPD_for_simd));
  // directive-name-A = "parallel", directive-name-B = "for simd",
  // only directive-name-B is loop-associated, so this is not a
  // composite construct, even though "for simd" is.
  ASSERT_FALSE(isCompositeConstruct(OMPD_parallel_for_simd));
}

TEST(Composition, IsCombinedConstruct) {
  // "parallel for simd" is a combined construct, see comment in
  // IsCompositeConstruct.
  ASSERT_TRUE(isCombinedConstruct(OMPD_parallel_for_simd));
  ASSERT_FALSE(isCombinedConstruct(OMPD_for_simd));
  ASSERT_TRUE(isCombinedConstruct(OMPD_parallel_for));
  ASSERT_FALSE(isCombinedConstruct(OMPD_parallel));
}
