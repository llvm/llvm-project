//===- llvm/unittests/Frontend/OpenMPComposeTest.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
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
