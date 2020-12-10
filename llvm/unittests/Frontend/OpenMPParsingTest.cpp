//===- llvm/unittest/IR/OpenMPIRParsingTest.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::omp;

namespace {

TEST(OpenMPParsingTest, OpenMPDirectiveKind) {
  EXPECT_EQ(getOpenMPDirectiveKind("foobar"), OMPD_unknown);

  EXPECT_EQ(getOpenMPDirectiveKind("for"), OMPD_for);
  EXPECT_EQ(getOpenMPDirectiveKind("simd"), OMPD_simd);
  EXPECT_EQ(getOpenMPDirectiveKind("for simd"), OMPD_for_simd);
}

TEST(OpenMPParsingTest, getOpenMPDirectiveName) {
  EXPECT_EQ(getOpenMPDirectiveName(OMPD_unknown), "unknown");

  EXPECT_EQ(getOpenMPDirectiveName(OMPD_for), "for");
  EXPECT_EQ(getOpenMPDirectiveName(OMPD_simd), "simd");
  EXPECT_EQ(getOpenMPDirectiveName(OMPD_for_simd), "for simd");
}

TEST(OpenMPParsingTest, getOpenMPClauseKind) {
  EXPECT_EQ(getOpenMPClauseKind("foobar"), OMPC_unknown);

  EXPECT_EQ(getOpenMPClauseKind("schedule"), OMPC_schedule);
  EXPECT_EQ(getOpenMPClauseKind("if"), OMPC_if);
}

TEST(OpenMPParsingTest, getOpenMPClauseName) {
  EXPECT_EQ(getOpenMPClauseName(OMPC_unknown), "unknown");

  EXPECT_EQ(getOpenMPClauseName(OMPC_schedule), "schedule");
  EXPECT_EQ(getOpenMPClauseName(OMPC_if), "if");
}

TEST(OpenMPParsingTest, isAllowedClauseForDirective) {
  EXPECT_TRUE(isAllowedClauseForDirective(OMPD_for, OMPC_schedule, 30));
  EXPECT_FALSE(isAllowedClauseForDirective(OMPD_for, OMPC_num_teams, 30));

  EXPECT_FALSE(isAllowedClauseForDirective(OMPD_for, OMPC_order, 30));
  EXPECT_FALSE(isAllowedClauseForDirective(OMPD_for, OMPC_order, 45));
  EXPECT_TRUE(isAllowedClauseForDirective(OMPD_for, OMPC_order, 50));
  EXPECT_TRUE(isAllowedClauseForDirective(OMPD_for, OMPC_order, 51));
}

TEST(OpenMPParsingTest, getOrderKind) {
  EXPECT_EQ(getOrderKind("foobar"), OMP_ORDER_concurrent);
  EXPECT_EQ(getOrderKind("default"), OMP_ORDER_concurrent);
}

TEST(OpenMPParsingTest, getProcBindKind) {
  EXPECT_EQ(getProcBindKind("foobar"), OMP_PROC_BIND_unknown);

  EXPECT_EQ(getProcBindKind("master"), OMP_PROC_BIND_master);
  EXPECT_EQ(getProcBindKind("close"), OMP_PROC_BIND_close);
  EXPECT_EQ(getProcBindKind("spread"), OMP_PROC_BIND_spread);
  EXPECT_EQ(getProcBindKind("default"), OMP_PROC_BIND_default);
  EXPECT_EQ(getProcBindKind("unknown"), OMP_PROC_BIND_unknown);
}

TEST(OpenMPParsingTest, getScheduleKind) {
  EXPECT_EQ(getScheduleKind("foobar"), OMP_SCHEDULE_Default);

  // FIXME: Why are these not lower case?
  EXPECT_EQ(getScheduleKind("Static"), OMP_SCHEDULE_Static);
  EXPECT_EQ(getScheduleKind("Dynamic"), OMP_SCHEDULE_Dynamic);
  EXPECT_EQ(getScheduleKind("Guided"), OMP_SCHEDULE_Guided);
  EXPECT_EQ(getScheduleKind("Auto"), OMP_SCHEDULE_Auto);
  EXPECT_EQ(getScheduleKind("Runtime"), OMP_SCHEDULE_Runtime);
  EXPECT_EQ(getScheduleKind("Default"), OMP_SCHEDULE_Default);
}

} // namespace
