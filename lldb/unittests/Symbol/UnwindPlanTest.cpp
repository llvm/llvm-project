//===-- UnwindPlanTest.cpp ------------------------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/UnwindPlan.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;

static UnwindPlan::RowSP make_simple_row(addr_t offset, uint64_t cfa_value) {
  UnwindPlan::RowSP row_sp = std::make_shared<UnwindPlan::Row>();
  row_sp->SetOffset(offset);
  row_sp->GetCFAValue().SetIsConstant(cfa_value);
  return row_sp;
}

TEST(UnwindPlan, InsertRow) {
  UnwindPlan::RowSP row1_sp = make_simple_row(0, 42);
  UnwindPlan::RowSP row2_sp = make_simple_row(0, 47);

  UnwindPlan plan(eRegisterKindGeneric);
  plan.InsertRow(row1_sp);
  EXPECT_THAT(plan.GetRowForFunctionOffset(0), testing::Pointee(*row1_sp));

  plan.InsertRow(row2_sp, /*replace_existing=*/false);
  EXPECT_THAT(plan.GetRowForFunctionOffset(0), testing::Pointee(*row1_sp));

  plan.InsertRow(row2_sp, /*replace_existing=*/true);
  EXPECT_THAT(plan.GetRowForFunctionOffset(0), testing::Pointee(*row2_sp));
}

TEST(UnwindPlan, GetRowForFunctionOffset) {
  UnwindPlan::RowSP row1_sp = make_simple_row(10, 42);
  UnwindPlan::RowSP row2_sp = make_simple_row(20, 47);

  UnwindPlan plan(eRegisterKindGeneric);
  plan.InsertRow(row1_sp);
  plan.InsertRow(row2_sp);

  EXPECT_THAT(plan.GetRowForFunctionOffset(0), nullptr);
  EXPECT_THAT(plan.GetRowForFunctionOffset(9), nullptr);
  EXPECT_THAT(plan.GetRowForFunctionOffset(10), testing::Pointee(*row1_sp));
  EXPECT_THAT(plan.GetRowForFunctionOffset(19), testing::Pointee(*row1_sp));
  EXPECT_THAT(plan.GetRowForFunctionOffset(20), testing::Pointee(*row2_sp));
  EXPECT_THAT(plan.GetRowForFunctionOffset(99), testing::Pointee(*row2_sp));
}

TEST(UnwindPlan, PlanValidAtAddress) {
  UnwindPlan::RowSP row1_sp = make_simple_row(0, 42);
  UnwindPlan::RowSP row2_sp = make_simple_row(10, 47);

  UnwindPlan plan(eRegisterKindGeneric);
  EXPECT_FALSE(plan.PlanValidAtAddress(Address(0)));

  plan.InsertRow(row2_sp);
  EXPECT_FALSE(plan.PlanValidAtAddress(Address(0)));

  plan.InsertRow(row1_sp);
  EXPECT_TRUE(plan.PlanValidAtAddress(Address(0)));
  EXPECT_TRUE(plan.PlanValidAtAddress(Address(10)));

  plan.SetPlanValidAddressRanges({AddressRange(0, 5), AddressRange(15, 5)});
  EXPECT_TRUE(plan.PlanValidAtAddress(Address(0)));
  EXPECT_FALSE(plan.PlanValidAtAddress(Address(5)));
  EXPECT_FALSE(plan.PlanValidAtAddress(Address(10)));
  EXPECT_TRUE(plan.PlanValidAtAddress(Address(15)));
  EXPECT_FALSE(plan.PlanValidAtAddress(Address(20)));
  EXPECT_FALSE(plan.PlanValidAtAddress(Address(25)));
}
