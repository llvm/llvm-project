//===- unittest/Support/RemarksAPITest.cpp - C++ API tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Remarks/Remark.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(RemarksAPI, Comparison) {
  remarks::Remark R;
  R.RemarkType = remarks::Type::Missed;
  R.PassName = "pass";
  R.RemarkName = "name";
  R.FunctionName = "func";
  R.Loc = remarks::RemarkLocation{"path", 3, 4};
  R.Hotness = 5;
  R.Args.emplace_back();
  R.Args.back().Key = "key";
  R.Args.back().Val = "value";
  R.Args.emplace_back();
  R.Args.back().Key = "keydebug";
  R.Args.back().Val = "valuedebug";
  R.Args.back().Loc = remarks::RemarkLocation{"argpath", 6, 7};

  // Check that == works.
  EXPECT_EQ(R, R);

  // Check that != works.
  remarks::Remark R2 = R.clone();
  R2.FunctionName = "func0";
  EXPECT_NE(R, R2);

  // Check that we iterate through all the arguments.
  remarks::Remark R3 = R.clone();
  R3.Args.back().Val = "not";
  EXPECT_NE(R, R3);
}

TEST(RemarksAPI, Clone) {
  remarks::Remark R;
  R.RemarkType = remarks::Type::Missed;
  R.PassName = "pass";
  R.RemarkName = "name";
  R.FunctionName = "func";
  R.Loc = remarks::RemarkLocation{"path", 3, 4};
  R.Hotness = 5;
  R.Args.emplace_back();
  R.Args.back().Key = "key";
  R.Args.back().Val = "value";
  R.Args.emplace_back();
  R.Args.back().Key = "keydebug";
  R.Args.back().Val = "valuedebug";
  R.Args.back().Loc = remarks::RemarkLocation{"argpath", 6, 7};

  // Check that clone works.
  remarks::Remark R2 = R.clone();
  EXPECT_EQ(R, R2);
}

TEST(RemarksAPI, ArgsAsMsg) {
  remarks::Remark R;
  R.RemarkType = remarks::Type::Missed;
  R.Args.emplace_back();
  R.Args.back().Key = "key";
  R.Args.back().Val = "can not do this ";
  R.Args.emplace_back();
  R.Args.back().Key = "keydebug";
  R.Args.back().Val = "because of that.";
  R.Args.back().Loc = remarks::RemarkLocation{"argpath", 6, 7};

  EXPECT_EQ(R.getArgsAsMsg(), "can not do this because of that.");
}
