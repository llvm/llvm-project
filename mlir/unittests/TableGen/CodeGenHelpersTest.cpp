//===- CodeGenHelpersTest.cpp - TableGen CodeGenHelpers Utility Tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/Format.h"
#include "gmock/gmock.h"

using mlir::tblgen::buildErrorStreamingString;
using mlir::tblgen::ErrorStreamType;
using mlir::tblgen::FmtContext;
using ::testing::StrEq;

TEST(CodeGenHelpersTest, BuildErrorStreamingString) {
  FmtContext ctx;
  ctx.withSelf("this_thing");
  std::string result1 =
      buildErrorStreamingString("here {{reformat($_self)}} is block", ctx,
                                ErrorStreamType::InsideOpError);
  EXPECT_THAT(result1,
              StrEq("here \") << reformat(this_thing) << (\" is block"));
  std::string result2 = buildErrorStreamingString(
      "here {{reformat($_self)}} is block", ctx, ErrorStreamType::InString);
  EXPECT_THAT(result2, StrEq("here \" << reformat(this_thing) << \" is block"));
}

