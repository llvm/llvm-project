//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"

#include "SBTestingSupport/SBTestUtilities.h"
#include "gtest/gtest.h"
#include <string>

using namespace lldb;
using namespace lldb_private;

class SBCommandReturnObjectTest : public SBDebuggerFixture {};

TEST_F(SBCommandReturnObjectTest, SetErrorKeepsTheErrorMessage) {
  SBCommandReturnObject result;
  SBError error;
  error.SetErrorString("boom");
  result.SetError(error, "fallback");
  EXPECT_FALSE(result.Succeeded());
  EXPECT_NE(std::string(result.GetError()).find("boom"), std::string::npos);
  EXPECT_EQ(std::string(result.GetError()).find("fallback"), std::string::npos);
}

TEST_F(SBCommandReturnObjectTest, SetErrorIgnoresSuccess) {
  SBCommandReturnObject result;
  SBError success("cleared");
  success.Clear();
  ASSERT_TRUE(success.IsValid());
  ASSERT_TRUE(success.Success());
  result.SetStatus(eReturnStatusSuccessContinuingNoResult);
  result.SetError(success, "fallback");
  EXPECT_EQ(result.GetStatus(), eReturnStatusSuccessContinuingNoResult);
  EXPECT_EQ(result.GetErrorSize(), 0u);
}

TEST_F(SBCommandReturnObjectTest, SetErrorSuccessResetsFailure) {
  SBCommandReturnObject result;
  SBError failure("boom");
  result.SetError(failure);
  ASSERT_FALSE(result.Succeeded());

  SBError success("cleared");
  success.Clear();
  result.SetError(success);
  EXPECT_EQ(result.GetStatus(), eReturnStatusSuccessFinishResult);
  EXPECT_NE(std::string(result.GetError()).find("boom"), std::string::npos);
}
