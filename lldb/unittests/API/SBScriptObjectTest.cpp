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

using namespace lldb;
using namespace lldb_private;

class SBScriptObjectTest : public SBDebuggerFixture {};

TEST_F(SBScriptObjectTest, Inequality) {
  int pointee = 0;
  int other_pointee = 0;
  SBScriptObject object(&pointee, eScriptLanguagePython);
  SBScriptObject copy = object;
  SBScriptObject other_pointer(&other_pointee, eScriptLanguagePython);
  SBScriptObject other_language(&pointee, eScriptLanguageLua);

  EXPECT_FALSE(object != object);
  EXPECT_FALSE(object != copy);
  EXPECT_TRUE(object != other_pointer);
  EXPECT_TRUE(object != other_language);
}
