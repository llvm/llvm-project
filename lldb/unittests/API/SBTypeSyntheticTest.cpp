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

class SBTypeSyntheticTest : public SBDebuggerFixture {};

TEST_F(SBTypeSyntheticTest, IsEqualToInvalid) {
  SBTypeSynthetic synthetic =
      SBTypeSynthetic::CreateWithClassName("module.Class");
  SBTypeSynthetic invalid;
  EXPECT_FALSE(synthetic.IsEqualTo(invalid));
}
