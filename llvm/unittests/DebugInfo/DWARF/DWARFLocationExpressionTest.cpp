//===- llvm/unittest/DebugInfo/DWARFLocationExpressionTest.cpp ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFLocationExpression.h"
#include "llvm/Support/ScopedPrinter.h"
#include "gtest/gtest.h"

using namespace llvm;
using object::SectionedAddress;

TEST(DWARFLocationExpression, Equality) {
  EXPECT_EQ((DWARFLocationExpression{None, {}}),
            (DWARFLocationExpression{None, {}}));
  EXPECT_NE((DWARFLocationExpression{DWARFAddressRange{1, 47}, {}}),
            (DWARFLocationExpression{DWARFAddressRange{1, 48}, {}}));
  EXPECT_NE((DWARFLocationExpression{DWARFAddressRange{1, 47}, {}}),
            (DWARFLocationExpression{DWARFAddressRange{1, 47}, {42}}));
}

TEST(DWARFLocationExpression, StreamingOperator) {
  EXPECT_EQ("None: 1, 2", to_string(DWARFLocationExpression{None, {1, 2}}));
  EXPECT_EQ(
      "[0x0000000000000042, 0x0000000000000047): 1",
      to_string(DWARFLocationExpression{DWARFAddressRange{0x42, 0x47}, {1}}));
}
