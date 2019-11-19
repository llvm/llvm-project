//===- ObjectFileTest.cpp - Tests for ObjectFile.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/ScopedPrinter.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

TEST(SectionedAddress, StreamingOperator) {
  EXPECT_EQ("SectionedAddress{0x00000047}", to_string(SectionedAddress{0x47}));
  EXPECT_EQ("SectionedAddress{0x00000047, 42}",
            to_string(SectionedAddress{0x47, 42}));
}
