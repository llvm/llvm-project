//===- unittest/Support/BitstreamRemarksFormatTest.cpp - BitCodes tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Remarks/BitstreamRemarkContainer.h"
#include "gtest/gtest.h"

using namespace llvm;

// The goal for this test is to observe test failures and carefully update the
// constants when they change.

// This should not change over time.
TEST(BitstreamRemarksFormat, Magic) {
  EXPECT_EQ(remarks::ContainerMagic, "RMRK");
}

// This should be updated whenever any of the tests below are modified.
TEST(BitstreamRemarksFormat, ContainerVersion) {
  EXPECT_EQ(remarks::CurrentContainerVersion, 0UL);
}

// The values of the current blocks should not change over time.
// When adding new blocks, make sure to append them to the enum.
TEST(BitstreamRemarksFormat, BlockIDs) {
  EXPECT_EQ(remarks::META_BLOCK_ID, 8);
  EXPECT_EQ(remarks::REMARK_BLOCK_ID, 9);
}

// The values of the current records should not change over time.
// When adding new records, make sure to append them to the enum.
TEST(BitstreamRemarksFormat, RecordIDs) {
  EXPECT_EQ(remarks::RECORD_FIRST, 1);
  EXPECT_EQ(remarks::RECORD_META_CONTAINER_INFO, 1);
  EXPECT_EQ(remarks::RECORD_META_REMARK_VERSION, 2);
  EXPECT_EQ(remarks::RECORD_META_STRTAB, 3);
  EXPECT_EQ(remarks::RECORD_META_EXTERNAL_FILE, 4);
  EXPECT_EQ(remarks::RECORD_REMARK_HEADER, 5);
  EXPECT_EQ(remarks::RECORD_REMARK_DEBUG_LOC, 6);
  EXPECT_EQ(remarks::RECORD_REMARK_HOTNESS, 7);
  EXPECT_EQ(remarks::RECORD_REMARK_ARG_WITH_DEBUGLOC, 8);
  EXPECT_EQ(remarks::RECORD_REMARK_ARG_WITHOUT_DEBUGLOC, 9);
  EXPECT_EQ(remarks::RECORD_LAST, 9);
}
