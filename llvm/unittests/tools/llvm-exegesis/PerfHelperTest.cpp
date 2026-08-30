//===-- PerfHelperTest.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PerfHelper.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#ifdef HAVE_LIBPFM
#include <perfmon/perf_event.h>
#endif

namespace llvm {
namespace exegesis {

namespace {

#ifdef HAVE_LIBPFM
TEST(RawPerfEventTest, ValidConstruction) {
  pfm::RawPerfEvent Event1(0x003c, 0x00);
  EXPECT_TRUE(Event1.valid());
  EXPECT_NE(Event1.attribute(), nullptr);
  EXPECT_EQ(Event1.name(), "raw:60:0");

  pfm::RawPerfEvent Event2(0x00c2, 0x01);
  EXPECT_TRUE(Event2.valid());
  EXPECT_NE(Event2.attribute(), nullptr);
  EXPECT_EQ(Event2.name(), "raw:194:1");

  pfm::RawPerfEvent Event3(0x0000, 0x00);
  EXPECT_TRUE(Event3.valid());
  EXPECT_NE(Event3.attribute(), nullptr);
  EXPECT_EQ(Event3.name(), "raw:0:0");
}

TEST(RawPerfEventTest, ConfigEncoding) {
  pfm::RawPerfEvent Event1(0x003c, 0x01);
  ASSERT_NE(Event1.attribute(), nullptr);
  EXPECT_EQ(Event1.attribute()->config, (0x01 << 8) | 0x003cULL);
  EXPECT_EQ(Event1.attribute()->type, PERF_TYPE_RAW);

  pfm::RawPerfEvent Event2(0x00c2, 0x01);
  ASSERT_NE(Event2.attribute(), nullptr);
  EXPECT_EQ(Event2.attribute()->config, (0x01 << 8) | 0x00c2ULL);
}

TEST(RawPerfEventTest, ExcludeFlags) {
  pfm::RawPerfEvent Event(0x003c, 0x00);
  ASSERT_NE(Event.attribute(), nullptr);
  EXPECT_EQ(Event.attribute()->exclude_kernel, 1u);
  EXPECT_EQ(Event.attribute()->exclude_hv, 1u);
}
#endif

} // namespace
} // namespace exegesis
} // namespace llvm
