//===-- PerfHelperTest.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PerfHelper.h"
#include "llvm/Config/config.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace exegesis {
namespace pfm {
namespace {

using ::testing::IsEmpty;
using ::testing::Not;

TEST(PerfHelperTest, FunctionalTest) {
#ifdef HAVE_LIBPFM
  ASSERT_FALSE(pfmInitialize());
  PerfEvent Event("CYCLES:u");
  ASSERT_TRUE(Event.valid());
  EXPECT_EQ(Event.name(), "CYCLES:u");
  EXPECT_THAT(Event.getPfmEventString(), Not(IsEmpty()));
  Counter Cnt(std::move(Event));
  Cnt.start();
  Cnt.stop();
  Cnt.read();
  pfmTerminate();
#else
  ASSERT_TRUE(pfmInitialize());
#endif
}

} // namespace
} // namespace pfm
} // namespace exegesis
} // namespace llvm
