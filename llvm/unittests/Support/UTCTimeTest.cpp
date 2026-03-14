//===- unittests/Support/UTCTimeTest.cpp ----------------- ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Chrono.h"
#include "gtest/gtest.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatProviders.h"
#include "llvm/Support/FormatVariadic.h"

namespace llvm {
namespace sys {
namespace {

TEST(UTCTime, convertutc) {
  // Get the current time.
  time_t currentTime;
  time(&currentTime);

  // Convert with toUtcTime.
  SmallString<15> customResultString;
  raw_svector_ostream T(customResultString);
  T << formatv("{0:%Y-%m-%d %H:%M:%S}", llvm::sys::toUtcTime(currentTime));

  // Convert with gmtime.
  char gmtimeResultString[20];
  std::tm *gmtimeResult = std::gmtime(&currentTime);
  assert(gmtimeResult != NULL);
  std::strftime(gmtimeResultString, 20, "%Y-%m-%d %H:%M:%S", gmtimeResult);

  // Compare the formatted strings.
  EXPECT_EQ(customResultString, StringRef(gmtimeResultString, 19));

}
} // namespace
} // namespace sys
} // namespace llvm
