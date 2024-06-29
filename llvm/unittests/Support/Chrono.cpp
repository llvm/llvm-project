//===- llvm/unittest/Support/Chrono.cpp - Time utilities tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Chrono.h"
#include "llvm/Support/FormatVariadic.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::sys;
using namespace std::chrono;

namespace {

TEST(Chrono, TimeTConversion) {
  EXPECT_EQ(time_t(0), toTimeT(toTimePoint(time_t(0))));
  EXPECT_EQ(time_t(1), toTimeT(toTimePoint(time_t(1))));
  EXPECT_EQ(time_t(47), toTimeT(toTimePoint(time_t(47))));

  TimePoint<> TP;
  EXPECT_EQ(TP, toTimePoint(toTimeT(TP)));
  TP += seconds(1);
  EXPECT_EQ(TP, toTimePoint(toTimeT(TP)));
  TP += hours(47);
  EXPECT_EQ(TP, toTimePoint(toTimeT(TP)));
}

TEST(Chrono, TimePointFormat) {
  using namespace std::chrono;
  struct tm TM {};
  TM.tm_year = 106;
  TM.tm_mon = 0;
  TM.tm_mday = 2;
  TM.tm_hour = 15;
  TM.tm_min = 4;
  TM.tm_sec = 5;
  TM.tm_isdst = -1;
  TimePoint<> T =
      system_clock::from_time_t(mktime(&TM)) + nanoseconds(123456789);
  TimePoint<> T2 =
      system_clock::from_time_t(mktime(&TM)) + nanoseconds(23456789);

  // operator<< uses the format YYYY-MM-DD HH:MM:SS.NNNNNNNNN
  std::string S;
  raw_string_ostream OS(S);
  OS << T;
  EXPECT_EQ("2006-01-02 15:04:05.123456789", OS.str());
  S.clear();
  OS << T2;
  EXPECT_EQ("2006-01-02 15:04:05.023456789", OS.str());

  // formatv default style matches operator<<.
  EXPECT_EQ("2006-01-02 15:04:05.123456789", formatv("{0}", T).str());
  EXPECT_EQ("2006-01-02 15:04:05.023456789", formatv("{0}", T2).str());
  // formatv supports strftime-style format strings.
  EXPECT_EQ("15:04:05", formatv("{0:%H:%M:%S}", T).str());
  // formatv supports our strftime extensions for sub-second precision.
  EXPECT_EQ("123", formatv("{0:%L}", T).str());
  EXPECT_EQ("123456", formatv("{0:%f}", T).str());
  EXPECT_EQ("123456789", formatv("{0:%N}", T).str());
  // our extensions don't interfere with %% escaping.
  EXPECT_EQ("%foo", formatv("{0:%%foo}", T).str());
}

// Test that toTimePoint and toTimeT can be called with a arguments with varying
// precisions.
TEST(Chrono, ImplicitConversions) {
  std::time_t TimeT = 47;
  TimePoint<seconds> Sec = toTimePoint(TimeT);
  TimePoint<milliseconds> Milli = toTimePoint(TimeT);
  TimePoint<microseconds> Micro = toTimePoint(TimeT);
  TimePoint<nanoseconds> Nano = toTimePoint(TimeT);
  EXPECT_EQ(Sec, Milli);
  EXPECT_EQ(Sec, Micro);
  EXPECT_EQ(Sec, Nano);
  EXPECT_EQ(TimeT, toTimeT(Sec));
  EXPECT_EQ(TimeT, toTimeT(Milli));
  EXPECT_EQ(TimeT, toTimeT(Micro));
  EXPECT_EQ(TimeT, toTimeT(Nano));
}

TEST(Chrono, DurationFormat) {
  EXPECT_EQ("1 h", formatv("{0}", hours(1)).str());
  EXPECT_EQ("1 m", formatv("{0}", minutes(1)).str());
  EXPECT_EQ("1 s", formatv("{0}", seconds(1)).str());
  EXPECT_EQ("1 ms", formatv("{0}", milliseconds(1)).str());
  EXPECT_EQ("1 us", formatv("{0}", microseconds(1)).str());
  EXPECT_EQ("1 ns", formatv("{0}", nanoseconds(1)).str());

  EXPECT_EQ("1 s", formatv("{0:+}", seconds(1)).str());
  EXPECT_EQ("1", formatv("{0:-}", seconds(1)).str());

  EXPECT_EQ("1000 ms", formatv("{0:ms}", seconds(1)).str());
  EXPECT_EQ("1000000 us", formatv("{0:us}", seconds(1)).str());
  EXPECT_EQ("1000", formatv("{0:ms-}", seconds(1)).str());

  EXPECT_EQ("1,000 ms", formatv("{0:+n}", milliseconds(1000)).str());
  EXPECT_EQ("0x3e8", formatv("{0:-x}", milliseconds(1000)).str());
  EXPECT_EQ("010", formatv("{0:-3}", milliseconds(10)).str());
  EXPECT_EQ("10,000", formatv("{0:ms-n}", seconds(10)).str());

  EXPECT_EQ("1.00 s", formatv("{0}", duration<float>(1)).str());
  EXPECT_EQ("0.123 s", formatv("{0:+3}", duration<float>(0.123f)).str());
  EXPECT_EQ("1.230e-01 s", formatv("{0:+e3}", duration<float>(0.123f)).str());

  typedef duration<float, std::ratio<60 * 60 * 24 * 14, 1000000>>
      microfortnights;
  EXPECT_EQ("1.00", formatv("{0:-}", microfortnights(1)).str());
  EXPECT_EQ("1209.60 ms", formatv("{0:ms}", microfortnights(1)).str());
}

} // anonymous namespace
