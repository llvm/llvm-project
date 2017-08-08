//===- llvm/unittest/Support/Chrono.cpp - Time utilities tests ------------===//
//
//           The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Chrono.h"
#include "llvm/ADT/SmallVector.h"
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

TEST(Chrono, StringConversion) {
  std::string S;
  raw_string_ostream OS(S);
  OS << system_clock::now();

  // Do a basic sanity check on the output.
  // The format we expect is YYYY-MM-DD HH:MM:SS.MMMUUUNNN
  StringRef Date, Time;
  std::tie(Date, Time) = StringRef(OS.str()).split(' ');

  SmallVector<StringRef, 3> Components;
  Date.split(Components, '-');
  ASSERT_EQ(3u, Components.size());
  EXPECT_EQ(4u, Components[0].size());
  EXPECT_EQ(2u, Components[1].size());
  EXPECT_EQ(2u, Components[2].size());

  StringRef Sec, Nano;
  std::tie(Sec, Nano) = Time.split('.');

  Components.clear();
  Sec.split(Components, ':');
  ASSERT_EQ(3u, Components.size());
  EXPECT_EQ(2u, Components[0].size());
  EXPECT_EQ(2u, Components[1].size());
  EXPECT_EQ(2u, Components[2].size());
  EXPECT_EQ(9u, Nano.size());
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
