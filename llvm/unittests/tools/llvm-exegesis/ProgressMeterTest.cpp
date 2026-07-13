//===-- ProgressMeterTest.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProgressMeter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace exegesis {

namespace {

struct PreprogrammedClock {
  using duration = std::chrono::seconds;
  using rep = duration::rep;
  using period = duration::period;
  using time_point = std::chrono::time_point<PreprogrammedClock, duration>;

  static constexpr bool is_steady = true;

  static time_point now() noexcept;
};

static int CurrentTimePoint = 0;
auto Pt(int i) {
  return PreprogrammedClock::time_point(PreprogrammedClock::duration(i));
}
static const std::array<const PreprogrammedClock::time_point, 10> TimePoints = {
    Pt(0),  Pt(5),  Pt(6),  Pt(20), Pt(20),
    Pt(35), Pt(37), Pt(75), Pt(77), Pt(100)};

PreprogrammedClock::time_point PreprogrammedClock::now() noexcept {
  time_point p = TimePoints[CurrentTimePoint];
  ++CurrentTimePoint;
  return p;
}

TEST(ProgressMeterTest, Integration) {
  CurrentTimePoint = 0;
  std::string TempString;
  raw_string_ostream SS(TempString);
  ProgressMeter<PreprogrammedClock> m(5, SS);
  for (int i = 0; i != 5; ++i)
    decltype(m)::ProgressMeterStep s(&m);
  ASSERT_EQ("Processing...  20%, ETA 00:20\n"
            "Processing...  40%, ETA 00:29\n"
            "Processing...  60%, ETA 00:23\n"
            "Processing...  80%, ETA 00:18\n"
            "Processing... 100%, ETA 00:00\n",
            TempString);
}

} // namespace
} // namespace exegesis
} // namespace llvm
