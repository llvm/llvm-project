//===-- UnixSignalsTest.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <string>

#include "gtest/gtest.h"

#include "lldb/Target/UnixSignals.h"
#include "llvm/Support/FormatVariadic.h"

using namespace lldb;
using namespace lldb_private;
using llvm::None;

class TestSignals : public UnixSignals {
public:
  TestSignals() {
    m_signals.clear();
    AddSignal(2, "SIG2", false, true, true, "DESC2");
    AddSignal(4, "SIG4", true, false, true, "DESC4");
    AddSignal(8, "SIG8", true, true, true, "DESC8");
    AddSignal(16, "SIG16", true, false, false, "DESC16");
  }
};

void ExpectEqArrays(llvm::ArrayRef<int32_t> expected,
                    llvm::ArrayRef<int32_t> observed, const char *file,
                    int line) {
  std::string location = llvm::formatv("{0}:{1}", file, line);
  ASSERT_EQ(expected.size(), observed.size()) << location;

  for (size_t i = 0; i < observed.size(); ++i) {
    ASSERT_EQ(expected[i], observed[i])
        << "array index: " << i << "location:" << location;
  }
}

#define EXPECT_EQ_ARRAYS(expected, observed)                                   \
  ExpectEqArrays((expected), (observed), __FILE__, __LINE__);

TEST(UnixSignalsTest, Iteration) {
  TestSignals signals;

  EXPECT_EQ(4, signals.GetNumSignals());
  EXPECT_EQ(2, signals.GetFirstSignalNumber());
  EXPECT_EQ(4, signals.GetNextSignalNumber(2));
  EXPECT_EQ(8, signals.GetNextSignalNumber(4));
  EXPECT_EQ(16, signals.GetNextSignalNumber(8));
  EXPECT_EQ(LLDB_INVALID_SIGNAL_NUMBER, signals.GetNextSignalNumber(16));
}

TEST(UnixSignalsTest, GetInfo) {
  TestSignals signals;

  bool should_suppress = false, should_stop = false, should_notify = false;
  int32_t signo = 4;
  std::string name =
      signals.GetSignalInfo(signo, should_suppress, should_stop, should_notify);
  EXPECT_EQ("SIG4", name);
  EXPECT_EQ(true, should_suppress);
  EXPECT_EQ(false, should_stop);
  EXPECT_EQ(true, should_notify);

  EXPECT_EQ(true, signals.GetShouldSuppress(signo));
  EXPECT_EQ(false, signals.GetShouldStop(signo));
  EXPECT_EQ(true, signals.GetShouldNotify(signo));
  EXPECT_EQ(name, signals.GetSignalAsCString(signo));
}

TEST(UnixSignalsTest, VersionChange) {
  TestSignals signals;

  int32_t signo = 8;
  uint64_t ver = signals.GetVersion();
  EXPECT_GT(ver, 0ull);
  EXPECT_EQ(true, signals.GetShouldSuppress(signo));
  EXPECT_EQ(true, signals.GetShouldStop(signo));
  EXPECT_EQ(true, signals.GetShouldNotify(signo));

  EXPECT_EQ(signals.GetVersion(), ver);

  signals.SetShouldSuppress(signo, false);
  EXPECT_LT(ver, signals.GetVersion());
  ver = signals.GetVersion();

  signals.SetShouldStop(signo, true);
  EXPECT_LT(ver, signals.GetVersion());
  ver = signals.GetVersion();

  signals.SetShouldNotify(signo, false);
  EXPECT_LT(ver, signals.GetVersion());
  ver = signals.GetVersion();

  EXPECT_EQ(false, signals.GetShouldSuppress(signo));
  EXPECT_EQ(true, signals.GetShouldStop(signo));
  EXPECT_EQ(false, signals.GetShouldNotify(signo));

  EXPECT_EQ(ver, signals.GetVersion());
}

TEST(UnixSignalsTest, GetFilteredSignals) {
  TestSignals signals;

  auto all_signals = signals.GetFilteredSignals(None, None, None);
  std::vector<int32_t> expected = {2, 4, 8, 16};
  EXPECT_EQ_ARRAYS(expected, all_signals);

  auto supressed = signals.GetFilteredSignals(true, None, None);
  expected = {4, 8, 16};
  EXPECT_EQ_ARRAYS(expected, supressed);

  auto not_supressed = signals.GetFilteredSignals(false, None, None);
  expected = {2};
  EXPECT_EQ_ARRAYS(expected, not_supressed);

  auto stopped = signals.GetFilteredSignals(None, true, None);
  expected = {2, 8};
  EXPECT_EQ_ARRAYS(expected, stopped);

  auto not_stopped = signals.GetFilteredSignals(None, false, None);
  expected = {4, 16};
  EXPECT_EQ_ARRAYS(expected, not_stopped);

  auto notified = signals.GetFilteredSignals(None, None, true);
  expected = {2, 4, 8};
  EXPECT_EQ_ARRAYS(expected, notified);

  auto not_notified = signals.GetFilteredSignals(None, None, false);
  expected = {16};
  EXPECT_EQ_ARRAYS(expected, not_notified);

  auto signal4 = signals.GetFilteredSignals(true, false, true);
  expected = {4};
  EXPECT_EQ_ARRAYS(expected, signal4);
}
