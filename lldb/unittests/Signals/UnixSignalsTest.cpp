//===-- UnixSignalsTest.cpp -----------------------------------------------===//
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

class TestSignals : public UnixSignals {
public:
  TestSignals() {
    m_signals.clear();
    AddSignal(2, "SIG2", false, true, true, "DESC2");
    AddSignal(4, "SIG4", true, false, true, "DESC4");
    AddSignal(8, "SIG8", true, true, true, "DESC8");
    AddSignal(16, "SIG16", true, false, false, "DESC16");
    AddSignalCode(16, 1, "a specific type of SIG16");
    AddSignalCode(16, 2, "SIG16 with a fault address",
                  SignalCodePrintOption::Address);
    AddSignalCode(16, 3, "bounds violation", SignalCodePrintOption::Bounds);
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

TEST(UnixSignalsTest, Reset) {
  TestSignals signals;
  bool stop_val     = signals.GetShouldStop(2);
  bool notify_val   = signals.GetShouldNotify(2);
  bool suppress_val = signals.GetShouldSuppress(2);
  
  // Change two, then reset one and make sure only that one was reset:
  EXPECT_EQ(true, signals.SetShouldNotify(2, !notify_val));
  EXPECT_EQ(true, signals.SetShouldSuppress(2, !suppress_val));
  EXPECT_EQ(true, signals.ResetSignal(2, false, true, false));
  EXPECT_EQ(stop_val, signals.GetShouldStop(2));
  EXPECT_EQ(notify_val, signals.GetShouldStop(2));
  EXPECT_EQ(!suppress_val, signals.GetShouldNotify(2));
  
  // Make sure reset with no arguments resets them all:
  EXPECT_EQ(true, signals.SetShouldSuppress(2, !suppress_val));
  EXPECT_EQ(true, signals.SetShouldNotify(2, !notify_val));
  EXPECT_EQ(true, signals.ResetSignal(2));
  EXPECT_EQ(stop_val, signals.GetShouldStop(2));
  EXPECT_EQ(notify_val, signals.GetShouldNotify(2));
  EXPECT_EQ(suppress_val, signals.GetShouldSuppress(2));
}

TEST(UnixSignalsTest, GetInfo) {
  TestSignals signals;

  bool should_suppress = false, should_stop = false, should_notify = false;
  int32_t signo = 4;
  bool success =
      signals.GetSignalInfo(signo, should_suppress, should_stop, should_notify);
  ASSERT_TRUE(success);
  EXPECT_EQ(true, should_suppress);
  EXPECT_EQ(false, should_stop);
  EXPECT_EQ(true, should_notify);

  EXPECT_EQ(true, signals.GetShouldSuppress(signo));
  EXPECT_EQ(false, signals.GetShouldStop(signo));
  EXPECT_EQ(true, signals.GetShouldNotify(signo));
}

TEST(UnixSignalsTest, GetAsStringRef) {
  TestSignals signals;

  ASSERT_EQ(llvm::StringRef(), signals.GetSignalAsStringRef(100));
  ASSERT_EQ("SIG16", signals.GetSignalAsStringRef(16));
}

TEST(UnixSignalsTest, GetAsString) {
  TestSignals signals;

  ASSERT_EQ("", signals.GetSignalDescription(100, std::nullopt));
  ASSERT_EQ("SIG16", signals.GetSignalDescription(16, std::nullopt));
  ASSERT_EQ("", signals.GetSignalDescription(100, 100));
  ASSERT_EQ("SIG16", signals.GetSignalDescription(16, 100));
  ASSERT_EQ("SIG16: a specific type of SIG16",
            signals.GetSignalDescription(16, 1));

  // Unknown code, won't use the address.
  ASSERT_EQ("SIG16", signals.GetSignalDescription(16, 100, 0xCAFEF00D));
  // Known code, that shouldn't print fault address.
  ASSERT_EQ("SIG16: a specific type of SIG16",
            signals.GetSignalDescription(16, 1, 0xCAFEF00D));
  // Known code that should.
  ASSERT_EQ("SIG16: SIG16 with a fault address (fault address: 0xcafef00d)",
            signals.GetSignalDescription(16, 2, 0xCAFEF00D));
  // No address given just print the code description.
  ASSERT_EQ("SIG16: SIG16 with a fault address",
            signals.GetSignalDescription(16, 2));

  const char *expected = "SIG16: bounds violation";
  // Must pass all needed info to get full output.
  ASSERT_EQ(expected, signals.GetSignalDescription(16, 3));
  ASSERT_EQ(expected, signals.GetSignalDescription(16, 3, 0xcafef00d));
  ASSERT_EQ(expected, signals.GetSignalDescription(16, 3, 0xcafef00d, 0x1234));

  ASSERT_EQ("SIG16: upper bound violation (fault address: 0x5679, lower bound: "
            "0x1234, upper bound: 0x5678)",
            signals.GetSignalDescription(16, 3, 0x5679, 0x1234, 0x5678));
  ASSERT_EQ("SIG16: lower bound violation (fault address: 0x1233, lower bound: "
            "0x1234, upper bound: 0x5678)",
            signals.GetSignalDescription(16, 3, 0x1233, 0x1234, 0x5678));
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

  auto all_signals =
      signals.GetFilteredSignals(std::nullopt, std::nullopt, std::nullopt);
  std::vector<int32_t> expected = {2, 4, 8, 16};
  EXPECT_EQ_ARRAYS(expected, all_signals);

  auto supressed = signals.GetFilteredSignals(true, std::nullopt, std::nullopt);
  expected = {4, 8, 16};
  EXPECT_EQ_ARRAYS(expected, supressed);

  auto not_supressed =
      signals.GetFilteredSignals(false, std::nullopt, std::nullopt);
  expected = {2};
  EXPECT_EQ_ARRAYS(expected, not_supressed);

  auto stopped = signals.GetFilteredSignals(std::nullopt, true, std::nullopt);
  expected = {2, 8};
  EXPECT_EQ_ARRAYS(expected, stopped);

  auto not_stopped =
      signals.GetFilteredSignals(std::nullopt, false, std::nullopt);
  expected = {4, 16};
  EXPECT_EQ_ARRAYS(expected, not_stopped);

  auto notified = signals.GetFilteredSignals(std::nullopt, std::nullopt, true);
  expected = {2, 4, 8};
  EXPECT_EQ_ARRAYS(expected, notified);

  auto not_notified =
      signals.GetFilteredSignals(std::nullopt, std::nullopt, false);
  expected = {16};
  EXPECT_EQ_ARRAYS(expected, not_notified);

  auto signal4 = signals.GetFilteredSignals(true, false, true);
  expected = {4};
  EXPECT_EQ_ARRAYS(expected, signal4);
}
