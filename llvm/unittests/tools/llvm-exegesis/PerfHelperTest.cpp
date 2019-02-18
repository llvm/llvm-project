//===-- PerfHelperTest.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  const PerfEvent SingleEvent("CYCLES:u");
  const auto &EmptyFn = []() {};
  std::string CallbackEventName;
  std::string CallbackEventNameFullyQualifed;
  int64_t CallbackEventCycles;
  Measure(llvm::makeArrayRef(SingleEvent),
          [&](const PerfEvent &Event, int64_t Value) {
            CallbackEventName = Event.name();
            CallbackEventNameFullyQualifed = Event.getPfmEventString();
            CallbackEventCycles = Value;
          },
          EmptyFn);
  EXPECT_EQ(CallbackEventName, "CYCLES:u");
  EXPECT_THAT(CallbackEventNameFullyQualifed, Not(IsEmpty()));
  pfmTerminate();
#else
  ASSERT_TRUE(pfmInitialize());
#endif
}

} // namespace
} // namespace pfm
} // namespace exegesis
} // namespace llvm
