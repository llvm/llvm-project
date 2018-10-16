//===-- PredicateTest.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Predicate.h"
#include "gtest/gtest.h"
#include <thread>

using namespace lldb_private;

TEST(Predicate, WaitForValueEqualTo) {
  Predicate<int> P(0);
  EXPECT_TRUE(P.WaitForValueEqualTo(0));
  EXPECT_FALSE(P.WaitForValueEqualTo(1, std::chrono::milliseconds(10)));

  std::thread Setter([&P] {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    P.SetValue(1, eBroadcastAlways);
  });
  EXPECT_TRUE(P.WaitForValueEqualTo(1));
  Setter.join();
}

TEST(Predicate, WaitForValueNotEqualTo) {
  Predicate<int> P(0);
  EXPECT_EQ(0, P.WaitForValueNotEqualTo(1));
  EXPECT_EQ(llvm::None,
            P.WaitForValueNotEqualTo(0, std::chrono::milliseconds(10)));
}
