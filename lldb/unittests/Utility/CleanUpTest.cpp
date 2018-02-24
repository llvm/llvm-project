//===-- CleanUpTest.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/CleanUp.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(CleanUpTest, no_args) {
  bool f = false;
  {
    CleanUp cleanup([&] { f = true; });
  }
  ASSERT_TRUE(f);
}

TEST(CleanUpTest, multiple_args) {
  bool f1 = false;
  bool f2 = false;
  bool f3 = false;
  {
    CleanUp cleanup(
        [](bool arg1, bool *arg2, bool &arg3) {
          ASSERT_FALSE(arg1);
          *arg2 = true;
          arg3 = true;
        },
        f1, &f2, f3);
  }
  ASSERT_TRUE(f2);
  ASSERT_FALSE(f3);
}

TEST(CleanUpTest, disable) {
  bool f = false;
  {
    CleanUp cleanup([&] { f = true; });
    cleanup.disable();
  }
  ASSERT_FALSE(f);
}
