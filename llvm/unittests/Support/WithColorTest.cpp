//===- WithColorTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/WithColor.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(WithColorTest, ColorMode) {
  {
    std::string S;
    llvm::raw_string_ostream OS(S);
    OS.enable_colors(true);

    WithColor(OS, HighlightColor::Error, ColorMode::Disable) << "test";
    EXPECT_EQ("test", OS.str());
  }

  {
    std::string S;
    llvm::raw_string_ostream OS(S);
    OS.enable_colors(true);

    WithColor(OS, HighlightColor::Error, ColorMode::Auto) << "test";
    EXPECT_EQ("test", OS.str());
  }

#ifdef LLVM_ON_UNIX
  {
    std::string S;
    llvm::raw_string_ostream OS(S);
    OS.enable_colors(true);

    WithColor(OS, HighlightColor::Error, ColorMode::Enable) << "test";
    EXPECT_EQ("\x1B[0;1;31mtest\x1B[0m", OS.str());
  }
#endif
}
