//===- llvm/unittest/ADT/StringSwitchTest.cpp - StringSwitch unit tests ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringSwitch.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(StringSwitchTest, Case) {
  auto Translate = [](StringRef S) {
    return llvm::StringSwitch<int>(S)
        .Case("0", 0)
        .Case("1", 1)
        .Case("2", 2)
        .Case("3", 3)
        .Case("4", 4)
        .Case("5", 5)
        .Case("6", 6)
        .Case("7", 7)
        .Case("8", 8)
        .Case("9", 9)
        .Case("A", 10)
        .Case("B", 11)
        .Case("C", 12)
        .Case("D", 13)
        .Case("E", 14)
        .Case("F", 15)
        .Default(-1);
  };
  EXPECT_EQ(1, Translate("1"));
  EXPECT_EQ(2, Translate("2"));
  EXPECT_EQ(11, Translate("B"));
  EXPECT_EQ(-1, Translate("b"));
  EXPECT_EQ(-1, Translate(""));
  EXPECT_EQ(-1, Translate("Test"));
}

TEST(StringSwitchTest, StartsWith) {
  auto Translate = [](StringRef S) {
    return llvm::StringSwitch<std::function<int(int, int)>>(S)
        .StartsWith("add", [](int X, int Y) { return X + Y; })
        .StartsWith("sub", [](int X, int Y) { return X - Y; })
        .StartsWith("mul", [](int X, int Y) { return X * Y; })
        .StartsWith("div", [](int X, int Y) { return X / Y; })
        .Default([](int X, int Y) { return 0; });
  };

  EXPECT_EQ(15, Translate("adder")(10, 5));
  EXPECT_EQ(5, Translate("subtracter")(10, 5));
  EXPECT_EQ(50, Translate("multiplier")(10, 5));
  EXPECT_EQ(2, Translate("divider")(10, 5));

  EXPECT_EQ(0, Translate("nothing")(10, 5));
  EXPECT_EQ(0, Translate("ADDER")(10, 5));
}

TEST(StringSwitchTest, EndsWith) {
  enum class Suffix { Possible, PastTense, Process, InProgressAction, Unknown };

  auto Translate = [](StringRef S) {
    return llvm::StringSwitch<Suffix>(S)
        .EndsWith("able", Suffix::Possible)
        .EndsWith("ed", Suffix::PastTense)
        .EndsWith("ation", Suffix::Process)
        .EndsWith("ing", Suffix::InProgressAction)
        .Default(Suffix::Unknown);
  };

  EXPECT_EQ(Suffix::Possible, Translate("optimizable"));
  EXPECT_EQ(Suffix::PastTense, Translate("optimized"));
  EXPECT_EQ(Suffix::Process, Translate("optimization"));
  EXPECT_EQ(Suffix::InProgressAction, Translate("optimizing"));
  EXPECT_EQ(Suffix::Unknown, Translate("optimizer"));
  EXPECT_EQ(Suffix::Unknown, Translate("OPTIMIZABLE"));
}

TEST(StringSwitchTest, Cases) {
  enum class OSType { Windows, Linux, Unknown };

  auto Translate = [](StringRef S) {
    return llvm::StringSwitch<OSType>(S)
        .Cases("wind\0ws", "win32", "winnt", OSType::Windows)
        .Cases("linux", "unix", "*nix", "posix", OSType::Linux)
        .Default(OSType::Unknown);
  };

  EXPECT_EQ(OSType::Windows, Translate(llvm::StringRef("wind\0ws", 7)));
  EXPECT_EQ(OSType::Windows, Translate("win32"));
  EXPECT_EQ(OSType::Windows, Translate("winnt"));

  EXPECT_EQ(OSType::Linux, Translate("linux"));
  EXPECT_EQ(OSType::Linux, Translate("unix"));
  EXPECT_EQ(OSType::Linux, Translate("*nix"));
  EXPECT_EQ(OSType::Linux, Translate("posix"));

  // Note that the whole null-terminator embedded string is required for the
  // case to match.
  EXPECT_EQ(OSType::Unknown, Translate("wind"));
  EXPECT_EQ(OSType::Unknown, Translate("Windows"));
  EXPECT_EQ(OSType::Unknown, Translate(""));
}
