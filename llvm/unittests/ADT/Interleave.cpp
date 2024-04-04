//===- unittests/ADT/Interleave.cpp - interleave unit tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace llvm;

TEST(InterleaveTest, Interleave) {
  std::string Str;
  raw_string_ostream OS(Str);

  // Check that interleave works on a SmallVector.
  SmallVector<const char *> Doodles = {"golden", "berna", "labra"};
  interleave(
      Doodles, OS, [&](const char *Name) { OS << Name << "doodle"; }, ", ");

  EXPECT_EQ(OS.str(), "goldendoodle, bernadoodle, labradoodle");
}

TEST(InterleaveTest, InterleaveComma) {
  std::string Str;
  raw_string_ostream OS(Str);

  // Check that interleaveComma uses ADL to find begin/end on an array.
  const StringRef LongDogs[] = {"dachshund", "doxie", "dackel", "teckel"};
  interleaveComma(LongDogs, OS);

  EXPECT_EQ(OS.str(), "dachshund, doxie, dackel, teckel");
}
