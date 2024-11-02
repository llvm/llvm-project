//===- llvm/unittest/Support/ModRefTest.cpp - ModRef tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ModRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <string>

using namespace llvm;

namespace {

// Verify that printing a MemoryEffects does not end with a ,.
TEST(ModRefTest, PrintMemoryEffects) {
  std::string S;
  raw_string_ostream OS(S);
  OS << MemoryEffects::none();
  OS.flush();
  EXPECT_EQ(S, "ArgMem: NoModRef, InaccessibleMem: NoModRef, Other: NoModRef");
}

} // namespace
