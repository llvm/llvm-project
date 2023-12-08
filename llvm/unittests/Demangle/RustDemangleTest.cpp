//===------------------ RustDemangleTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Demangle/Demangle.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cstdlib>

TEST(RustDemangle, Success) {
  char *Demangled = llvm::rustDemangle("_RNvC1a4main");
  EXPECT_STREQ(Demangled, "a::main");
  std::free(Demangled);
}

TEST(RustDemangle, Invalid) {
  char *Demangled = nullptr;

  // Invalid prefix.
  Demangled = llvm::rustDemangle("_ABCDEF");
  EXPECT_EQ(Demangled, nullptr);

  // Correct prefix but still invalid.
  Demangled = llvm::rustDemangle("_RRR");
  EXPECT_EQ(Demangled, nullptr);
}
