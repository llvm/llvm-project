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
  char *Demangled =
      llvm::rustDemangle("_RNvC1a4main", nullptr, nullptr, nullptr);
  EXPECT_STREQ(Demangled, "a::main");
  std::free(Demangled);

  // With status.
  int Status = 0;
  Demangled = llvm::rustDemangle("_RNvC1a4main", nullptr, nullptr, &Status);
  EXPECT_EQ(Status, llvm::demangle_success);
  EXPECT_STREQ(Demangled, "a::main");
  std::free(Demangled);

  // With status and length.
  size_t N = 0;
  Demangled = llvm::rustDemangle("_RNvC1a4main", nullptr, &N, &Status);
  EXPECT_EQ(Status, llvm::demangle_success);
  EXPECT_EQ(N, 8u);
  EXPECT_STREQ(Demangled, "a::main");
  std::free(Demangled);
}

TEST(RustDemangle, Invalid) {
  int Status = 0;
  char *Demangled = nullptr;

  // Invalid prefix.
  Demangled = llvm::rustDemangle("_ABCDEF", nullptr, nullptr, &Status);
  EXPECT_EQ(Status, llvm::demangle_invalid_mangled_name);
  EXPECT_EQ(Demangled, nullptr);

  // Correct prefix but still invalid.
  Demangled = llvm::rustDemangle("_RRR", nullptr, nullptr, &Status);
  EXPECT_EQ(Status, llvm::demangle_invalid_mangled_name);
  EXPECT_EQ(Demangled, nullptr);
}

TEST(RustDemangle, OutputBufferWithoutLength) {
  char *Buffer = static_cast<char *>(std::malloc(1024));
  ASSERT_NE(Buffer, nullptr);

  int Status = 0;
  char *Demangled =
      llvm::rustDemangle("_RNvC1a4main", Buffer, nullptr, &Status);

  EXPECT_EQ(Status, llvm::demangle_invalid_args);
  EXPECT_EQ(Demangled, nullptr);
  std::free(Buffer);
}

TEST(RustDemangle, OutputBuffer) {
  size_t N = 1024;
  char *Buffer = static_cast<char *>(std::malloc(N));
  ASSERT_NE(Buffer, nullptr);

  int Status = 0;
  char *Demangled = llvm::rustDemangle("_RNvC1a4main", Buffer, &N, &Status);

  EXPECT_EQ(Status, llvm::demangle_success);
  EXPECT_EQ(Demangled, Buffer);
  EXPECT_STREQ(Demangled, "a::main");
  std::free(Demangled);
}

TEST(RustDemangle, SmallOutputBuffer) {
  size_t N = 1;
  char *Buffer = static_cast<char *>(std::malloc(N));
  ASSERT_NE(Buffer, nullptr);

  int Status = 0;
  char *Demangled = llvm::rustDemangle("_RNvC1a4main", Buffer, &N, &Status);

  EXPECT_EQ(Status, llvm::demangle_success);
  EXPECT_STREQ(Demangled, "a::main");
  std::free(Demangled);
}
