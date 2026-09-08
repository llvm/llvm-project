//===- llvm/unittests/tools/llvm-cfi-verify/FileAnalysis.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-c/lto.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(deprecated, lto_runtime_lib_symbols_list) {
  size_t size = 0;
  const char *const *symbols = ::lto_runtime_lib_symbols_list(&size);
  (void)symbols;
  // The deprecated API always returns an empty list.
  ASSERT_EQ(size, (size_t)0);
}

TEST(valid_triples, lto_runtime_lib_symbols_list_for_triple) {
  size_t sizeA = 0;
  const char *const *symbolsA =
      ::lto_runtime_lib_symbols_list_for_triple("aarch64-gnu-linux", &sizeA);
  size_t sizeB = 0;
  const char *const *symbolsB =
      ::lto_runtime_lib_symbols_list_for_triple("x86_64-none-linux", &sizeB);
  // Valid triples return non-empty lists...
  ASSERT_NE(sizeA, (size_t)0);
  ASSERT_NE(sizeB, (size_t)0);
  // ...that, althought thet might hold the same list of symbols, are
  // stored in different places in memory.
  ASSERT_NE(symbolsA, symbolsB);
  // Re-invoking on the same triple returns the same pointer to the
  // list (the API caches the list).
  size_t sizeA2 = 0;
  ASSERT_EQ(symbolsA, ::lto_runtime_lib_symbols_list_for_triple(
                          "aarch64-gnu-linux", &sizeA2));
  ASSERT_EQ(sizeA, sizeA2);
}

TEST(invalid_triple, lto_runtime_lib_symbols_list_for_triple) {
  size_t size = 0;
  const char *const *symbols =
      ::lto_runtime_lib_symbols_list_for_triple("invalid-triple", &size);
  (void)symbols;
  ASSERT_EQ(size, (size_t)0);
}

TEST(normalised_triple, lto_runtime_lib_symbols_list_for_triple) {
  size_t sizeA = 0;
  size_t sizeB = 0;
  const char *const *symbolsA =
      ::lto_runtime_lib_symbols_list_for_triple("aarch64-apple-macho", &sizeA);
  const char *const *symbolsB =
      ::lto_runtime_lib_symbols_list_for_triple("aarch64-apple-unknown-macho", &sizeB);
  // Normalized triple have the same list, stored in the same place in
  // memory.
  ASSERT_NE(sizeA, (size_t)0);
  ASSERT_NE(sizeB, (size_t)0);
  ASSERT_EQ(sizeA, sizeB);
  ASSERT_EQ(symbolsA, symbolsB);
}

  
} // namespace
