//===- SymbolSizeTest.cpp - Tests for SymbolSize.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/SymbolSize.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

TEST(Object, SymbolSizeSort) {
  auto it = symbol_iterator(SymbolRef());
  std::vector<SymEntry> Syms{
      SymEntry{it, 0xffffffff00000000ull, 1, 0},
      SymEntry{it, 0x00ffffff00000000ull, 2, 0},
      SymEntry{it, 0x00ffffff000000ffull, 3, 0},
      SymEntry{it, 0x0000000100000000ull, 4, 0},
      SymEntry{it, 0x00000000000000ffull, 5, 0},
      SymEntry{it, 0x00000001000000ffull, 6, 0},
      SymEntry{it, 0x000000010000ffffull, 7, 0},
  };

  array_pod_sort(Syms.begin(), Syms.end(), compareAddress);

  for (unsigned I = 0, N = Syms.size(); I < N - 1; ++I) {
    EXPECT_LE(Syms[I].Address, Syms[I + 1].Address);
  }
}
