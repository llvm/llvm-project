//===- SymbolicFileTest.cpp - Tests for SymbolicFile.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "gtest/gtest.h"
#include <sstream>

TEST(Object, DataRefImplOstream) {
  std::string s;
  llvm::raw_string_ostream OS(s);
  llvm::object::DataRefImpl Data;
  Data.d.a = 0xeeee0000;
  Data.d.b = 0x0000ffff;

  static_assert(sizeof Data.p == sizeof(uint64_t) ||
                    sizeof Data.p == sizeof(uint32_t),
                "Test expected pointer type to be 32 or 64-bit.");

  char const *Expected;

  if (sizeof Data.p == sizeof(uint64_t)) {
    Expected = llvm::sys::IsLittleEndianHost
                             ? "(0xffffeeee0000 (0xeeee0000, 0x0000ffff))"
                             : "(0xeeee00000000ffff (0xeeee0000, 0x0000ffff))";
  }
  else {
    Expected = "(0xeeee0000 (0xeeee0000, 0x0000ffff))";
  }

  OS << Data;
  OS.flush();

  EXPECT_EQ(Expected, s);
}
