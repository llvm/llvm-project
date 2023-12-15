//===- TypeNameTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TypeName.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
namespace N1 {
struct S1 {};
class C1 {};
union U1 {};
}

TEST(TypeNameTest, Names) {
  struct S2 {};

  StringRef S1Name = getTypeName<N1::S1>();
  StringRef C1Name = getTypeName<N1::C1>();
  StringRef U1Name = getTypeName<N1::U1>();
  StringRef S2Name = getTypeName<S2>();

#if defined(__clang__) || defined(__GNUC__) || defined(__INTEL_COMPILER) ||    \
    defined(_MSC_VER)
  EXPECT_TRUE(S1Name.ends_with("::N1::S1")) << S1Name.str();
  EXPECT_TRUE(C1Name.ends_with("::N1::C1")) << C1Name.str();
  EXPECT_TRUE(U1Name.ends_with("::N1::U1")) << U1Name.str();
#ifdef __clang__
  EXPECT_TRUE(S2Name.ends_with("S2")) << S2Name.str();
#else
  EXPECT_TRUE(S2Name.endswith("::S2")) << S2Name.str();
#endif
#else
  EXPECT_EQ("UNKNOWN_TYPE", S1Name);
  EXPECT_EQ("UNKNOWN_TYPE", C1Name);
  EXPECT_EQ("UNKNOWN_TYPE", U1Name);
  EXPECT_EQ("UNKNOWN_TYPE", S2Name);
#endif
}

} // end anonymous namespace
