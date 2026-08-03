//===- AtomicScopeTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/AtomicScope.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(AtomicScopeTest, AbiValues) {
  EXPECT_EQ(static_cast<unsigned>(AtomicScope::System), 0u);
  EXPECT_EQ(static_cast<unsigned>(AtomicScope::Device), 1u);
  EXPECT_EQ(static_cast<unsigned>(AtomicScope::Workgroup), 2u);
  EXPECT_EQ(static_cast<unsigned>(AtomicScope::Wavefront), 3u);
  EXPECT_EQ(static_cast<unsigned>(AtomicScope::Single), 4u);
  EXPECT_EQ(static_cast<unsigned>(AtomicScope::Cluster), 5u);
}

TEST(AtomicScopeTest, UnknownStringIsNone) {
  Triple NVPTX("nvptx64-nvidia-cuda");
  EXPECT_FALSE(parseAtomicScopeIRString(NVPTX, "bogus").has_value());
  EXPECT_FALSE(parseAtomicScopeIRString(NVPTX, "agent").has_value());
}

TEST(AtomicScopeTest, NonGPUTargetIsNone) {
  Triple X86("x86_64-unknown-linux-gnu");
  EXPECT_FALSE(getAtomicScopeIRString(X86, AtomicScope::System).has_value());
  EXPECT_FALSE(parseAtomicScopeIRString(X86, "").has_value());
}

} // namespace
