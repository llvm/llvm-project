//===------- MemoryFlagsTest.cpp - Test MemoryFlags and related APIs ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/MemoryFlags.h"
#include "gtest/gtest.h"

#include <future>

using namespace llvm;
using namespace llvm::orc;

TEST(MemProtTest, Basics) {
  MemProt MPNone = MemProt::None;

  EXPECT_EQ(MPNone & MemProt::Read, MemProt::None);
  EXPECT_EQ(MPNone & MemProt::Write, MemProt::None);
  EXPECT_EQ(MPNone & MemProt::Exec, MemProt::None);

  EXPECT_EQ(MPNone | MemProt::Read, MemProt::Read);
  EXPECT_EQ(MPNone | MemProt::Write, MemProt::Write);
  EXPECT_EQ(MPNone | MemProt::Exec, MemProt::Exec);

  MemProt MPAll = MemProt::Read | MemProt::Write | MemProt::Exec;
  EXPECT_EQ(MPAll & MemProt::Read, MemProt::Read);
  EXPECT_EQ(MPAll & MemProt::Write, MemProt::Write);
  EXPECT_EQ(MPAll & MemProt::Exec, MemProt::Exec);
}

TEST(AllocGroupSmallMap, EmptyMap) {
  AllocGroupSmallMap<bool> EM;
  EXPECT_TRUE(EM.empty());
  EXPECT_EQ(EM.size(), 0u);
}

TEST(AllocGroupSmallMap, NonEmptyMap) {
  AllocGroupSmallMap<unsigned> NEM;
  NEM[MemProt::Read] = 42;

  EXPECT_FALSE(NEM.empty());
  EXPECT_EQ(NEM.size(), 1U);
  EXPECT_EQ(NEM[MemProt::Read], 42U);
  EXPECT_EQ(NEM.find(MemProt::Read), NEM.begin());
  EXPECT_EQ(NEM.find(MemProt::Read | MemProt::Write), NEM.end());

  NEM[MemProt::Read | MemProt::Write] = 7;
  EXPECT_EQ(NEM.size(), 2U);
  EXPECT_EQ(NEM.begin()->second, 42U);
  EXPECT_EQ((NEM.begin() + 1)->second, 7U);
}

