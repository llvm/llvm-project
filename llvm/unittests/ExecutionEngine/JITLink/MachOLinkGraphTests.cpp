//===------ MachOLinkGraphTests.cpp - Unit tests for MachO LinkGraphs -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JITLinkTestUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITLink/MachO.h"
#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"

#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::jitlink;

TEST(MachOLinkGraphTest, GetStandardSections) {
  // Check that LinkGraph construction works as expected.
  LinkGraph G("foo", std::make_shared<orc::SymbolStringPool>(),
              Triple("arm64-apple-darwin"), SubtargetFeatures(),
              getGenericEdgeKindName);

  auto &Data = getMachODefaultRWDataSection(G);
  EXPECT_TRUE(Data.empty());
  EXPECT_EQ(Data.getName(), orc::MachODataDataSectionName);
  EXPECT_EQ(Data.getMemProt(), orc::MemProt::Read | orc::MemProt::Write);

  auto &Text = getMachODefaultTextSection(G);
  EXPECT_TRUE(Text.empty());
  EXPECT_EQ(Text.getName(), orc::MachOTextTextSectionName);
  EXPECT_EQ(Text.getMemProt(), orc::MemProt::Read | orc::MemProt::Exec);
}
