//===------- AArch64Tests.cpp - Unit tests for the AArch64 backend --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <llvm/BinaryFormat/ELF.h>
#include <llvm/ExecutionEngine/JITLink/aarch64.h>

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::jitlink::aarch64;

TEST(AArch64, EmptyLinkGraph) {
  LinkGraph G("foo", std::make_shared<orc::SymbolStringPool>(),
              Triple("arm64-apple-darwin"), SubtargetFeatures(),
              getEdgeKindName);
  EXPECT_EQ(G.getName(), "foo");
  EXPECT_EQ(G.getTargetTriple().str(), "arm64-apple-darwin");
  EXPECT_EQ(G.getPointerSize(), 8U);
  EXPECT_EQ(G.getEndianness(), llvm::endianness::little);
  EXPECT_TRUE(G.external_symbols().empty());
  EXPECT_TRUE(G.absolute_symbols().empty());
  EXPECT_TRUE(G.defined_symbols().empty());
  EXPECT_TRUE(G.blocks().empty());
}

TEST(AArch64, GOTAndStubs) {
  LinkGraph G("foo", std::make_shared<orc::SymbolStringPool>(),
              Triple("arm64-apple-darwin"), SubtargetFeatures(),
              getEdgeKindName);

  auto &External = G.addExternalSymbol("external", 0, false);

  // First table accesses. We expect the graph to be empty:
  EXPECT_EQ(G.findSectionByName(GOTTableManager::getSectionName()), nullptr);
  EXPECT_EQ(G.findSectionByName(PLTTableManager::getSectionName()), nullptr);

  {
    // Create first GOT and PLT table managers and request a PLT stub. This
    // should force creation of both a PLT stub and GOT entry.
    GOTTableManager GOT(G);
    PLTTableManager PLT(G, GOT);

    PLT.getEntryForTarget(G, External);
  }

  auto *GOTSec = G.findSectionByName(GOTTableManager::getSectionName());
  EXPECT_NE(GOTSec, nullptr);
  if (GOTSec) {
    // Expect one entry in the GOT now.
    EXPECT_EQ(GOTSec->symbols_size(), 1U);
    EXPECT_EQ(GOTSec->blocks_size(), 1U);
  }

  auto *PLTSec = G.findSectionByName(PLTTableManager::getSectionName());
  EXPECT_NE(PLTSec, nullptr);
  if (PLTSec) {
    // Expect one entry in the PLT.
    EXPECT_EQ(PLTSec->symbols_size(), 1U);
    EXPECT_EQ(PLTSec->blocks_size(), 1U);
  }

  {
    // Create second GOT and PLT table managers and request a PLT stub. This
    // should force creation of both a PLT stub and GOT entry.
    GOTTableManager GOT(G);
    PLTTableManager PLT(G, GOT);

    PLT.getEntryForTarget(G, External);
  }

  EXPECT_EQ(G.findSectionByName(GOTTableManager::getSectionName()), GOTSec);
  if (GOTSec) {
    // Expect the same one entry in the GOT.
    EXPECT_EQ(GOTSec->symbols_size(), 1U);
    EXPECT_EQ(GOTSec->blocks_size(), 1U);
  }

  EXPECT_EQ(G.findSectionByName(PLTTableManager::getSectionName()), PLTSec);
  if (PLTSec) {
    // Expect the same one entry in the GOT.
    EXPECT_EQ(PLTSec->symbols_size(), 1U);
    EXPECT_EQ(PLTSec->blocks_size(), 1U);
  }
}
