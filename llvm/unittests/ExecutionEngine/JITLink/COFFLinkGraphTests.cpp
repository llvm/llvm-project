//===-- COFFLinkGraphTests.cpp - Unit tests for COFF LinkGraph utils ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JITLinkTestUtils.h"

#include "llvm/ExecutionEngine/JITLink/COFF.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::jitlink;

TEST(COFFLinkGraphTest, GetImageBaseSymbolReturnsNullWhenMissing) {
  LinkGraph G("foo", std::make_shared<orc::SymbolStringPool>(),
              Triple("x86_64-pc-windows-msvc"), SubtargetFeatures(),
              getGenericEdgeKindName);

  GetImageBaseSymbol GetIB;
  EXPECT_EQ(GetIB(G), nullptr);
}

TEST(COFFLinkGraphTest, GetImageBaseSymbolFindsExternal) {
  LinkGraph G("foo", std::make_shared<orc::SymbolStringPool>(),
              Triple("x86_64-pc-windows-msvc"), SubtargetFeatures(),
              getGenericEdgeKindName);

  auto &ExtSym = G.addExternalSymbol(G.intern("__ImageBase"), 0, false);
  GetImageBaseSymbol GetIB;
  EXPECT_EQ(GetIB(G), &ExtSym);
}

TEST(COFFLinkGraphTest, GetImageBaseSymbolFindsAbsolute) {
  LinkGraph G("foo", std::make_shared<orc::SymbolStringPool>(),
              Triple("x86_64-pc-windows-msvc"), SubtargetFeatures(),
              getGenericEdgeKindName);

  auto &AbsSym =
      G.addAbsoluteSymbol(G.intern("__ImageBase"), orc::ExecutorAddr(0x1000), 0,
                          Linkage::Strong, Scope::Default, true);
  GetImageBaseSymbol GetIB;
  EXPECT_EQ(GetIB(G), &AbsSym);
}

TEST(COFFLinkGraphTest, GetImageBaseSymbolFindsDefined) {
  LinkGraph G("foo", std::make_shared<orc::SymbolStringPool>(),
              Triple("x86_64-pc-windows-msvc"), SubtargetFeatures(),
              getGenericEdgeKindName);

  auto &Sec =
      G.createSection("__data", orc::MemProt::Read | orc::MemProt::Write);
  auto &B =
      G.createContentBlock(Sec, BlockContent, orc::ExecutorAddr(0x2000), 8, 0);
  auto &DefSym =
      G.addDefinedSymbol(B, 0, G.intern("__ImageBase"), 4, Linkage::Strong,
                         Scope::Default, false, false);
  GetImageBaseSymbol GetIB;
  EXPECT_EQ(GetIB(G), &DefSym);
}

TEST(COFFLinkGraphTest, GetImageBaseSymbolCachesResult) {
  LinkGraph G("foo", std::make_shared<orc::SymbolStringPool>(),
              Triple("x86_64-pc-windows-msvc"), SubtargetFeatures(),
              getGenericEdgeKindName);

  auto &ExtSym = G.addExternalSymbol(G.intern("__ImageBase"), 0, false);
  GetImageBaseSymbol GetIB;
  EXPECT_EQ(GetIB(G), &ExtSym);

  // Remove the symbol -- cached result should still be returned.
  G.removeExternalSymbol(ExtSym);
  EXPECT_EQ(GetIB(G), &ExtSym);
}

TEST(COFFLinkGraphTest, GetImageBaseSymbolReset) {
  LinkGraph G("foo", std::make_shared<orc::SymbolStringPool>(),
              Triple("x86_64-pc-windows-msvc"), SubtargetFeatures(),
              getGenericEdgeKindName);

  GetImageBaseSymbol GetIB;
  EXPECT_EQ(GetIB(G), nullptr);

  // Add a symbol and reset -- should now find it.
  auto &ExtSym = G.addExternalSymbol(G.intern("__ImageBase"), 0, false);
  GetIB.reset();
  EXPECT_EQ(GetIB(G), &ExtSym);

  // Reset with an explicit cache value of nullptr -- should return nullptr
  // without searching.
  GetIB.reset(nullptr);
  EXPECT_EQ(GetIB(G), nullptr);
}

TEST(COFFLinkGraphTest, GetImageBaseSymbolCustomName) {
  LinkGraph G("foo", std::make_shared<orc::SymbolStringPool>(),
              Triple("x86_64-pc-windows-msvc"), SubtargetFeatures(),
              getGenericEdgeKindName);

  G.addExternalSymbol(G.intern("__ImageBase"), 0, false);
  auto &CustomSym = G.addExternalSymbol(G.intern("__CustomBase"), 0, false);

  GetImageBaseSymbol GetIB("__CustomBase");
  EXPECT_EQ(GetIB(G), &CustomSym);
}
