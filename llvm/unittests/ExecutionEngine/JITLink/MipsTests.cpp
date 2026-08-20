//===-- MipsTests.cpp - Tests for MIPS JITLink utilities ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/mips.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::jitlink;

TEST(MipsJITLinkTest, N32PointerSize) {
  LinkGraph G("n32", std::make_shared<orc::SymbolStringPool>(),
              Triple("mips64el-unknown-linux-gnuabin32"), SubtargetFeatures(),
              mips::getEdgeKindName, 4);
  EXPECT_EQ(G.getPointerSize(), 4U);
  EXPECT_EQ(G.getEndianness(), endianness::little);
  EXPECT_EQ(mips::getPointerEdgeKind(G), mips::Pointer32);
}

TEST(MipsJITLinkTest, GOTAndStubs) {
  LinkGraph G("mips-r6", std::make_shared<orc::SymbolStringPool>(),
              Triple("mipsel-unknown-linux"), SubtargetFeatures("+mips32r6"),
              mips::getEdgeKindName);
  auto &GOT =
      G.createSection("$__GOT", orc::MemProt::Read | orc::MemProt::Write);
  auto &Stubs =
      G.createSection("$__STUBS", orc::MemProt::Read | orc::MemProt::Exec);
  auto &Target = G.addAbsoluteSymbol("target", orc::ExecutorAddr(0x12345678), 0,
                                     Linkage::Strong, Scope::Local, false);
  auto &Pointer = mips::createAnonymousPointer(G, GOT, &Target, 4);
  ASSERT_EQ(Pointer.getBlock().edges_size(), 1U);
  auto &PointerEdge = *Pointer.getBlock().edges().begin();
  EXPECT_EQ(PointerEdge.getKind(), mips::Pointer32);
  EXPECT_EQ(&PointerEdge.getTarget(), &Target);
  EXPECT_EQ(PointerEdge.getAddend(), 4);

  auto &Stub = mips::createAnonymousPointerJumpStub(G, Stubs, Pointer);
  ASSERT_EQ(Stub.getSize(), 20U);
  ASSERT_EQ(Stub.getBlock().edges_size(), 2U);
  auto StubEdge = Stub.getBlock().edges().begin();
  EXPECT_EQ(StubEdge->getKind(), mips::Hi16);
  EXPECT_EQ(&StubEdge->getTarget(), &Pointer);
  ++StubEdge;
  EXPECT_EQ(StubEdge->getKind(), mips::Lo16);
  EXPECT_EQ(&StubEdge->getTarget(), &Pointer);

  LinkGraph G64("mips64be-r2", std::make_shared<orc::SymbolStringPool>(),
                Triple("mips64-unknown-linux-gnuabi64"),
                SubtargetFeatures("+mips64r2"), mips::getEdgeKindName);
  auto &GOT64 =
      G64.createSection("$__GOT", orc::MemProt::Read | orc::MemProt::Write);
  auto &Stubs64 =
      G64.createSection("$__STUBS", orc::MemProt::Read | orc::MemProt::Exec);
  auto &Target64 =
      G64.addAbsoluteSymbol("target", orc::ExecutorAddr(0x123456789abcdef0ULL),
                            0, Linkage::Strong, Scope::Local, false);
  auto &Pointer64 = mips::createAnonymousPointer(G64, GOT64, &Target64);
  EXPECT_EQ(Pointer64.getBlock().edges().begin()->getKind(), mips::Pointer64);

  auto &Stub64 = mips::createAnonymousPointerJumpStub(G64, Stubs64, Pointer64);
  ASSERT_EQ(Stub64.getSize(), 36U);
  ASSERT_EQ(Stub64.getBlock().edges_size(), 4U);
  auto I = Stub64.getBlock().edges().begin();
  EXPECT_EQ((I++)->getKind(), mips::Highest16);
  EXPECT_EQ((I++)->getKind(), mips::Higher16);
  EXPECT_EQ((I++)->getKind(), mips::Hi16);
  EXPECT_EQ(I->getKind(), mips::Lo16);
}
