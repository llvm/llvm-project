//===------ LinkGraphTests.cpp - Unit tests for core JITLink classes ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Memory.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::jitlink;

static auto RWFlags =
    sys::Memory::ProtectionFlags(sys::Memory::MF_READ | sys::Memory::MF_WRITE);

static const char BlockContentBytes[] = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15,
                                         0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B,
                                         0x1C, 0x1D, 0x1E, 0x1F, 0x00};
static StringRef BlockContent(BlockContentBytes);

TEST(LinkGraphTest, Construction) {
  // Check that LinkGraph construction works as expected.
  LinkGraph G("foo", Triple("x86_64-apple-darwin"), 8, support::little,
              getGenericEdgeKindName);
  EXPECT_EQ(G.getName(), "foo");
  EXPECT_EQ(G.getTargetTriple().str(), "x86_64-apple-darwin");
  EXPECT_EQ(G.getPointerSize(), 8U);
  EXPECT_EQ(G.getEndianness(), support::little);
  EXPECT_TRUE(llvm::empty(G.external_symbols()));
  EXPECT_TRUE(llvm::empty(G.absolute_symbols()));
  EXPECT_TRUE(llvm::empty(G.defined_symbols()));
  EXPECT_TRUE(llvm::empty(G.blocks()));
}

TEST(LinkGraphTest, AddressAccess) {
  // Check that we can get addresses for blocks, symbols, and edges.
  LinkGraph G("foo", Triple("x86_64-apple-darwin"), 8, support::little,
              getGenericEdgeKindName);

  auto Sec1 = G.createSection("__data.1", RWFlags);
  auto &B1 = G.createContentBlock(Sec1, BlockContent, 0x1000, 8, 0);
  auto &S1 = G.addDefinedSymbol(B1, 4, "S1", 4, Linkage::Strong, Scope::Default,
                                false, false);
  B1.addEdge(Edge::FirstRelocation, 8, S1, 0);
  auto &E1 = *B1.edges().begin();

  EXPECT_EQ(B1.getAddress(), 0x1000U) << "Incorrect block address";
  EXPECT_EQ(S1.getAddress(), 0x1004U) << "Incorrect symbol address";
  EXPECT_EQ(B1.getFixupAddress(E1), 0x1008U) << "Incorrect fixup address";
}

TEST(LinkGraphTest, BlockAndSymbolIteration) {
  // Check that we can iterate over blocks within Sections and across sections.
  LinkGraph G("foo", Triple("x86_64-apple-darwin"), 8, support::little,
              getGenericEdgeKindName);
  auto &Sec1 = G.createSection("__data.1", RWFlags);
  auto &B1 = G.createContentBlock(Sec1, BlockContent, 0x1000, 8, 0);
  auto &B2 = G.createContentBlock(Sec1, BlockContent, 0x2000, 8, 0);
  auto &S1 = G.addDefinedSymbol(B1, 0, "S1", 4, Linkage::Strong, Scope::Default,
                                false, false);
  auto &S2 = G.addDefinedSymbol(B2, 4, "S2", 4, Linkage::Strong, Scope::Default,
                                false, false);

  auto &Sec2 = G.createSection("__data.2", RWFlags);
  auto &B3 = G.createContentBlock(Sec2, BlockContent, 0x3000, 8, 0);
  auto &B4 = G.createContentBlock(Sec2, BlockContent, 0x4000, 8, 0);
  auto &S3 = G.addDefinedSymbol(B3, 0, "S3", 4, Linkage::Strong, Scope::Default,
                                false, false);
  auto &S4 = G.addDefinedSymbol(B4, 4, "S4", 4, Linkage::Strong, Scope::Default,
                                false, false);

  // Check that iteration of blocks within a section behaves as expected.
  EXPECT_EQ(std::distance(Sec1.blocks().begin(), Sec1.blocks().end()), 2);
  EXPECT_TRUE(llvm::count(Sec1.blocks(), &B1));
  EXPECT_TRUE(llvm::count(Sec1.blocks(), &B2));

  // Check that iteration of symbols within a section behaves as expected.
  EXPECT_EQ(std::distance(Sec1.symbols().begin(), Sec1.symbols().end()), 2);
  EXPECT_TRUE(llvm::count(Sec1.symbols(), &S1));
  EXPECT_TRUE(llvm::count(Sec1.symbols(), &S2));

  // Check that iteration of blocks across sections behaves as expected.
  EXPECT_EQ(std::distance(G.blocks().begin(), G.blocks().end()), 4);
  EXPECT_TRUE(llvm::count(G.blocks(), &B1));
  EXPECT_TRUE(llvm::count(G.blocks(), &B2));
  EXPECT_TRUE(llvm::count(G.blocks(), &B3));
  EXPECT_TRUE(llvm::count(G.blocks(), &B4));

  // Check that iteration of defined symbols across sections behaves as
  // expected.
  EXPECT_EQ(
      std::distance(G.defined_symbols().begin(), G.defined_symbols().end()), 4);
  EXPECT_TRUE(llvm::count(G.defined_symbols(), &S1));
  EXPECT_TRUE(llvm::count(G.defined_symbols(), &S2));
  EXPECT_TRUE(llvm::count(G.defined_symbols(), &S3));
  EXPECT_TRUE(llvm::count(G.defined_symbols(), &S4));
}

TEST(LinkGraphTest, SplitBlock) {
  // Check that the LinkGraph::splitBlock test works as expected.

  const char BlockContentBytes[] = {0x10, 0x11, 0x12, 0x13, 0x14, 0x15,
                                    0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B,
                                    0x1C, 0x1D, 0x1E, 0x1F, 0x00};
  StringRef BlockContent(BlockContentBytes);

  LinkGraph G("foo", Triple("x86_64-apple-darwin"), 8, support::little,
              getGenericEdgeKindName);
  auto &Sec = G.createSection("__data", RWFlags);

  // Create the block to split.
  auto &B1 = G.createContentBlock(Sec, BlockContent, 0x1000, 8, 0);

  // Add some symbols to the block.
  auto &S1 = G.addDefinedSymbol(B1, 0, "S1", 4, Linkage::Strong, Scope::Default,
                                false, false);
  auto &S2 = G.addDefinedSymbol(B1, 4, "S2", 4, Linkage::Strong, Scope::Default,
                                false, false);
  auto &S3 = G.addDefinedSymbol(B1, 8, "S3", 4, Linkage::Strong, Scope::Default,
                                false, false);
  auto &S4 = G.addDefinedSymbol(B1, 12, "S4", 4, Linkage::Strong,
                                Scope::Default, false, false);

  // Add an extra block, EB, and target symbols, and use these to add edges
  // from B1 to EB.
  auto &EB = G.createContentBlock(Sec, BlockContent, 0x2000, 8, 0);
  auto &ES1 = G.addDefinedSymbol(EB, 0, "TS1", 4, Linkage::Strong,
                                 Scope::Default, false, false);
  auto &ES2 = G.addDefinedSymbol(EB, 4, "TS2", 4, Linkage::Strong,
                                 Scope::Default, false, false);
  auto &ES3 = G.addDefinedSymbol(EB, 8, "TS3", 4, Linkage::Strong,
                                 Scope::Default, false, false);
  auto &ES4 = G.addDefinedSymbol(EB, 12, "TS4", 4, Linkage::Strong,
                                 Scope::Default, false, false);

  // Add edges from B1 to EB.
  B1.addEdge(Edge::FirstRelocation, 0, ES1, 0);
  B1.addEdge(Edge::FirstRelocation, 4, ES2, 0);
  B1.addEdge(Edge::FirstRelocation, 8, ES3, 0);
  B1.addEdge(Edge::FirstRelocation, 12, ES4, 0);

  // Split B1.
  auto &B2 = G.splitBlock(B1, 8);

  // Check that the block addresses and content matches what we would expect.
  EXPECT_EQ(B1.getAddress(), 0x1008U);
  EXPECT_EQ(B1.getContent(), BlockContent.substr(8));

  EXPECT_EQ(B2.getAddress(), 0x1000U);
  EXPECT_EQ(B2.getContent(), BlockContent.substr(0, 8));

  // Check that symbols in B1 were transferred as expected:
  // We expect S1 and S2 to have been transferred to B2, and S3 and S4 to have
  // remained attached to B1. Symbols S3 and S4 should have had their offsets
  // slid to account for the change in address of B2.
  EXPECT_EQ(&S1.getBlock(), &B2);
  EXPECT_EQ(S1.getOffset(), 0U);

  EXPECT_EQ(&S2.getBlock(), &B2);
  EXPECT_EQ(S2.getOffset(), 4U);

  EXPECT_EQ(&S3.getBlock(), &B1);
  EXPECT_EQ(S3.getOffset(), 0U);

  EXPECT_EQ(&S4.getBlock(), &B1);
  EXPECT_EQ(S4.getOffset(), 4U);

  // Check that edges in B1 have been transferred as expected:
  // Both blocks should now have two edges each at offsets 0 and 4.
  EXPECT_EQ(llvm::size(B1.edges()), 2);
  if (size(B1.edges()) == 2) {
    auto *E1 = &*B1.edges().begin();
    auto *E2 = &*(B1.edges().begin() + 1);
    if (E2->getOffset() < E1->getOffset())
      std::swap(E1, E2);
    EXPECT_EQ(E1->getOffset(), 0U);
    EXPECT_EQ(E2->getOffset(), 4U);
  }

  EXPECT_EQ(llvm::size(B2.edges()), 2);
  if (size(B2.edges()) == 2) {
    auto *E1 = &*B2.edges().begin();
    auto *E2 = &*(B2.edges().begin() + 1);
    if (E2->getOffset() < E1->getOffset())
      std::swap(E1, E2);
    EXPECT_EQ(E1->getOffset(), 0U);
    EXPECT_EQ(E2->getOffset(), 4U);
  }
}
