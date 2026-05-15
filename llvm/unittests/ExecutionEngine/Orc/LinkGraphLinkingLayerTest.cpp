//===--- LinkGraphLinkingLayerTest.cpp - Unit tests for dep group calc ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/LinkGraphLinkingLayer.h"

#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::orc;

static const char BlockContentBytes[] = {0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0};
static ArrayRef<char> BlockContent(BlockContentBytes);

/// Test wrapper class that accesses LinkGraphLinkingLayer's private members
/// via the friend declaration.
class LinkGraphLinkingLayerTests : public testing::Test {
protected:
  using SymbolDepGroup = LinkGraphLinkingLayer::SymbolDepGroup;

  static SmallVector<SymbolDepGroup> calculateDepGroups(LinkGraph &G) {
    return LinkGraphLinkingLayer::calculateDepGroups(G);
  }

  static std::unique_ptr<LinkGraph> makeGraph(StringRef Name = "test") {
    return std::make_unique<LinkGraph>(
        Name.str(), std::make_shared<SymbolStringPool>(),
        Triple("x86_64-apple-darwin"), SubtargetFeatures(),
        getGenericEdgeKindName);
  }
};

// No blocks with non-local symbols: should produce no dep groups.
TEST_F(LinkGraphLinkingLayerTests, EmptyGraph) {
  auto G = makeGraph();
  auto DGs = calculateDepGroups(*G);
  EXPECT_TRUE(DGs.empty());
}

// A single block with a defined symbol and no edges: no deps, so no dep group.
TEST_F(LinkGraphLinkingLayerTests, SingleBlockNoDeps) {
  auto G = makeGraph();
  auto &Sec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &B =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x1000), 8, 0);
  G->addDefinedSymbol(B, 0, "foo", 4, Linkage::Strong, Scope::Default, false,
                      true);

  auto DGs = calculateDepGroups(*G);
  EXPECT_TRUE(DGs.empty());
}

// A single block with a defined symbol that depends on an external symbol.
TEST_F(LinkGraphLinkingLayerTests, SingleBlockExternalDep) {
  auto G = makeGraph();
  auto &Sec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &B =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x1000), 8, 0);
  G->addDefinedSymbol(B, 0, "foo", 4, Linkage::Strong, Scope::Default, false,
                      true);
  auto &ExtSym = G->addExternalSymbol("bar", 0, false);
  B.addEdge(Edge::FirstRelocation, 0, ExtSym, 0);

  auto DGs = calculateDepGroups(*G);
  ASSERT_EQ(DGs.size(), 1u);
  EXPECT_EQ(DGs[0].Defs.size(), 1u);
  EXPECT_EQ(DGs[0].Defs[0]->getName(), G->intern("foo"));
  EXPECT_EQ(DGs[0].Deps.size(), 1u);
  EXPECT_TRUE(DGs[0].Deps.count(&ExtSym));
}

// Two blocks, each with a defined symbol. Block A depends on Block B's symbol.
TEST_F(LinkGraphLinkingLayerTests, TwoBlocksDirectDep) {
  auto G = makeGraph();
  auto &Sec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &BA =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x1000), 8, 0);
  auto &BB =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x2000), 8, 0);
  G->addDefinedSymbol(BA, 0, "A", 4, Linkage::Strong, Scope::Default, false,
                      true);
  auto &SymB = G->addDefinedSymbol(BB, 0, "B", 4, Linkage::Strong,
                                   Scope::Default, false, true);
  // A -> B edge (through SymB).
  BA.addEdge(Edge::FirstRelocation, 0, SymB, 0);

  auto DGs = calculateDepGroups(*G);
  ASSERT_EQ(DGs.size(), 1u);
  // The dep group should contain A's def and B as a dependency.
  EXPECT_EQ(DGs[0].Defs.size(), 1u);
  EXPECT_EQ(DGs[0].Defs[0]->getName(), G->intern("A"));
  EXPECT_EQ(DGs[0].Deps.size(), 1u);
  EXPECT_TRUE(DGs[0].Deps.count(&SymB));
}

// Two independent blocks with no edges between them: no dep groups.
TEST_F(LinkGraphLinkingLayerTests, TwoIndependentBlocks) {
  auto G = makeGraph();
  auto &Sec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &BA =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x1000), 8, 0);
  auto &BB =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x2000), 8, 0);
  G->addDefinedSymbol(BA, 0, "A", 4, Linkage::Strong, Scope::Default, false,
                      true);
  G->addDefinedSymbol(BB, 0, "B", 4, Linkage::Strong, Scope::Default, false,
                      true);

  auto DGs = calculateDepGroups(*G);
  EXPECT_TRUE(DGs.empty());
}

// Block A -> anonymous block -> Block B. The anonymous block should be
// transparent, and A should transitively depend on B.
TEST_F(LinkGraphLinkingLayerTests, TransitiveThroughAnonymousBlock) {
  auto G = makeGraph();
  auto &Sec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &BA =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x1000), 8, 0);
  auto &BAnon =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x2000), 8, 0);
  auto &BB =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x3000), 8, 0);
  G->addDefinedSymbol(BA, 0, "A", 4, Linkage::Strong, Scope::Default, false,
                      true);
  // BAnon has no named symbols (anonymous block).
  auto &AnonSym = G->addAnonymousSymbol(BAnon, 0, 4, false, true);
  auto &SymB = G->addDefinedSymbol(BB, 0, "B", 4, Linkage::Strong,
                                   Scope::Default, false, true);

  // A -> AnonSym (in BAnon), BAnon -> SymB (in BB).
  BA.addEdge(Edge::FirstRelocation, 0, AnonSym, 0);
  BAnon.addEdge(Edge::FirstRelocation, 0, SymB, 0);

  auto DGs = calculateDepGroups(*G);
  ASSERT_EQ(DGs.size(), 1u);
  EXPECT_EQ(DGs[0].Defs.size(), 1u);
  EXPECT_EQ(DGs[0].Defs[0]->getName(), G->intern("A"));
  EXPECT_EQ(DGs[0].Deps.size(), 1u);
  EXPECT_TRUE(DGs[0].Deps.count(&SymB));
}

// Block A depends on an external and an absolute symbol. Only the external
// should appear in deps (absolutes are assumed ready).
TEST_F(LinkGraphLinkingLayerTests, AbsoluteSymbolIgnored) {
  auto G = makeGraph();
  auto &Sec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &B =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x1000), 8, 0);
  G->addDefinedSymbol(B, 0, "foo", 4, Linkage::Strong, Scope::Default, false,
                      true);
  auto &ExtSym = G->addExternalSymbol("ext", 0, false);
  auto &AbsSym = G->addAbsoluteSymbol("abs", ExecutorAddr(0x42), 0,
                                      Linkage::Strong, Scope::Default, true);
  B.addEdge(Edge::FirstRelocation, 0, ExtSym, 0);
  B.addEdge(Edge::FirstRelocation, 4, AbsSym, 0);

  auto DGs = calculateDepGroups(*G);
  ASSERT_EQ(DGs.size(), 1u);
  EXPECT_EQ(DGs[0].Deps.size(), 1u);
  EXPECT_TRUE(DGs[0].Deps.count(&ExtSym));
}

// Local-scoped symbols should not appear in dep groups.
TEST_F(LinkGraphLinkingLayerTests, LocalScopeSymbolsIgnored) {
  auto G = makeGraph();
  auto &Sec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &B =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x1000), 8, 0);
  G->addDefinedSymbol(B, 0, "local_sym", 4, Linkage::Strong, Scope::Local,
                      false, true);
  auto &ExtSym = G->addExternalSymbol("ext", 0, false);
  B.addEdge(Edge::FirstRelocation, 0, ExtSym, 0);

  auto DGs = calculateDepGroups(*G);
  // Local symbol shouldn't be tracked, so no dep group.
  EXPECT_TRUE(DGs.empty());
}

// Multiple symbols in the same block should end up in the same dep group.
TEST_F(LinkGraphLinkingLayerTests, MultipleSymbolsSameBlock) {
  auto G = makeGraph();
  auto &Sec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &B =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x1000), 8, 0);
  G->addDefinedSymbol(B, 0, "foo", 4, Linkage::Strong, Scope::Default, false,
                      true);
  G->addDefinedSymbol(B, 4, "bar", 4, Linkage::Strong, Scope::Default, false,
                      true);
  auto &ExtSym = G->addExternalSymbol("ext", 0, false);
  B.addEdge(Edge::FirstRelocation, 0, ExtSym, 0);

  auto DGs = calculateDepGroups(*G);
  ASSERT_EQ(DGs.size(), 1u);
  EXPECT_EQ(DGs[0].Defs.size(), 2u);
  EXPECT_EQ(DGs[0].Deps.size(), 1u);
  EXPECT_TRUE(DGs[0].Deps.count(&ExtSym));
}

// A -> B -> A cycle (through named symbols). Named blocks don't participate
// in the SCC graph — only anonymous blocks do. So A and B each get their own
// dep group, with the other as a dependency.
TEST_F(LinkGraphLinkingLayerTests, CycleTwoNamedBlocks) {
  auto G = makeGraph();
  auto &Sec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &BA =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x1000), 8, 0);
  auto &BB =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x2000), 8, 0);
  auto &SymA = G->addDefinedSymbol(BA, 0, "A", 4, Linkage::Strong,
                                   Scope::Default, false, true);
  auto &SymB = G->addDefinedSymbol(BB, 0, "B", 4, Linkage::Strong,
                                   Scope::Default, false, true);
  // A -> B, B -> A.
  BA.addEdge(Edge::FirstRelocation, 0, SymB, 0);
  BB.addEdge(Edge::FirstRelocation, 0, SymA, 0);

  auto DGs = calculateDepGroups(*G);
  // Each named block is its own SCC. A depends on B and B depends on A.
  ASSERT_EQ(DGs.size(), 2u);
  SymbolDepGroup *DGA = nullptr, *DGB = nullptr;
  for (auto &DG : DGs) {
    ASSERT_EQ(DG.Defs.size(), 1u);
    if (DG.Defs[0]->getName() == G->intern("A"))
      DGA = &DG;
    else if (DG.Defs[0]->getName() == G->intern("B"))
      DGB = &DG;
  }
  ASSERT_NE(DGA, nullptr);
  ASSERT_NE(DGB, nullptr);
  EXPECT_EQ(DGA->Deps.size(), 1u);
  EXPECT_TRUE(DGA->Deps.count(&SymB));
  EXPECT_EQ(DGB->Deps.size(), 1u);
  EXPECT_TRUE(DGB->Deps.count(&SymA));
}

// A -> B -> A cycle where A also depends on external "ext". Since named
// blocks don't form SCCs, A and B get separate dep groups.
TEST_F(LinkGraphLinkingLayerTests, CycleWithExternalDep) {
  auto G = makeGraph();
  auto &Sec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &BA =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x1000), 8, 0);
  auto &BB =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x2000), 8, 0);
  auto &SymA = G->addDefinedSymbol(BA, 0, "A", 4, Linkage::Strong,
                                   Scope::Default, false, true);
  auto &SymB = G->addDefinedSymbol(BB, 0, "B", 4, Linkage::Strong,
                                   Scope::Default, false, true);
  auto &ExtSym = G->addExternalSymbol("ext", 0, false);
  BA.addEdge(Edge::FirstRelocation, 0, SymB, 0);
  BA.addEdge(Edge::FirstRelocation, 4, ExtSym, 0);
  BB.addEdge(Edge::FirstRelocation, 0, SymA, 0);

  auto DGs = calculateDepGroups(*G);
  ASSERT_EQ(DGs.size(), 2u);
  SymbolDepGroup *DGA = nullptr, *DGB = nullptr;
  for (auto &DG : DGs) {
    ASSERT_EQ(DG.Defs.size(), 1u);
    if (DG.Defs[0]->getName() == G->intern("A"))
      DGA = &DG;
    else if (DG.Defs[0]->getName() == G->intern("B"))
      DGB = &DG;
  }
  ASSERT_NE(DGA, nullptr);
  ASSERT_NE(DGB, nullptr);
  // A depends on B and ext.
  EXPECT_EQ(DGA->Deps.size(), 2u);
  EXPECT_TRUE(DGA->Deps.count(&SymB));
  EXPECT_TRUE(DGA->Deps.count(&ExtSym));
  // B depends on A.
  EXPECT_EQ(DGB->Deps.size(), 1u);
  EXPECT_TRUE(DGB->Deps.count(&SymA));
}

// Chain through multiple anonymous blocks:
// A -> anon1 -> anon2 -> B
TEST_F(LinkGraphLinkingLayerTests, ChainThroughMultipleAnonymousBlocks) {
  auto G = makeGraph();
  auto &Sec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &BA =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x1000), 8, 0);
  auto &BAnon1 =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x2000), 8, 0);
  auto &BAnon2 =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x3000), 8, 0);
  auto &BB =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x4000), 8, 0);

  G->addDefinedSymbol(BA, 0, "A", 4, Linkage::Strong, Scope::Default, false,
                      true);
  auto &AnonSym1 = G->addAnonymousSymbol(BAnon1, 0, 4, false, true);
  auto &AnonSym2 = G->addAnonymousSymbol(BAnon2, 0, 4, false, true);
  auto &SymB = G->addDefinedSymbol(BB, 0, "B", 4, Linkage::Strong,
                                   Scope::Default, false, true);

  BA.addEdge(Edge::FirstRelocation, 0, AnonSym1, 0);
  BAnon1.addEdge(Edge::FirstRelocation, 0, AnonSym2, 0);
  BAnon2.addEdge(Edge::FirstRelocation, 0, SymB, 0);

  auto DGs = calculateDepGroups(*G);
  ASSERT_EQ(DGs.size(), 1u);
  EXPECT_EQ(DGs[0].Defs.size(), 1u);
  EXPECT_EQ(DGs[0].Defs[0]->getName(), G->intern("A"));
  EXPECT_EQ(DGs[0].Deps.size(), 1u);
  EXPECT_TRUE(DGs[0].Deps.count(&SymB));
}

// Diamond through anonymous blocks:
//   A -> anon1 -> B
//   A -> anon2 -> B
// A should depend on B (once).
TEST_F(LinkGraphLinkingLayerTests, DiamondThroughAnonymousBlocks) {
  auto G = makeGraph();
  auto &Sec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &BA =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x1000), 8, 0);
  auto &BAnon1 =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x2000), 8, 0);
  auto &BAnon2 =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x3000), 8, 0);
  auto &BB =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x4000), 8, 0);

  G->addDefinedSymbol(BA, 0, "A", 4, Linkage::Strong, Scope::Default, false,
                      true);
  auto &AnonSym1 = G->addAnonymousSymbol(BAnon1, 0, 4, false, true);
  auto &AnonSym2 = G->addAnonymousSymbol(BAnon2, 0, 4, false, true);
  auto &SymB = G->addDefinedSymbol(BB, 0, "B", 4, Linkage::Strong,
                                   Scope::Default, false, true);

  BA.addEdge(Edge::FirstRelocation, 0, AnonSym1, 0);
  BA.addEdge(Edge::FirstRelocation, 4, AnonSym2, 0);
  BAnon1.addEdge(Edge::FirstRelocation, 0, SymB, 0);
  BAnon2.addEdge(Edge::FirstRelocation, 0, SymB, 0);

  auto DGs = calculateDepGroups(*G);
  ASSERT_EQ(DGs.size(), 1u);
  EXPECT_EQ(DGs[0].Defs.size(), 1u);
  EXPECT_EQ(DGs[0].Defs[0]->getName(), G->intern("A"));
  EXPECT_EQ(DGs[0].Deps.size(), 1u);
  EXPECT_TRUE(DGs[0].Deps.count(&SymB));
}

// Cycle through anonymous blocks: A -> anon -> A.
// The anonymous block creates a path back to A, but since A is a named symbol
// the back-edge is a symbol dep (not an anonymous block dep). The SCC graph
// is A -> anon (no back-edge). The anon block's dep on A gets transitively
// merged into A's dep group, then filtered out (since A is a def), leaving
// an empty dep set. The group is then removed entirely.
TEST_F(LinkGraphLinkingLayerTests, CycleThroughAnonymousBlock) {
  auto G = makeGraph();
  auto &Sec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &BA =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x1000), 8, 0);
  auto &BAnon =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x2000), 8, 0);
  auto &SymA = G->addDefinedSymbol(BA, 0, "A", 4, Linkage::Strong,
                                   Scope::Default, false, true);
  auto &AnonSym = G->addAnonymousSymbol(BAnon, 0, 4, false, true);

  BA.addEdge(Edge::FirstRelocation, 0, AnonSym, 0);
  BAnon.addEdge(Edge::FirstRelocation, 0, SymA, 0);

  auto DGs = calculateDepGroups(*G);
  // A's self-dep gets filtered, leaving an empty dep set. Groups with empty
  // deps are removed, so no dep groups remain.
  EXPECT_TRUE(DGs.empty());
}

// Cycle through anonymous blocks with an external dep hanging off the anon
// block: A -> anon -> A, anon -> ext. A should depend on ext.
TEST_F(LinkGraphLinkingLayerTests, CycleThroughAnonymousBlockWithExternalDep) {
  auto G = makeGraph();
  auto &Sec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &BA =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x1000), 8, 0);
  auto &BAnon =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x2000), 8, 0);
  auto &SymA = G->addDefinedSymbol(BA, 0, "A", 4, Linkage::Strong,
                                   Scope::Default, false, true);
  auto &AnonSym = G->addAnonymousSymbol(BAnon, 0, 4, false, true);
  auto &ExtSym = G->addExternalSymbol("ext", 0, false);

  BA.addEdge(Edge::FirstRelocation, 0, AnonSym, 0);
  BAnon.addEdge(Edge::FirstRelocation, 0, SymA, 0);
  BAnon.addEdge(Edge::FirstRelocation, 4, ExtSym, 0);

  auto DGs = calculateDepGroups(*G);
  ASSERT_EQ(DGs.size(), 1u);
  EXPECT_EQ(DGs[0].Defs.size(), 1u);
  EXPECT_EQ(DGs[0].Defs[0]->getName(), G->intern("A"));
  EXPECT_EQ(DGs[0].Deps.size(), 1u);
  EXPECT_TRUE(DGs[0].Deps.count(&ExtSym));
}

// Two separate dep groups: A -> ext1, B -> ext2 (no connection between A and
// B).
TEST_F(LinkGraphLinkingLayerTests, TwoSeparateDepGroups) {
  auto G = makeGraph();
  auto &Sec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &BA =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x1000), 8, 0);
  auto &BB =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x2000), 8, 0);
  G->addDefinedSymbol(BA, 0, "A", 4, Linkage::Strong, Scope::Default, false,
                      true);
  G->addDefinedSymbol(BB, 0, "B", 4, Linkage::Strong, Scope::Default, false,
                      true);
  auto &Ext1 = G->addExternalSymbol("ext1", 0, false);
  auto &Ext2 = G->addExternalSymbol("ext2", 0, false);
  BA.addEdge(Edge::FirstRelocation, 0, Ext1, 0);
  BB.addEdge(Edge::FirstRelocation, 0, Ext2, 0);

  auto DGs = calculateDepGroups(*G);
  ASSERT_EQ(DGs.size(), 2u);

  // Find which group is which (order may vary).
  SymbolDepGroup *DGA = nullptr, *DGB = nullptr;
  for (auto &DG : DGs) {
    ASSERT_EQ(DG.Defs.size(), 1u);
    if (DG.Defs[0]->getName() == G->intern("A"))
      DGA = &DG;
    else if (DG.Defs[0]->getName() == G->intern("B"))
      DGB = &DG;
  }
  ASSERT_NE(DGA, nullptr);
  ASSERT_NE(DGB, nullptr);
  EXPECT_EQ(DGA->Deps.size(), 1u);
  EXPECT_TRUE(DGA->Deps.count(&Ext1));
  EXPECT_EQ(DGB->Deps.size(), 1u);
  EXPECT_TRUE(DGB->Deps.count(&Ext2));
}

// Dep group merging: A and B both transitively depend on ext through the same
// anonymous block, and neither BA nor BB introduces any additional deps of its
// own. This allows the second block processed to merge into the first's dep
// group via the single-source-group fast path.
// A -> anon -> ext
// B -> anon -> ext  (same anon block)
// A and B should be in the same dep group.
TEST_F(LinkGraphLinkingLayerTests, SharedAnonymousBlockMergesDepGroups) {
  auto G = makeGraph();
  auto &Sec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &BA =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x1000), 8, 0);
  auto &BB =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x2000), 8, 0);
  auto &BAnon =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x3000), 8, 0);
  G->addDefinedSymbol(BA, 0, "A", 4, Linkage::Strong, Scope::Default, false,
                      true);
  G->addDefinedSymbol(BB, 0, "B", 4, Linkage::Strong, Scope::Default, false,
                      true);
  auto &AnonSym = G->addAnonymousSymbol(BAnon, 0, 4, false, true);
  auto &ExtSym = G->addExternalSymbol("ext", 0, false);

  BA.addEdge(Edge::FirstRelocation, 0, AnonSym, 0);
  BB.addEdge(Edge::FirstRelocation, 0, AnonSym, 0);
  BAnon.addEdge(Edge::FirstRelocation, 0, ExtSym, 0);

  auto DGs = calculateDepGroups(*G);
  ASSERT_EQ(DGs.size(), 1u);
  EXPECT_EQ(DGs[0].Defs.size(), 2u);
  EXPECT_EQ(DGs[0].Deps.size(), 1u);
  EXPECT_TRUE(DGs[0].Deps.count(&ExtSym));
}

// Same as SharedAnonymousBlockMergesDepGroups, but BA and BB each introduce
// an additional direct dep (even the same one). This prevents merging — each
// block triggers the general-case merge path and gets its own dep group.
// A -> anon -> ext1, A -> ext2
// B -> anon -> ext1, B -> ext2
TEST_F(LinkGraphLinkingLayerTests,
       SharedAnonBlockWithExtraDepsSeparatesGroups) {
  auto G = makeGraph();
  auto &Sec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &BA =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x1000), 8, 0);
  auto &BB =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x2000), 8, 0);
  auto &BAnon =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x3000), 8, 0);
  G->addDefinedSymbol(BA, 0, "A", 4, Linkage::Strong, Scope::Default, false,
                      true);
  G->addDefinedSymbol(BB, 0, "B", 4, Linkage::Strong, Scope::Default, false,
                      true);
  auto &AnonSym = G->addAnonymousSymbol(BAnon, 0, 4, false, true);
  auto &Ext1 = G->addExternalSymbol("ext1", 0, false);
  auto &Ext2 = G->addExternalSymbol("ext2", 0, false);

  BA.addEdge(Edge::FirstRelocation, 0, AnonSym, 0);
  BA.addEdge(Edge::FirstRelocation, 4, Ext2, 0);
  BB.addEdge(Edge::FirstRelocation, 0, AnonSym, 0);
  BB.addEdge(Edge::FirstRelocation, 4, Ext2, 0);
  BAnon.addEdge(Edge::FirstRelocation, 0, Ext1, 0);

  auto DGs = calculateDepGroups(*G);
  ASSERT_EQ(DGs.size(), 2u);

  SymbolDepGroup *DGA = nullptr, *DGB = nullptr;
  for (auto &DG : DGs) {
    ASSERT_EQ(DG.Defs.size(), 1u);
    if (DG.Defs[0]->getName() == G->intern("A"))
      DGA = &DG;
    else if (DG.Defs[0]->getName() == G->intern("B"))
      DGB = &DG;
  }
  ASSERT_NE(DGA, nullptr);
  ASSERT_NE(DGB, nullptr);
  // Both groups should have ext1 (transitive) and ext2 (direct).
  EXPECT_EQ(DGA->Deps.size(), 2u);
  EXPECT_TRUE(DGA->Deps.count(&Ext1));
  EXPECT_TRUE(DGA->Deps.count(&Ext2));
  EXPECT_EQ(DGB->Deps.size(), 2u);
  EXPECT_TRUE(DGB->Deps.count(&Ext1));
  EXPECT_TRUE(DGB->Deps.count(&Ext2));
}

// Multi-node SCC in the anonymous block graph:
//   A -> anon1 -> anon2 -> anon1 (cycle), anon2 -> ext
// anon1 and anon2 form a two-node SCC. A depends transitively on ext through
// that SCC.
TEST_F(LinkGraphLinkingLayerTests, MultiNodeSCC) {
  auto G = makeGraph();
  auto &Sec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &BA =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x1000), 8, 0);
  auto &BAnon1 =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x2000), 8, 0);
  auto &BAnon2 =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x3000), 8, 0);

  G->addDefinedSymbol(BA, 0, "A", 4, Linkage::Strong, Scope::Default, false,
                      true);
  auto &AnonSym1 = G->addAnonymousSymbol(BAnon1, 0, 4, false, true);
  auto &AnonSym2 = G->addAnonymousSymbol(BAnon2, 0, 4, false, true);
  auto &ExtSym = G->addExternalSymbol("ext", 0, false);

  BA.addEdge(Edge::FirstRelocation, 0, AnonSym1, 0);
  BAnon1.addEdge(Edge::FirstRelocation, 0, AnonSym2, 0);
  BAnon2.addEdge(Edge::FirstRelocation, 0, AnonSym1, 0); // cycle
  BAnon2.addEdge(Edge::FirstRelocation, 4, ExtSym, 0);

  auto DGs = calculateDepGroups(*G);
  ASSERT_EQ(DGs.size(), 1u);
  EXPECT_EQ(DGs[0].Defs.size(), 1u);
  EXPECT_EQ(DGs[0].Defs[0]->getName(), G->intern("A"));
  EXPECT_EQ(DGs[0].Deps.size(), 1u);
  EXPECT_TRUE(DGs[0].Deps.count(&ExtSym));
}

// Multi-node SCC where different nodes in the cycle contribute different
// external deps:
//   A -> anon1 -> anon2 -> anon1 (cycle), anon1 -> ext1, anon2 -> ext2
// Both ext1 and ext2 should be merged into A's dep group.
TEST_F(LinkGraphLinkingLayerTests, MultiNodeSCCMergesExternalDeps) {
  auto G = makeGraph();
  auto &Sec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &BA =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x1000), 8, 0);
  auto &BAnon1 =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x2000), 8, 0);
  auto &BAnon2 =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x3000), 8, 0);

  G->addDefinedSymbol(BA, 0, "A", 4, Linkage::Strong, Scope::Default, false,
                      true);
  auto &AnonSym1 = G->addAnonymousSymbol(BAnon1, 0, 4, false, true);
  auto &AnonSym2 = G->addAnonymousSymbol(BAnon2, 0, 4, false, true);
  auto &Ext1 = G->addExternalSymbol("ext1", 0, false);
  auto &Ext2 = G->addExternalSymbol("ext2", 0, false);

  BA.addEdge(Edge::FirstRelocation, 0, AnonSym1, 0);
  BAnon1.addEdge(Edge::FirstRelocation, 0, AnonSym2, 0);
  BAnon1.addEdge(Edge::FirstRelocation, 4, Ext1, 0);
  BAnon2.addEdge(Edge::FirstRelocation, 0, AnonSym1, 0); // cycle
  BAnon2.addEdge(Edge::FirstRelocation, 4, Ext2, 0);

  auto DGs = calculateDepGroups(*G);
  ASSERT_EQ(DGs.size(), 1u);
  EXPECT_EQ(DGs[0].Defs.size(), 1u);
  EXPECT_EQ(DGs[0].Defs[0]->getName(), G->intern("A"));
  EXPECT_EQ(DGs[0].Deps.size(), 2u);
  EXPECT_TRUE(DGs[0].Deps.count(&Ext1));
  EXPECT_TRUE(DGs[0].Deps.count(&Ext2));
}

// Multi-node SCC where nodes in the cycle contribute a mix of external and
// named-block deps:
//   A -> anon1 -> anon2 -> anon1 (cycle), anon1 -> B (named), anon2 -> ext
// A should depend on both B and ext.
TEST_F(LinkGraphLinkingLayerTests, MultiNodeSCCMergesMixedDeps) {
  auto G = makeGraph();
  auto &Sec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &BA =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x1000), 8, 0);
  auto &BAnon1 =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x2000), 8, 0);
  auto &BAnon2 =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x3000), 8, 0);
  auto &BB =
      G->createContentBlock(Sec, BlockContent, ExecutorAddr(0x4000), 8, 0);

  G->addDefinedSymbol(BA, 0, "A", 4, Linkage::Strong, Scope::Default, false,
                      true);
  auto &AnonSym1 = G->addAnonymousSymbol(BAnon1, 0, 4, false, true);
  auto &AnonSym2 = G->addAnonymousSymbol(BAnon2, 0, 4, false, true);
  auto &SymB = G->addDefinedSymbol(BB, 0, "B", 4, Linkage::Strong,
                                   Scope::Default, false, true);
  auto &ExtSym = G->addExternalSymbol("ext", 0, false);

  BA.addEdge(Edge::FirstRelocation, 0, AnonSym1, 0);
  BAnon1.addEdge(Edge::FirstRelocation, 0, AnonSym2, 0);
  BAnon1.addEdge(Edge::FirstRelocation, 4, SymB, 0);
  BAnon2.addEdge(Edge::FirstRelocation, 0, AnonSym1, 0); // cycle
  BAnon2.addEdge(Edge::FirstRelocation, 4, ExtSym, 0);

  auto DGs = calculateDepGroups(*G);
  ASSERT_EQ(DGs.size(), 1u);
  EXPECT_EQ(DGs[0].Defs.size(), 1u);
  EXPECT_EQ(DGs[0].Defs[0]->getName(), G->intern("A"));
  EXPECT_EQ(DGs[0].Deps.size(), 2u);
  EXPECT_TRUE(DGs[0].Deps.count(&SymB));
  EXPECT_TRUE(DGs[0].Deps.count(&ExtSym));
}
