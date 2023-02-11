//===------ LinkGraphTests.cpp - Unit tests for core JITLink classes ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/Orc/ObjectFileInterface.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Memory.h"

#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::jitlink;

static const char BlockContentBytes[] = {
    0x54, 0x68, 0x65, 0x72, 0x65, 0x20, 0x77, 0x61, 0x73, 0x20, 0x6d, 0x6f,
    0x76, 0x65, 0x6d, 0x65, 0x6e, 0x74, 0x20, 0x61, 0x74, 0x20, 0x74, 0x68,
    0x65, 0x20, 0x73, 0x74, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x2c, 0x20, 0x66,
    0x6f, 0x72, 0x20, 0x74, 0x68, 0x65, 0x20, 0x77, 0x6f, 0x72, 0x64, 0x20,
    0x68, 0x61, 0x64, 0x20, 0x70, 0x61, 0x73, 0x73, 0x65, 0x64, 0x20, 0x61,
    0x72, 0x6f, 0x75, 0x6e, 0x64, 0x0a, 0x54, 0x68, 0x61, 0x74, 0x20, 0x74,
    0x68, 0x65, 0x20, 0x63, 0x6f, 0x6c, 0x74, 0x20, 0x66, 0x72, 0x6f, 0x6d,
    0x20, 0x4f, 0x6c, 0x64, 0x20, 0x52, 0x65, 0x67, 0x72, 0x65, 0x74, 0x20,
    0x68, 0x61, 0x64, 0x20, 0x67, 0x6f, 0x74, 0x20, 0x61, 0x77, 0x61, 0x79,
    0x2c, 0x0a, 0x41, 0x6e, 0x64, 0x20, 0x68, 0x61, 0x64, 0x20, 0x6a, 0x6f,
    0x69, 0x6e, 0x65, 0x64, 0x20, 0x74, 0x68, 0x65, 0x20, 0x77, 0x69, 0x6c,
    0x64, 0x20, 0x62, 0x75, 0x73, 0x68, 0x20, 0x68, 0x6f, 0x72, 0x73, 0x65,
    0x73, 0x20, 0x2d, 0x2d, 0x20, 0x68, 0x65, 0x20, 0x77, 0x61, 0x73, 0x20,
    0x77, 0x6f, 0x72, 0x74, 0x68, 0x20, 0x61, 0x20, 0x74, 0x68, 0x6f, 0x75,
    0x73, 0x61, 0x6e, 0x64, 0x20, 0x70, 0x6f, 0x75, 0x6e, 0x64, 0x2c, 0x0a,
    0x53, 0x6f, 0x20, 0x61, 0x6c, 0x6c, 0x20, 0x74, 0x68, 0x65, 0x20, 0x63,
    0x72, 0x61, 0x63, 0x6b, 0x73, 0x20, 0x68, 0x61, 0x64, 0x20, 0x67, 0x61,
    0x74, 0x68, 0x65, 0x72, 0x65, 0x64, 0x20, 0x74, 0x6f, 0x20, 0x74, 0x68,
    0x65, 0x20, 0x66, 0x72, 0x61, 0x79, 0x2e, 0x0a, 0x41, 0x6c, 0x6c, 0x20,
    0x74, 0x68, 0x65, 0x20, 0x74, 0x72, 0x69, 0x65, 0x64, 0x20, 0x61, 0x6e,
    0x64, 0x20, 0x6e, 0x6f, 0x74, 0x65, 0x64, 0x20, 0x72, 0x69, 0x64, 0x65,
    0x72, 0x73, 0x20, 0x66, 0x72, 0x6f, 0x6d, 0x20, 0x74, 0x68, 0x65, 0x20,
    0x73, 0x74, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x20, 0x6e, 0x65, 0x61,
    0x72, 0x20, 0x61, 0x6e, 0x64, 0x20, 0x66, 0x61, 0x72, 0x0a, 0x48, 0x61,
    0x64, 0x20, 0x6d, 0x75, 0x73, 0x74, 0x65, 0x72, 0x65, 0x64, 0x20, 0x61,
    0x74, 0x20, 0x74, 0x68, 0x65, 0x20, 0x68, 0x6f, 0x6d, 0x65, 0x73, 0x74,
    0x65, 0x61, 0x64, 0x20, 0x6f, 0x76, 0x65, 0x72, 0x6e, 0x69, 0x67, 0x68,
    0x74, 0x2c, 0x0a, 0x46, 0x6f, 0x72, 0x20, 0x74, 0x68, 0x65, 0x20, 0x62,
    0x75, 0x73, 0x68, 0x6d, 0x65, 0x6e, 0x20, 0x6c, 0x6f, 0x76, 0x65, 0x20,
    0x68, 0x61, 0x72, 0x64, 0x20, 0x72, 0x69, 0x64, 0x69, 0x6e, 0x67, 0x20,
    0x77, 0x68, 0x65, 0x72, 0x65, 0x20, 0x74, 0x68, 0x65, 0x20, 0x77, 0x69,
    0x6c, 0x64, 0x20, 0x62, 0x75, 0x73, 0x68, 0x20, 0x68, 0x6f, 0x72, 0x73,
    0x65, 0x73, 0x20, 0x61, 0x72, 0x65, 0x2c, 0x0a, 0x41, 0x6e, 0x64, 0x20,
    0x74, 0x68, 0x65, 0x20, 0x73, 0x74, 0x6f, 0x63, 0x6b, 0x2d, 0x68, 0x6f,
    0x72, 0x73, 0x65, 0x20, 0x73, 0x6e, 0x75, 0x66, 0x66, 0x73, 0x20, 0x74,
    0x68, 0x65, 0x20, 0x62, 0x61, 0x74, 0x74, 0x6c, 0x65, 0x20, 0x77, 0x69,
    0x74, 0x68, 0x20, 0x64, 0x65, 0x6c, 0x69, 0x67, 0x68, 0x74, 0x2e, 0x00};

static ArrayRef<char> BlockContent(BlockContentBytes);

TEST(LinkGraphTest, Construction) {
  // Check that LinkGraph construction works as expected.
  LinkGraph G("foo", Triple("x86_64-apple-darwin"), 8, support::little,
              getGenericEdgeKindName);
  EXPECT_EQ(G.getName(), "foo");
  EXPECT_EQ(G.getTargetTriple().str(), "x86_64-apple-darwin");
  EXPECT_EQ(G.getPointerSize(), 8U);
  EXPECT_EQ(G.getEndianness(), support::little);
  EXPECT_TRUE(G.external_symbols().empty());
  EXPECT_TRUE(G.absolute_symbols().empty());
  EXPECT_TRUE(G.defined_symbols().empty());
  EXPECT_TRUE(G.blocks().empty());
}

TEST(LinkGraphTest, AddressAccess) {
  // Check that we can get addresses for blocks, symbols, and edges.
  LinkGraph G("foo", Triple("x86_64-apple-darwin"), 8, support::little,
              getGenericEdgeKindName);

  auto &Sec1 =
      G.createSection("__data.1", orc::MemProt::Read | orc::MemProt::Write);
  orc::ExecutorAddr B1Addr(0x1000);
  auto &B1 = G.createContentBlock(Sec1, BlockContent, B1Addr, 8, 0);
  auto &S1 = G.addDefinedSymbol(B1, 4, "S1", 4, Linkage::Strong, Scope::Default,
                                false, false);
  B1.addEdge(Edge::FirstRelocation, 8, S1, 0);
  auto &E1 = *B1.edges().begin();

  EXPECT_EQ(B1.getAddress(), B1Addr) << "Incorrect block address";
  EXPECT_EQ(S1.getAddress(), B1Addr + 4) << "Incorrect symbol address";
  EXPECT_EQ(B1.getFixupAddress(E1), B1Addr + 8) << "Incorrect fixup address";
}

TEST(LinkGraphTest, BlockAndSymbolIteration) {
  // Check that we can iterate over blocks within Sections and across sections.
  LinkGraph G("foo", Triple("x86_64-apple-darwin"), 8, support::little,
              getGenericEdgeKindName);
  auto &Sec1 =
      G.createSection("__data.1", orc::MemProt::Read | orc::MemProt::Write);
  orc::ExecutorAddr B1Addr(0x1000);
  auto &B1 = G.createContentBlock(Sec1, BlockContent, B1Addr, 8, 0);
  orc::ExecutorAddr B2Addr(0x2000);
  auto &B2 = G.createContentBlock(Sec1, BlockContent, B2Addr, 8, 0);
  auto &S1 = G.addDefinedSymbol(B1, 0, "S1", 4, Linkage::Strong, Scope::Default,
                                false, false);
  auto &S2 = G.addDefinedSymbol(B2, 4, "S2", 4, Linkage::Strong, Scope::Default,
                                false, false);

  auto &Sec2 =
      G.createSection("__data.2", orc::MemProt::Read | orc::MemProt::Write);
  orc::ExecutorAddr B3Addr(0x3000);
  auto &B3 = G.createContentBlock(Sec2, BlockContent, B3Addr, 8, 0);
  orc::ExecutorAddr B4Addr(0x4000);
  auto &B4 = G.createContentBlock(Sec2, BlockContent, B4Addr, 8, 0);
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

TEST(LinkGraphTest, ContentAccessAndUpdate) {
  // Check that we can make a defined symbol external.
  LinkGraph G("foo", Triple("x86_64-apple-darwin"), 8, support::little,
              getGenericEdgeKindName);
  auto &Sec =
      G.createSection("__data", orc::MemProt::Read | orc::MemProt::Write);

  // Create an initial block.
  orc::ExecutorAddr BAddr(0x1000);
  auto &B = G.createContentBlock(Sec, BlockContent, BAddr, 8, 0);

  EXPECT_FALSE(B.isContentMutable()) << "Content unexpectedly mutable";
  EXPECT_EQ(B.getContent().data(), BlockContent.data())
      << "Unexpected block content data pointer";
  EXPECT_EQ(B.getContent().size(), BlockContent.size())
      << "Unexpected block content size";

  // Expect that attempting to get already-mutable content fails if the
  // content is not yet mutable (debug builds only).
#ifndef NDEBUG
  EXPECT_DEATH({ (void)B.getAlreadyMutableContent(); },
               "Content is not mutable")
      << "Unexpected mutable access allowed to immutable data";
#endif

  // Check that mutable content is copied on request as expected.
  auto MutableContent = B.getMutableContent(G);
  EXPECT_TRUE(B.isContentMutable()) << "Content unexpectedly immutable";
  EXPECT_NE(MutableContent.data(), BlockContent.data())
      << "Unexpected mutable content data pointer";
  EXPECT_EQ(MutableContent.size(), BlockContent.size())
      << "Unexpected mutable content size";
  EXPECT_TRUE(std::equal(MutableContent.begin(), MutableContent.end(),
                         BlockContent.begin()))
      << "Unexpected mutable content value";

  // Check that already-mutable content behaves as expected, with no
  // further copies.
  auto MutableContent2 = B.getMutableContent(G);
  EXPECT_TRUE(B.isContentMutable()) << "Content unexpectedly immutable";
  EXPECT_EQ(MutableContent2.data(), MutableContent.data())
      << "Unexpected mutable content 2 data pointer";
  EXPECT_EQ(MutableContent2.size(), MutableContent.size())
      << "Unexpected mutable content 2 size";

  // Check that getAlreadyMutableContent behaves as expected, with no
  // further copies.
  auto MutableContent3 = B.getMutableContent(G);
  EXPECT_TRUE(B.isContentMutable()) << "Content unexpectedly immutable";
  EXPECT_EQ(MutableContent3.data(), MutableContent.data())
      << "Unexpected mutable content 2 data pointer";
  EXPECT_EQ(MutableContent3.size(), MutableContent.size())
      << "Unexpected mutable content 2 size";

  // Check that we can obtain a writer and reader over the content.
  // Check that we can get a BinaryStreamReader for B.
  auto Writer = G.getBlockContentWriter(B);
  EXPECT_THAT_ERROR(Writer.writeInteger((uint32_t)0xcafef00d), Succeeded());

  auto Reader = G.getBlockContentReader(B);
  uint32_t Initial32Bits = 0;
  EXPECT_THAT_ERROR(Reader.readInteger(Initial32Bits), Succeeded());
  EXPECT_EQ(Initial32Bits, (uint32_t)0xcafef00d);

  // Set content back to immutable and check that everything behaves as
  // expected again.
  B.setContent(BlockContent);
  EXPECT_FALSE(B.isContentMutable()) << "Content unexpectedly mutable";
  EXPECT_EQ(B.getContent().data(), BlockContent.data())
      << "Unexpected block content data pointer";
  EXPECT_EQ(B.getContent().size(), BlockContent.size())
      << "Unexpected block content size";

  // Create an initially mutable block.
  auto &B2 = G.createMutableContentBlock(Sec, MutableContent,
                                         orc::ExecutorAddr(0x10000), 8, 0);

  EXPECT_TRUE(B2.isContentMutable()) << "Expected B2 content to be mutable";
}

TEST(LinkGraphTest, MakeExternal) {
  // Check that we can make defined and absolute symbols external.
  LinkGraph G("foo", Triple("x86_64-apple-darwin"), 8, support::little,
              getGenericEdgeKindName);
  auto &Sec =
      G.createSection("__data", orc::MemProt::Read | orc::MemProt::Write);

  // Create an initial block.
  auto &B1 =
      G.createContentBlock(Sec, BlockContent, orc::ExecutorAddr(0x1000), 8, 0);

  // Add a symbol to the block.
  auto &S1 = G.addDefinedSymbol(B1, 0, "S1", 4, Linkage::Strong, Scope::Default,
                                false, false);

  EXPECT_TRUE(S1.isDefined()) << "Symbol should be defined";
  EXPECT_FALSE(S1.isExternal()) << "Symbol should not be external";
  EXPECT_FALSE(S1.isAbsolute()) << "Symbol should not be absolute";
  EXPECT_TRUE(&S1.getBlock()) << "Symbol should have a non-null block";
  EXPECT_EQ(S1.getAddress(), orc::ExecutorAddr(0x1000))
      << "Unexpected symbol address";

  EXPECT_EQ(
      std::distance(G.defined_symbols().begin(), G.defined_symbols().end()), 1U)
      << "Unexpected number of defined symbols";
  EXPECT_EQ(
      std::distance(G.external_symbols().begin(), G.external_symbols().end()),
      0U)
      << "Unexpected number of external symbols";

  // Add an absolute symbol.
  auto &S2 = G.addAbsoluteSymbol("S2", orc::ExecutorAddr(0x2000), 0,
                                 Linkage::Strong, Scope::Default, true);

  EXPECT_TRUE(S2.isAbsolute()) << "Symbol should be absolute";
  EXPECT_EQ(
      std::distance(G.absolute_symbols().begin(), G.absolute_symbols().end()),
      1U)
      << "Unexpected number of symbols";

  // Make S1 and S2 external, confirm that the its flags are updated and that it
  // is moved from the defined/absolute symbols lists to the externals list.
  G.makeExternal(S1);
  G.makeExternal(S2);

  EXPECT_FALSE(S1.isDefined()) << "Symbol should not be defined";
  EXPECT_TRUE(S1.isExternal()) << "Symbol should be external";
  EXPECT_FALSE(S1.isAbsolute()) << "Symbol should not be absolute";
  EXPECT_FALSE(S2.isDefined()) << "Symbol should not be defined";
  EXPECT_TRUE(S2.isExternal()) << "Symbol should be external";
  EXPECT_FALSE(S2.isAbsolute()) << "Symbol should not be absolute";

  EXPECT_EQ(S1.getAddress(), orc::ExecutorAddr())
      << "Unexpected symbol address";
  EXPECT_EQ(S2.getAddress(), orc::ExecutorAddr())
      << "Unexpected symbol address";

  EXPECT_EQ(
      std::distance(G.defined_symbols().begin(), G.defined_symbols().end()), 0U)
      << "Unexpected number of defined symbols";
  EXPECT_EQ(
      std::distance(G.external_symbols().begin(), G.external_symbols().end()),
      2U)
      << "Unexpected number of external symbols";
  EXPECT_EQ(
      std::distance(G.absolute_symbols().begin(), G.absolute_symbols().end()),
      0U)
      << "Unexpected number of external symbols";
}

TEST(LinkGraphTest, MakeAbsolute) {
  // Check that we can make defined and external symbols absolute.
  LinkGraph G("foo", Triple("x86_64-apple-darwin"), 8, support::little,
              getGenericEdgeKindName);
  auto &Sec =
      G.createSection("__data", orc::MemProt::Read | orc::MemProt::Write);

  // Create an initial block.
  auto &B1 =
      G.createContentBlock(Sec, BlockContent, orc::ExecutorAddr(0x1000), 8, 0);

  // Add a symbol to the block.
  auto &S1 = G.addDefinedSymbol(B1, 0, "S1", 4, Linkage::Strong, Scope::Default,
                                false, false);

  EXPECT_TRUE(S1.isDefined()) << "Symbol should be defined";
  EXPECT_FALSE(S1.isExternal()) << "Symbol should not be external";
  EXPECT_FALSE(S1.isAbsolute()) << "Symbol should not be absolute";
  EXPECT_TRUE(&S1.getBlock()) << "Symbol should have a non-null block";
  EXPECT_EQ(S1.getAddress(), orc::ExecutorAddr(0x1000))
      << "Unexpected symbol address";

  EXPECT_EQ(
      std::distance(G.defined_symbols().begin(), G.defined_symbols().end()), 1U)
      << "Unexpected number of defined symbols";
  EXPECT_EQ(
      std::distance(G.external_symbols().begin(), G.external_symbols().end()),
      0U)
      << "Unexpected number of external symbols";

  // Add an external symbol.
  auto &S2 = G.addExternalSymbol("S2", 0, true);

  EXPECT_TRUE(S2.isExternal()) << "Symbol should be external";
  EXPECT_EQ(
      std::distance(G.external_symbols().begin(), G.external_symbols().end()),
      1U)
      << "Unexpected number of symbols";

  // Make S1 and S2 absolute, confirm that the its flags are updated and that it
  // is moved from the defined/external symbols lists to the absolutes list.
  orc::ExecutorAddr S1AbsAddr(0xA000);
  orc::ExecutorAddr S2AbsAddr(0xB000);
  G.makeAbsolute(S1, S1AbsAddr);
  G.makeAbsolute(S2, S2AbsAddr);

  EXPECT_FALSE(S1.isDefined()) << "Symbol should not be defined";
  EXPECT_FALSE(S1.isExternal()) << "Symbol should not be external";
  EXPECT_TRUE(S1.isAbsolute()) << "Symbol should be absolute";
  EXPECT_FALSE(S2.isDefined()) << "Symbol should not be defined";
  EXPECT_FALSE(S2.isExternal()) << "Symbol should not be absolute";
  EXPECT_TRUE(S2.isAbsolute()) << "Symbol should be absolute";

  EXPECT_EQ(S1.getAddress(), S1AbsAddr) << "Unexpected symbol address";
  EXPECT_EQ(S2.getAddress(), S2AbsAddr) << "Unexpected symbol address";

  EXPECT_EQ(
      std::distance(G.defined_symbols().begin(), G.defined_symbols().end()), 0U)
      << "Unexpected number of defined symbols";
  EXPECT_EQ(
      std::distance(G.external_symbols().begin(), G.external_symbols().end()),
      0U)
      << "Unexpected number of external symbols";
  EXPECT_EQ(
      std::distance(G.absolute_symbols().begin(), G.absolute_symbols().end()),
      2U)
      << "Unexpected number of external symbols";
}

TEST(LinkGraphTest, MakeDefined) {
  // Check that we can make an external symbol defined.
  LinkGraph G("foo", Triple("x86_64-apple-darwin"), 8, support::little,
              getGenericEdgeKindName);
  auto &Sec =
      G.createSection("__data", orc::MemProt::Read | orc::MemProt::Write);

  // Create an initial block.
  orc::ExecutorAddr B1Addr(0x1000);
  auto &B1 = G.createContentBlock(Sec, BlockContent, B1Addr, 8, 0);

  // Add an external symbol.
  auto &S1 = G.addExternalSymbol("S1", 4, true);

  EXPECT_FALSE(S1.isDefined()) << "Symbol should not be defined";
  EXPECT_TRUE(S1.isExternal()) << "Symbol should be external";
  EXPECT_FALSE(S1.isAbsolute()) << "Symbol should not be absolute";
  EXPECT_EQ(S1.getAddress(), orc::ExecutorAddr())
      << "Unexpected symbol address";

  EXPECT_EQ(
      std::distance(G.defined_symbols().begin(), G.defined_symbols().end()), 0U)
      << "Unexpected number of defined symbols";
  EXPECT_EQ(
      std::distance(G.external_symbols().begin(), G.external_symbols().end()),
      1U)
      << "Unexpected number of external symbols";

  // Make S1 defined, confirm that its flags are updated and that it is
  // moved from the defined symbols to the externals list.
  G.makeDefined(S1, B1, 0, 4, Linkage::Strong, Scope::Default, false);

  EXPECT_TRUE(S1.isDefined()) << "Symbol should be defined";
  EXPECT_FALSE(S1.isExternal()) << "Symbol should not be external";
  EXPECT_FALSE(S1.isAbsolute()) << "Symbol should not be absolute";
  EXPECT_TRUE(&S1.getBlock()) << "Symbol should have a non-null block";
  EXPECT_EQ(S1.getAddress(), orc::ExecutorAddr(0x1000U))
      << "Unexpected symbol address";

  EXPECT_EQ(
      std::distance(G.defined_symbols().begin(), G.defined_symbols().end()), 1U)
      << "Unexpected number of defined symbols";
  EXPECT_EQ(
      std::distance(G.external_symbols().begin(), G.external_symbols().end()),
      0U)
      << "Unexpected number of external symbols";
}

TEST(LinkGraphTest, TransferDefinedSymbol) {
  // Check that we can transfer a defined symbol from one block to another.
  LinkGraph G("foo", Triple("x86_64-apple-darwin"), 8, support::little,
              getGenericEdgeKindName);
  auto &Sec =
      G.createSection("__data", orc::MemProt::Read | orc::MemProt::Write);

  // Create initial blocks.
  orc::ExecutorAddr B1Addr(0x1000);
  auto &B1 = G.createContentBlock(Sec, BlockContent, B1Addr, 8, 0);
  orc::ExecutorAddr B2Addr(0x2000);
  auto &B2 = G.createContentBlock(Sec, BlockContent, B2Addr, 8, 0);
  orc::ExecutorAddr B3Addr(0x3000);
  auto &B3 = G.createContentBlock(Sec, BlockContent.slice(0, 32), B3Addr, 8, 0);

  // Add a symbol.
  auto &S1 = G.addDefinedSymbol(B1, 0, "S1", B1.getSize(), Linkage::Strong,
                                Scope::Default, false, false);

  // Transfer with zero offset, explicit size.
  G.transferDefinedSymbol(S1, B2, 0, 64);

  EXPECT_EQ(&S1.getBlock(), &B2) << "Block was not updated";
  EXPECT_EQ(S1.getOffset(), 0U) << "Unexpected offset";
  EXPECT_EQ(S1.getSize(), 64U) << "Size was not updated";

  // Transfer with non-zero offset, implicit truncation.
  G.transferDefinedSymbol(S1, B3, 16, std::nullopt);

  EXPECT_EQ(&S1.getBlock(), &B3) << "Block was not updated";
  EXPECT_EQ(S1.getOffset(), 16U) << "Offset was not updated";
  EXPECT_EQ(S1.getSize(), 16U) << "Size was not updated";
}

TEST(LinkGraphTest, TransferDefinedSymbolAcrossSections) {
  // Check that we can transfer a defined symbol from an existing block in one
  // section to another.
  LinkGraph G("foo", Triple("x86_64-apple-darwin"), 8, support::little,
              getGenericEdgeKindName);
  auto &Sec1 =
      G.createSection("__data.1", orc::MemProt::Read | orc::MemProt::Write);
  auto &Sec2 =
      G.createSection("__data.2", orc::MemProt::Read | orc::MemProt::Write);

  // Create blocks in each section.
  orc::ExecutorAddr B1Addr(0x1000);
  auto &B1 = G.createContentBlock(Sec1, BlockContent, B1Addr, 8, 0);
  orc::ExecutorAddr B2Addr(0x2000);
  auto &B2 = G.createContentBlock(Sec2, BlockContent, B2Addr, 8, 0);

  // Add a symbol to section 1.
  auto &S1 = G.addDefinedSymbol(B1, 0, "S1", B1.getSize(), Linkage::Strong,
                                Scope::Default, false, false);

  // Transfer with zero offset, explicit size to section 2.
  G.transferDefinedSymbol(S1, B2, 0, 64);

  EXPECT_EQ(&S1.getBlock(), &B2) << "Block was not updated";
  EXPECT_EQ(S1.getOffset(), 0U) << "Unexpected offset";
  EXPECT_EQ(S1.getSize(), 64U) << "Size was not updated";

  EXPECT_EQ(Sec1.symbols_size(), 0u) << "Symbol was not removed from Sec1";
  EXPECT_EQ(Sec2.symbols_size(), 1u) << "Symbol was not added to Sec2";
  if (Sec2.symbols_size() == 1) {
    EXPECT_EQ(*Sec2.symbols().begin(), &S1) << "Unexpected symbol";
  }
}

TEST(LinkGraphTest, TransferBlock) {
  // Check that we can transfer a block (and all associated symbols) from one
  // section to another.
  LinkGraph G("foo", Triple("x86_64-apple-darwin"), 8, support::little,
              getGenericEdgeKindName);
  auto &Sec1 =
      G.createSection("__data.1", orc::MemProt::Read | orc::MemProt::Write);
  auto &Sec2 =
      G.createSection("__data.2", orc::MemProt::Read | orc::MemProt::Write);

  // Create an initial block.
  orc::ExecutorAddr B1Addr(0x1000);
  auto &B1 = G.createContentBlock(Sec1, BlockContent, B1Addr, 8, 0);
  orc::ExecutorAddr B2Addr(0x2000);
  auto &B2 = G.createContentBlock(Sec1, BlockContent, B2Addr, 8, 0);

  // Add some symbols on B1...
  G.addDefinedSymbol(B1, 0, "S1", B1.getSize(), Linkage::Strong, Scope::Default,
                     false, false);
  G.addDefinedSymbol(B1, 1, "S2", B1.getSize() - 1, Linkage::Strong,
                     Scope::Default, false, false);

  // ... and on B2.
  G.addDefinedSymbol(B2, 0, "S3", B2.getSize(), Linkage::Strong, Scope::Default,
                     false, false);
  G.addDefinedSymbol(B2, 1, "S4", B2.getSize() - 1, Linkage::Strong,
                     Scope::Default, false, false);

  EXPECT_EQ(Sec1.blocks_size(), 2U) << "Expected two blocks in Sec1 initially";
  EXPECT_EQ(Sec1.symbols_size(), 4U)
      << "Expected four symbols in Sec1 initially";
  EXPECT_EQ(Sec2.blocks_size(), 0U) << "Expected zero blocks in Sec2 initially";
  EXPECT_EQ(Sec2.symbols_size(), 0U)
      << "Expected zero symbols in Sec2 initially";

  // Transfer with zero offset, explicit size.
  G.transferBlock(B1, Sec2);

  EXPECT_EQ(Sec1.blocks_size(), 1U)
      << "Expected one blocks in Sec1 after transfer";
  EXPECT_EQ(Sec1.symbols_size(), 2U)
      << "Expected two symbols in Sec1 after transfer";
  EXPECT_EQ(Sec2.blocks_size(), 1U)
      << "Expected one blocks in Sec2 after transfer";
  EXPECT_EQ(Sec2.symbols_size(), 2U)
      << "Expected two symbols in Sec2 after transfer";
}

TEST(LinkGraphTest, MergeSections) {
  // Check that we can transfer a block (and all associated symbols) from one
  // section to another.
  LinkGraph G("foo", Triple("x86_64-apple-darwin"), 8, support::little,
              getGenericEdgeKindName);
  auto &Sec1 =
      G.createSection("__data.1", orc::MemProt::Read | orc::MemProt::Write);
  auto &Sec2 =
      G.createSection("__data.2", orc::MemProt::Read | orc::MemProt::Write);
  auto &Sec3 =
      G.createSection("__data.3", orc::MemProt::Read | orc::MemProt::Write);

  // Create an initial block.
  orc::ExecutorAddr B1Addr(0x1000);
  auto &B1 = G.createContentBlock(Sec1, BlockContent, B1Addr, 8, 0);
  orc::ExecutorAddr B2Addr(0x2000);
  auto &B2 = G.createContentBlock(Sec2, BlockContent, B2Addr, 8, 0);
  orc::ExecutorAddr B3Addr(0x3000);
  auto &B3 = G.createContentBlock(Sec3, BlockContent, B3Addr, 8, 0);

  // Add a symbols for each block.
  G.addDefinedSymbol(B1, 0, "S1", B1.getSize(), Linkage::Strong, Scope::Default,
                     false, false);
  G.addDefinedSymbol(B2, 0, "S2", B2.getSize(), Linkage::Strong, Scope::Default,
                     false, false);
  G.addDefinedSymbol(B3, 0, "S3", B2.getSize(), Linkage::Strong, Scope::Default,
                     false, false);

  EXPECT_EQ(&B1.getSection(), &Sec1);
  EXPECT_EQ(&B2.getSection(), &Sec2);
  EXPECT_EQ(G.sections_size(), 3U) << "Expected three sections initially";
  EXPECT_EQ(Sec1.blocks_size(), 1U) << "Expected one block in Sec1 initially";
  EXPECT_EQ(Sec1.symbols_size(), 1U) << "Expected one symbol in Sec1 initially";
  EXPECT_EQ(Sec2.blocks_size(), 1U) << "Expected one block in Sec2 initially";
  EXPECT_EQ(Sec2.symbols_size(), 1U) << "Expected one symbol in Sec2 initially";
  EXPECT_EQ(Sec3.blocks_size(), 1U) << "Expected one block in Sec3 initially";
  EXPECT_EQ(Sec3.symbols_size(), 1U) << "Expected one symbol in Sec3 initially";

  // Check that self-merge is a no-op.
  G.mergeSections(Sec1, Sec1);

  EXPECT_EQ(&B1.getSection(), &Sec1)
      << "Expected B1.getSection() to remain unchanged";
  EXPECT_EQ(G.sections_size(), 3U)
      << "Expected three sections after first merge";
  EXPECT_EQ(Sec1.blocks_size(), 1U)
      << "Expected one block in Sec1 after first merge";
  EXPECT_EQ(Sec1.symbols_size(), 1U)
      << "Expected one symbol in Sec1 after first merge";
  EXPECT_EQ(Sec2.blocks_size(), 1U)
      << "Expected one block in Sec2 after first merge";
  EXPECT_EQ(Sec2.symbols_size(), 1U)
      << "Expected one symbol in Sec2 after first merge";
  EXPECT_EQ(Sec3.blocks_size(), 1U)
      << "Expected one block in Sec3 after first merge";
  EXPECT_EQ(Sec3.symbols_size(), 1U)
      << "Expected one symbol in Sec3 after first merge";

  // Merge Sec2 into Sec1, removing Sec2.
  G.mergeSections(Sec1, Sec2);

  EXPECT_EQ(&B2.getSection(), &Sec1)
      << "Expected B2.getSection() to have been changed to &Sec1";
  EXPECT_EQ(G.sections_size(), 2U)
      << "Expected two sections after section merge";
  EXPECT_EQ(Sec1.blocks_size(), 2U)
      << "Expected two blocks in Sec1 after section merge";
  EXPECT_EQ(Sec1.symbols_size(), 2U)
      << "Expected two symbols in Sec1 after section merge";
  EXPECT_EQ(Sec3.blocks_size(), 1U)
      << "Expected one block in Sec3 after section merge";
  EXPECT_EQ(Sec3.symbols_size(), 1U)
      << "Expected one symbol in Sec3 after section merge";

  G.mergeSections(Sec1, Sec3, true);

  EXPECT_EQ(G.sections_size(), 2U) << "Expected two sections after third merge";
  EXPECT_EQ(Sec1.blocks_size(), 3U)
      << "Expected three blocks in Sec1 after third merge";
  EXPECT_EQ(Sec1.symbols_size(), 3U)
      << "Expected three symbols in Sec1 after third merge";
  EXPECT_EQ(Sec3.blocks_size(), 0U)
      << "Expected one block in Sec3 after third merge";
  EXPECT_EQ(Sec3.symbols_size(), 0U)
      << "Expected one symbol in Sec3 after third merge";
}

TEST(LinkGraphTest, SplitBlock) {
  // Check that the LinkGraph::splitBlock test works as expected.
  LinkGraph G("foo", Triple("x86_64-apple-darwin"), 8, support::little,
              getGenericEdgeKindName);
  auto &Sec =
      G.createSection("__data", orc::MemProt::Read | orc::MemProt::Write);

  // Create the block to split.
  orc::ExecutorAddr B1Addr(0x1000);
  auto &B1 = G.createContentBlock(Sec, BlockContent, B1Addr, 8, 0);

  // Add some symbols to the block.
  auto &S1 = G.addDefinedSymbol(B1, 0, "S1", 4, Linkage::Strong, Scope::Default,
                                false, false);
  auto &S2 = G.addDefinedSymbol(B1, 4, "S2", 4, Linkage::Strong, Scope::Default,
                                false, false);
  auto &S3 = G.addDefinedSymbol(B1, 8, "S3", 4, Linkage::Strong, Scope::Default,
                                false, false);
  auto &S4 = G.addDefinedSymbol(B1, 12, "S4", 4, Linkage::Strong,
                                Scope::Default, false, false);
  // Add a symbol that extends beyond the split.
  auto &S5 = G.addDefinedSymbol(B1, 0, "S5", 16, Linkage::Strong,
                                Scope::Default, false, false);

  // Add an extra block, EB, and target symbols, and use these to add edges
  // from B1 to EB.
  orc::ExecutorAddr EBAddr(0x2000);
  auto &EB = G.createContentBlock(Sec, BlockContent, EBAddr, 8, 0);
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
  EXPECT_EQ(B1.getAddress(), B1Addr + 8);
  EXPECT_EQ(B1.getContent(), BlockContent.slice(8));

  EXPECT_EQ(B2.getAddress(), B1Addr);
  EXPECT_EQ(B2.getContent(), BlockContent.slice(0, 8));

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

  EXPECT_EQ(&S5.getBlock(), &B2);
  EXPECT_EQ(S5.getOffset(), 0U);
  // Size shrinks to fit.
  EXPECT_EQ(S5.getSize(), 8U);

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

TEST(LinkGraphTest, IsCStringBlockTest) {
  // Check that the LinkGraph::splitBlock test works as expected.
  LinkGraph G("foo", Triple("x86_64-apple-darwin"), 8, support::little,
              getGenericEdgeKindName);
  auto &Sec =
      G.createSection("__data", orc::MemProt::Read | orc::MemProt::Write);

  ArrayRef<char> CString = "hello, world!";
  ArrayRef<char> NotACString = {0, 1, 0, 1, 0};

  auto &CStringBlock =
      G.createContentBlock(Sec, CString, orc::ExecutorAddr(), 1, 0);
  auto &NotACStringBlock =
      G.createContentBlock(Sec, NotACString, orc::ExecutorAddr(), 1, 0);
  auto &SizeOneZeroFillBlock =
      G.createZeroFillBlock(Sec, 1, orc::ExecutorAddr(), 1, 0);
  auto &LargerZeroFillBlock =
      G.createZeroFillBlock(Sec, 2, orc::ExecutorAddr(), 1, 0);

  EXPECT_TRUE(isCStringBlock(CStringBlock));
  EXPECT_FALSE(isCStringBlock(NotACStringBlock));
  EXPECT_TRUE(isCStringBlock(SizeOneZeroFillBlock));
  EXPECT_FALSE(isCStringBlock(LargerZeroFillBlock));
}
