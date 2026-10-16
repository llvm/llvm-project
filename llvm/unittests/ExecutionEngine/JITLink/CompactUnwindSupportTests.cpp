//===-- CompactUnwindSupportTests.cpp - Compact unwind unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../../lib/ExecutionEngine/JITLink/CompactUnwindSupport.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::jitlink;

namespace {

struct TestTraits : CompactUnwindTraits<TestTraits, 8> {
  static constexpr endianness Endianness = endianness::little;
  static bool encodingSpecifiesDWARF(uint32_t Encoding) {
    return Encoding != 0;
  }
};

TEST(CompactUnwindSupportTest, SharedCodeBlockMatchesFDEsByAddress) {
  for (bool ReverseFDEs : {false, true}) {
    for (bool UseAddend : {false, true}) {
      SCOPED_TRACE(testing::Message() << ReverseFDEs << ", " << UseAddend);
      LinkGraph G("shared-block", std::make_shared<orc::SymbolStringPool>(),
                  Triple("arm64-apple-darwin"), 8, SubtargetFeatures(),
                  getGenericEdgeKindName);
      auto &Text = G.createSection("__TEXT,__text", orc::MemProt::Exec);
      char Code[8] = {};
      auto &CodeBlock =
          G.createContentBlock(Text, Code, orc::ExecutorAddr(0x1000), 4, 0);
      Symbol *Functions[] = {
          &G.addDefinedSymbol(CodeBlock, 0, "first", 4, Linkage::Strong,
                              Scope::Default, true, true),
          &G.addDefinedSymbol(CodeBlock, 4, "second", 4, Linkage::Strong,
                              Scope::Default, true, true)};
      auto &Frames = G.createSection("__TEXT,__eh_frame", orc::MemProt::Read);
      char CIEContent[8] = {}, FDEContent[2][16] = {};
      auto &CIEBlock = G.createContentBlock(Frames, CIEContent,
                                            orc::ExecutorAddr(0x2000), 8, 0);
      auto &CIE = G.addAnonymousSymbol(CIEBlock, 0, 8, false, false);
      Symbol *FDEs[2];
      for (unsigned I = 0; I != 2; ++I) {
        auto &FDEBlock = G.createContentBlock(
            Frames, FDEContent[I], orc::ExecutorAddr(0x2010 + 16 * I), 8, 0);
        FDEBlock.addEdge(Edge::FirstRelocation, 4, CIE, 0);
        FDEBlock.addEdge(Edge::FirstRelocation, 8, *Functions[I], 0);
        FDEs[I] = &G.addAnonymousSymbol(FDEBlock, 0, 16, false, false);
      }
      for (unsigned I = 0; I != 2; ++I)
        CodeBlock.addEdge(Edge::KeepAlive, 0, *FDEs[ReverseFDEs ? 1 - I : I],
                          0);

      auto &Compact =
          G.createSection("__LD,__compact_unwind", orc::MemProt::Read);
      char Records[2][TestTraits::Size] = {};
      Block *CompactBlocks[2];
      for (unsigned I = 0; I != 2; ++I) {
        support::endian::write32le(Records[I] + TestTraits::EncodingFieldOffset,
                                   1);
        CompactBlocks[I] = &G.createContentBlock(
            Compact, Records[I],
            orc::ExecutorAddr(0x3000 + TestTraits::Size * I), 8, 0);
        CompactBlocks[I]->addEdge(Edge::FirstRelocation, 0,
                                  *Functions[UseAddend ? 0 : I],
                                  UseAddend ? 4 * I : 0);
      }

      CompactUnwindManager<TestTraits> Manager(
          "__LD,__compact_unwind", "__TEXT,__unwind_info", "__TEXT,__eh_frame");
      auto Err = Manager.prepareForPrune(G);
      ASSERT_FALSE(!!Err) << toString(std::move(Err));
      for (unsigned I = 0; I != 2; ++I) {
        unsigned KeepAliveCount = 0;
        for (auto &E : CompactBlocks[I]->edges()) {
          if (E.getKind() != Edge::KeepAlive)
            continue;
          ++KeepAliveCount;
          EXPECT_EQ(&E.getTarget(), FDEs[I]);
        }
        EXPECT_EQ(KeepAliveCount, 1U);
      }
    }
  }
}

TEST(CompactUnwindSupportTest, RejectsAnUnrelatedFDEInTheSameCodeBlock) {
  LinkGraph G("missing-fde", std::make_shared<orc::SymbolStringPool>(),
              Triple("arm64-apple-darwin"), 8, SubtargetFeatures(),
              getGenericEdgeKindName);
  char Code[8] = {}, CIEContent[8] = {}, FDEContent[16] = {};
  char Record[TestTraits::Size] = {};
  auto &Text = G.createSection("__TEXT,__text", orc::MemProt::Exec);
  auto &CodeBlock =
      G.createContentBlock(Text, Code, orc::ExecutorAddr(0x1000), 4, 0);
  auto &First = G.addDefinedSymbol(CodeBlock, 0, "first", 4, Linkage::Strong,
                                   Scope::Default, true, true);
  auto &Second = G.addDefinedSymbol(CodeBlock, 4, "second", 4, Linkage::Strong,
                                    Scope::Default, true, true);
  auto &Frames = G.createSection("__TEXT,__eh_frame", orc::MemProt::Read);
  auto &CIEBlock =
      G.createContentBlock(Frames, CIEContent, orc::ExecutorAddr(0x2000), 8, 0);
  auto &CIE = G.addAnonymousSymbol(CIEBlock, 0, 8, false, false);
  auto &FDEBlock =
      G.createContentBlock(Frames, FDEContent, orc::ExecutorAddr(0x2010), 8, 0);
  FDEBlock.addEdge(Edge::FirstRelocation, 4, CIE, 0);
  FDEBlock.addEdge(Edge::FirstRelocation, 8, First, 0);
  auto &FDE = G.addAnonymousSymbol(FDEBlock, 0, 16, false, false);
  CodeBlock.addEdge(Edge::KeepAlive, 0, FDE, 0);
  auto &Compact = G.createSection("__LD,__compact_unwind", orc::MemProt::Read);
  support::endian::write32le(Record + TestTraits::EncodingFieldOffset, 1);
  auto &CompactBlock =
      G.createContentBlock(Compact, Record, orc::ExecutorAddr(0x3000), 8, 0);
  CompactBlock.addEdge(Edge::FirstRelocation, 0, Second, 0);

  CompactUnwindManager<TestTraits> Manager(
      "__LD,__compact_unwind", "__TEXT,__unwind_info", "__TEXT,__eh_frame");
  auto Err = Manager.prepareForPrune(G);
  ASSERT_TRUE(!!Err);
  EXPECT_NE(toString(std::move(Err)).find("no FDE was found"),
            std::string::npos);
}

} // namespace
