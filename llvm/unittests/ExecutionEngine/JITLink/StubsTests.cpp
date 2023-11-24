//===------ StubsTests.cpp - Unit tests for generic stub generation -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITLink/aarch64.h"
#include "llvm/ExecutionEngine/JITLink/i386.h"
#include "llvm/ExecutionEngine/JITLink/loongarch.h"
#include "llvm/ExecutionEngine/JITLink/x86_64.h"
#include "llvm/ExecutionEngine/Orc/ObjectFileInterface.h"
#include "llvm/Support/Memory.h"

#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::jitlink;

static std::pair<Symbol &, Symbol &>
GenerateStub(LinkGraph &G, size_t PointerSize, Edge::Kind PointerEdgeKind) {
  auto &FuncSymbol = G.addAbsoluteSymbol("Func", orc::ExecutorAddr(0x2000), 0,
                                         Linkage::Strong, Scope::Default, true);

  // Create a section for pointer symbols.
  auto &PointersSec =
      G.createSection("__pointers", orc::MemProt::Read | orc::MemProt::Write);

  // Create a section for jump stubs symbols.
  auto &StubsSec =
      G.createSection("__stubs", orc::MemProt::Read | orc::MemProt::Write);

  auto AnonymousPtrCreator = getAnonymousPointerCreator(G.getTargetTriple());
  EXPECT_TRUE(AnonymousPtrCreator);

  auto PointerSym = AnonymousPtrCreator(G, PointersSec, &FuncSymbol, 0);
  EXPECT_FALSE(errorToBool(PointerSym.takeError()));
  EXPECT_EQ(std::distance(PointerSym->getBlock().edges().begin(),
                          PointerSym->getBlock().edges().end()),
            1U);
  auto &DeltaEdge = *PointerSym->getBlock().edges().begin();
  EXPECT_EQ(DeltaEdge.getKind(), PointerEdgeKind);
  EXPECT_EQ(&DeltaEdge.getTarget(), &FuncSymbol);
  EXPECT_EQ(PointerSym->getBlock().getSize(), PointerSize);
  EXPECT_TRUE(all_of(PointerSym->getBlock().getContent(),
                     [](char x) { return x == 0; }));

  auto PtrJumpStubCreator = getPointerJumpStubCreator(G.getTargetTriple());
  EXPECT_TRUE(PtrJumpStubCreator);
  auto StubSym = PtrJumpStubCreator(G, StubsSec, *PointerSym);
  EXPECT_FALSE(errorToBool(StubSym.takeError()));
  return {*PointerSym, *StubSym};
}

TEST(StubsTest, StubsGeneration_x86_64) {
  const char PointerJumpStubContent[6] = {
      static_cast<char>(0xFFu), 0x25, 0x00, 0x00, 0x00, 0x00};
  LinkGraph G("foo", Triple("x86_64-apple-darwin"), 8, llvm::endianness::little,
              getGenericEdgeKindName);
  auto [PointerSym, StubSym] = GenerateStub(G, 8U, x86_64::Pointer64);

  EXPECT_EQ(std::distance(StubSym.getBlock().edges().begin(),
                          StubSym.getBlock().edges().end()),
            1U);
  auto &JumpEdge = *StubSym.getBlock().edges().begin();
  EXPECT_EQ(JumpEdge.getKind(), x86_64::BranchPCRel32);
  EXPECT_EQ(&JumpEdge.getTarget(), &PointerSym);
  EXPECT_EQ(StubSym.getBlock().getContent(),
            ArrayRef<char>(PointerJumpStubContent));
}

TEST(StubsTest, StubsGeneration_aarch64) {
  const char PointerJumpStubContent[12] = {
      0x10, 0x00, 0x00, (char)0x90u, // ADRP x16, <imm>@page21
      0x10, 0x02, 0x40, (char)0xf9u, // LDR x16, [x16, <imm>@pageoff12]
      0x00, 0x02, 0x1f, (char)0xd6u  // BR  x16
  };
  LinkGraph G("foo", Triple("aarch64-linux-gnu"), 8, llvm::endianness::little,
              getGenericEdgeKindName);
  auto [PointerSym, StubSym] = GenerateStub(G, 8U, aarch64::Pointer64);

  EXPECT_EQ(std::distance(StubSym.getBlock().edges().begin(),
                          StubSym.getBlock().edges().end()),
            2U);
  auto &AdrpHighEdge = *StubSym.getBlock().edges().begin();
  auto &LdrEdge = *++StubSym.getBlock().edges().begin();
  EXPECT_EQ(AdrpHighEdge.getKind(), aarch64::Page21);
  EXPECT_EQ(&AdrpHighEdge.getTarget(), &PointerSym);
  EXPECT_EQ(LdrEdge.getKind(), aarch64::PageOffset12);
  EXPECT_EQ(&LdrEdge.getTarget(), &PointerSym);
  EXPECT_EQ(StubSym.getBlock().getContent(),
            ArrayRef<char>(PointerJumpStubContent));
}

TEST(StubsTest, StubsGeneration_i386) {
  const char PointerJumpStubContent[6] = {
      static_cast<char>(0xFFu), 0x25, 0x00, 0x00, 0x00, 0x00};
  LinkGraph G("foo", Triple("i386-unknown-linux-gnu"), 4,
              llvm::endianness::little, getGenericEdgeKindName);
  auto [PointerSym, StubSym] = GenerateStub(G, 4U, i386::Pointer32);

  EXPECT_EQ(std::distance(StubSym.getBlock().edges().begin(),
                          StubSym.getBlock().edges().end()),
            1U);
  auto &JumpEdge = *StubSym.getBlock().edges().begin();
  EXPECT_EQ(JumpEdge.getKind(), i386::Pointer32);
  EXPECT_EQ(&JumpEdge.getTarget(), &PointerSym);
  EXPECT_EQ(StubSym.getBlock().getContent(),
            ArrayRef<char>(PointerJumpStubContent));
}

TEST(StubsTest, StubsGeneration_loongarch32) {
  const char PointerJumpStubContent[12] = {
      0x14,
      0x00,
      0x00,
      0x1a, // pcalau12i $t8, %page20(imm)
      static_cast<char>(0x94),
      0x02,
      static_cast<char>(0x80),
      0x28, // ld.d $t8, $t8, %pageoff12(imm)
      static_cast<char>(0x80),
      0x02,
      0x00,
      0x4c // jr $t8
  };
  LinkGraph G("foo", Triple("loongarch32"), 4, llvm::endianness::little,
              getGenericEdgeKindName);
  auto [PointerSym, StubSym] = GenerateStub(G, 4U, loongarch::Pointer32);

  EXPECT_EQ(std::distance(StubSym.getBlock().edges().begin(),
                          StubSym.getBlock().edges().end()),
            2U);
  auto &PageHighEdge = *StubSym.getBlock().edges().begin();
  auto &PageLowEdge = *++StubSym.getBlock().edges().begin();
  EXPECT_EQ(PageHighEdge.getKind(), loongarch::Page20);
  EXPECT_EQ(&PageHighEdge.getTarget(), &PointerSym);
  EXPECT_EQ(PageLowEdge.getKind(), loongarch::PageOffset12);
  EXPECT_EQ(&PageLowEdge.getTarget(), &PointerSym);
  EXPECT_EQ(StubSym.getBlock().getContent(),
            ArrayRef<char>(PointerJumpStubContent));
}

TEST(StubsTest, StubsGeneration_loongarch64) {
  const char PointerJumpStubContent[12] = {
      0x14,
      0x00,
      0x00,
      0x1a, // pcalau12i $t8, %page20(imm)
      static_cast<char>(0x94),
      0x02,
      static_cast<char>(0xc0),
      0x28, // ld.d $t8, $t8, %pageoff12(imm)
      static_cast<char>(0x80),
      0x02,
      0x00,
      0x4c // jr $t8
  };
  LinkGraph G("foo", Triple("loongarch64"), 8, llvm::endianness::little,
              getGenericEdgeKindName);
  auto [PointerSym, StubSym] = GenerateStub(G, 8U, loongarch::Pointer64);

  EXPECT_EQ(std::distance(StubSym.getBlock().edges().begin(),
                          StubSym.getBlock().edges().end()),
            2U);
  auto &PageHighEdge = *StubSym.getBlock().edges().begin();
  auto &PageLowEdge = *++StubSym.getBlock().edges().begin();
  EXPECT_EQ(PageHighEdge.getKind(), loongarch::Page20);
  EXPECT_EQ(&PageHighEdge.getTarget(), &PointerSym);
  EXPECT_EQ(PageLowEdge.getKind(), loongarch::PageOffset12);
  EXPECT_EQ(&PageLowEdge.getTarget(), &PointerSym);
  EXPECT_EQ(StubSym.getBlock().getContent(),
            ArrayRef<char>(PointerJumpStubContent));
}
