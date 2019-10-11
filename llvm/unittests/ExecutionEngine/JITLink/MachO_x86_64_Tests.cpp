//===--------- MachO_x86_64.cpp - Tests for JITLink MachO/x86-64 ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JITLinkTestCommon.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ExecutionEngine/JITLink/MachO_x86_64.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::jitlink::MachO_x86_64_Edges;

namespace {

class JITLinkTest_MachO_x86_64 : public JITLinkTestCommon,
                                 public testing::Test {
public:
  using BasicVerifyGraphFunction =
      std::function<void(LinkGraph &, const MCDisassembler &)>;

  void runBasicVerifyGraphTest(StringRef AsmSrc, StringRef Triple,
                               StringMap<JITEvaluatedSymbol> Externals,
                               bool PIC, bool LargeCodeModel,
                               MCTargetOptions Options,
                               BasicVerifyGraphFunction RunGraphTest) {
    auto TR = getTestResources(AsmSrc, Triple, PIC, LargeCodeModel,
                               std::move(Options));
    if (!TR) {
      dbgs() << "Skipping JITLInk unit test: " << toString(TR.takeError())
             << "\n";
      return;
    }

    auto JTCtx = std::make_unique<TestJITLinkContext>(
        **TR, [&](LinkGraph &G) { RunGraphTest(G, (*TR)->getDisassembler()); });

    JTCtx->externals() = std::move(Externals);

    jitLink_MachO_x86_64(std::move(JTCtx));
  }

protected:
  static void verifyIsPointerTo(LinkGraph &G, Block &B, Symbol &Target) {
    EXPECT_EQ(B.edges_size(), 1U) << "Incorrect number of edges for pointer";
    if (B.edges_size() != 1U)
      return;
    auto &E = *B.edges().begin();
    EXPECT_EQ(E.getOffset(), 0U) << "Expected edge offset of zero";
    EXPECT_EQ(E.getKind(), Pointer64)
        << "Expected pointer to have a pointer64 relocation";
    EXPECT_EQ(&E.getTarget(), &Target) << "Expected edge to point at target";
    EXPECT_THAT_EXPECTED(readInt<uint64_t>(G, B), HasValue(Target.getAddress()))
        << "Pointer does not point to target";
  }

  static void verifyGOTLoad(LinkGraph &G, Edge &E, Symbol &Target) {
    EXPECT_EQ(E.getAddend(), 0U) << "Expected GOT load to have a zero addend";
    EXPECT_TRUE(E.getTarget().isDefined())
        << "GOT entry should be a defined symbol";
    if (!E.getTarget().isDefined())
      return;

    verifyIsPointerTo(G, E.getTarget().getBlock(), Target);
  }

  static void verifyCall(const MCDisassembler &Dis, LinkGraph &G,
                         Block &CallerBlock, Edge &E, Symbol &Callee) {
    EXPECT_EQ(E.getKind(), Branch32) << "Edge is not a Branch32";
    EXPECT_EQ(E.getAddend(), 0U) << "Expected no addend on stub call";
    EXPECT_EQ(&E.getTarget(), &Callee)
        << "Edge does not point at expected callee";

    JITTargetAddress FixupAddress = CallerBlock.getAddress() + E.getOffset();
    uint64_t PCRelDelta = Callee.getAddress() - (FixupAddress + 4);

    EXPECT_THAT_EXPECTED(
        decodeImmediateOperand(Dis, CallerBlock, 0, E.getOffset() - 1),
        HasValue(PCRelDelta));
  }

  static void verifyIndirectCall(const MCDisassembler &Dis, LinkGraph &G,
                                 Block &CallerBlock, Edge &E, Symbol &Callee) {
    EXPECT_EQ(E.getKind(), PCRel32) << "Edge is not a PCRel32";
    EXPECT_EQ(E.getAddend(), 0) << "Expected no addend on stub cal";
    EXPECT_TRUE(E.getTarget().isDefined()) << "Target is not a defined symbol";
    if (!E.getTarget().isDefined())
      return;
    verifyIsPointerTo(G, E.getTarget().getBlock(), Callee);

    JITTargetAddress FixupAddress = CallerBlock.getAddress() + E.getOffset();
    uint64_t PCRelDelta = E.getTarget().getAddress() - (FixupAddress + 4);

    EXPECT_THAT_EXPECTED(
        decodeImmediateOperand(Dis, CallerBlock, 3, E.getOffset() - 2),
        HasValue(PCRelDelta));
  }

  static void verifyCallViaStub(const MCDisassembler &Dis, LinkGraph &G,
                                Block &CallerBlock, Edge &E, Symbol &Callee) {
    verifyCall(Dis, G, CallerBlock, E, E.getTarget());

    if (!E.getTarget().isDefined()) {
      ADD_FAILURE() << "Edge target is not a stub";
      return;
    }

    auto &StubBlock = E.getTarget().getBlock();
    EXPECT_EQ(StubBlock.edges_size(), 1U)
        << "Expected one edge from stub to target";

    auto &StubEdge = *StubBlock.edges().begin();

    verifyIndirectCall(Dis, G, StubBlock, StubEdge, Callee);
  }
};

} // end anonymous namespace

// Test each operation on LegacyObjectTransformLayer.
TEST_F(JITLinkTest_MachO_x86_64, BasicRelocations) {
  runBasicVerifyGraphTest(
      R"(
            .section        __TEXT,__text,regular,pure_instructions
            .build_version macos, 10, 14
            .globl  _bar
            .p2align        4, 0x90
    _bar:
            callq    _baz

            .globl  _foo
            .p2align        4, 0x90
    _foo:
            callq   _bar
    _foo.1:
            movq    _y@GOTPCREL(%rip), %rcx
    _foo.2:
            movq    _p(%rip), %rdx

            .section        __DATA,__data
            .globl  _x
            .p2align        2
    _x:
            .long   42

            .globl  _p
            .p2align        3
    _p:
            .quad   _x

    .subsections_via_symbols)",
      "x86_64-apple-macosx10.14",
      {{"_y", JITEvaluatedSymbol(0xdeadbeef, JITSymbolFlags::Exported)},
       {"_baz", JITEvaluatedSymbol(0xcafef00d, JITSymbolFlags::Exported)}},
      true, false, MCTargetOptions(),
      [](LinkGraph &G, const MCDisassembler &Dis) {
        // Name the symbols in the asm above.
        auto &Baz = symbol(G, "_baz");
        auto &Y = symbol(G, "_y");
        auto &Bar = symbol(G, "_bar");
        auto &Foo = symbol(G, "_foo");
        auto &Foo_1 = symbol(G, "_foo.1");
        auto &Foo_2 = symbol(G, "_foo.2");
        auto &X = symbol(G, "_x");
        auto &P = symbol(G, "_p");

        // Check unsigned reloc for _p
        {
          EXPECT_EQ(P.getBlock().edges_size(), 1U)
              << "Unexpected number of relocations";
          EXPECT_EQ(P.getBlock().edges().begin()->getKind(), Pointer64)
              << "Unexpected edge kind for _p";
          EXPECT_THAT_EXPECTED(readInt<uint64_t>(G, P.getBlock()),
                               HasValue(X.getAddress()))
              << "Unsigned relocation did not apply correctly";
        }

        // Check that _bar is a call-via-stub to _baz.
        // This will check that the call goes to a stub, that the stub is an
        // indirect call, and that the pointer for the indirect call points to
        // baz.
        {
          EXPECT_EQ(Bar.getBlock().edges_size(), 1U)
              << "Incorrect number of edges for bar";
          EXPECT_EQ(Bar.getBlock().edges().begin()->getKind(), Branch32)
              << "Unexpected edge kind for _bar";
          verifyCallViaStub(Dis, G, Bar.getBlock(),
                            *Bar.getBlock().edges().begin(), Baz);
        }

        // Check that _foo is a direct call to _bar.
        {
          EXPECT_EQ(Foo.getBlock().edges_size(), 1U)
              << "Incorrect number of edges for foo";
          EXPECT_EQ(Foo.getBlock().edges().begin()->getKind(), Branch32);
          verifyCall(Dis, G, Foo.getBlock(), *Foo.getBlock().edges().begin(),
                     Bar);
        }

        // Check .got load in _foo.1
        {
          EXPECT_EQ(Foo_1.getBlock().edges_size(), 1U)
              << "Incorrect number of edges for foo_1";
          EXPECT_EQ(Foo_1.getBlock().edges().begin()->getKind(), PCRel32);
          verifyGOTLoad(G, *Foo_1.getBlock().edges().begin(), Y);
        }

        // Check PCRel ref to _p in _foo.2
        {
          EXPECT_EQ(Foo_2.getBlock().edges_size(), 1U)
              << "Incorrect number of edges for foo_2";
          EXPECT_EQ(Foo_2.getBlock().edges().begin()->getKind(), PCRel32);

          JITTargetAddress FixupAddress =
              Foo_2.getBlock().getAddress() +
              Foo_2.getBlock().edges().begin()->getOffset();
          uint64_t PCRelDelta = P.getAddress() - (FixupAddress + 4);

          EXPECT_THAT_EXPECTED(
              decodeImmediateOperand(Dis, Foo_2.getBlock(), 4, 0),
              HasValue(PCRelDelta))
              << "PCRel load does not reference expected target";
        }
      });
}
