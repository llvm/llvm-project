//===---- JITLinkTest_MachO_x86_64.cpp - Tests for JITLink MachO/x86-64 ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JITLinkTestCommon.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ExecutionEngine/JITLink/JITLink_MachO_x86_64.h"
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
      std::function<void(AtomGraph &, const MCDisassembler &)>;

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

    auto JTCtx = llvm::make_unique<TestJITLinkContext>(
        **TR, [&](AtomGraph &G) { RunGraphTest(G, (*TR)->getDisassembler()); });

    JTCtx->externals() = std::move(Externals);

    jitLink_MachO_x86_64(std::move(JTCtx));
  }

protected:
  static void verifyIsPointerTo(AtomGraph &G, DefinedAtom &A, Atom &Target) {
    EXPECT_EQ(A.edges_size(), 1U) << "Incorrect number of edges for pointer";
    if (A.edges_size() != 1U)
      return;
    auto &E = *A.edges().begin();
    EXPECT_EQ(E.getKind(), Pointer64)
        << "Expected pointer to have a pointer64 relocation";
    EXPECT_EQ(&E.getTarget(), &Target) << "Expected edge to point at target";
    EXPECT_THAT_EXPECTED(readInt<uint64_t>(G, A), HasValue(Target.getAddress()))
        << "Pointer does not point to target";
  }

  static void verifyGOTLoad(AtomGraph &G, DefinedAtom &A, Edge &E,
                            Atom &Target) {
    EXPECT_EQ(E.getAddend(), 0U) << "Expected GOT load to have a zero addend";
    EXPECT_TRUE(E.getTarget().isDefined())
        << "GOT entry should be a defined atom";
    if (!E.getTarget().isDefined())
      return;

    verifyIsPointerTo(G, static_cast<DefinedAtom &>(E.getTarget()), Target);
  }

  static void verifyCall(const MCDisassembler &Dis, AtomGraph &G,
                         DefinedAtom &Caller, Edge &E, Atom &Callee) {
    EXPECT_EQ(E.getKind(), Branch32) << "Edge is not a Branch32";
    EXPECT_EQ(E.getAddend(), 0U) << "Expected no addend on stub call";
    EXPECT_EQ(&E.getTarget(), &Callee)
        << "Edge does not point at expected callee";

    JITTargetAddress FixupAddress = Caller.getAddress() + E.getOffset();
    uint64_t PCRelDelta = Callee.getAddress() - (FixupAddress + 4);

    EXPECT_THAT_EXPECTED(
        decodeImmediateOperand(Dis, Caller, 0, E.getOffset() - 1),
        HasValue(PCRelDelta));
  }

  static void verifyIndirectCall(const MCDisassembler &Dis, AtomGraph &G,
                                 DefinedAtom &Caller, Edge &E, Atom &Callee) {
    EXPECT_EQ(E.getKind(), PCRel32) << "Edge is not a PCRel32";
    EXPECT_EQ(E.getAddend(), 0) << "Expected no addend on stub cal";
    EXPECT_TRUE(E.getTarget().isDefined()) << "Target is not a defined atom";
    if (!E.getTarget().isDefined())
      return;
    verifyIsPointerTo(G, static_cast<DefinedAtom &>(E.getTarget()), Callee);

    JITTargetAddress FixupAddress = Caller.getAddress() + E.getOffset();
    uint64_t PCRelDelta = E.getTarget().getAddress() - (FixupAddress + 4);

    EXPECT_THAT_EXPECTED(
        decodeImmediateOperand(Dis, Caller, 3, E.getOffset() - 2),
        HasValue(PCRelDelta));
  }

  static void verifyCallViaStub(const MCDisassembler &Dis, AtomGraph &G,
                                DefinedAtom &Caller, Edge &E, Atom &Callee) {
    verifyCall(Dis, G, Caller, E, E.getTarget());

    if (!E.getTarget().isDefined()) {
      ADD_FAILURE() << "Edge target is not a stub";
      return;
    }

    auto &StubAtom = static_cast<DefinedAtom &>(E.getTarget());
    EXPECT_EQ(StubAtom.edges_size(), 1U)
        << "Expected one edge from stub to target";

    auto &StubEdge = *StubAtom.edges().begin();

    verifyIndirectCall(Dis, G, static_cast<DefinedAtom &>(StubAtom), StubEdge,
                       Callee);
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
      [](AtomGraph &G, const MCDisassembler &Dis) {
        // Name the atoms in the asm above.
        auto &Baz = atom(G, "_baz");
        auto &Y = atom(G, "_y");

        auto &Bar = definedAtom(G, "_bar");
        auto &Foo = definedAtom(G, "_foo");
        auto &Foo_1 = definedAtom(G, "_foo.1");
        auto &Foo_2 = definedAtom(G, "_foo.2");
        auto &X = definedAtom(G, "_x");
        auto &P = definedAtom(G, "_p");

        // Check unsigned reloc for _p
        {
          EXPECT_EQ(P.edges_size(), 1U) << "Unexpected number of relocations";
          EXPECT_EQ(P.edges().begin()->getKind(), Pointer64)
              << "Unexpected edge kind for _p";
          EXPECT_THAT_EXPECTED(readInt<uint64_t>(G, P),
                               HasValue(X.getAddress()))
              << "Unsigned relocation did not apply correctly";
        }

        // Check that _bar is a call-via-stub to _baz.
        // This will check that the call goes to a stub, that the stub is an
        // indirect call, and that the pointer for the indirect call points to
        // baz.
        {
          EXPECT_EQ(Bar.edges_size(), 1U)
              << "Incorrect number of edges for bar";
          EXPECT_EQ(Bar.edges().begin()->getKind(), Branch32)
              << "Unexpected edge kind for _bar";
          verifyCallViaStub(Dis, G, Bar, *Bar.edges().begin(), Baz);
        }

        // Check that _foo is a direct call to _bar.
        {
          EXPECT_EQ(Foo.edges_size(), 1U)
              << "Incorrect number of edges for foo";
          EXPECT_EQ(Foo.edges().begin()->getKind(), Branch32);
          verifyCall(Dis, G, Foo, *Foo.edges().begin(), Bar);
        }

        // Check .got load in _foo.1
        {
          EXPECT_EQ(Foo_1.edges_size(), 1U)
              << "Incorrect number of edges for foo_1";
          EXPECT_EQ(Foo_1.edges().begin()->getKind(), PCRel32);
          verifyGOTLoad(G, Foo_1, *Foo_1.edges().begin(), Y);
        }

        // Check PCRel ref to _p in _foo.2
        {
          EXPECT_EQ(Foo_2.edges_size(), 1U)
              << "Incorrect number of edges for foo_2";
          EXPECT_EQ(Foo_2.edges().begin()->getKind(), PCRel32);

          JITTargetAddress FixupAddress =
              Foo_2.getAddress() + Foo_2.edges().begin()->getOffset();
          uint64_t PCRelDelta = P.getAddress() - (FixupAddress + 4);

          EXPECT_THAT_EXPECTED(decodeImmediateOperand(Dis, Foo_2, 4, 0),
                               HasValue(PCRelDelta))
              << "PCRel load does not reference expected target";
        }
      });
}
