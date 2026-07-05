//===- llvm/unittest/CodeGen/GlobalISel/LegalizerInfoTest.cpp -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "GISelMITest.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace LegalizeActions;
using namespace LegalityPredicates;
using namespace LegalizeMutations;

// Define a couple of pretty printers to help debugging when things go wrong.
namespace llvm {
std::ostream &
operator<<(std::ostream &OS, const LegalizeAction Act) {
  switch (Act) {
  case Lower: OS << "Lower"; break;
  case Legal: OS << "Legal"; break;
  case NarrowScalar: OS << "NarrowScalar"; break;
  case WidenScalar:  OS << "WidenScalar"; break;
  case FewerElements:  OS << "FewerElements"; break;
  case MoreElements:  OS << "MoreElements"; break;
  case Libcall: OS << "Libcall"; break;
  case Custom: OS << "Custom"; break;
  case Bitcast: OS << "Bitcast"; break;
  case Unsupported: OS << "Unsupported"; break;
  case NotFound: OS << "NotFound"; break;
  }
  return OS;
}

std::ostream &operator<<(std::ostream &OS, const llvm::LegalizeActionStep Ty) {
  OS << "LegalizeActionStep(" << Ty.Action << ", " << Ty.TypeIdx << ", "
     << Ty.NewType << ')';
  return OS;
}
}


#define EXPECT_ACTION(Action, Index, Type, Query)                              \
  do {                                                                         \
    auto A = LI.getAction(Query);                                              \
    EXPECT_EQ(LegalizeActionStep(Action, Index, Type), A) << A;                \
  } while (0)

TEST(LegalizerInfoTest, RuleSets) {
  using namespace TargetOpcode;

  const LLT s5 = LLT::scalar(5);
  const LLT s8 = LLT::scalar(8);
  const LLT s16 = LLT::scalar(16);
  const LLT s32 = LLT::scalar(32);
  const LLT s33 = LLT::scalar(33);
  const LLT s64 = LLT::scalar(64);

  const LLT v2s5 = LLT::fixed_vector(2, 5);
  const LLT v2s8 = LLT::fixed_vector(2, 8);
  const LLT v2s16 = LLT::fixed_vector(2, 16);
  const LLT v2s32 = LLT::fixed_vector(2, 32);
  const LLT v3s32 = LLT::fixed_vector(3, 32);
  const LLT v4s32 = LLT::fixed_vector(4, 32);
  const LLT v8s32 = LLT::fixed_vector(8, 32);
  const LLT v2s33 = LLT::fixed_vector(2, 33);
  const LLT v2s64 = LLT::fixed_vector(2, 64);

  const LLT p0 = LLT::pointer(0, 32);
  const LLT v2p0 = LLT::fixed_vector(2, p0);
  const LLT v3p0 = LLT::fixed_vector(3, p0);
  const LLT v4p0 = LLT::fixed_vector(4, p0);

  const LLT s1 = LLT::scalar(1);
  const LLT v2s1 = LLT::fixed_vector(2, 1);
  const LLT v4s1 = LLT::fixed_vector(4, 1);

  {
    LegalizerInfo LI;

    LI.getActionDefinitionsBuilder(G_IMPLICIT_DEF)
      .legalFor({v4s32, v4p0})
      .moreElementsToNextPow2(0);

    EXPECT_ACTION(Unsupported, 0, LLT(), LegalityQuery(G_IMPLICIT_DEF, {s32}));
    EXPECT_ACTION(Unsupported, 0, LLT(), LegalityQuery(G_IMPLICIT_DEF, {v2s32}));
    EXPECT_ACTION(MoreElements, 0, v4p0, LegalityQuery(G_IMPLICIT_DEF, {v3p0}));
    EXPECT_ACTION(MoreElements, 0, v4s32, LegalityQuery(G_IMPLICIT_DEF, {v3s32}));
  }

  // Test minScalarOrElt
  {
    LegalizerInfo LI;
    LI.getActionDefinitionsBuilder(G_OR)
      .legalFor({s32})
      .minScalarOrElt(0, s32);

    EXPECT_ACTION(WidenScalar, 0, s32, LegalityQuery(G_OR, {s16}));
    EXPECT_ACTION(WidenScalar, 0, v2s32, LegalityQuery(G_OR, {v2s16}));
  }

  // Test maxScalarOrELt
  {
    LegalizerInfo LI;
    LI.getActionDefinitionsBuilder(G_AND)
      .legalFor({s16})
      .maxScalarOrElt(0, s16);

    EXPECT_ACTION(NarrowScalar, 0, s16, LegalityQuery(G_AND, {s32}));
    EXPECT_ACTION(NarrowScalar, 0, v2s16, LegalityQuery(G_AND, {v2s32}));
  }

  // Test clampScalarOrElt
  {
    LegalizerInfo LI;
    LI.getActionDefinitionsBuilder(G_XOR)
      .legalFor({s16})
      .clampScalarOrElt(0, s16, s32);

    EXPECT_ACTION(NarrowScalar, 0, s32, LegalityQuery(G_XOR, {s64}));
    EXPECT_ACTION(WidenScalar, 0, s16, LegalityQuery(G_XOR, {s8}));

    // Make sure the number of elements is preserved.
    EXPECT_ACTION(NarrowScalar, 0, v2s32, LegalityQuery(G_XOR, {v2s64}));
    EXPECT_ACTION(WidenScalar, 0, v2s16, LegalityQuery(G_XOR, {v2s8}));
  }

  // Test minScalar
  {
    LegalizerInfo LI;
    LI.getActionDefinitionsBuilder(G_OR)
      .legalFor({s32})
      .minScalar(0, s32);

    // Only handle scalars, ignore vectors.
    EXPECT_ACTION(WidenScalar, 0, s32, LegalityQuery(G_OR, {s16}));
    EXPECT_ACTION(Unsupported, 0, LLT(), LegalityQuery(G_OR, {v2s16}));
  }

  // Test minScalarIf
  {
    bool IfCond = true;
    LegalizerInfo LI;
    LI.getActionDefinitionsBuilder(G_OR)
      .legalFor({s32})
      .minScalarIf([&](const LegalityQuery &Query) {
                     return IfCond;
                   }, 0, s32);

    // Only handle scalars, ignore vectors.
    EXPECT_ACTION(WidenScalar, 0, s32, LegalityQuery(G_OR, {s16}));
    EXPECT_ACTION(Unsupported, 0, LLT(), LegalityQuery(G_OR, {v2s16}));

    IfCond = false;
    EXPECT_ACTION(Unsupported, 0, LLT(), LegalityQuery(G_OR, {s16}));
    EXPECT_ACTION(Unsupported, 0, LLT(), LegalityQuery(G_OR, {v2s16}));
  }

  // Test maxScalar
  {
    LegalizerInfo LI;
    LI.getActionDefinitionsBuilder(G_AND)
      .legalFor({s16})
      .maxScalar(0, s16);

    // Only handle scalars, ignore vectors.
    EXPECT_ACTION(NarrowScalar, 0, s16, LegalityQuery(G_AND, {s32}));
    EXPECT_ACTION(Unsupported, 0, LLT(), LegalityQuery(G_AND, {v2s32}));
  }

  // Test clampScalar
  {
    LegalizerInfo LI;

    LI.getActionDefinitionsBuilder(G_XOR)
      .legalFor({s16})
      .clampScalar(0, s16, s32);

    EXPECT_ACTION(NarrowScalar, 0, s32, LegalityQuery(G_XOR, {s64}));
    EXPECT_ACTION(WidenScalar, 0, s16, LegalityQuery(G_XOR, {s8}));

    // Only handle scalars, ignore vectors.
    EXPECT_ACTION(Unsupported, 0, LLT(), LegalityQuery(G_XOR, {v2s64}));
    EXPECT_ACTION(Unsupported, 0, LLT(), LegalityQuery(G_XOR, {v2s8}));
  }

  // Test widenScalarOrEltToNextPow2
  {
    LegalizerInfo LI;

    LI.getActionDefinitionsBuilder(G_AND)
      .legalFor({s32})
      .widenScalarOrEltToNextPow2(0, 32);

    // Handle scalars and vectors
    EXPECT_ACTION(WidenScalar, 0, s32, LegalityQuery(G_AND, {s5}));
    EXPECT_ACTION(WidenScalar, 0, v2s32, LegalityQuery(G_AND, {v2s5}));
    EXPECT_ACTION(WidenScalar, 0, s64, LegalityQuery(G_AND, {s33}));
    EXPECT_ACTION(WidenScalar, 0, v2s64, LegalityQuery(G_AND, {v2s33}));
  }

  // Test widenScalarToNextPow2
  {
    LegalizerInfo LI;

    LI.getActionDefinitionsBuilder(G_AND)
      .legalFor({s32})
      .widenScalarToNextPow2(0, 32);

    EXPECT_ACTION(WidenScalar, 0, s32, LegalityQuery(G_AND, {s5}));
    EXPECT_ACTION(WidenScalar, 0, s64, LegalityQuery(G_AND, {s33}));

    // Do nothing for vectors.
    EXPECT_ACTION(Unsupported, 0, LLT(), LegalityQuery(G_AND, {v2s5}));
    EXPECT_ACTION(Unsupported, 0, LLT(), LegalityQuery(G_AND, {v2s33}));
  }

  // Test changeElementCountTo
  {
    LegalizerInfo LI;

    // Type index form
    LI.getActionDefinitionsBuilder(G_SELECT)
      .moreElementsIf(isScalar(1), changeElementCountTo(1, 0));

    // Raw type form
    LI.getActionDefinitionsBuilder(G_ADD)
        .fewerElementsIf(typeIs(0, v4s32),
                         changeElementCountTo(0, ElementCount::getFixed(2)))
        .fewerElementsIf(typeIs(0, v8s32),
                         changeElementCountTo(0, ElementCount::getFixed(1)))
        .fewerElementsIf(typeIs(0, LLT::scalable_vector(4, s16)),
                         changeElementCountTo(0, ElementCount::getScalable(2)))
        .fewerElementsIf(typeIs(0, LLT::scalable_vector(8, s16)),
                         changeElementCountTo(0, ElementCount::getFixed(1)));


    EXPECT_ACTION(MoreElements, 1, v4s1, LegalityQuery(G_SELECT, {v4s32, s1}));
    EXPECT_ACTION(MoreElements, 1, v2s1, LegalityQuery(G_SELECT, {v2s32, s1}));
    EXPECT_ACTION(MoreElements, 1, v2s1, LegalityQuery(G_SELECT, {v2s32, s1}));
    EXPECT_ACTION(MoreElements, 1, v4s1, LegalityQuery(G_SELECT, {v4p0, s1}));

    EXPECT_ACTION(MoreElements, 1, LLT::scalable_vector(2, 1),
                  LegalityQuery(G_SELECT, {LLT::scalable_vector(2, 32), s1}));
    EXPECT_ACTION(MoreElements, 1, LLT::scalable_vector(4, 1),
                  LegalityQuery(G_SELECT, {LLT::scalable_vector(4, 32), s1}));
    EXPECT_ACTION(MoreElements, 1, LLT::scalable_vector(2, s1),
                  LegalityQuery(G_SELECT, {LLT::scalable_vector(2, p0), s1}));

    EXPECT_ACTION(FewerElements, 0, v2s32, LegalityQuery(G_ADD, {v4s32}));
    EXPECT_ACTION(FewerElements, 0, s32, LegalityQuery(G_ADD, {v8s32}));

    EXPECT_ACTION(FewerElements, 0, LLT::scalable_vector(2, 16),
                  LegalityQuery(G_ADD, {LLT::scalable_vector(4, 16)}));
    EXPECT_ACTION(FewerElements, 0, s16,
                  LegalityQuery(G_ADD, {LLT::scalable_vector(8, 16)}));
  }

  // Test minScalarEltSameAsIf
  {
    LegalizerInfo LI;

    LI.getActionDefinitionsBuilder(G_SELECT).minScalarEltSameAsIf(
        all(isVector(0), isVector(1)), 1, 0);
    LLT p1 = LLT::pointer(1, 32);
    LLT v2p1 = LLT::fixed_vector(2, p1);

    EXPECT_ACTION(WidenScalar, 1, v2s32, LegalityQuery(G_SELECT, {v2p0, v2s1}));
    EXPECT_ACTION(WidenScalar, 1, v2s32, LegalityQuery(G_SELECT, {v2p1, v2s1}));
  }
}

TEST(LegalizerInfoTest, MMOAlignment) {
  using namespace TargetOpcode;

  const LLT s32 = LLT::scalar(32);
  const LLT p0 = LLT::pointer(0, 64);

  {
    LegalizerInfo LI;
    LI.getActionDefinitionsBuilder(G_LOAD)
      .legalForTypesWithMemDesc({{s32, p0, s32, 32}});


    EXPECT_ACTION(
        Legal, 0, LLT(),
        LegalityQuery(G_LOAD, {s32, p0},
                      LegalityQuery::MemDesc{s32, 32, AtomicOrdering::NotAtomic,
                                             AtomicOrdering::NotAtomic}));
    EXPECT_ACTION(
        Unsupported, 0, LLT(),
        LegalityQuery(G_LOAD, {s32, p0},
                      LegalityQuery::MemDesc{s32, 16, AtomicOrdering::NotAtomic,
                                             AtomicOrdering::NotAtomic}));
    EXPECT_ACTION(
        Unsupported, 0, LLT(),
        LegalityQuery(G_LOAD, {s32, p0},
                      LegalityQuery::MemDesc{s32, 8, AtomicOrdering::NotAtomic,
                                             AtomicOrdering::NotAtomic}));
  }

  // Test that the maximum supported alignment value isn't truncated
  {
    // Maximum IR defined alignment in bytes.
    const uint64_t MaxAlignment = UINT64_C(1) << 29;
    const uint64_t MaxAlignInBits = 8 * MaxAlignment;
    LegalizerInfo LI;
    LI.getActionDefinitionsBuilder(G_LOAD)
      .legalForTypesWithMemDesc({{s32, p0, s32, MaxAlignInBits}});


    EXPECT_ACTION(
        Legal, 0, LLT(),
        LegalityQuery(G_LOAD, {s32, p0},
                      LegalityQuery::MemDesc{s32, MaxAlignInBits,
                                             AtomicOrdering::NotAtomic,
                                             AtomicOrdering::NotAtomic}));
    EXPECT_ACTION(
        Unsupported, 0, LLT(),
        LegalityQuery(G_LOAD, {s32, p0},
                      LegalityQuery::MemDesc{s32, 8, AtomicOrdering::NotAtomic,
                                             AtomicOrdering::NotAtomic}));
  }
}

// This code sequence doesn't do anything, but it covers a previously uncovered
// codepath that used to crash in MSVC x86_32 debug mode.
TEST(LegalizerInfoTest, MSVCDebugMiscompile) {
  const LLT S1 = LLT::scalar(1);
  const LLT P0 = LLT::pointer(0, 32);
  LegalizerInfo LI;
  auto Builder = LI.getActionDefinitionsBuilder(TargetOpcode::G_PTRTOINT);
  (void)Builder.legalForCartesianProduct({S1}, {P0});
}
