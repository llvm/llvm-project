//===- unittest/TableGen/AutomataTest.cpp - DFA tests ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Automaton.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
using testing::ContainerEq;
using testing::UnorderedElementsAre;

// Bring in the enums created by SearchableTables.td.
#define GET_SymKind_DECL
#define GET_BinRequirementKindEnum_DECL
#include "AutomataTables.inc"

// And bring in the automata from Automata.td.
#define GET_SimpleAutomaton_DECL
#define GET_TupleAutomaton_DECL
#define GET_NfaAutomaton_DECL
#define GET_BinPackerAutomaton_DECL
#include "AutomataAutomata.inc"

TEST(Automata, SimpleAutomatonAcceptsFromInitialState) {
  Automaton<SymKind> A{ArrayRef(SimpleAutomatonTransitions)};
  EXPECT_TRUE(A.add(SK_a));
  A.reset();
  EXPECT_TRUE(A.add(SK_b));
  A.reset();
  EXPECT_TRUE(A.add(SK_c));
  A.reset();
  EXPECT_FALSE(A.add(SK_d));
}

TEST(Automata, SimpleAutomatonAcceptsSequences) {
  Automaton<SymKind> A{ArrayRef(SimpleAutomatonTransitions)};
  // Test sequence <a b>
  A.reset();
  EXPECT_TRUE(A.add(SK_a));
  EXPECT_TRUE(A.add(SK_b));

  // Test sequence <a c> is rejected (c cannot get bit 0b10);
  A.reset();
  EXPECT_TRUE(A.add(SK_a));
  EXPECT_FALSE(A.add(SK_c));

  // Symmetric test: sequence <c a> is rejected.
  A.reset();
  EXPECT_TRUE(A.add(SK_c));
  EXPECT_FALSE(A.add(SK_a));
}

TEST(Automata, TupleAutomatonAccepts) {
  Automaton<TupleAutomatonAction> A{ArrayRef(TupleAutomatonTransitions)};
  A.reset();
  EXPECT_TRUE(
      A.add(TupleAutomatonAction{SK_a, SK_b, "yeet"}));
  A.reset();
  EXPECT_FALSE(
      A.add(TupleAutomatonAction{SK_a, SK_a, "yeet"}));
  A.reset();
  EXPECT_FALSE(
      A.add(TupleAutomatonAction{SK_a, SK_b, "feet"}));
  A.reset();
  EXPECT_TRUE(
      A.add(TupleAutomatonAction{SK_b, SK_b, "foo"}));
}

TEST(Automata, NfaAutomatonAccepts) {
  Automaton<SymKind> A{ArrayRef(NfaAutomatonTransitions)};

  // Test sequences <a a>, <a b>, <b a>, <b b>. All should be accepted.
  A.reset();
  EXPECT_TRUE(A.add(SK_a));
  EXPECT_TRUE(A.add(SK_a));
  A.reset();
  EXPECT_TRUE(A.add(SK_a));
  EXPECT_TRUE(A.add(SK_b));
  A.reset();
  EXPECT_TRUE(A.add(SK_b));
  EXPECT_TRUE(A.add(SK_a));
  A.reset();
  EXPECT_TRUE(A.add(SK_b));
  EXPECT_TRUE(A.add(SK_b));

  // Expect that <b b b> is not accepted.
  A.reset();
  EXPECT_TRUE(A.add(SK_b));
  EXPECT_TRUE(A.add(SK_b));
  EXPECT_FALSE(A.add(SK_b));
}

TEST(Automata, BinPackerAutomatonAccepts) {
  Automaton<BinPackerAutomatonAction> A{
      ArrayRef(BinPackerAutomatonTransitions)};

  // Expect that we can pack two double-bins in 0-4, then no more in 0-4.
  A.reset();
  EXPECT_TRUE(A.add(BRK_0_to_4_dbl));
  EXPECT_TRUE(A.add(BRK_0_to_4_dbl));
  EXPECT_FALSE(A.add(BRK_0_to_4));

  // Expect that we can pack two double-bins in 0-4, two more in 0-6 then no
  // more.
  A.reset();
  EXPECT_TRUE(A.add(BRK_0_to_4_dbl));
  EXPECT_TRUE(A.add(BRK_0_to_4_dbl));
  EXPECT_TRUE(A.add(BRK_0_to_6));
  EXPECT_TRUE(A.add(BRK_0_to_6));
  EXPECT_FALSE(A.add(BRK_0_to_6));

  // Expect that we can pack BRK_0_to_6 five times to occupy five bins, then
  // cannot allocate any double-bins.
  A.reset();
  for (unsigned I = 0; I < 5; ++I)
    EXPECT_TRUE(A.add(BRK_0_to_6));
  EXPECT_FALSE(A.add(BRK_0_to_6_dbl));
}

// The state we defined in TableGen uses the least significant 6 bits to represent a bin state.
#define BINS(a, b, c, d, e, f)                                                 \
  ((a << 5) | (b << 4) | (c << 3) | (d << 2) | (e << 1) | (f << 0))

TEST(Automata, BinPackerAutomatonExplains) {
  Automaton<BinPackerAutomatonAction> A{
      ArrayRef(BinPackerAutomatonTransitions),
      ArrayRef(BinPackerAutomatonTransitionInfo)};
  // Pack two double-bins in 0-4, then a single bin in 0-6.
  EXPECT_TRUE(A.add(BRK_0_to_4_dbl));
  EXPECT_TRUE(A.add(BRK_0_to_4_dbl));
  EXPECT_TRUE(A.add(BRK_0_to_6));
  EXPECT_THAT(
      A.getNfaPaths(),
      UnorderedElementsAre(
          // Allocate {0,1} first, then 6.
          ContainerEq(NfaPath{BINS(0, 0, 0, 0, 1, 1), BINS(0, 0, 1, 1, 1, 1),
                              BINS(1, 0, 1, 1, 1, 1)}),
          // Allocate {0,1} first, then 5.
          ContainerEq(NfaPath{BINS(0, 0, 0, 0, 1, 1), BINS(0, 0, 1, 1, 1, 1),
                              BINS(0, 1, 1, 1, 1, 1)}),
          // Allocate {2,3} first, then 6.
          ContainerEq(NfaPath{BINS(0, 0, 1, 1, 0, 0), BINS(0, 0, 1, 1, 1, 1),
                              BINS(1, 0, 1, 1, 1, 1)}),
          // Allocate {2,3} first, then 5.
          ContainerEq(NfaPath{BINS(0, 0, 1, 1, 0, 0), BINS(0, 0, 1, 1, 1, 1),
                              BINS(0, 1, 1, 1, 1, 1)})));
}
