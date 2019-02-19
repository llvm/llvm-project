//===- DeLICMTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "polly/DeLICM.h"
#include "polly/Support/ISLTools.h"
#include "gtest/gtest.h"
#include <isl/map.h>
#include <isl/set.h>
#include <isl/stream.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <memory>

using namespace llvm;
using namespace polly;

namespace {

/// Get the universes of all spaces in @p USet.
isl::union_set unionSpace(const isl::union_set &USet) {
  auto Result = isl::union_set::empty(USet.get_space());
  for (isl::set Set : USet.get_set_list()) {
    isl::space Space = Set.get_space();
    isl::set Universe = isl::set::universe(Space);
    Result = Result.add_set(Universe);
  }
  return Result;
}

void completeLifetime(isl::union_set Universe, isl::union_map OccupiedAndKnown,
                      isl::union_set &Occupied, isl::union_map &Known,
                      isl::union_set &Undef) {
  auto ParamSpace = Universe.get_space();

  if (Undef && !Occupied) {
    assert(!Occupied);
    Occupied = Universe.subtract(Undef);
  }

  if (OccupiedAndKnown) {
    assert(!Known);

    Known = isl::union_map::empty(ParamSpace);

    if (!Occupied)
      Occupied = OccupiedAndKnown.domain();

    for (isl::map Map : OccupiedAndKnown.get_map_list()) {
      if (!Map.has_tuple_name(isl::dim::out))
        continue;
      Known = Known.add_map(Map);
    }
  }

  if (!Undef) {
    assert(Occupied);
    Undef = Universe.subtract(Occupied);
  }

  if (!Known) { // By default, nothing is known.
    Known = isl::union_map::empty(ParamSpace);
  }

  // Conditions that must hold when returning.
  assert(Occupied);
  assert(Undef);
  assert(Known);
}

typedef struct {
  const char *OccupiedStr;
  const char *UndefStr;
  const char *WrittenStr;
} KnowledgeStr;

isl::union_set parseSetOrNull(isl_ctx *Ctx, const char *Str) {
  if (!Str)
    return nullptr;
  return isl::union_set(Ctx, Str);
}

isl::union_map parseMapOrNull(isl_ctx *Ctx, const char *Str) {
  if (!Str)
    return nullptr;
  return isl::union_map(Ctx, Str);
}

bool checkIsConflictingNonsymmetricCommon(
    isl_ctx *Ctx, isl::union_map ExistingOccupiedAndKnown,
    isl::union_set ExistingUnused, isl::union_map ExistingWritten,
    isl::union_map ProposedOccupiedAndKnown, isl::union_set ProposedUnused,
    isl::union_map ProposedWritten) {
  // Determine universe (set of all possible domains).
  auto Universe = isl::union_set::empty(isl::space::params_alloc(Ctx, 0));
  if (ExistingOccupiedAndKnown)
    Universe = Universe.unite(ExistingOccupiedAndKnown.domain());
  if (ExistingUnused)
    Universe = Universe.unite(ExistingUnused);
  if (ExistingWritten)
    Universe = Universe.unite(ExistingWritten.domain());
  if (ProposedOccupiedAndKnown)
    Universe = Universe.unite(ProposedOccupiedAndKnown.domain());
  if (ProposedUnused)
    Universe = Universe.unite(ProposedUnused);
  if (ProposedWritten)
    Universe = Universe.unite(ProposedWritten.domain());

  Universe = unionSpace(Universe);

  // Add a space the universe that does not occur anywhere else to ensure
  // robustness. Use &NewId to ensure that this Id is unique.
  isl::id NewId = isl::id::alloc(Ctx, "Unrelated", &NewId);
  // The space must contains at least one dimension to allow order
  // modifications.
  auto NewSpace = isl::space(Ctx, 0, 1);
  NewSpace = NewSpace.set_tuple_id(isl::dim::set, NewId);
  auto NewSet = isl::set::universe(NewSpace);
  Universe = Universe.add_set(NewSet);

  // Using the universe, fill missing data.
  isl::union_set ExistingOccupied;
  isl::union_map ExistingKnown;
  completeLifetime(Universe, ExistingOccupiedAndKnown, ExistingOccupied,
                   ExistingKnown, ExistingUnused);

  isl::union_set ProposedOccupied;
  isl::union_map ProposedKnown;
  completeLifetime(Universe, ProposedOccupiedAndKnown, ProposedOccupied,
                   ProposedKnown, ProposedUnused);

  auto Result = isConflicting(ExistingOccupied, ExistingUnused, ExistingKnown,
                              ExistingWritten, ProposedOccupied, ProposedUnused,
                              ProposedKnown, ProposedWritten);

  // isConflicting does not require ExistingOccupied nor ProposedUnused and are
  // implicitly assumed to be the remainder elements. Test the implicitness as
  // well.
  EXPECT_EQ(Result,
            isConflicting(ExistingOccupied, ExistingUnused, ExistingKnown,
                          ExistingWritten, ProposedOccupied, {}, ProposedKnown,
                          ProposedWritten));
  EXPECT_EQ(Result,
            isConflicting({}, ExistingUnused, ExistingKnown, ExistingWritten,
                          ProposedOccupied, ProposedUnused, ProposedKnown,
                          ProposedWritten));
  EXPECT_EQ(Result, isConflicting({}, ExistingUnused, ExistingKnown,
                                  ExistingWritten, ProposedOccupied, {},
                                  ProposedKnown, ProposedWritten));

  return Result;
}

bool checkIsConflictingNonsymmetricKnown(KnowledgeStr Existing,
                                         KnowledgeStr Proposed) {
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> Ctx(isl_ctx_alloc(),
                                                        &isl_ctx_free);

  // Parse knowledge.
  auto ExistingOccupiedAndKnown =
      parseMapOrNull(Ctx.get(), Existing.OccupiedStr);
  auto ExistingUnused = parseSetOrNull(Ctx.get(), Existing.UndefStr);
  auto ExistingWritten = parseMapOrNull(Ctx.get(), Existing.WrittenStr);

  auto ProposedOccupiedAndKnown =
      parseMapOrNull(Ctx.get(), Proposed.OccupiedStr);
  auto ProposedUnused = parseSetOrNull(Ctx.get(), Proposed.UndefStr);
  auto ProposedWritten = parseMapOrNull(Ctx.get(), Proposed.WrittenStr);

  return checkIsConflictingNonsymmetricCommon(
      Ctx.get(), ExistingOccupiedAndKnown, ExistingUnused, ExistingWritten,
      ProposedOccupiedAndKnown, ProposedUnused, ProposedWritten);
}

bool checkIsConflictingNonsymmetric(KnowledgeStr Existing,
                                    KnowledgeStr Proposed) {
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> Ctx(isl_ctx_alloc(),
                                                        &isl_ctx_free);

  // Parse knowledge.
  auto ExistingOccupied = parseSetOrNull(Ctx.get(), Existing.OccupiedStr);
  auto ExistingUnused = parseSetOrNull(Ctx.get(), Existing.UndefStr);
  auto ExistingWritten = parseSetOrNull(Ctx.get(), Existing.WrittenStr);

  auto ProposedOccupied = parseSetOrNull(Ctx.get(), Proposed.OccupiedStr);
  auto ProposedUnused = parseSetOrNull(Ctx.get(), Proposed.UndefStr);
  auto ProposedWritten = parseSetOrNull(Ctx.get(), Proposed.WrittenStr);

  return checkIsConflictingNonsymmetricCommon(
      Ctx.get(), isl::union_map::from_domain(ExistingOccupied), ExistingUnused,
      isl::union_map::from_domain(ExistingWritten),
      isl::union_map::from_domain(ProposedOccupied), ProposedUnused,
      isl::union_map::from_domain(ProposedWritten));
}

bool checkIsConflicting(KnowledgeStr Existing, KnowledgeStr Proposed) {
  auto Forward = checkIsConflictingNonsymmetric(Existing, Proposed);
  auto Backward = checkIsConflictingNonsymmetric(Proposed, Existing);

  // isConflicting should be symmetric.
  EXPECT_EQ(Forward, Backward);

  return Forward || Backward;
}

bool checkIsConflictingKnown(KnowledgeStr Existing, KnowledgeStr Proposed) {
  auto Forward = checkIsConflictingNonsymmetricKnown(Existing, Proposed);
  auto Backward = checkIsConflictingNonsymmetricKnown(Proposed, Existing);

  // checkIsConflictingKnown should be symmetric.
  EXPECT_EQ(Forward, Backward);

  return Forward || Backward;
}

TEST(DeLICM, isConflicting) {

  // Check occupied vs. occupied.
  EXPECT_TRUE(
      checkIsConflicting({"{ Dom[i] }", nullptr, "{}"}, {nullptr, "{}", "{}"}));
  EXPECT_TRUE(checkIsConflicting({"{ Dom[i] }", nullptr, "{}"},
                                 {"{ Dom[i] }", nullptr, "{}"}));
  EXPECT_FALSE(checkIsConflicting({"{ Dom[0] }", nullptr, "{}"},
                                  {nullptr, "{ Dom[0] }", "{}"}));
  EXPECT_FALSE(checkIsConflicting({"{ Dom[i] : i != 0 }", nullptr, "{}"},
                                  {"{ Dom[0] }", nullptr, "{}"}));

  // Check occupied vs. occupied with known values.
  EXPECT_FALSE(checkIsConflictingKnown({"{ Dom[i] -> Val[] }", nullptr, "{}"},
                                       {"{ Dom[i] -> Val[] }", nullptr, "{}"}));
  EXPECT_TRUE(checkIsConflictingKnown({"{ Dom[i] -> ValA[] }", nullptr, "{}"},
                                      {"{ Dom[i] -> ValB[] }", nullptr, "{}"}));
  EXPECT_TRUE(checkIsConflictingKnown({"{ Dom[i] -> Val[] }", nullptr, "{}"},
                                      {"{ Dom[i] -> [] }", nullptr, "{}"}));
  EXPECT_FALSE(checkIsConflictingKnown({"{ Dom[0] -> Val[] }", nullptr, "{}"},
                                       {nullptr, "{ Dom[0] }", "{}"}));
  EXPECT_FALSE(checkIsConflictingKnown(
      {"{ Dom[i] -> Val[]; Dom[i] -> Phi[] }", nullptr, "{}"},
      {"{ Dom[i] -> Val[] }", nullptr, "{}"}));

  // An implementation using subtract would have exponential runtime on patterns
  // such as this one.
  EXPECT_TRUE(checkIsConflictingKnown(
      {"{ Dom[i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15]"
       "-> Val[] }",
       nullptr, "{}"},
      {"[p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,q0,"
       "q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15] -> {"
       "Dom[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] -> Val[];"
       "Dom[p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15] -> Val[];"
       "Dom[q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15] -> Val[] }",
       "{}", "{}"}));

  // Check occupied vs. written.
  EXPECT_TRUE(
      checkIsConflicting({nullptr, "{}", "{}"}, {"{}", nullptr, "{ Dom[0] }"}));
  EXPECT_FALSE(
      checkIsConflicting({"{}", nullptr, "{}"}, {"{}", nullptr, "{ Dom[0] }"}));

  EXPECT_TRUE(checkIsConflicting({"{ Dom[i] }", nullptr, "{}"},
                                 {"{}", nullptr, "{ Dom[0] }"}));
  EXPECT_FALSE(checkIsConflicting({"{ DomA[i] }", nullptr, "{}"},
                                  {"{}", nullptr, "{ DomB[0] }"}));

  // Dom[1] represents the time between 0 and 1. Now Proposed writes at timestep
  // 0 such that will have a different value between 0 and 1. Hence it is
  // conflicting with Existing.
  EXPECT_TRUE(checkIsConflicting({"{ Dom[1] }", nullptr, "{}"},
                                 {"{}", nullptr, "{ Dom[0] }"}));
  EXPECT_FALSE(checkIsConflicting({"{ Dom[i] : i != 1 }", nullptr, "{}"},
                                  {"{}", nullptr, "{ Dom[0] }"}));

  // Check occupied vs. written with known values.
  EXPECT_FALSE(checkIsConflictingKnown({"{ Dom[i] -> Val[] }", nullptr, "{}"},
                                       {"{}", nullptr, "{ Dom[0] -> Val[] }"}));
  EXPECT_TRUE(checkIsConflictingKnown({"{ Dom[i] -> ValA[] }", nullptr, "{}"},
                                      {"{}", nullptr, "{ Dom[0] -> ValB[] }"}));
  EXPECT_TRUE(checkIsConflictingKnown({"{ Dom[i] -> Val[] }", nullptr, "{}"},
                                      {"{}", nullptr, "{ Dom[0] -> [] }"}));
  EXPECT_TRUE(checkIsConflictingKnown({"{ Dom[i] -> [] }", nullptr, "{}"},
                                      {"{}", nullptr, "{ Dom[0] -> Val[] }"}));

  // The same value can be known under multiple names, for instance a PHINode
  // has the same value as one of the incoming values. One matching pair
  // suffices.
  EXPECT_FALSE(checkIsConflictingKnown(
      {"{ Dom[i] -> Val[]; Dom[i] -> Phi[] }", nullptr, "{}"},
      {"{}", nullptr, "{ Dom[0] -> Val[] }"}));
  EXPECT_FALSE(checkIsConflictingKnown(
      {"{ Dom[i] -> Val[] }", nullptr, "{}"},
      {"{}", nullptr, "{ Dom[0] -> Val[]; Dom[0] -> Phi[] }"}));

  // Check written vs. written.
  EXPECT_TRUE(checkIsConflicting({"{}", nullptr, "{ Dom[0] }"},
                                 {"{}", nullptr, "{ Dom[0] }"}));
  EXPECT_FALSE(checkIsConflicting({"{}", nullptr, "{ Dom[-1] }"},
                                  {"{}", nullptr, "{ Dom[0] }"}));
  EXPECT_FALSE(checkIsConflicting({"{}", nullptr, "{ Dom[1] }"},
                                  {"{}", nullptr, "{ Dom[0] }"}));

  // Check written vs. written with known values.
  EXPECT_FALSE(checkIsConflictingKnown({"{}", nullptr, "{ Dom[0] -> Val[] }"},
                                       {"{}", nullptr, "{ Dom[0] -> Val[] }"}));
  EXPECT_TRUE(checkIsConflictingKnown({"{}", nullptr, "{ Dom[0] -> ValA[] }"},
                                      {"{}", nullptr, "{ Dom[0] -> ValB[] }"}));
  EXPECT_TRUE(checkIsConflictingKnown({"{}", nullptr, "{ Dom[0] -> Val[] }"},
                                      {"{}", nullptr, "{ Dom[0] -> [] }"}));
  EXPECT_FALSE(checkIsConflictingKnown(
      {"{}", nullptr, "{ Dom[0] -> Val[]}"},
      {"{}", nullptr, "{ Dom[0] -> Val[]; Dom[0] -> Phi[] }"}));
}
} // anonymous namespace
