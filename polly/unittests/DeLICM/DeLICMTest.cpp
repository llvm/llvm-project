//===- DeLICMTest.cpp ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "polly/DeLICM.h"
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
  auto Result = give(isl_union_set_empty(isl_union_set_get_space(USet.keep())));
  foreachElt(USet, [=, &Result](isl::set Set) {
    auto Space = give(isl_set_get_space(Set.keep()));
    auto Universe = give(isl_set_universe(Space.take()));
    Result = give(isl_union_set_add_set(Result.take(), Universe.take()));
  });
  return Result;
}

void completeLifetime(isl::union_set Universe, isl::union_set &Unknown,
                      isl::union_set &Undef) {
  if (!Unknown) {
    assert(Undef);
    Unknown = give(isl_union_set_subtract(Universe.copy(), Undef.copy()));
  }

  if (!Undef) {
    assert(Unknown);
    Undef = give(isl_union_set_subtract(Universe.copy(), Unknown.copy()));
  }
}

typedef struct {
  const char *OccupiedStr;
  const char *UndefStr;
  const char *WrittenStr;
} Knowledge;

isl::union_set parseSetOrNull(isl_ctx *Ctx, const char *Str) {
  if (!Str)
    return nullptr;
  return isl::union_set(Ctx, Str);
}

bool checkIsConflictingNonsymmetric(Knowledge Existing, Knowledge Proposed) {
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> Ctx(isl_ctx_alloc(),
                                                        &isl_ctx_free);

  // Parse knowledge.
  auto ExistingOccupied = parseSetOrNull(Ctx.get(), Existing.OccupiedStr);
  auto ExistingUnused = parseSetOrNull(Ctx.get(), Existing.UndefStr);
  auto ExistingWritten = parseSetOrNull(Ctx.get(), Existing.WrittenStr);

  auto ProposedOccupied = parseSetOrNull(Ctx.get(), Proposed.OccupiedStr);
  auto ProposedUnused = parseSetOrNull(Ctx.get(), Proposed.UndefStr);
  auto ProposedWritten = parseSetOrNull(Ctx.get(), Proposed.WrittenStr);

  // Determine universe (set of all possible domains).
  auto Universe =
      give(isl_union_set_empty(isl_space_params_alloc(Ctx.get(), 0)));
  if (ExistingOccupied)
    Universe =
        give(isl_union_set_union(Universe.take(), ExistingOccupied.copy()));
  if (ExistingUnused)
    Universe =
        give(isl_union_set_union(Universe.take(), ExistingUnused.copy()));
  if (ExistingWritten)
    Universe =
        give(isl_union_set_union(Universe.take(), ExistingWritten.copy()));
  if (ProposedOccupied)
    Universe =
        give(isl_union_set_union(Universe.take(), ProposedOccupied.copy()));
  if (ProposedUnused)
    Universe =
        give(isl_union_set_union(Universe.take(), ProposedUnused.copy()));
  if (ProposedWritten)
    Universe =
        give(isl_union_set_union(Universe.take(), ProposedWritten.copy()));
  Universe = unionSpace(Universe);

  // Add a space the universe that does not occur anywhere else to ensure
  // robustness. Use &NewId to ensure that this Id is unique.
  isl::id NewId = give(isl_id_alloc(Ctx.get(), "Unrelated", &NewId));
  // The space must contains at least one dimension to allow order
  // modifications.
  auto NewSpace = give(isl_space_set_alloc(Ctx.get(), 0, 1));
  NewSpace =
      give(isl_space_set_tuple_id(NewSpace.take(), isl_dim_set, NewId.copy()));
  auto NewSet = give(isl_set_universe(NewSpace.take()));
  Universe = give(isl_union_set_add_set(Universe.take(), NewSet.take()));

  // Using the universe, fill missing data.
  completeLifetime(Universe, ExistingOccupied, ExistingUnused);
  completeLifetime(Universe, ProposedOccupied, ProposedUnused);

  auto Result =
      isConflicting(ExistingOccupied, ExistingUnused, ExistingWritten,
                    ProposedOccupied, ProposedUnused, ProposedWritten);

  // isConflicting does not require ExistingOccupied nor ProposedUnused and are
  // implicitly assumed to be the remainder elements. Test the implicitness as
  // well.
  EXPECT_EQ(Result,
            isConflicting(ExistingOccupied, ExistingUnused, ExistingWritten,
                          ProposedOccupied, {}, ProposedWritten));
  EXPECT_EQ(Result,
            isConflicting({}, ExistingUnused, ExistingWritten, ProposedOccupied,
                          ProposedUnused, ProposedWritten));
  EXPECT_EQ(Result, isConflicting({}, ExistingUnused, ExistingWritten,
                                  ProposedOccupied, {}, ProposedWritten));

  return Result;
}

bool checkIsConflicting(Knowledge Existing, Knowledge Proposed) {
  auto Forward = checkIsConflictingNonsymmetric(Existing, Proposed);
  auto Backward = checkIsConflictingNonsymmetric(Proposed, Existing);

  // isConflicting should be symmetric.
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

  // Check written vs. written.
  EXPECT_TRUE(checkIsConflicting({"{}", nullptr, "{ Dom[0] }"},
                                 {"{}", nullptr, "{ Dom[0] }"}));
  EXPECT_FALSE(checkIsConflicting({"{}", nullptr, "{ Dom[-1] }"},
                                  {"{}", nullptr, "{ Dom[0] }"}));
  EXPECT_FALSE(checkIsConflicting({"{}", nullptr, "{ Dom[1] }"},
                                  {"{}", nullptr, "{ Dom[0] }"}));
}
} // anonymous namespace
