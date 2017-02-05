//===- IslTest.cpp ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "polly/Support/GICHelper.h"
#include "polly/Support/ISLTools.h"
#include "gtest/gtest.h"
#include "isl/stream.h"
#include "isl/val.h"

using namespace llvm;
using namespace polly;

static IslPtr<isl_space> parseSpace(isl_ctx *Ctx, const char *Str) {
  isl_stream *Stream = isl_stream_new_str(Ctx, Str);
  auto Obj = isl_stream_read_obj(Stream);

  IslPtr<isl_space> Result;
  if (Obj.type == isl_obj_set)
    Result = give(isl_set_get_space(static_cast<isl_set *>(Obj.v)));
  else if (Obj.type == isl_obj_map)
    Result = give(isl_map_get_space(static_cast<isl_map *>(Obj.v)));

  isl_stream_free(Stream);
  if (Obj.type)
    Obj.type->free(Obj.v);
  return Result;
}

#define SPACE(Str) parseSpace(Ctx.get(), Str)

#define SET(Str) give(isl_set_read_from_str(Ctx.get(), Str))
#define MAP(Str) give(isl_map_read_from_str(Ctx.get(), Str))

#define USET(Str) give(isl_union_set_read_from_str(Ctx.get(), Str))
#define UMAP(Str) give(isl_union_map_read_from_str(Ctx.get(), Str))

static bool operator==(const IslPtr<isl_space> &LHS,
                       const IslPtr<isl_space> &RHS) {
  auto IsEqual = isl_space_is_equal(LHS.keep(), RHS.keep());
  EXPECT_NE(isl_bool_error, IsEqual);
  return IsEqual;
}

static bool operator==(const IslPtr<isl_set> &LHS, const IslPtr<isl_set> &RHS) {
  auto IsEqual = isl_set_is_equal(LHS.keep(), RHS.keep());
  EXPECT_NE(isl_bool_error, IsEqual);
  return IsEqual;
}

static bool operator==(const IslPtr<isl_map> &LHS, const IslPtr<isl_map> &RHS) {
  auto IsEqual = isl_map_is_equal(LHS.keep(), RHS.keep());
  EXPECT_NE(isl_bool_error, IsEqual);
  return IsEqual;
}

static bool operator==(const IslPtr<isl_union_set> &LHS,
                       const IslPtr<isl_union_set> &RHS) {
  auto IsEqual = isl_union_set_is_equal(LHS.keep(), RHS.keep());
  EXPECT_NE(isl_bool_error, IsEqual);
  return IsEqual;
}

static bool operator==(const IslPtr<isl_union_map> &LHS,
                       const IslPtr<isl_union_map> &RHS) {
  auto IsEqual = isl_union_map_is_equal(LHS.keep(), RHS.keep());
  EXPECT_NE(isl_bool_error, IsEqual);
  return IsEqual;
}

namespace {

TEST(Isl, APIntToIslVal) {
  isl_ctx *IslCtx = isl_ctx_alloc();

  {
    APInt APZero(1, 0, true);
    auto *IslZero = isl_valFromAPInt(IslCtx, APZero, true);
    EXPECT_EQ(isl_bool_true, isl_val_is_zero(IslZero));
    isl_val_free(IslZero);
  }

  {
    APInt APNOne(1, -1, true);
    auto *IslNOne = isl_valFromAPInt(IslCtx, APNOne, true);
    EXPECT_EQ(isl_bool_true, isl_val_is_negone(IslNOne));
    isl_val_free(IslNOne);
  }

  {
    APInt APZero(1, 0, false);
    auto *IslZero = isl_valFromAPInt(IslCtx, APZero, false);
    EXPECT_EQ(isl_bool_true, isl_val_is_zero(IslZero));
    isl_val_free(IslZero);
  }

  {
    APInt APOne(1, 1, false);
    auto *IslOne = isl_valFromAPInt(IslCtx, APOne, false);
    EXPECT_EQ(isl_bool_true, isl_val_is_one(IslOne));
    isl_val_free(IslOne);
  }

  {
    APInt APNTwo(2, -2, true);
    auto *IslNTwo = isl_valFromAPInt(IslCtx, APNTwo, true);
    auto *IslNTwoCmp = isl_val_int_from_si(IslCtx, -2);
    EXPECT_EQ(isl_bool_true, isl_val_eq(IslNTwo, IslNTwoCmp));
    isl_val_free(IslNTwo);
    isl_val_free(IslNTwoCmp);
  }

  {
    APInt APNOne(32, -1, true);
    auto *IslNOne = isl_valFromAPInt(IslCtx, APNOne, true);
    EXPECT_EQ(isl_bool_true, isl_val_is_negone(IslNOne));
    isl_val_free(IslNOne);
  }

  {
    APInt APZero(32, 0, false);
    auto *IslZero = isl_valFromAPInt(IslCtx, APZero, false);
    EXPECT_EQ(isl_bool_true, isl_val_is_zero(IslZero));
    isl_val_free(IslZero);
  }

  {
    APInt APOne(32, 1, false);
    auto *IslOne = isl_valFromAPInt(IslCtx, APOne, false);
    EXPECT_EQ(isl_bool_true, isl_val_is_one(IslOne));
    isl_val_free(IslOne);
  }

  {
    APInt APTwo(32, 2, false);
    auto *IslTwo = isl_valFromAPInt(IslCtx, APTwo, false);
    EXPECT_EQ(0, isl_val_cmp_si(IslTwo, 2));
    isl_val_free(IslTwo);
  }

  {
    APInt APNOne(32, (1ull << 32) - 1, false);
    auto *IslNOne = isl_valFromAPInt(IslCtx, APNOne, false);
    auto *IslRef = isl_val_int_from_ui(IslCtx, (1ull << 32) - 1);
    EXPECT_EQ(isl_bool_true, isl_val_eq(IslNOne, IslRef));
    isl_val_free(IslNOne);
    isl_val_free(IslRef);
  }

  {
    APInt APLarge(130, 2, false);
    APLarge = APLarge.shl(70);
    auto *IslLarge = isl_valFromAPInt(IslCtx, APLarge, false);
    auto *IslRef = isl_val_int_from_ui(IslCtx, 71);
    IslRef = isl_val_2exp(IslRef);
    EXPECT_EQ(isl_bool_true, isl_val_eq(IslLarge, IslRef));
    isl_val_free(IslLarge);
    isl_val_free(IslRef);
  }

  isl_ctx_free(IslCtx);
}

TEST(Isl, IslValToAPInt) {
  isl_ctx *IslCtx = isl_ctx_alloc();

  {
    auto *IslNOne = isl_val_int_from_si(IslCtx, -1);
    auto APNOne = APIntFromVal(IslNOne);
    // Compare with the two's complement of -1 in a 1-bit integer.
    EXPECT_EQ(1, APNOne);
    EXPECT_EQ(1u, APNOne.getBitWidth());
  }

  {
    auto *IslNTwo = isl_val_int_from_si(IslCtx, -2);
    auto APNTwo = APIntFromVal(IslNTwo);
    // Compare with the two's complement of -2 in a 2-bit integer.
    EXPECT_EQ(2, APNTwo);
    EXPECT_EQ(2u, APNTwo.getBitWidth());
  }

  {
    auto *IslNThree = isl_val_int_from_si(IslCtx, -3);
    auto APNThree = APIntFromVal(IslNThree);
    // Compare with the two's complement of -3 in a 3-bit integer.
    EXPECT_EQ(5, APNThree);
    EXPECT_EQ(3u, APNThree.getBitWidth());
  }

  {
    auto *IslNFour = isl_val_int_from_si(IslCtx, -4);
    auto APNFour = APIntFromVal(IslNFour);
    // Compare with the two's complement of -4 in a 3-bit integer.
    EXPECT_EQ(4, APNFour);
    EXPECT_EQ(3u, APNFour.getBitWidth());
  }

  {
    auto *IslZero = isl_val_int_from_ui(IslCtx, 0);
    auto APZero = APIntFromVal(IslZero);
    EXPECT_EQ(0, APZero);
    EXPECT_EQ(1u, APZero.getBitWidth());
  }

  {
    auto *IslOne = isl_val_int_from_ui(IslCtx, 1);
    auto APOne = APIntFromVal(IslOne);
    EXPECT_EQ(1, APOne);
    EXPECT_EQ(2u, APOne.getBitWidth());
  }

  {
    auto *IslTwo = isl_val_int_from_ui(IslCtx, 2);
    auto APTwo = APIntFromVal(IslTwo);
    EXPECT_EQ(2, APTwo);
    EXPECT_EQ(3u, APTwo.getBitWidth());
  }

  {
    auto *IslThree = isl_val_int_from_ui(IslCtx, 3);
    auto APThree = APIntFromVal(IslThree);
    EXPECT_EQ(3, APThree);
    EXPECT_EQ(3u, APThree.getBitWidth());
  }

  {
    auto *IslFour = isl_val_int_from_ui(IslCtx, 4);
    auto APFour = APIntFromVal(IslFour);
    EXPECT_EQ(4, APFour);
    EXPECT_EQ(4u, APFour.getBitWidth());
  }

  {
    auto *IslNOne = isl_val_int_from_ui(IslCtx, (1ull << 32) - 1);
    auto APNOne = APIntFromVal(IslNOne);
    EXPECT_EQ((1ull << 32) - 1, APNOne);
    EXPECT_EQ(33u, APNOne.getBitWidth());
  }

  {
    auto *IslLargeNum = isl_val_int_from_ui(IslCtx, 60);
    IslLargeNum = isl_val_2exp(IslLargeNum);
    IslLargeNum = isl_val_sub_ui(IslLargeNum, 1);
    auto APLargeNum = APIntFromVal(IslLargeNum);
    EXPECT_EQ((1ull << 60) - 1, APLargeNum);
    EXPECT_EQ(61u, APLargeNum.getBitWidth());
  }

  {
    auto *IslExp = isl_val_int_from_ui(IslCtx, 500);
    auto *IslLargePow2 = isl_val_2exp(IslExp);
    auto APLargePow2 = APIntFromVal(IslLargePow2);
    EXPECT_TRUE(APLargePow2.isPowerOf2());
    EXPECT_EQ(502u, APLargePow2.getBitWidth());
    EXPECT_EQ(502u, APLargePow2.getMinSignedBits());
  }

  {
    auto *IslExp = isl_val_int_from_ui(IslCtx, 500);
    auto *IslLargeNPow2 = isl_val_neg(isl_val_2exp(IslExp));
    auto APLargeNPow2 = APIntFromVal(IslLargeNPow2);
    EXPECT_EQ(501u, APLargeNPow2.getBitWidth());
    EXPECT_EQ(501u, APLargeNPow2.getMinSignedBits());
    EXPECT_EQ(500, (-APLargeNPow2).exactLogBase2());
  }

  {
    auto *IslExp = isl_val_int_from_ui(IslCtx, 512);
    auto *IslLargePow2 = isl_val_2exp(IslExp);
    auto APLargePow2 = APIntFromVal(IslLargePow2);
    EXPECT_TRUE(APLargePow2.isPowerOf2());
    EXPECT_EQ(514u, APLargePow2.getBitWidth());
    EXPECT_EQ(514u, APLargePow2.getMinSignedBits());
  }

  {
    auto *IslExp = isl_val_int_from_ui(IslCtx, 512);
    auto *IslLargeNPow2 = isl_val_neg(isl_val_2exp(IslExp));
    auto APLargeNPow2 = APIntFromVal(IslLargeNPow2);
    EXPECT_EQ(513u, APLargeNPow2.getBitWidth());
    EXPECT_EQ(513u, APLargeNPow2.getMinSignedBits());
    EXPECT_EQ(512, (-APLargeNPow2).exactLogBase2());
  }

  isl_ctx_free(IslCtx);
}

TEST(Isl, Foreach) {
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> Ctx(isl_ctx_alloc(),
                                                        &isl_ctx_free);

  auto MapSpace = give(isl_space_alloc(Ctx.get(), 0, 1, 1));
  auto TestBMap = give(isl_basic_map_universe(MapSpace.copy()));
  TestBMap = give(isl_basic_map_fix_si(TestBMap.take(), isl_dim_in, 0, 0));
  TestBMap = give(isl_basic_map_fix_si(TestBMap.take(), isl_dim_out, 0, 0));
  auto TestMap = give(isl_map_from_basic_map(TestBMap.copy()));
  auto TestUMap = give(isl_union_map_from_map(TestMap.copy()));

  auto SetSpace = give(isl_space_set_alloc(Ctx.get(), 0, 1));
  auto TestBSet =
      give(isl_basic_set_from_point(isl_point_zero(SetSpace.copy())));
  auto TestSet = give(isl_set_from_basic_set(TestBSet.copy()));
  auto TestUSet = give(isl_union_set_from_set(TestSet.copy()));

  {
    auto NumBMaps = 0;
    foreachElt(TestMap, [&](IslPtr<isl_basic_map> BMap) {
      EXPECT_EQ(isl_bool_true,
                isl_basic_map_is_equal(BMap.keep(), TestBMap.keep()));
      NumBMaps++;
    });
    EXPECT_EQ(1, NumBMaps);
  }

  {
    auto NumBSets = 0;
    foreachElt(TestSet, [&](IslPtr<isl_basic_set> BSet) {
      EXPECT_EQ(isl_bool_true,
                isl_basic_set_is_equal(BSet.keep(), TestBSet.keep()));
      NumBSets++;
    });
    EXPECT_EQ(1, NumBSets);
  }

  {
    auto NumMaps = 0;
    foreachElt(TestUMap, [&](IslPtr<isl_map> Map) {
      EXPECT_EQ(isl_bool_true, isl_map_is_equal(Map.keep(), TestMap.keep()));
      NumMaps++;
    });
    EXPECT_EQ(1, NumMaps);
  }

  {
    auto NumSets = 0;
    foreachElt(TestUSet, [&](IslPtr<isl_set> Set) {
      EXPECT_EQ(isl_bool_true, isl_set_is_equal(Set.keep(), TestSet.keep()));
      NumSets++;
    });
    EXPECT_EQ(1, NumSets);
  }

  {
    auto UPwAff = give(isl_union_pw_aff_val_on_domain(TestUSet.copy(),
                                                      isl_val_zero(Ctx.get())));
    auto NumPwAffs = 0;
    foreachElt(UPwAff, [&](IslPtr<isl_pw_aff> PwAff) {
      EXPECT_EQ(isl_bool_true, isl_pw_aff_is_cst(PwAff.keep()));
      NumPwAffs++;
    });
    EXPECT_EQ(1, NumPwAffs);
  }

  {
    auto NumBMaps = 0;
    EXPECT_EQ(isl_stat_error,
              foreachEltWithBreak(
                  TestMap, [&](IslPtr<isl_basic_map> BMap) -> isl_stat {
                    EXPECT_EQ(isl_bool_true, isl_basic_map_is_equal(
                                                 BMap.keep(), TestBMap.keep()));
                    NumBMaps++;
                    return isl_stat_error;
                  }));
    EXPECT_EQ(1, NumBMaps);
  }

  {
    auto NumMaps = 0;
    EXPECT_EQ(
        isl_stat_error,
        foreachEltWithBreak(TestUMap, [&](IslPtr<isl_map> Map) -> isl_stat {
          EXPECT_EQ(isl_bool_true,
                    isl_map_is_equal(Map.keep(), TestMap.keep()));
          NumMaps++;
          return isl_stat_error;
        }));
    EXPECT_EQ(1, NumMaps);
  }

  {
    auto TestPwAff =
        give(isl_pw_aff_val_on_domain(TestSet.copy(), isl_val_zero(Ctx.get())));
    auto NumPieces = 0;
    foreachPieceWithBreak(
        TestPwAff,
        [&](IslPtr<isl_set> Domain, IslPtr<isl_aff> Aff) -> isl_stat {
          EXPECT_EQ(isl_bool_true,
                    isl_set_is_equal(Domain.keep(), TestSet.keep()));
          EXPECT_EQ(isl_bool_true, isl_aff_is_cst(Aff.keep()));
          NumPieces++;
          return isl_stat_error;
        });
    EXPECT_EQ(1, NumPieces);
  }
}

TEST(ISLTools, beforeScatter) {
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> Ctx(isl_ctx_alloc(),
                                                        &isl_ctx_free);

  // Basic usage with isl_map
  EXPECT_EQ(MAP("{ [] -> [i] : i <= 0 }"),
            beforeScatter(MAP("{ [] -> [0] }"), false));
  EXPECT_EQ(MAP("{ [] -> [i] : i < 0 }"),
            beforeScatter(MAP("{ [] -> [0] }"), true));

  // Basic usage with isl_union_map
  EXPECT_EQ(UMAP("{ A[] -> [i] : i <= 0; B[] -> [i] : i <= 0 }"),
            beforeScatter(UMAP("{ A[] -> [0]; B[] -> [0] }"), false));
  EXPECT_EQ(UMAP("{ A[] -> [i] : i < 0; B[] -> [i] : i < 0 }"),
            beforeScatter(UMAP("{ A[] -> [0]; B[] -> [0] }"), true));

  // More than one dimension
  EXPECT_EQ(UMAP("{ [] -> [i, j] : i < 0;  [] -> [i, j] : i = 0 and j <= 0 }"),
            beforeScatter(UMAP("{ [] -> [0, 0] }"), false));
  EXPECT_EQ(UMAP("{ [] -> [i, j] : i < 0;  [] -> [i, j] : i = 0 and j < 0 }"),
            beforeScatter(UMAP("{ [] -> [0, 0] }"), true));

  // Functional
  EXPECT_EQ(UMAP("{ [i] -> [j] : j <= i }"),
            beforeScatter(UMAP("{ [i] -> [i] }"), false));
  EXPECT_EQ(UMAP("{ [i] -> [j] : j < i }"),
            beforeScatter(UMAP("{ [i] -> [i] }"), true));

  // Parametrized
  EXPECT_EQ(UMAP("[i] -> { [] -> [j] : j <= i }"),
            beforeScatter(UMAP("[i] -> { [] -> [i] }"), false));
  EXPECT_EQ(UMAP("[i] -> { [] -> [j] : j < i }"),
            beforeScatter(UMAP("[i] -> { [] -> [i] }"), true));

  // More than one range
  EXPECT_EQ(UMAP("{ [] -> [i] : i <= 10 }"),
            beforeScatter(UMAP("{ [] -> [0]; [] -> [10] }"), false));
  EXPECT_EQ(UMAP("{ [] -> [i] : i < 10 }"),
            beforeScatter(UMAP("{ [] -> [0]; [] -> [10] }"), true));

  // Edge case: empty
  EXPECT_EQ(UMAP("{ [] -> [i] : 1 = 0 }"),
            beforeScatter(UMAP("{ [] -> [i] : 1 = 0 }"), false));
  EXPECT_EQ(UMAP("{ [] -> [i] : 1 = 0 }"),
            beforeScatter(UMAP("{ [] -> [i] : 1 = 0 }"), true));
}

TEST(ISLTools, afterScatter) {
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> Ctx(isl_ctx_alloc(),
                                                        &isl_ctx_free);

  // Basic usage with isl_map
  EXPECT_EQ(MAP("{ [] -> [i] : i >= 0 }"),
            afterScatter(MAP("{ [] -> [0] }"), false));
  EXPECT_EQ(MAP("{ [] -> [i] : i > 0 }"),
            afterScatter(MAP("{ [] -> [0] }"), true));

  // Basic usage with isl_union_map
  EXPECT_EQ(UMAP("{ A[] -> [i] : i >= 0; B[] -> [i] : i >= 0 }"),
            afterScatter(UMAP("{ A[] -> [0]; B[] -> [0] }"), false));
  EXPECT_EQ(UMAP("{ A[] -> [i] : i > 0; B[] -> [i] : i > 0 }"),
            afterScatter(UMAP("{ A[] -> [0]; B[] -> [0] }"), true));

  // More than one dimension
  EXPECT_EQ(UMAP("{ [] -> [i, j] : i > 0;  [] -> [i, j] : i = 0 and j >= 0 }"),
            afterScatter(UMAP("{ [] -> [0, 0] }"), false));
  EXPECT_EQ(UMAP("{ [] -> [i, j] : i > 0;  [] -> [i, j] : i = 0 and j > 0 }"),
            afterScatter(UMAP("{ [] -> [0, 0] }"), true));

  // Functional
  EXPECT_EQ(UMAP("{ [i] -> [j] : j >= i }"),
            afterScatter(UMAP("{ [i] -> [i] }"), false));
  EXPECT_EQ(UMAP("{ [i] -> [j] : j > i }"),
            afterScatter(UMAP("{ [i] -> [i] }"), true));

  // Parametrized
  EXPECT_EQ(UMAP("[i] -> { [] -> [j] : j >= i }"),
            afterScatter(UMAP("[i] -> { [] -> [i] }"), false));
  EXPECT_EQ(UMAP("[i] -> { [] -> [j] : j > i }"),
            afterScatter(UMAP("[i] -> { [] -> [i] }"), true));

  // More than one range
  EXPECT_EQ(UMAP("{ [] -> [i] : i >= 0 }"),
            afterScatter(UMAP("{ [] -> [0]; [] -> [10] }"), false));
  EXPECT_EQ(UMAP("{ [] -> [i] : i > 0 }"),
            afterScatter(UMAP("{ [] -> [0]; [] -> [10] }"), true));

  // Edge case: empty
  EXPECT_EQ(UMAP("{ }"), afterScatter(UMAP("{ }"), false));
  EXPECT_EQ(UMAP("{ }"), afterScatter(UMAP("{ }"), true));
}

TEST(ISLTools, betweenScatter) {
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> Ctx(isl_ctx_alloc(),
                                                        &isl_ctx_free);

  // Basic usage with isl_map
  EXPECT_EQ(MAP("{ [] -> [i] : 0 < i < 10 }"),
            betweenScatter(MAP("{ [] -> [0] }"), MAP("{ [] -> [10] }"), false,
                           false));
  EXPECT_EQ(
      MAP("{ [] -> [i] : 0 <= i < 10 }"),
      betweenScatter(MAP("{ [] -> [0] }"), MAP("{ [] -> [10] }"), true, false));
  EXPECT_EQ(
      MAP("{ [] -> [i] : 0 < i <= 10 }"),
      betweenScatter(MAP("{ [] -> [0] }"), MAP("{ [] -> [10] }"), false, true));
  EXPECT_EQ(
      MAP("{ [] -> [i] : 0 <= i <= 10 }"),
      betweenScatter(MAP("{ [] -> [0] }"), MAP("{ [] -> [10] }"), true, true));

  // Basic usage with isl_union_map
  EXPECT_EQ(UMAP("{ A[] -> [i] : 0 < i < 10; B[] -> [i] : 0 < i < 10 }"),
            betweenScatter(UMAP("{ A[] -> [0]; B[] -> [0] }"),
                           UMAP("{ A[] -> [10]; B[] -> [10] }"), false, false));
  EXPECT_EQ(UMAP("{ A[] -> [i] : 0 <= i < 10; B[] -> [i] : 0 <= i < 10 }"),
            betweenScatter(UMAP("{ A[] -> [0]; B[] -> [0] }"),
                           UMAP("{ A[] -> [10]; B[] -> [10] }"), true, false));
  EXPECT_EQ(UMAP("{ A[] -> [i] : 0 < i <= 10; B[] -> [i] : 0 < i <= 10 }"),
            betweenScatter(UMAP("{ A[] -> [0]; B[] -> [0] }"),
                           UMAP("{ A[] -> [10]; B[] -> [10] }"), false, true));
  EXPECT_EQ(UMAP("{ A[] -> [i] : 0 <= i <= 10; B[] -> [i] : 0 <= i <= 10 }"),
            betweenScatter(UMAP("{ A[] -> [0]; B[] -> [0] }"),
                           UMAP("{ A[] -> [10]; B[] -> [10] }"), true, true));
}

TEST(ISLTools, singleton) {
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> Ctx(isl_ctx_alloc(),
                                                        &isl_ctx_free);

  // No element found
  EXPECT_EQ(SET("{ [] : 1 = 0 }"), singleton(USET("{ }"), SPACE("{ [] }")));
  EXPECT_EQ(MAP("{ [] -> [] : 1 = 0  }"),
            singleton(UMAP("{  }"), SPACE("{ [] -> [] }")));

  // One element found
  EXPECT_EQ(SET("{ [] }"), singleton(USET("{ [] }"), SPACE("{ [] }")));
  EXPECT_EQ(MAP("{ [] -> [] }"),
            singleton(UMAP("{ [] -> [] }"), SPACE("{ [] -> [] }")));

  // Many elements found
  EXPECT_EQ(SET("{ [i] : 0 <= i < 10 }"),
            singleton(USET("{ [i] : 0 <= i < 10 }"), SPACE("{ [i] }")));
  EXPECT_EQ(
      MAP("{ [i] -> [i] : 0 <= i < 10 }"),
      singleton(UMAP("{ [i] -> [i] : 0 <= i < 10 }"), SPACE("{ [i] -> [j] }")));

  // Different parameters
  EXPECT_EQ(SET("[i] -> { [i] }"),
            singleton(USET("[i] -> { [i] }"), SPACE("{ [i] }")));
  EXPECT_EQ(MAP("[i] -> { [i] -> [i] }"),
            singleton(UMAP("[i] -> { [i] -> [i] }"), SPACE("{ [i] -> [j] }")));
}

TEST(ISLTools, getNumScatterDims) {
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> Ctx(isl_ctx_alloc(),
                                                        &isl_ctx_free);

  // Basic usage
  EXPECT_EQ(0u, getNumScatterDims(UMAP("{ [] -> [] }")));
  EXPECT_EQ(1u, getNumScatterDims(UMAP("{ [] -> [i] }")));
  EXPECT_EQ(2u, getNumScatterDims(UMAP("{ [] -> [i,j] }")));
  EXPECT_EQ(3u, getNumScatterDims(UMAP("{ [] -> [i,j,k] }")));

  // Different scatter spaces
  EXPECT_EQ(0u, getNumScatterDims(UMAP("{ A[] -> []; [] -> []}")));
  EXPECT_EQ(1u, getNumScatterDims(UMAP("{ A[] -> []; [] -> [i] }")));
  EXPECT_EQ(2u, getNumScatterDims(UMAP("{ A[] -> [i]; [] -> [i,j] }")));
  EXPECT_EQ(3u, getNumScatterDims(UMAP("{ A[] -> [i]; [] -> [i,j,k] }")));
}

TEST(ISLTools, getScatterSpace) {
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> Ctx(isl_ctx_alloc(),
                                                        &isl_ctx_free);

  // Basic usage
  EXPECT_EQ(SPACE("{ [] }"), getScatterSpace(UMAP("{ [] -> [] }")));
  EXPECT_EQ(SPACE("{ [i] }"), getScatterSpace(UMAP("{ [] -> [i] }")));
  EXPECT_EQ(SPACE("{ [i,j] }"), getScatterSpace(UMAP("{ [] -> [i,j] }")));
  EXPECT_EQ(SPACE("{ [i,j,k] }"), getScatterSpace(UMAP("{ [] -> [i,j,k] }")));

  // Different scatter spaces
  EXPECT_EQ(SPACE("{ [] }"), getScatterSpace(UMAP("{ A[] -> []; [] -> [] }")));
  EXPECT_EQ(SPACE("{ [i] }"),
            getScatterSpace(UMAP("{ A[] -> []; [] -> [i] }")));
  EXPECT_EQ(SPACE("{ [i,j] }"),
            getScatterSpace(UMAP("{ A[] -> [i]; [] -> [i,j] }")));
  EXPECT_EQ(SPACE("{ [i,j,k] }"),
            getScatterSpace(UMAP("{ A[] -> [i]; [] -> [i,j,k] }")));
}

TEST(ISLTools, makeIdentityMap) {
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> Ctx(isl_ctx_alloc(),
                                                        &isl_ctx_free);

  // Basic usage
  EXPECT_EQ(UMAP("{ [i] -> [i] }"), makeIdentityMap(USET("{ [0] }"), false));
  EXPECT_EQ(UMAP("{ [0] -> [0] }"), makeIdentityMap(USET("{ [0] }"), true));

  // Multiple spaces
  EXPECT_EQ(UMAP("{ [] -> []; [i] -> [i] }"),
            makeIdentityMap(USET("{ []; [0] }"), false));
  EXPECT_EQ(UMAP("{ [] -> []; [0] -> [0] }"),
            makeIdentityMap(USET("{ []; [0] }"), true));

  // Edge case: empty
  EXPECT_EQ(UMAP("{ }"), makeIdentityMap(USET("{ }"), false));
  EXPECT_EQ(UMAP("{ }"), makeIdentityMap(USET("{ }"), true));
}

TEST(ISLTools, reverseDomain) {
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> Ctx(isl_ctx_alloc(),
                                                        &isl_ctx_free);

  // Basic usage
  EXPECT_EQ(MAP("{ [B[] -> A[]] -> [] }"),
            reverseDomain(MAP("{ [A[] -> B[]] -> [] }")));
  EXPECT_EQ(UMAP("{ [B[] -> A[]] -> [] }"),
            reverseDomain(UMAP("{ [A[] -> B[]] -> [] }")));
}

TEST(ISLTools, shiftDim) {
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> Ctx(isl_ctx_alloc(),
                                                        &isl_ctx_free);

  // Basic usage
  EXPECT_EQ(SET("{ [1] }"), shiftDim(SET("{ [0] }"), 0, 1));
  EXPECT_EQ(USET("{ [1] }"), shiftDim(USET("{ [0] }"), 0, 1));

  // From-end indexing
  EXPECT_EQ(USET("{ [0,0,1] }"), shiftDim(USET("{ [0,0,0] }"), -1, 1));
  EXPECT_EQ(USET("{ [0,1,0] }"), shiftDim(USET("{ [0,0,0] }"), -2, 1));
  EXPECT_EQ(USET("{ [1,0,0] }"), shiftDim(USET("{ [0,0,0] }"), -3, 1));

  // Parametrized
  EXPECT_EQ(USET("[n] -> { [n+1] }"), shiftDim(USET("[n] -> { [n] }"), 0, 1));
}

TEST(DeLICM, computeReachingWrite) {
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> Ctx(isl_ctx_alloc(),
                                                        &isl_ctx_free);

  // Basic usage
  EXPECT_EQ(UMAP("{ [Elt[] -> [i]] -> Dom[] : 0 < i }"),
            computeReachingWrite(UMAP("{ Dom[] -> [0] }"),
                                 UMAP("{ Dom[] -> Elt[] }"), false, false,
                                 false));
  EXPECT_EQ(UMAP("{ [Elt[] -> [i]] -> Dom[] : 0 < i }"),
            computeReachingWrite(UMAP("{ Dom[] -> [0] }"),
                                 UMAP("{ Dom[] -> Elt[] }"), false, false,
                                 true));
  EXPECT_EQ(UMAP("{ [Elt[] -> [i]] -> Dom[] : 0 <= i }"),
            computeReachingWrite(UMAP("{ Dom[] -> [0] }"),
                                 UMAP("{ Dom[] -> Elt[] }"), false, true,
                                 false));
  EXPECT_EQ(UMAP("{ [Elt[] -> [i]] -> Dom[] : 0 <= i }"),
            computeReachingWrite(UMAP("{ Dom[] -> [0] }"),
                                 UMAP("{ Dom[] -> Elt[] }"), false, true,
                                 false));

  EXPECT_EQ(UMAP("{ [Elt[] -> [i]] -> Dom[] : i < 0 }"),
            computeReachingWrite(UMAP("{ Dom[] -> [0] }"),
                                 UMAP("{ Dom[] -> Elt[] }"), true, false,
                                 false));
  EXPECT_EQ(UMAP("{ [Elt[] -> [i]] -> Dom[] :  i <= 0 }"),
            computeReachingWrite(UMAP("{ Dom[] -> [0] }"),
                                 UMAP("{ Dom[] -> Elt[] }"), true, false,
                                 true));
  EXPECT_EQ(UMAP("{ [Elt[] -> [i]] -> Dom[] : i < 0 }"),
            computeReachingWrite(UMAP("{ Dom[] -> [0] }"),
                                 UMAP("{ Dom[] -> Elt[] }"), true, true,
                                 false));
  EXPECT_EQ(UMAP("{ [Elt[] -> [i]] -> Dom[] : i <= 0 }"),
            computeReachingWrite(UMAP("{ Dom[] -> [0] }"),
                                 UMAP("{ Dom[] -> Elt[] }"), true, true, true));

  // Two writes
  EXPECT_EQ(UMAP("{ [Elt[] -> [i]] -> Dom1[] : 0 < i < 10; [Elt[] -> [i]] -> "
                 "Dom2[] : 10 < i }"),
            computeReachingWrite(UMAP("{ Dom1[] -> [0]; Dom2[] -> [10] }"),
                                 UMAP("{ Dom1[] -> Elt[]; Dom2[] -> Elt[] }"),
                                 false, false, false));
  EXPECT_EQ(UMAP("{ [Elt[] -> [i]] -> Dom1[] : 0 <= i < 10; [Elt[] -> [i]] -> "
                 "Dom2[] : 10 <= i }"),
            computeReachingWrite(UMAP("{ Dom1[] -> [0]; Dom2[] -> [10] }"),
                                 UMAP("{ Dom1[] -> Elt[]; Dom2[] -> Elt[] }"),
                                 false, true, false));
  EXPECT_EQ(UMAP("{ [Elt[] -> [i]] -> Dom1[] : 0 < i <= 10; [Elt[] -> [i]] -> "
                 "Dom2[] : 10 < i }"),
            computeReachingWrite(UMAP("{ Dom1[] -> [0]; Dom2[] -> [10] }"),
                                 UMAP("{ Dom1[] -> Elt[]; Dom2[] -> Elt[] }"),
                                 false, false, true));
  EXPECT_EQ(UMAP("{ [Elt[] -> [i]] -> Dom1[] : 0 <= i <= 10; [Elt[] -> [i]] -> "
                 "Dom2[] : 10 <= i }"),
            computeReachingWrite(UMAP("{ Dom1[] -> [0]; Dom2[] -> [10] }"),
                                 UMAP("{ Dom1[] -> Elt[]; Dom2[] -> Elt[] }"),
                                 false, true, true));

  EXPECT_EQ(UMAP("{ [Elt[] -> [i]] -> Dom2[] : 0 < i < 10; [Elt[] -> [i]] -> "
                 "Dom1[] : i < 0 }"),
            computeReachingWrite(UMAP("{ Dom1[] -> [0]; Dom2[] -> [10] }"),
                                 UMAP("{ Dom1[] -> Elt[]; Dom2[] -> Elt[] }"),
                                 true, false, false));
  EXPECT_EQ(UMAP("{ [Elt[] -> [i]] -> Dom2[] : 0 <= i < 10; [Elt[] -> [i]] -> "
                 "Dom1[] : i < 0 }"),
            computeReachingWrite(UMAP("{ Dom1[] -> [0]; Dom2[] -> [10] }"),
                                 UMAP("{ Dom1[] -> Elt[]; Dom2[] -> Elt[] }"),
                                 true, true, false));
  EXPECT_EQ(UMAP("{ [Elt[] -> [i]] -> Dom2[] : 0 < i <= 10; [Elt[] -> [i]] -> "
                 "Dom1[] : i <= 0 }"),
            computeReachingWrite(UMAP("{ Dom1[] -> [0]; Dom2[] -> [10] }"),
                                 UMAP("{ Dom1[] -> Elt[]; Dom2[] -> Elt[] }"),
                                 true, false, true));
  EXPECT_EQ(UMAP("{ [Elt[] -> [i]] -> Dom2[] : 0 <= i <= 10; [Elt[] -> [i]] -> "
                 "Dom1[] : i <= 0 }"),
            computeReachingWrite(UMAP("{ Dom1[] -> [0]; Dom2[] -> [10] }"),
                                 UMAP("{ Dom1[] -> Elt[]; Dom2[] -> Elt[] }"),
                                 true, true, true));

  // Domain in same space
  EXPECT_EQ(UMAP("{ [Elt[] -> [i]] -> Dom[1] : 0 < i <= 10; [Elt[] -> [i]] -> "
                 "Dom[2] : 10 < i }"),
            computeReachingWrite(UMAP("{ Dom[i] -> [10i - 10] }"),
                                 UMAP("{ Dom[1] -> Elt[]; Dom[2] -> Elt[] }"),
                                 false, false, true));

  // Parametric
  EXPECT_EQ(UMAP("[p] -> { [Elt[] -> [i]] -> Dom[] : p < i }"),
            computeReachingWrite(UMAP("[p] -> { Dom[] -> [p] }"),
                                 UMAP("{ Dom[] -> Elt[] }"), false, false,
                                 false));

  // More realistic example (from reduction_embedded.ll)
  EXPECT_EQ(
      UMAP("{ [Elt[] -> [i]] -> Dom[0] : 0 < i <= 3; [Elt[] -> [i]] -> Dom[1] "
           ": 3 < i <= 6; [Elt[] -> [i]] -> Dom[2] : 6 < i <= 9; [Elt[] -> "
           "[i]] -> Dom[3] : 9 < i <= 12; [Elt[] -> [i]] -> Dom[4] : 12 < i }"),
      computeReachingWrite(UMAP("{ Dom[i] -> [3i] : 0 <= i <= 4 }"),
                           UMAP("{ Dom[i] -> Elt[] : 0 <= i <= 4 }"), false,
                           false, true));
}

TEST(DeLICM, computeArrayUnused) {
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> Ctx(isl_ctx_alloc(),
                                                        &isl_ctx_free);

  // The ReadEltInSameInst parameter doesn't matter in simple cases. To also
  // cover the parameter without duplicating the tests, this loops runs over
  // other in both settings.
  for (bool ReadEltInSameInst = false, Done = false; !Done;
       Done = ReadEltInSameInst, ReadEltInSameInst = true) {
    // Basic usage: one read, one write
    EXPECT_EQ(UMAP("{ Elt[] -> [i] : 0 < i < 10 }"),
              computeArrayUnused(UMAP("{ Read[] -> [0]; Write[] -> [10] }"),
                                 UMAP("{ Write[] -> Elt[] }"),
                                 UMAP("{ Read[] -> Elt[] }"), ReadEltInSameInst,
                                 false, false));
    EXPECT_EQ(UMAP("{ Elt[] -> [i] : 0 < i <= 10 }"),
              computeArrayUnused(UMAP("{ Read[] -> [0]; Write[] -> [10] }"),
                                 UMAP("{ Write[] -> Elt[] }"),
                                 UMAP("{ Read[] -> Elt[] }"), ReadEltInSameInst,
                                 false, true));
    EXPECT_EQ(UMAP("{ Elt[] -> [i] : 0 <= i < 10 }"),
              computeArrayUnused(UMAP("{ Read[] -> [0]; Write[] -> [10] }"),
                                 UMAP("{ Write[] -> Elt[] }"),
                                 UMAP("{ Read[] -> Elt[] }"), ReadEltInSameInst,
                                 true, false));
    EXPECT_EQ(UMAP("{ Elt[] -> [i] : 0 <= i <= 10 }"),
              computeArrayUnused(UMAP("{ Read[] -> [0]; Write[] -> [10] }"),
                                 UMAP("{ Write[] -> Elt[] }"),
                                 UMAP("{ Read[] -> Elt[] }"), ReadEltInSameInst,
                                 true, true));

    // Two reads
    EXPECT_EQ(UMAP("{ Elt[] -> [i] : 0 < i <= 10 }"),
              computeArrayUnused(
                  UMAP("{ Read[0] -> [-10]; Read[1] -> [0]; Write[] -> [10] }"),
                  UMAP("{ Write[] -> Elt[] }"), UMAP("{ Read[i] -> Elt[] }"),
                  ReadEltInSameInst, false, true));

    // Corner case: no writes
    EXPECT_EQ(UMAP("{}"),
              computeArrayUnused(UMAP("{ Read[] -> [0] }"), UMAP("{}"),
                                 UMAP("{ Read[] -> Elt[] }"), ReadEltInSameInst,
                                 false, false));

    // Corner case: no reads
    EXPECT_EQ(UMAP("{ Elt[] -> [i] : i <= 0 }"),
              computeArrayUnused(UMAP("{ Write[] -> [0] }"),
                                 UMAP("{ Write[] -> Elt[] }"), UMAP("{}"),
                                 ReadEltInSameInst, false, true));
  }

  // Read and write in same statement
  EXPECT_EQ(UMAP("{ Elt[] -> [i] : i < 0 }"),
            computeArrayUnused(UMAP("{ RW[] -> [0] }"),
                               UMAP("{ RW[] -> Elt[] }"),
                               UMAP("{ RW[] -> Elt[] }"), true, false, false));
  EXPECT_EQ(UMAP("{ Elt[] -> [i] : i <= 0 }"),
            computeArrayUnused(UMAP("{ RW[] -> [0] }"),
                               UMAP("{ RW[] -> Elt[] }"),
                               UMAP("{ RW[] -> Elt[] }"), true, false, true));
  EXPECT_EQ(UMAP("{ Elt[] -> [0] }"),
            computeArrayUnused(UMAP("{ RW[] -> [0] }"),
                               UMAP("{ RW[] -> Elt[] }"),
                               UMAP("{ RW[] -> Elt[] }"), false, true, true));
}

TEST(DeLICM, convertZoneToTimepoints) {
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> Ctx(isl_ctx_alloc(),
                                                        &isl_ctx_free);

  // Corner case: empty set
  EXPECT_EQ(USET("{}"), convertZoneToTimepoints(USET("{}"), false, false));
  EXPECT_EQ(USET("{}"), convertZoneToTimepoints(USET("{}"), true, false));
  EXPECT_EQ(USET("{}"), convertZoneToTimepoints(USET("{}"), false, true));
  EXPECT_EQ(USET("{}"), convertZoneToTimepoints(USET("{}"), true, true));

  // Basic usage
  EXPECT_EQ(USET("{}"), convertZoneToTimepoints(USET("{ [1] }"), false, false));
  EXPECT_EQ(USET("{ [0] }"),
            convertZoneToTimepoints(USET("{ [1] }"), true, false));
  EXPECT_EQ(USET("{ [1] }"),
            convertZoneToTimepoints(USET("{ [1] }"), false, true));
  EXPECT_EQ(USET("{ [0]; [1] }"),
            convertZoneToTimepoints(USET("{ [1] }"), true, true));

  // Non-adjacent ranges
  EXPECT_EQ(USET("{}"),
            convertZoneToTimepoints(USET("{ [1]; [11] }"), false, false));
  EXPECT_EQ(USET("{ [0]; [10] }"),
            convertZoneToTimepoints(USET("{ [1]; [11] }"), true, false));
  EXPECT_EQ(USET("{ [1]; [11] }"),
            convertZoneToTimepoints(USET("{ [1]; [11] }"), false, true));
  EXPECT_EQ(USET("{ [0]; [1]; [10]; [11] }"),
            convertZoneToTimepoints(USET("{ [1]; [11] }"), true, true));

  // Adjacent unit ranges
  EXPECT_EQ(
      USET("{ [i] : 0 < i < 10 }"),
      convertZoneToTimepoints(USET("{ [i] : 0 < i <= 10 }"), false, false));
  EXPECT_EQ(
      USET("{ [i] : 0 <= i < 10 }"),
      convertZoneToTimepoints(USET("{ [i] : 0 < i <= 10 }"), true, false));
  EXPECT_EQ(
      USET("{ [i] : 0 < i <= 10 }"),
      convertZoneToTimepoints(USET("{ [i] : 0 < i <= 10 }"), false, true));
  EXPECT_EQ(USET("{ [i] : 0 <= i <= 10 }"),
            convertZoneToTimepoints(USET("{ [i] : 0 < i <= 10 }"), true, true));

  // More than one dimension
  EXPECT_EQ(USET("{}"),
            convertZoneToTimepoints(USET("{ [0,1] }"), false, false));
  EXPECT_EQ(USET("{ [0,0] }"),
            convertZoneToTimepoints(USET("{ [0,1] }"), true, false));
  EXPECT_EQ(USET("{ [0,1] }"),
            convertZoneToTimepoints(USET("{ [0,1] }"), false, true));
  EXPECT_EQ(USET("{ [0,0]; [0,1] }"),
            convertZoneToTimepoints(USET("{ [0,1] }"), true, true));
}

} // anonymous namespace
