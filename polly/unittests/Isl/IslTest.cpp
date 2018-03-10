//===- IslTest.cpp ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "polly/Support/GICHelper.h"
#include "polly/Support/ISLOperators.h"
#include "polly/Support/ISLTools.h"
#include "gtest/gtest.h"
#include "isl/stream.h"
#include "isl/val.h"

using namespace llvm;
using namespace polly;

static isl::space parseSpace(isl_ctx *Ctx, const char *Str) {
  isl_stream *Stream = isl_stream_new_str(Ctx, Str);
  auto Obj = isl_stream_read_obj(Stream);

  isl::space Result;
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

#define SET(Str) isl::set(Ctx.get(), Str)
#define MAP(Str) isl::map(Ctx.get(), Str)

#define USET(Str) isl::union_set(Ctx.get(), Str)
#define UMAP(Str) isl::union_map(Ctx.get(), Str)

namespace isl {
inline namespace noexceptions {

static bool operator==(const isl::space &LHS, const isl::space &RHS) {
  return bool(LHS.is_equal(RHS));
}

static bool operator==(const isl::basic_set &LHS, const isl::basic_set &RHS) {
  return bool(LHS.is_equal(RHS));
}

static bool operator==(const isl::set &LHS, const isl::set &RHS) {
  return bool(LHS.is_equal(RHS));
}

static bool operator==(const isl::basic_map &LHS, const isl::basic_map &RHS) {
  return bool(LHS.is_equal(RHS));
}

static bool operator==(const isl::map &LHS, const isl::map &RHS) {
  return bool(LHS.is_equal(RHS));
}

static bool operator==(const isl::union_set &LHS, const isl::union_set &RHS) {
  return bool(LHS.is_equal(RHS));
}

static bool operator==(const isl::union_map &LHS, const isl::union_map &RHS) {
  return bool(LHS.is_equal(RHS));
}

static bool operator==(const isl::val &LHS, const isl::val &RHS) {
  return bool(LHS.eq(RHS));
}

static bool operator==(const isl::pw_aff &LHS, const isl::pw_aff &RHS) {
  return bool(LHS.is_equal(RHS));
}
} // namespace noexceptions
} // namespace isl

namespace {

TEST(Isl, APIntToIslVal) {
  isl_ctx *IslCtx = isl_ctx_alloc();

  {
    APInt APZero(1, 0, true);
    auto IslZero = valFromAPInt(IslCtx, APZero, true);
    EXPECT_TRUE(IslZero.is_zero());
  }

  {
    APInt APNOne(1, -1, true);
    auto IslNOne = valFromAPInt(IslCtx, APNOne, true);
    EXPECT_TRUE(IslNOne.is_negone());
  }

  {
    APInt APZero(1, 0, false);
    auto IslZero = valFromAPInt(IslCtx, APZero, false);
    EXPECT_TRUE(IslZero.is_zero());
  }

  {
    APInt APOne(1, 1, false);
    auto IslOne = valFromAPInt(IslCtx, APOne, false);
    EXPECT_TRUE(IslOne.is_one());
  }

  {
    APInt APNTwo(2, -2, true);
    auto IslNTwo = valFromAPInt(IslCtx, APNTwo, true);
    auto IslNTwoCmp = isl::val(IslCtx, -2);
    EXPECT_EQ(IslNTwo, IslNTwoCmp);
  }

  {
    APInt APNOne(32, -1, true);
    auto IslNOne = valFromAPInt(IslCtx, APNOne, true);
    EXPECT_TRUE(IslNOne.is_negone());
  }

  {
    APInt APZero(32, 0, false);
    auto IslZero = valFromAPInt(IslCtx, APZero, false);
    EXPECT_TRUE(IslZero.is_zero());
  }

  {
    APInt APOne(32, 1, false);
    auto IslOne = valFromAPInt(IslCtx, APOne, false);
    EXPECT_TRUE(IslOne.is_one());
  }

  {
    APInt APTwo(32, 2, false);
    auto IslTwo = valFromAPInt(IslCtx, APTwo, false);
    EXPECT_EQ(0, IslTwo.cmp_si(2));
  }

  {
    APInt APNOne(32, (1ull << 32) - 1, false);
    auto IslNOne = valFromAPInt(IslCtx, APNOne, false);
    auto IslRef = isl::val(IslCtx, 32).two_exp().sub_ui(1);
    EXPECT_EQ(IslNOne, IslRef);
  }

  {
    APInt APLarge(130, 2, false);
    APLarge = APLarge.shl(70);
    auto IslLarge = valFromAPInt(IslCtx, APLarge, false);
    auto IslRef = isl::val(IslCtx, 71);
    IslRef = IslRef.two_exp();
    EXPECT_EQ(IslLarge, IslRef);
  }

  isl_ctx_free(IslCtx);
}

TEST(Isl, IslValToAPInt) {
  isl_ctx *IslCtx = isl_ctx_alloc();

  {
    auto IslNOne = isl::val(IslCtx, -1);
    auto APNOne = APIntFromVal(IslNOne);
    // Compare with the two's complement of -1 in a 1-bit integer.
    EXPECT_EQ(1, APNOne);
    EXPECT_EQ(1u, APNOne.getBitWidth());
  }

  {
    auto IslNTwo = isl::val(IslCtx, -2);
    auto APNTwo = APIntFromVal(IslNTwo);
    // Compare with the two's complement of -2 in a 2-bit integer.
    EXPECT_EQ(2, APNTwo);
    EXPECT_EQ(2u, APNTwo.getBitWidth());
  }

  {
    auto IslNThree = isl::val(IslCtx, -3);
    auto APNThree = APIntFromVal(IslNThree);
    // Compare with the two's complement of -3 in a 3-bit integer.
    EXPECT_EQ(5, APNThree);
    EXPECT_EQ(3u, APNThree.getBitWidth());
  }

  {
    auto IslNFour = isl::val(IslCtx, -4);
    auto APNFour = APIntFromVal(IslNFour);
    // Compare with the two's complement of -4 in a 3-bit integer.
    EXPECT_EQ(4, APNFour);
    EXPECT_EQ(3u, APNFour.getBitWidth());
  }

  {
    auto IslZero = isl::val(IslCtx, 0);
    auto APZero = APIntFromVal(IslZero);
    EXPECT_EQ(0, APZero);
    EXPECT_EQ(1u, APZero.getBitWidth());
  }

  {
    auto IslOne = isl::val(IslCtx, 1);
    auto APOne = APIntFromVal(IslOne);
    EXPECT_EQ(1, APOne);
    EXPECT_EQ(2u, APOne.getBitWidth());
  }

  {
    auto IslTwo = isl::val(IslCtx, 2);
    auto APTwo = APIntFromVal(IslTwo);
    EXPECT_EQ(2, APTwo);
    EXPECT_EQ(3u, APTwo.getBitWidth());
  }

  {
    auto IslThree = isl::val(IslCtx, 3);
    auto APThree = APIntFromVal(IslThree);
    EXPECT_EQ(3, APThree);
    EXPECT_EQ(3u, APThree.getBitWidth());
  }

  {
    auto IslFour = isl::val(IslCtx, 4);
    auto APFour = APIntFromVal(IslFour);
    EXPECT_EQ(4, APFour);
    EXPECT_EQ(4u, APFour.getBitWidth());
  }

  {
    auto IslNOne = isl::val(IslCtx, 32).two_exp().sub_ui(1);
    auto APNOne = APIntFromVal(IslNOne);
    EXPECT_EQ((1ull << 32) - 1, APNOne);
    EXPECT_EQ(33u, APNOne.getBitWidth());
  }

  {
    auto IslLargeNum = isl::val(IslCtx, 60);
    IslLargeNum = IslLargeNum.two_exp();
    IslLargeNum = IslLargeNum.sub_ui(1);
    auto APLargeNum = APIntFromVal(IslLargeNum);
    EXPECT_EQ((1ull << 60) - 1, APLargeNum);
    EXPECT_EQ(61u, APLargeNum.getBitWidth());
  }

  {
    auto IslExp = isl::val(IslCtx, 500);
    auto IslLargePow2 = IslExp.two_exp();
    auto APLargePow2 = APIntFromVal(IslLargePow2);
    EXPECT_TRUE(APLargePow2.isPowerOf2());
    EXPECT_EQ(502u, APLargePow2.getBitWidth());
    EXPECT_EQ(502u, APLargePow2.getMinSignedBits());
  }

  {
    auto IslExp = isl::val(IslCtx, 500);
    auto IslLargeNPow2 = IslExp.two_exp().neg();
    auto APLargeNPow2 = APIntFromVal(IslLargeNPow2);
    EXPECT_EQ(501u, APLargeNPow2.getBitWidth());
    EXPECT_EQ(501u, APLargeNPow2.getMinSignedBits());
    EXPECT_EQ(500, (-APLargeNPow2).exactLogBase2());
  }

  {
    auto IslExp = isl::val(IslCtx, 512);
    auto IslLargePow2 = IslExp.two_exp();
    auto APLargePow2 = APIntFromVal(IslLargePow2);
    EXPECT_TRUE(APLargePow2.isPowerOf2());
    EXPECT_EQ(514u, APLargePow2.getBitWidth());
    EXPECT_EQ(514u, APLargePow2.getMinSignedBits());
  }

  {
    auto IslExp = isl::val(IslCtx, 512);
    auto IslLargeNPow2 = IslExp.two_exp().neg();
    auto APLargeNPow2 = APIntFromVal(IslLargeNPow2);
    EXPECT_EQ(513u, APLargeNPow2.getBitWidth());
    EXPECT_EQ(513u, APLargeNPow2.getMinSignedBits());
    EXPECT_EQ(512, (-APLargeNPow2).exactLogBase2());
  }

  isl_ctx_free(IslCtx);
}

TEST(Isl, Operators) {
  isl_ctx *IslCtx = isl_ctx_alloc();

  isl::val ValOne = isl::val(IslCtx, 1);
  isl::val ValTwo = isl::val(IslCtx, 2);
  isl::val ValThree = isl::val(IslCtx, 3);
  isl::val ValFour = isl::val(IslCtx, 4);

  isl::space Space = isl::space(IslCtx, 0, 0);
  isl::local_space LS = isl::local_space(Space);

  isl::pw_aff AffOne = isl::aff(LS, ValOne);
  isl::pw_aff AffTwo = isl::aff(LS, ValTwo);
  isl::pw_aff AffThree = isl::aff(LS, ValThree);
  isl::pw_aff AffFour = isl::aff(LS, ValFour);

  // Addition
  {
    EXPECT_EQ(AffOne + AffOne, AffTwo);
    EXPECT_EQ(AffOne + 1, AffTwo);
    EXPECT_EQ(1 + AffOne, AffTwo);
    EXPECT_EQ(AffOne + ValOne, AffTwo);
    EXPECT_EQ(ValOne + AffOne, AffTwo);
  }

  // Multiplication
  {
    EXPECT_EQ(AffTwo * AffTwo, AffFour);
    EXPECT_EQ(AffTwo * 2, AffFour);
    EXPECT_EQ(2 * AffTwo, AffFour);
    EXPECT_EQ(AffTwo * ValTwo, AffFour);
    EXPECT_EQ(ValTwo * AffTwo, AffFour);
  }

  // Subtraction
  {
    EXPECT_EQ(AffTwo - AffOne, AffOne);
    EXPECT_EQ(AffTwo - 1, AffOne);
    EXPECT_EQ(2 - AffOne, AffOne);
    EXPECT_EQ(AffTwo - ValOne, AffOne);
    EXPECT_EQ(ValTwo - AffOne, AffOne);
  }

  // Division
  {
    EXPECT_EQ(AffFour - AffTwo, AffTwo);
    EXPECT_EQ(AffFour - 2, AffTwo);
    EXPECT_EQ(4 - AffTwo, AffTwo);
    EXPECT_EQ(AffFour / ValTwo, AffTwo);
    EXPECT_EQ(AffFour / 2, AffTwo);
  }

  // Remainder
  {
    EXPECT_EQ(AffThree % AffTwo, AffOne);
    EXPECT_EQ(AffThree % 2, AffOne);
    EXPECT_EQ(3 % AffTwo, AffOne);
    EXPECT_EQ(AffThree % ValTwo, AffOne);
    EXPECT_EQ(ValThree % AffTwo, AffOne);
  }

  isl_ctx_free(IslCtx);
}

TEST(Isl, Foreach) {
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> Ctx(isl_ctx_alloc(),
                                                        &isl_ctx_free);

  auto MapSpace = isl::space(Ctx.get(), 0, 1, 1);
  auto TestBMap = isl::basic_map::universe(MapSpace);
  TestBMap = TestBMap.fix_si(isl::dim::out, 0, 0);
  TestBMap = TestBMap.fix_si(isl::dim::out, 0, 0);
  isl::map TestMap = TestBMap;
  isl::union_map TestUMap = TestMap;

  auto SetSpace = isl::space(Ctx.get(), 0, 1);
  isl::basic_set TestBSet = isl::point(SetSpace);
  isl::set TestSet = TestBSet;
  isl::union_set TestUSet = TestSet;

  {
    auto NumBMaps = 0;
    TestMap.foreach_basic_map([&](isl::basic_map BMap) -> isl::stat {
      EXPECT_EQ(BMap, TestBMap);
      NumBMaps++;
      return isl::stat::ok;
    });
    EXPECT_EQ(1, NumBMaps);
  }

  {
    auto NumBSets = 0;
    TestSet.foreach_basic_set([&](isl::basic_set BSet) -> isl::stat {
      EXPECT_EQ(BSet, TestBSet);
      NumBSets++;
      return isl::stat::ok;
    });
    EXPECT_EQ(1, NumBSets);
  }

  {
    auto NumMaps = 0;
    TestUMap.foreach_map([&](isl::map Map) -> isl::stat {
      EXPECT_EQ(Map, TestMap);
      NumMaps++;
      return isl::stat::ok;
    });
    EXPECT_EQ(1, NumMaps);
  }

  {
    auto NumSets = 0;
    TestUSet.foreach_set([&](isl::set Set) -> isl::stat {
      EXPECT_EQ(Set, TestSet);
      NumSets++;
      return isl::stat::ok;
    });
    EXPECT_EQ(1, NumSets);
  }

  {
    auto UPwAff = isl::union_pw_aff(TestUSet, isl::val::zero(Ctx.get()));
    auto NumPwAffs = 0;
    UPwAff.foreach_pw_aff([&](isl::pw_aff PwAff) -> isl::stat {
      EXPECT_TRUE(PwAff.is_cst());
      NumPwAffs++;
      return isl::stat::ok;
    });
    EXPECT_EQ(1, NumPwAffs);
  }

  {
    auto NumBMaps = 0;
    EXPECT_EQ(isl::stat::error,
              TestMap.foreach_basic_map([&](isl::basic_map BMap) -> isl::stat {
                EXPECT_EQ(BMap, TestBMap);
                NumBMaps++;
                return isl::stat::error;
              }));
    EXPECT_EQ(1, NumBMaps);
  }

  {
    auto NumMaps = 0;
    EXPECT_EQ(isl::stat::error,
              TestUMap.foreach_map([&](isl::map Map) -> isl::stat {
                EXPECT_EQ(Map, TestMap);
                NumMaps++;
                return isl::stat::error;
              }));
    EXPECT_EQ(1, NumMaps);
  }

  {
    auto TestPwAff = isl::pw_aff(TestSet, isl::val::zero(Ctx.get()));
    auto NumPieces = 0;
    TestPwAff.foreach_piece([&](isl::set Domain, isl::aff Aff) -> isl::stat {
      EXPECT_EQ(Domain, TestSet);
      EXPECT_TRUE(Aff.is_cst());
      NumPieces++;
      return isl::stat::error;
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

  // Union maps
  EXPECT_EQ(MAP("{ [1] -> [] }"),
            shiftDim(MAP("{ [0] -> [] }"), isl::dim::in, 0, 1));
  EXPECT_EQ(UMAP("{ [1] -> [] }"),
            shiftDim(UMAP("{ [0] -> [] }"), isl::dim::in, 0, 1));
  EXPECT_EQ(MAP("{ [] -> [1] }"),
            shiftDim(MAP("{ [] -> [0] }"), isl::dim::out, 0, 1));
  EXPECT_EQ(UMAP("{ [] -> [1] }"),
            shiftDim(UMAP("{ [] -> [0] }"), isl::dim::out, 0, 1));
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

    // Two writes
    EXPECT_EQ(
        UMAP("{ Elt[] -> [i] : i <= 10 }"),
        computeArrayUnused(UMAP("{ WriteA[] -> [0];  WriteB[] -> [10] }"),
                           UMAP("{ WriteA[] -> Elt[]; WriteB[] -> Elt[] }"),
                           UMAP("{}"), ReadEltInSameInst, false, true));

    // Two unused zones
    // read,write,read,write
    EXPECT_EQ(
        UMAP("{ Elt[] -> [i] : 0 < i <= 10; Elt[] -> [i] : 20 < i <= 30 }"),
        computeArrayUnused(UMAP("{ ReadA[] -> [0]; WriteA[] -> [10]; ReadB[] "
                                "-> [20]; WriteB[] -> [30] }"),
                           UMAP("{ WriteA[] -> Elt[]; WriteB[] -> Elt[] }"),
                           UMAP("{ ReadA[] -> Elt[];  ReadB[] -> Elt[] }"),
                           ReadEltInSameInst, false, true));

    // write, write
    EXPECT_EQ(
        UMAP("{ Elt[] -> [i] : i <= 10 }"),
        computeArrayUnused(
            UMAP("{ WriteA[] -> [0];  WriteB[] -> [10];  Read[] -> [20] }"),
            UMAP("{ WriteA[] -> Elt[]; WriteB[] -> Elt[] }"),
            UMAP("{ Read[] -> Elt[] }"), ReadEltInSameInst, false, true));

    // write, read
    EXPECT_EQ(UMAP("{ Elt[] -> [i] : i <= 0 }"),
              computeArrayUnused(UMAP("{ Write[] -> [0]; Read[] -> [10] }"),
                                 UMAP("{ Write[] -> Elt[] }"),
                                 UMAP("{ Read[] -> Elt[] }"), ReadEltInSameInst,
                                 false, true));

    // read, write, read
    EXPECT_EQ(UMAP("{ Elt[] -> [i] : 0 < i <= 10 }"),
              computeArrayUnused(
                  UMAP("{ ReadA[] -> [0]; Write[] -> [10]; ReadB[] -> [20] }"),
                  UMAP("{ Write[] -> Elt[] }"),
                  UMAP("{ ReadA[] -> Elt[];  ReadB[] -> Elt[] }"),
                  ReadEltInSameInst, false, true));

    // read, write, write
    EXPECT_EQ(
        UMAP("{ Elt[] -> [i] : 0 < i <= 20 }"),
        computeArrayUnused(
            UMAP("{ Read[] -> [0]; WriteA[] -> [10];  WriteB[] -> [20] }"),
            UMAP("{ WriteA[] -> Elt[]; WriteB[] -> Elt[] }"),
            UMAP("{ Read[] -> Elt[] }"), ReadEltInSameInst, false, true));

    // read, write, write, read
    EXPECT_EQ(
        UMAP("{ Elt[] -> [i] : 0 < i <= 20 }"),
        computeArrayUnused(UMAP("{ ReadA[] -> [0]; WriteA[] -> [10];  WriteB[] "
                                "-> [20]; ReadB[] -> [30] }"),
                           UMAP("{ WriteA[] -> Elt[]; WriteB[] -> Elt[] }"),
                           UMAP("{ ReadA[] -> Elt[];  ReadB[] -> Elt[] }"),
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

  // Map domains
  EXPECT_EQ(UMAP("{}"), convertZoneToTimepoints(UMAP("{ [1] -> [] }"),
                                                isl::dim::in, false, false));
  EXPECT_EQ(UMAP("{ [0] -> [] }"),
            convertZoneToTimepoints(UMAP("{ [1] -> [] }"), isl::dim::in, true,
                                    false));
  EXPECT_EQ(UMAP("{ [1] -> [] }"),
            convertZoneToTimepoints(UMAP("{ [1] -> [] }"), isl::dim::in, false,
                                    true));
  EXPECT_EQ(
      UMAP("{ [0] -> []; [1] -> [] }"),
      convertZoneToTimepoints(UMAP("{ [1] -> [] }"), isl::dim::in, true, true));

  // Map ranges
  EXPECT_EQ(UMAP("{}"), convertZoneToTimepoints(UMAP("{ [] -> [1] }"),
                                                isl::dim::out, false, false));
  EXPECT_EQ(UMAP("{ [] -> [0] }"),
            convertZoneToTimepoints(UMAP("{ [] -> [1] }"), isl::dim::out, true,
                                    false));
  EXPECT_EQ(UMAP("{ [] -> [1] }"),
            convertZoneToTimepoints(UMAP("{ [] -> [1] }"), isl::dim::out, false,
                                    true));
  EXPECT_EQ(UMAP("{ [] -> [0]; [] -> [1] }"),
            convertZoneToTimepoints(UMAP("{ [] -> [1] }"), isl::dim::out, true,
                                    true));
}

TEST(DeLICM, distribute) {
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> Ctx(isl_ctx_alloc(),
                                                        &isl_ctx_free);

  // Basic usage
  EXPECT_EQ(MAP("{ [Domain[] -> Range1[]] -> [Domain[] -> Range2[]] }"),
            distributeDomain(MAP("{ Domain[] -> [Range1[] -> Range2[]] }")));
  EXPECT_EQ(
      MAP("{ [Domain[i,j] -> Range1[i,k]] -> [Domain[i,j] -> Range2[j,k]] }"),
      distributeDomain(MAP("{ Domain[i,j] -> [Range1[i,k] -> Range2[j,k]] }")));

  // Union maps
  EXPECT_EQ(
      UMAP(
          "{ [DomainA[i,j] -> RangeA1[i,k]] -> [DomainA[i,j] -> RangeA2[j,k]];"
          "[DomainB[i,j] -> RangeB1[i,k]] -> [DomainB[i,j] -> RangeB2[j,k]] }"),
      distributeDomain(
          UMAP("{ DomainA[i,j] -> [RangeA1[i,k] -> RangeA2[j,k]];"
               "DomainB[i,j] -> [RangeB1[i,k] -> RangeB2[j,k]] }")));
}

TEST(DeLICM, lift) {
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> Ctx(isl_ctx_alloc(),
                                                        &isl_ctx_free);

  // Basic usage
  EXPECT_EQ(UMAP("{ [Factor[] -> Domain[]] -> [Factor[] -> Range[]] }"),
            liftDomains(UMAP("{ Domain[] -> Range[] }"), USET("{ Factor[] }")));
  EXPECT_EQ(UMAP("{ [Factor[l] -> Domain[i,k]] -> [Factor[l] -> Range[j,k]] }"),
            liftDomains(UMAP("{ Domain[i,k] -> Range[j,k] }"),
                        USET("{ Factor[l] }")));

  // Multiple maps in union
  EXPECT_EQ(
      UMAP("{ [FactorA[] -> DomainA[]] -> [FactorA[] -> RangeA[]];"
           "[FactorB[] -> DomainA[]] -> [FactorB[] -> RangeA[]];"
           "[FactorA[] -> DomainB[]] -> [FactorA[] -> RangeB[]];"
           "[FactorB[] -> DomainB[]] -> [FactorB[] -> RangeB[]] }"),
      liftDomains(UMAP("{ DomainA[] -> RangeA[]; DomainB[] -> RangeB[] }"),
                  USET("{ FactorA[]; FactorB[] }")));
}

TEST(DeLICM, apply) {
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> Ctx(isl_ctx_alloc(),
                                                        &isl_ctx_free);

  // Basic usage
  EXPECT_EQ(
      UMAP("{ [DomainDomain[] -> NewDomainRange[]] -> Range[] }"),
      applyDomainRange(UMAP("{ [DomainDomain[] -> DomainRange[]] -> Range[] }"),
                       UMAP("{ DomainRange[] -> NewDomainRange[] }")));
  EXPECT_EQ(
      UMAP("{ [DomainDomain[i,k] -> NewDomainRange[j,k,l]] -> Range[i,j] }"),
      applyDomainRange(
          UMAP("{ [DomainDomain[i,k] -> DomainRange[j,k]] -> Range[i,j] }"),
          UMAP("{ DomainRange[j,k] -> NewDomainRange[j,k,l] }")));

  // Multiple maps in union
  EXPECT_EQ(UMAP("{ [DomainDomainA[] -> NewDomainRangeA[]] -> RangeA[];"
                 "[DomainDomainB[] -> NewDomainRangeB[]] -> RangeB[] }"),
            applyDomainRange(
                UMAP("{ [DomainDomainA[] -> DomainRangeA[]] -> RangeA[];"
                     "[DomainDomainB[] -> DomainRangeB[]] -> RangeB[] }"),
                UMAP("{ DomainRangeA[] -> NewDomainRangeA[];"
                     "DomainRangeB[] -> NewDomainRangeB[] }")));
}

} // anonymous namespace
