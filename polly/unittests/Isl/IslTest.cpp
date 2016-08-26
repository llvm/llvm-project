//===- IslTest.cpp ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "polly/Support/GICHelper.h"
#include "gtest/gtest.h"
#include "isl/val.h"

using namespace llvm;
using namespace polly;

namespace {

TEST(Isl, APIntToIslVal) {
  isl_ctx *IslCtx = isl_ctx_alloc();

  {
    APInt APNOne(32, -1, true);
    auto *IslNOne = isl_valFromAPInt(IslCtx, APNOne, true);
    EXPECT_EQ(isl_bool_true, isl_val_is_negone(IslNOne));
    isl_val_free(IslNOne);
  }

  {
    APInt APZero(32, 0, false);
    auto *IslZero = isl_valFromAPInt(IslCtx, APZero, false);
    EXPECT_EQ(isl_val_is_zero(IslZero), isl_bool_true);
    isl_val_free(IslZero);
  }

  {
    APInt APOne(32, 1, false);
    auto *IslOne = isl_valFromAPInt(IslCtx, APOne, false);
    EXPECT_EQ(isl_val_is_one(IslOne), isl_bool_true);
    isl_val_free(IslOne);
  }

  {
    APInt APTwo(32, 2, false);
    auto *IslTwo = isl_valFromAPInt(IslCtx, APTwo, false);
    EXPECT_EQ(isl_val_cmp_si(IslTwo, 2), 0);
    isl_val_free(IslTwo);
  }

  {
    APInt APNOne(32, (1ull << 32) - 1, false);
    auto *IslNOne = isl_valFromAPInt(IslCtx, APNOne, false);
    auto *IslRef = isl_val_int_from_ui(IslCtx, (1ull << 32) - 1);
    EXPECT_EQ(isl_val_eq(IslNOne, IslRef), isl_bool_true);
    isl_val_free(IslNOne);
    isl_val_free(IslRef);
  }

  isl_ctx_free(IslCtx);
}

TEST(Isl, IslValToAPInt) {
  isl_ctx *IslCtx = isl_ctx_alloc();

  {
    auto *IslNOne = isl_val_int_from_si(IslCtx, -1);
    auto APNOne = APIntFromVal(IslNOne);
    // APInt has no sign bit, so never equals to a negative number.
    // FIXME: The canonical representation of a negative APInt is two's
    // complement.
    EXPECT_EQ(APNOne, 1);
  }

  {
    auto *IslZero = isl_val_int_from_ui(IslCtx, 0);
    auto APZero = APIntFromVal(IslZero);
    EXPECT_EQ(APZero, 0);
  }

  {
    auto *IslOne = isl_val_int_from_ui(IslCtx, 1);
    auto APOne = APIntFromVal(IslOne);
    EXPECT_EQ(APOne, 1);
  }

  {
    auto *IslTwo = isl_val_int_from_ui(IslCtx, 2);
    auto APTwo = APIntFromVal(IslTwo);
    EXPECT_EQ(APTwo, 2);
  }

  {
    auto *IslNOne = isl_val_int_from_ui(IslCtx, (1ull << 32) - 1);
    auto APNOne = APIntFromVal(IslNOne);
    EXPECT_EQ(APNOne, (1ull << 32) - 1);
  }

  isl_ctx_free(IslCtx);
}

} // anonymous namespace
