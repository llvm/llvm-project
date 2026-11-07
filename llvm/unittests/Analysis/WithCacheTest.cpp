//===- WithCacheTest.cpp - WithCache unit tests ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/WithCache.h"
#include "llvm/Analysis/SimplifyQuery.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Value.h"
#include "gtest/gtest.h"
#include <type_traits>

using namespace llvm;

// A WithCache<T *> binds by reference to a WithCache<const T *>, so both share
// a single KnownBits cache. Copying or assigning across the two types would
// silently duplicate the cache, so those operations are deleted to avoid the
// perf foot-gun.
static_assert(
    std::is_convertible_v<WithCache<Value *> &, WithCache<const Value *> &>);
static_assert(!std::is_constructible_v<WithCache<const Value *>,
                                       const WithCache<Value *> &>);
static_assert(!std::is_assignable_v<WithCache<const Value *> &,
                                    const WithCache<Value *> &>);

TEST(WithCacheTest, ConstViewSharesKnownBitsCache) {
  LLVMContext Ctx;
  DataLayout DL("");
  SimplifyQuery SQ(DL);
  Value *V = ConstantInt::get(Type::getInt32Ty(Ctx), 42);

  WithCache<Value *> Cache(V);

  auto LearnVIsConstant = [&](WithCache<const Value *> &ConstView) {
    EXPECT_EQ(&*Cache, V);
    EXPECT_EQ(ConstView.getValue(), V);
    EXPECT_TRUE(ConstView.getKnownBits(SQ).isConstant());
  };

  EXPECT_FALSE(Cache.hasKnownBits());
  LearnVIsConstant(Cache);
  EXPECT_TRUE(Cache.hasKnownBits());
}
