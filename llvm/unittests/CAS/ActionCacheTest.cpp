//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the tests for ActionCaches.
///
//===----------------------------------------------------------------------===//

#include "llvm/CAS/ActionCache.h"
#include "CASTestConfig.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::cas;

TEST_P(CASTest, ActionCacheHit) {
  std::shared_ptr<ObjectStore> CAS = createObjectStore();
  std::unique_ptr<ActionCache> Cache = createActionCache();

  std::optional<ObjectProxy> ID;
  ASSERT_THAT_ERROR(CAS->createProxy({}, "1").moveInto(ID), Succeeded());
  std::optional<CASID> ResultID;
  ASSERT_THAT_ERROR(Cache->put(*ID, *ID), Succeeded());
  ASSERT_THAT_ERROR(Cache->get(*ID).moveInto(ResultID), Succeeded());
  ASSERT_TRUE(ResultID);
  std::optional<ObjectRef> Result = CAS->getReference(*ResultID);
  ASSERT_TRUE(Result);
  ASSERT_EQ(*ID, *Result);
}

TEST_P(CASTest, ActionCacheMiss) {
  std::shared_ptr<ObjectStore> CAS = createObjectStore();
  std::unique_ptr<ActionCache> Cache = createActionCache();

  std::optional<ObjectProxy> ID1, ID2;
  ASSERT_THAT_ERROR(CAS->createProxy({}, "1").moveInto(ID1), Succeeded());
  ASSERT_THAT_ERROR(CAS->createProxy({}, "2").moveInto(ID2), Succeeded());
  ASSERT_THAT_ERROR(Cache->put(*ID1, *ID2), Succeeded());
  // This is a cache miss for looking up a key doesn't exist.
  std::optional<CASID> Result1;
  ASSERT_THAT_ERROR(Cache->get(*ID2).moveInto(Result1), Succeeded());
  ASSERT_FALSE(Result1);

  ASSERT_THAT_ERROR(Cache->put(*ID2, *ID1), Succeeded());
  // Cache hit after adding the value.
  std::optional<CASID> Result2;
  ASSERT_THAT_ERROR(Cache->get(*ID2).moveInto(Result2), Succeeded());
  ASSERT_TRUE(Result2);
  std::optional<ObjectRef> Ref = CAS->getReference(*Result2);
  ASSERT_TRUE(Ref);
  ASSERT_EQ(*ID1, *Ref);
}

TEST_P(CASTest, ActionCacheRewrite) {
  std::shared_ptr<ObjectStore> CAS = createObjectStore();
  std::unique_ptr<ActionCache> Cache = createActionCache();

  std::optional<ObjectProxy> ID1, ID2;
  ASSERT_THAT_ERROR(CAS->createProxy({}, "1").moveInto(ID1), Succeeded());
  ASSERT_THAT_ERROR(CAS->createProxy({}, "2").moveInto(ID2), Succeeded());
  ASSERT_THAT_ERROR(Cache->put(*ID1, *ID1), Succeeded());
  // Writing to the same key with different value is error.
  ASSERT_THAT_ERROR(Cache->put(*ID1, *ID2), Failed());
  // Writing the same value multiple times to the same key is fine.
  ASSERT_THAT_ERROR(Cache->put(*ID1, *ID1), Succeeded());
}
