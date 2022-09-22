//===- ActionCacheTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/ActionCache.h"
#include "CASTestConfig.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::cas;

TEST_P(CASTest, ActionCacheHit) {
  std::unique_ptr<ObjectStore> CAS = createObjectStore();
  std::unique_ptr<ActionCache> Cache = createActionCache();

  Optional<ObjectProxy> ID;
  ASSERT_THAT_ERROR(CAS->createProxy(None, "1").moveInto(ID), Succeeded());
  Optional<CASID> ResultID;
  ASSERT_THAT_ERROR(Cache->put(*ID, *ID), Succeeded());
  ASSERT_THAT_ERROR(Cache->get(*ID).moveInto(ResultID), Succeeded());
  ASSERT_TRUE(ResultID);
  Optional<ObjectRef> Result = CAS->getReference(*ResultID);
  ASSERT_TRUE(Result);
  ASSERT_EQ(*ID, *Result);
}

TEST_P(CASTest, ActionCacheMiss) {
  std::unique_ptr<ObjectStore> CAS = createObjectStore();
  std::unique_ptr<ActionCache> Cache = createActionCache();

  Optional<ObjectProxy> ID1, ID2;
  ASSERT_THAT_ERROR(CAS->createProxy(None, "1").moveInto(ID1), Succeeded());
  ASSERT_THAT_ERROR(CAS->createProxy(None, "2").moveInto(ID2), Succeeded());
  ASSERT_THAT_ERROR(Cache->put(*ID1, *ID2), Succeeded());
  // This is a cache miss for looking up a key doesn't exist.
  Optional<CASID> Result1;
  ASSERT_THAT_ERROR(Cache->get(*ID2).moveInto(Result1), Succeeded());
  ASSERT_FALSE(Result1);

  ASSERT_THAT_ERROR(Cache->put(*ID2, *ID1), Succeeded());
  // Cache hit after adding the value.
  Optional<CASID> Result2;
  ASSERT_THAT_ERROR(Cache->get(*ID2).moveInto(Result2), Succeeded());
  ASSERT_TRUE(Result2);
  Optional<ObjectRef> Ref = CAS->getReference(*Result2);
  ASSERT_TRUE(Ref);
  ASSERT_EQ(*ID1, *Ref);
}

TEST_P(CASTest, ActionCacheRewrite) {
  std::unique_ptr<ObjectStore> CAS = createObjectStore();
  std::unique_ptr<ActionCache> Cache = createActionCache();

  Optional<ObjectProxy> ID1, ID2;
  ASSERT_THAT_ERROR(CAS->createProxy(None, "1").moveInto(ID1), Succeeded());
  ASSERT_THAT_ERROR(CAS->createProxy(None, "2").moveInto(ID2), Succeeded());
  ASSERT_THAT_ERROR(Cache->put(*ID1, *ID1), Succeeded());
  // Writing to the same key with different value is error.
  ASSERT_THAT_ERROR(Cache->put(*ID1, *ID2), Failed());
  // Writing the same value multiple times to the same key is fine.
  ASSERT_THAT_ERROR(Cache->put(*ID1, *ID1), Succeeded());
}

TEST(OnDiskActionCache, ActionCacheResultInvalid) {
  unittest::TempDir Temp("on-disk-cache", /*Unique=*/true);
  std::unique_ptr<ObjectStore> CAS1 = createInMemoryCAS();
  std::unique_ptr<ObjectStore> CAS2 = createInMemoryCAS();

  Optional<ObjectProxy> ID1, ID2, ID3;
  ASSERT_THAT_ERROR(CAS1->createProxy(None, "1").moveInto(ID1), Succeeded());
  ASSERT_THAT_ERROR(CAS1->createProxy(None, "2").moveInto(ID2), Succeeded());
  ASSERT_THAT_ERROR(CAS2->createProxy(None, "1").moveInto(ID3), Succeeded());

  std::unique_ptr<ActionCache> Cache1 =
      cantFail(createOnDiskActionCache(Temp.path()));
  // Test put and get.
  ASSERT_THAT_ERROR(Cache1->put(*ID1, *ID2), Succeeded());
  Optional<CASID> Result;
  ASSERT_THAT_ERROR(Cache1->get(*ID1).moveInto(Result), Succeeded());
  ASSERT_TRUE(Result);

  // Create OnDiskCAS from the same location but a different underlying CAS.
  std::unique_ptr<ActionCache> Cache2 =
      cantFail(createOnDiskActionCache(Temp.path()));
  // Loading an key that points to an invalid object.
  Optional<CASID> Result2;
  // Get will work but the resulting CASID doesn't exist in ObjectStore.
  ASSERT_THAT_ERROR(Cache2->get(*ID3).moveInto(Result2), Succeeded());
  ASSERT_FALSE(CAS2->getReference(*Result2));
  // Write a different value will cause error.
  ASSERT_THAT_ERROR(Cache2->put(*ID3, *ID3), Failed());
}
