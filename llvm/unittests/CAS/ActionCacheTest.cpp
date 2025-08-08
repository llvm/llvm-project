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
#include "llvm/Support/FileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
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

#if LLVM_ENABLE_ONDISK_CAS
TEST(OnDiskActionCache, ActionCacheResultInvalid) {
  unittest::TempDir Temp("on-disk-cache", /*Unique=*/true);
  std::unique_ptr<ObjectStore> CAS1 = createInMemoryCAS();
  std::unique_ptr<ObjectStore> CAS2 = createInMemoryCAS();

  std::optional<ObjectProxy> ID1, ID2, ID3;
  ASSERT_THAT_ERROR(CAS1->createProxy({}, "1").moveInto(ID1), Succeeded());
  ASSERT_THAT_ERROR(CAS1->createProxy({}, "2").moveInto(ID2), Succeeded());
  ASSERT_THAT_ERROR(CAS2->createProxy({}, "1").moveInto(ID3), Succeeded());

#if !defined(LLVM_ENABLE_ONDISK_CAS)
  // The following won't work without LLVM_ENABLE_ONDISK_CAS enabled.
  // TODO: enable LLVM_ENABLE_ONDISK_CAS on windows
  return;
#endif

  std::unique_ptr<ActionCache> Cache1 =
      cantFail(createOnDiskActionCache(Temp.path()));
  // Test put and get.
  ASSERT_THAT_ERROR(Cache1->put(*ID1, *ID2), Succeeded());
  std::optional<CASID> Result;
  ASSERT_THAT_ERROR(Cache1->get(*ID1).moveInto(Result), Succeeded());
  ASSERT_TRUE(Result);

  // Create OnDiskCAS from the same location but a different underlying CAS.
  std::unique_ptr<ActionCache> Cache2 =
      cantFail(createOnDiskActionCache(Temp.path()));
  // Loading an key that points to an invalid object.
  std::optional<CASID> Result2;
  // Get will work but the resulting CASID doesn't exist in ObjectStore.
  ASSERT_THAT_ERROR(Cache2->get(*ID3).moveInto(Result2), Succeeded());
  ASSERT_FALSE(CAS2->getReference(*Result2));
  // Write a different value will cause error.
  ASSERT_THAT_ERROR(Cache2->put(*ID3, *ID3), Failed());
}
#endif

TEST_P(CASTest, ActionCacheAsync) {
  std::shared_ptr<ObjectStore> CAS = createObjectStore();
  std::unique_ptr<ActionCache> Cache = createActionCache();

  {
    std::optional<ObjectProxy> ID;
    ASSERT_THAT_ERROR(CAS->createProxy({}, "1").moveInto(ID), Succeeded());
    auto PutFuture = Cache->putFuture(*ID, *ID);
    ASSERT_THAT_ERROR(PutFuture.get().take(), Succeeded());
    auto GetFuture = Cache->getFuture(*ID);
    std::optional<CASID> ResultID;
    ASSERT_THAT_ERROR(GetFuture.get().take().moveInto(ResultID), Succeeded());
    ASSERT_TRUE(ResultID);
  }

  std::optional<ObjectProxy> ID2;
  ASSERT_THAT_ERROR(CAS->createProxy({}, "2").moveInto(ID2), Succeeded());
  {
    std::promise<AsyncErrorValue> Promise;
    auto Future = Promise.get_future();
    Cache->putAsync(*ID2, *ID2, false,
                    [Promise = std::move(Promise)](Error E) mutable {
                      Promise.set_value(std::move(E));
                    });
    ASSERT_THAT_ERROR(Future.get().take(), Succeeded());
  }
  {
    std::promise<AsyncCASIDValue> Promise;
    auto Future = Promise.get_future();
    Cache->getAsync(*ID2, false,
                    [Promise = std::move(Promise)](
                        Expected<std::optional<CASID>> Value) mutable {
                      Promise.set_value(std::move(Value));
                    });
    std::optional<CASID> ResultID;
    ASSERT_THAT_ERROR(Future.get().take().moveInto(ResultID), Succeeded());
    ASSERT_TRUE(ResultID);
  }
}
