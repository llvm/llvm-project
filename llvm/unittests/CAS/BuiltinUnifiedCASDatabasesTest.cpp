//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/BuiltinUnifiedCASDatabases.h"
#include "CASTestConfig.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/BuiltinCASContext.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/CAS/UnifiedOnDiskCache.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::unittest::cas;

TEST_F(OnDiskCASTest, UnifiedCASMaterializationCheckPreventsGarbageCollection) {
  unittest::TempDir Temp("on-disk-unified-cas", /*Unique=*/true);

  auto WithCAS = [&](llvm::function_ref<void(ObjectStore &)> Action) {
    std::pair<std::unique_ptr<ObjectStore>, std::unique_ptr<ActionCache>> DBs;
    ASSERT_THAT_ERROR(
        createOnDiskUnifiedCASDatabases(Temp.path()).moveInto(DBs),
        Succeeded());
    ObjectStore &CAS = *DBs.first;
    ASSERT_THAT_ERROR(CAS.setSizeLimit(1), Succeeded());
    Action(CAS);
  };

  std::optional<CASID> ID;

  // Create an object in the CAS.
  WithCAS([&ID](ObjectStore &CAS) {
    std::optional<ObjectRef> Ref;
    ASSERT_THAT_ERROR(CAS.store({}, "blah").moveInto(Ref), Succeeded());
    ASSERT_TRUE(Ref.has_value());

    ID = CAS.getID(*Ref);
  });

  // Check materialization and prune the storage.
  WithCAS([&ID](ObjectStore &CAS) {
    std::optional<ObjectRef> Ref = CAS.getReference(*ID);
    ASSERT_TRUE(Ref.has_value());

    std::optional<bool> IsMaterialized;
    ASSERT_THAT_ERROR(CAS.isMaterialized(*Ref).moveInto(IsMaterialized),
                      Succeeded());
    ASSERT_TRUE(IsMaterialized);

    ASSERT_THAT_ERROR(CAS.pruneStorageData(), Succeeded());
  });

  // Verify that the previous materialization check kept the object in the CAS.
  WithCAS([&ID](ObjectStore &CAS) {
    std::optional<ObjectRef> Ref = CAS.getReference(*ID);
    ASSERT_TRUE(Ref.has_value());

    std::optional<bool> IsMaterialized;
    ASSERT_THAT_ERROR(CAS.isMaterialized(*Ref).moveInto(IsMaterialized),
                      Succeeded());
    ASSERT_TRUE(IsMaterialized);
  });
}

TEST_F(OnDiskCASTest, UnifiedCASSharedPathDifferentHash) {
  unittest::TempDir Temp("on-disk-unified-cas", /*Unique=*/true);

  // On-disk CAS data of another hashing algorithm in the same path.
  {
    std::unique_ptr<ondisk::UnifiedOnDiskCache> UniDB;
    ASSERT_THAT_ERROR(ondisk::UnifiedOnDiskCache::open(
                          Temp.path(), /*SizeLimit=*/std::nullopt, "OtherHash",
                          sizeof(builtin::HashType))
                          .moveInto(UniDB),
                      Succeeded());
  }

  // The builtin on-disk CAS keeps its data in its own sub-directory.
  {
    std::pair<std::unique_ptr<ObjectStore>, std::unique_ptr<ActionCache>> DBs;
    ASSERT_THAT_ERROR(
        createOnDiskUnifiedCASDatabases(Temp.path()).moveInto(DBs),
        Succeeded());
    std::optional<ObjectRef> Ref;
    ASSERT_THAT_ERROR(DBs.first->store({}, "blah").moveInto(Ref), Succeeded());
  }
  std::string BuiltinDir(Temp.path(builtin::BuiltinCASContext::getHashName()));
  EXPECT_TRUE(sys::fs::exists(BuiltinDir + "/v1.1"));
  EXPECT_TRUE(sys::fs::exists(Temp.path("v1.1")));

  std::optional<ValidationResult> Result;
  ASSERT_THAT_ERROR(
      validateOnDiskUnifiedCASDatabasesIfNeeded(Temp.path(), /*CheckHash=*/true,
                                                /*ForceValidation=*/true)
          .moveInto(Result),
      Succeeded());
  EXPECT_EQ(Result, ValidationResult::Valid);

  // Recovery only affects the builtin on-disk CAS.
  ASSERT_FALSE(sys::fs::remove(BuiltinDir + "/v1.1/data.v1"));
  EXPECT_THAT_EXPECTED(
      validateOnDiskUnifiedCASDatabasesIfNeeded(Temp.path(), /*CheckHash=*/true,
                                                /*ForceValidation=*/true),
      Failed());
  ASSERT_THAT_ERROR(
      recoverOnDiskUnifiedCASDatabases(Temp.path()).moveInto(Result),
      Succeeded());
  EXPECT_EQ(Result, ValidationResult::Recovered);
  EXPECT_FALSE(sys::fs::exists(BuiltinDir + "/v1.1"));
  EXPECT_TRUE(sys::fs::exists(Temp.path("v1.1")));
}
