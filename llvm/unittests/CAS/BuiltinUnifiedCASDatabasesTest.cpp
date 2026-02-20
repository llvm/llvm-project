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
#include "llvm/CAS/ObjectStore.h"
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
