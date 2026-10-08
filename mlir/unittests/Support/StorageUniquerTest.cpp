//===- StorageUniquerTest.cpp - StorageUniquer Tests ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/StorageUniquer.h"
#include "gmock/gmock.h"

using namespace mlir;

namespace {
/// Simple storage class used for testing.
template <typename ConcreteT, typename... Args>
struct SimpleStorage : public StorageUniquer::BaseStorage {
  using Base = SimpleStorage<ConcreteT, Args...>;
  using KeyTy = std::tuple<Args...>;

  SimpleStorage(KeyTy key) : key(key) {}

  /// Get an instance of this storage instance.
  template <typename... ParamsT>
  static ConcreteT *get(StorageUniquer &uniquer, ParamsT &&...params) {
    return uniquer.get<ConcreteT>(
        /*initFn=*/{}, std::make_tuple(std::forward<ParamsT>(params)...));
  }

  /// Construct an instance with the given storage allocator.
  static ConcreteT *construct(StorageUniquer::StorageAllocator &alloc,
                              KeyTy key) {
    return new (alloc.allocate<ConcreteT>())
        ConcreteT(std::forward<KeyTy>(key));
  }
  bool operator==(const KeyTy &key) const { return this->key == key; }

  KeyTy key;
};
} // namespace

TEST(StorageUniquerTest, NonTrivialDestructor) {
  struct NonTrivialStorage : public SimpleStorage<NonTrivialStorage, bool *> {
    using Base::Base;
    ~NonTrivialStorage() {
      bool *wasDestructed = std::get<0>(key);
      *wasDestructed = true;
    }
  };

  // Verify that the storage instance destructor was properly called.
  bool wasDestructed = false;
  {
    StorageUniquer uniquer;
    uniquer.registerParametricStorageType<NonTrivialStorage>();
    NonTrivialStorage::get(uniquer, &wasDestructed);
  }

  EXPECT_TRUE(wasDestructed);
}

TEST(StorageUniquerTest, TransientScopeAndReset) {
  struct IntStorage : public SimpleStorage<IntStorage, int> {
    using Base::Base;
  };

  StorageUniquer uniquer;
  uniquer.registerParametricStorageType<IntStorage>();

  // Allocate in base layer.
  IntStorage *base1 = IntStorage::get(uniquer, 1);
  IntStorage *base2 = IntStorage::get(uniquer, 2);
  EXPECT_EQ(base1, IntStorage::get(uniquer, 1));
  EXPECT_EQ(base2, IntStorage::get(uniquer, 2));

  // Begin transient scope.
  EXPECT_FALSE(uniquer.isInTransientScope());
  uniquer.beginTransientScope();
  EXPECT_TRUE(uniquer.isInTransientScope());

  // Base instances are still found and have same pointer.
  EXPECT_EQ(base1, IntStorage::get(uniquer, 1));
  EXPECT_EQ(base2, IntStorage::get(uniquer, 2));

  // Allocate in transient layer.
  IntStorage *transient3 = IntStorage::get(uniquer, 3);
  IntStorage *transient4 = IntStorage::get(uniquer, 4);
  EXPECT_EQ(transient3, IntStorage::get(uniquer, 3));
  EXPECT_EQ(transient4, IntStorage::get(uniquer, 4));

  // End transient scope.
  uniquer.endTransientScope();
  EXPECT_FALSE(uniquer.isInTransientScope());

  // Base instances are still intact!
  EXPECT_EQ(base1, IntStorage::get(uniquer, 1));
  EXPECT_EQ(base2, IntStorage::get(uniquer, 2));

  // Re-allocating transient key 3 now produces a valid object.
  IntStorage *new3 = IntStorage::get(uniquer, 3);
  EXPECT_NE(new3, nullptr);
  EXPECT_EQ(std::get<0>(new3->key), 3);
}

TEST(StorageUniquerTest, DestructorOnEndTransientScope) {
  struct NonTrivialStorage : public SimpleStorage<NonTrivialStorage, bool *> {
    using Base::Base;
    ~NonTrivialStorage() {
      bool *wasDestructed = std::get<0>(key);
      *wasDestructed = true;
    }
  };

  bool baseDestructed = false;
  bool transientDestructed = false;
  {
    StorageUniquer uniquer;
    uniquer.registerParametricStorageType<NonTrivialStorage>();

    NonTrivialStorage::get(uniquer, &baseDestructed);

    uniquer.beginTransientScope();

    NonTrivialStorage::get(uniquer, &transientDestructed);

    EXPECT_FALSE(baseDestructed);
    EXPECT_FALSE(transientDestructed);

    // Ending transient scope should destroy only the transient instance.
    uniquer.endTransientScope();

    EXPECT_FALSE(baseDestructed);
    EXPECT_TRUE(transientDestructed);
  }

  // Destroying uniquer should destroy the remaining base instance.
  EXPECT_TRUE(baseDestructed);
}

TEST(StorageUniquerTest, MutableStorageInTransientScope) {
  struct MutableStorage : public SimpleStorage<MutableStorage, int> {
    using Base::Base;
    LogicalResult mutate(StorageUniquer::StorageAllocator &alloc, int newVal) {
      std::get<0>(key) = newVal;
      return success();
    }
  };

  StorageUniquer uniquer;
  uniquer.registerParametricStorageType<MutableStorage>();

  uniquer.beginTransientScope();
  MutableStorage *storage = MutableStorage::get(uniquer, 10);
  EXPECT_EQ(std::get<0>(storage->key), 10);

  EXPECT_TRUE(
      succeeded(uniquer.mutate(TypeID::get<MutableStorage>(), storage, 20)));
  EXPECT_EQ(std::get<0>(storage->key), 20);

  uniquer.endTransientScope();

  // Re-allocating after reset
  MutableStorage *newStorage = MutableStorage::get(uniquer, 10);
  EXPECT_EQ(std::get<0>(newStorage->key), 10);
}
