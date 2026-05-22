//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Locked.h"

#include "gtest/gtest.h"

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <type_traits>

using namespace lldb_private;

namespace {
struct Widget {
  int value = 0;
};
} // namespace

// Default mutex types match the LLDB conventions: recursive_mutex for write
// access, llvm::sys::RWMutex for read access.
static_assert(
    std::is_same_v<LockedPtr<Widget>::mutex_type, std::recursive_mutex>);
static_assert(
    std::is_same_v<LockedSP<Widget>::mutex_type, std::recursive_mutex>);
static_assert(
    std::is_same_v<LockedUP<Widget>::mutex_type, std::recursive_mutex>);
static_assert(
    std::is_same_v<SharedLockedPtr<Widget>::mutex_type, llvm::sys::RWMutex>);
static_assert(
    std::is_same_v<SharedLockedSP<Widget>::mutex_type, llvm::sys::RWMutex>);
static_assert(
    std::is_same_v<SharedLockedUP<Widget>::mutex_type, llvm::sys::RWMutex>);

// Force compile-time validation of every (pointer-flavor x mutex) combination
// the templates are intended to support.
template class lldb_private::Locked<Widget *, std::mutex>;
template class lldb_private::Locked<Widget *, std::recursive_mutex>;
template class lldb_private::Locked<Widget *, llvm::sys::RWMutex>;
template class lldb_private::Locked<std::shared_ptr<Widget>,
                                    std::recursive_mutex>;
template class lldb_private::Locked<std::unique_ptr<Widget>,
                                    std::recursive_mutex>;
template class lldb_private::SharedLocked<const Widget *, std::shared_mutex>;
template class lldb_private::SharedLocked<const Widget *, llvm::sys::RWMutex>;
template class lldb_private::SharedLocked<std::shared_ptr<const Widget>,
                                          llvm::sys::RWMutex>;
template class lldb_private::SharedLocked<std::unique_ptr<const Widget>,
                                          llvm::sys::RWMutex>;

// Locked is move-only; SharedLocked is copyable.
static_assert(!std::is_copy_constructible_v<LockedPtr<Widget>>);
static_assert(!std::is_copy_assignable_v<LockedPtr<Widget>>);
static_assert(std::is_move_constructible_v<LockedPtr<Widget>>);
static_assert(std::is_move_assignable_v<LockedPtr<Widget>>);

static_assert(std::is_copy_constructible_v<SharedLockedPtr<Widget>>);
static_assert(std::is_copy_assignable_v<SharedLockedPtr<Widget>>);
static_assert(std::is_move_constructible_v<SharedLockedPtr<Widget>>);
static_assert(std::is_move_assignable_v<SharedLockedPtr<Widget>>);

TEST(LockedTest, DefaultConstructed) {
  LockedPtr<Widget> handle;
  EXPECT_FALSE(handle);
  EXPECT_EQ(handle.get(), nullptr);

  SharedLockedPtr<Widget> shared;
  EXPECT_FALSE(shared);
  EXPECT_EQ(shared.get(), nullptr);
}

TEST(LockedTest, ExclusivePtrAccess) {
  std::recursive_mutex mutex;
  Widget widget;

  LockedPtr<Widget> handle(mutex, &widget);
  ASSERT_TRUE(handle);
  EXPECT_EQ(handle.get(), &widget);

  // A different thread cannot acquire the mutex while the handle holds it.
  std::thread other([&] { EXPECT_FALSE(mutex.try_lock()); });
  other.join();

  handle->value = 7;
  EXPECT_EQ((*handle).value, 7);
}

TEST(LockedTest, ExclusiveReleasesOnDestruction) {
  std::mutex mutex;
  Widget widget;
  {
    LockedPtr<Widget, std::mutex> handle(mutex, &widget);
    EXPECT_FALSE(mutex.try_lock());
  }
  EXPECT_TRUE(mutex.try_lock());
  mutex.unlock();
}

TEST(LockedTest, MoveTransfersLock) {
  std::mutex mutex;
  Widget widget;

  LockedPtr<Widget, std::mutex> first(mutex, &widget);
  EXPECT_FALSE(mutex.try_lock());

  LockedPtr<Widget, std::mutex> second = std::move(first);
  EXPECT_FALSE(mutex.try_lock());
  EXPECT_TRUE(second);
  EXPECT_EQ(second.get(), &widget);

  // The moved-from handle no longer points anywhere — the raw pointer must
  // be nulled out so operator bool can't claim ownership of a lock the
  // handle no longer holds.
  EXPECT_FALSE(first);
  EXPECT_EQ(first.get(), nullptr);
}

TEST(LockedTest, AcceptsExternallyAcquiredLock) {
  std::mutex mutex;
  Widget widget;

  std::unique_lock<std::mutex> lock(mutex);
  LockedPtr<Widget, std::mutex> handle(std::move(lock), &widget);
  ASSERT_TRUE(handle);
  EXPECT_FALSE(mutex.try_lock());
}

TEST(LockedTest, SharedPtrFlavor) {
  std::recursive_mutex mutex;
  auto widget_sp = std::make_shared<Widget>();
  widget_sp->value = 42;

  LockedSP<Widget> handle(mutex, widget_sp);
  ASSERT_TRUE(handle);
  EXPECT_EQ(handle.get(), widget_sp.get());
  EXPECT_EQ(handle->value, 42);
}

TEST(LockedTest, UniquePtrFlavor) {
  std::recursive_mutex mutex;
  auto widget_up = std::make_unique<Widget>();
  widget_up->value = 99;
  Widget *raw = widget_up.get();

  LockedUP<Widget> handle(mutex, std::move(widget_up));
  ASSERT_TRUE(handle);
  EXPECT_EQ(handle.get(), raw);
  EXPECT_EQ(handle->value, 99);
}

TEST(LockedTest, SharedAccessOnRWMutex) {
  llvm::sys::RWMutex mutex;
  Widget widget;
  widget.value = 5;

  SharedLockedPtr<Widget> reader(mutex, &widget);
  ASSERT_TRUE(reader);
  EXPECT_EQ(reader->value, 5);

  static_assert(std::is_same_v<decltype(reader.get()), const Widget *>,
                "shared access borrows a const-qualified pointer");
}

// Copies of a SharedLocked share the same reader lock; the lock is released
// only when the last copy goes away. Verified by polling try_lock() — it
// must keep failing while any copy is alive.
TEST(LockedTest, SharedAccessIsRefCounted) {
  llvm::sys::RWMutex mutex;
  Widget widget;

  std::optional<SharedLockedPtr<Widget>> first;
  first.emplace(mutex, &widget);
  EXPECT_FALSE(mutex.try_lock());

  std::optional<SharedLockedPtr<Widget>> second = first;
  ASSERT_TRUE(second);
  EXPECT_FALSE(mutex.try_lock());

  first.reset();
  // Second copy still holds the reader lock.
  EXPECT_FALSE(mutex.try_lock());

  second.reset();
  // All copies gone — the writer lock is now obtainable.
  EXPECT_TRUE(mutex.try_lock());
  mutex.unlock();
}

TEST(LockedTest, SharedAcceptsExternallyAcquiredLock) {
  llvm::sys::RWMutex mutex;
  Widget widget;

  std::shared_lock<llvm::sys::RWMutex> lock(mutex);
  SharedLockedPtr<Widget> handle(std::move(lock), &widget);
  ASSERT_TRUE(handle);
  EXPECT_FALSE(mutex.try_lock());
}

TEST(LockedTest, SharedMoveNullsSource) {
  llvm::sys::RWMutex mutex;
  Widget widget;

  SharedLockedPtr<Widget> first(mutex, &widget);
  SharedLockedPtr<Widget> second = std::move(first);
  EXPECT_TRUE(second);
  EXPECT_EQ(second.get(), &widget);
  EXPECT_FALSE(first);
  EXPECT_EQ(first.get(), nullptr);
}

TEST(LockedTest, ExclusiveAccessOnRWMutex) {
  llvm::sys::RWMutex mutex;
  Widget widget;
  LockedPtr<Widget, llvm::sys::RWMutex> writer(mutex, &widget);
  ASSERT_TRUE(writer);
  writer->value = 11;
  EXPECT_EQ(widget.value, 11);
}
