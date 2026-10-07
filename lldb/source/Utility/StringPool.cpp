//===-- StringPool.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/StringPool.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Threading.h"

#include <array>
#include <mutex>
#include <shared_mutex>
#include <utility>

#include <cstdint>
#include <cstring>

using namespace lldb_private;

#if !defined(__APPLE__)
using PoolMutex = std::shared_mutex;
#else
#include <os/lock.h>

namespace {
/// On Apple platforms os_unfair_lock is significantly faster than
/// pthread_rwlock for concurrent writes, and roughly on par for concurrent
/// reads.
///
/// The class satisfies both Lockable and SharedLockable so it composes with
/// std::lock_guard and std::shared_lock.
class PoolMutex {
public:
  void lock() { os_unfair_lock_lock(&m_lock); }
  void unlock() { os_unfair_lock_unlock(&m_lock); }
  void lock_shared() { os_unfair_lock_lock(&m_lock); }
  void unlock_shared() { os_unfair_lock_unlock(&m_lock); }

private:
  os_unfair_lock m_lock = OS_UNFAIR_LOCK_INIT;
};
} // namespace
#endif

namespace {
/// The default BumpPtrAllocatorImpl slab size.
constexpr size_t AllocatorSlabSize = 4096;
constexpr size_t SizeThreshold = AllocatorSlabSize;
constexpr size_t NumPools = 256;
/// Every pool shard has its own allocator which receives an equal share of
/// the string allocations. This means that when allocating many strings, every
/// allocator sees only its small share of allocations and assumes LLDB only
/// allocated a small amount of memory so far. In reality LLDB allocated a total
/// memory that is N times as large as what the allocator sees (where N is the
/// number of string pools). This causes that the BumpPtrAllocator continues a
/// long time to allocate memory in small chunks which only makes sense when
/// allocating a small amount of memory (which is true from the perspective of a
/// single allocator). On some systems doing all these small memory allocations
/// causes LLDB to spend a lot of time in malloc, so we need to force all these
/// allocators to behave like one allocator in terms of scaling their memory
/// allocations with increased demand. To do this we set the growth delay for
/// each single allocator to a rate so that our pool of allocators scales their
/// memory allocations similar to a single BumpPtrAllocatorImpl.
///
/// Currently we have 256 string pools and the normal growth delay of the
/// BumpPtrAllocatorImpl is 128 (i.e., the memory allocation size increases
/// every 128 full chunks), so by changing the delay to 1 we get a
/// total growth delay in our allocator collection of 256/1 = 256. This is
/// still only half as fast as a normal allocator but we can't go any faster
/// without decreasing the number of string pools.
constexpr size_t AllocatorGrowthDelay = 1;
using Allocator =
    llvm::BumpPtrAllocatorImpl<llvm::MallocAllocator, AllocatorSlabSize,
                               SizeThreshold, AllocatorGrowthDelay>;
using StringPoolValueType = const char *;
using StringMapType = llvm::StringMap<StringPoolValueType, Allocator>;
using StringPoolEntryType = llvm::StringMapEntry<StringPoolValueType>;

StringPoolEntryType &GetStringMapEntryFromKeyData(const char *keyData) {
  return StringPoolEntryType::GetStringMapEntryFromKeyData(keyData);
}
} // namespace

struct StringPool::PoolEntry {
  mutable PoolMutex m_mutex;
  StringMapType m_string_map;
  /// The exact number of bytes used by this pool.
  /// This excludes alignment, padding and redzones.
  std::size_t used_bytes = 0;
};

StringPool::StringPool() : m_string_pools(new PoolEntry[NumPools]) {}

StringPool::~StringPool() = default;

// Frameworks and dylibs aren't supposed to have global C++ initializers so we
// hide the string pool in a static function so that it will get initialized on
// the first call to this static function.
//
// Note, for now we make the string pool a pointer to the pool, because we
// can't guarantee that some objects won't get destroyed after the global
// destructor chain is run, and trying to make sure no destructors touch
// ConstStrings is difficult.  So we leak the pool instead.
StringPool &StringPool::GetGlobal() {
  static llvm::once_flag g_pool_initialization_flag;
  static StringPool *g_string_pool = nullptr;

  llvm::call_once(g_pool_initialization_flag,
                  []() { g_string_pool = new StringPool(); });

  return *g_string_pool;
}

static StringPool *g_system_pool = nullptr;

void StringPool::Initialize() {
  assert(!g_system_pool && "system pool already initialized");
  g_system_pool = &GetGlobal();
}

void StringPool::Terminate() { g_system_pool = nullptr; }

StringPoolRef StringPool::GetSystem() {
  assert(g_system_pool && "system pool not initialized");
  return StringPoolRef(*g_system_pool);
}

StringPool::PoolEntry &StringPool::selectPool(uint32_t h) {
  return m_string_pools[((h >> 24) ^ (h >> 16) ^ (h >> 8) ^ h) & 0xff];
}

StringPool::PoolEntry &StringPool::selectPool(llvm::StringRef s) {
  return selectPool(StringMapType::hash(s));
}

size_t StringPool::GetConstCStringLength(const char *ccstr) {
  if (ccstr != nullptr) {
    // Since the entry is read only, and we derive the entry entirely from
    // the pointer, we don't need the lock.
    const StringPoolEntryType &entry = GetStringMapEntryFromKeyData(ccstr);
    return entry.getKeyLength();
  }
  return 0;
}

const char *StringPool::GetMangledCounterpart(llvm::StringRef str) {
  const char *const ccstr = str.data();
  if (ccstr != nullptr) {
    const PoolEntry &pool = selectPool(str);
    std::shared_lock<PoolMutex> lock(pool.m_mutex);
    return GetStringMapEntryFromKeyData(ccstr).getValue();
  }
  return nullptr;
}

const char *StringPool::GetConstCString(const char *cstr) {
  if (cstr != nullptr)
    return GetConstCStringWithLength(cstr, strlen(cstr));
  return nullptr;
}

const char *StringPool::GetConstCStringWithLength(const char *cstr,
                                                  size_t cstr_len) {
  if (cstr != nullptr)
    return Intern(llvm::StringRef(cstr, cstr_len));
  return nullptr;
}

const char *StringPool::Intern(llvm::StringRef string_ref) {
  if (string_ref.data()) {
    const uint32_t string_hash = StringMapType::hash(string_ref);
    PoolEntry &pool = selectPool(string_hash);

    {
      std::shared_lock<PoolMutex> lock(pool.m_mutex);
      auto it = pool.m_string_map.find(string_ref, string_hash);
      if (it != pool.m_string_map.end())
        return it->getKeyData();
    }

    std::lock_guard<PoolMutex> lock(pool.m_mutex);
    pool.used_bytes += string_ref.size();
    StringPoolEntryType &entry =
        *pool.m_string_map
             .insert(std::make_pair(string_ref, nullptr), string_hash)
             .first;
    return entry.getKeyData();
  }
  return nullptr;
}

const char *
StringPool::GetConstCStringAndSetMangledCounterPart(llvm::StringRef demangled,
                                                    llvm::StringRef mangled) {
  const char *demangled_ccstr = nullptr;
  const char *const mangled_ccstr = mangled.data();

  {
    const uint32_t demangled_hash = StringMapType::hash(demangled);
    PoolEntry &pool = selectPool(demangled_hash);
    std::lock_guard<PoolMutex> lock(pool.m_mutex);

    // Make or update string pool entry with the mangled counterpart
    StringMapType &map = pool.m_string_map;
    auto [entry, inserted] =
        map.try_emplace_with_hash(demangled, demangled_hash);
    if (inserted)
      pool.used_bytes += demangled.size();

    entry->second = mangled_ccstr;

    // Extract the const version of the demangled_cstr
    demangled_ccstr = entry->getKeyData();
  }

  {
    // Now assign the demangled const string as the counterpart of the
    // mangled const string...
    PoolEntry &pool = selectPool(mangled);
    std::lock_guard<PoolMutex> lock(pool.m_mutex);
    GetStringMapEntryFromKeyData(mangled_ccstr).setValue(demangled_ccstr);
  }

  // Return the constant demangled C string
  return demangled_ccstr;
}

const char *StringPool::GetConstTrimmedCStringWithLength(const char *cstr,
                                                         size_t cstr_len) {
  if (cstr != nullptr) {
    const size_t trimmed_len = strnlen(cstr, cstr_len);
    return GetConstCStringWithLength(cstr, trimmed_len);
  }
  return nullptr;
}

ConstString::MemoryStats StringPool::GetMemoryStats() const {
  ConstString::MemoryStats stats;
  for (size_t i = 0; i < NumPools; ++i) {
    const PoolEntry &pool = m_string_pools[i];
    std::shared_lock<PoolMutex> lock(pool.m_mutex);
    const Allocator &alloc = pool.m_string_map.getAllocator();
    stats.bytes_total += alloc.getTotalMemory();
    stats.bytes_used += pool.used_bytes;
  }
  return stats;
}
