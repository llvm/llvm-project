//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file declares interface for OnDiskTrieRawHashMap, a thread-safe and
/// (mostly) lock-free hash map stored as trie and backed by persistent files on
/// disk.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_ONDISKTRIERAWHASHMAP_H
#define LLVM_CAS_ONDISKTRIERAWHASHMAP_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CAS/FileOffset.h"
#include "llvm/Support/Error.h"
#include <optional>

namespace llvm {

class raw_ostream;

namespace cas {

/// OnDiskTrieRawHashMap is a persistent trie data structure used as hash maps.
/// The keys are fixed length, and are expected to be binary hashes with a
/// normal distribution.
///
/// - Thread-safety is achieved through the use of atomics within a shared
///   memory mapping. Atomic access does not work on networked filesystems.
/// - Filesystem locks are used, but only sparingly:
///     - during initialization, for creating / opening an existing store;
///     - for the lifetime of the instance, a shared/reader lock is held
///     - during destruction, if there are no concurrent readers, to shrink the
///       files to their minimum size.
/// - Path is used as a directory:
///     - "index" stores the root trie and subtries.
///     - "data" stores (most of) the entries, like a bump-ptr-allocator.
///     - Large entries are stored externally in a file named by the key.
/// - Code is system-dependent and binary format itself is not portable. These
///   are not artifacts that can/should be moved between different systems; they
///   are only appropriate for local storage.
class OnDiskTrieRawHashMap {
public:
  LLVM_DUMP_METHOD void dump() const;
  void
  print(raw_ostream &OS,
        function_ref<void(ArrayRef<char>)> PrintRecordData = nullptr) const;

public:
  /// Const value proxy to access the records stored in TrieRawHashMap.
  struct ConstValueProxy {
    ConstValueProxy() = default;
    ConstValueProxy(ArrayRef<uint8_t> Hash, ArrayRef<char> Data)
        : Hash(Hash), Data(Data) {}
    ConstValueProxy(ArrayRef<uint8_t> Hash, StringRef Data)
        : Hash(Hash), Data(Data.begin(), Data.size()) {}

    ArrayRef<uint8_t> Hash;
    ArrayRef<char> Data;
  };

  /// Value proxy to access the records stored in TrieRawHashMap.
  struct ValueProxy {
    operator ConstValueProxy() const { return ConstValueProxy(Hash, Data); }

    ValueProxy() = default;
    ValueProxy(ArrayRef<uint8_t> Hash, MutableArrayRef<char> Data)
        : Hash(Hash), Data(Data) {}

    ArrayRef<uint8_t> Hash;
    MutableArrayRef<char> Data;
  };

  /// Validate the trie data structure.
  ///
  /// Callback receives the file offset to the data entry and the data stored.
  LLVM_ABI_FOR_TEST Error validate(
      function_ref<Error(FileOffset, ConstValueProxy)> RecordVerifier) const;

  /// Check the valid range of file offset for OnDiskTrieRawHashMap.
  static bool validOffset(FileOffset Offset) {
    return Offset.get() < (1LL << 48);
  }

public:
  /// Template class to implement a `pointer` type into the trie data structure.
  ///
  /// It provides pointer-like operation, e.g., dereference to get underlying
  /// data. It also reserves the top 16 bits of the pointer value, which can be
  /// used to pack additional information if needed.
  template <class ProxyT> class PointerImpl {
  public:
    FileOffset getOffset() const {
      return FileOffset(OffsetLow32 | (uint64_t)OffsetHigh16 << 32);
    }

    explicit operator bool() const { return IsValue; }

    const ProxyT &operator*() const {
      assert(IsValue);
      return Value;
    }
    const ProxyT *operator->() const {
      assert(IsValue);
      return &Value;
    }

    PointerImpl() = default;

  protected:
    PointerImpl(ProxyT Value, FileOffset Offset, bool IsValue = true)
        : Value(Value), OffsetLow32((uint64_t)Offset.get()),
          OffsetHigh16((uint64_t)Offset.get() >> 32), IsValue(IsValue) {
      if (IsValue)
        assert(validOffset(Offset));
    }

    ProxyT Value;
    uint32_t OffsetLow32 = 0;
    uint16_t OffsetHigh16 = 0;

    // True if points to a value (not a "nullptr"). Use an extra field because
    // 0 can be a valid offset.
    bool IsValue = false;
  };

  class OnDiskPtr;
  class ConstOnDiskPtr : public PointerImpl<ConstValueProxy> {
  public:
    ConstOnDiskPtr() = default;

  private:
    friend class OnDiskPtr;
    friend class OnDiskTrieRawHashMap;
    using ConstOnDiskPtr::PointerImpl::PointerImpl;
  };

  class OnDiskPtr : public PointerImpl<ValueProxy> {
  public:
    operator ConstOnDiskPtr() const {
      return ConstOnDiskPtr(Value, getOffset(), IsValue);
    }

    OnDiskPtr() = default;

  private:
    friend class OnDiskTrieRawHashMap;
    using OnDiskPtr::PointerImpl::PointerImpl;
  };

  /// Find the value from hash.
  ///
  /// \returns pointer to the value if exists, otherwise returns a non-value
  /// pointer that evaluates to `false` when convert to boolean.
  LLVM_ABI_FOR_TEST ConstOnDiskPtr find(ArrayRef<uint8_t> Hash) const;

  /// Helper function to recover a pointer into the trie from file offset.
  LLVM_ABI_FOR_TEST Expected<ConstOnDiskPtr>
  recoverFromFileOffset(FileOffset Offset) const;

  using LazyInsertOnConstructCB =
      function_ref<void(FileOffset TentativeOffset, ValueProxy TentativeValue)>;
  using LazyInsertOnLeakCB =
      function_ref<void(FileOffset TentativeOffset, ValueProxy TentativeValue,
                        FileOffset FinalOffset, ValueProxy FinalValue)>;

  /// Insert lazily.
  ///
  /// \p OnConstruct is called when ready to insert a value, after allocating
  /// space for the data. It is called at most once.
  ///
  /// \p OnLeak is called only if \p OnConstruct has been called and a race
  /// occurred before insertion, causing the tentative offset and data to be
  /// abandoned. This allows clients to clean up other results or update any
  /// references.
  ///
  /// NOTE: Does *not* guarantee that \p OnConstruct is only called on success.
  /// The in-memory \a TrieRawHashMap uses LazyAtomicPointer to synchronize
  /// simultaneous writes, but that seems dangerous to use in a memory-mapped
  /// file in case a process crashes in the busy state.
  LLVM_ABI_FOR_TEST Expected<OnDiskPtr>
  insertLazy(ArrayRef<uint8_t> Hash,
             LazyInsertOnConstructCB OnConstruct = nullptr,
             LazyInsertOnLeakCB OnLeak = nullptr);

  Expected<OnDiskPtr> insert(const ConstValueProxy &Value) {
    return insertLazy(Value.Hash, [&](FileOffset, ValueProxy Allocated) {
      assert(Allocated.Hash == Value.Hash);
      assert(Allocated.Data.size() == Value.Data.size());
      llvm::copy(Value.Data, Allocated.Data.begin());
    });
  }

  LLVM_ABI_FOR_TEST size_t size() const;
  LLVM_ABI_FOR_TEST size_t capacity() const;

  /// Gets or creates a file at \p Path with a hash-mapped trie named \p
  /// TrieName. The hash size is \p NumHashBits (in bits) and the records store
  /// data of size \p DataSize (in bytes).
  ///
  /// \p MaxFileSize controls the maximum file size to support, limiting the
  /// size of the \a mapped_file_region. \p NewFileInitialSize is the starting
  /// size if a new file is created.
  ///
  /// \p NewTableNumRootBits and \p NewTableNumSubtrieBits are hints to
  /// configure the trie, if it doesn't already exist.
  ///
  /// \pre NumHashBits is a multiple of 8 (byte-aligned).
  LLVM_ABI_FOR_TEST static Expected<OnDiskTrieRawHashMap>
  create(const Twine &Path, const Twine &TrieName, size_t NumHashBits,
         uint64_t DataSize, uint64_t MaxFileSize,
         std::optional<uint64_t> NewFileInitialSize,
         std::optional<size_t> NewTableNumRootBits = std::nullopt,
         std::optional<size_t> NewTableNumSubtrieBits = std::nullopt);

  LLVM_ABI_FOR_TEST OnDiskTrieRawHashMap(OnDiskTrieRawHashMap &&RHS);
  LLVM_ABI_FOR_TEST OnDiskTrieRawHashMap &operator=(OnDiskTrieRawHashMap &&RHS);
  LLVM_ABI_FOR_TEST ~OnDiskTrieRawHashMap();

private:
  struct ImplType;
  explicit OnDiskTrieRawHashMap(std::unique_ptr<ImplType> Impl);
  std::unique_ptr<ImplType> Impl;
};

} // namespace cas
} // namespace llvm

#endif // LLVM_CAS_ONDISKTRIERAWHASHMAP_H
