//===- OnDiskHashMappedTrie.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_ONDISKHASHMAPPEDTRIE_H
#define LLVM_CAS_ONDISKHASHMAPPEDTRIE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/FileSystem.h"
#include <atomic>
#include <mutex>
#include <optional>

namespace llvm {

class MemoryBuffer;
class raw_ostream;

namespace cas {

class FileOffset {
public:
  int64_t get() const { return Offset; }

  explicit operator bool() const { return Offset; }

  FileOffset() = default;
  explicit FileOffset(int64_t Offset) : Offset(Offset) { assert(Offset >= 0); }

private:
  int64_t Offset = 0;
};

/// On-disk hash-mapped trie. Thread-safe / lock-free.
///
/// This is an on-disk, (mostly) thread-safe key-value store that is (mostly)
/// lock-free. The keys are fixed length, and are expected to be binary hashes
/// with a normal distribution.
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
/// - Code is system-dependent (Windows not yet implemented), and binary format
///   itself is not portable. These are not artifacts that can/should be moved
///   between different systems; they are only appropriate for local storage.
///
/// FIXME: Add support for storing top-level metadata or identifiers that can
/// be created / read during initialization.
///
/// FIXME: Implement for Windows. See comment next to implementation of \a
/// OnDiskHashMappedTrie::MappedFileInfo::open().
class OnDiskHashMappedTrie {
public:
  LLVM_DUMP_METHOD void dump() const;
  void
  print(raw_ostream &OS,
        function_ref<void(ArrayRef<char>)> PrintRecordData = nullptr) const;

public:
  struct ConstValueProxy {
    ConstValueProxy() = default;
    ConstValueProxy(ArrayRef<uint8_t> Hash, ArrayRef<char> Data)
        : Hash(Hash), Data(Data) {}
    ConstValueProxy(ArrayRef<uint8_t> Hash, StringRef Data)
        : Hash(Hash), Data(Data.begin(), Data.size()) {}

    ArrayRef<uint8_t> Hash;
    ArrayRef<char> Data;
  };

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
  Error validate(
      function_ref<Error(FileOffset, ConstValueProxy)> RecordVerifier) const;

public:
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
    PointerImpl(FileOffset Offset, ProxyT Value)
        : PointerImpl(Value, Offset, /*IsValue=*/true) {}

    PointerImpl(ProxyT Value, FileOffset Offset, bool IsValue)
        : Value(Value), OffsetLow32((uint64_t)Offset.get()),
          OffsetHigh16((uint64_t)Offset.get() >> 32), IsValue(IsValue) {
      if (IsValue)
        checkOffset(Offset);
    }

    static void checkOffset(FileOffset Offset) {
      assert(Offset.get() > 0);
      assert((uint64_t)Offset.get() < (1LL << 48));
    }

    ProxyT Value;
    uint32_t OffsetLow32 = 0;
    uint16_t OffsetHigh16 = 0;
    bool IsValue = false;
  };

  class pointer;
  class const_pointer : public PointerImpl<ConstValueProxy> {
  public:
    const_pointer() = default;

  private:
    friend class pointer;
    friend class OnDiskHashMappedTrie;
    using const_pointer::PointerImpl::PointerImpl;
  };

  class pointer : public PointerImpl<ValueProxy> {
  public:
    operator const_pointer() const {
      return const_pointer(Value, getOffset(), IsValue);
    }

    pointer() = default;

  private:
    friend class OnDiskHashMappedTrie;
    using pointer::PointerImpl::PointerImpl;
  };

  pointer getMutablePointer(const_pointer CP) {
    if (!CP)
      return pointer();
    ValueProxy V{CP->Hash, MutableArrayRef(const_cast<char *>(CP->Data.data()),
                                           CP->Data.size())};
    return pointer(CP.getOffset(), V);
  }

  const_pointer find(ArrayRef<uint8_t> Hash) const;
  pointer find(ArrayRef<uint8_t> Hash) {
    return getMutablePointer(
        const_cast<const OnDiskHashMappedTrie *>(this)->find(Hash));
  }

  const_pointer recoverFromHashPointer(const uint8_t *HashBegin) const;
  pointer recoverFromHashPointer(const uint8_t *HashBegin) {
    return getMutablePointer(
        const_cast<const OnDiskHashMappedTrie *>(this)->recoverFromHashPointer(
            HashBegin));
  }

  const_pointer recoverFromFileOffset(FileOffset Offset) const;
  pointer recoverFromFileOffset(FileOffset Offset) {
    return getMutablePointer(
        const_cast<const OnDiskHashMappedTrie *>(this)->recoverFromFileOffset(
            Offset));
  }

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
  /// The in-memory \a HashMappedTrie uses LazyAtomicPointer to synchronize
  /// simultaneous writes, but that seems dangerous to use in a memory-mapped
  /// file in case a process crashes in the busy state.
  Expected<pointer> insertLazy(ArrayRef<uint8_t> Hash,
                               LazyInsertOnConstructCB OnConstruct = nullptr,
                               LazyInsertOnLeakCB OnLeak = nullptr);

  Expected<pointer> insert(const ConstValueProxy &Value) {
    return insertLazy(Value.Hash, [&](FileOffset, ValueProxy Allocated) {
      assert(Allocated.Hash == Value.Hash);
      assert(Allocated.Data.size() == Value.Data.size());
      llvm::copy(Value.Data, Allocated.Data.begin());
    });
  }

  size_t size() const;
  size_t capacity() const;

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
  ///
  /// TODO: Expose the internal DatabaseFile abstraction and add support for
  /// adding more tables to a single file.
  ///
  /// FIXME: Rename to getOrCreate().
  static Expected<OnDiskHashMappedTrie>
  create(const Twine &Path, const Twine &TrieName, size_t NumHashBits,
         uint64_t DataSize, uint64_t MaxFileSize,
         std::optional<uint64_t> NewFileInitialSize,
         std::optional<size_t> NewTableNumRootBits = std::nullopt,
         std::optional<size_t> NewTableNumSubtrieBits = std::nullopt);

  OnDiskHashMappedTrie(OnDiskHashMappedTrie &&RHS);
  OnDiskHashMappedTrie &operator=(OnDiskHashMappedTrie &&RHS);
  ~OnDiskHashMappedTrie();

private:
  struct ImplType;
  explicit OnDiskHashMappedTrie(std::unique_ptr<ImplType> Impl);
  std::unique_ptr<ImplType> Impl;
};

/// Sink for data. Stores variable length data with 8-byte alignment. Does not
/// track size of data, which is assumed to known from context, or embedded.
/// Uses 0-padding but does not guarantee 0-termination.
class OnDiskDataAllocator {
public:
  using ValueProxy = MutableArrayRef<char>;

  /// An iterator-like return value for data insertion. Maybe it should be
  /// called \c iterator, but it has no increment.
  class pointer {
  public:
    FileOffset getOffset() const { return Offset; }
    explicit operator bool() const { return bool(getOffset()); }
    const ValueProxy &operator*() const {
      assert(Offset && "Null dereference");
      return Value;
    }
    const ValueProxy *operator->() const {
      assert(Offset && "Null dereference");
      return &Value;
    }

    pointer() = default;

  private:
    friend class OnDiskDataAllocator;
    pointer(FileOffset Offset, ValueProxy Value)
        : Offset(Offset), Value(Value) {}
    FileOffset Offset;
    ValueProxy Value;
  };

  // Look up the data stored at the given offset.
  const char *beginData(FileOffset Offset) const;
  char *beginData(FileOffset Offset) {
    return const_cast<char *>(
        const_cast<const OnDiskDataAllocator *>(this)->beginData(Offset));
  }

  Expected<pointer> allocate(size_t Size);
  Expected<pointer> save(ArrayRef<char> Data) {
    auto P = allocate(Data.size());
    if (LLVM_UNLIKELY(!P))
      return P.takeError();
    llvm::copy(Data, (*P)->begin());
    return P;
  }
  Expected<pointer> save(StringRef Data) {
    return save(ArrayRef<char>(Data.begin(), Data.size()));
  }

  /// \returns the buffer that was allocated at \p create time, with size
  /// \p UserHeaderSize.
  MutableArrayRef<uint8_t> getUserHeader();

  size_t size() const;
  size_t capacity() const;

  static Expected<OnDiskDataAllocator>
  create(const Twine &Path, const Twine &TableName, uint64_t MaxFileSize,
         std::optional<uint64_t> NewFileInitialSize,
         uint32_t UserHeaderSize = 0,
         function_ref<void(void *)> UserHeaderInit = nullptr);

  OnDiskDataAllocator(OnDiskDataAllocator &&RHS);
  OnDiskDataAllocator &operator=(OnDiskDataAllocator &&RHS);

  // No copy. Just call \a create() again.
  OnDiskDataAllocator(const OnDiskDataAllocator &) = delete;
  OnDiskDataAllocator &operator=(const OnDiskDataAllocator &) = delete;

  ~OnDiskDataAllocator();

private:
  struct ImplType;
  explicit OnDiskDataAllocator(std::unique_ptr<ImplType> Impl);
  std::unique_ptr<ImplType> Impl;
};

} // namespace cas
} // namespace llvm

#endif // LLVM_CAS_ONDISKHASHMAPPEDTRIE_H
