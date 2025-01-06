//===- TrieRawHashMap.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_TRIERAWHASHMAP_H
#define LLVM_ADT_TRIERAWHASHMAP_H

#include "llvm/ADT/ArrayRef.h"
#include <atomic>
#include <optional>

namespace llvm {

class raw_ostream;

/// TrieRawHashMap - is a lock-free thread-safe trie that is can be used to
/// store/index data based on a hash value. It can be customized to work with
/// any hash algorithm or store any data.
///
/// Data structure:
/// Data node stored in the Trie contains both hash and data:
/// struct {
///    HashT Hash;
///    DataT Data;
/// };
///
/// Data is stored/indexed via a prefix tree, where each node in the tree can be
/// either the root, a sub-trie or a data node. Assuming a 4-bit hash and two
/// data objects {0001, A} and {0100, B}, it can be stored in a trie
/// (assuming Root has 2 bits, SubTrie has 1 bit):
///  +--------+
///  |Root[00]| -> {0001, A}
///  |    [01]| -> {0100, B}
///  |    [10]| (empty)
///  |    [11]| (empty)
///  +--------+
///
/// Inserting a new object {0010, C} will result in:
///  +--------+    +----------+
///  |Root[00]| -> |SubTrie[0]| -> {0001, A}
///  |        |    |       [1]| -> {0010, C}
///  |        |    +----------+
///  |    [01]| -> {0100, B}
///  |    [10]| (empty)
///  |    [11]| (empty)
///  +--------+
/// Note object A is sunk down to a sub-trie during the insertion. All the
/// nodes are inserted through compare-exchange to ensure thread-safe and
/// lock-free.
///
/// To find an object in the trie, walk the tree with prefix of the hash until
/// the data node is found. Then the hash is compared with the hash stored in
/// the data node to see if the is the same object.
///
/// Hash collision is not allowed so it is recommended to use trie with a
/// "strong" hashing algorithm. A well-distributed hash can also result in
/// better performance and memory usage.
///
/// It currently does not support iteration and deletion.

/// Base class for a lock-free thread-safe hash-mapped trie.
class ThreadSafeTrieRawHashMapBase {
public:
  static constexpr size_t TrieContentBaseSize = 4;
  static constexpr size_t DefaultNumRootBits = 6;
  static constexpr size_t DefaultNumSubtrieBits = 4;

private:
  template <class T> struct AllocValueType {
    char Base[TrieContentBaseSize];
    std::aligned_union_t<sizeof(T), T> Content;
  };

protected:
  template <class T>
  static constexpr size_t DefaultContentAllocSize = sizeof(AllocValueType<T>);

  template <class T>
  static constexpr size_t DefaultContentAllocAlign = alignof(AllocValueType<T>);

  template <class T>
  static constexpr size_t DefaultContentOffset =
      offsetof(AllocValueType<T>, Content);

public:
  static void *operator new(size_t Size) { return ::operator new(Size); }
  void operator delete(void *Ptr) { ::operator delete(Ptr); }

  LLVM_DUMP_METHOD void dump() const;
  void print(raw_ostream &OS) const;

protected:
  /// Result of a lookup. Suitable for an insertion hint. Maybe could be
  /// expanded into an iterator of sorts, but likely not useful (visiting
  /// everything in the trie should probably be done some way other than
  /// through an iterator pattern).
  class PointerBase {
  protected:
    void *get() const { return I == -2u ? P : nullptr; }

  public:
    PointerBase() noexcept = default;

  private:
    friend class ThreadSafeTrieRawHashMapBase;
    explicit PointerBase(void *Content) : P(Content), I(-2u) {}
    PointerBase(void *P, unsigned I, unsigned B) : P(P), I(I), B(B) {}

    bool isHint() const { return I != -1u && I != -2u; }

    void *P = nullptr;
    unsigned I = -1u;
    unsigned B = 0;
  };

  /// Find the stored content with hash.
  PointerBase find(ArrayRef<uint8_t> Hash) const;

  /// Insert and return the stored content.
  PointerBase
  insert(PointerBase Hint, ArrayRef<uint8_t> Hash,
         function_ref<const uint8_t *(void *Mem, ArrayRef<uint8_t> Hash)>
             Constructor);

  ThreadSafeTrieRawHashMapBase() = delete;

  ThreadSafeTrieRawHashMapBase(
      size_t ContentAllocSize, size_t ContentAllocAlign, size_t ContentOffset,
      std::optional<size_t> NumRootBits = std::nullopt,
      std::optional<size_t> NumSubtrieBits = std::nullopt);

  /// Destructor, which asserts if there's anything to do. Subclasses should
  /// call \a destroyImpl().
  ///
  /// \pre \a destroyImpl() was already called.
  ~ThreadSafeTrieRawHashMapBase();
  void destroyImpl(function_ref<void(void *ValueMem)> Destructor);

  ThreadSafeTrieRawHashMapBase(ThreadSafeTrieRawHashMapBase &&RHS);

  // Move assignment is not supported as it is not thread-safe.
  ThreadSafeTrieRawHashMapBase &
  operator=(ThreadSafeTrieRawHashMapBase &&RHS) = delete;

  // No copy.
  ThreadSafeTrieRawHashMapBase(const ThreadSafeTrieRawHashMapBase &) = delete;
  ThreadSafeTrieRawHashMapBase &
  operator=(const ThreadSafeTrieRawHashMapBase &) = delete;

  // Debug functions. Implementation details and not guaranteed to be
  // thread-safe.
  PointerBase getRoot() const;
  unsigned getStartBit(PointerBase P) const;
  unsigned getNumBits(PointerBase P) const;
  unsigned getNumSlotUsed(PointerBase P) const;
  std::string getTriePrefixAsString(PointerBase P) const;
  unsigned getNumTries() const;
  // Visit next trie in the allocation chain.
  PointerBase getNextTrie(PointerBase P) const;

private:
  friend class TrieRawHashMapTestHelper;
  const unsigned short ContentAllocSize;
  const unsigned short ContentAllocAlign;
  const unsigned short ContentOffset;
  unsigned short NumRootBits;
  unsigned short NumSubtrieBits;
  class ImplType;
  // ImplPtr is owned by ThreadSafeTrieRawHashMapBase and needs to be freed in
  // destroyImpl.
  std::atomic<ImplType *> ImplPtr;
  ImplType &getOrCreateImpl();
  ImplType *getImpl() const;
};

/// Lock-free thread-safe hash-mapped trie.
template <class T, size_t NumHashBytes>
class ThreadSafeTrieRawHashMap : public ThreadSafeTrieRawHashMapBase {
public:
  using HashT = std::array<uint8_t, NumHashBytes>;

  class LazyValueConstructor;
  struct value_type {
    const HashT Hash;
    T Data;

    value_type(value_type &&) = default;
    value_type(const value_type &) = default;

    value_type(ArrayRef<uint8_t> Hash, const T &Data)
        : Hash(makeHash(Hash)), Data(Data) {}
    value_type(ArrayRef<uint8_t> Hash, T &&Data)
        : Hash(makeHash(Hash)), Data(std::move(Data)) {}

  private:
    friend class LazyValueConstructor;

    struct EmplaceTag {};
    template <class... ArgsT>
    value_type(ArrayRef<uint8_t> Hash, EmplaceTag, ArgsT &&...Args)
        : Hash(makeHash(Hash)), Data(std::forward<ArgsT>(Args)...) {}

    static HashT makeHash(ArrayRef<uint8_t> HashRef) {
      HashT Hash;
      std::copy(HashRef.begin(), HashRef.end(), Hash.data());
      return Hash;
    }
  };

  using ThreadSafeTrieRawHashMapBase::operator delete;
  using HashType = HashT;

  using ThreadSafeTrieRawHashMapBase::dump;
  using ThreadSafeTrieRawHashMapBase::print;

private:
  template <class ValueT> class PointerImpl : PointerBase {
    friend class ThreadSafeTrieRawHashMap;

    ValueT *get() const {
      return reinterpret_cast<ValueT *>(PointerBase::get());
    }

  public:
    ValueT &operator*() const {
      assert(get());
      return *get();
    }
    ValueT *operator->() const {
      assert(get());
      return get();
    }
    explicit operator bool() const { return get(); }

    PointerImpl() = default;

  protected:
    PointerImpl(PointerBase Result) : PointerBase(Result) {}
  };

public:
  class pointer;
  class const_pointer;
  class pointer : public PointerImpl<value_type> {
    friend class ThreadSafeTrieRawHashMap;
    friend class const_pointer;

  public:
    pointer() = default;

  private:
    pointer(PointerBase Result) : pointer::PointerImpl(Result) {}
  };

  class const_pointer : public PointerImpl<const value_type> {
    friend class ThreadSafeTrieRawHashMap;

  public:
    const_pointer() = default;
    const_pointer(const pointer &P) : const_pointer::PointerImpl(P) {}

  private:
    const_pointer(PointerBase Result) : const_pointer::PointerImpl(Result) {}
  };

  class LazyValueConstructor {
  public:
    value_type &operator()(T &&RHS) {
      assert(Mem && "Constructor already called, or moved away");
      return assign(::new (Mem) value_type(Hash, std::move(RHS)));
    }
    value_type &operator()(const T &RHS) {
      assert(Mem && "Constructor already called, or moved away");
      return assign(::new (Mem) value_type(Hash, RHS));
    }
    template <class... ArgsT> value_type &emplace(ArgsT &&...Args) {
      assert(Mem && "Constructor already called, or moved away");
      return assign(::new (Mem)
                        value_type(Hash, typename value_type::EmplaceTag{},
                                   std::forward<ArgsT>(Args)...));
    }

    LazyValueConstructor(LazyValueConstructor &&RHS)
        : Mem(RHS.Mem), Result(RHS.Result), Hash(RHS.Hash) {
      RHS.Mem = nullptr; // Moved away, cannot call.
    }
    ~LazyValueConstructor() { assert(!Mem && "Constructor never called!"); }

  private:
    value_type &assign(value_type *V) {
      Mem = nullptr;
      Result = V;
      return *V;
    }
    friend class ThreadSafeTrieRawHashMap;
    LazyValueConstructor() = delete;
    LazyValueConstructor(void *Mem, value_type *&Result, ArrayRef<uint8_t> Hash)
        : Mem(Mem), Result(Result), Hash(Hash) {
      assert(Hash.size() == sizeof(HashT) && "Invalid hash");
      assert(Mem && "Invalid memory for construction");
    }
    void *Mem;
    value_type *&Result;
    ArrayRef<uint8_t> Hash;
  };

  /// Insert with a hint. Default-constructed hint will work, but it's
  /// recommended to start with a lookup to avoid overhead in object creation
  /// if it already exists.
  pointer insertLazy(const_pointer Hint, ArrayRef<uint8_t> Hash,
                     function_ref<void(LazyValueConstructor)> OnConstruct) {
    return pointer(ThreadSafeTrieRawHashMapBase::insert(
        Hint, Hash, [&](void *Mem, ArrayRef<uint8_t> Hash) {
          value_type *Result = nullptr;
          OnConstruct(LazyValueConstructor(Mem, Result, Hash));
          return Result->Hash.data();
        }));
  }

  pointer insertLazy(ArrayRef<uint8_t> Hash,
                     function_ref<void(LazyValueConstructor)> OnConstruct) {
    return insertLazy(const_pointer(), Hash, OnConstruct);
  }

  pointer insert(const_pointer Hint, value_type &&HashedData) {
    return insertLazy(Hint, HashedData.Hash, [&](LazyValueConstructor C) {
      C(std::move(HashedData.Data));
    });
  }

  pointer insert(const_pointer Hint, const value_type &HashedData) {
    return insertLazy(Hint, HashedData.Hash,
                      [&](LazyValueConstructor C) { C(HashedData.Data); });
  }

  pointer find(ArrayRef<uint8_t> Hash) {
    assert(Hash.size() == std::tuple_size<HashT>::value);
    return ThreadSafeTrieRawHashMapBase::find(Hash);
  }

  const_pointer find(ArrayRef<uint8_t> Hash) const {
    assert(Hash.size() == std::tuple_size<HashT>::value);
    return ThreadSafeTrieRawHashMapBase::find(Hash);
  }

  ThreadSafeTrieRawHashMap(std::optional<size_t> NumRootBits = std::nullopt,
                           std::optional<size_t> NumSubtrieBits = std::nullopt)
      : ThreadSafeTrieRawHashMapBase(DefaultContentAllocSize<value_type>,
                                     DefaultContentAllocAlign<value_type>,
                                     DefaultContentOffset<value_type>,
                                     NumRootBits, NumSubtrieBits) {}

  ~ThreadSafeTrieRawHashMap() {
    if constexpr (std::is_trivially_destructible<value_type>::value)
      this->destroyImpl(nullptr);
    else
      this->destroyImpl(
          [](void *P) { static_cast<value_type *>(P)->~value_type(); });
  }

  // Move constructor okay.
  ThreadSafeTrieRawHashMap(ThreadSafeTrieRawHashMap &&) = default;

  // No move assignment or any copy.
  ThreadSafeTrieRawHashMap &operator=(ThreadSafeTrieRawHashMap &&) = delete;
  ThreadSafeTrieRawHashMap(const ThreadSafeTrieRawHashMap &) = delete;
  ThreadSafeTrieRawHashMap &
  operator=(const ThreadSafeTrieRawHashMap &) = delete;
};

} // namespace llvm

#endif // LLVM_ADT_TRIERAWHASHMAP_H
