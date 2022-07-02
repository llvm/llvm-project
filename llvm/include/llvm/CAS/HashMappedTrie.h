//===- HashMappedTrie.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_HASHMAPPEDTRIE_H
#define LLVM_CAS_HASHMAPPEDTRIE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <atomic>

namespace llvm {

class MemoryBuffer;

namespace cas {

/// Base class for a lock-free thread-safe hash-mapped trie.
class ThreadSafeHashMappedTrieBase {
public:
  enum : size_t { TrieContentBaseSize = 4 };

private:
  template <class T> struct AllocValueType {
    char Base[TrieContentBaseSize];
    std::aligned_union_t<sizeof(T), T> Content;
  };

protected:
  template <class T> static constexpr size_t getContentAllocSize() {
    return sizeof(AllocValueType<T>);
  }
  template <class T> static constexpr size_t getContentAllocAlign() {
    return alignof(AllocValueType<T>);
  }
  template <class T> static constexpr size_t getContentOffset() {
    return offsetof(AllocValueType<T>, Content);
  }

public:
  void operator delete(void *Ptr) { ::free(Ptr); }

  static constexpr size_t DefaultNumRootBits = 6;
  static constexpr size_t DefaultNumSubtrieBits = 4;

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
    PointerBase() noexcept {}
    PointerBase(PointerBase &&) = default;
    PointerBase(const PointerBase &) = default;
    PointerBase &operator=(PointerBase &&) = default;
    PointerBase &operator=(const PointerBase &) = default;

  private:
    friend class ThreadSafeHashMappedTrieBase;
    explicit PointerBase(void *Content) : P(Content), I(-2u) {}
    PointerBase(void *P, unsigned I, unsigned B) : P(P), I(I), B(B) {}

    bool isHint() const { return I != -1u && I != -2u; }

    void *P = nullptr;
    unsigned I = -1u;
    unsigned B = 0;
  };

  PointerBase find(ArrayRef<uint8_t> Hash) const;

  /// Insert and return the stored content.
  PointerBase
  insert(PointerBase Hint, ArrayRef<uint8_t> Hash,
         function_ref<const uint8_t *(void *Mem, ArrayRef<uint8_t> Hash)>
             Constructor);

  ThreadSafeHashMappedTrieBase() = delete;

  ThreadSafeHashMappedTrieBase(size_t ContentAllocSize,
                               size_t ContentAllocAlign, size_t ContentOffset,
                               Optional<size_t> NumRootBits = None,
                               Optional<size_t> NumSubtrieBits = None);

  /// Destructor, which asserts if there's anything to do. Subclasses should
  /// call \a destroyImpl().
  ///
  /// \pre \a destroyImpl() was already called.
  ~ThreadSafeHashMappedTrieBase();
  void destroyImpl(function_ref<void (void *ValueMem)> Destructor);

  ThreadSafeHashMappedTrieBase(ThreadSafeHashMappedTrieBase &&RHS);

  // Move assignment can be implemented in a thread-safe way if NumRootBits and
  // NumSubtrieBits are stored inside the Root.
  ThreadSafeHashMappedTrieBase &
  operator=(ThreadSafeHashMappedTrieBase &&RHS) = delete;

  // No copy.
  ThreadSafeHashMappedTrieBase(const ThreadSafeHashMappedTrieBase &) = delete;
  ThreadSafeHashMappedTrieBase &
  operator=(const ThreadSafeHashMappedTrieBase &) = delete;

private:
  const unsigned short ContentAllocSize;
  const unsigned short ContentAllocAlign;
  const unsigned short ContentOffset;
  unsigned short NumRootBits;
  unsigned short NumSubtrieBits;
  struct ImplType;
  // ImplPtr is owned by ThreadSafeHashMappedTrieBase and needs to be freed in
  // destoryImpl.
  std::atomic<ImplType *> ImplPtr;
  ImplType &getOrCreateImpl();
  ImplType *getImpl() const;
};

/// Lock-free thread-safe hash-mapped trie.
template <class T, size_t NumHashBytes>
class ThreadSafeHashMappedTrie : ThreadSafeHashMappedTrieBase {
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
    value_type(ArrayRef<uint8_t> Hash, EmplaceTag, ArgsT &&... Args)
        : Hash(makeHash(Hash)), Data(std::forward<ArgsT>(Args)...) {}

    static HashT makeHash(ArrayRef<uint8_t> HashRef) {
      HashT Hash;
      std::copy(HashRef.begin(), HashRef.end(), Hash.data());
      return Hash;
    }
  };

  using ThreadSafeHashMappedTrieBase::operator delete;
  using HashType = HashT;

  using ThreadSafeHashMappedTrieBase::dump;
  using ThreadSafeHashMappedTrieBase::print;

private:
  template <class ValueT> class PointerImpl : PointerBase {
    friend class ThreadSafeHashMappedTrie;

    ValueT *get() const {
      if (void *B = PointerBase::get())
        return reinterpret_cast<ValueT *>(B);
      return nullptr;
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
    PointerImpl(PointerImpl &&) = default;
    PointerImpl(const PointerImpl &) = default;
    PointerImpl &operator=(PointerImpl &&) = default;
    PointerImpl &operator=(const PointerImpl &) = default;

  protected:
    PointerImpl(PointerBase Result) : PointerBase(Result) {}
  };

public:
  class pointer;
  class const_pointer;
  class pointer : public PointerImpl<value_type> {
    friend class ThreadSafeHashMappedTrie;
    friend class const_pointer;

  public:
    pointer() = default;
    pointer(pointer &&) = default;
    pointer(const pointer &) = default;
    pointer &operator=(pointer &&) = default;
    pointer &operator=(const pointer &) = default;

  private:
    pointer(PointerBase Result) : pointer::PointerImpl(Result) {}
  };

  class const_pointer : public PointerImpl<const value_type> {
    friend class ThreadSafeHashMappedTrie;

  public:
    const_pointer() = default;
    const_pointer(const_pointer &&) = default;
    const_pointer(const const_pointer &) = default;
    const_pointer &operator=(const_pointer &&) = default;
    const_pointer &operator=(const const_pointer &) = default;

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
    template <class... ArgsT> value_type &emplace(ArgsT &&... Args) {
      assert(Mem && "Constructor already called, or moved away");
      return assign(::new (Mem)
                        value_type(Hash, typename value_type::EmplaceTag{},
                                   std::forward<ArgsT>(Args)...));
    }

    LazyValueConstructor(LazyValueConstructor &&RHS)
        : Mem(RHS.Mem), Result(RHS.Result), Hash(RHS.Hash) {
      RHS.Mem = nullptr; // Moved away, cannot call.
    }
    ~LazyValueConstructor() {
      assert(!Mem && "Constructor never called!");
    }

  private:
    value_type &assign(value_type *V) {
      Mem = nullptr;
      Result = V;
      return *V;
    }
    friend class ThreadSafeHashMappedTrie;
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
    return pointer(ThreadSafeHashMappedTrieBase::insert(
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
    return insertLazy(Hint, HashedData.Hash,
                                   [&](LazyValueConstructor C) {
                                     C(std::move(HashedData.Data));
                                   });
  }

  pointer insert(const_pointer Hint, const value_type &HashedData) {
    return insertLazy(Hint, HashedData.Hash,
                      [&](LazyValueConstructor C) {
                        C(HashedData.Data);
                      });
  }

  pointer find(ArrayRef<uint8_t> Hash) {
    assert(Hash.size() == std::tuple_size<HashT>::value);
    return ThreadSafeHashMappedTrieBase::find(Hash);
  }

  const_pointer find(ArrayRef<uint8_t> Hash) const {
    assert(Hash.size() == std::tuple_size<HashT>::value);
    return ThreadSafeHashMappedTrieBase::find(Hash);
  }

  ThreadSafeHashMappedTrie(Optional<size_t> NumRootBits = None,
                           Optional<size_t> NumSubtrieBits = None)
      : ThreadSafeHashMappedTrieBase(getContentAllocSize<value_type>(),
                                     getContentAllocAlign<value_type>(),
                                     getContentOffset<value_type>(),
                                     NumRootBits, NumSubtrieBits) {}

  ~ThreadSafeHashMappedTrie() {
    if (std::is_trivially_destructible<value_type>::value)
      this->destroyImpl(nullptr);
    else
      this->destroyImpl(
          [](void *P) { static_cast<value_type *>(P)->~value_type(); });
  }

  // Move constructor okay.
  ThreadSafeHashMappedTrie(ThreadSafeHashMappedTrie &&) = default;

  // No move assignment or any copy.
  ThreadSafeHashMappedTrie &operator=(ThreadSafeHashMappedTrie &&) = delete;
  ThreadSafeHashMappedTrie(const ThreadSafeHashMappedTrie &) = delete;
  ThreadSafeHashMappedTrie &
  operator=(const ThreadSafeHashMappedTrie &) = delete;
};

} // namespace cas
} // namespace llvm

#endif // LLVM_CAS_HASHMAPPEDTRIE_H
