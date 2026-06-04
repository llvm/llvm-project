//===- llvm/ADT/DenseMap.h - Dense probed hash table ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the DenseMap class.
///
/// The hash table is linear-probing open addressing with tombstone-free
/// deletion (Knuth TAOCP 6.4 Algorithm R), power-of-two capacity, and a 0.75
/// maximum load factor. No sentinel key. Occupancy is stored in a packed
/// 1-bit-per-bucket "used" array.
///
/// `SmallDenseMap` adds an inline small buffer optimization.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_DENSEMAP_H
#define LLVM_ADT_DENSEMAP_H

#include "llvm/ADT/ADL.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/EpochTracker.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemAlloc.h"
#include "llvm/Support/ReverseIteration.h"
#include "llvm/Support/type_traits.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <new>
#include <type_traits>
#include <utility>

namespace llvm {

namespace detail {

// We extend a pair to allow users to override the bucket type with their own
// implementation without requiring two members.
template <typename KeyT, typename ValueT>
struct DenseMapPair : std::pair<KeyT, ValueT> {
  using std::pair<KeyT, ValueT>::pair;

  KeyT &getFirst() { return std::pair<KeyT, ValueT>::first; }
  const KeyT &getFirst() const { return std::pair<KeyT, ValueT>::first; }
  ValueT &getSecond() { return std::pair<KeyT, ValueT>::second; }
  const ValueT &getSecond() const { return std::pair<KeyT, ValueT>::second; }
};

} // end namespace detail

namespace densemap::detail {
using UsedT = uint32_t;

// Number of used words backing N buckets where N is zero or a power of two.
constexpr size_t usedWords(size_t N) {
  assert((N == 0 || isPowerOf2_64(N)) &&
         "bucket count must be zero or a power of two");
  return (N + 31) / 32;
}

inline bool used(const UsedT *U, size_t I) {
  return (U[I >> 5] >> (I & 31)) & 1;
}
inline void setUsed(UsedT *U, size_t I) { U[I >> 5] |= UsedT(1) << (I & 31); }
inline void unsetUsed(UsedT *U, size_t I) {
  U[I >> 5] &= ~(UsedT(1) << (I & 31));
}

// Invoke Func(I) for each occupied bucket index I in [0, N). Set always_inline;
// otherwise, for a heavy caller such as moveFrom's rehash, the inliner can
// leave it out of line and the per-element call dwarfs the work.
template <typename Fn>
LLVM_ATTRIBUTE_ALWAYS_INLINE void forEachUsed(const UsedT *U, unsigned N,
                                              Fn Func) {
  const unsigned NW = usedWords(N);
  for (unsigned W = 0; W != NW; ++W) {
    UsedT Bits = U[W];
    while (Bits) {
      Func((W << 5) + llvm::countr_zero(Bits));
      Bits &= Bits - 1;
    }
  }
}

// Buckets and the used array share one allocation: the bucket array first, then
// the used words.  NumBuckets is a power of two >= 4, so the bucket region size
// is a multiple of sizeof(UsedT) and the trailing used words are aligned.
template <typename BucketT> constexpr size_t allocAlign() {
  return std::max(alignof(BucketT), alignof(UsedT));
}
template <typename BucketT> size_t allocBytes(unsigned Num) {
  return sizeof(BucketT) * static_cast<size_t>(Num) +
         usedWords(Num) * sizeof(UsedT);
}

} // namespace densemap::detail

// Befriended below so DenseMapBase can expose its bucket-relocation callback
// erase to ValueHandleBase, the only caller that caches bucket pointers.
class ValueHandleBase;

template <typename KeyT, typename ValueT,
          typename KeyInfoT = DenseMapInfo<KeyT>,
          typename Bucket = llvm::detail::DenseMapPair<KeyT, ValueT>,
          bool IsConst = false>
class DenseMapIterator;

template <typename DerivedT, typename KeyT, typename ValueT, typename KeyInfoT,
          typename BucketT>
class DenseMapBase : public DebugEpochBase {
  template <typename T>
  using const_arg_type_t = typename const_pointer_or_const_ref<T>::type;

  using UsedT = llvm::densemap::detail::UsedT;

public:
  using size_type = unsigned;
  using key_type = KeyT;
  using mapped_type = ValueT;
  using value_type = BucketT;

  using iterator = DenseMapIterator<KeyT, ValueT, KeyInfoT, BucketT>;
  using const_iterator =
      DenseMapIterator<KeyT, ValueT, KeyInfoT, BucketT, true>;

  [[nodiscard]] inline iterator begin() {
    return iterator::makeBegin(getBuckets(), getUsed(), getNumBuckets(),
                               empty(), *this);
  }
  [[nodiscard]] inline iterator end() {
    return iterator::makeEnd(getBuckets(), getUsed(), getNumBuckets(), *this);
  }
  [[nodiscard]] inline const_iterator begin() const {
    return const_iterator::makeBegin(getBuckets(), getUsed(), getNumBuckets(),
                                     empty(), *this);
  }
  [[nodiscard]] inline const_iterator end() const {
    return const_iterator::makeEnd(getBuckets(), getUsed(), getNumBuckets(),
                                   *this);
  }

  // Return an iterator to iterate over keys in the map.
  [[nodiscard]] inline auto keys() {
    return map_range(*this, [](const BucketT &P) { return P.getFirst(); });
  }

  // Return an iterator to iterate over values in the map.
  [[nodiscard]] inline auto values() {
    return map_range(*this, [](const BucketT &P) { return P.getSecond(); });
  }

  [[nodiscard]] inline auto keys() const {
    return map_range(*this, [](const BucketT &P) { return P.getFirst(); });
  }

  [[nodiscard]] inline auto values() const {
    return map_range(*this, [](const BucketT &P) { return P.getSecond(); });
  }

  [[nodiscard]] bool empty() const { return getNumEntries() == 0; }
  [[nodiscard]] unsigned size() const { return getNumEntries(); }

  /// Grow the densemap so that it can contain at least \p NumEntries items
  /// before resizing again.
  void reserve(size_type NumEntries) {
    auto NumBuckets = getMinBucketToReserveForEntries(NumEntries);
    incrementEpoch();
    if (NumBuckets > getNumBuckets())
      grow(NumBuckets);
  }

  void clear() {
    incrementEpoch();
    if (getNumEntries() == 0)
      return;

    // If the capacity of the array is huge, and the # elements used is small,
    // shrink the array.
    if (getNumEntries() * 4 < getNumBuckets() && getNumBuckets() > 64) {
      shrink_and_clear();
      return;
    }

    destroyAll();
    std::memset(getUsed(), 0,
                llvm::densemap::detail::usedWords(getNumBuckets()) *
                    sizeof(UsedT));
    setNumEntries(0);
  }

  void shrink_and_clear() {
    auto [Reallocate, NewNumBuckets] = derived().planShrinkAndClear();
    destroyAll();
    if (!Reallocate) {
      initEmpty();
      return;
    }
    derived().deallocateBuckets();
    initWithExactBucketCount(NewNumBuckets);
  }

  /// Return true if the specified key is in the map, false otherwise.
  [[nodiscard]] bool contains(const_arg_type_t<KeyT> Val) const {
    return doFind(Val) != nullptr;
  }

  /// Return 1 if the specified key is in the map, 0 otherwise.
  [[nodiscard]] size_type count(const_arg_type_t<KeyT> Val) const {
    return contains(Val) ? 1 : 0;
  }

  [[nodiscard]] iterator find(const_arg_type_t<KeyT> Val) {
    return find_as(Val);
  }
  [[nodiscard]] const_iterator find(const_arg_type_t<KeyT> Val) const {
    return find_as(Val);
  }

  /// Alternate version of find() which allows a different, and possibly
  /// less expensive, key type.
  /// The DenseMapInfo is responsible for supplying methods
  /// getHashValue(LookupKeyT) and isEqual(LookupKeyT, KeyT) for each key
  /// type used.
  template <class LookupKeyT>
  [[nodiscard]] iterator find_as(const LookupKeyT &Val) {
    if (BucketT *Bucket = doFind(Val))
      return makeIterator(Bucket);
    return end();
  }
  template <class LookupKeyT>
  [[nodiscard]] const_iterator find_as(const LookupKeyT &Val) const {
    if (const BucketT *Bucket = doFind(Val))
      return makeConstIterator(Bucket);
    return end();
  }

  /// Return the entry for the specified key, or a default constructed value if
  /// no such entry exists.
  [[nodiscard]] ValueT lookup(const_arg_type_t<KeyT> Val) const {
    if (const BucketT *Bucket = doFind(Val))
      return Bucket->getSecond();
    return ValueT();
  }

  // Return the entry with the specified key, or \p Default. This variant is
  // useful, because `lookup` cannot be used with non-default-constructible
  // values.
  template <typename U = std::remove_cv_t<ValueT>>
  [[nodiscard]] ValueT lookup_or(const_arg_type_t<KeyT> Val,
                                 U &&Default) const {
    if (const BucketT *Bucket = doFind(Val))
      return Bucket->getSecond();
    return Default;
  }

  /// Return the entry for the specified key, or abort if no such entry exists.
  [[nodiscard]] ValueT &at(const_arg_type_t<KeyT> Val) {
    auto Iter = this->find(std::move(Val));
    assert(Iter != this->end() && "DenseMap::at failed due to a missing key");
    return Iter->second;
  }

  /// Return the entry for the specified key, or abort if no such entry exists.
  [[nodiscard]] const ValueT &at(const_arg_type_t<KeyT> Val) const {
    auto Iter = this->find(std::move(Val));
    assert(Iter != this->end() && "DenseMap::at failed due to a missing key");
    return Iter->second;
  }

  // Inserts key,value pair into the map if the key isn't already in the map.
  // If the key is already in the map, it returns false and doesn't update the
  // value.
  std::pair<iterator, bool> insert(const std::pair<KeyT, ValueT> &KV) {
    return try_emplace_impl(KV.first, KV.second);
  }

  // Inserts key,value pair into the map if the key isn't already in the map.
  // If the key is already in the map, it returns false and doesn't update the
  // value.
  std::pair<iterator, bool> insert(std::pair<KeyT, ValueT> &&KV) {
    return try_emplace_impl(std::move(KV.first), std::move(KV.second));
  }

  // Inserts key,value pair into the map if the key isn't already in the map.
  // The value is constructed in-place if the key is not in the map, otherwise
  // it is not moved.
  template <typename... Ts>
  std::pair<iterator, bool> try_emplace(KeyT &&Key, Ts &&...Args) {
    return try_emplace_impl(std::move(Key), std::forward<Ts>(Args)...);
  }

  // Inserts key,value pair into the map if the key isn't already in the map.
  // The value is constructed in-place if the key is not in the map, otherwise
  // it is not moved.
  template <typename... Ts>
  std::pair<iterator, bool> try_emplace(const KeyT &Key, Ts &&...Args) {
    return try_emplace_impl(Key, std::forward<Ts>(Args)...);
  }

  /// Alternate version of insert() which allows a different, and possibly
  /// less expensive, key type.
  /// The DenseMapInfo is responsible for supplying methods
  /// getHashValue(LookupKeyT) and isEqual(LookupKeyT, KeyT) for each key
  /// type used.
  template <typename LookupKeyT>
  std::pair<iterator, bool> insert_as(std::pair<KeyT, ValueT> &&KV,
                                      const LookupKeyT &Val) {
    BucketT *TheBucket;
    if (LookupBucketFor(Val, TheBucket))
      return {makeIterator(TheBucket), false}; // Already in map.

    // Otherwise, insert the new element.
    TheBucket = findBucketForInsertion(Val, TheBucket);
    ::new (&TheBucket->getFirst()) KeyT(std::move(KV.first));
    ::new (&TheBucket->getSecond()) ValueT(std::move(KV.second));
    return {makeIterator(TheBucket), true};
  }

  /// Range insertion of pairs.
  template <typename InputIt> void insert(InputIt I, InputIt E) {
    for (; I != E; ++I)
      insert(*I);
  }

  /// Inserts range of 'std::pair<KeyT, ValueT>' values into the map.
  template <typename Range> void insert_range(Range &&R) {
    insert(adl_begin(R), adl_end(R));
  }

  template <typename V>
  std::pair<iterator, bool> insert_or_assign(const KeyT &Key, V &&Val) {
    auto Ret = try_emplace(Key, std::forward<V>(Val));
    if (!Ret.second)
      Ret.first->second = std::forward<V>(Val);
    return Ret;
  }

  template <typename V>
  std::pair<iterator, bool> insert_or_assign(KeyT &&Key, V &&Val) {
    auto Ret = try_emplace(std::move(Key), std::forward<V>(Val));
    if (!Ret.second)
      Ret.first->second = std::forward<V>(Val);
    return Ret;
  }

  template <typename... Ts>
  std::pair<iterator, bool> emplace_or_assign(const KeyT &Key, Ts &&...Args) {
    auto Ret = try_emplace(Key, std::forward<Ts>(Args)...);
    if (!Ret.second)
      Ret.first->second = ValueT(std::forward<Ts>(Args)...);
    return Ret;
  }

  template <typename... Ts>
  std::pair<iterator, bool> emplace_or_assign(KeyT &&Key, Ts &&...Args) {
    auto Ret = try_emplace(std::move(Key), std::forward<Ts>(Args)...);
    if (!Ret.second)
      Ret.first->second = ValueT(std::forward<Ts>(Args)...);
    return Ret;
  }

  void eraseFromFilledBucket(BucketT *TheBucket) {
    eraseFromFilledBucket(TheBucket, [](BucketT &) {});
  }

  bool erase(const KeyT &Val) {
    BucketT *TheBucket = doFind(Val);
    if (!TheBucket)
      return false; // not in map.

    eraseFromFilledBucket(TheBucket);
    return true;
  }
  void erase(iterator I) { eraseFromFilledBucket(&*I); }

  /// Remove entries that match the given predicate. \p Pred is invoked
  /// with a reference to each live bucket and must not access the map being
  /// modified. This is the safe replacement for erase-while-iterating.
  ///
  /// Returns whether anything was removed. If so, all iterators and references
  /// into the map are invalidated.
  template <typename Predicate> bool remove_if(Predicate Pred) {
    UsedT *U = getUsed();
    unsigned NumBuckets = getNumBuckets();
    BucketT *B = getBuckets();
    bool Removed = false;
    for (unsigned I = 0; I != NumBuckets; ++I) {
      if (!llvm::densemap::detail::used(U, I))
        continue;
      if (Pred(B[I])) {
        B[I].getSecond().~ValueT();
        B[I].getFirst().~KeyT();
        llvm::densemap::detail::unsetUsed(U, I);
        decrementNumEntries();
        Removed = true;
      }
    }
    if (Removed) {
      incrementEpoch();
      this->grow(NumBuckets);
    }
    return Removed;
  }

  ValueT &operator[](const KeyT &Key) {
    return lookupOrInsertIntoBucket(Key).first->second;
  }

  ValueT &operator[](KeyT &&Key) {
    return lookupOrInsertIntoBucket(std::move(Key)).first->second;
  }

  /// Return true if the specified pointer points somewhere into the DenseMap's
  /// array of buckets (i.e. either to a key or value in the DenseMap).
  [[nodiscard]] bool isPointerIntoBucketsArray(const void *Ptr) const {
    return Ptr >= getBuckets() && Ptr < getBucketsEnd();
  }

  /// getPointerIntoBucketsArray() - Return an opaque pointer into the buckets
  /// array.  In conjunction with the previous method, this can be used to
  /// determine whether an insertion caused the DenseMap to reallocate.
  [[nodiscard]] const void *getPointerIntoBucketsArray() const {
    return getBuckets();
  }

  void swap(DerivedT &RHS) {
    this->incrementEpoch();
    RHS.incrementEpoch();
    derived().swapImpl(RHS);
  }

protected:
  DenseMapBase() = default;

  struct ExactBucketCount {};

  // A snapshot of the three fields the hot lookup paths need. Fetching them
  // together lets SmallDenseMap test its Small discriminator once rather than
  // once per accessor; for plain DenseMap it is three member loads either way.
  struct Rep {
    const BucketT *Buckets;
    const UsedT *Used;
    unsigned NumBuckets;
  };

  void initWithExactBucketCount(unsigned NewNumBuckets) {
    if (derived().allocateBuckets(NewNumBuckets))
      initEmpty();
    else
      setNumEntries(0);
  }

  void destroyAll() {
    // No need to iterate through the buckets if both KeyT and ValueT are
    // trivially destructible.
    if constexpr (std::is_trivially_destructible_v<KeyT> &&
                  std::is_trivially_destructible_v<ValueT>)
      return;

    if (getNumBuckets() == 0) // Nothing to do.
      return;

    BucketT *B = getBuckets();
    const UsedT *U = getUsed();
    const unsigned E = getNumBuckets();
    llvm::densemap::detail::forEachUsed(U, E, [&](unsigned I) {
      B[I].getSecond().~ValueT();
      B[I].getFirst().~KeyT();
    });
  }

  void initEmpty() {
    static_assert(std::is_base_of_v<DenseMapBase, DerivedT>,
                  "Must pass the derived type to this template!");
    setNumEntries(0);

    assert((getNumBuckets() & (getNumBuckets() - 1)) == 0 &&
           "# initial buckets must be a power of two!");
    std::memset(getUsed(), 0,
                llvm::densemap::detail::usedWords(getNumBuckets()) *
                    sizeof(UsedT));
  }

  /// Returns the number of buckets to allocate to ensure that the DenseMap can
  /// accommodate \p NumEntries without need to grow().
  unsigned getMinBucketToReserveForEntries(unsigned NumEntries) {
    // Ensure that "NumEntries * 4 < NumBuckets * 3"
    if (NumEntries == 0)
      return 0;
    // +1 is required because of the strict inequality.
    // For example, if NumEntries is 48, we need to return 128.
    return NextPowerOf2(NumEntries * 4 / 3 + 1);
  }

  // Move key/value from Other to *this.
  // Other is left in a valid but empty state.
  LLVM_ATTRIBUTE_NOINLINE void moveFrom(DerivedT &Other) {
    assert(getNumEntries() == 0 && "moveFrom requires an empty destination");
    BucketT *OtherB = Other.getBuckets();
    UsedT *OtherU = Other.getUsed();
    const unsigned E = Other.getNumBuckets();
    UsedT *U = getUsed();
    BucketT *B = getBuckets();
    const unsigned Mask = getNumBuckets() - 1;
    llvm::densemap::detail::forEachUsed(OtherU, E, [&](unsigned I) {
      // Find the first empty slot on this key's probe chain; there is no equal
      // key in the destination, so nothing to compare against.
      unsigned BucketNo = KeyInfoT::getHashValue(OtherB[I].getFirst()) & Mask;
      while (llvm::densemap::detail::used(U, BucketNo))
        BucketNo = (BucketNo + 1) & Mask;
      BucketT *DestBucket = B + BucketNo;
      ::new (&DestBucket->getFirst()) KeyT(std::move(OtherB[I].getFirst()));
      ::new (&DestBucket->getSecond()) ValueT(std::move(OtherB[I].getSecond()));
      llvm::densemap::detail::setUsed(U, BucketNo);

      // Free the moved-out key/value.
      OtherB[I].getSecond().~ValueT();
      OtherB[I].getFirst().~KeyT();
    });
    setNumEntries(Other.getNumEntries());
    Other.derived().kill();
  }

  LLVM_ATTRIBUTE_NOINLINE void copyFrom(const DerivedT &other) {
    this->destroyAll();
    derived().deallocateBuckets();
    setNumEntries(0);
    if (!derived().allocateBuckets(other.getNumBuckets())) {
      // The bucket list is empty.  No work to do.
      return;
    }

    assert(&other != this);
    assert(getNumBuckets() == other.getNumBuckets());

    setNumEntries(other.getNumEntries());

    BucketT *Buckets = getBuckets();
    const BucketT *OtherBuckets = other.getBuckets();
    const unsigned NumBuckets = getNumBuckets();
    UsedT *U = getUsed();
    const UsedT *OtherU = other.getUsed();
    std::memcpy(U, OtherU,
                llvm::densemap::detail::usedWords(NumBuckets) * sizeof(UsedT));
    if constexpr (std::is_trivially_copyable_v<KeyT> &&
                  std::is_trivially_copyable_v<ValueT>) {
      memcpy(reinterpret_cast<void *>(Buckets), OtherBuckets,
             NumBuckets * sizeof(BucketT));
    } else {
      llvm::densemap::detail::forEachUsed(U, NumBuckets, [&](unsigned I) {
        ::new (&Buckets[I].getFirst()) KeyT(OtherBuckets[I].getFirst());
        ::new (&Buckets[I].getSecond()) ValueT(OtherBuckets[I].getSecond());
      });
    }
  }

private:
  // ValueHandleBase caches pointers into the bucket array, so it needs the
  // callback erase below to fix them up as entries shift. It is the only
  // intended caller; do not add new ones.
  friend class ValueHandleBase;

  /// Erase the entry at \p TheBucket and close the resulting hole via Knuth
  /// TAOCP 6.4 Algorithm R. For callers that cache pointers into the bucket
  /// array, call \p OnMoved per shifted bucket.
  template <typename OnMovedT>
  LLVM_ATTRIBUTE_NOINLINE void eraseFromFilledBucket(BucketT *TheBucket,
                                                     OnMovedT &&OnMoved) {
    incrementEpoch();
    TheBucket->getSecond().~ValueT();
    TheBucket->getFirst().~KeyT();
    decrementNumEntries();

    BucketT *BucketsPtr = getBuckets();
    UsedT *U = getUsed();
    const unsigned Mask = getNumBuckets() - 1;
    unsigned I = TheBucket - BucketsPtr;
    unsigned J = I;
    while (true) {
      J = (J + 1) & Mask;
      BucketT &BJ = BucketsPtr[J];
      if (!llvm::densemap::detail::used(U, J))
        break;
      auto Ideal = KeyInfoT::getHashValue(BJ.getFirst());
      // If the hole (I) lies on the linear-probe chain from the home bucket
      // (Ideal) to J, shift J into the hole and make J the new hole.
      if (((I - Ideal) & Mask) < ((J - Ideal) & Mask)) {
        BucketT &BI = BucketsPtr[I];
        ::new (&BI.getFirst()) KeyT(std::move(BJ.getFirst()));
        ::new (&BI.getSecond()) ValueT(std::move(BJ.getSecond()));
        BJ.getSecond().~ValueT();
        BJ.getFirst().~KeyT();
        OnMoved(BI);
        I = J;
      }
    }
    llvm::densemap::detail::unsetUsed(U, I);
  }

  /// Erase \p Val and close the resulting hole by potentially shifting other
  /// entries into it. For callers that cache pointers into the bucket array,
  /// call \p OnMoved per shifted bucket.
  template <typename OnMovedT> bool erase(const KeyT &Val, OnMovedT &&OnMoved) {
    BucketT *TheBucket = doFind(Val);
    if (!TheBucket)
      return false;
    eraseFromFilledBucket(TheBucket, std::forward<OnMovedT>(OnMoved));
    return true;
  }

  DerivedT &derived() { return *static_cast<DerivedT *>(this); }
  const DerivedT &derived() const {
    return *static_cast<const DerivedT *>(this);
  }

  template <typename KeyArgT, typename... Ts>
  std::pair<BucketT *, bool> lookupOrInsertIntoBucket(KeyArgT &&Key,
                                                      Ts &&...Args) {
    BucketT *TheBucket = nullptr;
    if (LookupBucketFor(Key, TheBucket))
      return {TheBucket, false}; // Already in the map.

    // Otherwise, insert the new element.
    TheBucket = findBucketForInsertion(Key, TheBucket);
    ::new (&TheBucket->getFirst()) KeyT(std::forward<KeyArgT>(Key));
    ::new (&TheBucket->getSecond()) ValueT(std::forward<Ts>(Args)...);
    return {TheBucket, true};
  }

  template <typename KeyArgT, typename... Ts>
  std::pair<iterator, bool> try_emplace_impl(KeyArgT &&Key, Ts &&...Args) {
    auto [Bucket, Inserted] = lookupOrInsertIntoBucket(
        std::forward<KeyArgT>(Key), std::forward<Ts>(Args)...);
    return {makeIterator(Bucket), Inserted};
  }

  iterator makeIterator(BucketT *TheBucket) {
    return iterator::makeIterator(TheBucket, getBuckets(), getUsed(),
                                  getNumBuckets(), *this);
  }

  const_iterator makeConstIterator(const BucketT *TheBucket) const {
    return const_iterator::makeIterator(TheBucket, getBuckets(), getUsed(),
                                        getNumBuckets(), *this);
  }

  unsigned getNumEntries() const { return derived().getNumEntries(); }

  void setNumEntries(unsigned Num) { derived().setNumEntries(Num); }

  void incrementNumEntries() { setNumEntries(getNumEntries() + 1); }

  void decrementNumEntries() { setNumEntries(getNumEntries() - 1); }

  const BucketT *getBuckets() const { return derived().getBuckets(); }

  BucketT *getBuckets() { return derived().getBuckets(); }

  Rep getRep() const { return derived().getRep(); }

  const UsedT *getUsed() const { return derived().getUsed(); }

  UsedT *getUsed() { return derived().getUsed(); }

  unsigned getNumBuckets() const { return derived().getNumBuckets(); }

  BucketT *getBucketsEnd() { return getBuckets() + getNumBuckets(); }

  const BucketT *getBucketsEnd() const {
    return getBuckets() + getNumBuckets();
  }

  LLVM_ATTRIBUTE_NOINLINE void grow(unsigned MinNumBuckets) {
    unsigned NumBuckets = DerivedT::roundUpNumBuckets(MinNumBuckets);
    DerivedT Tmp(NumBuckets, ExactBucketCount{});
    Tmp.moveFrom(derived());
    if (derived().maybeMoveFast(std::move(Tmp)))
      return;
    initWithExactBucketCount(NumBuckets);
    moveFrom(Tmp);
  }

  template <typename LookupKeyT>
  BucketT *findBucketForInsertion(const LookupKeyT &Lookup,
                                  BucketT *TheBucket) {
    incrementEpoch();

    // Grow the table if the load factor would exceed 3/4 after insertion.
    // Linear probing with gap-closing deletion (Knuth Algorithm R) keeps
    // every chain compact and bounded by the table's empty-bucket count,
    // so no tombstone-driven resize is needed.
    unsigned NewNumEntries = getNumEntries() + 1;
    unsigned NumBuckets = getNumBuckets();
    if (LLVM_UNLIKELY(NewNumEntries * 4 >= NumBuckets * 3)) {
      this->grow(NumBuckets * 2);
      LookupBucketFor(Lookup, TheBucket);
    }
    assert(TheBucket);

    // Mark used. The caller will placement-construct the raw key/value.
    llvm::densemap::detail::setUsed(getUsed(), TheBucket - getBuckets());

    // Only update the state after we've grown our bucket space appropriately
    // so that when growing buckets we have self-consistent entry count.
    incrementNumEntries();
    return TheBucket;
  }

  template <typename LookupKeyT>
  const BucketT *doFind(const LookupKeyT &Val) const {
    auto [BucketsPtr, U, NumBuckets] = getRep();
    if (NumBuckets == 0)
      return nullptr;

    const unsigned Mask = NumBuckets - 1;
    unsigned BucketNo = KeyInfoT::getHashValue(Val) & Mask;
    while (true) {
      // An empty bucket terminates the probe: the key isn't in the map.
      if (LLVM_LIKELY(!llvm::densemap::detail::used(U, BucketNo)))
        return nullptr;
      const BucketT *Bucket = BucketsPtr + BucketNo;
      if (LLVM_LIKELY(KeyInfoT::isEqual(Val, Bucket->getFirst())))
        return Bucket;

      // Hash collision: continue linear probing.
      BucketNo = (BucketNo + 1) & Mask;
    }
  }

  template <typename LookupKeyT> BucketT *doFind(const LookupKeyT &Val) {
    return const_cast<BucketT *>(
        static_cast<const DenseMapBase *>(this)->doFind(Val));
  }

  /// Lookup the appropriate bucket for Val, returning it in FoundBucket. If the
  /// bucket contains the key and a value, this returns true, otherwise it
  /// returns a bucket with an empty marker and returns false.
  template <typename LookupKeyT>
  bool LookupBucketFor(const LookupKeyT &Val, BucketT *&FoundBucket) {
    auto [CBuckets, U, NumBuckets] = getRep();
    if (NumBuckets == 0) {
      FoundBucket = nullptr;
      return false;
    }
    // getRep() yields const pointers; this object is non-const, so recovering
    // a mutable bucket pointer is safe (mirrors the non-const getBuckets()).
    BucketT *BucketsPtr = const_cast<BucketT *>(CBuckets);

    const unsigned Mask = NumBuckets - 1;
    unsigned BucketNo = KeyInfoT::getHashValue(Val) & Mask;
    while (true) {
      BucketT *ThisBucket = BucketsPtr + BucketNo;
      // If we found an empty bucket, the key doesn't exist in the set.
      // Return it as the insertion point.
      if (LLVM_LIKELY(!llvm::densemap::detail::used(U, BucketNo))) {
        FoundBucket = ThisBucket;
        return false;
      }

      // Found Val's bucket?  If so, return it.
      if (LLVM_LIKELY(KeyInfoT::isEqual(Val, ThisBucket->getFirst()))) {
        FoundBucket = ThisBucket;
        return true;
      }

      // Hash collision: continue linear probing.
      BucketNo = (BucketNo + 1) & Mask;
    }
  }

public:
  /// Return the approximate size (in bytes) of the actual map.
  /// This is just the raw memory used by DenseMap.
  /// If entries are pointers to objects, the size of the referenced objects
  /// are not included.
  [[nodiscard]] size_t getMemorySize() const {
    return llvm::densemap::detail::allocBytes<BucketT>(getNumBuckets());
  }
};

/// Equality comparison for DenseMap.
///
/// Iterates over elements of LHS confirming that each (key, value) pair in LHS
/// is also in RHS, and that no additional pairs are in RHS.
/// Equivalent to N calls to RHS.find and N value comparisons. Amortized
/// complexity is linear, worst case is O(N^2) (if every hash collides).
template <typename DerivedT, typename KeyT, typename ValueT, typename KeyInfoT,
          typename BucketT>
[[nodiscard]] bool
operator==(const DenseMapBase<DerivedT, KeyT, ValueT, KeyInfoT, BucketT> &LHS,
           const DenseMapBase<DerivedT, KeyT, ValueT, KeyInfoT, BucketT> &RHS) {
  if (LHS.size() != RHS.size())
    return false;

  for (auto &KV : LHS) {
    auto I = RHS.find(KV.first);
    if (I == RHS.end() || I->second != KV.second)
      return false;
  }

  return true;
}

/// Inequality comparison for DenseMap.
///
/// Equivalent to !(LHS == RHS). See operator== for performance notes.
template <typename DerivedT, typename KeyT, typename ValueT, typename KeyInfoT,
          typename BucketT>
[[nodiscard]] bool
operator!=(const DenseMapBase<DerivedT, KeyT, ValueT, KeyInfoT, BucketT> &LHS,
           const DenseMapBase<DerivedT, KeyT, ValueT, KeyInfoT, BucketT> &RHS) {
  return !(LHS == RHS);
}

template <typename KeyT, typename ValueT,
          typename KeyInfoT = DenseMapInfo<KeyT>,
          typename BucketT = llvm::detail::DenseMapPair<KeyT, ValueT>>
class DenseMap : public DenseMapBase<DenseMap<KeyT, ValueT, KeyInfoT, BucketT>,
                                     KeyT, ValueT, KeyInfoT, BucketT> {
  friend class DenseMapBase<DenseMap, KeyT, ValueT, KeyInfoT, BucketT>;

  // Lift some types from the dependent base class into this class for
  // simplicity of referring to them.
  using BaseT = DenseMapBase<DenseMap, KeyT, ValueT, KeyInfoT, BucketT>;
  using UsedT = llvm::densemap::detail::UsedT;

  BucketT *Buckets = nullptr;
  UsedT *Used = nullptr;
  unsigned NumEntries = 0;
  unsigned NumBuckets = 0;

  explicit DenseMap(unsigned NumBuckets, typename BaseT::ExactBucketCount) {
    this->initWithExactBucketCount(NumBuckets);
  }

public:
  /// Create a DenseMap with an optional \p NumElementsToReserve to guarantee
  /// that this number of elements can be inserted in the map without grow().
  explicit DenseMap(unsigned NumElementsToReserve = 0)
      : DenseMap(BaseT::getMinBucketToReserveForEntries(NumElementsToReserve),
                 typename BaseT::ExactBucketCount{}) {}

  DenseMap(const DenseMap &other) : DenseMap() { this->copyFrom(other); }

  DenseMap(DenseMap &&other) : DenseMap() { this->swap(other); }

  template <typename InputIt>
  DenseMap(const InputIt &I, const InputIt &E) : DenseMap(std::distance(I, E)) {
    this->insert(I, E);
  }

  template <typename RangeT>
  DenseMap(llvm::from_range_t, const RangeT &Range)
      : DenseMap(adl_begin(Range), adl_end(Range)) {}

  DenseMap(std::initializer_list<typename BaseT::value_type> Vals)
      : DenseMap(Vals.begin(), Vals.end()) {}

  ~DenseMap() {
    this->destroyAll();
    deallocateBuckets();
  }

  DenseMap &operator=(const DenseMap &other) {
    if (&other != this)
      this->copyFrom(other);
    return *this;
  }

  DenseMap &operator=(DenseMap &&other) {
    this->destroyAll();
    deallocateBuckets();
    this->initWithExactBucketCount(0);
    this->swap(other);
    return *this;
  }

private:
  void swapImpl(DenseMap &RHS) {
    std::swap(Buckets, RHS.Buckets);
    std::swap(Used, RHS.Used);
    std::swap(NumEntries, RHS.NumEntries);
    std::swap(NumBuckets, RHS.NumBuckets);
  }

  unsigned getNumEntries() const { return NumEntries; }

  void setNumEntries(unsigned Num) { NumEntries = Num; }

  BucketT *getBuckets() const { return Buckets; }

  typename BaseT::Rep getRep() const { return {Buckets, Used, NumBuckets}; }

  UsedT *getUsed() const { return Used; }

  unsigned getNumBuckets() const { return NumBuckets; }

  void deallocateBuckets() {
    if (NumBuckets == 0)
      return;
    deallocate_buffer(Buckets,
                      llvm::densemap::detail::allocBytes<BucketT>(NumBuckets),
                      llvm::densemap::detail::allocAlign<BucketT>());
    Buckets = nullptr;
    Used = nullptr;
    NumBuckets = 0;
  }

  bool allocateBuckets(unsigned Num) {
    NumBuckets = Num;
    if (NumBuckets == 0) {
      Buckets = nullptr;
      Used = nullptr;
      return false;
    }

    auto *Storage = static_cast<char *>(
        allocate_buffer(llvm::densemap::detail::allocBytes<BucketT>(NumBuckets),
                        llvm::densemap::detail::allocAlign<BucketT>()));
    Buckets = reinterpret_cast<BucketT *>(Storage);
    // NumBuckets is a power of two >= 64, so the used array trailing the
    // buckets is aligned.
    assert(sizeof(BucketT) * NumBuckets % alignof(UsedT) == 0 &&
           "used array would be misaligned");
    Used = reinterpret_cast<UsedT *>(Storage + sizeof(BucketT) * NumBuckets);
    return true;
  }

  // Put the zombie instance in a known good state after a move.
  // deallocateBuckets() already resets to the empty state.
  void kill() { deallocateBuckets(); }

  static unsigned roundUpNumBuckets(unsigned MinNumBuckets) {
    return std::max(64u,
                    static_cast<unsigned>(NextPowerOf2(MinNumBuckets - 1)));
  }

  bool maybeMoveFast(DenseMap &&Other) {
    swapImpl(Other);
    return true;
  }

  // Plan how to shrink the bucket table.  Return:
  // - {false, 0} to reuse the existing bucket table
  // - {true, N} to reallocate a bucket table with N entries
  std::pair<bool, unsigned> planShrinkAndClear() const {
    unsigned NewNumBuckets = 0;
    if (NumEntries)
      NewNumBuckets = std::max(64u, 1u << (Log2_32_Ceil(NumEntries) + 1));
    if (NewNumBuckets == NumBuckets)
      return {false, 0};          // Reuse.
    return {true, NewNumBuckets}; // Reallocate.
  }
};

template <typename KeyT, typename ValueT, unsigned InlineBuckets = 4,
          typename KeyInfoT = DenseMapInfo<KeyT>,
          typename BucketT = llvm::detail::DenseMapPair<KeyT, ValueT>>
class SmallDenseMap
    : public DenseMapBase<
          SmallDenseMap<KeyT, ValueT, InlineBuckets, KeyInfoT, BucketT>, KeyT,
          ValueT, KeyInfoT, BucketT> {
  friend class DenseMapBase<SmallDenseMap, KeyT, ValueT, KeyInfoT, BucketT>;

  // Lift some types from the dependent base class into this class for
  // simplicity of referring to them.
  using BaseT = DenseMapBase<SmallDenseMap, KeyT, ValueT, KeyInfoT, BucketT>;
  using UsedT = llvm::densemap::detail::UsedT;

  static_assert(isPowerOf2_64(InlineBuckets),
                "InlineBuckets must be a power of 2.");

  // Number of used words backing the inline buckets (>= 1).
  static constexpr unsigned InlineUsedWords =
      llvm::densemap::detail::usedWords(InlineBuckets);

  unsigned Small : 1;
  unsigned NumEntries : 31;

  // Inline storage: the bucket array followed by the parallel used words.
  struct InlineRep {
    alignas(BucketT) char Buckets[sizeof(BucketT) * InlineBuckets];
    UsedT Used[InlineUsedWords];
  };
  struct LargeRep {
    BucketT *Buckets;
    UsedT *Used;
    unsigned NumBuckets;
  };

  // Discriminated by the Small bit.
  union {
    InlineRep Inline;
    LargeRep Large;
  } storage;

  SmallDenseMap(unsigned NumBuckets, typename BaseT::ExactBucketCount) {
    this->initWithExactBucketCount(NumBuckets);
  }

public:
  explicit SmallDenseMap(unsigned NumElementsToReserve = 0)
      : SmallDenseMap(
            BaseT::getMinBucketToReserveForEntries(NumElementsToReserve),
            typename BaseT::ExactBucketCount{}) {}

  SmallDenseMap(const SmallDenseMap &other) : SmallDenseMap() {
    this->copyFrom(other);
  }

  SmallDenseMap(SmallDenseMap &&other) : SmallDenseMap() { this->swap(other); }

  template <typename InputIt>
  SmallDenseMap(const InputIt &I, const InputIt &E)
      : SmallDenseMap(std::distance(I, E)) {
    this->insert(I, E);
  }

  template <typename RangeT>
  SmallDenseMap(llvm::from_range_t, const RangeT &Range)
      : SmallDenseMap(adl_begin(Range), adl_end(Range)) {}

  SmallDenseMap(std::initializer_list<typename BaseT::value_type> Vals)
      : SmallDenseMap(Vals.begin(), Vals.end()) {}

  ~SmallDenseMap() {
    this->destroyAll();
    deallocateBuckets();
  }

  SmallDenseMap &operator=(const SmallDenseMap &other) {
    if (&other != this)
      this->copyFrom(other);
    return *this;
  }

  SmallDenseMap &operator=(SmallDenseMap &&other) {
    this->destroyAll();
    deallocateBuckets();
    this->initWithExactBucketCount(0);
    this->swap(other);
    return *this;
  }

private:
  // Move-construct *Dst from *Src, then destroy *Src.  Dst is raw storage.
  static void relocateBucket(BucketT *Dst, BucketT *Src) {
    ::new (&Dst->getFirst()) KeyT(std::move(Src->getFirst()));
    ::new (&Dst->getSecond()) ValueT(std::move(Src->getSecond()));
    Src->getSecond().~ValueT();
    Src->getFirst().~KeyT();
  }

  void swapImpl(SmallDenseMap &RHS) {
    unsigned TmpNumEntries = RHS.NumEntries;
    RHS.NumEntries = NumEntries;
    NumEntries = TmpNumEntries;

    if (Small && RHS.Small) {
      // Both inline: swap the live bucket contents slot by slot, then the used
      // used words.  Buckets are raw storage, so a value may only move in one
      // direction when exactly one side is occupied.
      UsedT *LU = getInlineUsed(), *RU = RHS.getInlineUsed();
      BucketT *LB = getInlineBuckets(), *RB = RHS.getInlineBuckets();
      for (unsigned I = 0; I != InlineBuckets; ++I) {
        bool L = llvm::densemap::detail::used(LU, I);
        bool R = llvm::densemap::detail::used(RU, I);
        if (L && R) {
          // Both occupied: exchange through a temporary.
          alignas(BucketT) char Tmp[sizeof(BucketT)];
          BucketT *T = reinterpret_cast<BucketT *>(Tmp);
          relocateBucket(T, &LB[I]);
          relocateBucket(&LB[I], &RB[I]);
          relocateBucket(&RB[I], T);
        } else if (L) {
          relocateBucket(&RB[I], &LB[I]);
        } else if (R) {
          relocateBucket(&LB[I], &RB[I]);
        }
      }
      for (unsigned W = 0; W != InlineUsedWords; ++W)
        std::swap(LU[W], RU[W]);
      return;
    }
    if (!Small && !RHS.Small) {
      std::swap(storage.Large, RHS.storage.Large);
      return;
    }

    SmallDenseMap &SmallSide = Small ? *this : RHS;
    SmallDenseMap &LargeSide = Small ? RHS : *this;

    // Stash the large rep, then move the small side's inline contents into the
    // large side (which becomes inline), and finally install the rep on the
    // small side (which becomes large).
    LargeRep TmpRep = LargeSide.storage.Large;
    LargeSide.Small = true;
    {
      UsedT *SU = SmallSide.getInlineUsed(), *LU = LargeSide.getInlineUsed();
      BucketT *SB = SmallSide.getInlineBuckets(),
              *LB = LargeSide.getInlineBuckets();
      for (unsigned I = 0; I != InlineBuckets; ++I)
        if (llvm::densemap::detail::used(SU, I))
          relocateBucket(&LB[I], &SB[I]);
      for (unsigned W = 0; W != InlineUsedWords; ++W)
        LU[W] = SU[W];
    }
    SmallSide.Small = false;
    SmallSide.storage.Large = TmpRep;
  }

  unsigned getNumEntries() const { return NumEntries; }

  void setNumEntries(unsigned Num) {
    // NumEntries is hardcoded to be 31 bits wide.
    assert(Num < (1U << 31) && "Cannot support more than 1<<31 entries");
    NumEntries = Num;
  }

  const BucketT *getInlineBuckets() const {
    assert(Small);
    // Note that this cast does not violate aliasing rules as we assert that
    // the memory's dynamic type is the small, inline bucket buffer, and the
    // 'storage' is a POD containing a char buffer.
    return reinterpret_cast<const BucketT *>(storage.Inline.Buckets);
  }

  BucketT *getInlineBuckets() {
    assert(Small);
    return reinterpret_cast<BucketT *>(storage.Inline.Buckets);
  }

  const UsedT *getInlineUsed() const {
    assert(Small);
    return storage.Inline.Used;
  }

  UsedT *getInlineUsed() {
    assert(Small);
    return storage.Inline.Used;
  }

  const BucketT *getBuckets() const {
    return Small ? getInlineBuckets() : storage.Large.Buckets;
  }

  typename BaseT::Rep getRep() const {
    if (Small)
      return {getInlineBuckets(), getInlineUsed(), InlineBuckets};
    return {storage.Large.Buckets, storage.Large.Used,
            storage.Large.NumBuckets};
  }

  BucketT *getBuckets() {
    return const_cast<BucketT *>(
        const_cast<const SmallDenseMap *>(this)->getBuckets());
  }

  const UsedT *getUsed() const {
    return Small ? getInlineUsed() : storage.Large.Used;
  }

  UsedT *getUsed() {
    return const_cast<UsedT *>(
        const_cast<const SmallDenseMap *>(this)->getUsed());
  }

  unsigned getNumBuckets() const {
    return Small ? InlineBuckets : storage.Large.NumBuckets;
  }

  void deallocateBuckets() {
    // Fast path in case storage.Large.NumBuckets == 0, just like destroyAll.
    // This path is used to destruct zombie instances after moves.
    if (Small || storage.Large.NumBuckets == 0)
      return;

    deallocate_buffer(
        storage.Large.Buckets,
        llvm::densemap::detail::allocBytes<BucketT>(storage.Large.NumBuckets),
        llvm::densemap::detail::allocAlign<BucketT>());
    storage.Large.NumBuckets = 0;
  }

  bool allocateBuckets(unsigned Num) {
    if (Num <= InlineBuckets) {
      Small = true;
      return true;
    }
    Small = false;
    auto *S = static_cast<char *>(
        allocate_buffer(llvm::densemap::detail::allocBytes<BucketT>(Num),
                        llvm::densemap::detail::allocAlign<BucketT>()));
    storage.Large.Buckets = reinterpret_cast<BucketT *>(S);
    storage.Large.Used = reinterpret_cast<UsedT *>(S + sizeof(BucketT) * Num);
    storage.Large.NumBuckets = Num;
    return true;
  }

  // Put the zombie instance in a known good state after a move.
  void kill() {
    deallocateBuckets();
    Small = false;
    storage.Large = LargeRep{nullptr, nullptr, 0};
  }

  static unsigned roundUpNumBuckets(unsigned MinNumBuckets) {
    if (MinNumBuckets <= InlineBuckets)
      return InlineBuckets;
    return std::max(64u,
                    static_cast<unsigned>(NextPowerOf2(MinNumBuckets - 1)));
  }

  bool maybeMoveFast(SmallDenseMap &&Other) {
    if (Other.Small)
      return false;

    Small = false;
    NumEntries = Other.NumEntries;
    storage.Large = Other.storage.Large;
    Other.storage.Large.NumBuckets = 0;
    return true;
  }

  // Plan how to shrink the bucket table.  Return:
  // - {false, 0} to reuse the existing bucket table
  // - {true, N} to reallocate a bucket table with N entries
  std::pair<bool, unsigned> planShrinkAndClear() const {
    unsigned NewNumBuckets = 0;
    if (!this->empty()) {
      NewNumBuckets = 1u << (Log2_32_Ceil(this->size()) + 1);
      if (NewNumBuckets > InlineBuckets)
        NewNumBuckets = std::max(64u, NewNumBuckets);
    }
    bool Reuse = Small ? NewNumBuckets <= InlineBuckets
                       : NewNumBuckets == storage.Large.NumBuckets;
    if (Reuse)
      return {false, 0};          // Reuse.
    return {true, NewNumBuckets}; // Reallocate.
  }
};

template <typename KeyT, typename ValueT, typename KeyInfoT, typename Bucket,
          bool IsConst>
class DenseMapIterator : DebugEpochBase::HandleBase {
  friend class DenseMapIterator<KeyT, ValueT, KeyInfoT, Bucket, true>;
  friend class DenseMapIterator<KeyT, ValueT, KeyInfoT, Bucket, false>;

  using UsedT = llvm::densemap::detail::UsedT;

public:
  using difference_type = ptrdiff_t;
  using value_type = std::conditional_t<IsConst, const Bucket, Bucket>;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::forward_iterator_tag;

private:
  using BucketItTy =
      std::conditional_t<shouldReverseIterate<KeyT>(),
                         std::reverse_iterator<pointer>, pointer>;

  BucketItTy Ptr = {};
  BucketItTy End = {};
  // The non-reversed bucket base and the parallel used array.  They map a
  // bucket back to its index so AdvancePastEmptyBuckets can consult the bits.
  pointer Buckets = {};
  const UsedT *Used = {};

  DenseMapIterator(BucketItTy Pos, BucketItTy E, pointer BucketsBase,
                   const UsedT *U, const DebugEpochBase &Epoch)
      : DebugEpochBase::HandleBase(&Epoch), Ptr(Pos), End(E),
        Buckets(BucketsBase), Used(U) {
    assert(isHandleInSync() && "invalid construction!");
  }

public:
  DenseMapIterator() = default;

  static DenseMapIterator makeBegin(pointer Buckets, const UsedT *Used,
                                    unsigned NumBuckets, bool IsEmpty,
                                    const DebugEpochBase &Epoch) {
    // When the map is empty, avoid the overhead of advancing/retreating past
    // empty buckets.
    if (IsEmpty)
      return makeEnd(Buckets, Used, NumBuckets, Epoch);
    auto R = maybeReverse(llvm::make_range(Buckets, Buckets + NumBuckets));
    DenseMapIterator Iter(R.begin(), R.end(), Buckets, Used, Epoch);
    Iter.AdvancePastEmptyBuckets();
    return Iter;
  }

  static DenseMapIterator makeEnd(pointer Buckets, const UsedT *Used,
                                  unsigned NumBuckets,
                                  const DebugEpochBase &Epoch) {
    auto R = maybeReverse(llvm::make_range(Buckets, Buckets + NumBuckets));
    return DenseMapIterator(R.end(), R.end(), Buckets, Used, Epoch);
  }

  static DenseMapIterator makeIterator(pointer P, pointer Buckets,
                                       const UsedT *Used, unsigned NumBuckets,
                                       const DebugEpochBase &Epoch) {
    auto R = maybeReverse(llvm::make_range(Buckets, Buckets + NumBuckets));
    constexpr int Offset = shouldReverseIterate<KeyT>() ? 1 : 0;
    return DenseMapIterator(BucketItTy(P + Offset), R.end(), Buckets, Used,
                            Epoch);
  }

  // Converting ctor from non-const iterators to const iterators. SFINAE'd out
  // for const iterator destinations so it doesn't end up as a user defined copy
  // constructor.
  template <bool IsConstSrc,
            typename = std::enable_if_t<!IsConstSrc && IsConst>>
  DenseMapIterator(
      const DenseMapIterator<KeyT, ValueT, KeyInfoT, Bucket, IsConstSrc> &I)
      : DebugEpochBase::HandleBase(I), Ptr(I.Ptr), End(I.End),
        Buckets(I.Buckets), Used(I.Used) {}

  [[nodiscard]] reference operator*() const {
    assert(isHandleInSync() && "invalid iterator access!");
    assert(Ptr != End && "dereferencing end() iterator");
    return *Ptr;
  }
  [[nodiscard]] pointer operator->() const { return &operator*(); }

  [[nodiscard]] friend bool operator==(const DenseMapIterator &LHS,
                                       const DenseMapIterator &RHS) {
    assert((!LHS.getEpochAddress() || LHS.isHandleInSync()) &&
           "handle not in sync!");
    assert((!RHS.getEpochAddress() || RHS.isHandleInSync()) &&
           "handle not in sync!");
    assert(LHS.getEpochAddress() == RHS.getEpochAddress() &&
           "comparing incomparable iterators!");
    return LHS.Ptr == RHS.Ptr;
  }

  [[nodiscard]] friend bool operator!=(const DenseMapIterator &LHS,
                                       const DenseMapIterator &RHS) {
    return !(LHS == RHS);
  }

  inline DenseMapIterator &operator++() { // Preincrement
    assert(isHandleInSync() && "invalid iterator access!");
    assert(Ptr != End && "incrementing end() iterator");
    ++Ptr;
    AdvancePastEmptyBuckets();
    return *this;
  }
  DenseMapIterator operator++(int) { // Postincrement
    assert(isHandleInSync() && "invalid iterator access!");
    DenseMapIterator tmp = *this;
    ++*this;
    return tmp;
  }

private:
  void AdvancePastEmptyBuckets() {
    if constexpr (shouldReverseIterate<KeyT>()) {
      while (Ptr != End && !llvm::densemap::detail::used(Used, &*Ptr - Buckets))
        ++Ptr;
    } else {
      // Forward iteration skips empty buckets a used-word (32 buckets) at a
      // time: scan from the current index for the next set occupancy bit.
      const size_t N = End - Buckets;
      size_t I = Ptr - Buckets;
      if (I >= N) {
        Ptr = End;
        return;
      }
      const size_t NW = llvm::densemap::detail::usedWords(N);
      size_t W = I >> 5;
      UsedT Bits = Used[W] & (~UsedT(0) << (I & 31));
      while (Bits == 0) {
        if (++W == NW) {
          Ptr = End;
          return;
        }
        Bits = Used[W];
      }
      Ptr = Buckets + ((W << 5) + llvm::countr_zero(Bits));
    }
  }

  static auto maybeReverse(iterator_range<pointer> Range) {
    if constexpr (shouldReverseIterate<KeyT>())
      return reverse(Range);
    else
      return Range;
  }
};

template <typename KeyT, typename ValueT, typename KeyInfoT>
[[nodiscard]] inline size_t
capacity_in_bytes(const DenseMap<KeyT, ValueT, KeyInfoT> &X) {
  return X.getMemorySize();
}

} // end namespace llvm

#endif // LLVM_ADT_DENSEMAP_H
