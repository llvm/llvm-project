//===- llvm/ADT/DenseMap.h - Dense probed hash table ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the DenseMap class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_DENSEMAP_H
#define LLVM_ADT_DENSEMAP_H

#include "llvm/Support/DataTypes.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <utility>

namespace llvm {

template<typename T>
struct DenseMapInfo {
  //static inline T getEmptyKey();
  //static inline T getTombstoneKey();
  //static unsigned getHashValue(const T &Val);
  //static bool isEqual(const T &LHS, const T &RHS);
  //static bool isPod()
};

// Provide DenseMapInfo for all pointers.
template<typename T>
struct DenseMapInfo<T*> {
  static inline T* getEmptyKey() { return reinterpret_cast<T*>(-1); }
  static inline T* getTombstoneKey() { return reinterpret_cast<T*>(-2); }
  static unsigned getHashValue(const T *PtrVal) {
    return (unsigned((uintptr_t)PtrVal) >> 4) ^
           (unsigned((uintptr_t)PtrVal) >> 9);
  }
  static bool isEqual(const T *LHS, const T *RHS) { return LHS == RHS; }
  static bool isPod() { return true; }
};

// Provide DenseMapInfo for unsigned ints.
template<> struct DenseMapInfo<unsigned> {
  static inline unsigned getEmptyKey() { return ~0; }
  static inline unsigned getTombstoneKey() { return ~0 - 1; }
  static unsigned getHashValue(const unsigned& Val) { return Val * 37; }
  static bool isPod() { return true; }
  static bool isEqual(const unsigned& LHS, const unsigned& RHS) {
  return LHS == RHS;
  }
};

// Provide DenseMapInfo for unsigned longs.
template<> struct DenseMapInfo<unsigned long> {
  static inline unsigned long getEmptyKey() { return ~0L; }
  static inline unsigned long getTombstoneKey() { return ~0L - 1L; }
  static unsigned getHashValue(const unsigned long& Val) {
    return (unsigned)(Val * 37L);
  }
  static bool isPod() { return true; }
  static bool isEqual(const unsigned long& LHS, const unsigned long& RHS) {
  return LHS == RHS;
  }
};

// Provide DenseMapInfo for all pairs whose members have info.
template<typename T, typename U>
struct DenseMapInfo<std::pair<T, U> > {
  typedef std::pair<T, U> Pair;
  typedef DenseMapInfo<T> FirstInfo;
  typedef DenseMapInfo<U> SecondInfo;

  static inline Pair getEmptyKey() {
    return std::make_pair(FirstInfo::getEmptyKey(),
                          SecondInfo::getEmptyKey());
  }
  static inline Pair getTombstoneKey() {
    return std::make_pair(FirstInfo::getTombstoneKey(),
                            SecondInfo::getEmptyKey());
  }
  static unsigned getHashValue(const Pair& PairVal) {
    uint64_t key = (uint64_t)FirstInfo::getHashValue(PairVal.first) << 32
          | (uint64_t)SecondInfo::getHashValue(PairVal.second);
    key += ~(key << 32);
    key ^= (key >> 22);
    key += ~(key << 13);
    key ^= (key >> 8);
    key += (key << 3);
    key ^= (key >> 15);
    key += ~(key << 27);
    key ^= (key >> 31);
    return (unsigned)key;
  }
  static bool isEqual(const Pair& LHS, const Pair& RHS) { return LHS == RHS; }
  static bool isPod() { return FirstInfo::isPod() && SecondInfo::isPod(); }
};

template<typename KeyT, typename ValueT,
         typename KeyInfoT = DenseMapInfo<KeyT>,
         typename ValueInfoT = DenseMapInfo<ValueT> >
class DenseMapIterator;
template<typename KeyT, typename ValueT,
         typename KeyInfoT = DenseMapInfo<KeyT>,
         typename ValueInfoT = DenseMapInfo<ValueT> >
class DenseMapConstIterator;

template<typename KeyT, typename ValueT,
         typename KeyInfoT = DenseMapInfo<KeyT>,
         typename ValueInfoT = DenseMapInfo<ValueT> >
class DenseMap {
  typedef std::pair<KeyT, ValueT> BucketT;
  unsigned NumBuckets;
  BucketT *Buckets;

  unsigned NumEntries;
  unsigned NumTombstones;
public:
  typedef KeyT key_type;
  typedef ValueT mapped_type;
  typedef BucketT value_type;

  DenseMap(const DenseMap& other) {
    NumBuckets = 0;
    CopyFrom(other);
  }

  explicit DenseMap(unsigned NumInitBuckets = 64) {
    init(NumInitBuckets);
  }

  ~DenseMap() {
    const KeyT EmptyKey = getEmptyKey(), TombstoneKey = getTombstoneKey();
    for (BucketT *P = Buckets, *E = Buckets+NumBuckets; P != E; ++P) {
      if (!KeyInfoT::isEqual(P->first, EmptyKey) &&
          !KeyInfoT::isEqual(P->first, TombstoneKey))
        P->second.~ValueT();
      P->first.~KeyT();
    }
    operator delete(Buckets);
  }

  typedef DenseMapIterator<KeyT, ValueT, KeyInfoT> iterator;
  typedef DenseMapConstIterator<KeyT, ValueT, KeyInfoT> const_iterator;
  inline iterator begin() {
     return iterator(Buckets, Buckets+NumBuckets);
  }
  inline iterator end() {
    return iterator(Buckets+NumBuckets, Buckets+NumBuckets);
  }
  inline const_iterator begin() const {
    return const_iterator(Buckets, Buckets+NumBuckets);
  }
  inline const_iterator end() const {
    return const_iterator(Buckets+NumBuckets, Buckets+NumBuckets);
  }

  bool empty() const { return NumEntries == 0; }
  unsigned size() const { return NumEntries; }

  /// Grow the densemap so that it has at least Size buckets. Does not shrink
  void resize(size_t Size) { grow(Size); }

  void clear() {
    // If the capacity of the array is huge, and the # elements used is small,
    // shrink the array.
    if (NumEntries * 4 < NumBuckets && NumBuckets > 64) {
      shrink_and_clear();
      return;
    }

    const KeyT EmptyKey = getEmptyKey(), TombstoneKey = getTombstoneKey();
    for (BucketT *P = Buckets, *E = Buckets+NumBuckets; P != E; ++P) {
      if (!KeyInfoT::isEqual(P->first, EmptyKey)) {
        if (!KeyInfoT::isEqual(P->first, TombstoneKey)) {
          P->second.~ValueT();
          --NumEntries;
        }
        P->first = EmptyKey;
      }
    }
    assert(NumEntries == 0 && "Node count imbalance!");
    NumTombstones = 0;
  }

  /// count - Return true if the specified key is in the map.
  bool count(const KeyT &Val) const {
    BucketT *TheBucket;
    return LookupBucketFor(Val, TheBucket);
  }

  iterator find(const KeyT &Val) {
    BucketT *TheBucket;
    if (LookupBucketFor(Val, TheBucket))
      return iterator(TheBucket, Buckets+NumBuckets);
    return end();
  }
  const_iterator find(const KeyT &Val) const {
    BucketT *TheBucket;
    if (LookupBucketFor(Val, TheBucket))
      return const_iterator(TheBucket, Buckets+NumBuckets);
    return end();
  }

  /// lookup - Return the entry for the specified key, or a default
  /// constructed value if no such entry exists.
  ValueT lookup(const KeyT &Val) const {
    BucketT *TheBucket;
    if (LookupBucketFor(Val, TheBucket))
      return TheBucket->second;
    return ValueT();
  }

  std::pair<iterator, bool> insert(const std::pair<KeyT, ValueT> &KV) {
    BucketT *TheBucket;
    if (LookupBucketFor(KV.first, TheBucket))
      return std::make_pair(iterator(TheBucket, Buckets+NumBuckets),
                            false); // Already in map.

    // Otherwise, insert the new element.
    TheBucket = InsertIntoBucket(KV.first, KV.second, TheBucket);
    return std::make_pair(iterator(TheBucket, Buckets+NumBuckets),
                          true);
  }

  /// insert - Range insertion of pairs.
  template<typename InputIt>
  void insert(InputIt I, InputIt E) {
    for (; I != E; ++I)
      insert(*I);
  }


  bool erase(const KeyT &Val) {
    BucketT *TheBucket;
    if (!LookupBucketFor(Val, TheBucket))
      return false; // not in map.

    TheBucket->second.~ValueT();
    TheBucket->first = getTombstoneKey();
    --NumEntries;
    ++NumTombstones;
    return true;
  }
  bool erase(iterator I) {
    BucketT *TheBucket = &*I;
    TheBucket->second.~ValueT();
    TheBucket->first = getTombstoneKey();
    --NumEntries;
    ++NumTombstones;
    return true;
  }

  value_type& FindAndConstruct(const KeyT &Key) {
    BucketT *TheBucket;
    if (LookupBucketFor(Key, TheBucket))
      return *TheBucket;

    return *InsertIntoBucket(Key, ValueT(), TheBucket);
  }

  ValueT &operator[](const KeyT &Key) {
    return FindAndConstruct(Key).second;
  }

  DenseMap& operator=(const DenseMap& other) {
    CopyFrom(other);
    return *this;
  }

private:
  void CopyFrom(const DenseMap& other) {
    if (NumBuckets != 0 && (!KeyInfoT::isPod() || !ValueInfoT::isPod())) {
      const KeyT EmptyKey = getEmptyKey(), TombstoneKey = getTombstoneKey();
      for (BucketT *P = Buckets, *E = Buckets+NumBuckets; P != E; ++P) {
        if (!KeyInfoT::isEqual(P->first, EmptyKey) &&
            !KeyInfoT::isEqual(P->first, TombstoneKey))
          P->second.~ValueT();
        P->first.~KeyT();
      }
    }

    NumEntries = other.NumEntries;
    NumTombstones = other.NumTombstones;

    if (NumBuckets)
      operator delete(Buckets);
    Buckets = static_cast<BucketT*>(operator new(sizeof(BucketT) *
                                                 other.NumBuckets));

    if (KeyInfoT::isPod() && ValueInfoT::isPod())
      memcpy(Buckets, other.Buckets, other.NumBuckets * sizeof(BucketT));
    else
      for (size_t i = 0; i < other.NumBuckets; ++i) {
        new (&Buckets[i].first) KeyT(other.Buckets[i].first);
        if (!KeyInfoT::isEqual(Buckets[i].first, getEmptyKey()) &&
            !KeyInfoT::isEqual(Buckets[i].first, getTombstoneKey()))
          new (&Buckets[i].second) ValueT(other.Buckets[i].second);
      }
    NumBuckets = other.NumBuckets;
  }

  BucketT *InsertIntoBucket(const KeyT &Key, const ValueT &Value,
                            BucketT *TheBucket) {
    // If the load of the hash table is more than 3/4, or if fewer than 1/8 of
    // the buckets are empty (meaning that many are filled with tombstones),
    // grow the table.
    //
    // The later case is tricky.  For example, if we had one empty bucket with
    // tons of tombstones, failing lookups (e.g. for insertion) would have to
    // probe almost the entire table until it found the empty bucket.  If the
    // table completely filled with tombstones, no lookup would ever succeed,
    // causing infinite loops in lookup.
    if (NumEntries*4 >= NumBuckets*3 ||
        NumBuckets-(NumEntries+NumTombstones) < NumBuckets/8) {
      this->grow(NumBuckets * 2);
      LookupBucketFor(Key, TheBucket);
    }
    ++NumEntries;

    // If we are writing over a tombstone, remember this.
    if (!KeyInfoT::isEqual(TheBucket->first, getEmptyKey()))
      --NumTombstones;

    TheBucket->first = Key;
    new (&TheBucket->second) ValueT(Value);
    return TheBucket;
  }

  static unsigned getHashValue(const KeyT &Val) {
    return KeyInfoT::getHashValue(Val);
  }
  static const KeyT getEmptyKey() {
    return KeyInfoT::getEmptyKey();
  }
  static const KeyT getTombstoneKey() {
    return KeyInfoT::getTombstoneKey();
  }

  /// LookupBucketFor - Lookup the appropriate bucket for Val, returning it in
  /// FoundBucket.  If the bucket contains the key and a value, this returns
  /// true, otherwise it returns a bucket with an empty marker or tombstone and
  /// returns false.
  bool LookupBucketFor(const KeyT &Val, BucketT *&FoundBucket) const {
    unsigned BucketNo = getHashValue(Val);
    unsigned ProbeAmt = 1;
    BucketT *BucketsPtr = Buckets;

    // FoundTombstone - Keep track of whether we find a tombstone while probing.
    BucketT *FoundTombstone = 0;
    const KeyT EmptyKey = getEmptyKey();
    const KeyT TombstoneKey = getTombstoneKey();
    assert(!KeyInfoT::isEqual(Val, EmptyKey) &&
           !KeyInfoT::isEqual(Val, TombstoneKey) &&
           "Empty/Tombstone value shouldn't be inserted into map!");

    while (1) {
      BucketT *ThisBucket = BucketsPtr + (BucketNo & (NumBuckets-1));
      // Found Val's bucket?  If so, return it.
      if (KeyInfoT::isEqual(ThisBucket->first, Val)) {
        FoundBucket = ThisBucket;
        return true;
      }

      // If we found an empty bucket, the key doesn't exist in the set.
      // Insert it and return the default value.
      if (KeyInfoT::isEqual(ThisBucket->first, EmptyKey)) {
        // If we've already seen a tombstone while probing, fill it in instead
        // of the empty bucket we eventually probed to.
        if (FoundTombstone) ThisBucket = FoundTombstone;
        FoundBucket = FoundTombstone ? FoundTombstone : ThisBucket;
        return false;
      }

      // If this is a tombstone, remember it.  If Val ends up not in the map, we
      // prefer to return it than something that would require more probing.
      if (KeyInfoT::isEqual(ThisBucket->first, TombstoneKey) && !FoundTombstone)
        FoundTombstone = ThisBucket;  // Remember the first tombstone found.

      // Otherwise, it's a hash collision or a tombstone, continue quadratic
      // probing.
      BucketNo += ProbeAmt++;
    }
  }

  void init(unsigned InitBuckets) {
    NumEntries = 0;
    NumTombstones = 0;
    NumBuckets = InitBuckets;
    assert(InitBuckets && (InitBuckets & (InitBuckets-1)) == 0 &&
           "# initial buckets must be a power of two!");
    Buckets = static_cast<BucketT*>(operator new(sizeof(BucketT)*InitBuckets));
    // Initialize all the keys to EmptyKey.
    const KeyT EmptyKey = getEmptyKey();
    for (unsigned i = 0; i != InitBuckets; ++i)
      new (&Buckets[i].first) KeyT(EmptyKey);
  }

  void grow(unsigned AtLeast) {
    unsigned OldNumBuckets = NumBuckets;
    BucketT *OldBuckets = Buckets;

    // Double the number of buckets.
    while (NumBuckets <= AtLeast)
      NumBuckets <<= 1;
    NumTombstones = 0;
    Buckets = static_cast<BucketT*>(operator new(sizeof(BucketT)*NumBuckets));

    // Initialize all the keys to EmptyKey.
    const KeyT EmptyKey = getEmptyKey();
    for (unsigned i = 0, e = NumBuckets; i != e; ++i)
      new (&Buckets[i].first) KeyT(EmptyKey);

    // Insert all the old elements.
    const KeyT TombstoneKey = getTombstoneKey();
    for (BucketT *B = OldBuckets, *E = OldBuckets+OldNumBuckets; B != E; ++B) {
      if (!KeyInfoT::isEqual(B->first, EmptyKey) &&
          !KeyInfoT::isEqual(B->first, TombstoneKey)) {
        // Insert the key/value into the new table.
        BucketT *DestBucket;
        bool FoundVal = LookupBucketFor(B->first, DestBucket);
        FoundVal = FoundVal; // silence warning.
        assert(!FoundVal && "Key already in new map?");
        DestBucket->first = B->first;
        new (&DestBucket->second) ValueT(B->second);

        // Free the value.
        B->second.~ValueT();
      }
      B->first.~KeyT();
    }

    // Free the old table.
    operator delete(OldBuckets);
  }

  void shrink_and_clear() {
    unsigned OldNumBuckets = NumBuckets;
    BucketT *OldBuckets = Buckets;

    // Reduce the number of buckets.
    NumBuckets = NumEntries > 32 ? 1 << (Log2_32_Ceil(NumEntries) + 1)
                                 : 64;
    NumTombstones = 0;
    Buckets = static_cast<BucketT*>(operator new(sizeof(BucketT)*NumBuckets));

    // Initialize all the keys to EmptyKey.
    const KeyT EmptyKey = getEmptyKey();
    for (unsigned i = 0, e = NumBuckets; i != e; ++i)
      new (&Buckets[i].first) KeyT(EmptyKey);

    // Free the old buckets.
    const KeyT TombstoneKey = getTombstoneKey();
    for (BucketT *B = OldBuckets, *E = OldBuckets+OldNumBuckets; B != E; ++B) {
      if (!KeyInfoT::isEqual(B->first, EmptyKey) &&
          !KeyInfoT::isEqual(B->first, TombstoneKey)) {
        // Free the value.
        B->second.~ValueT();
      }
      B->first.~KeyT();
    }

    // Free the old table.
    operator delete(OldBuckets);

    NumEntries = 0;
  }
};

template<typename KeyT, typename ValueT, typename KeyInfoT, typename ValueInfoT>
class DenseMapIterator {
  typedef std::pair<KeyT, ValueT> BucketT;
protected:
  const BucketT *Ptr, *End;
public:
  DenseMapIterator(void) : Ptr(0), End(0) {}

  DenseMapIterator(const BucketT *Pos, const BucketT *E) : Ptr(Pos), End(E) {
    AdvancePastEmptyBuckets();
  }

  std::pair<KeyT, ValueT> &operator*() const {
    return *const_cast<BucketT*>(Ptr);
  }
  std::pair<KeyT, ValueT> *operator->() const {
    return const_cast<BucketT*>(Ptr);
  }

  bool operator==(const DenseMapIterator &RHS) const {
    return Ptr == RHS.Ptr;
  }
  bool operator!=(const DenseMapIterator &RHS) const {
    return Ptr != RHS.Ptr;
  }

  inline DenseMapIterator& operator++() {          // Preincrement
    ++Ptr;
    AdvancePastEmptyBuckets();
    return *this;
  }
  DenseMapIterator operator++(int) {        // Postincrement
    DenseMapIterator tmp = *this; ++*this; return tmp;
  }

private:
  void AdvancePastEmptyBuckets() {
    const KeyT Empty = KeyInfoT::getEmptyKey();
    const KeyT Tombstone = KeyInfoT::getTombstoneKey();

    while (Ptr != End &&
           (KeyInfoT::isEqual(Ptr->first, Empty) ||
            KeyInfoT::isEqual(Ptr->first, Tombstone)))
      ++Ptr;
  }
};

template<typename KeyT, typename ValueT, typename KeyInfoT, typename ValueInfoT>
class DenseMapConstIterator : public DenseMapIterator<KeyT, ValueT, KeyInfoT> {
public:
  DenseMapConstIterator(void) : DenseMapIterator<KeyT, ValueT, KeyInfoT>() {}
  DenseMapConstIterator(const std::pair<KeyT, ValueT> *Pos,
                        const std::pair<KeyT, ValueT> *E)
    : DenseMapIterator<KeyT, ValueT, KeyInfoT>(Pos, E) {
  }
  const std::pair<KeyT, ValueT> &operator*() const {
    return *this->Ptr;
  }
  const std::pair<KeyT, ValueT> *operator->() const {
    return this->Ptr;
  }
};

} // end namespace llvm

#endif
