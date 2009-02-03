//===--- StringMap.h - String Hash table map interface ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the StringMap class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STRINGMAP_H
#define LLVM_ADT_STRINGMAP_H

#include "llvm/Support/Allocator.h"
#include <cstring>
#include <string>

namespace llvm {
  template<typename ValueT>
  class StringMapConstIterator;
  template<typename ValueT>
  class StringMapIterator;
  template<typename ValueTy>
  class StringMapEntry;

/// StringMapEntryInitializer - This datatype can be partially specialized for
/// various datatypes in a stringmap to allow them to be initialized when an
/// entry is default constructed for the map.
template<typename ValueTy>
class StringMapEntryInitializer {
public:
  template <typename InitTy>
  static void Initialize(StringMapEntry<ValueTy> &T, InitTy InitVal) {
  }
};


/// StringMapEntryBase - Shared base class of StringMapEntry instances.
class StringMapEntryBase {
  unsigned StrLen;
public:
  explicit StringMapEntryBase(unsigned Len) : StrLen(Len) {}

  unsigned getKeyLength() const { return StrLen; }
};

/// StringMapImpl - This is the base class of StringMap that is shared among
/// all of its instantiations.
class StringMapImpl {
public:
  /// ItemBucket - The hash table consists of an array of these.  If Item is
  /// non-null, this is an extant entry, otherwise, it is a hole.
  struct ItemBucket {
    /// FullHashValue - This remembers the full hash value of the key for
    /// easy scanning.
    unsigned FullHashValue;

    /// Item - This is a pointer to the actual item object.
    StringMapEntryBase *Item;
  };

protected:
  ItemBucket *TheTable;
  unsigned NumBuckets;
  unsigned NumItems;
  unsigned NumTombstones;
  unsigned ItemSize;
protected:
  explicit StringMapImpl(unsigned itemSize) : ItemSize(itemSize) {
    // Initialize the map with zero buckets to allocation.
    TheTable = 0;
    NumBuckets = 0;
    NumItems = 0;
    NumTombstones = 0;
  }
  StringMapImpl(unsigned InitSize, unsigned ItemSize);
  void RehashTable();

  /// ShouldRehash - Return true if the table should be rehashed after a new
  /// element was recently inserted.
  bool ShouldRehash() const {
    // If the hash table is now more than 3/4 full, or if fewer than 1/8 of
    // the buckets are empty (meaning that many are filled with tombstones),
    // grow the table.
    return NumItems*4 > NumBuckets*3 ||
           NumBuckets-(NumItems+NumTombstones) < NumBuckets/8;
  }

  /// LookupBucketFor - Look up the bucket that the specified string should end
  /// up in.  If it already exists as a key in the map, the Item pointer for the
  /// specified bucket will be non-null.  Otherwise, it will be null.  In either
  /// case, the FullHashValue field of the bucket will be set to the hash value
  /// of the string.
  unsigned LookupBucketFor(const char *KeyStart, const char *KeyEnd);

  /// FindKey - Look up the bucket that contains the specified key. If it exists
  /// in the map, return the bucket number of the key.  Otherwise return -1.
  /// This does not modify the map.
  int FindKey(const char *KeyStart, const char *KeyEnd) const;

  /// RemoveKey - Remove the specified StringMapEntry from the table, but do not
  /// delete it.  This aborts if the value isn't in the table.
  void RemoveKey(StringMapEntryBase *V);

  /// RemoveKey - Remove the StringMapEntry for the specified key from the
  /// table, returning it.  If the key is not in the table, this returns null.
  StringMapEntryBase *RemoveKey(const char *KeyStart, const char *KeyEnd);
private:
  void init(unsigned Size);
public:
  static StringMapEntryBase *getTombstoneVal() {
    return (StringMapEntryBase*)-1;
  }

  unsigned getNumBuckets() const { return NumBuckets; }
  unsigned getNumItems() const { return NumItems; }

  bool empty() const { return NumItems == 0; }
  unsigned size() const { return NumItems; }
};

/// StringMapEntry - This is used to represent one value that is inserted into
/// a StringMap.  It contains the Value itself and the key: the string length
/// and data.
template<typename ValueTy>
class StringMapEntry : public StringMapEntryBase {
public:
  ValueTy second;

  explicit StringMapEntry(unsigned strLen)
    : StringMapEntryBase(strLen), second() {}
  StringMapEntry(unsigned strLen, const ValueTy &V)
    : StringMapEntryBase(strLen), second(V) {}

  const ValueTy &getValue() const { return second; }
  ValueTy &getValue() { return second; }

  void setValue(const ValueTy &V) { second = V; }

  /// getKeyData - Return the start of the string data that is the key for this
  /// value.  The string data is always stored immediately after the
  /// StringMapEntry object.
  const char *getKeyData() const {return reinterpret_cast<const char*>(this+1);}

  const char *first() const { return getKeyData(); }

  /// Create - Create a StringMapEntry for the specified key and default
  /// construct the value.
  template<typename AllocatorTy, typename InitType>
  static StringMapEntry *Create(const char *KeyStart, const char *KeyEnd,
                                AllocatorTy &Allocator,
                                InitType InitVal) {
    unsigned KeyLength = static_cast<unsigned>(KeyEnd-KeyStart);

    // Okay, the item doesn't already exist, and 'Bucket' is the bucket to fill
    // in.  Allocate a new item with space for the string at the end and a null
    // terminator.

    unsigned AllocSize = static_cast<unsigned>(sizeof(StringMapEntry))+
      KeyLength+1;
    unsigned Alignment = alignof<StringMapEntry>();

    StringMapEntry *NewItem =
      static_cast<StringMapEntry*>(Allocator.Allocate(AllocSize,Alignment));

    // Default construct the value.
    new (NewItem) StringMapEntry(KeyLength);

    // Copy the string information.
    char *StrBuffer = const_cast<char*>(NewItem->getKeyData());
    memcpy(StrBuffer, KeyStart, KeyLength);
    StrBuffer[KeyLength] = 0;  // Null terminate for convenience of clients.

    // Initialize the value if the client wants to.
    StringMapEntryInitializer<ValueTy>::Initialize(*NewItem, InitVal);
    return NewItem;
  }

  template<typename AllocatorTy>
  static StringMapEntry *Create(const char *KeyStart, const char *KeyEnd,
                                AllocatorTy &Allocator) {
    return Create(KeyStart, KeyEnd, Allocator, 0);
  }


  /// Create - Create a StringMapEntry with normal malloc/free.
  template<typename InitType>
  static StringMapEntry *Create(const char *KeyStart, const char *KeyEnd,
                                InitType InitVal) {
    MallocAllocator A;
    return Create(KeyStart, KeyEnd, A, InitVal);
  }

  static StringMapEntry *Create(const char *KeyStart, const char *KeyEnd) {
    return Create(KeyStart, KeyEnd, 0);
  }

  /// GetStringMapEntryFromValue - Given a value that is known to be embedded
  /// into a StringMapEntry, return the StringMapEntry itself.
  static StringMapEntry &GetStringMapEntryFromValue(ValueTy &V) {
    StringMapEntry *EPtr = 0;
    char *Ptr = reinterpret_cast<char*>(&V) -
                  (reinterpret_cast<char*>(&EPtr->second) -
                   reinterpret_cast<char*>(EPtr));
    return *reinterpret_cast<StringMapEntry*>(Ptr);
  }
  static const StringMapEntry &GetStringMapEntryFromValue(const ValueTy &V) {
    return GetStringMapEntryFromValue(const_cast<ValueTy&>(V));
  }

  /// Destroy - Destroy this StringMapEntry, releasing memory back to the
  /// specified allocator.
  template<typename AllocatorTy>
  void Destroy(AllocatorTy &Allocator) {
    // Free memory referenced by the item.
    this->~StringMapEntry();
    Allocator.Deallocate(this);
  }

  /// Destroy this object, releasing memory back to the malloc allocator.
  void Destroy() {
    MallocAllocator A;
    Destroy(A);
  }
};


/// StringMap - This is an unconventional map that is specialized for handling
/// keys that are "strings", which are basically ranges of bytes. This does some
/// funky memory allocation and hashing things to make it extremely efficient,
/// storing the string data *after* the value in the map.
template<typename ValueTy, typename AllocatorTy = MallocAllocator>
class StringMap : public StringMapImpl {
  AllocatorTy Allocator;
  typedef StringMapEntry<ValueTy> MapEntryTy;
public:
  StringMap() : StringMapImpl(static_cast<unsigned>(sizeof(MapEntryTy))) {}
  explicit StringMap(unsigned InitialSize)
    : StringMapImpl(InitialSize, static_cast<unsigned>(sizeof(MapEntryTy))) {}

  AllocatorTy &getAllocator() { return Allocator; }
  const AllocatorTy &getAllocator() const { return Allocator; }

  typedef const char* key_type;
  typedef ValueTy mapped_type;
  typedef StringMapEntry<ValueTy> value_type;
  typedef size_t size_type;

  typedef StringMapConstIterator<ValueTy> const_iterator;
  typedef StringMapIterator<ValueTy> iterator;

  iterator begin() {
    return iterator(TheTable, NumBuckets == 0);
  }
  iterator end() {
    return iterator(TheTable+NumBuckets, true);
  }
  const_iterator begin() const {
    return const_iterator(TheTable, NumBuckets == 0);
  }
  const_iterator end() const {
    return const_iterator(TheTable+NumBuckets, true);
  }

  iterator find(const char *KeyStart, const char *KeyEnd) {
    int Bucket = FindKey(KeyStart, KeyEnd);
    if (Bucket == -1) return end();
    return iterator(TheTable+Bucket);
  }
  iterator find(const char *Key) {
    return find(Key, Key + strlen(Key));
  }
  iterator find(const std::string &Key) {
    const char* key_start = (Key.empty() ? NULL : &Key[0]);
    return find(key_start, key_start + Key.size());
  }

  const_iterator find(const char *KeyStart, const char *KeyEnd) const {
    int Bucket = FindKey(KeyStart, KeyEnd);
    if (Bucket == -1) return end();
    return const_iterator(TheTable+Bucket);
  }
  const_iterator find(const char *Key) const {
    return find(Key, Key + strlen(Key));
  }
  const_iterator find(const std::string &Key) const {
    const char* key_start = (Key.empty() ? NULL : &Key[0]);
    return find(key_start, key_start + Key.size());
  }

  ValueTy& operator[](const char *Key) {
    value_type& entry = GetOrCreateValue(Key, Key + strlen(Key));
    return entry.getValue();
  }
  ValueTy& operator[](const std::string &Key) {
    const char* key_start = (Key.empty() ? NULL : &Key[0]);
    value_type& entry = GetOrCreateValue(key_start, key_start + Key.size());
    return entry.getValue();
  }

  size_type count(const char *KeyStart, const char *KeyEnd) const {
    return find(KeyStart, KeyEnd) == end() ? 0 : 1;
  }
  size_type count(const char *Key) const {
    return count(Key, Key + strlen(Key));
  }
  size_type count(const std::string &Key) const {
    const char* key_start = (Key.empty() ? NULL : &Key[0]);
    return count(key_start, key_start + Key.size());
  }

  /// insert - Insert the specified key/value pair into the map.  If the key
  /// already exists in the map, return false and ignore the request, otherwise
  /// insert it and return true.
  bool insert(MapEntryTy *KeyValue) {
    unsigned BucketNo =
      LookupBucketFor(KeyValue->getKeyData(),
                      KeyValue->getKeyData()+KeyValue->getKeyLength());
    ItemBucket &Bucket = TheTable[BucketNo];
    if (Bucket.Item && Bucket.Item != getTombstoneVal())
      return false;  // Already exists in map.

    if (Bucket.Item == getTombstoneVal())
      --NumTombstones;
    Bucket.Item = KeyValue;
    ++NumItems;

    if (ShouldRehash())
      RehashTable();
    return true;
  }

  // clear - Empties out the StringMap
  void clear() {
    if (empty()) return;

    // Zap all values, resetting the keys back to non-present (not tombstone),
    // which is safe because we're removing all elements.
    for (ItemBucket *I = TheTable, *E = TheTable+NumBuckets; I != E; ++I) {
      if (I->Item && I->Item != getTombstoneVal()) {
        static_cast<MapEntryTy*>(I->Item)->Destroy(Allocator);
        I->Item = 0;
      }
    }

    NumItems = 0;
  }

  /// GetOrCreateValue - Look up the specified key in the table.  If a value
  /// exists, return it.  Otherwise, default construct a value, insert it, and
  /// return.
  template <typename InitTy>
  StringMapEntry<ValueTy> &GetOrCreateValue(const char *KeyStart,
                                            const char *KeyEnd,
                                            InitTy Val) {
    unsigned BucketNo = LookupBucketFor(KeyStart, KeyEnd);
    ItemBucket &Bucket = TheTable[BucketNo];
    if (Bucket.Item && Bucket.Item != getTombstoneVal())
      return *static_cast<MapEntryTy*>(Bucket.Item);

    MapEntryTy *NewItem = MapEntryTy::Create(KeyStart, KeyEnd, Allocator, Val);

    if (Bucket.Item == getTombstoneVal())
      --NumTombstones;
    ++NumItems;

    // Fill in the bucket for the hash table.  The FullHashValue was already
    // filled in by LookupBucketFor.
    Bucket.Item = NewItem;

    if (ShouldRehash())
      RehashTable();
    return *NewItem;
  }

  StringMapEntry<ValueTy> &GetOrCreateValue(const char *KeyStart,
                                            const char *KeyEnd) {
    return GetOrCreateValue(KeyStart, KeyEnd, 0);
  }

  /// remove - Remove the specified key/value pair from the map, but do not
  /// erase it.  This aborts if the key is not in the map.
  void remove(MapEntryTy *KeyValue) {
    RemoveKey(KeyValue);
  }

  void erase(iterator I) {
    MapEntryTy &V = *I;
    remove(&V);
    V.Destroy(Allocator);
  }

  bool erase(const char *Key) {
    iterator I = find(Key);
    if (I == end()) return false;
    erase(I);
    return true;
  }

  bool erase(const std::string &Key) {
    iterator I = find(Key);
    if (I == end()) return false;
    erase(I);
    return true;
  }

  ~StringMap() {
    clear();
    free(TheTable);
  }
private:
  StringMap(const StringMap &);  // FIXME: Implement.
  void operator=(const StringMap &);  // FIXME: Implement.
};


template<typename ValueTy>
class StringMapConstIterator {
protected:
  StringMapImpl::ItemBucket *Ptr;
public:
  typedef StringMapEntry<ValueTy> value_type;

  explicit StringMapConstIterator(StringMapImpl::ItemBucket *Bucket,
                                  bool NoAdvance = false)
  : Ptr(Bucket) {
    if (!NoAdvance) AdvancePastEmptyBuckets();
  }

  const value_type &operator*() const {
    return *static_cast<StringMapEntry<ValueTy>*>(Ptr->Item);
  }
  const value_type *operator->() const {
    return static_cast<StringMapEntry<ValueTy>*>(Ptr->Item);
  }

  bool operator==(const StringMapConstIterator &RHS) const {
    return Ptr == RHS.Ptr;
  }
  bool operator!=(const StringMapConstIterator &RHS) const {
    return Ptr != RHS.Ptr;
  }

  inline StringMapConstIterator& operator++() {          // Preincrement
    ++Ptr;
    AdvancePastEmptyBuckets();
    return *this;
  }
  StringMapConstIterator operator++(int) {        // Postincrement
    StringMapConstIterator tmp = *this; ++*this; return tmp;
  }

private:
  void AdvancePastEmptyBuckets() {
    while (Ptr->Item == 0 || Ptr->Item == StringMapImpl::getTombstoneVal())
      ++Ptr;
  }
};

template<typename ValueTy>
class StringMapIterator : public StringMapConstIterator<ValueTy> {
public:
  explicit StringMapIterator(StringMapImpl::ItemBucket *Bucket,
                             bool NoAdvance = false)
    : StringMapConstIterator<ValueTy>(Bucket, NoAdvance) {
  }
  StringMapEntry<ValueTy> &operator*() const {
    return *static_cast<StringMapEntry<ValueTy>*>(this->Ptr->Item);
  }
  StringMapEntry<ValueTy> *operator->() const {
    return static_cast<StringMapEntry<ValueTy>*>(this->Ptr->Item);
  }
};

}

#endif
