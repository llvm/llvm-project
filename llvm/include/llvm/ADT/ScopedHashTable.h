//===- ScopedHashTable.h - A simple scoped hash table ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements an efficient scoped hash table, which is useful for
// things like dominator-based optimizations.  This allows clients to do things
// like this:
//
//  ScopedHashTable<int, int> HT;
//  {
//    ScopedHashTableScope<int, int> Scope1(HT);
//    HT.insert(0, 0);
//    HT.insert(1, 1);
//    {
//      ScopedHashTableScope<int, int> Scope2(HT);
//      HT.insert(0, 42);
//    }
//  }
//
// Looking up the value for "0" in the Scope2 block will return 42.  Looking
// up the value for 0 before 42 is inserted or after Scope2 is popped will
// return 0.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SCOPEDHASHTABLE_H
#define LLVM_ADT_SCOPEDHASHTABLE_H

#include <cassert>
#include "llvm/ADT/DenseMap.h"

namespace llvm {

template <typename K, typename V>
class ScopedHashTable;

template <typename K, typename V>
class ScopedHashTableVal {
  ScopedHashTableVal *NextInScope;
  ScopedHashTableVal *NextForKey;
  K Key;
  V Val;
public:
  ScopedHashTableVal(ScopedHashTableVal *nextInScope,
                     ScopedHashTableVal *nextForKey, const K &key, const V &val)
    : NextInScope(nextInScope), NextForKey(nextForKey), Key(key), Val(val) {
  }

  const K &getKey() const { return Key; }
  const V &getValue() const { return Val; }
  V &getValue() { return Val; }

  ScopedHashTableVal *getNextForKey() { return NextForKey; }
  const ScopedHashTableVal *getNextForKey() const { return NextForKey; }
public:
  ScopedHashTableVal *getNextInScope() { return NextInScope; }
};

template <typename K, typename V>
class ScopedHashTableScope {
  /// HT - The hashtable that we are active for.
  ScopedHashTable<K, V> &HT;

  /// PrevScope - This is the scope that we are shadowing in HT.
  ScopedHashTableScope *PrevScope;

  /// LastValInScope - This is the last value that was inserted for this scope
  /// or null if none have been inserted yet.
  ScopedHashTableVal<K,V> *LastValInScope;
  void operator=(ScopedHashTableScope&);       // DO NOT IMPLEMENT
  ScopedHashTableScope(ScopedHashTableScope&); // DO NOT IMPLEMENT
public:
  ScopedHashTableScope(ScopedHashTable<K, V> &HT);
  ~ScopedHashTableScope();

private:
  friend class ScopedHashTable<K, V>;
  ScopedHashTableVal<K, V> *getLastValInScope() { return LastValInScope; }
  void setLastValInScope(ScopedHashTableVal<K,V> *Val) { LastValInScope = Val; }
};


template <typename K, typename V>
class ScopedHashTableIterator {
  ScopedHashTableVal<K,V> *Node;
public:
  ScopedHashTableIterator(ScopedHashTableVal<K,V> *node) : Node(node){}

  V &operator*() const {
    assert(Node && "Dereference end()");
    return Node->getValue();
  }
  V *operator->() const {
    return &Node->getValue();
  }

  bool operator==(const ScopedHashTableIterator &RHS) const {
    return Node == RHS.Node;
  }
  bool operator!=(const ScopedHashTableIterator &RHS) const {
    return Node != RHS.Node;
  }

  inline ScopedHashTableIterator& operator++() {          // Preincrement
    assert(Node && "incrementing past end()");
    Node = Node->getNextForKey();
    return *this;
  }
  ScopedHashTableIterator operator++(int) {        // Postincrement
    ScopedHashTableIterator tmp = *this; ++*this; return tmp;
  }
};


template <typename K, typename V>
class ScopedHashTable {
  DenseMap<K, ScopedHashTableVal<K,V>*> TopLevelMap;
  ScopedHashTableScope<K, V> *CurScope;
  ScopedHashTable(const ScopedHashTable&); // NOT YET IMPLEMENTED
  void operator=(const ScopedHashTable&);  // NOT YET IMPLEMENTED
  friend class ScopedHashTableScope<K, V>;
public:
  ScopedHashTable() : CurScope(0) {}
  ~ScopedHashTable() {
    assert(CurScope == 0 && TopLevelMap.empty() && "Scope imbalance!");
  }

  void insert(const K &Key, const V &Val) {
    assert(CurScope && "No scope active!");

    ScopedHashTableVal<K,V> *&KeyEntry = TopLevelMap[Key];

    KeyEntry = new ScopedHashTableVal<K,V>(CurScope->getLastValInScope(),
                                           KeyEntry, Key, Val);
    CurScope->setLastValInScope(KeyEntry);
  }

  typedef ScopedHashTableIterator<K, V> iterator;

  iterator end() { return iterator(0); }

  iterator begin(const K &Key) {
    typename DenseMap<K, ScopedHashTableVal<K,V>*>::iterator I =
      TopLevelMap.find(Key);
    if (I == TopLevelMap.end()) return end();
    return iterator(I->second);
  }
};

/// ScopedHashTableScope ctor - Install this as the current scope for the hash
/// table.
template <typename K, typename V>
ScopedHashTableScope<K, V>::ScopedHashTableScope(ScopedHashTable<K, V> &ht)
  : HT(ht) {
  PrevScope = HT.CurScope;
  HT.CurScope = this;
  LastValInScope = 0;
}

template <typename K, typename V>
ScopedHashTableScope<K, V>::~ScopedHashTableScope() {
  assert(HT.CurScope == this && "Scope imbalance!");
  HT.CurScope = PrevScope;

  // Pop and delete all values corresponding to this scope.
  while (ScopedHashTableVal<K, V> *ThisEntry = LastValInScope) {
    // Pop this value out of the TopLevelMap.
    if (ThisEntry->getNextForKey() == 0) {
      assert(HT.TopLevelMap[ThisEntry->getKey()] == ThisEntry &&
             "Scope imbalance!");
      HT.TopLevelMap.erase(ThisEntry->getKey());
    } else {
      ScopedHashTableVal<K,V> *&KeyEntry = HT.TopLevelMap[ThisEntry->getKey()];
      assert(KeyEntry == ThisEntry && "Scope imbalance!");
      KeyEntry = ThisEntry->getNextForKey();
    }

    // Pop this value out of the scope.
    LastValInScope = ThisEntry->getNextInScope();

    // Delete this entry.
    delete ThisEntry;
  }
}

} // end namespace llvm

#endif
