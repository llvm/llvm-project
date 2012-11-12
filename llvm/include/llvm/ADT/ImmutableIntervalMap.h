//===--- ImmutableIntervalMap.h - Immutable (functional) map  ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ImmutableIntervalMap class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_IMMUTABLE_INTERVAL_MAP_H
#define LLVM_ADT_IMMUTABLE_INTERVAL_MAP_H

#include "llvm/ADT/ImmutableMap.h"

namespace llvm {

class Interval {
private:
  int64_t Start;
  int64_t End;

public:
  Interval(int64_t S, int64_t E) : Start(S), End(E) {}

  int64_t getStart() const { return Start; }
  int64_t getEnd() const { return End; }
};

template <typename T>
struct ImutIntervalInfo {
  typedef const std::pair<Interval, T> value_type;
  typedef const value_type &value_type_ref;
  typedef const Interval key_type;
  typedef const Interval &key_type_ref;
  typedef const T data_type;
  typedef const T &data_type_ref;

  static key_type_ref KeyOfValue(value_type_ref V) {
    return V.first;
  }

  static data_type_ref DataOfValue(value_type_ref V) {
    return V.second;
  }

  static bool isEqual(key_type_ref L, key_type_ref R) {
    return L.getStart() == R.getStart() && L.getEnd() == R.getEnd();
  }

  static bool isDataEqual(data_type_ref L, data_type_ref R) {
    return ImutContainerInfo<T>::isEqual(L,R);
  }

  static bool isLess(key_type_ref L, key_type_ref R) {
    // Assume L and R does not overlap.
    if (L.getStart() < R.getStart()) {
      assert(L.getEnd() < R.getStart());
      return true;
    } else if (L.getStart() == R.getStart()) {
      assert(L.getEnd() == R.getEnd());
      return false;
    } else {
      assert(L.getStart() > R.getEnd());
      return false;
    }
  }

  static bool isContainedIn(key_type_ref K, key_type_ref L) {
    if (K.getStart() >= L.getStart() && K.getEnd() <= L.getEnd())
      return true;
    else
      return false;
  }

  static void Profile(FoldingSetNodeID &ID, value_type_ref V) {
    ID.AddInteger(V.first.getStart());
    ID.AddInteger(V.first.getEnd());
    ImutProfileInfo<T>::Profile(ID, V.second);
  }
};

template <typename ImutInfo>
class ImutIntervalAVLFactory : public ImutAVLFactory<ImutInfo> {
  typedef ImutAVLTree<ImutInfo> TreeTy;
  typedef typename ImutInfo::value_type     value_type;
  typedef typename ImutInfo::value_type_ref value_type_ref;
  typedef typename ImutInfo::key_type       key_type;
  typedef typename ImutInfo::key_type_ref   key_type_ref;
  typedef typename ImutInfo::data_type      data_type;
  typedef typename ImutInfo::data_type_ref  data_type_ref;

public:
  ImutIntervalAVLFactory(BumpPtrAllocator &Alloc) 
    : ImutAVLFactory<ImutInfo>(Alloc) {}

  TreeTy *Add(TreeTy *T, value_type_ref V) {
    T = add_internal(V,T);
    this->MarkImmutable(T);
    return T;
  }

  TreeTy *Find(TreeTy *T, key_type_ref K) {
    if (!T)
      return NULL;

    key_type_ref CurrentKey = ImutInfo::KeyOfValue(this->getValue(T));

    if (ImutInfo::isContainedIn(K, CurrentKey))
      return T;
    else if (ImutInfo::isLess(K, CurrentKey))
      return Find(this->getLeft(T), K);
    else
      return Find(this->getRight(T), K);
  }

private:
  TreeTy *add_internal(value_type_ref V, TreeTy *T) {
    key_type_ref K = ImutInfo::KeyOfValue(V);
    T = removeAllOverlaps(T, K);
    if (this->isEmpty(T))
      return this->CreateNode(NULL, V, NULL);

    assert(!T->isMutable());

    key_type_ref KCurrent = ImutInfo::KeyOfValue(this->Value(T));

    if (ImutInfo::isLess(K, KCurrent))
      return this->Balance(add_internal(V, this->Left(T)), this->Value(T), 
                                        this->Right(T));
    else
      return this->Balance(this->Left(T), this->Value(T), 
                           add_internal(V, this->Right(T)));
  }

  // Remove all overlaps from T.
  TreeTy *removeAllOverlaps(TreeTy *T, key_type_ref K) {
    bool Changed;
    do {
      Changed = false;
      T = removeOverlap(T, K, Changed);
      this->markImmutable(T);
    } while (Changed);

    return T;
  }

  // Remove one overlap from T.
  TreeTy *removeOverlap(TreeTy *T, key_type_ref K, bool &Changed) {
    if (!T)
      return NULL;
    Interval CurrentK = ImutInfo::KeyOfValue(this->Value(T));

    // If current key does not overlap the inserted key.
    if (CurrentK.getStart() > K.getEnd())
      return this->Balance(removeOverlap(this->Left(T), K, Changed),
                           this->Value(T), this->Right(T));
    else if (CurrentK.getEnd() < K.getStart())
      return this->Balance(this->Left(T), this->Value(T), 
                           removeOverlap(this->Right(T), K, Changed));

    // Current key overlaps with the inserted key.
    // Remove the current key.
    Changed = true;
    data_type_ref OldData = ImutInfo::DataOfValue(this->Value(T));
    T = this->Remove_internal(CurrentK, T);
    // Add back the unoverlapped part of the current key.
    if (CurrentK.getStart() < K.getStart()) {
      if (CurrentK.getEnd() <= K.getEnd()) {
        Interval NewK(CurrentK.getStart(), K.getStart()-1);
        return add_internal(std::make_pair(NewK, OldData), T);
      } else {
        Interval NewK1(CurrentK.getStart(), K.getStart()-1);
        T = add_internal(std::make_pair(NewK1, OldData), T); 

        Interval NewK2(K.getEnd()+1, CurrentK.getEnd());
        return add_internal(std::make_pair(NewK2, OldData), T);
      }
    } else {
      if (CurrentK.getEnd() > K.getEnd()) {
        Interval NewK(K.getEnd()+1, CurrentK.getEnd());
        return add_internal(std::make_pair(NewK, OldData), T);
      } else
        return T;
    }
  }
};

/// ImmutableIntervalMap maps an interval [start, end] to a value. The intervals
/// in the map are guaranteed to be disjoint.
template <typename ValT>
class ImmutableIntervalMap 
  : public ImmutableMap<Interval, ValT, ImutIntervalInfo<ValT> > {

  typedef typename ImutIntervalInfo<ValT>::value_type      value_type;
  typedef typename ImutIntervalInfo<ValT>::value_type_ref  value_type_ref;
  typedef typename ImutIntervalInfo<ValT>::key_type        key_type;
  typedef typename ImutIntervalInfo<ValT>::key_type_ref    key_type_ref;
  typedef typename ImutIntervalInfo<ValT>::data_type       data_type;
  typedef typename ImutIntervalInfo<ValT>::data_type_ref   data_type_ref;
  typedef ImutAVLTree<ImutIntervalInfo<ValT> > TreeTy;

public:
  explicit ImmutableIntervalMap(TreeTy *R) 
    : ImmutableMap<Interval, ValT, ImutIntervalInfo<ValT> >(R) {}

  class Factory {
    ImutIntervalAVLFactory<ImutIntervalInfo<ValT> > F;

  public:
    Factory(BumpPtrAllocator& Alloc) : F(Alloc) {}

    ImmutableIntervalMap getEmptyMap() { 
      return ImmutableIntervalMap(F.getEmptyTree()); 
    }

    ImmutableIntervalMap add(ImmutableIntervalMap Old, 
                             key_type_ref K, data_type_ref D) {
      TreeTy *T = F.add(Old.Root, std::pair<key_type, data_type>(K, D));
      return ImmutableIntervalMap(F.getCanonicalTree(T));
    }

    ImmutableIntervalMap remove(ImmutableIntervalMap Old, key_type_ref K) {
      TreeTy *T = F.remove(Old.Root, K);
      return ImmutableIntervalMap(F.getCanonicalTree(T));
    }

    data_type *lookup(ImmutableIntervalMap M, key_type_ref K) {
      TreeTy *T = F.Find(M.getRoot(), K);
      if (T)
        return &T->getValue().second;
      else
        return 0;
    }
  };

private:
  // For ImmutableIntervalMap, the lookup operation has to be done by the 
  // factory.
  data_type* lookup(key_type_ref K) const;
};

} // end namespace llvm

#endif
