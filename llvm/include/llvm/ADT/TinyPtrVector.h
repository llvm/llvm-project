//===- llvm/ADT/TinyPtrVector.h - 'Normally tiny' vectors -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_TINYPTRVECTOR_H
#define LLVM_ADT_TINYPTRVECTOR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include <cassert>
#include <cstddef>
#include <iterator>

namespace llvm {

/// TinyPtrVector - This class is specialized for cases where there are
/// normally 0 or 1 element in a vector, but is general enough to go beyond that
/// when required.
///
template <typename EltTy> class LLVM_ATTRIBUTE_WARN_UNUSED TinyPtrVector {
public:
  using VecTy = SmallVector<EltTy, 4>;
  using value_type = typename VecTy::value_type;
  // A TinyPtrVector has three different possible representations internally:
  // - A null VecTy *. This is an empty vector.
  // - An EltTy. This is a one-element vector.
  // - A non-null VecTy *. This is a pointer to a SmallVector (which may have
  //   any number of elements, including zero or one).
  //
  // EltTy must be the first pointer type so we can expose begin()/end()
  // for a one-element vector using getAddrOfPtr1().
  using PtrUnion = PointerUnion<EltTy, VecTy *>;

private:
  static constexpr VecTy *EmptyVal = nullptr;

  bool isEmptyForm() const { return isa<VecTy *>(Val) && !cast<VecTy *>(Val); }
  bool isSingleElementForm() const { return isa<EltTy>(Val); }
  bool isMultiElementForm() const { return dyn_cast<VecTy *>(Val); }

  PtrUnion Val;

public:
  // FIXME: Add a constructor to PtrUnion that allows marking this constexpr.
  TinyPtrVector() : Val(EmptyVal) {}

  ~TinyPtrVector() {
    if (VecTy *V = dyn_cast<VecTy *>(Val))
      delete V;
  }

  TinyPtrVector(const TinyPtrVector &RHS) : Val(RHS.Val) {
    if (VecTy *V = dyn_cast<VecTy *>(Val))
      Val = new VecTy(*V);
  }

  TinyPtrVector &operator=(const TinyPtrVector &RHS) {
    if (this == &RHS)
      return *this;
    if (RHS.empty()) {
      this->clear();
      return *this;
    }

    // Try to squeeze into the single slot. If it won't fit, allocate a copied
    // vector.
    if (!isMultiElementForm()) {
      if (RHS.size() == 1)
        Val = RHS.front();
      else
        Val = new VecTy(*cast<VecTy *>(RHS.Val));
      return *this;
    }

    // If we have a full vector allocated, try to re-use it.
    if (RHS.isSingleElementForm()) {
      cast<VecTy *>(Val)->clear();
      cast<VecTy *>(Val)->push_back(RHS.front());
    } else {
      *cast<VecTy *>(Val) = *cast<VecTy *>(RHS.Val);
    }
    return *this;
  }

  TinyPtrVector(TinyPtrVector &&RHS) : Val(RHS.Val) { RHS.Val = EmptyVal; }

  TinyPtrVector &operator=(TinyPtrVector &&RHS) {
    if (this == &RHS)
      return *this;
    if (RHS.empty()) {
      this->clear();
      return *this;
    }

    // If this vector has been allocated on the heap, re-use it if cheap. If it
    // would require more copying, just delete it and we'll steal the other
    // side.
    if (VecTy *V = dyn_cast<VecTy *>(Val)) {
      if (isa<EltTy>(RHS.Val)) {
        V->clear();
        V->push_back(RHS.front());
        RHS.Val = EmptyVal;
        return *this;
      }
      delete V;
    }

    Val = RHS.Val;
    RHS.Val = EmptyVal;
    return *this;
  }

  TinyPtrVector(std::initializer_list<EltTy> IL)
      : Val(IL.size() == 0   ? PtrUnion(EmptyVal)
            : IL.size() == 1 ? PtrUnion(*IL.begin())
                             : PtrUnion(new VecTy(IL.begin(), IL.end()))) {}

  /// Constructor from an ArrayRef.
  ///
  /// This also is a constructor for individual array elements due to the single
  /// element constructor for ArrayRef.
  explicit TinyPtrVector(ArrayRef<EltTy> Elts)
      : Val(Elts.empty() ? PtrUnion(EmptyVal)
            : Elts.size() == 1
                ? PtrUnion(Elts[0])
                : PtrUnion(new VecTy(Elts.begin(), Elts.end()))) {}

  TinyPtrVector(size_t Count, EltTy Value)
      : Val(Count == 0   ? PtrUnion(EmptyVal)
            : Count == 1 ? PtrUnion(Value)
                         : PtrUnion(new VecTy(Count, Value))) {}

  bool empty() const {
    // This vector can be empty if it contains no element, or if it
    // contains a pointer to an empty vector.
    if (isSingleElementForm())
      return false;
    if (isEmptyForm())
      return true;
    return cast<VecTy *>(Val)->empty();
  }

  unsigned size() const {
    if (isSingleElementForm())
      return 1;
    if (isEmptyForm())
      return 0;
    return cast<VecTy *>(Val)->size();
  }

  using iterator = EltTy *;
  using const_iterator = const EltTy *;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  iterator begin() {
    if (isSingleElementForm())
      return Val.getAddrOfPtr1();

    if (isEmptyForm()) {
      // Any valid pointer with correct alignment works here. Using the same
      // value as the one-element case slightly reduces the complexity of the
      // generated code.
      return reinterpret_cast<EltTy *>(this);
    }

    return cast<VecTy *>(Val)->begin();
  }

  iterator end() {
    if (isSingleElementForm())
      return begin() + 1;

    if (isEmptyForm())
      return begin();

    return cast<VecTy *>(Val)->end();
  }

  const_iterator begin() const {
    return (const_iterator)const_cast<TinyPtrVector*>(this)->begin();
  }

  const_iterator end() const {
    return (const_iterator)const_cast<TinyPtrVector*>(this)->end();
  }

  reverse_iterator rbegin() { return reverse_iterator(end()); }
  reverse_iterator rend() { return reverse_iterator(begin()); }

  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }

  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  EltTy *data() { return begin(); }
  const EltTy *data() const { return begin(); }

  EltTy operator[](unsigned i) const {
    assert(!empty() && "vector empty");
    if (isSingleElementForm()) {
      assert(i == 0 && "tinyvector index out of range");
      return cast<EltTy>(Val);
    }

    assert(i < cast<VecTy *>(Val)->size() && "tinyvector index out of range");
    return (*cast<VecTy *>(Val))[i];
  }

  EltTy front() const {
    assert(!empty() && "vector empty");
    if (isSingleElementForm())
      return cast<EltTy>(Val);
    return cast<VecTy *>(Val)->front();
  }

  EltTy back() const {
    assert(!empty() && "vector empty");
    if (isSingleElementForm())
      return cast<EltTy>(Val);
    return cast<VecTy *>(Val)->back();
  }

  void push_back(EltTy NewVal) {
    // If we have nothing, add something.
    if (isEmptyForm()) {
      Val = NewVal;
      return;
    }

    // If we have a single value, convert to a vector.
    if (isSingleElementForm()) {
      EltTy V = cast<EltTy>(Val);
      Val = new VecTy({V, NewVal});
      return;
    }

    // Add the new value, we know we have a vector.
    cast<VecTy *>(Val)->push_back(NewVal);
  }

  void pop_back() {
    // If we have a single value, convert to empty.
    assert(!empty() && "vector empty");
    if (isSingleElementForm())
      Val = EmptyVal;
    else
      cast<VecTy *>(Val)->pop_back();
  }

  void clear() {
    // If we have a single value, convert to empty.
    if (isSingleElementForm()) {
      Val = EmptyVal;
    } else if (cast<VecTy *>(Val)) {
      // If we have a vector form, just clear it.
      cast<VecTy *>(Val)->clear();
    }
  }

  iterator erase(iterator I) {
    assert(I >= begin() && "Iterator to erase is out of bounds.");
    assert(I < end() && "Erasing at past-the-end iterator.");

    // If we have a single value, convert to empty.
    if (isSingleElementForm()) {
      if (I == begin())
        Val = EmptyVal;
    } else {
      // multiple items in a vector; just do the erase, there is no
      // benefit to collapsing back to a pointer
      return cast<VecTy *>(Val)->erase(I);
    }
    return end();
  }

  iterator erase(iterator S, iterator E) {
    assert(S >= begin() && "Range to erase is out of bounds.");
    assert(S <= E && "Trying to erase invalid range.");
    assert(E <= end() && "Trying to erase past the end.");

    if (isSingleElementForm()) {
      if (S == begin() && S != E)
        Val = EmptyVal;
    } else if (isMultiElementForm()) {
      return cast<VecTy *>(Val)->erase(S, E);
    }
    return end();
  }

  iterator insert(iterator I, const EltTy &Elt) {
    assert(I >= this->begin() && "Insertion iterator is out of bounds.");
    assert(I <= this->end() && "Inserting past the end of the vector.");
    if (I == end()) {
      push_back(Elt);
      return std::prev(end());
    }
    if (isSingleElementForm()) {
      EltTy V = cast<EltTy>(Val);
      assert(I == begin());
      Val = Elt;
      push_back(V);
      return begin();
    }

    return cast<VecTy *>(Val)->insert(I, Elt);
  }

  template<typename ItTy>
  iterator insert(iterator I, ItTy From, ItTy To) {
    assert(I >= this->begin() && "Insertion iterator is out of bounds.");
    assert(I <= this->end() && "Inserting past the end of the vector.");
    if (From == To)
      return I;

    ptrdiff_t Offset = I - begin();
    if (isEmptyForm()) {
      if (std::next(From) == To) {
        Val = *From;
        return begin();
      }

      Val = new VecTy();
    } else if (isSingleElementForm()) {
      EltTy V = cast<EltTy>(Val);
      Val = new VecTy();
      cast<VecTy *>(Val)->push_back(V);
    }
    return cast<VecTy *>(Val)->insert(begin() + Offset, From, To);
  }
};

} // end namespace llvm

#endif // LLVM_ADT_TINYPTRVECTOR_H
