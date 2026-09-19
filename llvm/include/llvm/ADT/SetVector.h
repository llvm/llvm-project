//===- llvm/ADT/SetVector.h - Set with insert order iteration ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a set that has insertion order iteration
/// characteristics. This is useful for keeping a set of things that need to be
/// visited later but in a deterministic order (insertion order). The interface
/// is purposefully minimal.
///
/// This file defines SetVector and SmallSetVector, which performs no
/// allocations if the SetVector has less than a certain number of elements.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SETVECTOR_H
#define LLVM_ADT_SETVECTOR_H

#include "llvm/ADT/ADL.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include <cassert>
#include <initializer_list>

namespace llvm {

/// A vector that has set insertion semantics.
///
/// This adapter class provides a way to keep a set of things that also has the
/// property of a deterministic iteration order. The order of iteration is the
/// order of insertion.
///
/// The key and value types are derived from the Set and Vector types
/// respectively. This allows the vector-type operations and set-type operations
/// to have different types.
///
/// No constraint is placed on the key and value types, although it is assumed
/// that value_type can be converted into key_type for insertion. Users must be
/// aware of any loss of information in this conversion. For example, setting
/// value_type to float and key_type to int can produce very surprising results,
/// but it is not explicitly disallowed.
///
/// The parameter N specifies the "small" size of the container, which is the
/// number of elements upto which a linear scan over the Vector will be used
/// when searching for elements instead of checking Set, due to it being better
/// for performance. A value of 0 means that this mode of operation is not used,
/// and is the default value.
namespace detail {

template <typename T>
struct alignas(SmallVectorAlignmentAndSize<T>) SetVectorSmallVectorImpl
    : public SmallVectorImpl<T> {
  using SmallVectorImpl<T>::getFirstEl;
  using SmallVectorImpl<T>::isSmall;

  explicit SetVectorSmallVectorImpl(unsigned N) : SmallVectorImpl<T>(N) {}
  ~SetVectorSmallVectorImpl() = default;
  SetVectorSmallVectorImpl(const SetVectorSmallVectorImpl &) = delete;
  SetVectorSmallVectorImpl &
  operator=(const SetVectorSmallVectorImpl &) = delete;

  SetVectorSmallVectorImpl &operator=(SetVectorSmallVectorImpl &&RHS) {
    SmallVectorImpl<T>::operator=(std::move(RHS));
    return *this;
  }

  void destroy_elements() { this->destroy_range(this->begin(), this->end()); }

  void resetInlineCapacity(unsigned N) {
    if (this->isSmall() && this->capacity() == 0)
      this->set_allocation_range(this->getFirstEl(), N);
  }
};

template <typename T, typename Vector> struct SetVectorVectorStorage {
  using type = Vector;
};

template <typename T> struct SetVectorVectorStorage<T, SmallVectorImpl<T>> {
  using type = SetVectorSmallVectorImpl<T>;
};

template <typename T, typename Vector, typename Set, unsigned N = 0>
class SetVectorBase {
  static_assert(N <= 32, "Small size should be less than or equal to 32!");

  using const_arg_type =
      typename const_pointer_or_const_ref<typename Set::key_type>::type;

protected:
  using vector_storage_type = typename SetVectorVectorStorage<T, Vector>::type;

  SetVectorBase() = default;
  explicit SetVectorBase(unsigned InlineCapacity) : vector_(InlineCapacity) {}
  SetVectorBase(const SetVectorBase &) = default;
  SetVectorBase(SetVectorBase &&) = default;
  SetVectorBase &operator=(const SetVectorBase &) = default;
  SetVectorBase &operator=(SetVectorBase &&) = default;
  ~SetVectorBase() = default;

  void copyFrom(const SetVectorBase &RHS) {
    set_.clear();
    vector_.assign(RHS.vector_.begin(), RHS.vector_.end());
    if constexpr (std::is_base_of_v<SmallVectorImpl<T>, Vector>) {
      if (!vector_.isSmall() || vector_.size() > 32)
        makeBig();
    } else if constexpr (canBeSmall()) {
      if (vector_.size() > N)
        makeBig();
    } else {
      set_ = RHS.set_;
    }
  }

  void moveFrom(SetVectorBase &&RHS) {
    set_.clear();
    if constexpr (std::is_base_of_v<SmallVectorImpl<T>, Vector>) {
      if (!RHS.vector_.isSmall()) {
        set_ = std::move(RHS.set_);
        vector_ = std::move(RHS.vector_);
        RHS.set_.clear();
      } else {
        vector_ = std::move(RHS.vector_);
        RHS.set_.clear();
        if (!vector_.isSmall() || vector_.size() > 32)
          makeBig();
      }
    } else {
      set_ = std::move(RHS.set_);
      vector_ = std::move(RHS.vector_);
    }
  }

public:
  using value_type = typename Vector::value_type;
  using key_type = typename Set::key_type;
  using reference = value_type &;
  using const_reference = const value_type &;
  using set_type = Set;
  using vector_type = Vector;
  using iterator = typename vector_type::const_iterator;
  using const_iterator = typename vector_type::const_iterator;
  using reverse_iterator = typename vector_type::const_reverse_iterator;
  using const_reverse_iterator = typename vector_type::const_reverse_iterator;
  using size_type = typename vector_type::size_type;

  [[nodiscard]] ArrayRef<value_type> getArrayRef() const { return vector_; }

  /// Determine if the SetVector is empty or not.
  [[nodiscard]] bool empty() const { return vector_.empty(); }

  /// Determine the number of elements in the SetVector.
  [[nodiscard]] size_type size() const { return vector_.size(); }

  /// Reserve space in the SetVector if supported by the underlying containers.
  void reserve(size_type Size) {
    vector_.reserve(Size);
    set_.reserve(Size);
  }

  /// Get an iterator to the beginning of the SetVector.
  [[nodiscard]] iterator begin() { return vector_.begin(); }

  /// Get a const_iterator to the beginning of the SetVector.
  [[nodiscard]] const_iterator begin() const { return vector_.begin(); }

  /// Get an iterator to the end of the SetVector.
  [[nodiscard]] iterator end() { return vector_.end(); }

  /// Get a const_iterator to the end of the SetVector.
  [[nodiscard]] const_iterator end() const { return vector_.end(); }

  /// Get an reverse_iterator to the end of the SetVector.
  [[nodiscard]] reverse_iterator rbegin() { return vector_.rbegin(); }

  /// Get a const_reverse_iterator to the end of the SetVector.
  [[nodiscard]] const_reverse_iterator rbegin() const {
    return vector_.rbegin();
  }

  /// Get a reverse_iterator to the beginning of the SetVector.
  [[nodiscard]] reverse_iterator rend() { return vector_.rend(); }

  /// Get a const_reverse_iterator to the beginning of the SetVector.
  [[nodiscard]] const_reverse_iterator rend() const { return vector_.rend(); }

  /// Return the first element of the SetVector.
  [[nodiscard]] const value_type &front() const {
    assert(!empty() && "Cannot call front() on empty SetVector!");
    return vector_.front();
  }

  /// Return the last element of the SetVector.
  [[nodiscard]] const value_type &back() const {
    assert(!empty() && "Cannot call back() on empty SetVector!");
    return vector_.back();
  }

  /// Index into the SetVector.
  const_reference operator[](size_type n) const {
    assert(n < vector_.size() && "SetVector access out of range!");
    return vector_[n];
  }

  /// Insert a new element into the SetVector.
  /// \returns true if the element was inserted into the SetVector.
  bool insert(const value_type &X) {
    if constexpr (canBeSmall())
      if (isSmall()) {
        if (!llvm::is_contained(vector_, X)) {
          vector_.push_back(X);
          if constexpr (std::is_base_of_v<SmallVectorImpl<T>, Vector>) {
            if (!vector_.isSmall() || vector_.size() > 32)
              makeBig();
          } else {
            if (vector_.size() > N)
              makeBig();
          }
          return true;
        }
        return false;
      }

    bool result = set_.insert(X).second;
    if (result)
      vector_.push_back(X);
    return result;
  }

  /// Insert a range of elements into the SetVector.
  template<typename It>
  void insert(It Start, It End) {
    for (; Start != End; ++Start)
      insert(*Start);
  }

  template <typename Range> void insert_range(Range &&R) {
    insert(adl_begin(R), adl_end(R));
  }

  /// Remove an item from the set vector.
  bool remove(const value_type& X) {
    if constexpr (canBeSmall())
      if (isSmall()) {
        typename vector_type::iterator I = find(vector_, X);
        if (I != vector_.end()) {
          vector_.erase(I);
          return true;
        }
        return false;
      }

    if (set_.erase(X)) {
      typename vector_type::iterator I = find(vector_, X);
      assert(I != vector_.end() && "Corrupted SetVector instances!");
      vector_.erase(I);
      return true;
    }
    return false;
  }

  /// Erase a single element from the set vector.
  /// \returns an iterator pointing to the next element that followed the
  /// element erased. This is the end of the SetVector if the last element is
  /// erased.
  iterator erase(const_iterator I) {
    if constexpr (canBeSmall())
      if (isSmall())
        return vector_.erase(I);

    const key_type &V = *I;
    assert(set_.count(V) && "Corrupted SetVector instances!");
    set_.erase(V);
    return vector_.erase(I);
  }

  /// Remove items from the set vector based on a predicate function.
  ///
  /// This is intended to be equivalent to the following code, if we could
  /// write it:
  ///
  /// \code
  ///   V.erase(remove_if(V, P), V.end());
  /// \endcode
  ///
  /// However, SetVector doesn't expose non-const iterators, making any
  /// algorithm like remove_if impossible to use.
  ///
  /// \returns true if any element is removed.
  template <typename UnaryPredicate>
  bool remove_if(UnaryPredicate P) {
    typename vector_type::iterator I = [this, P] {
      if constexpr (canBeSmall())
        if (isSmall())
          return llvm::remove_if(vector_, P);

      return llvm::remove_if(vector_, [&](const value_type &V) {
        if (P(V)) {
          set_.erase(V);
          return true;
        }
        return false;
      });
    }();

    if (I == vector_.end())
      return false;
    vector_.erase(I, vector_.end());
    return true;
  }

  /// Check if the SetVector contains the given key.
  [[nodiscard]] bool contains(const_arg_type key) const {
    if constexpr (canBeSmall())
      if (isSmall())
        return is_contained(vector_, key);

    return is_contained(set_, key);
  }

  /// Count the number of elements of a given key in the SetVector.
  /// \returns 0 if the element is not in the SetVector, 1 if it is.
  [[nodiscard]] size_type count(const_arg_type key) const {
    return contains(key) ? 1 : 0;
  }

  /// Completely clear the SetVector
  void clear() {
    set_.clear();
    vector_.clear();
  }

  /// Remove the last element of the SetVector.
  void pop_back() {
    assert(!empty() && "Cannot remove an element from an empty SetVector!");
    if (!isSmall())
      set_.erase(back());
    vector_.pop_back();
  }

  [[nodiscard]] value_type pop_back_val() {
    value_type Ret = back();
    pop_back();
    return Ret;
  }

  [[nodiscard]] bool operator==(const SetVectorBase &that) const {
    return vector_ == that.vector_;
  }

  [[nodiscard]] bool operator!=(const SetVectorBase &that) const {
    return vector_ != that.vector_;
  }

  /// Compute This := This u S, return whether 'This' changed.
  /// TODO: We should be able to use set_union from SetOperations.h, but
  ///       SetVector interface is inconsistent with DenseSet.
  template <class STy>
  bool set_union(const STy &S) {
    bool Changed = false;

    for (const auto &Elem : S)
      if (insert(Elem))
        Changed = true;

    return Changed;
  }

  /// Compute This := This - B
  /// TODO: We should be able to use set_subtract from SetOperations.h, but
  ///       SetVector interface is inconsistent with DenseSet.
  template <class STy>
  void set_subtract(const STy &S) {
    for (const auto &Elem : S)
      remove(Elem);
  }

  void swap(SetVectorBase &RHS) {
    set_.swap(RHS.set_);
    vector_.swap(RHS.vector_);
    if constexpr (std::is_base_of_v<SmallVectorImpl<T>, Vector>) {
      if (!vector_.isSmall() && isSmall())
        makeBig();
      if (!RHS.vector_.isSmall() && RHS.isSmall())
        RHS.makeBig();
    }
  }

protected:
  [[nodiscard]] static constexpr bool canBeSmall() {
    return std::is_base_of_v<SmallVectorImpl<T>, Vector> || N != 0;
  }

  [[nodiscard]] bool isSmall() const { return set_.empty(); }

  void makeBig() {
    if constexpr (canBeSmall())
      for (const auto &entry : vector_)
        set_.insert(entry);
  }

  set_type set_;               ///< The set.
  vector_storage_type vector_; ///< The vector.
};

} // end namespace detail

template <typename T, typename Vector = SmallVector<T, 0>,
          typename Set = DenseSet<T>, unsigned N = 0>
class SetVector : public detail::SetVectorBase<T, Vector, Set, N> {
  using Base = detail::SetVectorBase<T, Vector, Set, N>;

public:
  /// Construct an empty SetVector
  SetVector() = default;

  /// Initialize a SetVector with a range of elements
  template <typename It> SetVector(It Start, It End) {
    this->insert(Start, End);
  }

  template <typename Range>
  SetVector(llvm::from_range_t, Range &&R)
      : SetVector(adl_begin(R), adl_end(R)) {}

  SetVector(std::initializer_list<T> IL) { this->insert(IL.begin(), IL.end()); }

  /// Clear the SetVector and return the underlying vector.
  [[nodiscard]] Vector takeVector() {
    this->set_.clear();
    return std::move(this->vector_);
  }
};

template <typename T, typename Set>
class SetVector<T, SmallVectorImpl<T>, Set, 0>
    : public detail::SetVectorBase<T, SmallVectorImpl<T>, Set, 0> {
  using Base = detail::SetVectorBase<T, SmallVectorImpl<T>, Set, 0>;

  static_assert(alignof(Set) <= alignof(detail::SetVectorSmallVectorImpl<T>),
                "Set alignment must not exceed vector alignment");

protected:
  explicit SetVector(unsigned InlineCapacity) : Base(InlineCapacity) {}
  SetVector(const SetVector &) = default;
  SetVector(SetVector &&) = default;
  SetVector &operator=(const SetVector &) = default;
  SetVector &operator=(SetVector &&) = default;
  ~SetVector() = default;

public:
  template <unsigned RetN = 0> [[nodiscard]] SmallVector<T, RetN> takeVector() {
    this->set_.clear();
    return SmallVector<T, RetN>(std::move(this->vector_));
  }
};

template <typename T, unsigned VecN, typename Set, unsigned N>
class SetVector<T, SmallVector<T, VecN>, Set, N>
    : public SetVector<T, SmallVectorImpl<T>, Set, 0>,
      private SmallVectorStorage<T, VecN> {
  static_assert(VecN <= 32, "Small size should be less than or equal to 32!");
  static_assert(N <= 32, "Small size should be less than or equal to 32!");
  using Base = SetVector<T, SmallVectorImpl<T>, Set, 0>;

  void checkStorageLayout() const {
    if constexpr (VecN > 0)
      assert(this->vector_.getFirstEl() ==
                 static_cast<const void *>(
                     static_cast<const SmallVectorStorage<T, VecN> *>(this)) &&
             "SmallSetVector inline storage layout mismatch!");
  }

public:
  using vector_type = SmallVector<T, VecN>;

  SetVector() : Base(VecN) { checkStorageLayout(); }

  template <typename It> SetVector(It Start, It End) : Base(VecN) {
    checkStorageLayout();
    this->insert(Start, End);
  }

  template <typename Range>
  SetVector(llvm::from_range_t, Range &&R) : Base(VecN) {
    checkStorageLayout();
    this->insert_range(std::forward<Range>(R));
  }

  SetVector(std::initializer_list<T> IL) : Base(VecN) {
    checkStorageLayout();
    this->insert(IL.begin(), IL.end());
  }

  SetVector(const SetVector &RHS) : Base(VecN) {
    checkStorageLayout();
    this->copyFrom(RHS);
  }

  SetVector(const Base &RHS) : Base(VecN) {
    checkStorageLayout();
    this->copyFrom(RHS);
  }

  SetVector(SetVector &&RHS) : Base(VecN) {
    checkStorageLayout();
    this->moveFrom(std::move(RHS));
  }

  SetVector(Base &&RHS) : Base(VecN) {
    checkStorageLayout();
    this->moveFrom(std::move(RHS));
  }

  SetVector &operator=(const SetVector &RHS) {
    if (this != &RHS)
      this->copyFrom(RHS);
    return *this;
  }

  SetVector &operator=(const Base &RHS) {
    if (this != &RHS)
      this->copyFrom(RHS);
    return *this;
  }

  SetVector &operator=(SetVector &&RHS) {
    if (this != &RHS)
      this->moveFrom(std::move(RHS));
    return *this;
  }

  SetVector &operator=(Base &&RHS) {
    if (this != &RHS)
      this->moveFrom(std::move(RHS));
    return *this;
  }

  ~SetVector() { this->vector_.destroy_elements(); }

  [[nodiscard]] SmallVector<T, VecN> takeVector() {
    this->set_.clear();
    SmallVector<T, VecN> Ret(std::move(this->vector_));
    this->vector_.resetInlineCapacity(VecN);
    return Ret;
  }
};

/// Common base class for SmallSetVector and SetVector that erases the inline
/// capacity N from the type, similar to SmallVectorImpl.
template <typename T, typename Set = DenseSet<T>>
using SmallSetVectorImpl = SetVector<T, SmallVectorImpl<T>, Set, 0>;

/// A SetVector that performs no allocations if smaller than
/// a certain size.
template <typename T, unsigned N>
class SmallSetVector : public SetVector<T, SmallVector<T, N>, DenseSet<T>, N> {
public:
  using SetVector<T, SmallVector<T, N>, DenseSet<T>, N>::SetVector;
};

} // end namespace llvm

namespace std {

/// Implement std::swap in terms of SetVector swap.
template <typename T, typename V, typename S, unsigned N>
inline void swap(llvm::SetVector<T, V, S, N> &LHS,
                 llvm::SetVector<T, V, S, N> &RHS) {
  LHS.swap(RHS);
}

/// Implement std::swap in terms of SmallSetVector swap.
template<typename T, unsigned N>
inline void
swap(llvm::SmallSetVector<T, N> &LHS, llvm::SmallSetVector<T, N> &RHS) {
  LHS.swap(RHS);
}

} // end namespace std

#endif // LLVM_ADT_SETVECTOR_H
