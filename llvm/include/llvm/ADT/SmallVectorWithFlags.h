//===- SmallVectorWithFlags.h - Small vector with flags ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SMALLVECTORWITHFLAGS_H
#define LLVM_ADT_SMALLVECTORWITHFLAGS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

/// An N-independent interface for SmallVectorWithFlags. Element operations,
/// including assignment and swap through this base, preserve each vector's
/// flags. This is a separate interface from SmallVectorImpl<T>: an ordinary
/// SmallVectorImpl cannot interpret a capacity field holding flags.
template <typename T, unsigned FlagBits = 1>
class SmallVectorWithFlagsImpl
    : public detail::SmallVectorImplBase<T, FlagBits> {
  using Base = detail::SmallVectorImplBase<T, FlagBits>;

  static_assert(FlagBits > 0, "SmallVectorWithFlags needs at least one flag");

protected:
  explicit SmallVectorWithFlagsImpl(unsigned N) : Base(N) {}
  ~SmallVectorWithFlagsImpl() = default;

public:
  static constexpr unsigned NumFlagBits = FlagBits;

  /// Access the complete flag bit mask. Bits at or above NumFlagBits are
  /// undefined behavior in setFlags; debug builds assert on them.
  using Base::getFlags;
  using Base::setFlags;

  bool getFlag(unsigned Index) const {
    assert(Index < FlagBits && "flag index out of range");
    return (getFlags() >> Index) & 1;
  }

  void setFlag(unsigned Index, bool Value) {
    assert(Index < FlagBits && "flag index out of range");
    unsigned Mask = 1u << Index;
    setFlags(Value ? getFlags() | Mask : getFlags() & ~Mask);
  }

  SmallVectorWithFlagsImpl(const SmallVectorWithFlagsImpl &) = delete;

  SmallVectorWithFlagsImpl &operator=(const SmallVectorWithFlagsImpl &RHS) {
    Base::operator=(RHS);
    return *this;
  }

  SmallVectorWithFlagsImpl &operator=(SmallVectorWithFlagsImpl &&RHS) {
    Base::operator=(std::move(RHS));
    return *this;
  }

  bool operator==(const SmallVectorWithFlagsImpl &RHS) const {
    return getFlags() == RHS.getFlags() && Base::operator==(RHS);
  }

  bool operator!=(const SmallVectorWithFlagsImpl &RHS) const {
    return !(*this == RHS);
  }

  /// Order by elements first, then by flags when the elements are equivalent.
  bool operator<(const SmallVectorWithFlagsImpl &RHS) const {
    if (Base::operator<(RHS))
      return true;
    if (RHS.Base::operator<(*this))
      return false;
    return getFlags() < RHS.getFlags();
  }

  bool operator>(const SmallVectorWithFlagsImpl &RHS) const {
    return RHS < *this;
  }

  bool operator<=(const SmallVectorWithFlagsImpl &RHS) const {
    return !(RHS < *this);
  }

  bool operator>=(const SmallVectorWithFlagsImpl &RHS) const {
    return !(*this < RHS);
  }
};

/// A SmallVector with one to four user flag bits stored in the high bits of
/// its capacity field, without increasing its object size or inline capacity.
/// This packs a vector and a few bools that would otherwise be padded to a
/// word. The maximum capacity is divided by 2^FlagBits: with a 32-bit size
/// type and four flag bits, 2^28 - 1 elements.
///
/// Flags initially contain zero. Element operations preserve flags. Copying or
/// moving a complete SmallVectorWithFlags copies its flags; moving leaves the
/// source flags unchanged. Swapping complete containers exchanges flags as well
/// as elements. Equality compares both elements and flags. Ordering compares
/// elements lexicographically, then flags. ArrayRef conversion exposes only the
/// elements; use it explicitly when an element-only comparison is intended.
///
/// This container shares SmallVector's lack of exception safety. Use ArrayRef
/// for read-only interfaces. Output parameters can use
/// SmallVectorWithFlagsImpl<T, FlagBits> independently of the inline capacity.
template <typename T,
          unsigned N = CalculateSmallVectorDefaultInlinedElements<T>::value,
          unsigned FlagBits = 1>
class LLVM_GSL_OWNER SmallVectorWithFlags
    : public SmallVectorWithFlagsImpl<T, FlagBits>,
      public SmallVectorStorage<T, N> {
  using Base = SmallVectorWithFlagsImpl<T, FlagBits>;

  static_assert(N <= Base::SizeTypeMax(), "inline capacity exceeds maximum");

public:
  SmallVectorWithFlags() : Base(N) {}

  ~SmallVectorWithFlags() { this->destroy_range(this->begin(), this->end()); }

  explicit SmallVectorWithFlags(size_t Size) : Base(N) { this->resize(Size); }

  SmallVectorWithFlags(size_t Size, const T &Value) : Base(N) {
    this->assign(Size, Value);
  }

  template <typename It, typename = EnableIfConvertibleToInputIterator<It>>
  SmallVectorWithFlags(It Begin, It End) : Base(N) {
    this->append(Begin, End);
  }

  template <typename RangeT>
  explicit SmallVectorWithFlags(const iterator_range<RangeT> &Range) : Base(N) {
    this->append(Range.begin(), Range.end());
  }

  SmallVectorWithFlags(std::initializer_list<T> Values) : Base(N) {
    this->append(Values);
  }

  template <typename U,
            typename = std::enable_if_t<std::is_convertible_v<U, T>>>
  explicit SmallVectorWithFlags(ArrayRef<U> Values) : Base(N) {
    this->append(Values.begin(), Values.end());
  }

  /// Moving from the N-independent interface takes the flags as well, since a
  /// new container has none of its own yet.
  SmallVectorWithFlags(Base &&RHS) : Base(N) {
    if (!RHS.empty()) {
      // See SmallVector(SmallVectorImpl &&) for the rationale of this branch.
      if constexpr (std::is_trivially_move_assignable_v<T>)
        Base::operator=(std::move(RHS));
      else
        this->moveConstructFrom(std::move(RHS));
    }
    this->setFlags(RHS.getFlags());
  }

  SmallVectorWithFlags(const SmallVectorWithFlags &RHS) : Base(N) {
    if (!RHS.empty())
      Base::operator=(RHS);
    this->setFlags(RHS.getFlags());
  }

  SmallVectorWithFlags(SmallVectorWithFlags &&RHS)
      : SmallVectorWithFlags(static_cast<Base &&>(RHS)) {}

  SmallVectorWithFlags &operator=(const SmallVectorWithFlags &RHS) {
    Base::operator=(RHS);
    this->setFlags(RHS.getFlags());
    return *this;
  }

  SmallVectorWithFlags &operator=(SmallVectorWithFlags &&RHS) {
    Base::operator=(std::move(RHS));
    this->setFlags(RHS.getFlags());
    return *this;
  }

  template <unsigned M>
  SmallVectorWithFlags &
  operator=(const SmallVectorWithFlags<T, M, FlagBits> &RHS) {
    Base::operator=(RHS);
    this->setFlags(RHS.getFlags());
    return *this;
  }

  template <unsigned M>
  SmallVectorWithFlags &operator=(SmallVectorWithFlags<T, M, FlagBits> &&RHS) {
    Base::operator=(std::move(RHS));
    this->setFlags(RHS.getFlags());
    return *this;
  }

  /// Assigning through the N-independent interface updates only the elements.
  SmallVectorWithFlags &operator=(const Base &RHS) {
    Base::operator=(RHS);
    return *this;
  }

  SmallVectorWithFlags &operator=(Base &&RHS) {
    Base::operator=(std::move(RHS));
    return *this;
  }

  SmallVectorWithFlags &operator=(std::initializer_list<T> Values) {
    this->assign(Values);
    return *this;
  }

  /// Swapping with the N-independent interface keeps flags with their owners.
  using Base::swap;

  /// Swapping complete containers exchanges the flags along with the elements,
  /// even across inline capacities.
  template <unsigned M> void swap(SmallVectorWithFlags<T, M, FlagBits> &RHS) {
    Base::swap(RHS);
    unsigned Flags = this->getFlags();
    this->setFlags(RHS.getFlags());
    RHS.setFlags(Flags);
  }
};

template <typename T, unsigned N, unsigned FlagBits>
void swap(SmallVectorWithFlags<T, N, FlagBits> &LHS,
          SmallVectorWithFlags<T, N, FlagBits> &RHS) {
  LHS.swap(RHS);
}

template <typename T, unsigned N, unsigned FlagBits>
size_t capacity_in_bytes(const SmallVectorWithFlags<T, N, FlagBits> &V) {
  return V.capacity_in_bytes();
}

template <typename T, unsigned FlagBits>
ArrayRef(const SmallVectorWithFlagsImpl<T, FlagBits> &) -> ArrayRef<T>;

template <typename T, unsigned N, unsigned FlagBits>
ArrayRef(const SmallVectorWithFlags<T, N, FlagBits> &) -> ArrayRef<T>;

template <typename T, unsigned FlagBits>
MutableArrayRef(SmallVectorWithFlagsImpl<T, FlagBits> &) -> MutableArrayRef<T>;

template <typename T, unsigned N, unsigned FlagBits>
MutableArrayRef(SmallVectorWithFlags<T, N, FlagBits> &) -> MutableArrayRef<T>;

} // namespace llvm

#endif // LLVM_ADT_SMALLVECTORWITHFLAGS_H
