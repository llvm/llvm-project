//===- SmallVectorWithFlags.h - Small vector with flags ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines SmallVectorWithFlags, a SmallVector with user flag bits
/// packed into its capacity field.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SMALLVECTORWITHFLAGS_H
#define LLVM_ADT_SMALLVECTORWITHFLAGS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

/// An N-independent interface for SmallVectorWithFlags. The flags are part of
/// the vector's value: assignment and swap through this base transfer them
/// along with the elements, as do equality and ordering. This is a separate
/// interface from SmallVectorImpl<T>: an ordinary SmallVectorImpl cannot
/// interpret a capacity field holding flags.
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

  /// The complete flag bit mask; see SmallVectorHeader.
  using Base::getFlags;
  using Base::setFlags;

  /// Return flag bit \p Index.
  bool getFlag(unsigned Index) const {
    assert(Index < FlagBits && "flag index out of range");
    return (getFlags() >> Index) & 1;
  }

  /// Set flag bit \p Index to \p Value.
  void setFlag(unsigned Index, bool Value) {
    assert(Index < FlagBits && "flag index out of range");
    unsigned Mask = 1u << Index;
    setFlags(Value ? getFlags() | Mask : getFlags() & ~Mask);
  }

  SmallVectorWithFlagsImpl(const SmallVectorWithFlagsImpl &) = delete;

  SmallVectorWithFlagsImpl &operator=(const SmallVectorWithFlagsImpl &RHS) {
    Base::operator=(RHS);
    setFlags(RHS.getFlags());
    return *this;
  }

  /// Moving leaves the source flags unchanged.
  SmallVectorWithFlagsImpl &operator=(SmallVectorWithFlagsImpl &&RHS) {
    // Taking over a heap buffer takes its capacity word, flags included.
    bool TakesBuffer = !RHS.isSmall();
    Base::operator=(std::move(RHS));
    if (!TakesBuffer)
      setFlags(RHS.getFlags());
    return *this;
  }

  void swap(SmallVectorWithFlagsImpl &RHS) {
    // Swapping two heap buffers swaps their capacity words, flags included.
    if (this->isSmall() || RHS.isSmall()) {
      unsigned Flags = getFlags();
      setFlags(RHS.getFlags());
      RHS.setFlags(Flags);
    }
    Base::swap(RHS);
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
/// The maximum capacity is divided by 2^FlagBits: with a 32-bit size type and
/// four flag bits, 2^28 - 1 elements.
///
/// Flags initially contain zero; the flags-first constructors take the
/// complete flag mask. Element operations preserve flags. Copying, moving, and
/// swapping transfer the flags along with the elements, whether done on the
/// complete container or through SmallVectorWithFlagsImpl; moving leaves the
/// source flags unchanged. Moving into a plain SmallVector<T, N> drops the
/// flags. Equality compares both elements and flags. Ordering compares
/// elements lexicographically, then flags. ArrayRef conversion exposes only
/// the elements; use it explicitly when an element-only comparison is
/// intended.
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

  /// The flags-first constructors take the complete flag mask; see setFlags.
  template <typename It, typename = EnableIfConvertibleToInputIterator<It>>
  SmallVectorWithFlags(unsigned Flags, It Begin, It End)
      : SmallVectorWithFlags(Begin, End) {
    this->setFlags(Flags);
  }

  SmallVectorWithFlags(unsigned Flags, Base &&RHS)
      : SmallVectorWithFlags(std::move(RHS)) {
    this->setFlags(Flags);
  }

  SmallVectorWithFlags(Base &&RHS) : Base(N) {
    if (!RHS.empty()) {
      // See SmallVector(SmallVectorImpl &&) for the rationale of this branch.
      if constexpr (std::is_trivially_move_assignable_v<T>)
        Base::operator=(std::move(RHS));
      else
        this->moveConstructFrom(std::move(RHS));
    }
    // Base::operator= transfers the flags; the empty and moveConstructFrom
    // paths do not.
    this->setFlags(RHS.getFlags());
  }

  SmallVectorWithFlags(const SmallVectorWithFlags &RHS) : Base(N) {
    if (!RHS.empty())
      Base::operator=(RHS);
    // Base::operator= transfers the flags; the empty path does not.
    this->setFlags(RHS.getFlags());
  }

  SmallVectorWithFlags(SmallVectorWithFlags &&RHS)
      : SmallVectorWithFlags(static_cast<Base &&>(RHS)) {}

  SmallVectorWithFlags &operator=(const SmallVectorWithFlags &RHS) {
    Base::operator=(RHS);
    return *this;
  }

  SmallVectorWithFlags &operator=(SmallVectorWithFlags &&RHS) {
    Base::operator=(std::move(RHS));
    return *this;
  }

  // Also covers SmallVectorWithFlags with other inline capacities.
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
};

/// Also covers complete containers, including across inline capacities.
template <typename T, unsigned FlagBits>
void swap(SmallVectorWithFlagsImpl<T, FlagBits> &LHS,
          SmallVectorWithFlagsImpl<T, FlagBits> &RHS) {
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
