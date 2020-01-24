//===-- llvm/Support/Alignment.h - Useful alignment functions ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains types to represent alignments.
// They are instrumented to guarantee some invariants are preserved and prevent
// invalid manipulations.
//
// - Align represents an alignment in bytes, it is always set and always a valid
// power of two, its minimum value is 1 which means no alignment requirements.
//
// - MaybeAlign is an optional type, it may be undefined or set. When it's set
// you can get the underlying Align type by using the getValue() method.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ALIGNMENT_H_
#define LLVM_SUPPORT_ALIGNMENT_H_

#include "llvm/ADT/Optional.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <limits>

namespace llvm {

#define ALIGN_CHECK_ISPOSITIVE(decl)                                           \
  assert(decl > 0 && (#decl " should be defined"))
#define ALIGN_CHECK_ISSET(decl)                                                \
  assert(decl.hasValue() && (#decl " should be defined"))

/// This struct is a compact representation of a valid (non-zero power of two)
/// alignment.
/// It is suitable for use as static global constants.
struct Align {
private:
  uint8_t ShiftValue = 0; /// The log2 of the required alignment.
                          /// ShiftValue is less than 64 by construction.

  friend struct MaybeAlign;
  friend unsigned Log2(Align);
  friend bool operator==(Align Lhs, Align Rhs);
  friend bool operator!=(Align Lhs, Align Rhs);
  friend bool operator<=(Align Lhs, Align Rhs);
  friend bool operator>=(Align Lhs, Align Rhs);
  friend bool operator<(Align Lhs, Align Rhs);
  friend bool operator>(Align Lhs, Align Rhs);
  friend unsigned encode(struct MaybeAlign A);
  friend struct MaybeAlign decodeMaybeAlign(unsigned Value);

  /// A trivial type to allow construction of constexpr Align.
  /// This is currently needed to workaround a bug in GCC 5.3 which prevents
  /// definition of constexpr assign operators.
  /// https://stackoverflow.com/questions/46756288/explicitly-defaulted-function-cannot-be-declared-as-constexpr-because-the-implic
  /// FIXME: Remove this, make all assign operators constexpr and introduce user
  /// defined literals when we don't have to support GCC 5.3 anymore.
  /// https://llvm.org/docs/GettingStarted.html#getting-a-modern-host-c-toolchain
  struct LogValue {
    uint8_t Log;
  };

public:
  /// Default is byte-aligned.
  constexpr Align() = default;
  /// Do not perform checks in case of copy/move construct/assign, because the
  /// checks have been performed when building `Other`.
  constexpr Align(const Align &Other) = default;
  constexpr Align(Align &&Other) = default;
  Align &operator=(const Align &Other) = default;
  Align &operator=(Align &&Other) = default;

  explicit Align(uint64_t Value) {
    assert(Value > 0 && "Value must not be 0");
    assert(llvm::isPowerOf2_64(Value) && "Alignment is not a power of 2");
    ShiftValue = Log2_64(Value);
    assert(ShiftValue < 64 && "Broken invariant");
  }

  /// This is a hole in the type system and should not be abused.
  /// Needed to interact with C for instance.
  uint64_t value() const { return uint64_t(1) << ShiftValue; }

  /// Returns a default constructed Align which corresponds to no alignment.
  /// It was decided to deprecate Align::None because it's too close to
  /// llvm::None which can be used to initialize `MaybeAlign`.
  /// MaybeAlign = llvm::None means unspecified alignment,
  /// Align = Align::None() means alignment of one byte.
  LLVM_ATTRIBUTE_DEPRECATED(constexpr static const Align None(),
                            "Use Align() or Align(1) instead") {
    return Align();
  }

  /// Allow constructions of constexpr Align.
  template <size_t kValue> constexpr static LogValue Constant() {
    return LogValue{static_cast<uint8_t>(CTLog2<kValue>())};
  }

  /// Allow constructions of constexpr Align from types.
  /// Compile time equivalent to Align(alignof(T)).
  template <typename T> constexpr static LogValue Of() {
    return Constant<std::alignment_of<T>::value>();
  }

  /// Constexpr constructor from LogValue type.
  constexpr Align(LogValue CA) : ShiftValue(CA.Log) {}
};

/// Treats the value 0 as a 1, so Align is always at least 1.
inline Align assumeAligned(uint64_t Value) {
  return Value ? Align(Value) : Align();
}

/// This struct is a compact representation of a valid (power of two) or
/// undefined (0) alignment.
struct MaybeAlign : public llvm::Optional<Align> {
private:
  using UP = llvm::Optional<Align>;

public:
  /// Default is undefined.
  MaybeAlign() = default;
  /// Do not perform checks in case of copy/move construct/assign, because the
  /// checks have been performed when building `Other`.
  MaybeAlign(const MaybeAlign &Other) = default;
  MaybeAlign &operator=(const MaybeAlign &Other) = default;
  MaybeAlign(MaybeAlign &&Other) = default;
  MaybeAlign &operator=(MaybeAlign &&Other) = default;

  /// Use llvm::Optional<Align> constructor.
  using UP::UP;

  explicit MaybeAlign(uint64_t Value) {
    assert((Value == 0 || llvm::isPowerOf2_64(Value)) &&
           "Alignment is neither 0 nor a power of 2");
    if (Value)
      emplace(Value);
  }

  /// For convenience, returns a valid alignment or 1 if undefined.
  Align valueOrOne() const { return hasValue() ? getValue() : Align(); }
};

/// Checks that SizeInBytes is a multiple of the alignment.
inline bool isAligned(Align Lhs, uint64_t SizeInBytes) {
  return SizeInBytes % Lhs.value() == 0;
}

/// Checks that SizeInBytes is a multiple of the alignment.
/// Returns false if the alignment is undefined.
inline bool isAligned(MaybeAlign Lhs, uint64_t SizeInBytes) {
  ALIGN_CHECK_ISSET(Lhs);
  return SizeInBytes % (*Lhs).value() == 0;
}

/// Checks that Addr is a multiple of the alignment.
inline bool isAddrAligned(Align Lhs, const void *Addr) {
  return isAligned(Lhs, reinterpret_cast<uintptr_t>(Addr));
}

/// Returns a multiple of A needed to store `Size` bytes.
inline uint64_t alignTo(uint64_t Size, Align A) {
  const uint64_t value = A.value();
  // The following line is equivalent to `(Size + value - 1) / value * value`.

  // The division followed by a multiplication can be thought of as a right
  // shift followed by a left shift which zeros out the extra bits produced in
  // the bump; `~(value - 1)` is a mask where all those bits being zeroed out
  // are just zero.

  // Most compilers can generate this code but the pattern may be missed when
  // multiple functions gets inlined.
  return (Size + value - 1) & ~(value - 1);
}

/// Returns a multiple of A needed to store `Size` bytes.
/// Returns `Size` if current alignment is undefined.
inline uint64_t alignTo(uint64_t Size, MaybeAlign A) {
  return A ? alignTo(Size, A.getValue()) : Size;
}

/// Aligns `Addr` to `Alignment` bytes, rounding up.
inline uintptr_t alignAddr(const void *Addr, Align Alignment) {
  uintptr_t ArithAddr = reinterpret_cast<uintptr_t>(Addr);
  assert(static_cast<uintptr_t>(ArithAddr + Alignment.value() - 1) >=
             ArithAddr &&
         "Overflow");
  return alignTo(ArithAddr, Alignment);
}

/// Returns the offset to the next integer (mod 2**64) that is greater than
/// or equal to \p Value and is a multiple of \p Align.
inline uint64_t offsetToAlignment(uint64_t Value, Align Alignment) {
  return alignTo(Value, Alignment) - Value;
}

/// Returns the necessary adjustment for aligning `Addr` to `Alignment`
/// bytes, rounding up.
inline uint64_t offsetToAlignedAddr(const void *Addr, Align Alignment) {
  return offsetToAlignment(reinterpret_cast<uintptr_t>(Addr), Alignment);
}

/// Returns the log2 of the alignment.
inline unsigned Log2(Align A) { return A.ShiftValue; }

/// Returns the log2 of the alignment.
/// \pre A must be defined.
inline unsigned Log2(MaybeAlign A) {
  ALIGN_CHECK_ISSET(A);
  return Log2(A.getValue());
}

/// Returns the alignment that satisfies both alignments.
/// Same semantic as MinAlign.
inline Align commonAlignment(Align A, Align B) { return std::min(A, B); }

/// Returns the alignment that satisfies both alignments.
/// Same semantic as MinAlign.
inline Align commonAlignment(Align A, uint64_t Offset) {
  return Align(MinAlign(A.value(), Offset));
}

/// Returns the alignment that satisfies both alignments.
/// Same semantic as MinAlign.
inline MaybeAlign commonAlignment(MaybeAlign A, MaybeAlign B) {
  return A && B ? commonAlignment(*A, *B) : A ? A : B;
}

/// Returns the alignment that satisfies both alignments.
/// Same semantic as MinAlign.
inline MaybeAlign commonAlignment(MaybeAlign A, uint64_t Offset) {
  return MaybeAlign(MinAlign((*A).value(), Offset));
}

/// Returns a representation of the alignment that encodes undefined as 0.
inline unsigned encode(MaybeAlign A) { return A ? A->ShiftValue + 1 : 0; }

/// Dual operation of the encode function above.
inline MaybeAlign decodeMaybeAlign(unsigned Value) {
  if (Value == 0)
    return MaybeAlign();
  Align Out;
  Out.ShiftValue = Value - 1;
  return Out;
}

/// Returns a representation of the alignment, the encoded value is positive by
/// definition.
inline unsigned encode(Align A) { return encode(MaybeAlign(A)); }

/// Comparisons between Align and scalars. Rhs must be positive.
inline bool operator==(Align Lhs, uint64_t Rhs) {
  ALIGN_CHECK_ISPOSITIVE(Rhs);
  return Lhs.value() == Rhs;
}
inline bool operator!=(Align Lhs, uint64_t Rhs) {
  ALIGN_CHECK_ISPOSITIVE(Rhs);
  return Lhs.value() != Rhs;
}
inline bool operator<=(Align Lhs, uint64_t Rhs) {
  ALIGN_CHECK_ISPOSITIVE(Rhs);
  return Lhs.value() <= Rhs;
}
inline bool operator>=(Align Lhs, uint64_t Rhs) {
  ALIGN_CHECK_ISPOSITIVE(Rhs);
  return Lhs.value() >= Rhs;
}
inline bool operator<(Align Lhs, uint64_t Rhs) {
  ALIGN_CHECK_ISPOSITIVE(Rhs);
  return Lhs.value() < Rhs;
}
inline bool operator>(Align Lhs, uint64_t Rhs) {
  ALIGN_CHECK_ISPOSITIVE(Rhs);
  return Lhs.value() > Rhs;
}

/// Comparisons between MaybeAlign and scalars.
inline bool operator==(MaybeAlign Lhs, uint64_t Rhs) {
  return Lhs ? (*Lhs).value() == Rhs : Rhs == 0;
}
inline bool operator!=(MaybeAlign Lhs, uint64_t Rhs) {
  return Lhs ? (*Lhs).value() != Rhs : Rhs != 0;
}
inline bool operator<=(MaybeAlign Lhs, uint64_t Rhs) {
  ALIGN_CHECK_ISSET(Lhs);
  ALIGN_CHECK_ISPOSITIVE(Rhs);
  return (*Lhs).value() <= Rhs;
}
inline bool operator>=(MaybeAlign Lhs, uint64_t Rhs) {
  ALIGN_CHECK_ISSET(Lhs);
  ALIGN_CHECK_ISPOSITIVE(Rhs);
  return (*Lhs).value() >= Rhs;
}
inline bool operator<(MaybeAlign Lhs, uint64_t Rhs) {
  ALIGN_CHECK_ISSET(Lhs);
  ALIGN_CHECK_ISPOSITIVE(Rhs);
  return (*Lhs).value() < Rhs;
}
inline bool operator>(MaybeAlign Lhs, uint64_t Rhs) {
  ALIGN_CHECK_ISSET(Lhs);
  ALIGN_CHECK_ISPOSITIVE(Rhs);
  return (*Lhs).value() > Rhs;
}

/// Comparisons operators between Align.
inline bool operator==(Align Lhs, Align Rhs) {
  return Lhs.ShiftValue == Rhs.ShiftValue;
}
inline bool operator!=(Align Lhs, Align Rhs) {
  return Lhs.ShiftValue != Rhs.ShiftValue;
}
inline bool operator<=(Align Lhs, Align Rhs) {
  return Lhs.ShiftValue <= Rhs.ShiftValue;
}
inline bool operator>=(Align Lhs, Align Rhs) {
  return Lhs.ShiftValue >= Rhs.ShiftValue;
}
inline bool operator<(Align Lhs, Align Rhs) {
  return Lhs.ShiftValue < Rhs.ShiftValue;
}
inline bool operator>(Align Lhs, Align Rhs) {
  return Lhs.ShiftValue > Rhs.ShiftValue;
}

/// Comparisons operators between Align and MaybeAlign.
inline bool operator==(Align Lhs, MaybeAlign Rhs) {
  ALIGN_CHECK_ISSET(Rhs);
  return Lhs.value() == (*Rhs).value();
}
inline bool operator!=(Align Lhs, MaybeAlign Rhs) {
  ALIGN_CHECK_ISSET(Rhs);
  return Lhs.value() != (*Rhs).value();
}
inline bool operator<=(Align Lhs, MaybeAlign Rhs) {
  ALIGN_CHECK_ISSET(Rhs);
  return Lhs.value() <= (*Rhs).value();
}
inline bool operator>=(Align Lhs, MaybeAlign Rhs) {
  ALIGN_CHECK_ISSET(Rhs);
  return Lhs.value() >= (*Rhs).value();
}
inline bool operator<(Align Lhs, MaybeAlign Rhs) {
  ALIGN_CHECK_ISSET(Rhs);
  return Lhs.value() < (*Rhs).value();
}
inline bool operator>(Align Lhs, MaybeAlign Rhs) {
  ALIGN_CHECK_ISSET(Rhs);
  return Lhs.value() > (*Rhs).value();
}

/// Comparisons operators between MaybeAlign and Align.
inline bool operator==(MaybeAlign Lhs, Align Rhs) {
  ALIGN_CHECK_ISSET(Lhs);
  return Lhs && (*Lhs).value() == Rhs.value();
}
inline bool operator!=(MaybeAlign Lhs, Align Rhs) {
  ALIGN_CHECK_ISSET(Lhs);
  return Lhs && (*Lhs).value() != Rhs.value();
}
inline bool operator<=(MaybeAlign Lhs, Align Rhs) {
  ALIGN_CHECK_ISSET(Lhs);
  return Lhs && (*Lhs).value() <= Rhs.value();
}
inline bool operator>=(MaybeAlign Lhs, Align Rhs) {
  ALIGN_CHECK_ISSET(Lhs);
  return Lhs && (*Lhs).value() >= Rhs.value();
}
inline bool operator<(MaybeAlign Lhs, Align Rhs) {
  ALIGN_CHECK_ISSET(Lhs);
  return Lhs && (*Lhs).value() < Rhs.value();
}
inline bool operator>(MaybeAlign Lhs, Align Rhs) {
  ALIGN_CHECK_ISSET(Lhs);
  return Lhs && (*Lhs).value() > Rhs.value();
}

inline Align operator/(Align Lhs, uint64_t Divisor) {
  assert(llvm::isPowerOf2_64(Divisor) &&
         "Divisor must be positive and a power of 2");
  assert(Lhs != 1 && "Can't halve byte alignment");
  return Align(Lhs.value() / Divisor);
}

inline MaybeAlign operator/(MaybeAlign Lhs, uint64_t Divisor) {
  assert(llvm::isPowerOf2_64(Divisor) &&
         "Divisor must be positive and a power of 2");
  return Lhs ? Lhs.getValue() / Divisor : MaybeAlign();
}

inline Align max(MaybeAlign Lhs, Align Rhs) {
  return Lhs && *Lhs > Rhs ? *Lhs : Rhs;
}

inline Align max(Align Lhs, MaybeAlign Rhs) {
  return Rhs && *Rhs > Lhs ? *Rhs : Lhs;
}

#undef ALIGN_CHECK_ISPOSITIVE
#undef ALIGN_CHECK_ISSET

} // namespace llvm

#endif // LLVM_SUPPORT_ALIGNMENT_H_
