//===- llvm/Support/PointerLikeTypeTraits.h - Pointer Traits ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PointerLikeTypeTraits class.  This allows data
// structures to reason about pointers and other things that are pointer sized.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_POINTERLIKETYPETRAITS_H
#define LLVM_SUPPORT_POINTERLIKETYPETRAITS_H

#include "llvm/Support/DataTypes.h"
#include <type_traits>

namespace llvm {

/// A traits type that is used to handle pointer types and things that are just
/// wrappers for pointers as a uniform entity.
template <typename T> struct PointerLikeTypeTraits;

namespace detail {
/// A tiny meta function to compute the log2 of a compile time constant.
template <size_t N>
struct ConstantLog2
    : std::integral_constant<size_t, ConstantLog2<N / 2>::value + 1> {};
template <> struct ConstantLog2<1> : std::integral_constant<size_t, 0> {};

// Provide a trait to check if T is pointer-like.
template <typename T, typename U = void> struct HasPointerLikeTypeTraits {
  static const bool value = false;
};

// sizeof(T) is valid only for a complete T.
template <typename T> struct HasPointerLikeTypeTraits<
  T, decltype((sizeof(PointerLikeTypeTraits<T>) + sizeof(T)), void())> {
  static const bool value = true;
};

template <typename T> struct IsPointerLike {
  static const bool value = HasPointerLikeTypeTraits<T>::value;
};

template <typename T> struct IsPointerLike<T *> {
  static const bool value = true;
};
} // namespace detail

// Provide PointerLikeTypeTraits for non-cvr pointers.
template <typename T> struct PointerLikeTypeTraits<T *> {
  static inline void *getAsVoidPointer(T *P) { return P; }
  static inline T *getFromVoidPointer(void *P) { return static_cast<T *>(P); }

  enum { NumLowBitsAvailable = detail::ConstantLog2<alignof(T)>::value };
};

template <> struct PointerLikeTypeTraits<void *> {
  static inline void *getAsVoidPointer(void *P) { return P; }
  static inline void *getFromVoidPointer(void *P) { return P; }

  /// Note, we assume here that void* is related to raw malloc'ed memory and
  /// that malloc returns objects at least 4-byte aligned. However, this may be
  /// wrong, or pointers may be from something other than malloc. In this case,
  /// you should specify a real typed pointer or avoid this template.
  ///
  /// All clients should use assertions to do a run-time check to ensure that
  /// this is actually true.
  enum { NumLowBitsAvailable = 2 };
};

// Provide PointerLikeTypeTraits for const things.
template <typename T> struct PointerLikeTypeTraits<const T> {
  typedef PointerLikeTypeTraits<T> NonConst;

  static inline const void *getAsVoidPointer(const T P) {
    return NonConst::getAsVoidPointer(P);
  }
  static inline const T getFromVoidPointer(const void *P) {
    return NonConst::getFromVoidPointer(const_cast<void *>(P));
  }
  enum { NumLowBitsAvailable = NonConst::NumLowBitsAvailable };
};

// Provide PointerLikeTypeTraits for const pointers.
template <typename T> struct PointerLikeTypeTraits<const T *> {
  typedef PointerLikeTypeTraits<T *> NonConst;

  static inline const void *getAsVoidPointer(const T *P) {
    return NonConst::getAsVoidPointer(const_cast<T *>(P));
  }
  static inline const T *getFromVoidPointer(const void *P) {
    return NonConst::getFromVoidPointer(const_cast<void *>(P));
  }
  enum { NumLowBitsAvailable = NonConst::NumLowBitsAvailable };
};

// Provide PointerLikeTypeTraits for uintptr_t.
template <> struct PointerLikeTypeTraits<uintptr_t> {
  static inline void *getAsVoidPointer(uintptr_t P) {
    return reinterpret_cast<void *>(P);
  }
  static inline uintptr_t getFromVoidPointer(void *P) {
    return reinterpret_cast<uintptr_t>(P);
  }
  // No bits are available!
  enum { NumLowBitsAvailable = 0 };
};

} // end namespace llvm

#endif
