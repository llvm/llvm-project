//===--- TrailingObjects.h - Variable-length classes ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header defines support for implementing classes that have
/// some trailing object (or arrays of objects) appended to them. The
/// main purpose is to make it obvious where this idiom is being used,
/// and to make the usage more idiomatic and more difficult to get
/// wrong.
///
/// The TrailingObject template abstracts away the reinterpret_cast,
/// pointer arithmetic, and size calculations used for the allocation
/// and access of appended arrays of objects, as well as asserts that
/// the alignment of the classes involved are appropriate for the
/// usage. Additionally, it ensures that the base type is final --
/// deriving from a class that expects data appended immediately after
/// it is typically not safe.
///
/// Users are expected to derive from this template, and provide
/// numTrailingObjects implementations for each trailing type,
/// e.g. like this sample:
///
/// \code
/// class VarLengthObj : private TrailingObjects<VarLengthObj, int, double> {
///   friend TrailingObjects;
///
///   unsigned NumInts, NumDoubles;
///   size_t numTrailingObjects(OverloadToken<int>) const { return NumInts; }
///   size_t numTrailingObjects(OverloadToken<double>) const {
///     return NumDoubles;
///   }
///  };
/// \endcode
///
/// You can access the appended arrays via 'getTrailingObjects', and
/// determine the size needed for allocation via
/// 'additionalSizeToAlloc' and 'totalSizeToAlloc'.
///
/// All the methods implemented by this class are are intended for use
/// by the implementation of the class, not as part of its interface
/// (thus, private inheritance is suggested).
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_TRAILINGOBJECTS_H
#define LLVM_SUPPORT_TRAILINGOBJECTS_H

#include <new>
#include <type_traits>
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/type_traits.h"

namespace llvm {

/// The base class for TrailingObjects* classes.
class TrailingObjectsBase {
protected:
  /// OverloadToken's purpose is to allow specifying function overloads
  /// for different types, without actually taking the types as
  /// parameters. (Necessary because member function templates cannot
  /// be specialized, so overloads must be used instead of
  /// specialization.)
  template <typename T> struct OverloadToken {};
};

// Internally used to indicate that the user didn't supply this value,
// so the explicit-specialization for fewer args will be used.
class NoTrailingTypeArg {};

// TODO: Consider using a single variadic implementation instead of
// multiple copies of the TrailingObjects template? [but, variadic
// template recursive implementations are annoying...]

/// This is the two-type version of the TrailingObjects template; see
/// file docstring for details.
template <typename BaseTy, typename TrailingTy1,
          typename TrailingTy2 = NoTrailingTypeArg>
class TrailingObjects : public TrailingObjectsBase {
private:
  // Contains static_assert statements for the alignment of the
  // types. Must not be at class-level, because BaseTy isn't complete
  // at class instantiation time, but will be by the time this
  // function is instantiated.
  static void verifyTrailingObjectsAssertions() {
    static_assert(llvm::AlignOf<BaseTy>::Alignment >=
                      llvm::AlignOf<TrailingTy1>::Alignment,
                  "TrailingTy1 requires more alignment than BaseTy provides");
    static_assert(
        llvm::AlignOf<TrailingTy1>::Alignment >=
            llvm::AlignOf<TrailingTy2>::Alignment,
        "TrailingTy2 requires more alignment than TrailingTy1 provides");

#ifdef LLVM_IS_FINAL
    static_assert(LLVM_IS_FINAL(BaseTy), "BaseTy must be final.");
#endif
  }

  // The next four functions are internal helpers for getTrailingObjects.
  static const TrailingTy1 *getTrailingObjectsImpl(const BaseTy *Obj,
                                                   OverloadToken<TrailingTy1>) {
    return reinterpret_cast<const TrailingTy1 *>(Obj + 1);
  }

  static TrailingTy1 *getTrailingObjectsImpl(BaseTy *Obj,
                                             OverloadToken<TrailingTy1>) {
    return reinterpret_cast<TrailingTy1 *>(Obj + 1);
  }

  static const TrailingTy2 *getTrailingObjectsImpl(const BaseTy *Obj,
                                                   OverloadToken<TrailingTy2>) {
    return reinterpret_cast<const TrailingTy2 *>(
        getTrailingObjectsImpl(Obj, OverloadToken<TrailingTy1>()) +
        Obj->numTrailingObjects(OverloadToken<TrailingTy1>()));
  }

  static TrailingTy2 *getTrailingObjectsImpl(BaseTy *Obj,
                                             OverloadToken<TrailingTy2>) {
    return reinterpret_cast<TrailingTy2 *>(
        getTrailingObjectsImpl(Obj, OverloadToken<TrailingTy1>()) +
        Obj->numTrailingObjects(OverloadToken<TrailingTy1>()));
  }

protected:
  /// Returns a pointer to the trailing object array of the given type
  /// (which must be one of those specified in the class template). The
  /// array may have zero or more elements in it.
  template <typename T> const T *getTrailingObjects() const {
    verifyTrailingObjectsAssertions();
    // Forwards to an impl function with overloads, since member
    // function templates can't be specialized.
    return getTrailingObjectsImpl(static_cast<const BaseTy *>(this),
                                  OverloadToken<T>());
  }

  /// Returns a pointer to the trailing object array of the given type
  /// (which must be one of those specified in the class template). The
  /// array may have zero or more elements in it.
  template <typename T> T *getTrailingObjects() {
    verifyTrailingObjectsAssertions();
    // Forwards to an impl function with overloads, since member
    // function templates can't be specialized.
    return getTrailingObjectsImpl(static_cast<BaseTy *>(this),
                                  OverloadToken<T>());
  }

  /// Returns the size of the trailing data, if an object were
  /// allocated with the given counts (The counts are in the same order
  /// as the template arguments). This does not include the size of the
  /// base object.  The template arguments must be the same as those
  /// used in the class; they are supplied here redundantly only so
  /// that it's clear what the counts are counting in callers.
  template <typename Ty1, typename Ty2,
            typename std::enable_if<std::is_same<Ty1, TrailingTy1>::value &&
                                        std::is_same<Ty2, TrailingTy2>::value,
                                    int>::type = 0>
  static LLVM_CONSTEXPR size_t additionalSizeToAlloc(size_t Count1, size_t Count2) {
    return sizeof(TrailingTy1) * Count1 + sizeof(TrailingTy2) * Count2;
  }

  /// Returns the total size of an object if it were allocated with the
  /// given trailing object counts. This is the same as
  /// additionalSizeToAlloc, except it *does* include the size of the base
  /// object.
  template <typename Ty1, typename Ty2>
  static LLVM_CONSTEXPR size_t totalSizeToAlloc(size_t Count1, size_t Count2) {
    return sizeof(BaseTy) + additionalSizeToAlloc<Ty1, Ty2>(Count1, Count2);
  }
};

/// This is the one-type version of the TrailingObjects template. See
/// the two-type version for more documentation.
template <typename BaseTy, typename TrailingTy1>
class TrailingObjects<BaseTy, TrailingTy1, NoTrailingTypeArg>
    : public TrailingObjectsBase {
private:
  static void verifyTrailingObjectsAssertions() {
    static_assert(llvm::AlignOf<BaseTy>::Alignment >=
                      llvm::AlignOf<TrailingTy1>::Alignment,
                  "TrailingTy1 requires more alignment than BaseTy provides");

#ifdef LLVM_IS_FINAL
    static_assert(LLVM_IS_FINAL(BaseTy), "BaseTy must be final.");
#endif
  }

  static const TrailingTy1 *getTrailingObjectsImpl(const BaseTy *Obj,
                                                   OverloadToken<TrailingTy1>) {
    return reinterpret_cast<const TrailingTy1 *>(Obj + 1);
  }

  static TrailingTy1 *getTrailingObjectsImpl(BaseTy *Obj,
                                             OverloadToken<TrailingTy1>) {
    return reinterpret_cast<TrailingTy1 *>(Obj + 1);
  }

protected:
  template <typename T> const T *getTrailingObjects() const {
    verifyTrailingObjectsAssertions();
    return getTrailingObjectsImpl(static_cast<const BaseTy *>(this),
                                  OverloadToken<T>());
  }

  template <typename T> T *getTrailingObjects() {
    verifyTrailingObjectsAssertions();
    return getTrailingObjectsImpl(static_cast<BaseTy *>(this),
                                  OverloadToken<T>());
  }

  template <typename Ty1,
            typename std::enable_if<std::is_same<Ty1, TrailingTy1>::value,
                                    int>::type = 0>
  static LLVM_CONSTEXPR size_t additionalSizeToAlloc(size_t Count1) {
    return sizeof(TrailingTy1) * Count1;
  }

  template <typename Ty1>
  static LLVM_CONSTEXPR size_t totalSizeToAlloc(size_t Count1) {
    return sizeof(BaseTy) + additionalSizeToAlloc<Ty1>(Count1);
  }
};

} // end namespace llvm

#endif
