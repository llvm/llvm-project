//===- STLExtras.h - STL-like extensions that are used by MLIR --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains stuff that should be arguably sunk down to the LLVM
// Support/STLExtras.h file over time.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_STLEXTRAS_H
#define MLIR_SUPPORT_STLEXTRAS_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {

namespace detail {
template <typename RangeT>
using ValueOfRange = typename std::remove_reference<decltype(
    *std::begin(std::declval<RangeT &>()))>::type;
} // end namespace detail

/// An STL-style algorithm similar to std::for_each that applies a second
/// functor between every pair of elements.
///
/// This provides the control flow logic to, for example, print a
/// comma-separated list:
/// \code
///   interleave(names.begin(), names.end(),
///              [&](StringRef name) { os << name; },
///              [&] { os << ", "; });
/// \endcode
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor,
          typename = typename std::enable_if<
              !std::is_constructible<StringRef, UnaryFunctor>::value &&
              !std::is_constructible<StringRef, NullaryFunctor>::value>::type>
inline void interleave(ForwardIterator begin, ForwardIterator end,
                       UnaryFunctor each_fn, NullaryFunctor between_fn) {
  if (begin == end)
    return;
  each_fn(*begin);
  ++begin;
  for (; begin != end; ++begin) {
    between_fn();
    each_fn(*begin);
  }
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor,
          typename = typename std::enable_if<
              !std::is_constructible<StringRef, UnaryFunctor>::value &&
              !std::is_constructible<StringRef, NullaryFunctor>::value>::type>
inline void interleave(const Container &c, UnaryFunctor each_fn,
                       NullaryFunctor between_fn) {
  interleave(c.begin(), c.end(), each_fn, between_fn);
}

/// Overload of interleave for the common case of string separator.
template <typename Container, typename UnaryFunctor, typename raw_ostream,
          typename T = detail::ValueOfRange<Container>>
inline void interleave(const Container &c, raw_ostream &os,
                       UnaryFunctor each_fn, const StringRef &separator) {
  interleave(c.begin(), c.end(), each_fn, [&] { os << separator; });
}
template <typename Container, typename raw_ostream,
          typename T = detail::ValueOfRange<Container>>
inline void interleave(const Container &c, raw_ostream &os,
                       const StringRef &separator) {
  interleave(
      c, os, [&](const T &a) { os << a; }, separator);
}

template <typename Container, typename UnaryFunctor, typename raw_ostream,
          typename T = detail::ValueOfRange<Container>>
inline void interleaveComma(const Container &c, raw_ostream &os,
                            UnaryFunctor each_fn) {
  interleave(c, os, each_fn, ", ");
}
template <typename Container, typename raw_ostream,
          typename T = detail::ValueOfRange<Container>>
inline void interleaveComma(const Container &c, raw_ostream &os) {
  interleaveComma(c, os, [&](const T &a) { os << a; });
}

} // end namespace mlir

#endif // MLIR_SUPPORT_STLEXTRAS_H
