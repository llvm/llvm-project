//===-- raw-ostream.h - Support for printing using raw_ostream --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file is not part of gtest, but extends it to support LLVM libraries.
// This is not a public API for testing - it's a detail of LLVM's gtest.
//
// gtest allows providing printers for custom types by defining operator<<.
// In LLVM, operator<< usually takes llvm:raw_ostream& instead of std::ostream&.
//
// This file defines a template printable(V), which returns a version of V that
// can be streamed into a std::ostream.
//
// This interface is chosen so that in the default case (printable(V) is V),
// the main gtest code calls operator<<(OS, V) itself. gtest-printers carefully
// controls the lookup to enable fallback printing (see testing::internal2).
//===----------------------------------------------------------------------===//

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_CUSTOM_RAW_OSTREAM_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_CUSTOM_RAW_OSTREAM_H_

namespace llvm_gtest {
// StreamSwitch is a trait that tells us how to stream a T into a std::ostream.
// By default, we just stream the T directly. We'll specialize this later.
template <typename T, typename Enable = void> struct StreamSwitch {
  static const T& printable(const T& V) { return V; }
};

// printable() returns a version of its argument that can be streamed into a
// std::ostream. This may be the argument itself, or some other representation.
template <typename T>
auto printable(const T &V) -> decltype(StreamSwitch<T>::printable(V)) {
  // We delegate to the trait, to allow partial specialization.
  return StreamSwitch<T>::printable(V);
}
} // namespace llvm_gtest

// If raw_ostream support is enabled, we specialize for types with operator<<
// that takes a raw_ostream.
#if !GTEST_NO_LLVM_RAW_OSTREAM
#include "llvm/ADT/Optional.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include <ostream>
namespace llvm_gtest {

// The printable() of a raw_ostream-enabled type T is a RawStreamProxy<T>.
// It uses raw_os_ostream to write the wrapped value to a std::ostream.
template <typename T>
struct RawStreamProxy {
  const T& V;
  friend std::ostream &operator<<(std::ostream &S, const RawStreamProxy<T> &V) {
    llvm::raw_os_ostream OS(S);
    OS << V.V;
    return S;
  }
};

// We enable raw_ostream treatment if `(raw_ostream&) << (const T&)` is valid.
// We don't want implicit conversions on the RHS (e.g. to bool!), so "consume"
// the possible conversion by passing something convertible to const T& instead.
template <typename T> struct ConvertibleTo { operator T(); };
template <typename T>
struct StreamSwitch<T, decltype((void)(std::declval<llvm::raw_ostream &>()
                                       << ConvertibleTo<const T &>()))> {
  static const RawStreamProxy<T> printable(const T &V) { return {V}; }
};

// llvm::Optional has a template operator<<, which means it will not accept any
// implicit conversions, so we need to special-case it here.
template <typename T>
struct StreamSwitch<llvm::Optional<T>,
                    decltype((void)(std::declval<llvm::raw_ostream &>()
                                    << std::declval<llvm::Optional<T>>()))> {
  static const RawStreamProxy<llvm::Optional<T>>
  printable(const llvm::Optional<T> &V) {
    return {V};
  }
};
} // namespace llvm_gtest
#endif  // !GTEST_NO_LLVM_RAW_OSTREAM

#endif // GTEST_INCLUDE_GTEST_INTERNAL_CUSTOM_RAW_OSTREAM_H_
