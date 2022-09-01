//===- llvm/Testing/ADT/StringMapEntry.h ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TESTING_ADT_STRINGMAPENTRY_H_
#define LLVM_TESTING_ADT_STRINGMAPENTRY_H_

#include "llvm/ADT/StringMapEntry.h"
#include <ostream>
#include <type_traits>

namespace llvm {
namespace detail {

template <typename T, typename = std::void_t<>>
struct CanOutputToOStream : std::false_type {};

template <typename T>
struct CanOutputToOStream<T, std::void_t<decltype(std::declval<std::ostream &>()
                                                  << std::declval<T>())>>
    : std::true_type {};

} // namespace detail

/// Support for printing to std::ostream, for use with e.g. producing more
/// useful error messages with Google Test.
template <typename T>
std::ostream &operator<<(std::ostream &OS, const StringMapEntry<T> &E) {
  OS << "{\"" << E.getKey().data() << "\": ";
  if constexpr (detail::CanOutputToOStream<decltype(E.getValue())>::value) {
    OS << E.getValue();
  } else {
    OS << "non-printable value";
  }
  return OS << "}";
}

} // namespace llvm

#endif
