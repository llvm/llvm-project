//===- llvm/ADT/StringViewExtras.h - Useful string_view functions C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains some functions that are useful when dealing with
/// string_views.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STRINGVIEWEXTRAS_H
#define LLVM_ADT_STRINGVIEWEXTRAS_H

#include <string_view>

namespace llvm {

// FIXME: std::string_view::starts_with is not available until C++20. Once LLVM
// is upgraded to C++20, remove this header and users.

inline constexpr bool starts_with(std::string_view self, char C) {
  return !self.empty() && self.front() == C;
}
inline constexpr bool starts_with(std::string_view haystack,
                                  std::string_view needle) {
  return haystack.substr(0, needle.size()) == needle;
}

} // end namespace llvm

#endif // LLVM_ADT_STRINGVIEWEXTRAS_H
