// Copyright 2015, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// This file provides an injection point for custom printers in a local
// installation of gTest.
// It will be included from gtest-printers.h and the overrides in this file
// will be visible to everyone.
//
// Injection point for custom user configurations. See README for details
//
// ** Custom implementation starts here **

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_CUSTOM_GTEST_PRINTERS_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_CUSTOM_GTEST_PRINTERS_H_

#if !GTEST_NO_LLVM_SUPPORT
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include <ostream>
// Printing of llvm String types.
// gtest sees these as containers of char (they have nested iterator types),
// so their operator<< is never considered unless we provide PrintTo().
// PrintStringTo provides quotes and escaping, at the cost of a copy.
namespace llvm {
inline void PrintTo(llvm::StringRef S, std::ostream *OS) {
  *OS << ::testing::PrintToString(S.str());
}
// We need both SmallString<N> and SmallVectorImpl<char> overloads:
//  - the SmallString<N> template is needed as overload resolution will
//    instantiate generic PrintTo<T> rather than do derived-to-base conversion
//  - but SmallVectorImpl<char> is sometimes the actual static type, in code
//    that erases the small size
template <unsigned N>
inline void PrintTo(const SmallString<N> &S, std::ostream *OS) {
  *OS << ::testing::PrintToString(std::string(S.data(), S.size()));
}
inline void PrintTo(const SmallVectorImpl<char> &S, std::ostream *OS) {
  *OS << ::testing::PrintToString(std::string(S.data(), S.size()));
}

// DenseMap's entries inherit from std::pair, and should act like pairs.
// However gTest's provided `PrintTo(pair<K,V>)` template won't deduce K and V
// because of the needed derived-to-base conversion.
namespace detail {
template <typename K, typename V>
inline void PrintTo(const DenseMapPair<K, V> &Pair, std::ostream *OS) {
  *OS << ::testing::PrintToString(static_cast<const std::pair<K, V> &>(Pair));
}
} // namespace detail

} // namespace llvm
#endif // !GTEST_NO_LLVM_SUPPORT

#endif  // GTEST_INCLUDE_GTEST_INTERNAL_CUSTOM_GTEST_PRINTERS_H_
