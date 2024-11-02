//===- OptionStrCmp.h - Option String Comparison ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_OPTIONSTRCMP_H
#define LLVM_SUPPORT_OPTIONSTRCMP_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {

// Comparison function for Option strings (option names & prefixes).
// The ordering is *almost* case-insensitive lexicographic, with an exception.
// '\0' comes at the end of the alphabet instead of the beginning (thus options
// precede any other options which prefix them). Additionally, if two options
// are identical ignoring case, they are ordered according to case sensitive
// ordering if `FallbackCaseSensitive` is true.
int StrCmpOptionName(StringRef A, StringRef B,
                     bool FallbackCaseSensitive = true);

// Comparison function for Option prefixes.
int StrCmpOptionPrefixes(ArrayRef<StringRef> APrefixes,
                         ArrayRef<StringRef> BPrefixes);

} // namespace llvm

#endif // LLVM_SUPPORT_OPTIONSTRCMP_H
