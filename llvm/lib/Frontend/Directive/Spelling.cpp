//===-------------------------------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/Directive/Spelling.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"

#include <cassert>

using namespace llvm;

static bool Contains(directive::VersionRange V, int P) {
  return V.Min <= P && P <= V.Max;
}

llvm::StringRef llvm::directive::FindName(
    llvm::iterator_range<const directive::Spelling *> Range, unsigned Version) {
  assert(llvm::isInt<8 * sizeof(int)>(Version) && "Version value out of range");

  int V = Version;
  // Do a linear search to find the first Spelling that contains Version.
  // The condition "contains(S, Version)" does not partition the list of
  // spellings, so std::[lower|upper]_bound cannot be used.
  // In practice the list of spellings is expected to be very short, so
  // linear search seems appropriate. In general, an interval tree may be
  // a better choice, but in this case it may be an overkill.
  for (auto &S : Range) {
    if (Contains(S.Versions, V))
      return S.Name;
  }
  return StringRef();
}
