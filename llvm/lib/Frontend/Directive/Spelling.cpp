//===-- Spelling.cpp ---------------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/Directive/Spelling.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"

#include <cassert>

llvm::StringRef llvm::directive::FindName(
    llvm::iterator_range<const llvm::directive::Spelling *> Range,
    unsigned Version) {
  assert(llvm::isInt<8 * sizeof(int)>(Version) && "Version value out of range");

  int V = Version;
  Spelling Tmp{StringRef(), {V, V}};
  auto F =
      llvm::lower_bound(Range, Tmp, [](const Spelling &A, const Spelling &B) {
        return A.Versions < B.Versions;
      });
  if (F != Range.end())
    return F->Name;
  return StringRef();
}
