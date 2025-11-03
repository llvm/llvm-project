//===-------------------------------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_FRONTEND_DIRECTIVE_SPELLING_H
#define LLVM_FRONTEND_DIRECTIVE_SPELLING_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Compiler.h"

#include <limits>
#include <tuple>

namespace llvm::directive {

struct VersionRange {
  static constexpr int MaxValue = std::numeric_limits<int>::max();
  // The default "Version" value in get<Lang><Enum>Name() is 0, include that
  // in the maximum range.
  int Min = 0;
  int Max = MaxValue;

  bool operator<(const VersionRange &R) const {
    return std::tie(Min, Max) < std::tie(R.Min, R.Max);
  }
};

struct Spelling {
  StringRef Name;
  VersionRange Versions;
};

LLVM_ABI StringRef FindName(llvm::iterator_range<const Spelling *>,
                            unsigned Version);

} // namespace llvm::directive

#endif // LLVM_FRONTEND_DIRECTIVE_SPELLING_H
