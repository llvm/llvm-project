//===-- GlobPattern.h - glob pattern matcher implementation -*- C++ -*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a glob pattern matcher. The glob pattern is the
// rule used by the shell.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_GLOB_PATTERN_H
#define LLVM_SUPPORT_GLOB_PATTERN_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <vector>

// This class represents a glob pattern. Supported metacharacters
// are "*", "?", "[<chars>]" and "[^<chars>]".
namespace llvm {
class BitVector;
template <typename T> class ArrayRef;

class GlobPattern {
public:
  static Expected<GlobPattern> create(StringRef Pat);
  bool match(StringRef S) const;

private:
  bool matchOne(ArrayRef<BitVector> Pat, StringRef S) const;

  // Parsed glob pattern.
  std::vector<BitVector> Tokens;

  // The following members are for optimization.
  Optional<StringRef> Exact;
  Optional<StringRef> Prefix;
  Optional<StringRef> Suffix;
};
}

#endif // LLVM_SUPPORT_GLOB_PATTERN_H
