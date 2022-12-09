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

#ifndef LLVM_SUPPORT_GLOBPATTERN_H
#define LLVM_SUPPORT_GLOBPATTERN_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Error.h"
#include <optional>
#include <vector>

// This class represents a glob pattern. Supported metacharacters
// are "*", "?", "\", "[<chars>]", "[^<chars>]", and "[!<chars>]".
namespace llvm {

template <typename T> class ArrayRef;
class StringRef;

class GlobPattern {
public:
  static Expected<GlobPattern> create(StringRef Pat);
  bool match(StringRef S) const;

  // Returns true for glob pattern "*". Can be used to avoid expensive
  // preparation/acquisition of the input for match().
  bool isTrivialMatchAll() const {
    if (Prefix && Prefix->empty()) {
      assert(!Suffix);
      return true;
    }
    return false;
  }

private:
  bool matchOne(ArrayRef<BitVector> Pat, StringRef S) const;

  // Parsed glob pattern.
  std::vector<BitVector> Tokens;

  // The following members are for optimization.
  std::optional<StringRef> Exact;
  std::optional<StringRef> Prefix;
  std::optional<StringRef> Suffix;
};
}

#endif // LLVM_SUPPORT_GLOBPATTERN_H
