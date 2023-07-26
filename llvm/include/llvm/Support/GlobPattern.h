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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <optional>

// This class represents a glob pattern. Supported metacharacters
// are "*", "?", "\", "[<chars>]", "[^<chars>]", and "[!<chars>]".
namespace llvm {

class GlobPattern {
public:
  static Expected<GlobPattern> create(StringRef Pat);
  bool match(StringRef S) const;

  // Returns true for glob pattern "*". Can be used to avoid expensive
  // preparation/acquisition of the input for match().
  bool isTrivialMatchAll() const { return Prefix.empty() && Pat == "*"; }

private:
  bool matchOne(StringRef Str) const;

  // Brackets with their end position and matched bytes.
  struct Bracket {
    const char *Next;
    BitVector Bytes;
  };
  SmallVector<Bracket, 0> Brackets;

  StringRef Prefix, Pat;
};
}

#endif // LLVM_SUPPORT_GLOBPATTERN_H
