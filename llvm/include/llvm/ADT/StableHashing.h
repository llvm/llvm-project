//===- llvm/ADT/StableHashing.h - Utilities for stable hashing * C++ *-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides types and functions for computing and combining stable
// hashes. Stable hashes can be useful for hashing across different modules,
// processes, machines, or compiler runs for a specific compiler version. It
// currently employs the xxh3_64bits hashing algorithm. Be aware that this
// implementation may be adjusted or updated as improvements to the compiler are
// made.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STABLEHASHING_H
#define LLVM_ADT_STABLEHASHING_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/xxhash.h"

namespace llvm {

/// An opaque object representing a stable hash code. It can be serialized,
/// deserialized, and is stable across processes and executions.
using stable_hash = uint64_t;

inline stable_hash stable_hash_combine(ArrayRef<stable_hash> Buffer) {
  const uint8_t *Ptr = reinterpret_cast<const uint8_t *>(Buffer.data());
  size_t Size = Buffer.size() * sizeof(stable_hash);
  return xxh3_64bits(ArrayRef<uint8_t>(Ptr, Size));
}

inline stable_hash stable_hash_combine(stable_hash A, stable_hash B) {
  stable_hash Hashes[2] = {A, B};
  return stable_hash_combine(Hashes);
}

inline stable_hash stable_hash_combine(stable_hash A, stable_hash B,
                                       stable_hash C) {
  stable_hash Hashes[3] = {A, B, C};
  return stable_hash_combine(Hashes);
}

inline stable_hash stable_hash_combine(stable_hash A, stable_hash B,
                                       stable_hash C, stable_hash D) {
  stable_hash Hashes[4] = {A, B, C, D};
  return stable_hash_combine(Hashes);
}

// Removes suffixes introduced by LLVM from the name to enhance stability and
// maintain closeness to the original name across different builds.
inline StringRef get_stable_name(StringRef Name) {
  // Return the part after ".content." that represents contents.
  StringRef S0 = Name.rsplit(".content.").second;
  if (!S0.empty())
    return S0;

  // Ignore these suffixes.
  StringRef P1 = Name.rsplit(".llvm.").first;
  return P1.rsplit(".__uniq.").first;
}

// Generates a consistent hash value for a given input name across different
// program executions and environments. This function first converts the input
// name into a stable form using the `get_stable_name` function, and then
// computes a hash of this stable name. For instance, `foo.llvm.1234` would have
// the same hash as `foo.llvm.5678.
inline stable_hash stable_hash_name(StringRef Name) {
  return xxh3_64bits(get_stable_name(Name));
}

} // namespace llvm

#endif
