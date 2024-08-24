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

} // namespace llvm

#endif
