//===- llvm/Support/Hash.h - Hash functions --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides hash functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_HASH_H
#define LLVM_SUPPORT_HASH_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace llvm {

enum class KCFIHashAlgorithm { xxHash64, FNV1a };

/// Parse a KCFI hash algorithm name.
/// Returns xxHash64 if the name is not recognized.
LLVM_ABI KCFIHashAlgorithm parseKCFIHashAlgorithm(StringRef Name);

/// Convert a KCFI hash algorithm enum to its string representation.
LLVM_ABI StringRef stringifyKCFIHashAlgorithm(KCFIHashAlgorithm Algorithm);

/// Compute KCFI type ID from mangled type name.
/// The algorithm can be xxHash64 or FNV-1a.
LLVM_ABI uint32_t getKCFITypeID(StringRef MangledTypeName,
                                KCFIHashAlgorithm Algorithm);

} // end namespace llvm

#endif // LLVM_SUPPORT_HASH_H
