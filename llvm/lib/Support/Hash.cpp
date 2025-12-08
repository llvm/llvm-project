//===- Hash.cpp - Hash functions ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements hash functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Hash.h"
#include "llvm/Support/xxhash.h"

using namespace llvm;

KCFIHashAlgorithm llvm::parseKCFIHashAlgorithm(StringRef Name) {
  if (Name == "FNV-1a")
    return KCFIHashAlgorithm::FNV1a;
  // Default to xxHash64 for backward compatibility
  return KCFIHashAlgorithm::xxHash64;
}

StringRef llvm::stringifyKCFIHashAlgorithm(KCFIHashAlgorithm Algorithm) {
  switch (Algorithm) {
  case KCFIHashAlgorithm::xxHash64:
    return "xxHash64";
  case KCFIHashAlgorithm::FNV1a:
    return "FNV-1a";
  }
  llvm_unreachable("Unknown KCFI hash algorithm");
}

uint32_t llvm::getKCFITypeID(StringRef MangledTypeName,
                             KCFIHashAlgorithm Algorithm) {
  switch (Algorithm) {
  case KCFIHashAlgorithm::xxHash64:
    // Use lower 32 bits of xxHash64
    return static_cast<uint32_t>(xxHash64(MangledTypeName));
  case KCFIHashAlgorithm::FNV1a:
    // FNV-1a hash (32-bit)
    uint32_t Hash = 2166136261u; // FNV offset basis
    for (unsigned char C : MangledTypeName) {
      Hash ^= C;
      Hash *= 16777619u; // FNV prime
    }
    return Hash;
  }
  llvm_unreachable("Unknown KCFI hash algorithm");
}
