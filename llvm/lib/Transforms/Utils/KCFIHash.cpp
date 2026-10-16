//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/KCFIHash.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace support;

// xxHash64 is a deprecated pre-xxh3 hash, retained here only as the default
// KCFI type-ID hash for ABI compatibility.

static uint64_t rotl64(uint64_t X, size_t R) {
  return (X << R) | (X >> (64 - R));
}

constexpr uint64_t PRIME64_1 = 11400714785074694791ULL;
constexpr uint64_t PRIME64_2 = 14029467366897019727ULL;
constexpr uint64_t PRIME64_3 = 1609587929392839161ULL;
constexpr uint64_t PRIME64_4 = 9650029242287828579ULL;
constexpr uint64_t PRIME64_5 = 2870177450012600261ULL;

static uint64_t round(uint64_t Acc, uint64_t Input) {
  Acc += Input * PRIME64_2;
  Acc = rotl64(Acc, 31);
  Acc *= PRIME64_1;
  return Acc;
}

static uint64_t mergeRound(uint64_t Acc, uint64_t Val) {
  Val = round(0, Val);
  Acc ^= Val;
  Acc = Acc * PRIME64_1 + PRIME64_4;
  return Acc;
}

static uint64_t avalanche(uint64_t H) {
  H ^= H >> 33;
  H *= PRIME64_2;
  H ^= H >> 29;
  H *= PRIME64_3;
  H ^= H >> 32;
  return H;
}

static uint64_t xxHash64(const uint8_t *P, size_t Len) {
  const uint8_t *const BEnd = P + Len;
  uint64_t H64;

  if (Len >= 32) {
    const uint8_t *const Limit = BEnd - 32;
    uint64_t V1 = PRIME64_1 + PRIME64_2;
    uint64_t V2 = PRIME64_2;
    uint64_t V3 = 0;
    uint64_t V4 = -PRIME64_1;

    do {
      V1 = round(V1, endian::read64le(P));
      P += 8;
      V2 = round(V2, endian::read64le(P));
      P += 8;
      V3 = round(V3, endian::read64le(P));
      P += 8;
      V4 = round(V4, endian::read64le(P));
      P += 8;
    } while (P <= Limit);

    H64 = rotl64(V1, 1) + rotl64(V2, 7) + rotl64(V3, 12) + rotl64(V4, 18);
    H64 = mergeRound(H64, V1);
    H64 = mergeRound(H64, V2);
    H64 = mergeRound(H64, V3);
    H64 = mergeRound(H64, V4);
  } else {
    H64 = PRIME64_5;
  }

  H64 += (uint64_t)Len;

  while (reinterpret_cast<uintptr_t>(P) + 8 <=
         reinterpret_cast<uintptr_t>(BEnd)) {
    H64 ^= round(0, endian::read64le(P));
    H64 = rotl64(H64, 27) * PRIME64_1 + PRIME64_4;
    P += 8;
  }

  if (reinterpret_cast<uintptr_t>(P) + 4 <= reinterpret_cast<uintptr_t>(BEnd)) {
    H64 ^= (uint64_t)endian::read32le(P) * PRIME64_1;
    H64 = rotl64(H64, 23) * PRIME64_2 + PRIME64_3;
    P += 4;
  }

  while (P < BEnd) {
    H64 ^= (*P) * PRIME64_5;
    H64 = rotl64(H64, 11) * PRIME64_1;
    ++P;
  }

  return avalanche(H64);
}

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
    return static_cast<uint32_t>(
        xxHash64(reinterpret_cast<const uint8_t *>(MangledTypeName.data()),
                 MangledTypeName.size()));
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
