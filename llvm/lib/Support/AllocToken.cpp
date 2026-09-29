//===- AllocToken.cpp - Allocation Token Calculation ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definition of AllocToken modes and shared calculation of stateless token IDs.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/AllocToken.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SipHash.h"

using namespace llvm;

std::optional<AllocTokenMode>
llvm::getAllocTokenModeFromString(StringRef Name) {
  return StringSwitch<std::optional<AllocTokenMode>>(Name)
      .Case("increment", AllocTokenMode::Increment)
      .Case("random", AllocTokenMode::Random)
      .Case("typehash", AllocTokenMode::TypeHash)
      .Case("typehashpointersplit", AllocTokenMode::TypeHashPointerSplit)
      .Case("typefunchash", AllocTokenMode::TypeFuncHash)
      .Case("typefunchashpointersplit",
            AllocTokenMode::TypeFuncHashPointerSplit)
      .Case("default", DefaultAllocTokenMode)
      .Default(std::nullopt);
}

StringRef llvm::getAllocTokenModeAsString(AllocTokenMode Mode) {
  switch (Mode) {
  case AllocTokenMode::Increment:
    return "increment";
  case AllocTokenMode::Random:
    return "random";
  case AllocTokenMode::TypeHash:
    return "typehash";
  case AllocTokenMode::TypeHashPointerSplit:
    return "typehashpointersplit";
  case AllocTokenMode::TypeFuncHash:
    return "typefunchash";
  case AllocTokenMode::TypeFuncHashPointerSplit:
    return "typefunchashpointersplit";
  }
  llvm_unreachable("Unknown AllocTokenMode");
}

static uint64_t getStableHash(const AllocTokenMetadata &Metadata,
                              uint64_t MaxTokens) {
  return getStableSipHash(Metadata.TypeName) % MaxTokens;
}

/// The token ID is split into bitfields: the upper bits hold the type name
/// hash, and the lower bits hold the hash of the name of the function
/// containing the allocation. With pointer split, the most significant bit is
/// set for types that contain pointers. Uses Log2(MaxTokens) bits, so that the
/// token ID is always less than MaxTokens.
static uint64_t getTypeFuncHash(const AllocTokenMetadata &Metadata,
                                uint64_t MaxTokens, bool PointerSplit) {
  const unsigned Bits = Log2_64(MaxTokens);
  assert(Bits >= (PointerSplit ? 3u : 2u) && "MaxTokens too small");
  // If the number of bits is odd, the type name hash gets the extra bit.
  const unsigned FuncBits = Bits / 2;
  unsigned TypeBits = Bits - FuncBits;
  uint64_t Token = 0;
  if (PointerSplit) {
    --TypeBits;
    Token = uint64_t(Metadata.ContainsPointer) << (Bits - 1);
  }
  // An empty type name denotes an unknown type.
  if (!Metadata.TypeName.empty())
    Token |= (getStableSipHash(Metadata.TypeName) &
              maskTrailingOnes<uint64_t>(TypeBits))
             << FuncBits;
  Token |= getStableSipHash(*Metadata.FunctionName) &
           maskTrailingOnes<uint64_t>(FuncBits);
  return Token;
}

std::optional<uint64_t> llvm::getAllocToken(AllocTokenMode Mode,
                                            const AllocTokenMetadata &Metadata,
                                            uint64_t MaxTokens) {
  assert(MaxTokens && "Must provide non-zero max tokens");

  switch (Mode) {
  case AllocTokenMode::Increment:
  case AllocTokenMode::Random:
    // Stateful modes cannot be implemented as a pure function.
    return std::nullopt;

  case AllocTokenMode::TypeFuncHash:
  case AllocTokenMode::TypeFuncHashPointerSplit:
    // Depends on the function containing the allocation, which may be unknown
    // (e.g. in constant expressions).
    if (!Metadata.FunctionName)
      return std::nullopt;
    return getTypeFuncHash(Metadata, MaxTokens,
                           Mode == AllocTokenMode::TypeFuncHashPointerSplit);

  case AllocTokenMode::TypeHash:
    return getStableHash(Metadata, MaxTokens);

  case AllocTokenMode::TypeHashPointerSplit: {
    if (MaxTokens == 1)
      return 0;
    const uint64_t HalfTokens = MaxTokens / 2;
    uint64_t Hash = getStableHash(Metadata, HalfTokens);
    if (Metadata.ContainsPointer)
      Hash += HalfTokens;
    return Hash;
  }
  }

  llvm_unreachable("");
}
