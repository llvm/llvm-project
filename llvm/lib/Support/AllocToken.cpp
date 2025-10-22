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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SipHash.h"

using namespace llvm;

static uint64_t getStableHash(const AllocTokenMetadata &Metadata,
                              uint64_t MaxTokens) {
  return getStableSipHash(Metadata.TypeName) % MaxTokens;
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
