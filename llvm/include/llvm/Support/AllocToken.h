//===- llvm/Support/AllocToken.h - Allocation Token Calculation -----*- C++ -*//
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

#ifndef LLVM_SUPPORT_ALLOCTOKEN_H
#define LLVM_SUPPORT_ALLOCTOKEN_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <optional>

namespace llvm {

/// Modes for generating allocation token IDs.
enum class AllocTokenMode {
  /// Incrementally increasing token ID.
  Increment,

  /// Simple mode that returns a statically-assigned random token ID.
  Random,

  /// Token ID based on allocated type hash.
  TypeHash,

  /// Token ID based on allocated type hash, where the top half ID-space is
  /// reserved for types that contain pointers and the bottom half for types
  /// that do not contain pointers.
  TypeHashPointerSplit,
};

/// The default allocation token mode.
inline constexpr AllocTokenMode DefaultAllocTokenMode =
    AllocTokenMode::TypeHashPointerSplit;

/// Returns the AllocTokenMode from its canonical string name; if an invalid
/// name was provided returns nullopt.
LLVM_ABI std::optional<AllocTokenMode>
getAllocTokenModeFromString(StringRef Name);

/// Metadata about an allocation used to generate a token ID.
struct AllocTokenMetadata {
  SmallString<64> TypeName;
  bool ContainsPointer;
};

/// Calculates stable allocation token ID. Returns std::nullopt for stateful
/// modes that are only available in the AllocToken pass.
///
/// \param Mode The token generation mode.
/// \param Metadata The metadata about the allocation.
/// \param MaxTokens The maximum number of tokens (must not be 0)
/// \return The calculated allocation token ID, or std::nullopt.
LLVM_ABI std::optional<uint64_t>
getAllocToken(AllocTokenMode Mode, const AllocTokenMetadata &Metadata,
              uint64_t MaxTokens);

} // end namespace llvm

#endif // LLVM_SUPPORT_ALLOCTOKEN_H
