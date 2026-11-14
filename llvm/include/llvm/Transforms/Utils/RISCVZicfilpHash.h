//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers for computing the 20-bit RISCV-Zicfilp landing pad label from a
// mangled type name.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_RISCVZICFILPHASH_H
#define LLVM_TRANSFORMS_UTILS_RISCVZICFILPHASH_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MD5.h"
#include <cstdint>

namespace llvm {
namespace RISCVCFI {
constexpr uint32_t ZicfilpFuncSigFallbackLabel = 0xfea61;

/// Apply the psABI 128-bit-MD5-to-20-bit-label reduction.
LLVM_ABI uint32_t reduceZicfilpFuncSigLabel(const MD5::MD5Result &Digest);

/// Compute MD5(Signature), then apply the psABI reduction.
LLVM_ABI uint32_t getZicfilpFuncSigLabel(StringRef Signature);
} // namespace RISCVCFI
} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_RISCVZICFILPHASH_H
