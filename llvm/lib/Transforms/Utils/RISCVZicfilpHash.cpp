//===-- RISCVZicfilpHash.cpp - helpers to generate lpad labels ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generates RISCV-Zicfilp landing pad labels
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/RISCVZicfilpHash.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"

namespace llvm {
namespace RISCVCFI {
uint32_t reduceZicfilpFuncSigLabel(const llvm::MD5::MD5Result &Digest) {
  // per riscv-elf-psabi (draft
  // https://github.com/riscv-non-isa/riscv-elf-psabi-doc/pull/434), the rules
  // to determine the label are:
  // 1. use the lower 20 bits of the hash
  // 2. if the current value is 0, use the next 20 bits
  // 3. if fewer than 20 bits remain, pad with 0
  // 4. if the value is still 0, hash the string "RISC-V" and use the lower 20
  // bits of its hash. (this value is constant: 0xfea61. no need to compute
  // every time.)

  // Treat the digest as a single little-endian 128-bit integer. Zero-extend it
  // so that the final 20-bit chunk is padded with zeros (rule 3).
  const uint64_t Words[] = {Digest.low(), Digest.high()};
  llvm::APInt Hash = llvm::APInt(128, Words).zext(140);

  uint32_t Label = 0;
  for (unsigned Shift = 0; Shift < 128 && !Label; Shift += 20)
    // extractBitsAsZExtValue doesn't pad the selected bit range, it pads the returned uint64_t.
    // ordinary extractBits would return an APInt which needs to be unwrapped with .getZExtValue before casting to uint32_t.
    Label = static_cast<uint32_t>(Hash.extractBitsAsZExtValue(20, Shift));
  if (!Label)
    Label = ZicfilpFuncSigFallbackLabel;

  return Label;
}

/// Compute MD5(Signature), then apply the psABI reduction.
uint32_t getZicfilpFuncSigLabel(StringRef Signature) {
  llvm::MD5 Hash;
  Hash.update(Signature);
  llvm::MD5::MD5Result Result;
  Hash.final(Result);
  return reduceZicfilpFuncSigLabel(Result);
}
} // namespace RISCVCFI
} // namespace llvm