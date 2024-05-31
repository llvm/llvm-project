//===--- SipHash.h - An ABI-stable string SipHash ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A family of ABI-stable string hash algorithms based on SipHash, currently
// used to compute ptrauth discriminators.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SIPHASH_H
#define LLVM_SUPPORT_SIPHASH_H

#include <cstdint>

namespace llvm {
class StringRef;

/// Compute a stable 64-bit hash of the given string.
///
/// The exact algorithm is the little-endian interpretation of the
/// non-doubled (i.e. 64-bit) result of applying a SipHash-2-4 using
/// a specific key value which can be found in the source.
///
/// By "stable" we mean that the result of this hash algorithm will
/// the same across different compiler versions and target platforms.
uint64_t getPointerAuthStableSipHash64(StringRef S);

/// Compute a stable non-zero 16-bit hash of the given string.
///
/// This computes the full getPointerAuthStableSipHash64, but additionally
/// truncates it down to a non-zero 16-bit value.
///
/// We use a 16-bit discriminator because ARM64 can efficiently load
/// a 16-bit immediate into the high bits of a register without disturbing
/// the remainder of the value, which serves as a nice blend operation.
/// 16 bits is also sufficiently compact to not inflate a loader relocation.
/// We disallow zero to guarantee a different discriminator from the places
/// in the ABI that use a constant zero.
uint64_t getPointerAuthStableSipHash16(StringRef S);

} // end namespace llvm

#endif
