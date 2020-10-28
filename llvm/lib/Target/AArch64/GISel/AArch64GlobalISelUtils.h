//===- AArch64GlobalISelUtils.h ----------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file APIs for AArch64-specific helper functions used in the GlobalISel
/// pipeline.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_GISEL_AARCH64GLOBALISELUTILS_H
#define LLVM_LIB_TARGET_AARCH64_GISEL_AARCH64GLOBALISELUTILS_H

#include <cstdint>

namespace llvm {
namespace AArch64GISelUtils {

/// \returns true if \p C is a legal immediate operand for an arithmetic
/// instruction.
constexpr bool isLegalArithImmed(const uint64_t C) {
  return (C >> 12 == 0) || ((C & 0xFFFULL) == 0 && C >> 24 == 0);
}

} // namespace AArch64GISelUtils
} // namespace llvm

#endif
