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

#include "llvm/ADT/Optional.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/Register.h"
#include <cstdint>

namespace llvm {
namespace AArch64GISelUtils {

/// \returns true if \p C is a legal immediate operand for an arithmetic
/// instruction.
constexpr bool isLegalArithImmed(const uint64_t C) {
  return (C >> 12 == 0) || ((C & 0xFFFULL) == 0 && C >> 24 == 0);
}

/// \returns A value when \p MI is a vector splat of a Register or constant.
/// Checks for generic opcodes and AArch64-specific generic opcodes.
Optional<RegOrConstant> getAArch64VectorSplat(const MachineInstr &MI,
                                              const MachineRegisterInfo &MRI);

/// \returns A value when \p MI is a constant vector splat.
/// Checks for generic opcodes and AArch64-specific generic opcodes.
Optional<int64_t> getAArch64VectorSplatScalar(const MachineInstr &MI,
                                              const MachineRegisterInfo &MRI);

} // namespace AArch64GISelUtils
} // namespace llvm

#endif
