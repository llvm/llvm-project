//=-- RISCVStateAttributes.h - Helper for interpreting RISC-V attributes -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVSTATEATTRIBUTES_H
#define LLVM_LIB_TARGET_RISCV_RISCVSTATEATTRIBUTES_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"

namespace llvm {
namespace RISCVState {

inline constexpr StringLiteral Attributes[] = {
    "riscv_in", "riscv_out", "riscv_inout", "riscv_preserves", "riscv_new"};

inline bool hasAttribute(const Function &F) {
  return any_of(Attributes, [&F](StringRef A) { return F.hasFnAttribute(A); });
}

} // namespace RISCVState
} // namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVSTATEATTRIBUTES_H
