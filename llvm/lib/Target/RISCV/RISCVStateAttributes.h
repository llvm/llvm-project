//=-- RISCVStateAttributes.h - Helper for interpreting RISC-V attributes -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVSTATEATTRIBUTES_H
#define LLVM_LIB_TARGET_RISCV_RISCVSTATEATTRIBUTES_H

#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"

namespace llvm {
namespace RISCVState {

inline bool hasAttribute(const Function &F) {
  for (Attribute A : F.getAttributes().getFnAttrs()) {
    if (!A.isStringAttribute())
      continue;

    StringRef Kind = A.getKindAsString();
    if (Kind == "riscv_in" || Kind == "riscv_out" ||
        Kind == "riscv_inout" || Kind == "riscv_preserves" ||
        Kind == "riscv_new")
      return true;
  }
  return false;
}

} // namespace RISCVState
} // namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVSTATEATTRIBUTES_H
