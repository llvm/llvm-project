//===- RISCVMatInt.h - Immediate materialisation ---------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_MATINT_H
#define LLVM_LIB_TARGET_RISCV_MATINT_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MachineValueType.h"
#include <cstdint>

namespace llvm {

namespace RISCVMatInt {
struct Inst {
  unsigned Opc;
  int64_t Imm;

  Inst(unsigned Opc, int64_t Imm) : Opc(Opc), Imm(Imm) {}
};
using InstSeq = SmallVector<Inst, 8>;

// Helper to generate an instruction sequence that will materialise the given
// immediate value into a register. A sequence of instructions represented by
// a simple struct produced rather than directly emitting the instructions in
// order to allow this helper to be used from both the MC layer and during
// instruction selection.
void generateInstSeq(int64_t Val, bool IsRV64, InstSeq &Res);

// Helper to estimate the number of instructions required to materialise the
// given immediate value into a register. This estimate does not account for
// `Val` possibly fitting into an immediate, and so may over-estimate.
//
// This will attempt to produce instructions to materialise `Val` as an
// `Size`-bit immediate. `IsRV64` should match the target architecture.
int getIntMatCost(const APInt &Val, unsigned Size, bool IsRV64);
} // namespace RISCVMatInt
} // namespace llvm
#endif
