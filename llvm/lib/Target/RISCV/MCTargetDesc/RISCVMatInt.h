//===- RISCVMatInt.h - Immediate materialisation ---------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_MCTARGETDESC_MATINT_H
#define LLVM_LIB_TARGET_RISCV_MCTARGETDESC_MATINT_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include <cstdint>

namespace llvm {
class APInt;

namespace RISCVMatInt {

enum OpndKind {
  RegImm, // ADDI/ADDIW/SLLI/SRLI/BSETI/BCLRI
  Imm,    // LUI
  RegReg // SH1ADD/SH2ADD/SH3ADD
};

class Inst {
  unsigned Opc;
  // Reg0 and Reg1 are the offset in the containing sequence which
  // define the vreg used as the respective operand (if any).  Note
  // that a sequence implicitly starts with X0, so an offset one
  // past the start of the sequence is valid, and means X0.
  uint8_t Reg0 : 4;
  uint8_t Reg1 : 4;
  int32_t Imm : 24; // The largest value we need to store is 20 bits.

public:
  Inst(unsigned Opc, int64_t I, uint8_t R0, uint8_t R1)
    : Opc(Opc), Reg0(R0), Reg1(R1), Imm(I) {
    assert(I == Imm && "truncated");
    assert(Reg0 == R0 && Reg1 == R1 && "truncated");
  }

  unsigned getOpcode() const { return Opc; }
  int64_t getImm() const { return Imm; }
  uint8_t getReg0() const { return Reg0; }
  uint8_t getReg1() const { return Reg1; }

  OpndKind getOpndKind() const;
};
static_assert(sizeof(Inst) == 8);

using InstSeq = SmallVector<Inst, 8>;

// Helper to generate an instruction sequence that will materialise the given
// immediate value into a register. A sequence of instructions represented by a
// simple struct is produced rather than directly emitting the instructions in
// order to allow this helper to be used from both the MC layer and during
// instruction selection.
InstSeq generateInstSeq(int64_t Val, const FeatureBitset &ActiveFeatures,
                        bool AllowMultipleVRegs = false);

// Helper to estimate the number of instructions required to materialise the
// given immediate value into a register. This estimate does not account for
// `Val` possibly fitting into an immediate, and so may over-estimate.
//
// This will attempt to produce instructions to materialise `Val` as an
// `Size`-bit immediate.
//
// If CompressionCost is true it will use a different cost calculation if RVC is
// enabled. This should be used to compare two different sequences to determine
// which is more compressible.
int getIntMatCost(const APInt &Val, unsigned Size,
                  const FeatureBitset &ActiveFeatures,
                  bool CompressionCost = false,
                  bool AllowMultipleVRegs = false);
} // namespace RISCVMatInt
} // namespace llvm
#endif
