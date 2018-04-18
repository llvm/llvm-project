//===-- RISCVMCPseudoExpansion.cpp - RISCV MC Pseudo Expansion ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// This file provides helpers to expand pseudo MC instructions that are usable
/// in the AsmParser and the AsmPrinter.
///
//===----------------------------------------------------------------------===//

#include "RISCVMCPseudoExpansion.h"
#include "RISCVMCTargetDesc.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/MathExtras.h"

#include <cassert>

using namespace llvm;

void llvm::emitRISCVLoadImm(unsigned DestReg, int64_t Value, MCStreamer &Out,
                            const MCSubtargetInfo *STI) {
  if (isInt<32>(Value)) {
    // Emits the MC instructions for loading a 32-bit constant into a register.
    //
    // Depending on the active bits in the immediate Value v, the following
    // instruction sequences are emitted:
    //
    // v == 0                        : ADDI(W)
    // v[0,12) != 0 && v[12,32) == 0 : ADDI(W)
    // v[0,12) == 0 && v[12,32) != 0 : LUI
    // v[0,32) != 0                  : LUI+ADDI(W)
    //
    int64_t Hi20 = ((Value + 0x800) >> 12) & 0xFFFFF;
    int64_t Lo12 = SignExtend64<12>(Value);
    unsigned SrcReg = RISCV::X0;

    if (Hi20) {
      Out.EmitInstruction(
          MCInstBuilder(RISCV::LUI).addReg(DestReg).addImm(Hi20), *STI);
      SrcReg = DestReg;
    }

    if (Lo12 || Hi20 == 0) {
      unsigned AddiOpcode =
          STI->hasFeature(RISCV::Feature64Bit) ? RISCV::ADDIW : RISCV::ADDI;
      Out.EmitInstruction(
          MCInstBuilder(AddiOpcode).addReg(DestReg).addReg(SrcReg).addImm(Lo12),
          *STI);
    }
    return;
  }
  assert(STI->hasFeature(RISCV::Feature64Bit) &&
         "Target must be 64-bit to support a >32-bit constant");

  // In the worst case, for a full 64-bit constant, a sequence of 8 instructions
  // (i.e., LUI+ADDIW+SLLI+ADDI+SLLI+ADDI+SLLI+ADDI) has to be emmitted. Note
  // that the first two instructions (LUI+ADDIW) can contribute up to 32 bits
  // while the following ADDI instructions contribute up to 12 bits each.
  //
  // On the first glance, implementing this seems to be possible by simply
  // emitting the most significant 32 bits (LUI+ADDIW) followed by as many left
  // shift (SLLI) and immediate additions (ADDI) as needed. However, due to the
  // fact that ADDI performs a sign extended addition, doing it like that would
  // only be possible when at most 11 bits of the ADDI instructions are used.
  // Using all 12 bits of the ADDI instructions, like done by GAS, actually
  // requires that the constant is processed starting with the least significant
  // bit.
  //
  // In the following, constants are processed from LSB to MSB but instruction
  // emission is performed from MSB to LSB by recursively calling
  // emitRISCVLoadImm. In each recursion, first the lowest 12 bits are removed
  // from the constant and the optimal shift amount, which can be greater than
  // 12 bits if the constant is sparse, is determined. Then, the shifted
  // remaining constant is processed recursively and gets emitted as soon as it
  // fits into 32 bits. The emission of the shifts and additions is subsequently
  // performed when the recursion returns.
  //
  int64_t Lo12 = SignExtend64<12>(Value);
  int64_t Hi52 = (Value + 0x800) >> 12;
  int ShiftAmount = 12 + findFirstSet((uint64_t)Hi52);
  Hi52 = SignExtend64(Hi52 >> (ShiftAmount - 12), 64 - ShiftAmount);

  emitRISCVLoadImm(DestReg, Hi52, Out, STI);

  Out.EmitInstruction(MCInstBuilder(RISCV::SLLI)
                          .addReg(DestReg)
                          .addReg(DestReg)
                          .addImm(ShiftAmount),
                      *STI);

  if (Lo12)
    Out.EmitInstruction(
        MCInstBuilder(RISCV::ADDI).addReg(DestReg).addReg(DestReg).addImm(Lo12),
        *STI);
}
