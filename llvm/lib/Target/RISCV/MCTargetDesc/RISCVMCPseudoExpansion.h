//===-- RISCVMCPseudoExpansion.h - RISCV MC Pseudo Expansion ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// This file describes helpers to expand pseudo MC instructions that are usable
/// in the AsmParser and the AsmPrinter.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVMCPSEUDOEXPANSION_H
#define LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVMCPSEUDOEXPANSION_H

#include <cstdint>

namespace llvm {

class MCStreamer;
class MCSubtargetInfo;

void emitRISCVLoadImm(unsigned DestReg, int64_t Value, MCStreamer &Out,
                      const MCSubtargetInfo *STI);

} // namespace llvm

#endif
