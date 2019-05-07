//===-- DPUInstructionDecoder.cpp - Disassembler for DPU
//-------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Generated DPU instruction decoder class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DPU_DISASSEMBLER_DPUINSTRUCTIONDECODER_H
#define LLVM_LIB_TARGET_DPU_DISASSEMBLER_DPUINSTRUCTIONDECODER_H

#include "DPUOperandDecoder.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"

namespace llvm {
class DPUInstructionDecoder {
public:
  DPUInstructionDecoder() = default;

  MCDisassembler::DecodeStatus getInstruction(MCInst &MI, uint64_t &Insn,
                                              DPUOperandDecoder &DAsm,
                                              bool useSugar) const;
};
} // end namespace llvm

#endif // LLVM_LIB_TARGET_DPU_DISASSEMBLER_DPUINSTRUCTIONDECODER_H
