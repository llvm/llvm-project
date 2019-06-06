//===- DPUOperandDecoder.cpp - Disassembler for DPU -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of the DPU Disassembler.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DPU_DISASSEMBLER_DPUOPERANDDECODER_H
#define LLVM_LIB_TARGET_DPU_DISASSEMBLER_DPUOPERANDDECODER_H

#include "MCTargetDesc/DPUAsmCondition.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"

namespace llvm {
using namespace DPUAsmCondition;

class DPUOperandDecoder {
private:
  uint64_t Address;
  const MCDisassembler *Decoder;

public:
  DPUOperandDecoder(uint64_t PC, const MCDisassembler *DAsm)
      : Address{PC}, Decoder{DAsm} {};

  MCDisassembler::DecodeStatus Decode_endian(MCInst &MI, uint64_t Value);

  MCDisassembler::DecodeStatus Decode_ra(MCInst &MI, uint64_t Value);

  MCDisassembler::DecodeStatus Decode_rb(MCInst &MI, uint64_t Value);

  MCDisassembler::DecodeStatus Decode_db(MCInst &MI, uint64_t Value);

  MCDisassembler::DecodeStatus Decode_imm(MCInst &MI, int32_t Value);

  MCDisassembler::DecodeStatus Decode_imm(MCInst &MI, int32_t Value,
                                          int32_t ValueSize);

  MCDisassembler::DecodeStatus Decode_immDma(MCInst &MI, int32_t Value);

  MCDisassembler::DecodeStatus Decode_cc(MCInst &MI, ConditionClass ccClass,
                                         uint64_t Value);

  MCDisassembler::DecodeStatus Decode_false_cc(MCInst &MI);

  MCDisassembler::DecodeStatus Decode_pc(MCInst &MI, uint64_t Value);

  MCDisassembler::DecodeStatus Decode_off(MCInst &MI, int32_t Value);

  MCDisassembler::DecodeStatus Decode_off(MCInst &MI, int32_t Value,
                                          int32_t ValueSize);

  MCDisassembler::DecodeStatus Decode_rc(MCInst &MI, uint64_t Value);

  MCDisassembler::DecodeStatus Decode_dc(MCInst &MI, uint64_t Value);

  MCDisassembler::DecodeStatus Decode_shift(MCInst &MI, uint64_t Value);

  MCDisassembler::DecodeStatus Decode_zero(MCInst &MI, uint64_t Value);
};
} // end namespace llvm

#endif // LLVM_LIB_TARGET_DPU_DISASSEMBLER_DPUOPERANDDECODER_H
