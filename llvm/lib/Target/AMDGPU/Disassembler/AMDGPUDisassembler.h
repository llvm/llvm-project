//===-- AMDGPUDisassembler.hpp - Disassembler for AMDGPU ISA ---*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// This file contains declaration for AMDGPU ISA disassembler
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_DISASSEMBLER_AMDGPUDISASSEMBLER_H
#define LLVM_LIB_TARGET_AMDGPU_DISASSEMBLER_AMDGPUDISASSEMBLER_H

#include "llvm/MC/MCDisassembler/MCDisassembler.h"

namespace llvm {

  class MCContext;
  class MCInst;
  class MCSubtargetInfo;

  class AMDGPUDisassembler : public MCDisassembler {
  private:
    /// true if 32-bit literal constant is placed after instruction
    mutable bool HasLiteral;
    mutable ArrayRef<uint8_t> Bytes;

  public:
    AMDGPUDisassembler(const MCSubtargetInfo &STI, MCContext &Ctx) :
      MCDisassembler(STI, Ctx), HasLiteral(false) {}

    ~AMDGPUDisassembler() {}

    DecodeStatus getInstruction(MCInst &MI, uint64_t &Size,
                                ArrayRef<uint8_t> Bytes, uint64_t Address,
                                raw_ostream &WS, raw_ostream &CS) const override;

    /// Decode inline float value in SRC field
    DecodeStatus DecodeImmedFloat(unsigned Imm, uint32_t &F) const;
    /// Decode inline double value in SRC field
    DecodeStatus DecodeImmedDouble(unsigned Imm, uint64_t &D) const;
    /// Decode inline integer value in SRC field
    DecodeStatus DecodeImmedInteger(unsigned Imm, int64_t &I) const;
    /// Decode VGPR register
    DecodeStatus DecodeVgprRegister(unsigned Val, unsigned &RegID,
                                    unsigned Size = 32) const;
    /// Decode SGPR register
    DecodeStatus DecodeSgprRegister(unsigned Val, unsigned &RegID,
                                    unsigned Size = 32) const;
    /// Decode 32-bit register in SRC field
    DecodeStatus DecodeSrc32Register(unsigned Val, unsigned &RegID) const;
    /// Decode 64-bit register in SRC field
    DecodeStatus DecodeSrc64Register(unsigned Val, unsigned &RegID) const;

    /// Decode literal constant after instruction
    DecodeStatus DecodeLiteralConstant(MCInst &Inst, uint64_t &Literal) const;

    DecodeStatus DecodeVGPR_32RegisterClass(MCInst &Inst, unsigned Imm,
                                            uint64_t Addr) const;

    DecodeStatus DecodeVSRegisterClass(MCInst &Inst, unsigned Imm,
                                       uint64_t Addr, bool Is32) const;

    DecodeStatus DecodeVS_32RegisterClass(MCInst &Inst, unsigned Imm,
                                          uint64_t Addr) const;

    DecodeStatus DecodeVS_64RegisterClass(MCInst &Inst, unsigned Imm,
                                          uint64_t Addr) const;

    DecodeStatus DecodeVReg_64RegisterClass(MCInst &Inst, unsigned Imm,
                                            uint64_t Addr) const;
  };
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_DISASSEMBLER_AMDGPUDISASSEMBLER_H
