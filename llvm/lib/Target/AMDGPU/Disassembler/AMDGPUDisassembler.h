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
  public:
    AMDGPUDisassembler(const MCSubtargetInfo &STI, MCContext &Ctx) :
      MCDisassembler(STI, Ctx) {}

    ~AMDGPUDisassembler() {}

    DecodeStatus getInstruction(MCInst &MI, uint64_t &Size,
                                ArrayRef<uint8_t> Bytes, uint64_t Address,
                                raw_ostream &WS, raw_ostream &CS) const override;

    /// Decode inline float value in VSrc field
    DecodeStatus DecodeLitFloat(unsigned Imm, uint32_t& F) const;
    /// Decode inline integer value in VSrc field
    DecodeStatus DecodeLitInteger(unsigned Imm, int64_t& I) const;
    /// Decode VGPR register
    DecodeStatus DecodeVgprRegister(unsigned Val, unsigned& RegID) const;
    /// Decode SGPR register
    DecodeStatus DecodeSgprRegister(unsigned Val, unsigned& RegID) const;
    /// Decode register in VSrc field
    DecodeStatus DecodeSrcRegister(unsigned Val, unsigned& RegID) const;

    DecodeStatus DecodeVS_32RegisterClass(MCInst &Inst, unsigned Imm, 
                                          uint64_t Addr) const;

    DecodeStatus DecodeVGPR_32RegisterClass(MCInst &Inst, unsigned Imm, 
                                            uint64_t Addr) const;
  };
} // namespace llvm

#endif //LLVM_LIB_TARGET_AMDGPU_DISASSEMBLER_AMDGPUDISASSEMBLER_H
