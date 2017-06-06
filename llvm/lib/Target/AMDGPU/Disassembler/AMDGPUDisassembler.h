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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCDisassembler/MCRelocationInfo.h"
#include "llvm/MC/MCDisassembler/MCSymbolizer.h"
#include <algorithm>
#include <cstdint>
#include <memory>

namespace llvm {

class MCContext;
class MCInst;
class MCOperand;
class MCSubtargetInfo;
class Twine;

//===----------------------------------------------------------------------===//
// AMDGPUDisassembler
//===----------------------------------------------------------------------===//

class AMDGPUDisassembler : public MCDisassembler {
private:
  mutable ArrayRef<uint8_t> Bytes;
  mutable uint32_t Literal;
  mutable bool HasLiteral;

public:
  AMDGPUDisassembler(const MCSubtargetInfo &STI, MCContext &Ctx) :
    MCDisassembler(STI, Ctx) {}

  ~AMDGPUDisassembler() override = default;

  DecodeStatus getInstruction(MCInst &MI, uint64_t &Size,
                              ArrayRef<uint8_t> Bytes, uint64_t Address,
                              raw_ostream &WS, raw_ostream &CS) const override;

  const char* getRegClassName(unsigned RegClassID) const;

  MCOperand createRegOperand(unsigned int RegId) const;
  MCOperand createRegOperand(unsigned RegClassID, unsigned Val) const;
  MCOperand createSRegOperand(unsigned SRegClassID, unsigned Val) const;

  MCOperand errOperand(unsigned V, const Twine& ErrMsg) const;

  DecodeStatus tryDecodeInst(const uint8_t* Table,
                              MCInst &MI,
                              uint64_t Inst,
                              uint64_t Address) const;

  MCOperand decodeOperand_VGPR_32(unsigned Val) const;
  MCOperand decodeOperand_VS_32(unsigned Val) const;
  MCOperand decodeOperand_VS_64(unsigned Val) const;
  MCOperand decodeOperand_VSrc16(unsigned Val) const;
  MCOperand decodeOperand_VSrcV216(unsigned Val) const;

  MCOperand decodeOperand_VReg_64(unsigned Val) const;
  MCOperand decodeOperand_VReg_96(unsigned Val) const;
  MCOperand decodeOperand_VReg_128(unsigned Val) const;

  MCOperand decodeOperand_SReg_32(unsigned Val) const;
  MCOperand decodeOperand_SReg_32_XM0_XEXEC(unsigned Val) const;
  MCOperand decodeOperand_SReg_64(unsigned Val) const;
  MCOperand decodeOperand_SReg_64_XEXEC(unsigned Val) const;
  MCOperand decodeOperand_SReg_128(unsigned Val) const;
  MCOperand decodeOperand_SReg_256(unsigned Val) const;
  MCOperand decodeOperand_SReg_512(unsigned Val) const;

  enum OpWidthTy {
    OPW32,
    OPW64,
    OPW128,
    OPW16,
    OPWV216,
    OPW_LAST_,
    OPW_FIRST_ = OPW32
  };

  unsigned getVgprClassId(const OpWidthTy Width) const;
  unsigned getSgprClassId(const OpWidthTy Width) const;
  unsigned getTtmpClassId(const OpWidthTy Width) const;

  static MCOperand decodeIntImmed(unsigned Imm);
  static MCOperand decodeFPImmed(OpWidthTy Width, unsigned Imm);
  MCOperand decodeLiteralConstant() const;

  MCOperand decodeSrcOp(const OpWidthTy Width, unsigned Val) const;
  MCOperand decodeSpecialReg32(unsigned Val) const;
  MCOperand decodeSpecialReg64(unsigned Val) const;

  MCOperand decodeSDWA9Src(const OpWidthTy Width, unsigned Val) const;
  MCOperand decodeSDWA9Src16(unsigned Val) const;
  MCOperand decodeSDWA9Src32(unsigned Val) const;
  MCOperand decodeSDWA9VopcDst(unsigned Val) const;
};

//===----------------------------------------------------------------------===//
// AMDGPUSymbolizer
//===----------------------------------------------------------------------===//

class AMDGPUSymbolizer : public MCSymbolizer {
private:
  void *DisInfo;

public:
  AMDGPUSymbolizer(MCContext &Ctx, std::unique_ptr<MCRelocationInfo> &&RelInfo,
                   void *disInfo)
                   : MCSymbolizer(Ctx, std::move(RelInfo)), DisInfo(disInfo) {}

  bool tryAddingSymbolicOperand(MCInst &Inst, raw_ostream &cStream,
                                int64_t Value, uint64_t Address,
                                bool IsBranch, uint64_t Offset,
                                uint64_t InstSize) override;

  void tryAddingPcLoadReferenceComment(raw_ostream &cStream,
                                       int64_t Value,
                                       uint64_t Address) override;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_DISASSEMBLER_AMDGPUDISASSEMBLER_H
