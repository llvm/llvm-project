//===- AMDGPUDisassembler.hpp - Disassembler for AMDGPU ISA -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/DataExtractor.h"
#include <memory>

namespace llvm {

class MCInst;
class MCOperand;
class MCSubtargetInfo;
class Twine;

//===----------------------------------------------------------------------===//
// AMDGPUDisassembler
//===----------------------------------------------------------------------===//

class AMDGPUDisassembler : public MCDisassembler {
private:
  std::unique_ptr<MCInstrInfo const> const MCII;
  const MCRegisterInfo &MRI;
  const unsigned TargetMaxInstBytes;
  mutable ArrayRef<uint8_t> Bytes;
  mutable uint32_t Literal;
  mutable bool HasLiteral;

public:
  AMDGPUDisassembler(const MCSubtargetInfo &STI, MCContext &Ctx,
                     MCInstrInfo const *MCII);
  ~AMDGPUDisassembler() override = default;

  DecodeStatus getInstruction(MCInst &MI, uint64_t &Size,
                              ArrayRef<uint8_t> Bytes, uint64_t Address,
                              raw_ostream &CS) const override;

  const char* getRegClassName(unsigned RegClassID) const;

  MCOperand createRegOperand(unsigned int RegId) const;
  MCOperand createRegOperand(unsigned RegClassID, unsigned Val) const;
  MCOperand createSRegOperand(unsigned SRegClassID, unsigned Val) const;

  MCOperand errOperand(unsigned V, const Twine& ErrMsg) const;

  DecodeStatus tryDecodeInst(const uint8_t* Table, MCInst &MI, uint64_t Inst,
                             uint64_t Address) const;

  Optional<DecodeStatus> onSymbolStart(SymbolInfoTy &Symbol, uint64_t &Size,
                                       ArrayRef<uint8_t> Bytes,
                                       uint64_t Address,
                                       raw_ostream &CStream) const override;

  DecodeStatus decodeKernelDescriptor(StringRef KdName, ArrayRef<uint8_t> Bytes,
                                      uint64_t KdAddress) const;

  DecodeStatus
  decodeKernelDescriptorDirective(DataExtractor::Cursor &Cursor,
                                  ArrayRef<uint8_t> Bytes,
                                  raw_string_ostream &KdStream) const;

  /// Decode as directives that handle COMPUTE_PGM_RSRC1.
  /// \param FourByteBuffer - Bytes holding contents of COMPUTE_PGM_RSRC1.
  /// \param KdStream       - Stream to write the disassembled directives to.
  // NOLINTNEXTLINE(readability-identifier-naming)
  DecodeStatus decodeCOMPUTE_PGM_RSRC1(uint32_t FourByteBuffer,
                                       raw_string_ostream &KdStream) const;

  /// Decode as directives that handle COMPUTE_PGM_RSRC2.
  /// \param FourByteBuffer - Bytes holding contents of COMPUTE_PGM_RSRC2.
  /// \param KdStream       - Stream to write the disassembled directives to.
  // NOLINTNEXTLINE(readability-identifier-naming)
  DecodeStatus decodeCOMPUTE_PGM_RSRC2(uint32_t FourByteBuffer,
                                       raw_string_ostream &KdStream) const;

  DecodeStatus convertFMAanyK(MCInst &MI, int ImmLitIdx) const;
  DecodeStatus convertSDWAInst(MCInst &MI) const;
  DecodeStatus convertDPP8Inst(MCInst &MI) const;
  DecodeStatus convertMIMGInst(MCInst &MI) const;

  MCOperand decodeOperand_VGPR_32(unsigned Val) const;
  MCOperand decodeOperand_VRegOrLds_32(unsigned Val) const;

  MCOperand decodeOperand_VS_32(unsigned Val) const;
  MCOperand decodeOperand_VS_64(unsigned Val) const;
  MCOperand decodeOperand_VS_128(unsigned Val) const;
  MCOperand decodeOperand_VSrc16(unsigned Val) const;
  MCOperand decodeOperand_VSrcV216(unsigned Val) const;
  MCOperand decodeOperand_VSrcV232(unsigned Val) const;

  MCOperand decodeOperand_VReg_64(unsigned Val) const;
  MCOperand decodeOperand_VReg_96(unsigned Val) const;
  MCOperand decodeOperand_VReg_128(unsigned Val) const;
  MCOperand decodeOperand_VReg_256(unsigned Val) const;
  MCOperand decodeOperand_VReg_512(unsigned Val) const;
  MCOperand decodeOperand_VReg_1024(unsigned Val) const;

  MCOperand decodeOperand_SReg_32(unsigned Val) const;
  MCOperand decodeOperand_SReg_32_XM0_XEXEC(unsigned Val) const;
  MCOperand decodeOperand_SReg_32_XEXEC_HI(unsigned Val) const;
  MCOperand decodeOperand_SRegOrLds_32(unsigned Val) const;
  MCOperand decodeOperand_SReg_64(unsigned Val) const;
  MCOperand decodeOperand_SReg_64_XEXEC(unsigned Val) const;
  MCOperand decodeOperand_SReg_128(unsigned Val) const;
  MCOperand decodeOperand_SReg_256(unsigned Val) const;
  MCOperand decodeOperand_SReg_512(unsigned Val) const;

  MCOperand decodeOperand_AGPR_32(unsigned Val) const;
  MCOperand decodeOperand_AReg_64(unsigned Val) const;
  MCOperand decodeOperand_AReg_128(unsigned Val) const;
  MCOperand decodeOperand_AReg_256(unsigned Val) const;
  MCOperand decodeOperand_AReg_512(unsigned Val) const;
  MCOperand decodeOperand_AReg_1024(unsigned Val) const;
  MCOperand decodeOperand_AV_32(unsigned Val) const;
  MCOperand decodeOperand_AV_64(unsigned Val) const;
  MCOperand decodeOperand_AV_128(unsigned Val) const;
  MCOperand decodeOperand_AV_512(unsigned Val) const;

  enum OpWidthTy {
    OPW32,
    OPW64,
    OPW96,
    OPW128,
    OPW160,
    OPW256,
    OPW512,
    OPW1024,
    OPW16,
    OPWV216,
    OPWV232,
    OPW_LAST_,
    OPW_FIRST_ = OPW32
  };

  unsigned getVgprClassId(const OpWidthTy Width) const;
  unsigned getAgprClassId(const OpWidthTy Width) const;
  unsigned getSgprClassId(const OpWidthTy Width) const;
  unsigned getTtmpClassId(const OpWidthTy Width) const;

  static MCOperand decodeIntImmed(unsigned Imm);
  static MCOperand decodeFPImmed(OpWidthTy Width, unsigned Imm);
  MCOperand decodeMandatoryLiteralConstant(unsigned Imm) const;
  MCOperand decodeLiteralConstant() const;

  MCOperand decodeSrcOp(const OpWidthTy Width, unsigned Val,
                        bool MandatoryLiteral = false) const;
  MCOperand decodeDstOp(const OpWidthTy Width, unsigned Val) const;
  MCOperand decodeSpecialReg32(unsigned Val) const;
  MCOperand decodeSpecialReg64(unsigned Val) const;

  MCOperand decodeSDWASrc(const OpWidthTy Width, unsigned Val) const;
  MCOperand decodeSDWASrc16(unsigned Val) const;
  MCOperand decodeSDWASrc32(unsigned Val) const;
  MCOperand decodeSDWAVopcDst(unsigned Val) const;

  MCOperand decodeBoolReg(unsigned Val) const;

  int getTTmpIdx(unsigned Val) const;

  const MCInstrInfo *getMCII() const { return MCII.get(); }

  bool isVI() const;
  bool isGFX9() const;
  bool isGFX90A() const;
  bool isGFX9Plus() const;
  bool isGFX10() const;
  bool isGFX10Plus() const;

  bool hasArchitectedFlatScratch() const;
};

//===----------------------------------------------------------------------===//
// AMDGPUSymbolizer
//===----------------------------------------------------------------------===//

class AMDGPUSymbolizer : public MCSymbolizer {
private:
  void *DisInfo;
  std::vector<uint64_t> ReferencedAddresses;

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

  ArrayRef<uint64_t> getReferencedAddresses() const override {
    return ReferencedAddresses;
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_DISASSEMBLER_AMDGPUDISASSEMBLER_H
