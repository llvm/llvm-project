//===- AMDGPUDisassembler.cpp - Disassembler for AMDGPU ISA ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// This file contains definition for AMDGPU ISA disassembler
//
//===----------------------------------------------------------------------===//

// ToDo: What to do with instruction suffixes (v_mov_b32 vs v_mov_b32_e32)?

#include "Disassembler/AMDGPUDisassembler.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "SIRegisterInfo.h"
#include "TargetInfo/AMDGPUTargetInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm-c/DisassemblerTypes.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDecoderOps.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-disassembler"

#define SGPR_MAX                                                               \
  (isGFX10Plus() ? AMDGPU::EncValues::SGPR_MAX_GFX10                           \
                 : AMDGPU::EncValues::SGPR_MAX_SI)

using DecodeStatus = llvm::MCDisassembler::DecodeStatus;

AMDGPUDisassembler::AMDGPUDisassembler(const MCSubtargetInfo &STI,
                                       MCContext &Ctx,
                                       MCInstrInfo const *MCII) :
  MCDisassembler(STI, Ctx), MCII(MCII), MRI(*Ctx.getRegisterInfo()),
  TargetMaxInstBytes(Ctx.getAsmInfo()->getMaxInstLength(&STI)) {

  // ToDo: AMDGPUDisassembler supports only VI ISA.
  if (!STI.getFeatureBits()[AMDGPU::FeatureGCN3Encoding] && !isGFX10Plus())
    report_fatal_error("Disassembly not yet supported for subtarget");
}

inline static MCDisassembler::DecodeStatus
addOperand(MCInst &Inst, const MCOperand& Opnd) {
  Inst.addOperand(Opnd);
  return Opnd.isValid() ?
    MCDisassembler::Success :
    MCDisassembler::Fail;
}

static int insertNamedMCOperand(MCInst &MI, const MCOperand &Op,
                                uint16_t NameIdx) {
  int OpIdx = AMDGPU::getNamedOperandIdx(MI.getOpcode(), NameIdx);
  if (OpIdx != -1) {
    auto I = MI.begin();
    std::advance(I, OpIdx);
    MI.insert(I, Op);
  }
  return OpIdx;
}

static DecodeStatus decodeSoppBrTarget(MCInst &Inst, unsigned Imm,
                                       uint64_t Addr,
                                       const MCDisassembler *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);

  // Our branches take a simm16, but we need two extra bits to account for the
  // factor of 4.
  APInt SignedOffset(18, Imm * 4, true);
  int64_t Offset = (SignedOffset.sext(64) + 4 + Addr).getSExtValue();

  if (DAsm->tryAddingSymbolicOperand(Inst, Offset, Addr, true, 2, 2, 0))
    return MCDisassembler::Success;
  return addOperand(Inst, MCOperand::createImm(Imm));
}

static DecodeStatus decodeSMEMOffset(MCInst &Inst, unsigned Imm, uint64_t Addr,
                                     const MCDisassembler *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  int64_t Offset;
  if (DAsm->isVI()) {         // VI supports 20-bit unsigned offsets.
    Offset = Imm & 0xFFFFF;
  } else {                    // GFX9+ supports 21-bit signed offsets.
    Offset = SignExtend64<21>(Imm);
  }
  return addOperand(Inst, MCOperand::createImm(Offset));
}

static DecodeStatus decodeBoolReg(MCInst &Inst, unsigned Val, uint64_t Addr,
                                  const MCDisassembler *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeBoolReg(Val));
}

#define DECODE_OPERAND(StaticDecoderName, DecoderName)                         \
  static DecodeStatus StaticDecoderName(MCInst &Inst, unsigned Imm,            \
                                        uint64_t /*Addr*/,                     \
                                        const MCDisassembler *Decoder) {       \
    auto DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);              \
    return addOperand(Inst, DAsm->DecoderName(Imm));                           \
  }

#define DECODE_OPERAND_REG(RegClass) \
DECODE_OPERAND(Decode##RegClass##RegisterClass, decodeOperand_##RegClass)

DECODE_OPERAND_REG(VGPR_32)
DECODE_OPERAND_REG(VGPR_32_Lo128)
DECODE_OPERAND_REG(VRegOrLds_32)
DECODE_OPERAND_REG(VS_32)
DECODE_OPERAND_REG(VS_64)
DECODE_OPERAND_REG(VS_128)

DECODE_OPERAND_REG(VReg_64)
DECODE_OPERAND_REG(VReg_96)
DECODE_OPERAND_REG(VReg_128)
DECODE_OPERAND_REG(VReg_256)
DECODE_OPERAND_REG(VReg_512)
DECODE_OPERAND_REG(VReg_1024)

DECODE_OPERAND_REG(SReg_32)
DECODE_OPERAND_REG(SReg_32_XM0_XEXEC)
DECODE_OPERAND_REG(SReg_32_XEXEC_HI)
DECODE_OPERAND_REG(SRegOrLds_32)
DECODE_OPERAND_REG(SReg_64)
DECODE_OPERAND_REG(SReg_64_XEXEC)
DECODE_OPERAND_REG(SReg_128)
DECODE_OPERAND_REG(SReg_256)
DECODE_OPERAND_REG(SReg_512)

DECODE_OPERAND_REG(AGPR_32)
DECODE_OPERAND_REG(AReg_64)
DECODE_OPERAND_REG(AReg_128)
DECODE_OPERAND_REG(AReg_256)
DECODE_OPERAND_REG(AReg_512)
DECODE_OPERAND_REG(AReg_1024)
DECODE_OPERAND_REG(AV_32)
DECODE_OPERAND_REG(AV_64)
DECODE_OPERAND_REG(AV_128)
DECODE_OPERAND_REG(AVDst_128)
DECODE_OPERAND_REG(AVDst_512)

static DecodeStatus decodeOperand_VSrc16(MCInst &Inst, unsigned Imm,
                                         uint64_t Addr,
                                         const MCDisassembler *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeOperand_VSrc16(Imm));
}

static DecodeStatus decodeOperand_VSrcV216(MCInst &Inst, unsigned Imm,
                                           uint64_t Addr,
                                           const MCDisassembler *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeOperand_VSrcV216(Imm));
}

static DecodeStatus decodeOperand_VSrcV232(MCInst &Inst, unsigned Imm,
                                           uint64_t Addr,
                                           const MCDisassembler *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeOperand_VSrcV232(Imm));
}

static DecodeStatus decodeOperand_VS_16(MCInst &Inst, unsigned Imm,
                                        uint64_t Addr,
                                        const MCDisassembler *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeOperand_VSrc16(Imm));
}

static DecodeStatus decodeOperand_VS_32(MCInst &Inst, unsigned Imm,
                                        uint64_t Addr,
                                        const MCDisassembler *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeOperand_VS_32(Imm));
}

static DecodeStatus decodeOperand_AReg_64(MCInst &Inst, unsigned Imm,
                                          uint64_t Addr,
                                          const MCDisassembler *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeSrcOp(AMDGPUDisassembler::OPW64, Imm | 512));
}

static DecodeStatus decodeOperand_AReg_128(MCInst &Inst, unsigned Imm,
                                           uint64_t Addr,
                                           const MCDisassembler *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeSrcOp(AMDGPUDisassembler::OPW128, Imm | 512));
}

static DecodeStatus decodeOperand_AReg_256(MCInst &Inst, unsigned Imm,
                                           uint64_t Addr,
                                           const MCDisassembler *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeSrcOp(AMDGPUDisassembler::OPW256, Imm | 512));
}

static DecodeStatus decodeOperand_AReg_512(MCInst &Inst, unsigned Imm,
                                           uint64_t Addr,
                                           const MCDisassembler *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeSrcOp(AMDGPUDisassembler::OPW512, Imm | 512));
}

static DecodeStatus decodeOperand_AReg_1024(MCInst &Inst, unsigned Imm,
                                            uint64_t Addr,
                                            const MCDisassembler *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeSrcOp(AMDGPUDisassembler::OPW1024, Imm | 512));
}

static DecodeStatus decodeOperand_VReg_64(MCInst &Inst, unsigned Imm,
                                          uint64_t Addr,
                                          const MCDisassembler *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeSrcOp(AMDGPUDisassembler::OPW64, Imm));
}

static DecodeStatus decodeOperand_VReg_128(MCInst &Inst, unsigned Imm,
                                           uint64_t Addr,
                                           const MCDisassembler *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeSrcOp(AMDGPUDisassembler::OPW128, Imm));
}

static DecodeStatus decodeOperand_VReg_256(MCInst &Inst, unsigned Imm,
                                           uint64_t Addr,
                                           const MCDisassembler *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeSrcOp(AMDGPUDisassembler::OPW256, Imm));
}

static DecodeStatus decodeOperand_VReg_512(MCInst &Inst, unsigned Imm,
                                           uint64_t Addr,
                                           const MCDisassembler *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeSrcOp(AMDGPUDisassembler::OPW512, Imm));
}

static DecodeStatus decodeOperand_VReg_1024(MCInst &Inst, unsigned Imm,
                                            uint64_t Addr,
                                            const MCDisassembler *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeSrcOp(AMDGPUDisassembler::OPW1024, Imm));
}

static DecodeStatus decodeOperand_f32kimm(MCInst &Inst, unsigned Imm,
                                          uint64_t Addr,
                                          const MCDisassembler *Decoder) {
  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);
  return addOperand(Inst, DAsm->decodeMandatoryLiteralConstant(Imm));
}

static DecodeStatus decodeOperand_f16kimm(MCInst &Inst, unsigned Imm,
                                          uint64_t Addr,
                                          const MCDisassembler *Decoder) {
  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);
  return addOperand(Inst, DAsm->decodeMandatoryLiteralConstant(Imm));
}

static DecodeStatus
decodeOperand_VS_16_Deferred(MCInst &Inst, unsigned Imm, uint64_t Addr,
                             const MCDisassembler *Decoder) {
  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);
  return addOperand(
      Inst, DAsm->decodeSrcOp(llvm::AMDGPUDisassembler::OPW16, Imm, true));
}

static DecodeStatus
decodeOperand_VS_32_Deferred(MCInst &Inst, unsigned Imm, uint64_t Addr,
                             const MCDisassembler *Decoder) {
  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);
  return addOperand(
      Inst, DAsm->decodeSrcOp(llvm::AMDGPUDisassembler::OPW32, Imm, true));
}

static DecodeStatus decodeOperandVOPDDstY(MCInst &Inst, unsigned Val,
                                          uint64_t Addr, const void *Decoder) {
  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);
  return addOperand(Inst, DAsm->decodeVOPDDstYOp(Inst, Val));
}

static bool IsAGPROperand(const MCInst &Inst, int OpIdx,
                          const MCRegisterInfo *MRI) {
  if (OpIdx < 0)
    return false;

  const MCOperand &Op = Inst.getOperand(OpIdx);
  if (!Op.isReg())
    return false;

  unsigned Sub = MRI->getSubReg(Op.getReg(), AMDGPU::sub0);
  auto Reg = Sub ? Sub : Op.getReg();
  return Reg >= AMDGPU::AGPR0 && Reg <= AMDGPU::AGPR255;
}

static DecodeStatus decodeOperand_AVLdSt_Any(MCInst &Inst, unsigned Imm,
                                             AMDGPUDisassembler::OpWidthTy Opw,
                                             const MCDisassembler *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  if (!DAsm->isGFX90A()) {
    Imm &= 511;
  } else {
    // If atomic has both vdata and vdst their register classes are tied.
    // The bit is decoded along with the vdst, first operand. We need to
    // change register class to AGPR if vdst was AGPR.
    // If a DS instruction has both data0 and data1 their register classes
    // are also tied.
    unsigned Opc = Inst.getOpcode();
    uint64_t TSFlags = DAsm->getMCII()->get(Opc).TSFlags;
    uint16_t DataNameIdx = (TSFlags & SIInstrFlags::DS) ? AMDGPU::OpName::data0
                                                        : AMDGPU::OpName::vdata;
    const MCRegisterInfo *MRI = DAsm->getContext().getRegisterInfo();
    int DataIdx = AMDGPU::getNamedOperandIdx(Opc, DataNameIdx);
    if ((int)Inst.getNumOperands() == DataIdx) {
      int DstIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vdst);
      if (IsAGPROperand(Inst, DstIdx, MRI))
        Imm |= 512;
    }

    if (TSFlags & SIInstrFlags::DS) {
      int Data2Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::data1);
      if ((int)Inst.getNumOperands() == Data2Idx &&
          IsAGPROperand(Inst, DataIdx, MRI))
        Imm |= 512;
    }
  }
  return addOperand(Inst, DAsm->decodeSrcOp(Opw, Imm | 256));
}

static DecodeStatus
DecodeAVLdSt_32RegisterClass(MCInst &Inst, unsigned Imm, uint64_t Addr,
                             const MCDisassembler *Decoder) {
  return decodeOperand_AVLdSt_Any(Inst, Imm,
                                  AMDGPUDisassembler::OPW32, Decoder);
}

static DecodeStatus
DecodeAVLdSt_64RegisterClass(MCInst &Inst, unsigned Imm, uint64_t Addr,
                             const MCDisassembler *Decoder) {
  return decodeOperand_AVLdSt_Any(Inst, Imm,
                                  AMDGPUDisassembler::OPW64, Decoder);
}

static DecodeStatus
DecodeAVLdSt_96RegisterClass(MCInst &Inst, unsigned Imm, uint64_t Addr,
                             const MCDisassembler *Decoder) {
  return decodeOperand_AVLdSt_Any(Inst, Imm,
                                  AMDGPUDisassembler::OPW96, Decoder);
}

static DecodeStatus
DecodeAVLdSt_128RegisterClass(MCInst &Inst, unsigned Imm, uint64_t Addr,
                              const MCDisassembler *Decoder) {
  return decodeOperand_AVLdSt_Any(Inst, Imm,
                                  AMDGPUDisassembler::OPW128, Decoder);
}

static DecodeStatus decodeOperand_SReg_32(MCInst &Inst, unsigned Imm,
                                          uint64_t Addr,
                                          const MCDisassembler *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeOperand_SReg_32(Imm));
}

#define DECODE_SDWA(DecName) \
DECODE_OPERAND(decodeSDWA##DecName, decodeSDWA##DecName)

DECODE_SDWA(Src32)
DECODE_SDWA(Src16)
DECODE_SDWA(VopcDst)

#include "AMDGPUGenDisassemblerTables.inc"

//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

template <typename T> static inline T eatBytes(ArrayRef<uint8_t>& Bytes) {
  assert(Bytes.size() >= sizeof(T));
  const auto Res = support::endian::read<T, support::endianness::little>(Bytes.data());
  Bytes = Bytes.slice(sizeof(T));
  return Res;
}

static inline DecoderUInt128 eat12Bytes(ArrayRef<uint8_t> &Bytes) {
  assert(Bytes.size() >= 12);
  uint64_t Lo = support::endian::read<uint64_t, support::endianness::little>(
      Bytes.data());
  Bytes = Bytes.slice(8);
  uint64_t Hi = support::endian::read<uint32_t, support::endianness::little>(
      Bytes.data());
  Bytes = Bytes.slice(4);
  return DecoderUInt128(Lo, Hi);
}

// The disassembler is greedy, so we need to check FI operand value to
// not parse a dpp if the correct literal is not set. For dpp16 the
// autogenerated decoder checks the dpp literal
static bool isValidDPP8(const MCInst &MI) {
  using namespace llvm::AMDGPU::DPP;
  int FiIdx = AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::fi);
  assert(FiIdx != -1);
  if ((unsigned)FiIdx >= MI.getNumOperands())
    return false;
  unsigned Fi = MI.getOperand(FiIdx).getImm();
  return Fi == DPP8_FI_0 || Fi == DPP8_FI_1;
}

DecodeStatus AMDGPUDisassembler::getInstruction(MCInst &MI, uint64_t &Size,
                                                ArrayRef<uint8_t> Bytes_,
                                                uint64_t Address,
                                                raw_ostream &CS) const {
  CommentStream = &CS;
  bool IsSDWA = false;

  unsigned MaxInstBytesNum = std::min((size_t)TargetMaxInstBytes, Bytes_.size());
  Bytes = Bytes_.slice(0, MaxInstBytesNum);

  DecodeStatus Res = MCDisassembler::Fail;
  do {
    // ToDo: better to switch encoding length using some bit predicate
    // but it is unknown yet, so try all we can

    // Try to decode DPP and SDWA first to solve conflict with VOP1 and VOP2
    // encodings
    if (isGFX11Plus() && Bytes.size() >= 12 ) {
      DecoderUInt128 DecW = eat12Bytes(Bytes);
      Res = tryDecodeInst(DecoderTableDPP8GFX1196, MI, DecW,
                                          Address);
      if (Res && convertDPP8Inst(MI) == MCDisassembler::Success)
        break;
      MI = MCInst(); // clear
      Res = tryDecodeInst(DecoderTableDPPGFX1196, MI, DecW,
                                          Address);
      if (Res) {
        if (MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::VOP3P)
          convertVOP3PDPPInst(MI);
        else if (AMDGPU::isVOPC64DPP(MI.getOpcode()))
          convertVOPCDPPInst(MI); // Special VOP3 case
        else {
          assert(MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::VOP3);
          convertVOP3DPPInst(MI); // Regular VOP3 case
        }
        break;
      }
      Res = tryDecodeInst(DecoderTableGFX1196, MI, DecW, Address);
      if (Res)
        break;
    }
    // Reinitialize Bytes
    Bytes = Bytes_.slice(0, MaxInstBytesNum);

    if (Bytes.size() >= 8) {
      const uint64_t QW = eatBytes<uint64_t>(Bytes);

      if (STI.getFeatureBits()[AMDGPU::FeatureGFX10_BEncoding]) {
        Res = tryDecodeInst(DecoderTableGFX10_B64, MI, QW, Address);
        if (Res) {
          if (AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::dpp8)
              == -1)
            break;
          if (convertDPP8Inst(MI) == MCDisassembler::Success)
            break;
          MI = MCInst(); // clear
        }
      }

      Res = tryDecodeInst(DecoderTableDPP864, MI, QW, Address);
      if (Res && convertDPP8Inst(MI) == MCDisassembler::Success)
        break;
      MI = MCInst(); // clear

      Res = tryDecodeInst(DecoderTableDPP8GFX1164, MI, QW, Address);
      if (Res && convertDPP8Inst(MI) == MCDisassembler::Success)
        break;
      MI = MCInst(); // clear

      Res = tryDecodeInst(DecoderTableDPP64, MI, QW, Address);
      if (Res) break;

      Res = tryDecodeInst(DecoderTableDPPGFX1164, MI, QW, Address);
      if (Res) {
        if (MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::VOPC)
          convertVOPCDPPInst(MI);
        break;
      }

      Res = tryDecodeInst(DecoderTableSDWA64, MI, QW, Address);
      if (Res) { IsSDWA = true;  break; }

      Res = tryDecodeInst(DecoderTableSDWA964, MI, QW, Address);
      if (Res) { IsSDWA = true;  break; }

      Res = tryDecodeInst(DecoderTableSDWA1064, MI, QW, Address);
      if (Res) { IsSDWA = true;  break; }

      if (STI.getFeatureBits()[AMDGPU::FeatureUnpackedD16VMem]) {
        Res = tryDecodeInst(DecoderTableGFX80_UNPACKED64, MI, QW, Address);
        if (Res)
          break;
      }

      // Some GFX9 subtargets repurposed the v_mad_mix_f32, v_mad_mixlo_f16 and
      // v_mad_mixhi_f16 for FMA variants. Try to decode using this special
      // table first so we print the correct name.
      if (STI.getFeatureBits()[AMDGPU::FeatureFmaMixInsts]) {
        Res = tryDecodeInst(DecoderTableGFX9_DL64, MI, QW, Address);
        if (Res)
          break;
      }
    }

    // Reinitialize Bytes as DPP64 could have eaten too much
    Bytes = Bytes_.slice(0, MaxInstBytesNum);

    // Try decode 32-bit instruction
    if (Bytes.size() < 4) break;
    const uint32_t DW = eatBytes<uint32_t>(Bytes);
    Res = tryDecodeInst(DecoderTableGFX832, MI, DW, Address);
    if (Res) break;

    Res = tryDecodeInst(DecoderTableAMDGPU32, MI, DW, Address);
    if (Res) break;

    Res = tryDecodeInst(DecoderTableGFX932, MI, DW, Address);
    if (Res) break;

    if (STI.getFeatureBits()[AMDGPU::FeatureGFX90AInsts]) {
      Res = tryDecodeInst(DecoderTableGFX90A32, MI, DW, Address);
      if (Res)
        break;
    }

    if (STI.getFeatureBits()[AMDGPU::FeatureGFX10_BEncoding]) {
      Res = tryDecodeInst(DecoderTableGFX10_B32, MI, DW, Address);
      if (Res) break;
    }

    Res = tryDecodeInst(DecoderTableGFX1032, MI, DW, Address);
    if (Res) break;

    Res = tryDecodeInst(DecoderTableGFX1132, MI, DW, Address);
    if (Res) break;

    if (Bytes.size() < 4) break;
    const uint64_t QW = ((uint64_t)eatBytes<uint32_t>(Bytes) << 32) | DW;

    if (STI.getFeatureBits()[AMDGPU::FeatureGFX940Insts]) {
      Res = tryDecodeInst(DecoderTableGFX94064, MI, QW, Address);
      if (Res)
        break;
    }

    if (STI.getFeatureBits()[AMDGPU::FeatureGFX90AInsts]) {
      Res = tryDecodeInst(DecoderTableGFX90A64, MI, QW, Address);
      if (Res)
        break;
    }

    Res = tryDecodeInst(DecoderTableGFX864, MI, QW, Address);
    if (Res) break;

    Res = tryDecodeInst(DecoderTableAMDGPU64, MI, QW, Address);
    if (Res) break;

    Res = tryDecodeInst(DecoderTableGFX964, MI, QW, Address);
    if (Res) break;

    Res = tryDecodeInst(DecoderTableGFX1064, MI, QW, Address);
    if (Res) break;

    Res = tryDecodeInst(DecoderTableGFX1164, MI, QW, Address);
    if (Res)
      break;

    Res = tryDecodeInst(DecoderTableWMMAGFX1164, MI, QW, Address);
  } while (false);

  if (Res && (MI.getOpcode() == AMDGPU::V_MAC_F32_e64_vi ||
              MI.getOpcode() == AMDGPU::V_MAC_F32_e64_gfx6_gfx7 ||
              MI.getOpcode() == AMDGPU::V_MAC_F32_e64_gfx10 ||
              MI.getOpcode() == AMDGPU::V_MAC_LEGACY_F32_e64_gfx6_gfx7 ||
              MI.getOpcode() == AMDGPU::V_MAC_LEGACY_F32_e64_gfx10 ||
              MI.getOpcode() == AMDGPU::V_MAC_F16_e64_vi ||
              MI.getOpcode() == AMDGPU::V_FMAC_F64_e64_gfx90a ||
              MI.getOpcode() == AMDGPU::V_FMAC_F32_e64_vi ||
              MI.getOpcode() == AMDGPU::V_FMAC_F32_e64_gfx10 ||
              MI.getOpcode() == AMDGPU::V_FMAC_F32_e64_gfx11 ||
              MI.getOpcode() == AMDGPU::V_FMAC_LEGACY_F32_e64_gfx10 ||
              MI.getOpcode() == AMDGPU::V_FMAC_DX9_ZERO_F32_e64_gfx11 ||
              MI.getOpcode() == AMDGPU::V_FMAC_F16_e64_gfx10 ||
              MI.getOpcode() == AMDGPU::V_FMAC_F16_t16_e64_gfx11)) {
    // Insert dummy unused src2_modifiers.
    insertNamedMCOperand(MI, MCOperand::createImm(0),
                         AMDGPU::OpName::src2_modifiers);
  }

  if (Res && (MCII->get(MI.getOpcode()).TSFlags &
          (SIInstrFlags::MUBUF | SIInstrFlags::FLAT | SIInstrFlags::SMRD))) {
    int CPolPos = AMDGPU::getNamedOperandIdx(MI.getOpcode(),
                                             AMDGPU::OpName::cpol);
    if (CPolPos != -1) {
      unsigned CPol =
          (MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::IsAtomicRet) ?
              AMDGPU::CPol::GLC : 0;
      if (MI.getNumOperands() <= (unsigned)CPolPos) {
        insertNamedMCOperand(MI, MCOperand::createImm(CPol),
                             AMDGPU::OpName::cpol);
      } else if (CPol) {
        MI.getOperand(CPolPos).setImm(MI.getOperand(CPolPos).getImm() | CPol);
      }
    }
  }

  if (Res && (MCII->get(MI.getOpcode()).TSFlags &
              (SIInstrFlags::MTBUF | SIInstrFlags::MUBUF)) &&
             (STI.getFeatureBits()[AMDGPU::FeatureGFX90AInsts])) {
    // GFX90A lost TFE, its place is occupied by ACC.
    int TFEOpIdx =
        AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::tfe);
    if (TFEOpIdx != -1) {
      auto TFEIter = MI.begin();
      std::advance(TFEIter, TFEOpIdx);
      MI.insert(TFEIter, MCOperand::createImm(0));
    }
  }

  if (Res && (MCII->get(MI.getOpcode()).TSFlags &
              (SIInstrFlags::MTBUF | SIInstrFlags::MUBUF))) {
    int SWZOpIdx =
        AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::swz);
    if (SWZOpIdx != -1) {
      auto SWZIter = MI.begin();
      std::advance(SWZIter, SWZOpIdx);
      MI.insert(SWZIter, MCOperand::createImm(0));
    }
  }

  if (Res && (MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::MIMG)) {
    int VAddr0Idx =
        AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::vaddr0);
    int RsrcIdx =
        AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::srsrc);
    unsigned NSAArgs = RsrcIdx - VAddr0Idx - 1;
    if (VAddr0Idx >= 0 && NSAArgs > 0) {
      unsigned NSAWords = (NSAArgs + 3) / 4;
      if (Bytes.size() < 4 * NSAWords) {
        Res = MCDisassembler::Fail;
      } else {
        for (unsigned i = 0; i < NSAArgs; ++i) {
          const unsigned VAddrIdx = VAddr0Idx + 1 + i;
          auto VAddrRCID = MCII->get(MI.getOpcode()).OpInfo[VAddrIdx].RegClass;
          MI.insert(MI.begin() + VAddrIdx,
                    createRegOperand(VAddrRCID, Bytes[i]));
        }
        Bytes = Bytes.slice(4 * NSAWords);
      }
    }

    if (Res)
      Res = convertMIMGInst(MI);
  }

  if (Res && (MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::EXP))
    Res = convertEXPInst(MI);

  if (Res && (MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::VINTERP))
    Res = convertVINTERPInst(MI);

  if (Res && IsSDWA)
    Res = convertSDWAInst(MI);

  int VDstIn_Idx = AMDGPU::getNamedOperandIdx(MI.getOpcode(),
                                              AMDGPU::OpName::vdst_in);
  if (VDstIn_Idx != -1) {
    int Tied = MCII->get(MI.getOpcode()).getOperandConstraint(VDstIn_Idx,
                           MCOI::OperandConstraint::TIED_TO);
    if (Tied != -1 && (MI.getNumOperands() <= (unsigned)VDstIn_Idx ||
         !MI.getOperand(VDstIn_Idx).isReg() ||
         MI.getOperand(VDstIn_Idx).getReg() != MI.getOperand(Tied).getReg())) {
      if (MI.getNumOperands() > (unsigned)VDstIn_Idx)
        MI.erase(&MI.getOperand(VDstIn_Idx));
      insertNamedMCOperand(MI,
        MCOperand::createReg(MI.getOperand(Tied).getReg()),
        AMDGPU::OpName::vdst_in);
    }
  }

  int ImmLitIdx =
      AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::imm);
  bool isVOP2 = MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::VOP2;
  if (Res && ImmLitIdx != -1 && (isVOP2 || AMDGPU::isVOPD(MI.getOpcode())))
    Res = convertFMAanyK(MI, ImmLitIdx);

  // if the opcode was not recognized we'll assume a Size of 4 bytes
  // (unless there are fewer bytes left)
  Size = Res ? (MaxInstBytesNum - Bytes.size())
             : std::min((size_t)4, Bytes_.size());
  return Res;
}

DecodeStatus AMDGPUDisassembler::convertEXPInst(MCInst &MI) const {
  if (STI.getFeatureBits()[AMDGPU::FeatureGFX11]) {
    // The MCInst still has these fields even though they are no longer encoded
    // in the GFX11 instruction.
    insertNamedMCOperand(MI, MCOperand::createImm(0), AMDGPU::OpName::vm);
    insertNamedMCOperand(MI, MCOperand::createImm(0), AMDGPU::OpName::compr);
  }
  return MCDisassembler::Success;
}

DecodeStatus AMDGPUDisassembler::convertVINTERPInst(MCInst &MI) const {
  if (MI.getOpcode() == AMDGPU::V_INTERP_P10_F16_F32_inreg_gfx11 ||
      MI.getOpcode() == AMDGPU::V_INTERP_P10_RTZ_F16_F32_inreg_gfx11 ||
      MI.getOpcode() == AMDGPU::V_INTERP_P2_F16_F32_inreg_gfx11 ||
      MI.getOpcode() == AMDGPU::V_INTERP_P2_RTZ_F16_F32_inreg_gfx11) {
    // The MCInst has this field that is not directly encoded in the
    // instruction.
    insertNamedMCOperand(MI, MCOperand::createImm(0), AMDGPU::OpName::op_sel);
  }
  return MCDisassembler::Success;
}

DecodeStatus AMDGPUDisassembler::convertSDWAInst(MCInst &MI) const {
  if (STI.getFeatureBits()[AMDGPU::FeatureGFX9] ||
      STI.getFeatureBits()[AMDGPU::FeatureGFX10]) {
    if (AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::sdst) != -1)
      // VOPC - insert clamp
      insertNamedMCOperand(MI, MCOperand::createImm(0), AMDGPU::OpName::clamp);
  } else if (STI.getFeatureBits()[AMDGPU::FeatureVolcanicIslands]) {
    int SDst = AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::sdst);
    if (SDst != -1) {
      // VOPC - insert VCC register as sdst
      insertNamedMCOperand(MI, createRegOperand(AMDGPU::VCC),
                           AMDGPU::OpName::sdst);
    } else {
      // VOP1/2 - insert omod if present in instruction
      insertNamedMCOperand(MI, MCOperand::createImm(0), AMDGPU::OpName::omod);
    }
  }
  return MCDisassembler::Success;
}

struct VOPModifiers {
  unsigned OpSel = 0;
  unsigned OpSelHi = 0;
  unsigned NegLo = 0;
  unsigned NegHi = 0;
};

// Reconstruct values of VOP3/VOP3P operands such as op_sel.
// Note that these values do not affect disassembler output,
// so this is only necessary for consistency with src_modifiers.
static VOPModifiers collectVOPModifiers(const MCInst &MI,
                                        bool IsVOP3P = false) {
  VOPModifiers Modifiers;
  unsigned Opc = MI.getOpcode();
  const int ModOps[] = {AMDGPU::OpName::src0_modifiers,
                        AMDGPU::OpName::src1_modifiers,
                        AMDGPU::OpName::src2_modifiers};
  for (int J = 0; J < 3; ++J) {
    int OpIdx = AMDGPU::getNamedOperandIdx(Opc, ModOps[J]);
    if (OpIdx == -1)
      continue;

    unsigned Val = MI.getOperand(OpIdx).getImm();

    Modifiers.OpSel |= !!(Val & SISrcMods::OP_SEL_0) << J;
    if (IsVOP3P) {
      Modifiers.OpSelHi |= !!(Val & SISrcMods::OP_SEL_1) << J;
      Modifiers.NegLo |= !!(Val & SISrcMods::NEG) << J;
      Modifiers.NegHi |= !!(Val & SISrcMods::NEG_HI) << J;
    } else if (J == 0) {
      Modifiers.OpSel |= !!(Val & SISrcMods::DST_OP_SEL) << 3;
    }
  }

  return Modifiers;
}

// MAC opcodes have special old and src2 operands.
// src2 is tied to dst, while old is not tied (but assumed to be).
bool AMDGPUDisassembler::isMacDPP(MCInst &MI) const {
  constexpr int DST_IDX = 0;
  auto Opcode = MI.getOpcode();
  const auto &Desc = MCII->get(Opcode);
  auto OldIdx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::old);

  if (OldIdx != -1 && Desc.getOperandConstraint(
                          OldIdx, MCOI::OperandConstraint::TIED_TO) == -1) {
    assert(AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src2) != -1);
    assert(Desc.getOperandConstraint(
               AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src2),
               MCOI::OperandConstraint::TIED_TO) == DST_IDX);
    (void)DST_IDX;
    return true;
  }

  return false;
}

// Create dummy old operand and insert dummy unused src2_modifiers
void AMDGPUDisassembler::convertMacDPPInst(MCInst &MI) const {
  assert(MI.getNumOperands() + 1 < MCII->get(MI.getOpcode()).getNumOperands());
  insertNamedMCOperand(MI, MCOperand::createReg(0), AMDGPU::OpName::old);
  insertNamedMCOperand(MI, MCOperand::createImm(0),
                       AMDGPU::OpName::src2_modifiers);
}

// We must check FI == literal to reject not genuine dpp8 insts, and we must
// first add optional MI operands to check FI
DecodeStatus AMDGPUDisassembler::convertDPP8Inst(MCInst &MI) const {
  unsigned Opc = MI.getOpcode();
  if (MCII->get(Opc).TSFlags & SIInstrFlags::VOP3P) {
    convertVOP3PDPPInst(MI);
  } else if ((MCII->get(Opc).TSFlags & SIInstrFlags::VOPC) ||
             AMDGPU::isVOPC64DPP(Opc)) {
    convertVOPCDPPInst(MI);
  } else {
    if (isMacDPP(MI))
      convertMacDPPInst(MI);

    unsigned DescNumOps = MCII->get(Opc).getNumOperands();
    if (MI.getNumOperands() < DescNumOps &&
        AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::op_sel) != -1) {
      auto Mods = collectVOPModifiers(MI);
      insertNamedMCOperand(MI, MCOperand::createImm(Mods.OpSel),
                           AMDGPU::OpName::op_sel);
    } else {
      // Insert dummy unused src modifiers.
      if (MI.getNumOperands() < DescNumOps &&
          AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0_modifiers) != -1)
        insertNamedMCOperand(MI, MCOperand::createImm(0),
                             AMDGPU::OpName::src0_modifiers);

      if (MI.getNumOperands() < DescNumOps &&
          AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src1_modifiers) != -1)
        insertNamedMCOperand(MI, MCOperand::createImm(0),
                             AMDGPU::OpName::src1_modifiers);
    }
  }
  return isValidDPP8(MI) ? MCDisassembler::Success : MCDisassembler::SoftFail;
}

DecodeStatus AMDGPUDisassembler::convertVOP3DPPInst(MCInst &MI) const {
  if (isMacDPP(MI))
    convertMacDPPInst(MI);

  unsigned Opc = MI.getOpcode();
  unsigned DescNumOps = MCII->get(Opc).getNumOperands();
  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::op_sel) != -1) {
    auto Mods = collectVOPModifiers(MI);
    insertNamedMCOperand(MI, MCOperand::createImm(Mods.OpSel),
                         AMDGPU::OpName::op_sel);
  }
  return MCDisassembler::Success;
}

// Note that before gfx10, the MIMG encoding provided no information about
// VADDR size. Consequently, decoded instructions always show address as if it
// has 1 dword, which could be not really so.
DecodeStatus AMDGPUDisassembler::convertMIMGInst(MCInst &MI) const {

  int VDstIdx = AMDGPU::getNamedOperandIdx(MI.getOpcode(),
                                           AMDGPU::OpName::vdst);

  int VDataIdx = AMDGPU::getNamedOperandIdx(MI.getOpcode(),
                                            AMDGPU::OpName::vdata);
  int VAddr0Idx =
      AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::vaddr0);
  int DMaskIdx = AMDGPU::getNamedOperandIdx(MI.getOpcode(),
                                            AMDGPU::OpName::dmask);

  int TFEIdx   = AMDGPU::getNamedOperandIdx(MI.getOpcode(),
                                            AMDGPU::OpName::tfe);
  int D16Idx   = AMDGPU::getNamedOperandIdx(MI.getOpcode(),
                                            AMDGPU::OpName::d16);

  const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(MI.getOpcode());
  const AMDGPU::MIMGBaseOpcodeInfo *BaseOpcode =
      AMDGPU::getMIMGBaseOpcodeInfo(Info->BaseOpcode);

  assert(VDataIdx != -1);
  if (BaseOpcode->BVH) {
    // Add A16 operand for intersect_ray instructions
    if (AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::a16) > -1) {
      addOperand(MI, MCOperand::createImm(1));
    }
    return MCDisassembler::Success;
  }

  bool IsAtomic = (VDstIdx != -1);
  bool IsGather4 = MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::Gather4;
  bool IsNSA = false;
  unsigned AddrSize = Info->VAddrDwords;

  if (isGFX10Plus()) {
    unsigned DimIdx =
        AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::dim);
    int A16Idx =
        AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::a16);
    const AMDGPU::MIMGDimInfo *Dim =
        AMDGPU::getMIMGDimInfoByEncoding(MI.getOperand(DimIdx).getImm());
    const bool IsA16 = (A16Idx != -1 && MI.getOperand(A16Idx).getImm());

    AddrSize =
        AMDGPU::getAddrSizeMIMGOp(BaseOpcode, Dim, IsA16, AMDGPU::hasG16(STI));

    IsNSA = Info->MIMGEncoding == AMDGPU::MIMGEncGfx10NSA ||
            Info->MIMGEncoding == AMDGPU::MIMGEncGfx11NSA;
    if (!IsNSA) {
      if (AddrSize > 8)
        AddrSize = 16;
    } else {
      if (AddrSize > Info->VAddrDwords) {
        // The NSA encoding does not contain enough operands for the combination
        // of base opcode / dimension. Should this be an error?
        return MCDisassembler::Success;
      }
    }
  }

  unsigned DMask = MI.getOperand(DMaskIdx).getImm() & 0xf;
  unsigned DstSize = IsGather4 ? 4 : std::max(countPopulation(DMask), 1u);

  bool D16 = D16Idx >= 0 && MI.getOperand(D16Idx).getImm();
  if (D16 && AMDGPU::hasPackedD16(STI)) {
    DstSize = (DstSize + 1) / 2;
  }

  if (TFEIdx != -1 && MI.getOperand(TFEIdx).getImm())
    DstSize += 1;

  if (DstSize == Info->VDataDwords && AddrSize == Info->VAddrDwords)
    return MCDisassembler::Success;

  int NewOpcode =
      AMDGPU::getMIMGOpcode(Info->BaseOpcode, Info->MIMGEncoding, DstSize, AddrSize);
  if (NewOpcode == -1)
    return MCDisassembler::Success;

  // Widen the register to the correct number of enabled channels.
  unsigned NewVdata = AMDGPU::NoRegister;
  if (DstSize != Info->VDataDwords) {
    auto DataRCID = MCII->get(NewOpcode).OpInfo[VDataIdx].RegClass;

    // Get first subregister of VData
    unsigned Vdata0 = MI.getOperand(VDataIdx).getReg();
    unsigned VdataSub0 = MRI.getSubReg(Vdata0, AMDGPU::sub0);
    Vdata0 = (VdataSub0 != 0)? VdataSub0 : Vdata0;

    NewVdata = MRI.getMatchingSuperReg(Vdata0, AMDGPU::sub0,
                                       &MRI.getRegClass(DataRCID));
    if (NewVdata == AMDGPU::NoRegister) {
      // It's possible to encode this such that the low register + enabled
      // components exceeds the register count.
      return MCDisassembler::Success;
    }
  }

  // If not using NSA on GFX10+, widen address register to correct size.
  unsigned NewVAddr0 = AMDGPU::NoRegister;
  if (isGFX10Plus() && !IsNSA && AddrSize != Info->VAddrDwords) {
    unsigned VAddr0 = MI.getOperand(VAddr0Idx).getReg();
    unsigned VAddrSub0 = MRI.getSubReg(VAddr0, AMDGPU::sub0);
    VAddr0 = (VAddrSub0 != 0) ? VAddrSub0 : VAddr0;

    auto AddrRCID = MCII->get(NewOpcode).OpInfo[VAddr0Idx].RegClass;
    NewVAddr0 = MRI.getMatchingSuperReg(VAddr0, AMDGPU::sub0,
                                        &MRI.getRegClass(AddrRCID));
    if (NewVAddr0 == AMDGPU::NoRegister)
      return MCDisassembler::Success;
  }

  MI.setOpcode(NewOpcode);

  if (NewVdata != AMDGPU::NoRegister) {
    MI.getOperand(VDataIdx) = MCOperand::createReg(NewVdata);

    if (IsAtomic) {
      // Atomic operations have an additional operand (a copy of data)
      MI.getOperand(VDstIdx) = MCOperand::createReg(NewVdata);
    }
  }

  if (NewVAddr0 != AMDGPU::NoRegister) {
    MI.getOperand(VAddr0Idx) = MCOperand::createReg(NewVAddr0);
  } else if (IsNSA) {
    assert(AddrSize <= Info->VAddrDwords);
    MI.erase(MI.begin() + VAddr0Idx + AddrSize,
             MI.begin() + VAddr0Idx + Info->VAddrDwords);
  }

  return MCDisassembler::Success;
}

// Opsel and neg bits are used in src_modifiers and standalone operands. Autogen
// decoder only adds to src_modifiers, so manually add the bits to the other
// operands.
DecodeStatus AMDGPUDisassembler::convertVOP3PDPPInst(MCInst &MI) const {
  unsigned Opc = MI.getOpcode();
  unsigned DescNumOps = MCII->get(Opc).getNumOperands();
  auto Mods = collectVOPModifiers(MI, true);

  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vdst_in) != -1)
    insertNamedMCOperand(MI, MCOperand::createImm(0), AMDGPU::OpName::vdst_in);

  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::op_sel) != -1)
    insertNamedMCOperand(MI, MCOperand::createImm(Mods.OpSel),
                         AMDGPU::OpName::op_sel);
  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::op_sel_hi) != -1)
    insertNamedMCOperand(MI, MCOperand::createImm(Mods.OpSelHi),
                         AMDGPU::OpName::op_sel_hi);
  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::neg_lo) != -1)
    insertNamedMCOperand(MI, MCOperand::createImm(Mods.NegLo),
                         AMDGPU::OpName::neg_lo);
  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::neg_hi) != -1)
    insertNamedMCOperand(MI, MCOperand::createImm(Mods.NegHi),
                         AMDGPU::OpName::neg_hi);

  return MCDisassembler::Success;
}

// Create dummy old operand and insert optional operands
DecodeStatus AMDGPUDisassembler::convertVOPCDPPInst(MCInst &MI) const {
  unsigned Opc = MI.getOpcode();
  unsigned DescNumOps = MCII->get(Opc).getNumOperands();

  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::old) != -1)
    insertNamedMCOperand(MI, MCOperand::createReg(0), AMDGPU::OpName::old);

  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0_modifiers) != -1)
    insertNamedMCOperand(MI, MCOperand::createImm(0),
                         AMDGPU::OpName::src0_modifiers);

  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src1_modifiers) != -1)
    insertNamedMCOperand(MI, MCOperand::createImm(0),
                         AMDGPU::OpName::src1_modifiers);
  return MCDisassembler::Success;
}

DecodeStatus AMDGPUDisassembler::convertFMAanyK(MCInst &MI,
                                                int ImmLitIdx) const {
  assert(HasLiteral && "Should have decoded a literal");
  const MCInstrDesc &Desc = MCII->get(MI.getOpcode());
  unsigned DescNumOps = Desc.getNumOperands();
  insertNamedMCOperand(MI, MCOperand::createImm(Literal),
                       AMDGPU::OpName::immDeferred);
  assert(DescNumOps == MI.getNumOperands());
  for (unsigned I = 0; I < DescNumOps; ++I) {
    auto &Op = MI.getOperand(I);
    auto OpType = Desc.OpInfo[I].OperandType;
    bool IsDeferredOp = (OpType == AMDGPU::OPERAND_REG_IMM_FP32_DEFERRED ||
                         OpType == AMDGPU::OPERAND_REG_IMM_FP16_DEFERRED);
    if (Op.isImm() && Op.getImm() == AMDGPU::EncValues::LITERAL_CONST &&
        IsDeferredOp)
      Op.setImm(Literal);
  }
  return MCDisassembler::Success;
}

const char* AMDGPUDisassembler::getRegClassName(unsigned RegClassID) const {
  return getContext().getRegisterInfo()->
    getRegClassName(&AMDGPUMCRegisterClasses[RegClassID]);
}

inline
MCOperand AMDGPUDisassembler::errOperand(unsigned V,
                                         const Twine& ErrMsg) const {
  *CommentStream << "Error: " + ErrMsg;

  // ToDo: add support for error operands to MCInst.h
  // return MCOperand::createError(V);
  return MCOperand();
}

inline
MCOperand AMDGPUDisassembler::createRegOperand(unsigned int RegId) const {
  return MCOperand::createReg(AMDGPU::getMCReg(RegId, STI));
}

inline
MCOperand AMDGPUDisassembler::createRegOperand(unsigned RegClassID,
                                               unsigned Val) const {
  const auto& RegCl = AMDGPUMCRegisterClasses[RegClassID];
  if (Val >= RegCl.getNumRegs())
    return errOperand(Val, Twine(getRegClassName(RegClassID)) +
                           ": unknown register " + Twine(Val));
  return createRegOperand(RegCl.getRegister(Val));
}

inline
MCOperand AMDGPUDisassembler::createSRegOperand(unsigned SRegClassID,
                                                unsigned Val) const {
  // ToDo: SI/CI have 104 SGPRs, VI - 102
  // Valery: here we accepting as much as we can, let assembler sort it out
  int shift = 0;
  switch (SRegClassID) {
  case AMDGPU::SGPR_32RegClassID:
  case AMDGPU::TTMP_32RegClassID:
    break;
  case AMDGPU::SGPR_64RegClassID:
  case AMDGPU::TTMP_64RegClassID:
    shift = 1;
    break;
  case AMDGPU::SGPR_128RegClassID:
  case AMDGPU::TTMP_128RegClassID:
  // ToDo: unclear if s[100:104] is available on VI. Can we use VCC as SGPR in
  // this bundle?
  case AMDGPU::SGPR_256RegClassID:
  case AMDGPU::TTMP_256RegClassID:
    // ToDo: unclear if s[96:104] is available on VI. Can we use VCC as SGPR in
  // this bundle?
  case AMDGPU::SGPR_512RegClassID:
  case AMDGPU::TTMP_512RegClassID:
    shift = 2;
    break;
  // ToDo: unclear if s[88:104] is available on VI. Can we use VCC as SGPR in
  // this bundle?
  default:
    llvm_unreachable("unhandled register class");
  }

  if (Val % (1 << shift)) {
    *CommentStream << "Warning: " << getRegClassName(SRegClassID)
                   << ": scalar reg isn't aligned " << Val;
  }

  return createRegOperand(SRegClassID, Val >> shift);
}

MCOperand AMDGPUDisassembler::decodeOperand_VS_32(unsigned Val) const {
  return decodeSrcOp(OPW32, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VS_64(unsigned Val) const {
  return decodeSrcOp(OPW64, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VS_128(unsigned Val) const {
  return decodeSrcOp(OPW128, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VSrc16(unsigned Val) const {
  return decodeSrcOp(OPW16, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VSrcV216(unsigned Val) const {
  return decodeSrcOp(OPWV216, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VSrcV232(unsigned Val) const {
  return decodeSrcOp(OPWV232, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VGPR_32_Lo128(unsigned Val) const {
  return createRegOperand(AMDGPU::VGPR_32_Lo128RegClassID, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VGPR_32(unsigned Val) const {
  // Some instructions have operand restrictions beyond what the encoding
  // allows. Some ordinarily VSrc_32 operands are VGPR_32, so clear the extra
  // high bit.
  Val &= 255;

  return createRegOperand(AMDGPU::VGPR_32RegClassID, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VRegOrLds_32(unsigned Val) const {
  return decodeSrcOp(OPW32, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_AGPR_32(unsigned Val) const {
  return createRegOperand(AMDGPU::AGPR_32RegClassID, Val & 255);
}

MCOperand AMDGPUDisassembler::decodeOperand_AReg_64(unsigned Val) const {
  return createRegOperand(AMDGPU::AReg_64RegClassID, Val & 255);
}

MCOperand AMDGPUDisassembler::decodeOperand_AReg_128(unsigned Val) const {
  return createRegOperand(AMDGPU::AReg_128RegClassID, Val & 255);
}

MCOperand AMDGPUDisassembler::decodeOperand_AReg_256(unsigned Val) const {
  return createRegOperand(AMDGPU::AReg_256RegClassID, Val & 255);
}

MCOperand AMDGPUDisassembler::decodeOperand_AReg_512(unsigned Val) const {
  return createRegOperand(AMDGPU::AReg_512RegClassID, Val & 255);
}

MCOperand AMDGPUDisassembler::decodeOperand_AReg_1024(unsigned Val) const {
  return createRegOperand(AMDGPU::AReg_1024RegClassID, Val & 255);
}

MCOperand AMDGPUDisassembler::decodeOperand_AV_32(unsigned Val) const {
  return decodeSrcOp(OPW32, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_AV_64(unsigned Val) const {
  return decodeSrcOp(OPW64, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_AV_128(unsigned Val) const {
  return decodeSrcOp(OPW128, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_AVDst_128(unsigned Val) const {
  using namespace AMDGPU::EncValues;
  assert((Val & IS_VGPR) == 0); // Val{8} is not encoded but assumed to be 1.
  return decodeSrcOp(OPW128, Val | IS_VGPR);
}

MCOperand AMDGPUDisassembler::decodeOperand_AVDst_512(unsigned Val) const {
  using namespace AMDGPU::EncValues;
  assert((Val & IS_VGPR) == 0); // Val{8} is not encoded but assumed to be 1.
  return decodeSrcOp(OPW512, Val | IS_VGPR);
}

MCOperand AMDGPUDisassembler::decodeOperand_VReg_64(unsigned Val) const {
  return createRegOperand(AMDGPU::VReg_64RegClassID, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VReg_96(unsigned Val) const {
  return createRegOperand(AMDGPU::VReg_96RegClassID, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VReg_128(unsigned Val) const {
  return createRegOperand(AMDGPU::VReg_128RegClassID, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VReg_256(unsigned Val) const {
  return createRegOperand(AMDGPU::VReg_256RegClassID, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VReg_512(unsigned Val) const {
  return createRegOperand(AMDGPU::VReg_512RegClassID, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_VReg_1024(unsigned Val) const {
  return createRegOperand(AMDGPU::VReg_1024RegClassID, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_SReg_32(unsigned Val) const {
  // table-gen generated disassembler doesn't care about operand types
  // leaving only registry class so SSrc_32 operand turns into SReg_32
  // and therefore we accept immediates and literals here as well
  return decodeSrcOp(OPW32, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_SReg_32_XM0_XEXEC(
  unsigned Val) const {
  // SReg_32_XM0 is SReg_32 without M0 or EXEC_LO/EXEC_HI
  return decodeOperand_SReg_32(Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_SReg_32_XEXEC_HI(
  unsigned Val) const {
  // SReg_32_XM0 is SReg_32 without EXEC_HI
  return decodeOperand_SReg_32(Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_SRegOrLds_32(unsigned Val) const {
  // table-gen generated disassembler doesn't care about operand types
  // leaving only registry class so SSrc_32 operand turns into SReg_32
  // and therefore we accept immediates and literals here as well
  return decodeSrcOp(OPW32, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_SReg_64(unsigned Val) const {
  return decodeSrcOp(OPW64, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_SReg_64_XEXEC(unsigned Val) const {
  return decodeSrcOp(OPW64, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_SReg_128(unsigned Val) const {
  return decodeSrcOp(OPW128, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_SReg_256(unsigned Val) const {
  return decodeDstOp(OPW256, Val);
}

MCOperand AMDGPUDisassembler::decodeOperand_SReg_512(unsigned Val) const {
  return decodeDstOp(OPW512, Val);
}

// Decode Literals for insts which always have a literal in the encoding
MCOperand
AMDGPUDisassembler::decodeMandatoryLiteralConstant(unsigned Val) const {
  if (HasLiteral) {
    assert(
        AMDGPU::hasVOPD(STI) &&
        "Should only decode multiple kimm with VOPD, check VSrc operand types");
    if (Literal != Val)
      return errOperand(Val, "More than one unique literal is illegal");
  }
  HasLiteral = true;
  Literal = Val;
  return MCOperand::createImm(Literal);
}

MCOperand AMDGPUDisassembler::decodeLiteralConstant() const {
  // For now all literal constants are supposed to be unsigned integer
  // ToDo: deal with signed/unsigned 64-bit integer constants
  // ToDo: deal with float/double constants
  if (!HasLiteral) {
    if (Bytes.size() < 4) {
      return errOperand(0, "cannot read literal, inst bytes left " +
                        Twine(Bytes.size()));
    }
    HasLiteral = true;
    Literal = eatBytes<uint32_t>(Bytes);
  }
  return MCOperand::createImm(Literal);
}

MCOperand AMDGPUDisassembler::decodeIntImmed(unsigned Imm) {
  using namespace AMDGPU::EncValues;

  assert(Imm >= INLINE_INTEGER_C_MIN && Imm <= INLINE_INTEGER_C_MAX);
  return MCOperand::createImm((Imm <= INLINE_INTEGER_C_POSITIVE_MAX) ?
    (static_cast<int64_t>(Imm) - INLINE_INTEGER_C_MIN) :
    (INLINE_INTEGER_C_POSITIVE_MAX - static_cast<int64_t>(Imm)));
      // Cast prevents negative overflow.
}

static int64_t getInlineImmVal32(unsigned Imm) {
  switch (Imm) {
  case 240:
    return FloatToBits(0.5f);
  case 241:
    return FloatToBits(-0.5f);
  case 242:
    return FloatToBits(1.0f);
  case 243:
    return FloatToBits(-1.0f);
  case 244:
    return FloatToBits(2.0f);
  case 245:
    return FloatToBits(-2.0f);
  case 246:
    return FloatToBits(4.0f);
  case 247:
    return FloatToBits(-4.0f);
  case 248: // 1 / (2 * PI)
    return 0x3e22f983;
  default:
    llvm_unreachable("invalid fp inline imm");
  }
}

static int64_t getInlineImmVal64(unsigned Imm) {
  switch (Imm) {
  case 240:
    return DoubleToBits(0.5);
  case 241:
    return DoubleToBits(-0.5);
  case 242:
    return DoubleToBits(1.0);
  case 243:
    return DoubleToBits(-1.0);
  case 244:
    return DoubleToBits(2.0);
  case 245:
    return DoubleToBits(-2.0);
  case 246:
    return DoubleToBits(4.0);
  case 247:
    return DoubleToBits(-4.0);
  case 248: // 1 / (2 * PI)
    return 0x3fc45f306dc9c882;
  default:
    llvm_unreachable("invalid fp inline imm");
  }
}

static int64_t getInlineImmVal16(unsigned Imm) {
  switch (Imm) {
  case 240:
    return 0x3800;
  case 241:
    return 0xB800;
  case 242:
    return 0x3C00;
  case 243:
    return 0xBC00;
  case 244:
    return 0x4000;
  case 245:
    return 0xC000;
  case 246:
    return 0x4400;
  case 247:
    return 0xC400;
  case 248: // 1 / (2 * PI)
    return 0x3118;
  default:
    llvm_unreachable("invalid fp inline imm");
  }
}

MCOperand AMDGPUDisassembler::decodeFPImmed(OpWidthTy Width, unsigned Imm) {
  assert(Imm >= AMDGPU::EncValues::INLINE_FLOATING_C_MIN
      && Imm <= AMDGPU::EncValues::INLINE_FLOATING_C_MAX);

  // ToDo: case 248: 1/(2*PI) - is allowed only on VI
  switch (Width) {
  case OPW32:
  case OPW128: // splat constants
  case OPW512:
  case OPW1024:
  case OPWV232:
    return MCOperand::createImm(getInlineImmVal32(Imm));
  case OPW64:
  case OPW256:
    return MCOperand::createImm(getInlineImmVal64(Imm));
  case OPW16:
  case OPWV216:
    return MCOperand::createImm(getInlineImmVal16(Imm));
  default:
    llvm_unreachable("implement me");
  }
}

unsigned AMDGPUDisassembler::getVgprClassId(const OpWidthTy Width) const {
  using namespace AMDGPU;

  assert(OPW_FIRST_ <= Width && Width < OPW_LAST_);
  switch (Width) {
  default: // fall
  case OPW32:
  case OPW16:
  case OPWV216:
    return VGPR_32RegClassID;
  case OPW64:
  case OPWV232: return VReg_64RegClassID;
  case OPW96: return VReg_96RegClassID;
  case OPW128: return VReg_128RegClassID;
  case OPW160: return VReg_160RegClassID;
  case OPW256: return VReg_256RegClassID;
  case OPW512: return VReg_512RegClassID;
  case OPW1024: return VReg_1024RegClassID;
  }
}

unsigned AMDGPUDisassembler::getAgprClassId(const OpWidthTy Width) const {
  using namespace AMDGPU;

  assert(OPW_FIRST_ <= Width && Width < OPW_LAST_);
  switch (Width) {
  default: // fall
  case OPW32:
  case OPW16:
  case OPWV216:
    return AGPR_32RegClassID;
  case OPW64:
  case OPWV232: return AReg_64RegClassID;
  case OPW96: return AReg_96RegClassID;
  case OPW128: return AReg_128RegClassID;
  case OPW160: return AReg_160RegClassID;
  case OPW256: return AReg_256RegClassID;
  case OPW512: return AReg_512RegClassID;
  case OPW1024: return AReg_1024RegClassID;
  }
}


unsigned AMDGPUDisassembler::getSgprClassId(const OpWidthTy Width) const {
  using namespace AMDGPU;

  assert(OPW_FIRST_ <= Width && Width < OPW_LAST_);
  switch (Width) {
  default: // fall
  case OPW32:
  case OPW16:
  case OPWV216:
    return SGPR_32RegClassID;
  case OPW64:
  case OPWV232: return SGPR_64RegClassID;
  case OPW96: return SGPR_96RegClassID;
  case OPW128: return SGPR_128RegClassID;
  case OPW160: return SGPR_160RegClassID;
  case OPW256: return SGPR_256RegClassID;
  case OPW512: return SGPR_512RegClassID;
  }
}

unsigned AMDGPUDisassembler::getTtmpClassId(const OpWidthTy Width) const {
  using namespace AMDGPU;

  assert(OPW_FIRST_ <= Width && Width < OPW_LAST_);
  switch (Width) {
  default: // fall
  case OPW32:
  case OPW16:
  case OPWV216:
    return TTMP_32RegClassID;
  case OPW64:
  case OPWV232: return TTMP_64RegClassID;
  case OPW128: return TTMP_128RegClassID;
  case OPW256: return TTMP_256RegClassID;
  case OPW512: return TTMP_512RegClassID;
  }
}

int AMDGPUDisassembler::getTTmpIdx(unsigned Val) const {
  using namespace AMDGPU::EncValues;

  unsigned TTmpMin = isGFX9Plus() ? TTMP_GFX9PLUS_MIN : TTMP_VI_MIN;
  unsigned TTmpMax = isGFX9Plus() ? TTMP_GFX9PLUS_MAX : TTMP_VI_MAX;

  return (TTmpMin <= Val && Val <= TTmpMax)? Val - TTmpMin : -1;
}

MCOperand AMDGPUDisassembler::decodeSrcOp(const OpWidthTy Width, unsigned Val,
                                          bool MandatoryLiteral) const {
  using namespace AMDGPU::EncValues;

  assert(Val < 1024); // enum10

  bool IsAGPR = Val & 512;
  Val &= 511;

  if (VGPR_MIN <= Val && Val <= VGPR_MAX) {
    return createRegOperand(IsAGPR ? getAgprClassId(Width)
                                   : getVgprClassId(Width), Val - VGPR_MIN);
  }
  if (Val <= SGPR_MAX) {
    // "SGPR_MIN <= Val" is always true and causes compilation warning.
    static_assert(SGPR_MIN == 0);
    return createSRegOperand(getSgprClassId(Width), Val - SGPR_MIN);
  }

  int TTmpIdx = getTTmpIdx(Val);
  if (TTmpIdx >= 0) {
    return createSRegOperand(getTtmpClassId(Width), TTmpIdx);
  }

  if (INLINE_INTEGER_C_MIN <= Val && Val <= INLINE_INTEGER_C_MAX)
    return decodeIntImmed(Val);

  if (INLINE_FLOATING_C_MIN <= Val && Val <= INLINE_FLOATING_C_MAX)
    return decodeFPImmed(Width, Val);

  if (Val == LITERAL_CONST) {
    if (MandatoryLiteral)
      // Keep a sentinel value for deferred setting
      return MCOperand::createImm(LITERAL_CONST);
    else
      return decodeLiteralConstant();
  }

  switch (Width) {
  case OPW32:
  case OPW16:
  case OPWV216:
    return decodeSpecialReg32(Val);
  case OPW64:
  case OPWV232:
    return decodeSpecialReg64(Val);
  default:
    llvm_unreachable("unexpected immediate type");
  }
}

MCOperand AMDGPUDisassembler::decodeDstOp(const OpWidthTy Width, unsigned Val) const {
  using namespace AMDGPU::EncValues;

  assert(Val < 128);
  assert(Width == OPW256 || Width == OPW512);

  if (Val <= SGPR_MAX) {
    // "SGPR_MIN <= Val" is always true and causes compilation warning.
    static_assert(SGPR_MIN == 0);
    return createSRegOperand(getSgprClassId(Width), Val - SGPR_MIN);
  }

  int TTmpIdx = getTTmpIdx(Val);
  if (TTmpIdx >= 0) {
    return createSRegOperand(getTtmpClassId(Width), TTmpIdx);
  }

  llvm_unreachable("unknown dst register");
}

// Bit 0 of DstY isn't stored in the instruction, because it's always the
// opposite of bit 0 of DstX.
MCOperand AMDGPUDisassembler::decodeVOPDDstYOp(MCInst &Inst,
                                               unsigned Val) const {
  int VDstXInd =
      AMDGPU::getNamedOperandIdx(Inst.getOpcode(), AMDGPU::OpName::vdstX);
  assert(VDstXInd != -1);
  assert(Inst.getOperand(VDstXInd).isReg());
  unsigned XDstReg = MRI.getEncodingValue(Inst.getOperand(VDstXInd).getReg());
  Val |= ~XDstReg & 1;
  auto Width = llvm::AMDGPUDisassembler::OPW32;
  return createRegOperand(getVgprClassId(Width), Val);
}

MCOperand AMDGPUDisassembler::decodeSpecialReg32(unsigned Val) const {
  using namespace AMDGPU;

  switch (Val) {
  case 102: return createRegOperand(FLAT_SCR_LO);
  case 103: return createRegOperand(FLAT_SCR_HI);
  case 104: return createRegOperand(XNACK_MASK_LO);
  case 105: return createRegOperand(XNACK_MASK_HI);
  case 106: return createRegOperand(VCC_LO);
  case 107: return createRegOperand(VCC_HI);
  case 108: return createRegOperand(TBA_LO);
  case 109: return createRegOperand(TBA_HI);
  case 110: return createRegOperand(TMA_LO);
  case 111: return createRegOperand(TMA_HI);
  case 124:
    return isGFX11Plus() ? createRegOperand(SGPR_NULL) : createRegOperand(M0);
  case 125:
    return isGFX11Plus() ? createRegOperand(M0) : createRegOperand(SGPR_NULL);
  case 126: return createRegOperand(EXEC_LO);
  case 127: return createRegOperand(EXEC_HI);
  case 235: return createRegOperand(SRC_SHARED_BASE);
  case 236: return createRegOperand(SRC_SHARED_LIMIT);
  case 237: return createRegOperand(SRC_PRIVATE_BASE);
  case 238: return createRegOperand(SRC_PRIVATE_LIMIT);
  case 239: return createRegOperand(SRC_POPS_EXITING_WAVE_ID);
  case 251: return createRegOperand(SRC_VCCZ);
  case 252: return createRegOperand(SRC_EXECZ);
  case 253: return createRegOperand(SRC_SCC);
  case 254: return createRegOperand(LDS_DIRECT);
  default: break;
  }
  return errOperand(Val, "unknown operand encoding " + Twine(Val));
}

MCOperand AMDGPUDisassembler::decodeSpecialReg64(unsigned Val) const {
  using namespace AMDGPU;

  switch (Val) {
  case 102: return createRegOperand(FLAT_SCR);
  case 104: return createRegOperand(XNACK_MASK);
  case 106: return createRegOperand(VCC);
  case 108: return createRegOperand(TBA);
  case 110: return createRegOperand(TMA);
  case 124:
    if (isGFX11Plus())
      return createRegOperand(SGPR_NULL);
    break;
  case 125:
    if (!isGFX11Plus())
      return createRegOperand(SGPR_NULL);
    break;
  case 126: return createRegOperand(EXEC);
  case 235: return createRegOperand(SRC_SHARED_BASE);
  case 236: return createRegOperand(SRC_SHARED_LIMIT);
  case 237: return createRegOperand(SRC_PRIVATE_BASE);
  case 238: return createRegOperand(SRC_PRIVATE_LIMIT);
  case 239: return createRegOperand(SRC_POPS_EXITING_WAVE_ID);
  case 251: return createRegOperand(SRC_VCCZ);
  case 252: return createRegOperand(SRC_EXECZ);
  case 253: return createRegOperand(SRC_SCC);
  default: break;
  }
  return errOperand(Val, "unknown operand encoding " + Twine(Val));
}

MCOperand AMDGPUDisassembler::decodeSDWASrc(const OpWidthTy Width,
                                            const unsigned Val) const {
  using namespace AMDGPU::SDWA;
  using namespace AMDGPU::EncValues;

  if (STI.getFeatureBits()[AMDGPU::FeatureGFX9] ||
      STI.getFeatureBits()[AMDGPU::FeatureGFX10]) {
    // XXX: cast to int is needed to avoid stupid warning:
    // compare with unsigned is always true
    if (int(SDWA9EncValues::SRC_VGPR_MIN) <= int(Val) &&
        Val <= SDWA9EncValues::SRC_VGPR_MAX) {
      return createRegOperand(getVgprClassId(Width),
                              Val - SDWA9EncValues::SRC_VGPR_MIN);
    }
    if (SDWA9EncValues::SRC_SGPR_MIN <= Val &&
        Val <= (isGFX10Plus() ? SDWA9EncValues::SRC_SGPR_MAX_GFX10
                              : SDWA9EncValues::SRC_SGPR_MAX_SI)) {
      return createSRegOperand(getSgprClassId(Width),
                               Val - SDWA9EncValues::SRC_SGPR_MIN);
    }
    if (SDWA9EncValues::SRC_TTMP_MIN <= Val &&
        Val <= SDWA9EncValues::SRC_TTMP_MAX) {
      return createSRegOperand(getTtmpClassId(Width),
                               Val - SDWA9EncValues::SRC_TTMP_MIN);
    }

    const unsigned SVal = Val - SDWA9EncValues::SRC_SGPR_MIN;

    if (INLINE_INTEGER_C_MIN <= SVal && SVal <= INLINE_INTEGER_C_MAX)
      return decodeIntImmed(SVal);

    if (INLINE_FLOATING_C_MIN <= SVal && SVal <= INLINE_FLOATING_C_MAX)
      return decodeFPImmed(Width, SVal);

    return decodeSpecialReg32(SVal);
  } else if (STI.getFeatureBits()[AMDGPU::FeatureVolcanicIslands]) {
    return createRegOperand(getVgprClassId(Width), Val);
  }
  llvm_unreachable("unsupported target");
}

MCOperand AMDGPUDisassembler::decodeSDWASrc16(unsigned Val) const {
  return decodeSDWASrc(OPW16, Val);
}

MCOperand AMDGPUDisassembler::decodeSDWASrc32(unsigned Val) const {
  return decodeSDWASrc(OPW32, Val);
}

MCOperand AMDGPUDisassembler::decodeSDWAVopcDst(unsigned Val) const {
  using namespace AMDGPU::SDWA;

  assert((STI.getFeatureBits()[AMDGPU::FeatureGFX9] ||
          STI.getFeatureBits()[AMDGPU::FeatureGFX10]) &&
         "SDWAVopcDst should be present only on GFX9+");

  bool IsWave64 = STI.getFeatureBits()[AMDGPU::FeatureWavefrontSize64];

  if (Val & SDWA9EncValues::VOPC_DST_VCC_MASK) {
    Val &= SDWA9EncValues::VOPC_DST_SGPR_MASK;

    int TTmpIdx = getTTmpIdx(Val);
    if (TTmpIdx >= 0) {
      auto TTmpClsId = getTtmpClassId(IsWave64 ? OPW64 : OPW32);
      return createSRegOperand(TTmpClsId, TTmpIdx);
    } else if (Val > SGPR_MAX) {
      return IsWave64 ? decodeSpecialReg64(Val)
                      : decodeSpecialReg32(Val);
    } else {
      return createSRegOperand(getSgprClassId(IsWave64 ? OPW64 : OPW32), Val);
    }
  } else {
    return createRegOperand(IsWave64 ? AMDGPU::VCC : AMDGPU::VCC_LO);
  }
}

MCOperand AMDGPUDisassembler::decodeBoolReg(unsigned Val) const {
  return STI.getFeatureBits()[AMDGPU::FeatureWavefrontSize64] ?
    decodeOperand_SReg_64(Val) : decodeOperand_SReg_32(Val);
}

bool AMDGPUDisassembler::isVI() const {
  return STI.getFeatureBits()[AMDGPU::FeatureVolcanicIslands];
}

bool AMDGPUDisassembler::isGFX9() const { return AMDGPU::isGFX9(STI); }

bool AMDGPUDisassembler::isGFX90A() const {
  return STI.getFeatureBits()[AMDGPU::FeatureGFX90AInsts];
}

bool AMDGPUDisassembler::isGFX9Plus() const { return AMDGPU::isGFX9Plus(STI); }

bool AMDGPUDisassembler::isGFX10() const { return AMDGPU::isGFX10(STI); }

bool AMDGPUDisassembler::isGFX10Plus() const {
  return AMDGPU::isGFX10Plus(STI);
}

bool AMDGPUDisassembler::isGFX11() const {
  return STI.getFeatureBits()[AMDGPU::FeatureGFX11];
}

bool AMDGPUDisassembler::isGFX11Plus() const {
  return AMDGPU::isGFX11Plus(STI);
}


bool AMDGPUDisassembler::hasArchitectedFlatScratch() const {
  return STI.getFeatureBits()[AMDGPU::FeatureArchitectedFlatScratch];
}

//===----------------------------------------------------------------------===//
// AMDGPU specific symbol handling
//===----------------------------------------------------------------------===//
#define PRINT_DIRECTIVE(DIRECTIVE, MASK)                                       \
  do {                                                                         \
    KdStream << Indent << DIRECTIVE " "                                        \
             << ((FourByteBuffer & MASK) >> (MASK##_SHIFT)) << '\n';           \
  } while (0)

// NOLINTNEXTLINE(readability-identifier-naming)
MCDisassembler::DecodeStatus AMDGPUDisassembler::decodeCOMPUTE_PGM_RSRC1(
    uint32_t FourByteBuffer, raw_string_ostream &KdStream) const {
  using namespace amdhsa;
  StringRef Indent = "\t";

  // We cannot accurately backward compute #VGPRs used from
  // GRANULATED_WORKITEM_VGPR_COUNT. But we are concerned with getting the same
  // value of GRANULATED_WORKITEM_VGPR_COUNT in the reassembled binary. So we
  // simply calculate the inverse of what the assembler does.

  uint32_t GranulatedWorkitemVGPRCount =
      (FourByteBuffer & COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT) >>
      COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT;

  uint32_t NextFreeVGPR = (GranulatedWorkitemVGPRCount + 1) *
                          AMDGPU::IsaInfo::getVGPREncodingGranule(&STI);

  KdStream << Indent << ".amdhsa_next_free_vgpr " << NextFreeVGPR << '\n';

  // We cannot backward compute values used to calculate
  // GRANULATED_WAVEFRONT_SGPR_COUNT. Hence the original values for following
  // directives can't be computed:
  // .amdhsa_reserve_vcc
  // .amdhsa_reserve_flat_scratch
  // .amdhsa_reserve_xnack_mask
  // They take their respective default values if not specified in the assembly.
  //
  // GRANULATED_WAVEFRONT_SGPR_COUNT
  //    = f(NEXT_FREE_SGPR + VCC + FLAT_SCRATCH + XNACK_MASK)
  //
  // We compute the inverse as though all directives apart from NEXT_FREE_SGPR
  // are set to 0. So while disassembling we consider that:
  //
  // GRANULATED_WAVEFRONT_SGPR_COUNT
  //    = f(NEXT_FREE_SGPR + 0 + 0 + 0)
  //
  // The disassembler cannot recover the original values of those 3 directives.

  uint32_t GranulatedWavefrontSGPRCount =
      (FourByteBuffer & COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT) >>
      COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT;

  if (isGFX10Plus() && GranulatedWavefrontSGPRCount)
    return MCDisassembler::Fail;

  uint32_t NextFreeSGPR = (GranulatedWavefrontSGPRCount + 1) *
                          AMDGPU::IsaInfo::getSGPREncodingGranule(&STI);

  KdStream << Indent << ".amdhsa_reserve_vcc " << 0 << '\n';
  if (!hasArchitectedFlatScratch())
    KdStream << Indent << ".amdhsa_reserve_flat_scratch " << 0 << '\n';
  KdStream << Indent << ".amdhsa_reserve_xnack_mask " << 0 << '\n';
  KdStream << Indent << ".amdhsa_next_free_sgpr " << NextFreeSGPR << "\n";

  if (FourByteBuffer & COMPUTE_PGM_RSRC1_PRIORITY)
    return MCDisassembler::Fail;

  PRINT_DIRECTIVE(".amdhsa_float_round_mode_32",
                  COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_32);
  PRINT_DIRECTIVE(".amdhsa_float_round_mode_16_64",
                  COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_16_64);
  PRINT_DIRECTIVE(".amdhsa_float_denorm_mode_32",
                  COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_32);
  PRINT_DIRECTIVE(".amdhsa_float_denorm_mode_16_64",
                  COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64);

  if (FourByteBuffer & COMPUTE_PGM_RSRC1_PRIV)
    return MCDisassembler::Fail;

  PRINT_DIRECTIVE(".amdhsa_dx10_clamp", COMPUTE_PGM_RSRC1_ENABLE_DX10_CLAMP);

  if (FourByteBuffer & COMPUTE_PGM_RSRC1_DEBUG_MODE)
    return MCDisassembler::Fail;

  PRINT_DIRECTIVE(".amdhsa_ieee_mode", COMPUTE_PGM_RSRC1_ENABLE_IEEE_MODE);

  if (FourByteBuffer & COMPUTE_PGM_RSRC1_BULKY)
    return MCDisassembler::Fail;

  if (FourByteBuffer & COMPUTE_PGM_RSRC1_CDBG_USER)
    return MCDisassembler::Fail;

  PRINT_DIRECTIVE(".amdhsa_fp16_overflow", COMPUTE_PGM_RSRC1_FP16_OVFL);

  if (FourByteBuffer & COMPUTE_PGM_RSRC1_RESERVED0)
    return MCDisassembler::Fail;

  if (isGFX10Plus()) {
    PRINT_DIRECTIVE(".amdhsa_workgroup_processor_mode",
                    COMPUTE_PGM_RSRC1_WGP_MODE);
    PRINT_DIRECTIVE(".amdhsa_memory_ordered", COMPUTE_PGM_RSRC1_MEM_ORDERED);
    PRINT_DIRECTIVE(".amdhsa_forward_progress", COMPUTE_PGM_RSRC1_FWD_PROGRESS);
  }
  return MCDisassembler::Success;
}

// NOLINTNEXTLINE(readability-identifier-naming)
MCDisassembler::DecodeStatus AMDGPUDisassembler::decodeCOMPUTE_PGM_RSRC2(
    uint32_t FourByteBuffer, raw_string_ostream &KdStream) const {
  using namespace amdhsa;
  StringRef Indent = "\t";
  if (hasArchitectedFlatScratch())
    PRINT_DIRECTIVE(".amdhsa_enable_private_segment",
                    COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT);
  else
    PRINT_DIRECTIVE(".amdhsa_system_sgpr_private_segment_wavefront_offset",
                    COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT);
  PRINT_DIRECTIVE(".amdhsa_system_sgpr_workgroup_id_x",
                  COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X);
  PRINT_DIRECTIVE(".amdhsa_system_sgpr_workgroup_id_y",
                  COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y);
  PRINT_DIRECTIVE(".amdhsa_system_sgpr_workgroup_id_z",
                  COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z);
  PRINT_DIRECTIVE(".amdhsa_system_sgpr_workgroup_info",
                  COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO);
  PRINT_DIRECTIVE(".amdhsa_system_vgpr_workitem_id",
                  COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID);

  if (FourByteBuffer & COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_ADDRESS_WATCH)
    return MCDisassembler::Fail;

  if (FourByteBuffer & COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_MEMORY)
    return MCDisassembler::Fail;

  if (FourByteBuffer & COMPUTE_PGM_RSRC2_GRANULATED_LDS_SIZE)
    return MCDisassembler::Fail;

  PRINT_DIRECTIVE(
      ".amdhsa_exception_fp_ieee_invalid_op",
      COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION);
  PRINT_DIRECTIVE(".amdhsa_exception_fp_denorm_src",
                  COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE);
  PRINT_DIRECTIVE(
      ".amdhsa_exception_fp_ieee_div_zero",
      COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO);
  PRINT_DIRECTIVE(".amdhsa_exception_fp_ieee_overflow",
                  COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW);
  PRINT_DIRECTIVE(".amdhsa_exception_fp_ieee_underflow",
                  COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW);
  PRINT_DIRECTIVE(".amdhsa_exception_fp_ieee_inexact",
                  COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT);
  PRINT_DIRECTIVE(".amdhsa_exception_int_div_zero",
                  COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_INT_DIVIDE_BY_ZERO);

  if (FourByteBuffer & COMPUTE_PGM_RSRC2_RESERVED0)
    return MCDisassembler::Fail;

  return MCDisassembler::Success;
}

#undef PRINT_DIRECTIVE

MCDisassembler::DecodeStatus
AMDGPUDisassembler::decodeKernelDescriptorDirective(
    DataExtractor::Cursor &Cursor, ArrayRef<uint8_t> Bytes,
    raw_string_ostream &KdStream) const {
#define PRINT_DIRECTIVE(DIRECTIVE, MASK)                                       \
  do {                                                                         \
    KdStream << Indent << DIRECTIVE " "                                        \
             << ((TwoByteBuffer & MASK) >> (MASK##_SHIFT)) << '\n';            \
  } while (0)

  uint16_t TwoByteBuffer = 0;
  uint32_t FourByteBuffer = 0;

  StringRef ReservedBytes;
  StringRef Indent = "\t";

  assert(Bytes.size() == 64);
  DataExtractor DE(Bytes, /*IsLittleEndian=*/true, /*AddressSize=*/8);

  switch (Cursor.tell()) {
  case amdhsa::GROUP_SEGMENT_FIXED_SIZE_OFFSET:
    FourByteBuffer = DE.getU32(Cursor);
    KdStream << Indent << ".amdhsa_group_segment_fixed_size " << FourByteBuffer
             << '\n';
    return MCDisassembler::Success;

  case amdhsa::PRIVATE_SEGMENT_FIXED_SIZE_OFFSET:
    FourByteBuffer = DE.getU32(Cursor);
    KdStream << Indent << ".amdhsa_private_segment_fixed_size "
             << FourByteBuffer << '\n';
    return MCDisassembler::Success;

  case amdhsa::KERNARG_SIZE_OFFSET:
    FourByteBuffer = DE.getU32(Cursor);
    KdStream << Indent << ".amdhsa_kernarg_size "
             << FourByteBuffer << '\n';
    return MCDisassembler::Success;

  case amdhsa::RESERVED0_OFFSET:
    // 4 reserved bytes, must be 0.
    ReservedBytes = DE.getBytes(Cursor, 4);
    for (int I = 0; I < 4; ++I) {
      if (ReservedBytes[I] != 0) {
        return MCDisassembler::Fail;
      }
    }
    return MCDisassembler::Success;

  case amdhsa::KERNEL_CODE_ENTRY_BYTE_OFFSET_OFFSET:
    // KERNEL_CODE_ENTRY_BYTE_OFFSET
    // So far no directive controls this for Code Object V3, so simply skip for
    // disassembly.
    DE.skip(Cursor, 8);
    return MCDisassembler::Success;

  case amdhsa::RESERVED1_OFFSET:
    // 20 reserved bytes, must be 0.
    ReservedBytes = DE.getBytes(Cursor, 20);
    for (int I = 0; I < 20; ++I) {
      if (ReservedBytes[I] != 0) {
        return MCDisassembler::Fail;
      }
    }
    return MCDisassembler::Success;

  case amdhsa::COMPUTE_PGM_RSRC3_OFFSET:
    // COMPUTE_PGM_RSRC3
    //  - Only set for GFX10, GFX6-9 have this to be 0.
    //  - Currently no directives directly control this.
    FourByteBuffer = DE.getU32(Cursor);
    if (!isGFX10Plus() && FourByteBuffer) {
      return MCDisassembler::Fail;
    }
    return MCDisassembler::Success;

  case amdhsa::COMPUTE_PGM_RSRC1_OFFSET:
    FourByteBuffer = DE.getU32(Cursor);
    if (decodeCOMPUTE_PGM_RSRC1(FourByteBuffer, KdStream) ==
        MCDisassembler::Fail) {
      return MCDisassembler::Fail;
    }
    return MCDisassembler::Success;

  case amdhsa::COMPUTE_PGM_RSRC2_OFFSET:
    FourByteBuffer = DE.getU32(Cursor);
    if (decodeCOMPUTE_PGM_RSRC2(FourByteBuffer, KdStream) ==
        MCDisassembler::Fail) {
      return MCDisassembler::Fail;
    }
    return MCDisassembler::Success;

  case amdhsa::KERNEL_CODE_PROPERTIES_OFFSET:
    using namespace amdhsa;
    TwoByteBuffer = DE.getU16(Cursor);

    if (!hasArchitectedFlatScratch())
      PRINT_DIRECTIVE(".amdhsa_user_sgpr_private_segment_buffer",
                      KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER);
    PRINT_DIRECTIVE(".amdhsa_user_sgpr_dispatch_ptr",
                    KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR);
    PRINT_DIRECTIVE(".amdhsa_user_sgpr_queue_ptr",
                    KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR);
    PRINT_DIRECTIVE(".amdhsa_user_sgpr_kernarg_segment_ptr",
                    KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR);
    PRINT_DIRECTIVE(".amdhsa_user_sgpr_dispatch_id",
                    KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID);
    if (!hasArchitectedFlatScratch())
      PRINT_DIRECTIVE(".amdhsa_user_sgpr_flat_scratch_init",
                      KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT);
    PRINT_DIRECTIVE(".amdhsa_user_sgpr_private_segment_size",
                    KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE);

    if (TwoByteBuffer & KERNEL_CODE_PROPERTY_RESERVED0)
      return MCDisassembler::Fail;

    // Reserved for GFX9
    if (isGFX9() &&
        (TwoByteBuffer & KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32)) {
      return MCDisassembler::Fail;
    } else if (isGFX10Plus()) {
      PRINT_DIRECTIVE(".amdhsa_wavefront_size32",
                      KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);
    }

    PRINT_DIRECTIVE(".amdhsa_uses_dynamic_stack",
                    KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK);

    if (TwoByteBuffer & KERNEL_CODE_PROPERTY_RESERVED1)
      return MCDisassembler::Fail;

    return MCDisassembler::Success;

  case amdhsa::RESERVED2_OFFSET:
    // 6 bytes from here are reserved, must be 0.
    ReservedBytes = DE.getBytes(Cursor, 6);
    for (int I = 0; I < 6; ++I) {
      if (ReservedBytes[I] != 0)
        return MCDisassembler::Fail;
    }
    return MCDisassembler::Success;

  default:
    llvm_unreachable("Unhandled index. Case statements cover everything.");
    return MCDisassembler::Fail;
  }
#undef PRINT_DIRECTIVE
}

MCDisassembler::DecodeStatus AMDGPUDisassembler::decodeKernelDescriptor(
    StringRef KdName, ArrayRef<uint8_t> Bytes, uint64_t KdAddress) const {
  // CP microcode requires the kernel descriptor to be 64 aligned.
  if (Bytes.size() != 64 || KdAddress % 64 != 0)
    return MCDisassembler::Fail;

  std::string Kd;
  raw_string_ostream KdStream(Kd);
  KdStream << ".amdhsa_kernel " << KdName << '\n';

  DataExtractor::Cursor C(0);
  while (C && C.tell() < Bytes.size()) {
    MCDisassembler::DecodeStatus Status =
        decodeKernelDescriptorDirective(C, Bytes, KdStream);

    cantFail(C.takeError());

    if (Status == MCDisassembler::Fail)
      return MCDisassembler::Fail;
  }
  KdStream << ".end_amdhsa_kernel\n";
  outs() << KdStream.str();
  return MCDisassembler::Success;
}

Optional<MCDisassembler::DecodeStatus>
AMDGPUDisassembler::onSymbolStart(SymbolInfoTy &Symbol, uint64_t &Size,
                                  ArrayRef<uint8_t> Bytes, uint64_t Address,
                                  raw_ostream &CStream) const {
  // Right now only kernel descriptor needs to be handled.
  // We ignore all other symbols for target specific handling.
  // TODO:
  // Fix the spurious symbol issue for AMDGPU kernels. Exists for both Code
  // Object V2 and V3 when symbols are marked protected.

  // amd_kernel_code_t for Code Object V2.
  if (Symbol.Type == ELF::STT_AMDGPU_HSA_KERNEL) {
    Size = 256;
    return MCDisassembler::Fail;
  }

  // Code Object V3 kernel descriptors.
  StringRef Name = Symbol.Name;
  if (Symbol.Type == ELF::STT_OBJECT && Name.endswith(StringRef(".kd"))) {
    Size = 64; // Size = 64 regardless of success or failure.
    return decodeKernelDescriptor(Name.drop_back(3), Bytes, Address);
  }
  return None;
}

//===----------------------------------------------------------------------===//
// AMDGPUSymbolizer
//===----------------------------------------------------------------------===//

// Try to find symbol name for specified label
bool AMDGPUSymbolizer::tryAddingSymbolicOperand(
    MCInst &Inst, raw_ostream & /*cStream*/, int64_t Value,
    uint64_t /*Address*/, bool IsBranch, uint64_t /*Offset*/,
    uint64_t /*OpSize*/, uint64_t /*InstSize*/) {

  if (!IsBranch) {
    return false;
  }

  auto *Symbols = static_cast<SectionSymbolsTy *>(DisInfo);
  if (!Symbols)
    return false;

  auto Result = llvm::find_if(*Symbols, [Value](const SymbolInfoTy &Val) {
    return Val.Addr == static_cast<uint64_t>(Value) &&
           Val.Type == ELF::STT_NOTYPE;
  });
  if (Result != Symbols->end()) {
    auto *Sym = Ctx.getOrCreateSymbol(Result->Name);
    const auto *Add = MCSymbolRefExpr::create(Sym, Ctx);
    Inst.addOperand(MCOperand::createExpr(Add));
    return true;
  }
  // Add to list of referenced addresses, so caller can synthesize a label.
  ReferencedAddresses.push_back(static_cast<uint64_t>(Value));
  return false;
}

void AMDGPUSymbolizer::tryAddingPcLoadReferenceComment(raw_ostream &cStream,
                                                       int64_t Value,
                                                       uint64_t Address) {
  llvm_unreachable("unimplemented");
}

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

static MCSymbolizer *createAMDGPUSymbolizer(const Triple &/*TT*/,
                              LLVMOpInfoCallback /*GetOpInfo*/,
                              LLVMSymbolLookupCallback /*SymbolLookUp*/,
                              void *DisInfo,
                              MCContext *Ctx,
                              std::unique_ptr<MCRelocationInfo> &&RelInfo) {
  return new AMDGPUSymbolizer(*Ctx, std::move(RelInfo), DisInfo);
}

static MCDisassembler *createAMDGPUDisassembler(const Target &T,
                                                const MCSubtargetInfo &STI,
                                                MCContext &Ctx) {
  return new AMDGPUDisassembler(STI, Ctx, T.createMCInstrInfo());
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeAMDGPUDisassembler() {
  TargetRegistry::RegisterMCDisassembler(getTheGCNTarget(),
                                         createAMDGPUDisassembler);
  TargetRegistry::RegisterMCSymbolizer(getTheGCNTarget(),
                                       createAMDGPUSymbolizer);
}
