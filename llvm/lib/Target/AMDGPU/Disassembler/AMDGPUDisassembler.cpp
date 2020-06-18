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
#include "AMDGPU.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "TargetInfo/AMDGPUTargetInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm-c/Disassembler.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixedLenDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <tuple>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-disassembler"

#define SGPR_MAX (isGFX10() ? AMDGPU::EncValues::SGPR_MAX_GFX10 \
                            : AMDGPU::EncValues::SGPR_MAX_SI)

using DecodeStatus = llvm::MCDisassembler::DecodeStatus;

AMDGPUDisassembler::AMDGPUDisassembler(const MCSubtargetInfo &STI,
                                       MCContext &Ctx,
                                       MCInstrInfo const *MCII) :
  MCDisassembler(STI, Ctx), MCII(MCII), MRI(*Ctx.getRegisterInfo()),
  TargetMaxInstBytes(Ctx.getAsmInfo()->getMaxInstLength(&STI)) {

  // ToDo: AMDGPUDisassembler supports only VI ISA.
  if (!STI.getFeatureBits()[AMDGPU::FeatureGCN3Encoding] && !isGFX10())
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
                                       uint64_t Addr, const void *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);

  // Our branches take a simm16, but we need two extra bits to account for the
  // factor of 4.
  APInt SignedOffset(18, Imm * 4, true);
  int64_t Offset = (SignedOffset.sext(64) + 4 + Addr).getSExtValue();

  if (DAsm->tryAddingSymbolicOperand(Inst, Offset, Addr, true, 2, 2))
    return MCDisassembler::Success;
  return addOperand(Inst, MCOperand::createImm(Imm));
}

static DecodeStatus decodeSMEMOffset(MCInst &Inst, unsigned Imm,
                                     uint64_t Addr, const void *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  int64_t Offset;
  if (DAsm->isVI()) {         // VI supports 20-bit unsigned offsets.
    Offset = Imm & 0xFFFFF;
  } else {                    // GFX9+ supports 21-bit signed offsets.
    Offset = SignExtend64<21>(Imm);
  }
  return addOperand(Inst, MCOperand::createImm(Offset));
}

static DecodeStatus decodeBoolReg(MCInst &Inst, unsigned Val,
                                  uint64_t Addr, const void *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeBoolReg(Val));
}

#define DECODE_OPERAND(StaticDecoderName, DecoderName) \
static DecodeStatus StaticDecoderName(MCInst &Inst, \
                                       unsigned Imm, \
                                       uint64_t /*Addr*/, \
                                       const void *Decoder) { \
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder); \
  return addOperand(Inst, DAsm->DecoderName(Imm)); \
}

#define DECODE_OPERAND_REG(RegClass) \
DECODE_OPERAND(Decode##RegClass##RegisterClass, decodeOperand_##RegClass)

DECODE_OPERAND_REG(VGPR_32)
DECODE_OPERAND_REG(VRegOrLds_32)
DECODE_OPERAND_REG(VS_32)
DECODE_OPERAND_REG(VS_64)
DECODE_OPERAND_REG(VS_128)

DECODE_OPERAND_REG(VReg_64)
DECODE_OPERAND_REG(VReg_96)
DECODE_OPERAND_REG(VReg_128)

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
DECODE_OPERAND_REG(AReg_128)
DECODE_OPERAND_REG(AReg_512)
DECODE_OPERAND_REG(AReg_1024)
DECODE_OPERAND_REG(AV_32)
DECODE_OPERAND_REG(AV_64)

static DecodeStatus decodeOperand_VSrc16(MCInst &Inst,
                                         unsigned Imm,
                                         uint64_t Addr,
                                         const void *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeOperand_VSrc16(Imm));
}

static DecodeStatus decodeOperand_VSrcV216(MCInst &Inst,
                                         unsigned Imm,
                                         uint64_t Addr,
                                         const void *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeOperand_VSrcV216(Imm));
}

static DecodeStatus decodeOperand_VS_16(MCInst &Inst,
                                        unsigned Imm,
                                        uint64_t Addr,
                                        const void *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeOperand_VSrc16(Imm));
}

static DecodeStatus decodeOperand_VS_32(MCInst &Inst,
                                        unsigned Imm,
                                        uint64_t Addr,
                                        const void *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeOperand_VS_32(Imm));
}

static DecodeStatus decodeOperand_AReg_128(MCInst &Inst,
                                           unsigned Imm,
                                           uint64_t Addr,
                                           const void *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeSrcOp(AMDGPUDisassembler::OPW128, Imm | 512));
}

static DecodeStatus decodeOperand_AReg_512(MCInst &Inst,
                                           unsigned Imm,
                                           uint64_t Addr,
                                           const void *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeSrcOp(AMDGPUDisassembler::OPW512, Imm | 512));
}

static DecodeStatus decodeOperand_AReg_1024(MCInst &Inst,
                                            unsigned Imm,
                                            uint64_t Addr,
                                            const void *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeSrcOp(AMDGPUDisassembler::OPW1024, Imm | 512));
}

static DecodeStatus decodeOperand_SReg_32(MCInst &Inst,
                                          unsigned Imm,
                                          uint64_t Addr,
                                          const void *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeOperand_SReg_32(Imm));
}

static DecodeStatus decodeOperand_VGPR_32(MCInst &Inst,
                                         unsigned Imm,
                                         uint64_t Addr,
                                         const void *Decoder) {
  auto DAsm = static_cast<const AMDGPUDisassembler*>(Decoder);
  return addOperand(Inst, DAsm->decodeSrcOp(AMDGPUDisassembler::OPW32, Imm));
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

DecodeStatus AMDGPUDisassembler::tryDecodeInst(const uint8_t* Table,
                                               MCInst &MI,
                                               uint64_t Inst,
                                               uint64_t Address) const {
  assert(MI.getOpcode() == 0);
  assert(MI.getNumOperands() == 0);
  MCInst TmpInst;
  HasLiteral = false;
  const auto SavedBytes = Bytes;
  if (decodeInstruction(Table, TmpInst, Inst, Address, this, STI)) {
    MI = TmpInst;
    return MCDisassembler::Success;
  }
  Bytes = SavedBytes;
  return MCDisassembler::Fail;
}

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

      Res = tryDecodeInst(DecoderTableDPP64, MI, QW, Address);
      if (Res) break;

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

    if (STI.getFeatureBits()[AMDGPU::FeatureGFX10_BEncoding]) {
      Res = tryDecodeInst(DecoderTableGFX10_B32, MI, DW, Address);
      if (Res) break;
    }

    Res = tryDecodeInst(DecoderTableGFX1032, MI, DW, Address);
    if (Res) break;

    if (Bytes.size() < 4) break;
    const uint64_t QW = ((uint64_t)eatBytes<uint32_t>(Bytes) << 32) | DW;
    Res = tryDecodeInst(DecoderTableGFX864, MI, QW, Address);
    if (Res) break;

    Res = tryDecodeInst(DecoderTableAMDGPU64, MI, QW, Address);
    if (Res) break;

    Res = tryDecodeInst(DecoderTableGFX964, MI, QW, Address);
    if (Res) break;

    Res = tryDecodeInst(DecoderTableGFX1064, MI, QW, Address);
  } while (false);

  if (Res && (MI.getOpcode() == AMDGPU::V_MAC_F32_e64_vi ||
              MI.getOpcode() == AMDGPU::V_MAC_F32_e64_gfx6_gfx7 ||
              MI.getOpcode() == AMDGPU::V_MAC_F32_e64_gfx10 ||
              MI.getOpcode() == AMDGPU::V_MAC_F16_e64_vi ||
              MI.getOpcode() == AMDGPU::V_FMAC_F32_e64_vi ||
              MI.getOpcode() == AMDGPU::V_FMAC_F32_e64_gfx10 ||
              MI.getOpcode() == AMDGPU::V_FMAC_F16_e64_gfx10)) {
    // Insert dummy unused src2_modifiers.
    insertNamedMCOperand(MI, MCOperand::createImm(0),
                         AMDGPU::OpName::src2_modifiers);
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
          MI.insert(MI.begin() + VAddr0Idx + 1 + i,
                    decodeOperand_VGPR_32(Bytes[i]));
        }
        Bytes = Bytes.slice(4 * NSAWords);
      }
    }

    if (Res)
      Res = convertMIMGInst(MI);
  }

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

  // if the opcode was not recognized we'll assume a Size of 4 bytes
  // (unless there are fewer bytes left)
  Size = Res ? (MaxInstBytesNum - Bytes.size())
             : std::min((size_t)4, Bytes_.size());
  return Res;
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

DecodeStatus AMDGPUDisassembler::convertDPP8Inst(MCInst &MI) const {
  unsigned Opc = MI.getOpcode();
  unsigned DescNumOps = MCII->get(Opc).getNumOperands();

  // Insert dummy unused src modifiers.
  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0_modifiers) != -1)
    insertNamedMCOperand(MI, MCOperand::createImm(0),
                         AMDGPU::OpName::src0_modifiers);

  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src1_modifiers) != -1)
    insertNamedMCOperand(MI, MCOperand::createImm(0),
                         AMDGPU::OpName::src1_modifiers);

  return isValidDPP8(MI) ? MCDisassembler::Success : MCDisassembler::SoftFail;
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

  assert(VDataIdx != -1);
  assert(DMaskIdx != -1);
  assert(TFEIdx != -1);

  const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(MI.getOpcode());
  bool IsAtomic = (VDstIdx != -1);
  bool IsGather4 = MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::Gather4;

  bool IsNSA = false;
  unsigned AddrSize = Info->VAddrDwords;

  if (STI.getFeatureBits()[AMDGPU::FeatureGFX10]) {
    unsigned DimIdx =
        AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::dim);
    const AMDGPU::MIMGBaseOpcodeInfo *BaseOpcode =
        AMDGPU::getMIMGBaseOpcodeInfo(Info->BaseOpcode);
    const AMDGPU::MIMGDimInfo *Dim =
        AMDGPU::getMIMGDimInfoByEncoding(MI.getOperand(DimIdx).getImm());

    AddrSize = BaseOpcode->NumExtraArgs +
               (BaseOpcode->Gradients ? Dim->NumGradients : 0) +
               (BaseOpcode->Coordinates ? Dim->NumCoords : 0) +
               (BaseOpcode->LodOrClampOrMip ? 1 : 0);
    IsNSA = Info->MIMGEncoding == AMDGPU::MIMGEncGfx10NSA;
    if (!IsNSA) {
      if (AddrSize > 8)
        AddrSize = 16;
      else if (AddrSize > 4)
        AddrSize = 8;
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

  // FIXME: Add tfe support
  if (MI.getOperand(TFEIdx).getImm())
    return MCDisassembler::Success;

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

  unsigned NewVAddr0 = AMDGPU::NoRegister;
  if (STI.getFeatureBits()[AMDGPU::FeatureGFX10] && !IsNSA &&
      AddrSize != Info->VAddrDwords) {
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

MCOperand AMDGPUDisassembler::decodeOperand_AReg_128(unsigned Val) const {
  return createRegOperand(AMDGPU::AReg_128RegClassID, Val & 255);
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
    return MCOperand::createImm(getInlineImmVal32(Imm));
  case OPW64:
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
  case OPW64: return VReg_64RegClassID;
  case OPW128: return VReg_128RegClassID;
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
  case OPW64: return AReg_64RegClassID;
  case OPW128: return AReg_128RegClassID;
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
  case OPW64: return SGPR_64RegClassID;
  case OPW128: return SGPR_128RegClassID;
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
  case OPW64: return TTMP_64RegClassID;
  case OPW128: return TTMP_128RegClassID;
  case OPW256: return TTMP_256RegClassID;
  case OPW512: return TTMP_512RegClassID;
  }
}

int AMDGPUDisassembler::getTTmpIdx(unsigned Val) const {
  using namespace AMDGPU::EncValues;

  unsigned TTmpMin =
      (isGFX9() || isGFX10()) ? TTMP_GFX9_GFX10_MIN : TTMP_VI_MIN;
  unsigned TTmpMax =
      (isGFX9() || isGFX10()) ? TTMP_GFX9_GFX10_MAX : TTMP_VI_MAX;

  return (TTmpMin <= Val && Val <= TTmpMax)? Val - TTmpMin : -1;
}

MCOperand AMDGPUDisassembler::decodeSrcOp(const OpWidthTy Width, unsigned Val) const {
  using namespace AMDGPU::EncValues;

  assert(Val < 1024); // enum10

  bool IsAGPR = Val & 512;
  Val &= 511;

  if (VGPR_MIN <= Val && Val <= VGPR_MAX) {
    return createRegOperand(IsAGPR ? getAgprClassId(Width)
                                   : getVgprClassId(Width), Val - VGPR_MIN);
  }
  if (Val <= SGPR_MAX) {
    assert(SGPR_MIN == 0); // "SGPR_MIN <= Val" is always true and causes compilation warning.
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

  if (Val == LITERAL_CONST)
    return decodeLiteralConstant();

  switch (Width) {
  case OPW32:
  case OPW16:
  case OPWV216:
    return decodeSpecialReg32(Val);
  case OPW64:
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
    assert(SGPR_MIN == 0); // "SGPR_MIN <= Val" is always true and causes compilation warning.
    return createSRegOperand(getSgprClassId(Width), Val - SGPR_MIN);
  }

  int TTmpIdx = getTTmpIdx(Val);
  if (TTmpIdx >= 0) {
    return createSRegOperand(getTtmpClassId(Width), TTmpIdx);
  }

  llvm_unreachable("unknown dst register");
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
  case 124: return createRegOperand(M0);
  case 125: return createRegOperand(SGPR_NULL);
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
  case 125: return createRegOperand(SGPR_NULL);
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
        Val <= (isGFX10() ? SDWA9EncValues::SRC_SGPR_MAX_GFX10
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

bool AMDGPUDisassembler::isGFX9() const {
  return STI.getFeatureBits()[AMDGPU::FeatureGFX9];
}

bool AMDGPUDisassembler::isGFX10() const {
  return STI.getFeatureBits()[AMDGPU::FeatureGFX10];
}

//===----------------------------------------------------------------------===//
// AMDGPUSymbolizer
//===----------------------------------------------------------------------===//

// Try to find symbol name for specified label
bool AMDGPUSymbolizer::tryAddingSymbolicOperand(MCInst &Inst,
                                raw_ostream &/*cStream*/, int64_t Value,
                                uint64_t /*Address*/, bool IsBranch,
                                uint64_t /*Offset*/, uint64_t /*InstSize*/) {

  if (!IsBranch) {
    return false;
  }

  auto *Symbols = static_cast<SectionSymbolsTy *>(DisInfo);
  if (!Symbols)
    return false;

  auto Result = std::find_if(Symbols->begin(), Symbols->end(),
                             [Value](const SymbolInfoTy& Val) {
                                return Val.Addr == static_cast<uint64_t>(Value)
                                    && Val.Type == ELF::STT_NOTYPE;
                             });
  if (Result != Symbols->end()) {
    auto *Sym = Ctx.getOrCreateSymbol(Result->Name);
    const auto *Add = MCSymbolRefExpr::create(Sym, Ctx);
    Inst.addOperand(MCOperand::createExpr(Add));
    return true;
  }
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
