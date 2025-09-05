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
#include "MCTargetDesc/AMDGPUMCExpr.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "SIRegisterInfo.h"
#include "TargetInfo/AMDGPUTargetInfo.h"
#include "Utils/AMDGPUAsmUtils.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm-c/DisassemblerTypes.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDecoder.h"
#include "llvm/MC/MCDecoderOps.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;
using namespace llvm::MCD;

#define DEBUG_TYPE "amdgpu-disassembler"

#define SGPR_MAX                                                               \
  (isGFX10Plus() ? AMDGPU::EncValues::SGPR_MAX_GFX10                           \
                 : AMDGPU::EncValues::SGPR_MAX_SI)

using DecodeStatus = llvm::MCDisassembler::DecodeStatus;

static int64_t getInlineImmValF16(unsigned Imm);
static int64_t getInlineImmValBF16(unsigned Imm);
static int64_t getInlineImmVal32(unsigned Imm);
static int64_t getInlineImmVal64(unsigned Imm);

AMDGPUDisassembler::AMDGPUDisassembler(const MCSubtargetInfo &STI,
                                       MCContext &Ctx, MCInstrInfo const *MCII)
    : MCDisassembler(STI, Ctx), MCII(MCII), MRI(*Ctx.getRegisterInfo()),
      MAI(*Ctx.getAsmInfo()),
      HwModeRegClass(STI.getHwMode(MCSubtargetInfo::HwMode_RegInfo)),
      TargetMaxInstBytes(MAI.getMaxInstLength(&STI)),
      CodeObjectVersion(AMDGPU::getDefaultAMDHSACodeObjectVersion()) {
  // ToDo: AMDGPUDisassembler supports only VI ISA.
  if (!STI.hasFeature(AMDGPU::FeatureGCN3Encoding) && !isGFX10Plus())
    reportFatalUsageError("disassembly not yet supported for subtarget");

  for (auto [Symbol, Code] : AMDGPU::UCVersion::getGFXVersions())
    createConstantSymbolExpr(Symbol, Code);

  UCVersionW64Expr = createConstantSymbolExpr("UC_VERSION_W64_BIT", 0x2000);
  UCVersionW32Expr = createConstantSymbolExpr("UC_VERSION_W32_BIT", 0x4000);
  UCVersionMDPExpr = createConstantSymbolExpr("UC_VERSION_MDP_BIT", 0x8000);
}

void AMDGPUDisassembler::setABIVersion(unsigned Version) {
  CodeObjectVersion = AMDGPU::getAMDHSACodeObjectVersion(Version);
}

inline static MCDisassembler::DecodeStatus
addOperand(MCInst &Inst, const MCOperand& Opnd) {
  Inst.addOperand(Opnd);
  return Opnd.isValid() ?
    MCDisassembler::Success :
    MCDisassembler::Fail;
}

static int insertNamedMCOperand(MCInst &MI, const MCOperand &Op,
                                AMDGPU::OpName Name) {
  int OpIdx = AMDGPU::getNamedOperandIdx(MI.getOpcode(), Name);
  if (OpIdx != -1) {
    auto *I = MI.begin();
    std::advance(I, OpIdx);
    MI.insert(I, Op);
  }
  return OpIdx;
}

static DecodeStatus decodeSOPPBrTarget(MCInst &Inst, unsigned Imm,
                                       uint64_t Addr,
                                       const MCDisassembler *Decoder) {
  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);

  // Our branches take a simm16.
  int64_t Offset = SignExtend64<16>(Imm) * 4 + 4 + Addr;

  if (DAsm->tryAddingSymbolicOperand(Inst, Offset, Addr, true, 2, 2, 0))
    return MCDisassembler::Success;
  return addOperand(Inst, MCOperand::createImm(Imm));
}

static DecodeStatus decodeSMEMOffset(MCInst &Inst, unsigned Imm, uint64_t Addr,
                                     const MCDisassembler *Decoder) {
  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);
  int64_t Offset;
  if (DAsm->isGFX12Plus()) { // GFX12 supports 24-bit signed offsets.
    Offset = SignExtend64<24>(Imm);
  } else if (DAsm->isVI()) { // VI supports 20-bit unsigned offsets.
    Offset = Imm & 0xFFFFF;
  } else { // GFX9+ supports 21-bit signed offsets.
    Offset = SignExtend64<21>(Imm);
  }
  return addOperand(Inst, MCOperand::createImm(Offset));
}

static DecodeStatus decodeBoolReg(MCInst &Inst, unsigned Val, uint64_t Addr,
                                  const MCDisassembler *Decoder) {
  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);
  return addOperand(Inst, DAsm->decodeBoolReg(Inst, Val));
}

static DecodeStatus decodeSplitBarrier(MCInst &Inst, unsigned Val,
                                       uint64_t Addr,
                                       const MCDisassembler *Decoder) {
  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);
  return addOperand(Inst, DAsm->decodeSplitBarrier(Inst, Val));
}

static DecodeStatus decodeDpp8FI(MCInst &Inst, unsigned Val, uint64_t Addr,
                                 const MCDisassembler *Decoder) {
  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);
  return addOperand(Inst, DAsm->decodeDpp8FI(Val));
}

#define DECODE_OPERAND(StaticDecoderName, DecoderName)                         \
  static DecodeStatus StaticDecoderName(MCInst &Inst, unsigned Imm,            \
                                        uint64_t /*Addr*/,                     \
                                        const MCDisassembler *Decoder) {       \
    auto DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);              \
    return addOperand(Inst, DAsm->DecoderName(Imm));                           \
  }

// Decoder for registers, decode directly using RegClassID. Imm(8-bit) is
// number of register. Used by VGPR only and AGPR only operands.
#define DECODE_OPERAND_REG_8(RegClass)                                         \
  static DecodeStatus Decode##RegClass##RegisterClass(                         \
      MCInst &Inst, unsigned Imm, uint64_t /*Addr*/,                           \
      const MCDisassembler *Decoder) {                                         \
    assert(Imm < (1 << 8) && "8-bit encoding");                                \
    auto DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);              \
    return addOperand(                                                         \
        Inst, DAsm->createRegOperand(AMDGPU::RegClass##RegClassID, Imm));      \
  }

#define DECODE_SrcOp(Name, EncSize, OpWidth, EncImm)                           \
  static DecodeStatus Name(MCInst &Inst, unsigned Imm, uint64_t /*Addr*/,      \
                           const MCDisassembler *Decoder) {                    \
    assert(Imm < (1 << EncSize) && #EncSize "-bit encoding");                  \
    auto DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);              \
    return addOperand(Inst, DAsm->decodeSrcOp(Inst, OpWidth, EncImm));         \
  }

static DecodeStatus decodeSrcOp(MCInst &Inst, unsigned EncSize,
                                unsigned OpWidth, unsigned Imm, unsigned EncImm,
                                const MCDisassembler *Decoder) {
  assert(Imm < (1U << EncSize) && "Operand doesn't fit encoding!");
  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);
  return addOperand(Inst, DAsm->decodeSrcOp(Inst, OpWidth, EncImm));
}

// Decoder for registers. Imm(7-bit) is number of register, uses decodeSrcOp to
// get register class. Used by SGPR only operands.
#define DECODE_OPERAND_SREG_7(RegClass, OpWidth)                               \
  DECODE_SrcOp(Decode##RegClass##RegisterClass, 7, OpWidth, Imm)

#define DECODE_OPERAND_SREG_8(RegClass, OpWidth)                               \
  DECODE_SrcOp(Decode##RegClass##RegisterClass, 8, OpWidth, Imm)

// Decoder for registers. Imm(10-bit): Imm{7-0} is number of register,
// Imm{9} is acc(agpr or vgpr) Imm{8} should be 0 (see VOP3Pe_SMFMAC).
// Set Imm{8} to 1 (IS_VGPR) to decode using 'enum10' from decodeSrcOp.
// Used by AV_ register classes (AGPR or VGPR only register operands).
template <unsigned OpWidth>
static DecodeStatus decodeAV10(MCInst &Inst, unsigned Imm, uint64_t /* Addr */,
                               const MCDisassembler *Decoder) {
  return decodeSrcOp(Inst, 10, OpWidth, Imm, Imm | AMDGPU::EncValues::IS_VGPR,
                     Decoder);
}

// Decoder for Src(9-bit encoding) registers only.
template <unsigned OpWidth>
static DecodeStatus decodeSrcReg9(MCInst &Inst, unsigned Imm,
                                  uint64_t /* Addr */,
                                  const MCDisassembler *Decoder) {
  return decodeSrcOp(Inst, 9, OpWidth, Imm, Imm, Decoder);
}

// Decoder for Src(9-bit encoding) AGPR, register number encoded in 9bits, set
// Imm{9} to 1 (set acc) and decode using 'enum10' from decodeSrcOp, registers
// only.
template <unsigned OpWidth>
static DecodeStatus decodeSrcA9(MCInst &Inst, unsigned Imm, uint64_t /* Addr */,
                                const MCDisassembler *Decoder) {
  return decodeSrcOp(Inst, 9, OpWidth, Imm, Imm | 512, Decoder);
}

// Decoder for 'enum10' from decodeSrcOp, Imm{0-8} is 9-bit Src encoding
// Imm{9} is acc, registers only.
template <unsigned OpWidth>
static DecodeStatus decodeSrcAV10(MCInst &Inst, unsigned Imm,
                                  uint64_t /* Addr */,
                                  const MCDisassembler *Decoder) {
  return decodeSrcOp(Inst, 10, OpWidth, Imm, Imm, Decoder);
}

// Decoder for RegisterOperands using 9-bit Src encoding. Operand can be
// register from RegClass or immediate. Registers that don't belong to RegClass
// will be decoded and InstPrinter will report warning. Immediate will be
// decoded into constant matching the OperandType (important for floating point
// types).
template <unsigned OpWidth>
static DecodeStatus decodeSrcRegOrImm9(MCInst &Inst, unsigned Imm,
                                       uint64_t /* Addr */,
                                       const MCDisassembler *Decoder) {
  return decodeSrcOp(Inst, 9, OpWidth, Imm, Imm, Decoder);
}

// Decoder for Src(9-bit encoding) AGPR or immediate. Set Imm{9} to 1 (set acc)
// and decode using 'enum10' from decodeSrcOp.
template <unsigned OpWidth>
static DecodeStatus decodeSrcRegOrImmA9(MCInst &Inst, unsigned Imm,
                                        uint64_t /* Addr */,
                                        const MCDisassembler *Decoder) {
  return decodeSrcOp(Inst, 9, OpWidth, Imm, Imm | 512, Decoder);
}

// Default decoders generated by tablegen: 'Decode<RegClass>RegisterClass'
// when RegisterClass is used as an operand. Most often used for destination
// operands.

DECODE_OPERAND_REG_8(VGPR_32)
DECODE_OPERAND_REG_8(VGPR_32_Lo128)
DECODE_OPERAND_REG_8(VReg_64)
DECODE_OPERAND_REG_8(VReg_96)
DECODE_OPERAND_REG_8(VReg_128)
DECODE_OPERAND_REG_8(VReg_192)
DECODE_OPERAND_REG_8(VReg_256)
DECODE_OPERAND_REG_8(VReg_288)
DECODE_OPERAND_REG_8(VReg_320)
DECODE_OPERAND_REG_8(VReg_352)
DECODE_OPERAND_REG_8(VReg_384)
DECODE_OPERAND_REG_8(VReg_512)
DECODE_OPERAND_REG_8(VReg_1024)

DECODE_OPERAND_SREG_7(SReg_32, 32)
DECODE_OPERAND_SREG_7(SReg_32_XM0, 32)
DECODE_OPERAND_SREG_7(SReg_32_XEXEC, 32)
DECODE_OPERAND_SREG_7(SReg_32_XM0_XEXEC, 32)
DECODE_OPERAND_SREG_7(SReg_32_XEXEC_HI, 32)
DECODE_OPERAND_SREG_7(SReg_64_XEXEC, 64)
DECODE_OPERAND_SREG_7(SReg_64_XEXEC_XNULL, 64)
DECODE_OPERAND_SREG_7(SReg_96, 96)
DECODE_OPERAND_SREG_7(SReg_128, 128)
DECODE_OPERAND_SREG_7(SReg_128_XNULL, 128)
DECODE_OPERAND_SREG_7(SReg_256, 256)
DECODE_OPERAND_SREG_7(SReg_256_XNULL, 256)
DECODE_OPERAND_SREG_7(SReg_512, 512)

DECODE_OPERAND_SREG_8(SReg_64, 64)

DECODE_OPERAND_REG_8(AGPR_32)
DECODE_OPERAND_REG_8(AReg_64)
DECODE_OPERAND_REG_8(AReg_128)
DECODE_OPERAND_REG_8(AReg_256)
DECODE_OPERAND_REG_8(AReg_512)
DECODE_OPERAND_REG_8(AReg_1024)

static DecodeStatus DecodeVGPR_16RegisterClass(MCInst &Inst, unsigned Imm,
                                               uint64_t /*Addr*/,
                                               const MCDisassembler *Decoder) {
  assert(isUInt<10>(Imm) && "10-bit encoding expected");
  assert((Imm & (1 << 8)) == 0 && "Imm{8} should not be used");

  bool IsHi = Imm & (1 << 9);
  unsigned RegIdx = Imm & 0xff;
  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);
  return addOperand(Inst, DAsm->createVGPR16Operand(RegIdx, IsHi));
}

static DecodeStatus
DecodeVGPR_16_Lo128RegisterClass(MCInst &Inst, unsigned Imm, uint64_t /*Addr*/,
                                 const MCDisassembler *Decoder) {
  assert(isUInt<8>(Imm) && "8-bit encoding expected");

  bool IsHi = Imm & (1 << 7);
  unsigned RegIdx = Imm & 0x7f;
  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);
  return addOperand(Inst, DAsm->createVGPR16Operand(RegIdx, IsHi));
}

template <unsigned OpWidth>
static DecodeStatus decodeOperand_VSrcT16_Lo128(MCInst &Inst, unsigned Imm,
                                                uint64_t /*Addr*/,
                                                const MCDisassembler *Decoder) {
  assert(isUInt<9>(Imm) && "9-bit encoding expected");

  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);
  if (Imm & AMDGPU::EncValues::IS_VGPR) {
    bool IsHi = Imm & (1 << 7);
    unsigned RegIdx = Imm & 0x7f;
    return addOperand(Inst, DAsm->createVGPR16Operand(RegIdx, IsHi));
  }
  return addOperand(Inst, DAsm->decodeNonVGPRSrcOp(Inst, OpWidth, Imm & 0xFF));
}

template <unsigned OpWidth>
static DecodeStatus decodeOperand_VSrcT16(MCInst &Inst, unsigned Imm,
                                          uint64_t /*Addr*/,
                                          const MCDisassembler *Decoder) {
  assert(isUInt<10>(Imm) && "10-bit encoding expected");

  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);
  if (Imm & AMDGPU::EncValues::IS_VGPR) {
    bool IsHi = Imm & (1 << 9);
    unsigned RegIdx = Imm & 0xff;
    return addOperand(Inst, DAsm->createVGPR16Operand(RegIdx, IsHi));
  }
  return addOperand(Inst, DAsm->decodeNonVGPRSrcOp(Inst, OpWidth, Imm & 0xFF));
}

static DecodeStatus decodeOperand_VGPR_16(MCInst &Inst, unsigned Imm,
                                          uint64_t /*Addr*/,
                                          const MCDisassembler *Decoder) {
  assert(isUInt<10>(Imm) && "10-bit encoding expected");
  assert(Imm & AMDGPU::EncValues::IS_VGPR && "VGPR expected");

  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);

  bool IsHi = Imm & (1 << 9);
  unsigned RegIdx = Imm & 0xff;
  return addOperand(Inst, DAsm->createVGPR16Operand(RegIdx, IsHi));
}

static DecodeStatus decodeOperand_KImmFP(MCInst &Inst, unsigned Imm,
                                         uint64_t Addr,
                                         const MCDisassembler *Decoder) {
  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);
  return addOperand(Inst, DAsm->decodeMandatoryLiteralConstant(Imm));
}

static DecodeStatus decodeOperand_KImmFP64(MCInst &Inst, uint64_t Imm,
                                           uint64_t Addr,
                                           const MCDisassembler *Decoder) {
  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);
  return addOperand(Inst, DAsm->decodeMandatoryLiteral64Constant(Imm));
}

static DecodeStatus decodeOperandVOPDDstY(MCInst &Inst, unsigned Val,
                                          uint64_t Addr, const void *Decoder) {
  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);
  return addOperand(Inst, DAsm->decodeVOPDDstYOp(Inst, Val));
}

static DecodeStatus decodeAVLdSt(MCInst &Inst, unsigned Imm, unsigned Opw,
                                 const MCDisassembler *Decoder) {
  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);
  return addOperand(Inst, DAsm->decodeSrcOp(Inst, Opw, Imm | 256));
}

template <unsigned Opw>
static DecodeStatus decodeAVLdSt(MCInst &Inst, unsigned Imm,
                                 uint64_t /* Addr */,
                                 const MCDisassembler *Decoder) {
  return decodeAVLdSt(Inst, Imm, Opw, Decoder);
}

static DecodeStatus decodeOperand_VSrc_f64(MCInst &Inst, unsigned Imm,
                                           uint64_t Addr,
                                           const MCDisassembler *Decoder) {
  assert(Imm < (1 << 9) && "9-bit encoding");
  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);
  return addOperand(Inst, DAsm->decodeSrcOp(Inst, 64, Imm));
}

#define DECODE_SDWA(DecName) \
DECODE_OPERAND(decodeSDWA##DecName, decodeSDWA##DecName)

DECODE_SDWA(Src32)
DECODE_SDWA(Src16)
DECODE_SDWA(VopcDst)

static DecodeStatus decodeVersionImm(MCInst &Inst, unsigned Imm,
                                     uint64_t /* Addr */,
                                     const MCDisassembler *Decoder) {
  const auto *DAsm = static_cast<const AMDGPUDisassembler *>(Decoder);
  return addOperand(Inst, DAsm->decodeVersionImm(Imm));
}

#include "AMDGPUGenDisassemblerTables.inc"

namespace {
// Define bitwidths for various types used to instantiate the decoder.
template <> constexpr uint32_t InsnBitWidth<uint32_t> = 32;
template <> constexpr uint32_t InsnBitWidth<uint64_t> = 64;
template <> constexpr uint32_t InsnBitWidth<std::bitset<96>> = 96;
template <> constexpr uint32_t InsnBitWidth<std::bitset<128>> = 128;
} // namespace

//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

template <typename InsnType>
DecodeStatus AMDGPUDisassembler::tryDecodeInst(const uint8_t *Table, MCInst &MI,
                                               InsnType Inst, uint64_t Address,
                                               raw_ostream &Comments) const {
  assert(MI.getOpcode() == 0);
  assert(MI.getNumOperands() == 0);
  MCInst TmpInst;
  HasLiteral = false;
  const auto SavedBytes = Bytes;

  SmallString<64> LocalComments;
  raw_svector_ostream LocalCommentStream(LocalComments);
  CommentStream = &LocalCommentStream;

  DecodeStatus Res =
      decodeInstruction(Table, TmpInst, Inst, Address, this, STI);

  CommentStream = nullptr;

  if (Res != MCDisassembler::Fail) {
    MI = TmpInst;
    Comments << LocalComments;
    return MCDisassembler::Success;
  }
  Bytes = SavedBytes;
  return MCDisassembler::Fail;
}

template <typename InsnType>
DecodeStatus
AMDGPUDisassembler::tryDecodeInst(const uint8_t *Table1, const uint8_t *Table2,
                                  MCInst &MI, InsnType Inst, uint64_t Address,
                                  raw_ostream &Comments) const {
  for (const uint8_t *T : {Table1, Table2}) {
    if (DecodeStatus Res = tryDecodeInst(T, MI, Inst, Address, Comments))
      return Res;
  }
  return MCDisassembler::Fail;
}

template <typename T> static inline T eatBytes(ArrayRef<uint8_t>& Bytes) {
  assert(Bytes.size() >= sizeof(T));
  const auto Res =
      support::endian::read<T, llvm::endianness::little>(Bytes.data());
  Bytes = Bytes.slice(sizeof(T));
  return Res;
}

static inline std::bitset<96> eat12Bytes(ArrayRef<uint8_t> &Bytes) {
  using namespace llvm::support::endian;
  assert(Bytes.size() >= 12);
  std::bitset<96> Lo(read<uint64_t, endianness::little>(Bytes.data()));
  Bytes = Bytes.slice(8);
  std::bitset<96> Hi(read<uint32_t, endianness::little>(Bytes.data()));
  Bytes = Bytes.slice(4);
  return (Hi << 64) | Lo;
}

static inline std::bitset<128> eat16Bytes(ArrayRef<uint8_t> &Bytes) {
  using namespace llvm::support::endian;
  assert(Bytes.size() >= 16);
  std::bitset<128> Lo(read<uint64_t, endianness::little>(Bytes.data()));
  Bytes = Bytes.slice(8);
  std::bitset<128> Hi(read<uint64_t, endianness::little>(Bytes.data()));
  Bytes = Bytes.slice(8);
  return (Hi << 64) | Lo;
}

void AMDGPUDisassembler::decodeImmOperands(MCInst &MI,
                                           const MCInstrInfo &MCII) const {
  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  for (auto [OpNo, OpDesc] : enumerate(Desc.operands())) {
    if (OpNo >= MI.getNumOperands())
      continue;

    // TODO: Fix V_DUAL_FMAMK_F32_X_FMAAK_F32_gfx12 vsrc operands,
    // defined to take VGPR_32, but in reality allowing inline constants.
    bool IsSrc = AMDGPU::OPERAND_SRC_FIRST <= OpDesc.OperandType &&
                 OpDesc.OperandType <= AMDGPU::OPERAND_SRC_LAST;
    if (!IsSrc && OpDesc.OperandType != MCOI::OPERAND_REGISTER)
      continue;

    MCOperand &Op = MI.getOperand(OpNo);
    if (!Op.isImm())
      continue;
    int64_t Imm = Op.getImm();
    if (AMDGPU::EncValues::INLINE_INTEGER_C_MIN <= Imm &&
        Imm <= AMDGPU::EncValues::INLINE_INTEGER_C_MAX) {
      Op = decodeIntImmed(Imm);
      continue;
    }

    if (Imm == AMDGPU::EncValues::LITERAL_CONST) {
      Op = decodeLiteralConstant(
          Desc, OpDesc, OpDesc.OperandType == AMDGPU::OPERAND_REG_IMM_FP64);
      continue;
    }

    if (AMDGPU::EncValues::INLINE_FLOATING_C_MIN <= Imm &&
        Imm <= AMDGPU::EncValues::INLINE_FLOATING_C_MAX) {
      switch (OpDesc.OperandType) {
      case AMDGPU::OPERAND_REG_IMM_BF16:
      case AMDGPU::OPERAND_REG_IMM_V2BF16:
      case AMDGPU::OPERAND_REG_INLINE_C_BF16:
      case AMDGPU::OPERAND_REG_INLINE_C_V2BF16:
        Imm = getInlineImmValBF16(Imm);
        break;
      case AMDGPU::OPERAND_REG_IMM_FP16:
      case AMDGPU::OPERAND_REG_IMM_INT16:
      case AMDGPU::OPERAND_REG_IMM_V2FP16:
      case AMDGPU::OPERAND_REG_INLINE_C_FP16:
      case AMDGPU::OPERAND_REG_INLINE_C_INT16:
      case AMDGPU::OPERAND_REG_INLINE_C_V2FP16:
        Imm = getInlineImmValF16(Imm);
        break;
      case AMDGPU::OPERAND_REG_IMM_FP64:
      case AMDGPU::OPERAND_REG_IMM_INT64:
      case AMDGPU::OPERAND_REG_INLINE_AC_FP64:
      case AMDGPU::OPERAND_REG_INLINE_C_FP64:
      case AMDGPU::OPERAND_REG_INLINE_C_INT64:
        Imm = getInlineImmVal64(Imm);
        break;
      default:
        Imm = getInlineImmVal32(Imm);
      }
      Op.setImm(Imm);
    }
  }
}

DecodeStatus AMDGPUDisassembler::getInstruction(MCInst &MI, uint64_t &Size,
                                                ArrayRef<uint8_t> Bytes_,
                                                uint64_t Address,
                                                raw_ostream &CS) const {
  unsigned MaxInstBytesNum = std::min((size_t)TargetMaxInstBytes, Bytes_.size());
  Bytes = Bytes_.slice(0, MaxInstBytesNum);

  // In case the opcode is not recognized we'll assume a Size of 4 bytes (unless
  // there are fewer bytes left). This will be overridden on success.
  Size = std::min((size_t)4, Bytes_.size());

  do {
    // ToDo: better to switch encoding length using some bit predicate
    // but it is unknown yet, so try all we can

    // Try to decode DPP and SDWA first to solve conflict with VOP1 and VOP2
    // encodings
    if (isGFX1250() && Bytes.size() >= 16) {
      std::bitset<128> DecW = eat16Bytes(Bytes);
      if (tryDecodeInst(DecoderTableGFX1250128, MI, DecW, Address, CS))
        break;
      Bytes = Bytes_.slice(0, MaxInstBytesNum);
    }

    if (isGFX11Plus() && Bytes.size() >= 12) {
      std::bitset<96> DecW = eat12Bytes(Bytes);

      if (isGFX11() &&
          tryDecodeInst(DecoderTableGFX1196, DecoderTableGFX11_FAKE1696, MI,
                        DecW, Address, CS))
        break;

      if (isGFX1250() &&
          tryDecodeInst(DecoderTableGFX125096, DecoderTableGFX1250_FAKE1696, MI,
                        DecW, Address, CS))
        break;

      if (isGFX12() &&
          tryDecodeInst(DecoderTableGFX1296, DecoderTableGFX12_FAKE1696, MI,
                        DecW, Address, CS))
        break;

      if (isGFX12() &&
          tryDecodeInst(DecoderTableGFX12W6496, MI, DecW, Address, CS))
        break;

      if (STI.hasFeature(AMDGPU::Feature64BitLiterals)) {
        // Return 8 bytes for a potential literal.
        Bytes = Bytes_.slice(4, MaxInstBytesNum - 4);

        if (isGFX1250() &&
            tryDecodeInst(DecoderTableGFX125096, MI, DecW, Address, CS))
          break;
      }

      // Reinitialize Bytes
      Bytes = Bytes_.slice(0, MaxInstBytesNum);

    } else if (Bytes.size() >= 16 &&
               STI.hasFeature(AMDGPU::FeatureGFX950Insts)) {
      std::bitset<128> DecW = eat16Bytes(Bytes);
      if (tryDecodeInst(DecoderTableGFX940128, MI, DecW, Address, CS))
        break;

      // Reinitialize Bytes
      Bytes = Bytes_.slice(0, MaxInstBytesNum);
    }

    if (Bytes.size() >= 8) {
      const uint64_t QW = eatBytes<uint64_t>(Bytes);

      if (STI.hasFeature(AMDGPU::FeatureGFX10_BEncoding) &&
          tryDecodeInst(DecoderTableGFX10_B64, MI, QW, Address, CS))
        break;

      if (STI.hasFeature(AMDGPU::FeatureUnpackedD16VMem) &&
          tryDecodeInst(DecoderTableGFX80_UNPACKED64, MI, QW, Address, CS))
        break;

      if (STI.hasFeature(AMDGPU::FeatureGFX950Insts) &&
          tryDecodeInst(DecoderTableGFX95064, MI, QW, Address, CS))
        break;

      // Some GFX9 subtargets repurposed the v_mad_mix_f32, v_mad_mixlo_f16 and
      // v_mad_mixhi_f16 for FMA variants. Try to decode using this special
      // table first so we print the correct name.
      if (STI.hasFeature(AMDGPU::FeatureFmaMixInsts) &&
          tryDecodeInst(DecoderTableGFX9_DL64, MI, QW, Address, CS))
        break;

      if (STI.hasFeature(AMDGPU::FeatureGFX940Insts) &&
          tryDecodeInst(DecoderTableGFX94064, MI, QW, Address, CS))
        break;

      if (STI.hasFeature(AMDGPU::FeatureGFX90AInsts) &&
          tryDecodeInst(DecoderTableGFX90A64, MI, QW, Address, CS))
        break;

      if ((isVI() || isGFX9()) &&
          tryDecodeInst(DecoderTableGFX864, MI, QW, Address, CS))
        break;

      if (isGFX9() && tryDecodeInst(DecoderTableGFX964, MI, QW, Address, CS))
        break;

      if (isGFX10() && tryDecodeInst(DecoderTableGFX1064, MI, QW, Address, CS))
        break;

      if (isGFX1250() &&
          tryDecodeInst(DecoderTableGFX125064, DecoderTableGFX1250_FAKE1664, MI,
                        QW, Address, CS))
        break;

      if (isGFX12() &&
          tryDecodeInst(DecoderTableGFX1264, DecoderTableGFX12_FAKE1664, MI, QW,
                        Address, CS))
        break;

      if (isGFX11() &&
          tryDecodeInst(DecoderTableGFX1164, DecoderTableGFX11_FAKE1664, MI, QW,
                        Address, CS))
        break;

      if (isGFX11() &&
          tryDecodeInst(DecoderTableGFX11W6464, MI, QW, Address, CS))
        break;

      if (isGFX12() &&
          tryDecodeInst(DecoderTableGFX12W6464, MI, QW, Address, CS))
        break;

      // Reinitialize Bytes
      Bytes = Bytes_.slice(0, MaxInstBytesNum);
    }

    // Try decode 32-bit instruction
    if (Bytes.size() >= 4) {
      const uint32_t DW = eatBytes<uint32_t>(Bytes);

      if ((isVI() || isGFX9()) &&
          tryDecodeInst(DecoderTableGFX832, MI, DW, Address, CS))
        break;

      if (tryDecodeInst(DecoderTableAMDGPU32, MI, DW, Address, CS))
        break;

      if (isGFX9() && tryDecodeInst(DecoderTableGFX932, MI, DW, Address, CS))
        break;

      if (STI.hasFeature(AMDGPU::FeatureGFX950Insts) &&
          tryDecodeInst(DecoderTableGFX95032, MI, DW, Address, CS))
        break;

      if (STI.hasFeature(AMDGPU::FeatureGFX90AInsts) &&
          tryDecodeInst(DecoderTableGFX90A32, MI, DW, Address, CS))
        break;

      if (STI.hasFeature(AMDGPU::FeatureGFX10_BEncoding) &&
          tryDecodeInst(DecoderTableGFX10_B32, MI, DW, Address, CS))
        break;

      if (isGFX10() && tryDecodeInst(DecoderTableGFX1032, MI, DW, Address, CS))
        break;

      if (isGFX11() &&
          tryDecodeInst(DecoderTableGFX1132, DecoderTableGFX11_FAKE1632, MI, DW,
                        Address, CS))
        break;

      if (isGFX1250() &&
          tryDecodeInst(DecoderTableGFX125032, DecoderTableGFX1250_FAKE1632, MI,
                        DW, Address, CS))
        break;

      if (isGFX12() &&
          tryDecodeInst(DecoderTableGFX1232, DecoderTableGFX12_FAKE1632, MI, DW,
                        Address, CS))
        break;
    }

    return MCDisassembler::Fail;
  } while (false);

  DecodeStatus Status = MCDisassembler::Success;

  decodeImmOperands(MI, *MCII);

  if (MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::DPP) {
    if (isMacDPP(MI))
      convertMacDPPInst(MI);

    if (MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::VOP3P)
      convertVOP3PDPPInst(MI);
    else if (MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::VOPC)
      convertVOPCDPPInst(MI); // Special VOP3 case
    else if (AMDGPU::isVOPC64DPP(MI.getOpcode()))
      convertVOPC64DPPInst(MI); // Special VOP3 case
    else if (AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::dpp8) !=
             -1)
      convertDPP8Inst(MI);
    else if (MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::VOP3)
      convertVOP3DPPInst(MI); // Regular VOP3 case
  }

  convertTrue16OpSel(MI);

  if (AMDGPU::isMAC(MI.getOpcode())) {
    // Insert dummy unused src2_modifiers.
    insertNamedMCOperand(MI, MCOperand::createImm(0),
                         AMDGPU::OpName::src2_modifiers);
  }

  if (MI.getOpcode() == AMDGPU::V_CVT_SR_BF8_F32_e64_dpp ||
      MI.getOpcode() == AMDGPU::V_CVT_SR_FP8_F32_e64_dpp) {
    // Insert dummy unused src2_modifiers.
    insertNamedMCOperand(MI, MCOperand::createImm(0),
                         AMDGPU::OpName::src2_modifiers);
  }

  if ((MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::DS) &&
      !AMDGPU::hasGDS(STI)) {
    insertNamedMCOperand(MI, MCOperand::createImm(0), AMDGPU::OpName::gds);
  }

  if (MCII->get(MI.getOpcode()).TSFlags &
      (SIInstrFlags::MUBUF | SIInstrFlags::FLAT | SIInstrFlags::SMRD)) {
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

  if ((MCII->get(MI.getOpcode()).TSFlags &
       (SIInstrFlags::MTBUF | SIInstrFlags::MUBUF)) &&
      (STI.hasFeature(AMDGPU::FeatureGFX90AInsts))) {
    // GFX90A lost TFE, its place is occupied by ACC.
    int TFEOpIdx =
        AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::tfe);
    if (TFEOpIdx != -1) {
      auto *TFEIter = MI.begin();
      std::advance(TFEIter, TFEOpIdx);
      MI.insert(TFEIter, MCOperand::createImm(0));
    }
  }

  // Validate buffer instruction offsets for GFX12+ - must not be a negative.
  if (isGFX12Plus() && isBufferInstruction(MI)) {
    int OffsetIdx =
        AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::offset);
    if (OffsetIdx != -1) {
      uint32_t Imm = MI.getOperand(OffsetIdx).getImm();
      int64_t SignedOffset = SignExtend64<24>(Imm);
      if (SignedOffset < 0)
        return MCDisassembler::Fail;
    }
  }

  if (MCII->get(MI.getOpcode()).TSFlags &
      (SIInstrFlags::MTBUF | SIInstrFlags::MUBUF)) {
    int SWZOpIdx =
        AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::swz);
    if (SWZOpIdx != -1) {
      auto *SWZIter = MI.begin();
      std::advance(SWZIter, SWZOpIdx);
      MI.insert(SWZIter, MCOperand::createImm(0));
    }
  }

  const MCInstrDesc &Desc = MCII->get(MI.getOpcode());
  if (Desc.TSFlags & SIInstrFlags::MIMG) {
    int VAddr0Idx =
        AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::vaddr0);
    int RsrcIdx =
        AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::srsrc);
    unsigned NSAArgs = RsrcIdx - VAddr0Idx - 1;
    if (VAddr0Idx >= 0 && NSAArgs > 0) {
      unsigned NSAWords = (NSAArgs + 3) / 4;
      if (Bytes.size() < 4 * NSAWords)
        return MCDisassembler::Fail;
      for (unsigned i = 0; i < NSAArgs; ++i) {
        const unsigned VAddrIdx = VAddr0Idx + 1 + i;
        auto VAddrRCID =
            MCII->getOpRegClassID(Desc.operands()[VAddrIdx], HwModeRegClass);
        MI.insert(MI.begin() + VAddrIdx, createRegOperand(VAddrRCID, Bytes[i]));
      }
      Bytes = Bytes.slice(4 * NSAWords);
    }

    convertMIMGInst(MI);
  }

  if (MCII->get(MI.getOpcode()).TSFlags &
      (SIInstrFlags::VIMAGE | SIInstrFlags::VSAMPLE))
    convertMIMGInst(MI);

  if (MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::EXP)
    convertEXPInst(MI);

  if (MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::VINTERP)
    convertVINTERPInst(MI);

  if (MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::SDWA)
    convertSDWAInst(MI);

  if (MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::IsMAI)
    convertMAIInst(MI);

  if (MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::IsWMMA)
    convertWMMAInst(MI);

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

  bool IsSOPK = MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::SOPK;
  if (AMDGPU::hasNamedOperand(MI.getOpcode(), AMDGPU::OpName::imm) && !IsSOPK)
    convertFMAanyK(MI);

  // Some VOPC instructions, e.g., v_cmpx_f_f64, use VOP3 encoding and
  // have EXEC as implicit destination. Issue a warning if encoding for
  // vdst is not EXEC.
  if ((MCII->get(MI.getOpcode()).TSFlags & SIInstrFlags::VOP3) &&
      MCII->get(MI.getOpcode()).hasImplicitDefOfPhysReg(AMDGPU::EXEC)) {
    auto ExecEncoding = MRI.getEncodingValue(AMDGPU::EXEC_LO);
    if (Bytes_[0] != ExecEncoding)
      Status = MCDisassembler::SoftFail;
  }

  Size = MaxInstBytesNum - Bytes.size();
  return Status;
}

void AMDGPUDisassembler::convertEXPInst(MCInst &MI) const {
  if (STI.hasFeature(AMDGPU::FeatureGFX11Insts)) {
    // The MCInst still has these fields even though they are no longer encoded
    // in the GFX11 instruction.
    insertNamedMCOperand(MI, MCOperand::createImm(0), AMDGPU::OpName::vm);
    insertNamedMCOperand(MI, MCOperand::createImm(0), AMDGPU::OpName::compr);
  }
}

void AMDGPUDisassembler::convertVINTERPInst(MCInst &MI) const {
  convertTrue16OpSel(MI);
  if (MI.getOpcode() == AMDGPU::V_INTERP_P10_F16_F32_inreg_t16_gfx11 ||
      MI.getOpcode() == AMDGPU::V_INTERP_P10_F16_F32_inreg_fake16_gfx11 ||
      MI.getOpcode() == AMDGPU::V_INTERP_P10_F16_F32_inreg_t16_gfx12 ||
      MI.getOpcode() == AMDGPU::V_INTERP_P10_F16_F32_inreg_fake16_gfx12 ||
      MI.getOpcode() == AMDGPU::V_INTERP_P10_RTZ_F16_F32_inreg_t16_gfx11 ||
      MI.getOpcode() == AMDGPU::V_INTERP_P10_RTZ_F16_F32_inreg_fake16_gfx11 ||
      MI.getOpcode() == AMDGPU::V_INTERP_P10_RTZ_F16_F32_inreg_t16_gfx12 ||
      MI.getOpcode() == AMDGPU::V_INTERP_P10_RTZ_F16_F32_inreg_fake16_gfx12 ||
      MI.getOpcode() == AMDGPU::V_INTERP_P2_F16_F32_inreg_t16_gfx11 ||
      MI.getOpcode() == AMDGPU::V_INTERP_P2_F16_F32_inreg_fake16_gfx11 ||
      MI.getOpcode() == AMDGPU::V_INTERP_P2_F16_F32_inreg_t16_gfx12 ||
      MI.getOpcode() == AMDGPU::V_INTERP_P2_F16_F32_inreg_fake16_gfx12 ||
      MI.getOpcode() == AMDGPU::V_INTERP_P2_RTZ_F16_F32_inreg_t16_gfx11 ||
      MI.getOpcode() == AMDGPU::V_INTERP_P2_RTZ_F16_F32_inreg_fake16_gfx11 ||
      MI.getOpcode() == AMDGPU::V_INTERP_P2_RTZ_F16_F32_inreg_t16_gfx12 ||
      MI.getOpcode() == AMDGPU::V_INTERP_P2_RTZ_F16_F32_inreg_fake16_gfx12) {
    // The MCInst has this field that is not directly encoded in the
    // instruction.
    insertNamedMCOperand(MI, MCOperand::createImm(0), AMDGPU::OpName::op_sel);
  }
}

void AMDGPUDisassembler::convertSDWAInst(MCInst &MI) const {
  if (STI.hasFeature(AMDGPU::FeatureGFX9) ||
      STI.hasFeature(AMDGPU::FeatureGFX10)) {
    if (AMDGPU::hasNamedOperand(MI.getOpcode(), AMDGPU::OpName::sdst))
      // VOPC - insert clamp
      insertNamedMCOperand(MI, MCOperand::createImm(0), AMDGPU::OpName::clamp);
  } else if (STI.hasFeature(AMDGPU::FeatureVolcanicIslands)) {
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
}

/// Adjust the register values used by V_MFMA_F8F6F4_f8_f8 instructions to the
/// appropriate subregister for the used format width.
static void adjustMFMA_F8F6F4OpRegClass(const MCRegisterInfo &MRI,
                                        MCOperand &MO, uint8_t NumRegs) {
  switch (NumRegs) {
  case 4:
    return MO.setReg(MRI.getSubReg(MO.getReg(), AMDGPU::sub0_sub1_sub2_sub3));
  case 6:
    return MO.setReg(
        MRI.getSubReg(MO.getReg(), AMDGPU::sub0_sub1_sub2_sub3_sub4_sub5));
  case 8:
    if (MCRegister NewReg = MRI.getSubReg(
            MO.getReg(), AMDGPU::sub0_sub1_sub2_sub3_sub4_sub5_sub6_sub7)) {
      MO.setReg(NewReg);
    }
    return;
  case 12: {
    // There is no 384-bit subreg index defined.
    MCRegister BaseReg = MRI.getSubReg(MO.getReg(), AMDGPU::sub0);
    MCRegister NewReg = MRI.getMatchingSuperReg(
        BaseReg, AMDGPU::sub0, &MRI.getRegClass(AMDGPU::VReg_384RegClassID));
    return MO.setReg(NewReg);
  }
  case 16:
    // No-op in cases where one operand is still f8/bf8.
    return;
  default:
    llvm_unreachable("Unexpected size for mfma/wmma f8f6f4 operand");
  }
}

/// f8f6f4 instructions have different pseudos depending on the used formats. In
/// the disassembler table, we only have the variants with the largest register
/// classes which assume using an fp8/bf8 format for both operands. The actual
/// register class depends on the format in blgp and cbsz operands. Adjust the
/// register classes depending on the used format.
void AMDGPUDisassembler::convertMAIInst(MCInst &MI) const {
  int BlgpIdx =
      AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::blgp);
  if (BlgpIdx == -1)
    return;

  int CbszIdx =
      AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::cbsz);

  unsigned CBSZ = MI.getOperand(CbszIdx).getImm();
  unsigned BLGP = MI.getOperand(BlgpIdx).getImm();

  const AMDGPU::MFMA_F8F6F4_Info *AdjustedRegClassOpcode =
      AMDGPU::getMFMA_F8F6F4_WithFormatArgs(CBSZ, BLGP, MI.getOpcode());
  if (!AdjustedRegClassOpcode ||
      AdjustedRegClassOpcode->Opcode == MI.getOpcode())
    return;

  MI.setOpcode(AdjustedRegClassOpcode->Opcode);
  int Src0Idx =
      AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::src0);
  int Src1Idx =
      AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::src1);
  adjustMFMA_F8F6F4OpRegClass(MRI, MI.getOperand(Src0Idx),
                              AdjustedRegClassOpcode->NumRegsSrcA);
  adjustMFMA_F8F6F4OpRegClass(MRI, MI.getOperand(Src1Idx),
                              AdjustedRegClassOpcode->NumRegsSrcB);
}

void AMDGPUDisassembler::convertWMMAInst(MCInst &MI) const {
  int FmtAIdx =
      AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::matrix_a_fmt);
  if (FmtAIdx == -1)
    return;

  int FmtBIdx =
      AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::matrix_b_fmt);

  unsigned FmtA = MI.getOperand(FmtAIdx).getImm();
  unsigned FmtB = MI.getOperand(FmtBIdx).getImm();

  const AMDGPU::MFMA_F8F6F4_Info *AdjustedRegClassOpcode =
      AMDGPU::getWMMA_F8F6F4_WithFormatArgs(FmtA, FmtB, MI.getOpcode());
  if (!AdjustedRegClassOpcode ||
      AdjustedRegClassOpcode->Opcode == MI.getOpcode())
    return;

  MI.setOpcode(AdjustedRegClassOpcode->Opcode);
  int Src0Idx =
      AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::src0);
  int Src1Idx =
      AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::src1);
  adjustMFMA_F8F6F4OpRegClass(MRI, MI.getOperand(Src0Idx),
                              AdjustedRegClassOpcode->NumRegsSrcA);
  adjustMFMA_F8F6F4OpRegClass(MRI, MI.getOperand(Src1Idx),
                              AdjustedRegClassOpcode->NumRegsSrcB);
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
  const AMDGPU::OpName ModOps[] = {AMDGPU::OpName::src0_modifiers,
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

// Instructions decode the op_sel/suffix bits into the src_modifier
// operands. Copy those bits into the src operands for true16 VGPRs.
void AMDGPUDisassembler::convertTrue16OpSel(MCInst &MI) const {
  const unsigned Opc = MI.getOpcode();
  const MCRegisterClass &ConversionRC =
      MRI.getRegClass(AMDGPU::VGPR_16RegClassID);
  constexpr std::array<std::tuple<AMDGPU::OpName, AMDGPU::OpName, unsigned>, 4>
      OpAndOpMods = {{{AMDGPU::OpName::src0, AMDGPU::OpName::src0_modifiers,
                       SISrcMods::OP_SEL_0},
                      {AMDGPU::OpName::src1, AMDGPU::OpName::src1_modifiers,
                       SISrcMods::OP_SEL_0},
                      {AMDGPU::OpName::src2, AMDGPU::OpName::src2_modifiers,
                       SISrcMods::OP_SEL_0},
                      {AMDGPU::OpName::vdst, AMDGPU::OpName::src0_modifiers,
                       SISrcMods::DST_OP_SEL}}};
  for (const auto &[OpName, OpModsName, OpSelMask] : OpAndOpMods) {
    int OpIdx = AMDGPU::getNamedOperandIdx(Opc, OpName);
    int OpModsIdx = AMDGPU::getNamedOperandIdx(Opc, OpModsName);
    if (OpIdx == -1 || OpModsIdx == -1)
      continue;
    MCOperand &Op = MI.getOperand(OpIdx);
    if (!Op.isReg())
      continue;
    if (!ConversionRC.contains(Op.getReg()))
      continue;
    unsigned OpEnc = MRI.getEncodingValue(Op.getReg());
    const MCOperand &OpMods = MI.getOperand(OpModsIdx);
    unsigned ModVal = OpMods.getImm();
    if (ModVal & OpSelMask) { // isHi
      unsigned RegIdx = OpEnc & AMDGPU::HWEncoding::REG_IDX_MASK;
      Op.setReg(ConversionRC.getRegister(RegIdx * 2 + 1));
    }
  }
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
    assert(AMDGPU::hasNamedOperand(Opcode, AMDGPU::OpName::src2));
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

void AMDGPUDisassembler::convertDPP8Inst(MCInst &MI) const {
  unsigned Opc = MI.getOpcode();

  int VDstInIdx =
      AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::vdst_in);
  if (VDstInIdx != -1)
    insertNamedMCOperand(MI, MI.getOperand(0), AMDGPU::OpName::vdst_in);

  unsigned DescNumOps = MCII->get(Opc).getNumOperands();
  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::hasNamedOperand(Opc, AMDGPU::OpName::op_sel)) {
    convertTrue16OpSel(MI);
    auto Mods = collectVOPModifiers(MI);
    insertNamedMCOperand(MI, MCOperand::createImm(Mods.OpSel),
                         AMDGPU::OpName::op_sel);
  } else {
    // Insert dummy unused src modifiers.
    if (MI.getNumOperands() < DescNumOps &&
        AMDGPU::hasNamedOperand(Opc, AMDGPU::OpName::src0_modifiers))
      insertNamedMCOperand(MI, MCOperand::createImm(0),
                           AMDGPU::OpName::src0_modifiers);

    if (MI.getNumOperands() < DescNumOps &&
        AMDGPU::hasNamedOperand(Opc, AMDGPU::OpName::src1_modifiers))
      insertNamedMCOperand(MI, MCOperand::createImm(0),
                           AMDGPU::OpName::src1_modifiers);
  }
}

void AMDGPUDisassembler::convertVOP3DPPInst(MCInst &MI) const {
  convertTrue16OpSel(MI);

  int VDstInIdx =
      AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::vdst_in);
  if (VDstInIdx != -1)
    insertNamedMCOperand(MI, MI.getOperand(0), AMDGPU::OpName::vdst_in);

  unsigned Opc = MI.getOpcode();
  unsigned DescNumOps = MCII->get(Opc).getNumOperands();
  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::hasNamedOperand(Opc, AMDGPU::OpName::op_sel)) {
    auto Mods = collectVOPModifiers(MI);
    insertNamedMCOperand(MI, MCOperand::createImm(Mods.OpSel),
                         AMDGPU::OpName::op_sel);
  }
}

// Given a wide tuple \p Reg check if it will overflow 256 registers.
// \returns \p Reg on success or NoRegister otherwise.
static unsigned CheckVGPROverflow(unsigned Reg, const MCRegisterClass &RC,
                                  const MCRegisterInfo &MRI) {
  unsigned NumRegs = RC.getSizeInBits() / 32;
  MCRegister Sub0 = MRI.getSubReg(Reg, AMDGPU::sub0);
  if (!Sub0)
    return Reg;

  MCRegister BaseReg;
  if (MRI.getRegClass(AMDGPU::VGPR_32RegClassID).contains(Sub0))
    BaseReg = AMDGPU::VGPR0;
  else if (MRI.getRegClass(AMDGPU::AGPR_32RegClassID).contains(Sub0))
    BaseReg = AMDGPU::AGPR0;

  assert(BaseReg && "Only vector registers expected");

  return (Sub0 - BaseReg + NumRegs <= 256) ? Reg : AMDGPU::NoRegister;
}

// Note that before gfx10, the MIMG encoding provided no information about
// VADDR size. Consequently, decoded instructions always show address as if it
// has 1 dword, which could be not really so.
void AMDGPUDisassembler::convertMIMGInst(MCInst &MI) const {
  auto TSFlags = MCII->get(MI.getOpcode()).TSFlags;

  int VDstIdx = AMDGPU::getNamedOperandIdx(MI.getOpcode(),
                                           AMDGPU::OpName::vdst);

  int VDataIdx = AMDGPU::getNamedOperandIdx(MI.getOpcode(),
                                            AMDGPU::OpName::vdata);
  int VAddr0Idx =
      AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::vaddr0);
  AMDGPU::OpName RsrcOpName = (TSFlags & SIInstrFlags::MIMG)
                                  ? AMDGPU::OpName::srsrc
                                  : AMDGPU::OpName::rsrc;
  int RsrcIdx = AMDGPU::getNamedOperandIdx(MI.getOpcode(), RsrcOpName);
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
    addOperand(MI, MCOperand::createImm(BaseOpcode->A16));
    return;
  }

  bool IsAtomic = (VDstIdx != -1);
  bool IsGather4 = TSFlags & SIInstrFlags::Gather4;
  bool IsVSample = TSFlags & SIInstrFlags::VSAMPLE;
  bool IsNSA = false;
  bool IsPartialNSA = false;
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

    // VSAMPLE insts that do not use vaddr3 behave the same as NSA forms.
    // VIMAGE insts other than BVH never use vaddr4.
    IsNSA = Info->MIMGEncoding == AMDGPU::MIMGEncGfx10NSA ||
            Info->MIMGEncoding == AMDGPU::MIMGEncGfx11NSA ||
            Info->MIMGEncoding == AMDGPU::MIMGEncGfx12;
    if (!IsNSA) {
      if (!IsVSample && AddrSize > 12)
        AddrSize = 16;
    } else {
      if (AddrSize > Info->VAddrDwords) {
        if (!STI.hasFeature(AMDGPU::FeaturePartialNSAEncoding)) {
          // The NSA encoding does not contain enough operands for the
          // combination of base opcode / dimension. Should this be an error?
          return;
        }
        IsPartialNSA = true;
      }
    }
  }

  unsigned DMask = MI.getOperand(DMaskIdx).getImm() & 0xf;
  unsigned DstSize = IsGather4 ? 4 : std::max(llvm::popcount(DMask), 1);

  bool D16 = D16Idx >= 0 && MI.getOperand(D16Idx).getImm();
  if (D16 && AMDGPU::hasPackedD16(STI)) {
    DstSize = (DstSize + 1) / 2;
  }

  if (TFEIdx != -1 && MI.getOperand(TFEIdx).getImm())
    DstSize += 1;

  if (DstSize == Info->VDataDwords && AddrSize == Info->VAddrDwords)
    return;

  int NewOpcode =
      AMDGPU::getMIMGOpcode(Info->BaseOpcode, Info->MIMGEncoding, DstSize, AddrSize);
  if (NewOpcode == -1)
    return;

  // Widen the register to the correct number of enabled channels.
  MCRegister NewVdata;
  if (DstSize != Info->VDataDwords) {
    auto DataRCID = MCII->getOpRegClassID(
        MCII->get(NewOpcode).operands()[VDataIdx], HwModeRegClass);

    // Get first subregister of VData
    MCRegister Vdata0 = MI.getOperand(VDataIdx).getReg();
    MCRegister VdataSub0 = MRI.getSubReg(Vdata0, AMDGPU::sub0);
    Vdata0 = (VdataSub0 != 0)? VdataSub0 : Vdata0;

    const MCRegisterClass &NewRC = MRI.getRegClass(DataRCID);
    NewVdata = MRI.getMatchingSuperReg(Vdata0, AMDGPU::sub0, &NewRC);
    NewVdata = CheckVGPROverflow(NewVdata, NewRC, MRI);
    if (!NewVdata) {
      // It's possible to encode this such that the low register + enabled
      // components exceeds the register count.
      return;
    }
  }

  // If not using NSA on GFX10+, widen vaddr0 address register to correct size.
  // If using partial NSA on GFX11+ widen last address register.
  int VAddrSAIdx = IsPartialNSA ? (RsrcIdx - 1) : VAddr0Idx;
  MCRegister NewVAddrSA;
  if (STI.hasFeature(AMDGPU::FeatureNSAEncoding) && (!IsNSA || IsPartialNSA) &&
      AddrSize != Info->VAddrDwords) {
    MCRegister VAddrSA = MI.getOperand(VAddrSAIdx).getReg();
    MCRegister VAddrSubSA = MRI.getSubReg(VAddrSA, AMDGPU::sub0);
    VAddrSA = VAddrSubSA ? VAddrSubSA : VAddrSA;

    auto AddrRCID = MCII->getOpRegClassID(
        MCII->get(NewOpcode).operands()[VAddrSAIdx], HwModeRegClass);

    const MCRegisterClass &NewRC = MRI.getRegClass(AddrRCID);
    NewVAddrSA = MRI.getMatchingSuperReg(VAddrSA, AMDGPU::sub0, &NewRC);
    NewVAddrSA = CheckVGPROverflow(NewVAddrSA, NewRC, MRI);
    if (!NewVAddrSA)
      return;
  }

  MI.setOpcode(NewOpcode);

  if (NewVdata != AMDGPU::NoRegister) {
    MI.getOperand(VDataIdx) = MCOperand::createReg(NewVdata);

    if (IsAtomic) {
      // Atomic operations have an additional operand (a copy of data)
      MI.getOperand(VDstIdx) = MCOperand::createReg(NewVdata);
    }
  }

  if (NewVAddrSA) {
    MI.getOperand(VAddrSAIdx) = MCOperand::createReg(NewVAddrSA);
  } else if (IsNSA) {
    assert(AddrSize <= Info->VAddrDwords);
    MI.erase(MI.begin() + VAddr0Idx + AddrSize,
             MI.begin() + VAddr0Idx + Info->VAddrDwords);
  }
}

// Opsel and neg bits are used in src_modifiers and standalone operands. Autogen
// decoder only adds to src_modifiers, so manually add the bits to the other
// operands.
void AMDGPUDisassembler::convertVOP3PDPPInst(MCInst &MI) const {
  unsigned Opc = MI.getOpcode();
  unsigned DescNumOps = MCII->get(Opc).getNumOperands();
  auto Mods = collectVOPModifiers(MI, true);

  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::hasNamedOperand(Opc, AMDGPU::OpName::vdst_in))
    insertNamedMCOperand(MI, MCOperand::createImm(0), AMDGPU::OpName::vdst_in);

  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::hasNamedOperand(Opc, AMDGPU::OpName::op_sel))
    insertNamedMCOperand(MI, MCOperand::createImm(Mods.OpSel),
                         AMDGPU::OpName::op_sel);
  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::hasNamedOperand(Opc, AMDGPU::OpName::op_sel_hi))
    insertNamedMCOperand(MI, MCOperand::createImm(Mods.OpSelHi),
                         AMDGPU::OpName::op_sel_hi);
  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::hasNamedOperand(Opc, AMDGPU::OpName::neg_lo))
    insertNamedMCOperand(MI, MCOperand::createImm(Mods.NegLo),
                         AMDGPU::OpName::neg_lo);
  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::hasNamedOperand(Opc, AMDGPU::OpName::neg_hi))
    insertNamedMCOperand(MI, MCOperand::createImm(Mods.NegHi),
                         AMDGPU::OpName::neg_hi);
}

// Create dummy old operand and insert optional operands
void AMDGPUDisassembler::convertVOPCDPPInst(MCInst &MI) const {
  unsigned Opc = MI.getOpcode();
  unsigned DescNumOps = MCII->get(Opc).getNumOperands();

  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::hasNamedOperand(Opc, AMDGPU::OpName::old))
    insertNamedMCOperand(MI, MCOperand::createReg(0), AMDGPU::OpName::old);

  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::hasNamedOperand(Opc, AMDGPU::OpName::src0_modifiers))
    insertNamedMCOperand(MI, MCOperand::createImm(0),
                         AMDGPU::OpName::src0_modifiers);

  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::hasNamedOperand(Opc, AMDGPU::OpName::src1_modifiers))
    insertNamedMCOperand(MI, MCOperand::createImm(0),
                         AMDGPU::OpName::src1_modifiers);
}

void AMDGPUDisassembler::convertVOPC64DPPInst(MCInst &MI) const {
  unsigned Opc = MI.getOpcode();
  unsigned DescNumOps = MCII->get(Opc).getNumOperands();

  convertTrue16OpSel(MI);

  if (MI.getNumOperands() < DescNumOps &&
      AMDGPU::hasNamedOperand(Opc, AMDGPU::OpName::op_sel)) {
    VOPModifiers Mods = collectVOPModifiers(MI);
    insertNamedMCOperand(MI, MCOperand::createImm(Mods.OpSel),
                         AMDGPU::OpName::op_sel);
  }
}

void AMDGPUDisassembler::convertFMAanyK(MCInst &MI) const {
  assert(HasLiteral && "Should have decoded a literal");
  insertNamedMCOperand(MI, MCOperand::createImm(Literal), AMDGPU::OpName::immX);
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
  case AMDGPU::SGPR_96RegClassID:
  case AMDGPU::TTMP_96RegClassID:
  case AMDGPU::SGPR_128RegClassID:
  case AMDGPU::TTMP_128RegClassID:
  // ToDo: unclear if s[100:104] is available on VI. Can we use VCC as SGPR in
  // this bundle?
  case AMDGPU::SGPR_256RegClassID:
  case AMDGPU::TTMP_256RegClassID:
    // ToDo: unclear if s[96:104] is available on VI. Can we use VCC as SGPR in
  // this bundle?
  case AMDGPU::SGPR_288RegClassID:
  case AMDGPU::TTMP_288RegClassID:
  case AMDGPU::SGPR_320RegClassID:
  case AMDGPU::TTMP_320RegClassID:
  case AMDGPU::SGPR_352RegClassID:
  case AMDGPU::TTMP_352RegClassID:
  case AMDGPU::SGPR_384RegClassID:
  case AMDGPU::TTMP_384RegClassID:
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

MCOperand AMDGPUDisassembler::createVGPR16Operand(unsigned RegIdx,
                                                  bool IsHi) const {
  unsigned RegIdxInVGPR16 = RegIdx * 2 + (IsHi ? 1 : 0);
  return createRegOperand(AMDGPU::VGPR_16RegClassID, RegIdxInVGPR16);
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

MCOperand
AMDGPUDisassembler::decodeMandatoryLiteral64Constant(uint64_t Val) const {
  if (HasLiteral) {
    if (Literal64 != Val)
      return errOperand(Val, "More than one unique literal is illegal");
  }
  HasLiteral = true;
  Literal = Literal64 = Val;

  bool UseLit64 = Lo_32(Literal64) != 0;
  return UseLit64 ? MCOperand::createExpr(AMDGPUMCExpr::createLit(
                        LitModifier::Lit64, Literal64, getContext()))
                  : MCOperand::createImm(Literal64);
}

MCOperand AMDGPUDisassembler::decodeLiteralConstant(const MCInstrDesc &Desc,
                                                    const MCOperandInfo &OpDesc,
                                                    bool ExtendFP64) const {
  // For now all literal constants are supposed to be unsigned integer
  // ToDo: deal with signed/unsigned 64-bit integer constants
  // ToDo: deal with float/double constants
  if (!HasLiteral) {
    if (Bytes.size() < 4) {
      return errOperand(0, "cannot read literal, inst bytes left " +
                        Twine(Bytes.size()));
    }
    HasLiteral = true;
    Literal = Literal64 = eatBytes<uint32_t>(Bytes);
    if (ExtendFP64)
      Literal64 <<= 32;
  }

  int64_t Val = ExtendFP64 ? Literal64 : Literal;

  bool CanUse64BitLiterals =
      STI.hasFeature(AMDGPU::Feature64BitLiterals) &&
      !(Desc.TSFlags & (SIInstrFlags::VOP3 | SIInstrFlags::VOP3P));

  bool UseLit64 = false;
  if (CanUse64BitLiterals) {
    if (OpDesc.OperandType == AMDGPU::OPERAND_REG_IMM_INT64 ||
        OpDesc.OperandType == AMDGPU::OPERAND_REG_INLINE_C_INT64)
      UseLit64 = !isInt<32>(Val) || !isUInt<32>(Val);
    else if (OpDesc.OperandType == AMDGPU::OPERAND_REG_IMM_FP64 ||
             OpDesc.OperandType == AMDGPU::OPERAND_REG_INLINE_C_FP64 ||
             OpDesc.OperandType == AMDGPU::OPERAND_REG_INLINE_AC_FP64)
      UseLit64 = Lo_32(Val) != 0;
  }

  return UseLit64 ? MCOperand::createExpr(AMDGPUMCExpr::createLit(
                        LitModifier::Lit64, Val, getContext()))
                  : MCOperand::createImm(Val);
}

MCOperand
AMDGPUDisassembler::decodeLiteral64Constant(const MCInst &Inst) const {
  assert(STI.hasFeature(AMDGPU::Feature64BitLiterals));

  if (!HasLiteral) {
    if (Bytes.size() < 8) {
      return errOperand(0, "cannot read literal64, inst bytes left " +
                               Twine(Bytes.size()));
    }
    HasLiteral = true;
    Literal64 = eatBytes<uint64_t>(Bytes);
  }

  bool UseLit64 = false;
  const MCInstrDesc &Desc = MCII->get(Inst.getOpcode());
  const MCOperandInfo &OpDesc = Desc.operands()[Inst.getNumOperands()];
  if (OpDesc.OperandType == AMDGPU::OPERAND_REG_IMM_INT64 ||
      OpDesc.OperandType == AMDGPU::OPERAND_REG_INLINE_C_INT64) {
    UseLit64 = !isInt<32>(Literal64) || !isUInt<32>(Literal64);
  } else {
    assert(OpDesc.OperandType == AMDGPU::OPERAND_REG_IMM_FP64 ||
           OpDesc.OperandType == AMDGPU::OPERAND_REG_INLINE_C_FP64 ||
           OpDesc.OperandType == AMDGPU::OPERAND_REG_INLINE_AC_FP64);
    UseLit64 = Lo_32(Literal64) != 0;
  }

  return UseLit64 ? MCOperand::createExpr(AMDGPUMCExpr::createLit(
                        LitModifier::Lit64, Literal64, getContext()))
                  : MCOperand::createImm(Literal64);
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
    return llvm::bit_cast<uint32_t>(0.5f);
  case 241:
    return llvm::bit_cast<uint32_t>(-0.5f);
  case 242:
    return llvm::bit_cast<uint32_t>(1.0f);
  case 243:
    return llvm::bit_cast<uint32_t>(-1.0f);
  case 244:
    return llvm::bit_cast<uint32_t>(2.0f);
  case 245:
    return llvm::bit_cast<uint32_t>(-2.0f);
  case 246:
    return llvm::bit_cast<uint32_t>(4.0f);
  case 247:
    return llvm::bit_cast<uint32_t>(-4.0f);
  case 248: // 1 / (2 * PI)
    return 0x3e22f983;
  default:
    llvm_unreachable("invalid fp inline imm");
  }
}

static int64_t getInlineImmVal64(unsigned Imm) {
  switch (Imm) {
  case 240:
    return llvm::bit_cast<uint64_t>(0.5);
  case 241:
    return llvm::bit_cast<uint64_t>(-0.5);
  case 242:
    return llvm::bit_cast<uint64_t>(1.0);
  case 243:
    return llvm::bit_cast<uint64_t>(-1.0);
  case 244:
    return llvm::bit_cast<uint64_t>(2.0);
  case 245:
    return llvm::bit_cast<uint64_t>(-2.0);
  case 246:
    return llvm::bit_cast<uint64_t>(4.0);
  case 247:
    return llvm::bit_cast<uint64_t>(-4.0);
  case 248: // 1 / (2 * PI)
    return 0x3fc45f306dc9c882;
  default:
    llvm_unreachable("invalid fp inline imm");
  }
}

static int64_t getInlineImmValF16(unsigned Imm) {
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

static int64_t getInlineImmValBF16(unsigned Imm) {
  switch (Imm) {
  case 240:
    return 0x3F00;
  case 241:
    return 0xBF00;
  case 242:
    return 0x3F80;
  case 243:
    return 0xBF80;
  case 244:
    return 0x4000;
  case 245:
    return 0xC000;
  case 246:
    return 0x4080;
  case 247:
    return 0xC080;
  case 248: // 1 / (2 * PI)
    return 0x3E22;
  default:
    llvm_unreachable("invalid fp inline imm");
  }
}

unsigned AMDGPUDisassembler::getVgprClassId(unsigned Width) const {
  using namespace AMDGPU;

  switch (Width) {
  case 16:
  case 32:
    return VGPR_32RegClassID;
  case 64:
    return VReg_64RegClassID;
  case 96:
    return VReg_96RegClassID;
  case 128:
    return VReg_128RegClassID;
  case 160:
    return VReg_160RegClassID;
  case 192:
    return VReg_192RegClassID;
  case 256:
    return VReg_256RegClassID;
  case 288:
    return VReg_288RegClassID;
  case 320:
    return VReg_320RegClassID;
  case 352:
    return VReg_352RegClassID;
  case 384:
    return VReg_384RegClassID;
  case 512:
    return VReg_512RegClassID;
  case 1024:
    return VReg_1024RegClassID;
  }
  llvm_unreachable("Invalid register width!");
}

unsigned AMDGPUDisassembler::getAgprClassId(unsigned Width) const {
  using namespace AMDGPU;

  switch (Width) {
  case 16:
  case 32:
    return AGPR_32RegClassID;
  case 64:
    return AReg_64RegClassID;
  case 96:
    return AReg_96RegClassID;
  case 128:
    return AReg_128RegClassID;
  case 160:
    return AReg_160RegClassID;
  case 256:
    return AReg_256RegClassID;
  case 288:
    return AReg_288RegClassID;
  case 320:
    return AReg_320RegClassID;
  case 352:
    return AReg_352RegClassID;
  case 384:
    return AReg_384RegClassID;
  case 512:
    return AReg_512RegClassID;
  case 1024:
    return AReg_1024RegClassID;
  }
  llvm_unreachable("Invalid register width!");
}

unsigned AMDGPUDisassembler::getSgprClassId(unsigned Width) const {
  using namespace AMDGPU;

  switch (Width) {
  case 16:
  case 32:
    return SGPR_32RegClassID;
  case 64:
    return SGPR_64RegClassID;
  case 96:
    return SGPR_96RegClassID;
  case 128:
    return SGPR_128RegClassID;
  case 160:
    return SGPR_160RegClassID;
  case 256:
    return SGPR_256RegClassID;
  case 288:
    return SGPR_288RegClassID;
  case 320:
    return SGPR_320RegClassID;
  case 352:
    return SGPR_352RegClassID;
  case 384:
    return SGPR_384RegClassID;
  case 512:
    return SGPR_512RegClassID;
  }
  llvm_unreachable("Invalid register width!");
}

unsigned AMDGPUDisassembler::getTtmpClassId(unsigned Width) const {
  using namespace AMDGPU;

  switch (Width) {
  case 16:
  case 32:
    return TTMP_32RegClassID;
  case 64:
    return TTMP_64RegClassID;
  case 128:
    return TTMP_128RegClassID;
  case 256:
    return TTMP_256RegClassID;
  case 288:
    return TTMP_288RegClassID;
  case 320:
    return TTMP_320RegClassID;
  case 352:
    return TTMP_352RegClassID;
  case 384:
    return TTMP_384RegClassID;
  case 512:
    return TTMP_512RegClassID;
  }
  llvm_unreachable("Invalid register width!");
}

int AMDGPUDisassembler::getTTmpIdx(unsigned Val) const {
  using namespace AMDGPU::EncValues;

  unsigned TTmpMin = isGFX9Plus() ? TTMP_GFX9PLUS_MIN : TTMP_VI_MIN;
  unsigned TTmpMax = isGFX9Plus() ? TTMP_GFX9PLUS_MAX : TTMP_VI_MAX;

  return (TTmpMin <= Val && Val <= TTmpMax)? Val - TTmpMin : -1;
}

MCOperand AMDGPUDisassembler::decodeSrcOp(const MCInst &Inst, unsigned Width,
                                          unsigned Val) const {
  using namespace AMDGPU::EncValues;

  assert(Val < 1024); // enum10

  bool IsAGPR = Val & 512;
  Val &= 511;

  if (VGPR_MIN <= Val && Val <= VGPR_MAX) {
    return createRegOperand(IsAGPR ? getAgprClassId(Width)
                                   : getVgprClassId(Width), Val - VGPR_MIN);
  }
  return decodeNonVGPRSrcOp(Inst, Width, Val & 0xFF);
}

MCOperand AMDGPUDisassembler::decodeNonVGPRSrcOp(const MCInst &Inst,
                                                 unsigned Width,
                                                 unsigned Val) const {
  // Cases when Val{8} is 1 (vgpr, agpr or true 16 vgpr) should have been
  // decoded earlier.
  assert(Val < (1 << 8) && "9-bit Src encoding when Val{8} is 0");
  using namespace AMDGPU::EncValues;

  if (Val <= SGPR_MAX) {
    // "SGPR_MIN <= Val" is always true and causes compilation warning.
    static_assert(SGPR_MIN == 0);
    return createSRegOperand(getSgprClassId(Width), Val - SGPR_MIN);
  }

  int TTmpIdx = getTTmpIdx(Val);
  if (TTmpIdx >= 0) {
    return createSRegOperand(getTtmpClassId(Width), TTmpIdx);
  }

  if ((INLINE_INTEGER_C_MIN <= Val && Val <= INLINE_INTEGER_C_MAX) ||
      (INLINE_FLOATING_C_MIN <= Val && Val <= INLINE_FLOATING_C_MAX) ||
      Val == LITERAL_CONST)
    return MCOperand::createImm(Val);

  if (Val == LITERAL64_CONST && STI.hasFeature(AMDGPU::Feature64BitLiterals)) {
    return decodeLiteral64Constant(Inst);
  }

  switch (Width) {
  case 32:
  case 16:
    return decodeSpecialReg32(Val);
  case 64:
    return decodeSpecialReg64(Val);
  case 96:
  case 128:
  case 256:
  case 512:
    return decodeSpecialReg96Plus(Val);
  default:
    llvm_unreachable("unexpected immediate type");
  }
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
  return createRegOperand(getVgprClassId(32), Val);
}

MCOperand AMDGPUDisassembler::decodeSpecialReg32(unsigned Val) const {
  using namespace AMDGPU;

  switch (Val) {
  // clang-format off
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
  case 230: return createRegOperand(SRC_FLAT_SCRATCH_BASE_LO);
  case 231: return createRegOperand(SRC_FLAT_SCRATCH_BASE_HI);
  case 235: return createRegOperand(SRC_SHARED_BASE_LO);
  case 236: return createRegOperand(SRC_SHARED_LIMIT_LO);
  case 237: return createRegOperand(SRC_PRIVATE_BASE_LO);
  case 238: return createRegOperand(SRC_PRIVATE_LIMIT_LO);
  case 239: return createRegOperand(SRC_POPS_EXITING_WAVE_ID);
  case 251: return createRegOperand(SRC_VCCZ);
  case 252: return createRegOperand(SRC_EXECZ);
  case 253: return createRegOperand(SRC_SCC);
  case 254: return createRegOperand(LDS_DIRECT);
  default: break;
    // clang-format on
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
  case 230: return createRegOperand(SRC_FLAT_SCRATCH_BASE_LO);
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

MCOperand AMDGPUDisassembler::decodeSpecialReg96Plus(unsigned Val) const {
  using namespace AMDGPU;

  switch (Val) {
  case 124:
    if (isGFX11Plus())
      return createRegOperand(SGPR_NULL);
    break;
  case 125:
    if (!isGFX11Plus())
      return createRegOperand(SGPR_NULL);
    break;
  default:
    break;
  }
  return errOperand(Val, "unknown operand encoding " + Twine(Val));
}

MCOperand AMDGPUDisassembler::decodeSDWASrc(unsigned Width,
                                            const unsigned Val) const {
  using namespace AMDGPU::SDWA;
  using namespace AMDGPU::EncValues;

  if (STI.hasFeature(AMDGPU::FeatureGFX9) ||
      STI.hasFeature(AMDGPU::FeatureGFX10)) {
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

    if ((INLINE_INTEGER_C_MIN <= SVal && SVal <= INLINE_INTEGER_C_MAX) ||
        (INLINE_FLOATING_C_MIN <= SVal && SVal <= INLINE_FLOATING_C_MAX))
      return MCOperand::createImm(SVal);

    return decodeSpecialReg32(SVal);
  }
  if (STI.hasFeature(AMDGPU::FeatureVolcanicIslands))
    return createRegOperand(getVgprClassId(Width), Val);
  llvm_unreachable("unsupported target");
}

MCOperand AMDGPUDisassembler::decodeSDWASrc16(unsigned Val) const {
  return decodeSDWASrc(16, Val);
}

MCOperand AMDGPUDisassembler::decodeSDWASrc32(unsigned Val) const {
  return decodeSDWASrc(32, Val);
}

MCOperand AMDGPUDisassembler::decodeSDWAVopcDst(unsigned Val) const {
  using namespace AMDGPU::SDWA;

  assert((STI.hasFeature(AMDGPU::FeatureGFX9) ||
          STI.hasFeature(AMDGPU::FeatureGFX10)) &&
         "SDWAVopcDst should be present only on GFX9+");

  bool IsWave32 = STI.hasFeature(AMDGPU::FeatureWavefrontSize32);

  if (Val & SDWA9EncValues::VOPC_DST_VCC_MASK) {
    Val &= SDWA9EncValues::VOPC_DST_SGPR_MASK;

    int TTmpIdx = getTTmpIdx(Val);
    if (TTmpIdx >= 0) {
      auto TTmpClsId = getTtmpClassId(IsWave32 ? 32 : 64);
      return createSRegOperand(TTmpClsId, TTmpIdx);
    }
    if (Val > SGPR_MAX) {
      return IsWave32 ? decodeSpecialReg32(Val) : decodeSpecialReg64(Val);
    }
    return createSRegOperand(getSgprClassId(IsWave32 ? 32 : 64), Val);
  }
  return createRegOperand(IsWave32 ? AMDGPU::VCC_LO : AMDGPU::VCC);
}

MCOperand AMDGPUDisassembler::decodeBoolReg(const MCInst &Inst,
                                            unsigned Val) const {
  return STI.hasFeature(AMDGPU::FeatureWavefrontSize32)
             ? decodeSrcOp(Inst, 32, Val)
             : decodeSrcOp(Inst, 64, Val);
}

MCOperand AMDGPUDisassembler::decodeSplitBarrier(const MCInst &Inst,
                                                 unsigned Val) const {
  return decodeSrcOp(Inst, 32, Val);
}

MCOperand AMDGPUDisassembler::decodeDpp8FI(unsigned Val) const {
  if (Val != AMDGPU::DPP::DPP8_FI_0 && Val != AMDGPU::DPP::DPP8_FI_1)
    return MCOperand();
  return MCOperand::createImm(Val);
}

MCOperand AMDGPUDisassembler::decodeVersionImm(unsigned Imm) const {
  using VersionField = AMDGPU::EncodingField<7, 0>;
  using W64Bit = AMDGPU::EncodingBit<13>;
  using W32Bit = AMDGPU::EncodingBit<14>;
  using MDPBit = AMDGPU::EncodingBit<15>;
  using Encoding = AMDGPU::EncodingFields<VersionField, W64Bit, W32Bit, MDPBit>;

  auto [Version, W64, W32, MDP] = Encoding::decode(Imm);

  // Decode into a plain immediate if any unused bits are raised.
  if (Encoding::encode(Version, W64, W32, MDP) != Imm)
    return MCOperand::createImm(Imm);

  const auto &Versions = AMDGPU::UCVersion::getGFXVersions();
  const auto *I = find_if(
      Versions, [Version = Version](const AMDGPU::UCVersion::GFXVersion &V) {
        return V.Code == Version;
      });
  MCContext &Ctx = getContext();
  const MCExpr *E;
  if (I == Versions.end())
    E = MCConstantExpr::create(Version, Ctx);
  else
    E = MCSymbolRefExpr::create(Ctx.getOrCreateSymbol(I->Symbol), Ctx);

  if (W64)
    E = MCBinaryExpr::createOr(E, UCVersionW64Expr, Ctx);
  if (W32)
    E = MCBinaryExpr::createOr(E, UCVersionW32Expr, Ctx);
  if (MDP)
    E = MCBinaryExpr::createOr(E, UCVersionMDPExpr, Ctx);

  return MCOperand::createExpr(E);
}

bool AMDGPUDisassembler::isVI() const {
  return STI.hasFeature(AMDGPU::FeatureVolcanicIslands);
}

bool AMDGPUDisassembler::isGFX9() const { return AMDGPU::isGFX9(STI); }

bool AMDGPUDisassembler::isGFX90A() const {
  return STI.hasFeature(AMDGPU::FeatureGFX90AInsts);
}

bool AMDGPUDisassembler::isGFX9Plus() const { return AMDGPU::isGFX9Plus(STI); }

bool AMDGPUDisassembler::isGFX10() const { return AMDGPU::isGFX10(STI); }

bool AMDGPUDisassembler::isGFX10Plus() const {
  return AMDGPU::isGFX10Plus(STI);
}

bool AMDGPUDisassembler::isGFX11() const {
  return STI.hasFeature(AMDGPU::FeatureGFX11);
}

bool AMDGPUDisassembler::isGFX11Plus() const {
  return AMDGPU::isGFX11Plus(STI);
}

bool AMDGPUDisassembler::isGFX12() const {
  return STI.hasFeature(AMDGPU::FeatureGFX12);
}

bool AMDGPUDisassembler::isGFX12Plus() const {
  return AMDGPU::isGFX12Plus(STI);
}

bool AMDGPUDisassembler::isGFX1250() const { return AMDGPU::isGFX1250(STI); }

bool AMDGPUDisassembler::hasArchitectedFlatScratch() const {
  return STI.hasFeature(AMDGPU::FeatureArchitectedFlatScratch);
}

bool AMDGPUDisassembler::hasKernargPreload() const {
  return AMDGPU::hasKernargPreload(STI);
}

//===----------------------------------------------------------------------===//
// AMDGPU specific symbol handling
//===----------------------------------------------------------------------===//

/// Print a string describing the reserved bit range specified by Mask with
/// offset BaseBytes for use in error comments. Mask is a single continuous
/// range of 1s surrounded by zeros. The format here is meant to align with the
/// tables that describe these bits in llvm.org/docs/AMDGPUUsage.html.
static SmallString<32> getBitRangeFromMask(uint32_t Mask, unsigned BaseBytes) {
  SmallString<32> Result;
  raw_svector_ostream S(Result);

  int TrailingZeros = llvm::countr_zero(Mask);
  int PopCount = llvm::popcount(Mask);

  if (PopCount == 1) {
    S << "bit (" << (TrailingZeros + BaseBytes * CHAR_BIT) << ')';
  } else {
    S << "bits in range ("
      << (TrailingZeros + PopCount - 1 + BaseBytes * CHAR_BIT) << ':'
      << (TrailingZeros + BaseBytes * CHAR_BIT) << ')';
  }

  return Result;
}

#define GET_FIELD(MASK) (AMDHSA_BITS_GET(FourByteBuffer, MASK))
#define PRINT_DIRECTIVE(DIRECTIVE, MASK)                                       \
  do {                                                                         \
    KdStream << Indent << DIRECTIVE " " << GET_FIELD(MASK) << '\n';            \
  } while (0)
#define PRINT_PSEUDO_DIRECTIVE_COMMENT(DIRECTIVE, MASK)                        \
  do {                                                                         \
    KdStream << Indent << MAI.getCommentString() << ' ' << DIRECTIVE " "       \
             << GET_FIELD(MASK) << '\n';                                       \
  } while (0)

#define CHECK_RESERVED_BITS_IMPL(MASK, DESC, MSG)                              \
  do {                                                                         \
    if (FourByteBuffer & (MASK)) {                                             \
      return createStringError(std::errc::invalid_argument,                    \
                               "kernel descriptor " DESC                       \
                               " reserved %s set" MSG,                         \
                               getBitRangeFromMask((MASK), 0).c_str());        \
    }                                                                          \
  } while (0)

#define CHECK_RESERVED_BITS(MASK) CHECK_RESERVED_BITS_IMPL(MASK, #MASK, "")
#define CHECK_RESERVED_BITS_MSG(MASK, MSG)                                     \
  CHECK_RESERVED_BITS_IMPL(MASK, #MASK, ", " MSG)
#define CHECK_RESERVED_BITS_DESC(MASK, DESC)                                   \
  CHECK_RESERVED_BITS_IMPL(MASK, DESC, "")
#define CHECK_RESERVED_BITS_DESC_MSG(MASK, DESC, MSG)                          \
  CHECK_RESERVED_BITS_IMPL(MASK, DESC, ", " MSG)

// NOLINTNEXTLINE(readability-identifier-naming)
Expected<bool> AMDGPUDisassembler::decodeCOMPUTE_PGM_RSRC1(
    uint32_t FourByteBuffer, raw_string_ostream &KdStream) const {
  using namespace amdhsa;
  StringRef Indent = "\t";

  // We cannot accurately backward compute #VGPRs used from
  // GRANULATED_WORKITEM_VGPR_COUNT. But we are concerned with getting the same
  // value of GRANULATED_WORKITEM_VGPR_COUNT in the reassembled binary. So we
  // simply calculate the inverse of what the assembler does.

  uint32_t GranulatedWorkitemVGPRCount =
      GET_FIELD(COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);

  uint32_t NextFreeVGPR =
      (GranulatedWorkitemVGPRCount + 1) *
      AMDGPU::IsaInfo::getVGPREncodingGranule(&STI, EnableWavefrontSize32);

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
      GET_FIELD(COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);

  if (isGFX10Plus())
    CHECK_RESERVED_BITS_MSG(COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                            "must be zero on gfx10+");

  uint32_t NextFreeSGPR = (GranulatedWavefrontSGPRCount + 1) *
                          AMDGPU::IsaInfo::getSGPREncodingGranule(&STI);

  KdStream << Indent << ".amdhsa_reserve_vcc " << 0 << '\n';
  if (!hasArchitectedFlatScratch())
    KdStream << Indent << ".amdhsa_reserve_flat_scratch " << 0 << '\n';
  bool ReservedXnackMask = STI.hasFeature(AMDGPU::FeatureXNACK);
  assert(!ReservedXnackMask || STI.hasFeature(AMDGPU::FeatureSupportsXNACK));
  KdStream << Indent << ".amdhsa_reserve_xnack_mask " << ReservedXnackMask
           << '\n';
  KdStream << Indent << ".amdhsa_next_free_sgpr " << NextFreeSGPR << "\n";

  CHECK_RESERVED_BITS(COMPUTE_PGM_RSRC1_PRIORITY);

  PRINT_DIRECTIVE(".amdhsa_float_round_mode_32",
                  COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_32);
  PRINT_DIRECTIVE(".amdhsa_float_round_mode_16_64",
                  COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_16_64);
  PRINT_DIRECTIVE(".amdhsa_float_denorm_mode_32",
                  COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_32);
  PRINT_DIRECTIVE(".amdhsa_float_denorm_mode_16_64",
                  COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64);

  CHECK_RESERVED_BITS(COMPUTE_PGM_RSRC1_PRIV);

  if (!isGFX12Plus())
    PRINT_DIRECTIVE(".amdhsa_dx10_clamp",
                    COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_DX10_CLAMP);

  CHECK_RESERVED_BITS(COMPUTE_PGM_RSRC1_DEBUG_MODE);

  if (!isGFX12Plus())
    PRINT_DIRECTIVE(".amdhsa_ieee_mode",
                    COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_IEEE_MODE);

  CHECK_RESERVED_BITS(COMPUTE_PGM_RSRC1_BULKY);
  CHECK_RESERVED_BITS(COMPUTE_PGM_RSRC1_CDBG_USER);

  // Bits [26].
  if (isGFX9Plus()) {
    PRINT_DIRECTIVE(".amdhsa_fp16_overflow", COMPUTE_PGM_RSRC1_GFX9_PLUS_FP16_OVFL);
  } else {
    CHECK_RESERVED_BITS_DESC_MSG(COMPUTE_PGM_RSRC1_GFX6_GFX8_RESERVED0,
                                 "COMPUTE_PGM_RSRC1", "must be zero pre-gfx9");
  }

  // Bits [27].
  if (isGFX1250()) {
    PRINT_PSEUDO_DIRECTIVE_COMMENT("FLAT_SCRATCH_IS_NV",
                                   COMPUTE_PGM_RSRC1_GFX125_FLAT_SCRATCH_IS_NV);
  } else {
    CHECK_RESERVED_BITS_DESC(COMPUTE_PGM_RSRC1_GFX6_GFX120_RESERVED1,
                             "COMPUTE_PGM_RSRC1");
  }

  // Bits [28].
  CHECK_RESERVED_BITS_DESC(COMPUTE_PGM_RSRC1_RESERVED2, "COMPUTE_PGM_RSRC1");

  // Bits [29-31].
  if (isGFX10Plus()) {
    // WGP_MODE is not available on GFX1250.
    if (!isGFX1250()) {
      PRINT_DIRECTIVE(".amdhsa_workgroup_processor_mode",
                      COMPUTE_PGM_RSRC1_GFX10_PLUS_WGP_MODE);
    }
    PRINT_DIRECTIVE(".amdhsa_memory_ordered", COMPUTE_PGM_RSRC1_GFX10_PLUS_MEM_ORDERED);
    PRINT_DIRECTIVE(".amdhsa_forward_progress", COMPUTE_PGM_RSRC1_GFX10_PLUS_FWD_PROGRESS);
  } else {
    CHECK_RESERVED_BITS_DESC(COMPUTE_PGM_RSRC1_GFX6_GFX9_RESERVED3,
                             "COMPUTE_PGM_RSRC1");
  }

  if (isGFX12Plus())
    PRINT_DIRECTIVE(".amdhsa_round_robin_scheduling",
                    COMPUTE_PGM_RSRC1_GFX12_PLUS_ENABLE_WG_RR_EN);

  return true;
}

// NOLINTNEXTLINE(readability-identifier-naming)
Expected<bool> AMDGPUDisassembler::decodeCOMPUTE_PGM_RSRC2(
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

  CHECK_RESERVED_BITS(COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_ADDRESS_WATCH);
  CHECK_RESERVED_BITS(COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_MEMORY);
  CHECK_RESERVED_BITS(COMPUTE_PGM_RSRC2_GRANULATED_LDS_SIZE);

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

  CHECK_RESERVED_BITS_DESC(COMPUTE_PGM_RSRC2_RESERVED0, "COMPUTE_PGM_RSRC2");

  return true;
}

// NOLINTNEXTLINE(readability-identifier-naming)
Expected<bool> AMDGPUDisassembler::decodeCOMPUTE_PGM_RSRC3(
    uint32_t FourByteBuffer, raw_string_ostream &KdStream) const {
  using namespace amdhsa;
  StringRef Indent = "\t";
  if (isGFX90A()) {
    KdStream << Indent << ".amdhsa_accum_offset "
             << (GET_FIELD(COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET) + 1) * 4
             << '\n';

    PRINT_DIRECTIVE(".amdhsa_tg_split", COMPUTE_PGM_RSRC3_GFX90A_TG_SPLIT);

    CHECK_RESERVED_BITS_DESC_MSG(COMPUTE_PGM_RSRC3_GFX90A_RESERVED0,
                                 "COMPUTE_PGM_RSRC3", "must be zero on gfx90a");
    CHECK_RESERVED_BITS_DESC_MSG(COMPUTE_PGM_RSRC3_GFX90A_RESERVED1,
                                 "COMPUTE_PGM_RSRC3", "must be zero on gfx90a");
  } else if (isGFX10Plus()) {
    // Bits [0-3].
    if (!isGFX12Plus()) {
      if (!EnableWavefrontSize32 || !*EnableWavefrontSize32) {
        PRINT_DIRECTIVE(".amdhsa_shared_vgpr_count",
                        COMPUTE_PGM_RSRC3_GFX10_GFX11_SHARED_VGPR_COUNT);
      } else {
        PRINT_PSEUDO_DIRECTIVE_COMMENT(
            "SHARED_VGPR_COUNT",
            COMPUTE_PGM_RSRC3_GFX10_GFX11_SHARED_VGPR_COUNT);
      }
    } else {
      CHECK_RESERVED_BITS_DESC_MSG(COMPUTE_PGM_RSRC3_GFX12_PLUS_RESERVED0,
                                   "COMPUTE_PGM_RSRC3",
                                   "must be zero on gfx12+");
    }

    // Bits [4-11].
    if (isGFX11()) {
      PRINT_DIRECTIVE(".amdhsa_inst_pref_size",
                      COMPUTE_PGM_RSRC3_GFX11_INST_PREF_SIZE);
      PRINT_PSEUDO_DIRECTIVE_COMMENT("TRAP_ON_START",
                                     COMPUTE_PGM_RSRC3_GFX11_TRAP_ON_START);
      PRINT_PSEUDO_DIRECTIVE_COMMENT("TRAP_ON_END",
                                     COMPUTE_PGM_RSRC3_GFX11_TRAP_ON_END);
    } else if (isGFX12Plus()) {
      PRINT_DIRECTIVE(".amdhsa_inst_pref_size",
                      COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE);
    } else {
      CHECK_RESERVED_BITS_DESC_MSG(COMPUTE_PGM_RSRC3_GFX10_RESERVED1,
                                   "COMPUTE_PGM_RSRC3",
                                   "must be zero on gfx10");
    }

    // Bits [12].
    CHECK_RESERVED_BITS_DESC_MSG(COMPUTE_PGM_RSRC3_GFX10_PLUS_RESERVED2,
                                 "COMPUTE_PGM_RSRC3", "must be zero on gfx10+");

    // Bits [13].
    if (isGFX12Plus()) {
      PRINT_PSEUDO_DIRECTIVE_COMMENT("GLG_EN",
                                     COMPUTE_PGM_RSRC3_GFX12_PLUS_GLG_EN);
    } else {
      CHECK_RESERVED_BITS_DESC_MSG(COMPUTE_PGM_RSRC3_GFX10_GFX11_RESERVED3,
                                   "COMPUTE_PGM_RSRC3",
                                   "must be zero on gfx10 or gfx11");
    }

    // Bits [14-21].
    if (isGFX1250()) {
      PRINT_DIRECTIVE(".amdhsa_named_barrier_count",
                      COMPUTE_PGM_RSRC3_GFX125_NAMED_BAR_CNT);
      PRINT_PSEUDO_DIRECTIVE_COMMENT(
          "ENABLE_DYNAMIC_VGPR", COMPUTE_PGM_RSRC3_GFX125_ENABLE_DYNAMIC_VGPR);
      PRINT_PSEUDO_DIRECTIVE_COMMENT("TCP_SPLIT",
                                     COMPUTE_PGM_RSRC3_GFX125_TCP_SPLIT);
      PRINT_PSEUDO_DIRECTIVE_COMMENT(
          "ENABLE_DIDT_THROTTLE",
          COMPUTE_PGM_RSRC3_GFX125_ENABLE_DIDT_THROTTLE);
    } else {
      CHECK_RESERVED_BITS_DESC_MSG(COMPUTE_PGM_RSRC3_GFX10_GFX120_RESERVED4,
                                   "COMPUTE_PGM_RSRC3",
                                   "must be zero on gfx10+");
    }

    // Bits [22-30].
    CHECK_RESERVED_BITS_DESC_MSG(COMPUTE_PGM_RSRC3_GFX10_PLUS_RESERVED5,
                                 "COMPUTE_PGM_RSRC3", "must be zero on gfx10+");

    // Bits [31].
    if (isGFX11Plus()) {
      PRINT_PSEUDO_DIRECTIVE_COMMENT("IMAGE_OP",
                                     COMPUTE_PGM_RSRC3_GFX11_PLUS_IMAGE_OP);
    } else {
      CHECK_RESERVED_BITS_DESC_MSG(COMPUTE_PGM_RSRC3_GFX10_RESERVED6,
                                   "COMPUTE_PGM_RSRC3",
                                   "must be zero on gfx10");
    }
  } else if (FourByteBuffer) {
    return createStringError(
        std::errc::invalid_argument,
        "kernel descriptor COMPUTE_PGM_RSRC3 must be all zero before gfx9");
  }
  return true;
}
#undef PRINT_PSEUDO_DIRECTIVE_COMMENT
#undef PRINT_DIRECTIVE
#undef GET_FIELD
#undef CHECK_RESERVED_BITS_IMPL
#undef CHECK_RESERVED_BITS
#undef CHECK_RESERVED_BITS_MSG
#undef CHECK_RESERVED_BITS_DESC
#undef CHECK_RESERVED_BITS_DESC_MSG

/// Create an error object to return from onSymbolStart for reserved kernel
/// descriptor bits being set.
static Error createReservedKDBitsError(uint32_t Mask, unsigned BaseBytes,
                                       const char *Msg = "") {
  return createStringError(
      std::errc::invalid_argument, "kernel descriptor reserved %s set%s%s",
      getBitRangeFromMask(Mask, BaseBytes).c_str(), *Msg ? ", " : "", Msg);
}

/// Create an error object to return from onSymbolStart for reserved kernel
/// descriptor bytes being set.
static Error createReservedKDBytesError(unsigned BaseInBytes,
                                        unsigned WidthInBytes) {
  // Create an error comment in the same format as the "Kernel Descriptor"
  // table here: https://llvm.org/docs/AMDGPUUsage.html#kernel-descriptor .
  return createStringError(
      std::errc::invalid_argument,
      "kernel descriptor reserved bits in range (%u:%u) set",
      (BaseInBytes + WidthInBytes) * CHAR_BIT - 1, BaseInBytes * CHAR_BIT);
}

Expected<bool> AMDGPUDisassembler::decodeKernelDescriptorDirective(
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
    return true;

  case amdhsa::PRIVATE_SEGMENT_FIXED_SIZE_OFFSET:
    FourByteBuffer = DE.getU32(Cursor);
    KdStream << Indent << ".amdhsa_private_segment_fixed_size "
             << FourByteBuffer << '\n';
    return true;

  case amdhsa::KERNARG_SIZE_OFFSET:
    FourByteBuffer = DE.getU32(Cursor);
    KdStream << Indent << ".amdhsa_kernarg_size "
             << FourByteBuffer << '\n';
    return true;

  case amdhsa::RESERVED0_OFFSET:
    // 4 reserved bytes, must be 0.
    ReservedBytes = DE.getBytes(Cursor, 4);
    for (int I = 0; I < 4; ++I) {
      if (ReservedBytes[I] != 0)
        return createReservedKDBytesError(amdhsa::RESERVED0_OFFSET, 4);
    }
    return true;

  case amdhsa::KERNEL_CODE_ENTRY_BYTE_OFFSET_OFFSET:
    // KERNEL_CODE_ENTRY_BYTE_OFFSET
    // So far no directive controls this for Code Object V3, so simply skip for
    // disassembly.
    DE.skip(Cursor, 8);
    return true;

  case amdhsa::RESERVED1_OFFSET:
    // 20 reserved bytes, must be 0.
    ReservedBytes = DE.getBytes(Cursor, 20);
    for (int I = 0; I < 20; ++I) {
      if (ReservedBytes[I] != 0)
        return createReservedKDBytesError(amdhsa::RESERVED1_OFFSET, 20);
    }
    return true;

  case amdhsa::COMPUTE_PGM_RSRC3_OFFSET:
    FourByteBuffer = DE.getU32(Cursor);
    return decodeCOMPUTE_PGM_RSRC3(FourByteBuffer, KdStream);

  case amdhsa::COMPUTE_PGM_RSRC1_OFFSET:
    FourByteBuffer = DE.getU32(Cursor);
    return decodeCOMPUTE_PGM_RSRC1(FourByteBuffer, KdStream);

  case amdhsa::COMPUTE_PGM_RSRC2_OFFSET:
    FourByteBuffer = DE.getU32(Cursor);
    return decodeCOMPUTE_PGM_RSRC2(FourByteBuffer, KdStream);

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
      return createReservedKDBitsError(KERNEL_CODE_PROPERTY_RESERVED0,
                                       amdhsa::KERNEL_CODE_PROPERTIES_OFFSET);

    // Reserved for GFX9
    if (isGFX9() &&
        (TwoByteBuffer & KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32)) {
      return createReservedKDBitsError(
          KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32,
          amdhsa::KERNEL_CODE_PROPERTIES_OFFSET, "must be zero on gfx9");
    }
    if (isGFX10Plus()) {
      PRINT_DIRECTIVE(".amdhsa_wavefront_size32",
                      KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);
    }

    if (CodeObjectVersion >= AMDGPU::AMDHSA_COV5)
      PRINT_DIRECTIVE(".amdhsa_uses_dynamic_stack",
                      KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK);

    if (TwoByteBuffer & KERNEL_CODE_PROPERTY_RESERVED1) {
      return createReservedKDBitsError(KERNEL_CODE_PROPERTY_RESERVED1,
                                       amdhsa::KERNEL_CODE_PROPERTIES_OFFSET);
    }

    return true;

  case amdhsa::KERNARG_PRELOAD_OFFSET:
    using namespace amdhsa;
    TwoByteBuffer = DE.getU16(Cursor);
    if (TwoByteBuffer & KERNARG_PRELOAD_SPEC_LENGTH) {
      PRINT_DIRECTIVE(".amdhsa_user_sgpr_kernarg_preload_length",
                      KERNARG_PRELOAD_SPEC_LENGTH);
    }

    if (TwoByteBuffer & KERNARG_PRELOAD_SPEC_OFFSET) {
      PRINT_DIRECTIVE(".amdhsa_user_sgpr_kernarg_preload_offset",
                      KERNARG_PRELOAD_SPEC_OFFSET);
    }
    return true;

  case amdhsa::RESERVED3_OFFSET:
    // 4 bytes from here are reserved, must be 0.
    ReservedBytes = DE.getBytes(Cursor, 4);
    for (int I = 0; I < 4; ++I) {
      if (ReservedBytes[I] != 0)
        return createReservedKDBytesError(amdhsa::RESERVED3_OFFSET, 4);
    }
    return true;

  default:
    llvm_unreachable("Unhandled index. Case statements cover everything.");
    return true;
  }
#undef PRINT_DIRECTIVE
}

Expected<bool> AMDGPUDisassembler::decodeKernelDescriptor(
    StringRef KdName, ArrayRef<uint8_t> Bytes, uint64_t KdAddress) const {

  // CP microcode requires the kernel descriptor to be 64 aligned.
  if (Bytes.size() != 64 || KdAddress % 64 != 0)
    return createStringError(std::errc::invalid_argument,
                             "kernel descriptor must be 64-byte aligned");

  // FIXME: We can't actually decode "in order" as is done below, as e.g. GFX10
  // requires us to know the setting of .amdhsa_wavefront_size32 in order to
  // accurately produce .amdhsa_next_free_vgpr, and they appear in the wrong
  // order. Workaround this by first looking up .amdhsa_wavefront_size32 here
  // when required.
  if (isGFX10Plus()) {
    uint16_t KernelCodeProperties =
        support::endian::read16(&Bytes[amdhsa::KERNEL_CODE_PROPERTIES_OFFSET],
                                llvm::endianness::little);
    EnableWavefrontSize32 =
        AMDHSA_BITS_GET(KernelCodeProperties,
                        amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);
  }

  std::string Kd;
  raw_string_ostream KdStream(Kd);
  KdStream << ".amdhsa_kernel " << KdName << '\n';

  DataExtractor::Cursor C(0);
  while (C && C.tell() < Bytes.size()) {
    Expected<bool> Res = decodeKernelDescriptorDirective(C, Bytes, KdStream);

    cantFail(C.takeError());

    if (!Res)
      return Res;
  }
  KdStream << ".end_amdhsa_kernel\n";
  outs() << KdStream.str();
  return true;
}

Expected<bool> AMDGPUDisassembler::onSymbolStart(SymbolInfoTy &Symbol,
                                                 uint64_t &Size,
                                                 ArrayRef<uint8_t> Bytes,
                                                 uint64_t Address) const {
  // Right now only kernel descriptor needs to be handled.
  // We ignore all other symbols for target specific handling.
  // TODO:
  // Fix the spurious symbol issue for AMDGPU kernels. Exists for both Code
  // Object V2 and V3 when symbols are marked protected.

  // amd_kernel_code_t for Code Object V2.
  if (Symbol.Type == ELF::STT_AMDGPU_HSA_KERNEL) {
    Size = 256;
    return createStringError(std::errc::invalid_argument,
                             "code object v2 is not supported");
  }

  // Code Object V3 kernel descriptors.
  StringRef Name = Symbol.Name;
  if (Symbol.Type == ELF::STT_OBJECT && Name.ends_with(StringRef(".kd"))) {
    Size = 64; // Size = 64 regardless of success or failure.
    return decodeKernelDescriptor(Name.drop_back(3), Bytes, Address);
  }

  return false;
}

const MCExpr *AMDGPUDisassembler::createConstantSymbolExpr(StringRef Id,
                                                           int64_t Val) {
  MCContext &Ctx = getContext();
  MCSymbol *Sym = Ctx.getOrCreateSymbol(Id);
  // Note: only set value to Val on a new symbol in case an dissassembler
  // has already been initialized in this context.
  if (!Sym->isVariable()) {
    Sym->setVariableValue(MCConstantExpr::create(Val, Ctx));
  } else {
    int64_t Res = ~Val;
    bool Valid = Sym->getVariableValue()->evaluateAsAbsolute(Res);
    if (!Valid || Res != Val)
      Ctx.reportWarning(SMLoc(), "unsupported redefinition of " + Id);
  }
  return MCSymbolRefExpr::create(Sym, Ctx);
}

bool AMDGPUDisassembler::isBufferInstruction(const MCInst &MI) const {
  const uint64_t TSFlags = MCII->get(MI.getOpcode()).TSFlags;

  // Check for MUBUF and MTBUF instructions
  if (TSFlags & (SIInstrFlags::MTBUF | SIInstrFlags::MUBUF))
    return true;

  // Check for SMEM buffer instructions (S_BUFFER_* instructions)
  if ((TSFlags & SIInstrFlags::SMRD) && AMDGPU::getSMEMIsBuffer(MI.getOpcode()))
    return true;

  return false;
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

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeAMDGPUDisassembler() {
  TargetRegistry::RegisterMCDisassembler(getTheGCNTarget(),
                                         createAMDGPUDisassembler);
  TargetRegistry::RegisterMCSymbolizer(getTheGCNTarget(),
                                       createAMDGPUSymbolizer);
}
