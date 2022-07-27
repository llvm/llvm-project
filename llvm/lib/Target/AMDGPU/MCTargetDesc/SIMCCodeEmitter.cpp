//===-- SIMCCodeEmitter.cpp - SI Code Emitter -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// The SI code emitter produces machine code that can be executed
/// directly on the GPU device.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/AMDGPUFixupKinds.h"
#include "MCTargetDesc/AMDGPUMCCodeEmitter.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/APInt.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/Casting.h"

using namespace llvm;

namespace {

class SIMCCodeEmitter : public  AMDGPUMCCodeEmitter {
  const MCRegisterInfo &MRI;

  /// Encode an fp or int literal
  uint32_t getLitEncoding(const MCOperand &MO, const MCOperandInfo &OpInfo,
                          const MCSubtargetInfo &STI) const;

public:
  SIMCCodeEmitter(const MCInstrInfo &mcii, MCContext &ctx)
      : AMDGPUMCCodeEmitter(mcii), MRI(*ctx.getRegisterInfo()) {}
  SIMCCodeEmitter(const SIMCCodeEmitter &) = delete;
  SIMCCodeEmitter &operator=(const SIMCCodeEmitter &) = delete;

  /// Encode the instruction and write it to the OS.
  void encodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;

  /// \returns the encoding for an MCOperand.
  void getMachineOpValue(const MCInst &MI, const MCOperand &MO, APInt &Op,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;

  /// Use a fixup to encode the simm16 field for SOPP branch
  ///        instructions.
  void getSOPPBrEncoding(const MCInst &MI, unsigned OpNo, APInt &Op,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;

  void getSMEMOffsetEncoding(const MCInst &MI, unsigned OpNo, APInt &Op,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const override;

  void getSDWASrcEncoding(const MCInst &MI, unsigned OpNo, APInt &Op,
                          SmallVectorImpl<MCFixup> &Fixups,
                          const MCSubtargetInfo &STI) const override;

  void getSDWAVopcDstEncoding(const MCInst &MI, unsigned OpNo, APInt &Op,
                              SmallVectorImpl<MCFixup> &Fixups,
                              const MCSubtargetInfo &STI) const override;

  void getAVOperandEncoding(const MCInst &MI, unsigned OpNo, APInt &Op,
                            SmallVectorImpl<MCFixup> &Fixups,
                            const MCSubtargetInfo &STI) const override;

private:
  uint64_t getImplicitOpSelHiEncoding(int Opcode) const;
  void getMachineOpValueCommon(const MCInst &MI, const MCOperand &MO,
                               unsigned OpNo, APInt &Op,
                               SmallVectorImpl<MCFixup> &Fixups,
                               const MCSubtargetInfo &STI) const;
};

} // end anonymous namespace

MCCodeEmitter *llvm::createSIMCCodeEmitter(const MCInstrInfo &MCII,
                                           MCContext &Ctx) {
  return new SIMCCodeEmitter(MCII, Ctx);
}

// Returns the encoding value to use if the given integer is an integer inline
// immediate value, or 0 if it is not.
template <typename IntTy>
static uint32_t getIntInlineImmEncoding(IntTy Imm) {
  if (Imm >= 0 && Imm <= 64)
    return 128 + Imm;

  if (Imm >= -16 && Imm <= -1)
    return 192 + std::abs(Imm);

  return 0;
}

static uint32_t getLit16IntEncoding(uint16_t Val, const MCSubtargetInfo &STI) {
  uint16_t IntImm = getIntInlineImmEncoding(static_cast<int16_t>(Val));
  return IntImm == 0 ? 255 : IntImm;
}

static uint32_t getLit16Encoding(uint16_t Val, const MCSubtargetInfo &STI) {
  uint16_t IntImm = getIntInlineImmEncoding(static_cast<int16_t>(Val));
  if (IntImm != 0)
    return IntImm;

  if (Val == 0x3800) // 0.5
    return 240;

  if (Val == 0xB800) // -0.5
    return 241;

  if (Val == 0x3C00) // 1.0
    return 242;

  if (Val == 0xBC00) // -1.0
    return 243;

  if (Val == 0x4000) // 2.0
    return 244;

  if (Val == 0xC000) // -2.0
    return 245;

  if (Val == 0x4400) // 4.0
    return 246;

  if (Val == 0xC400) // -4.0
    return 247;

  if (Val == 0x3118 && // 1.0 / (2.0 * pi)
      STI.getFeatureBits()[AMDGPU::FeatureInv2PiInlineImm])
    return 248;

  return 255;
}

static uint32_t getLit32Encoding(uint32_t Val, const MCSubtargetInfo &STI) {
  uint32_t IntImm = getIntInlineImmEncoding(static_cast<int32_t>(Val));
  if (IntImm != 0)
    return IntImm;

  if (Val == FloatToBits(0.5f))
    return 240;

  if (Val == FloatToBits(-0.5f))
    return 241;

  if (Val == FloatToBits(1.0f))
    return 242;

  if (Val == FloatToBits(-1.0f))
    return 243;

  if (Val == FloatToBits(2.0f))
    return 244;

  if (Val == FloatToBits(-2.0f))
    return 245;

  if (Val == FloatToBits(4.0f))
    return 246;

  if (Val == FloatToBits(-4.0f))
    return 247;

  if (Val == 0x3e22f983 && // 1.0 / (2.0 * pi)
      STI.getFeatureBits()[AMDGPU::FeatureInv2PiInlineImm])
    return 248;

  return 255;
}

static uint32_t getLit64Encoding(uint64_t Val, const MCSubtargetInfo &STI) {
  uint32_t IntImm = getIntInlineImmEncoding(static_cast<int64_t>(Val));
  if (IntImm != 0)
    return IntImm;

  if (Val == DoubleToBits(0.5))
    return 240;

  if (Val == DoubleToBits(-0.5))
    return 241;

  if (Val == DoubleToBits(1.0))
    return 242;

  if (Val == DoubleToBits(-1.0))
    return 243;

  if (Val == DoubleToBits(2.0))
    return 244;

  if (Val == DoubleToBits(-2.0))
    return 245;

  if (Val == DoubleToBits(4.0))
    return 246;

  if (Val == DoubleToBits(-4.0))
    return 247;

  if (Val == 0x3fc45f306dc9c882 && // 1.0 / (2.0 * pi)
      STI.getFeatureBits()[AMDGPU::FeatureInv2PiInlineImm])
    return 248;

  return 255;
}

uint32_t SIMCCodeEmitter::getLitEncoding(const MCOperand &MO,
                                         const MCOperandInfo &OpInfo,
                                         const MCSubtargetInfo &STI) const {
  int64_t Imm;
  if (MO.isExpr()) {
    const auto *C = dyn_cast<MCConstantExpr>(MO.getExpr());
    if (!C)
      return 255;

    Imm = C->getValue();
  } else {

    assert(!MO.isDFPImm());

    if (!MO.isImm())
      return ~0;

    Imm = MO.getImm();
  }

  switch (OpInfo.OperandType) {
  case AMDGPU::OPERAND_REG_IMM_INT32:
  case AMDGPU::OPERAND_REG_IMM_FP32:
  case AMDGPU::OPERAND_REG_IMM_FP32_DEFERRED:
  case AMDGPU::OPERAND_REG_INLINE_C_INT32:
  case AMDGPU::OPERAND_REG_INLINE_C_FP32:
  case AMDGPU::OPERAND_REG_INLINE_AC_INT32:
  case AMDGPU::OPERAND_REG_INLINE_AC_FP32:
  case AMDGPU::OPERAND_REG_IMM_V2INT32:
  case AMDGPU::OPERAND_REG_IMM_V2FP32:
  case AMDGPU::OPERAND_REG_INLINE_C_V2INT32:
  case AMDGPU::OPERAND_REG_INLINE_C_V2FP32:
    return getLit32Encoding(static_cast<uint32_t>(Imm), STI);

  case AMDGPU::OPERAND_REG_IMM_INT64:
  case AMDGPU::OPERAND_REG_IMM_FP64:
  case AMDGPU::OPERAND_REG_INLINE_C_INT64:
  case AMDGPU::OPERAND_REG_INLINE_C_FP64:
  case AMDGPU::OPERAND_REG_INLINE_AC_FP64:
    return getLit64Encoding(static_cast<uint64_t>(Imm), STI);

  case AMDGPU::OPERAND_REG_IMM_INT16:
  case AMDGPU::OPERAND_REG_INLINE_C_INT16:
  case AMDGPU::OPERAND_REG_INLINE_AC_INT16:
    return getLit16IntEncoding(static_cast<uint16_t>(Imm), STI);
  case AMDGPU::OPERAND_REG_IMM_FP16:
  case AMDGPU::OPERAND_REG_IMM_FP16_DEFERRED:
  case AMDGPU::OPERAND_REG_INLINE_C_FP16:
  case AMDGPU::OPERAND_REG_INLINE_AC_FP16:
    // FIXME Is this correct? What do inline immediates do on SI for f16 src
    // which does not have f16 support?
    return getLit16Encoding(static_cast<uint16_t>(Imm), STI);
  case AMDGPU::OPERAND_REG_IMM_V2INT16:
  case AMDGPU::OPERAND_REG_IMM_V2FP16: {
    if (!isUInt<16>(Imm) && STI.getFeatureBits()[AMDGPU::FeatureVOP3Literal])
      return getLit32Encoding(static_cast<uint32_t>(Imm), STI);
    if (OpInfo.OperandType == AMDGPU::OPERAND_REG_IMM_V2FP16)
      return getLit16Encoding(static_cast<uint16_t>(Imm), STI);
    LLVM_FALLTHROUGH;
  }
  case AMDGPU::OPERAND_REG_INLINE_C_V2INT16:
  case AMDGPU::OPERAND_REG_INLINE_AC_V2INT16:
    return getLit16IntEncoding(static_cast<uint16_t>(Imm), STI);
  case AMDGPU::OPERAND_REG_INLINE_C_V2FP16:
  case AMDGPU::OPERAND_REG_INLINE_AC_V2FP16: {
    uint16_t Lo16 = static_cast<uint16_t>(Imm);
    uint32_t Encoding = getLit16Encoding(Lo16, STI);
    return Encoding;
  }
  case AMDGPU::OPERAND_KIMM32:
  case AMDGPU::OPERAND_KIMM16:
    return MO.getImm();
  default:
    llvm_unreachable("invalid operand size");
  }
}

uint64_t SIMCCodeEmitter::getImplicitOpSelHiEncoding(int Opcode) const {
  using namespace AMDGPU::VOP3PEncoding;
  using namespace AMDGPU::OpName;

  if (AMDGPU::getNamedOperandIdx(Opcode, op_sel_hi) != -1) {
    if (AMDGPU::getNamedOperandIdx(Opcode, src2) != -1)
      return 0;
    if (AMDGPU::getNamedOperandIdx(Opcode, src1) != -1)
      return OP_SEL_HI_2;
    if (AMDGPU::getNamedOperandIdx(Opcode, src0) != -1)
      return OP_SEL_HI_1 | OP_SEL_HI_2;
  }
  return OP_SEL_HI_0 | OP_SEL_HI_1 | OP_SEL_HI_2;
}

static bool isVCMPX64(const MCInstrDesc &Desc) {
  return (Desc.TSFlags & SIInstrFlags::VOP3) &&
         Desc.hasImplicitDefOfPhysReg(AMDGPU::EXEC);
}

void SIMCCodeEmitter::encodeInstruction(const MCInst &MI, raw_ostream &OS,
                                        SmallVectorImpl<MCFixup> &Fixups,
                                        const MCSubtargetInfo &STI) const {
  int Opcode = MI.getOpcode();
  APInt Encoding, Scratch;
  getBinaryCodeForInstr(MI, Fixups, Encoding, Scratch,  STI);
  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  unsigned bytes = Desc.getSize();

  // Set unused op_sel_hi bits to 1 for VOP3P and MAI instructions.
  // Note that accvgpr_read/write are MAI, have src0, but do not use op_sel.
  if ((Desc.TSFlags & SIInstrFlags::VOP3P) ||
      Opcode == AMDGPU::V_ACCVGPR_READ_B32_vi ||
      Opcode == AMDGPU::V_ACCVGPR_WRITE_B32_vi) {
    Encoding |= getImplicitOpSelHiEncoding(Opcode);
  }

  // GFX11 v_cmpx opcodes promoted to VOP3 have implied dst=EXEC.
  // Documentation requires dst to be encoded as EXEC (0x7E),
  // but it looks like the actual value encoded for dst operand
  // is ignored by HW. It was decided to define dst as "do not care"
  // in td files to allow disassembler accept any dst value.
  // However, dst is encoded as EXEC for compatibility with SP3.
  if (AMDGPU::isGFX11Plus(STI) && isVCMPX64(Desc)) {
    assert((Encoding & 0xFF) == 0);
    Encoding |= MRI.getEncodingValue(AMDGPU::EXEC_LO);
  }

  for (unsigned i = 0; i < bytes; i++) {
    OS.write((uint8_t)Encoding.extractBitsAsZExtValue(8, 8 * i));
  }

  // NSA encoding.
  if (AMDGPU::isGFX10Plus(STI) && Desc.TSFlags & SIInstrFlags::MIMG) {
    int vaddr0 = AMDGPU::getNamedOperandIdx(MI.getOpcode(),
                                            AMDGPU::OpName::vaddr0);
    int srsrc = AMDGPU::getNamedOperandIdx(MI.getOpcode(),
                                           AMDGPU::OpName::srsrc);
    assert(vaddr0 >= 0 && srsrc > vaddr0);
    unsigned NumExtraAddrs = srsrc - vaddr0 - 1;
    unsigned NumPadding = (-NumExtraAddrs) & 3;

    for (unsigned i = 0; i < NumExtraAddrs; ++i) {
      getMachineOpValue(MI, MI.getOperand(vaddr0 + 1 + i), Encoding, Fixups,
                        STI);
      OS.write((uint8_t)Encoding.getLimitedValue());
    }
    for (unsigned i = 0; i < NumPadding; ++i)
      OS.write(0);
  }

  if ((bytes > 8 && STI.getFeatureBits()[AMDGPU::FeatureVOP3Literal]) ||
      (bytes > 4 && !STI.getFeatureBits()[AMDGPU::FeatureVOP3Literal]))
    return;

  // Do not print literals from SISrc Operands for insts with mandatory literals
  int ImmLitIdx =
      AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::imm);
  if (ImmLitIdx != -1)
    return;

  // Check for additional literals
  for (unsigned i = 0, e = Desc.getNumOperands(); i < e; ++i) {

    // Check if this operand should be encoded as [SV]Src
    if (!AMDGPU::isSISrcOperand(Desc, i))
      continue;

    // Is this operand a literal immediate?
    const MCOperand &Op = MI.getOperand(i);
    if (getLitEncoding(Op, Desc.OpInfo[i], STI) != 255)
      continue;

    // Yes! Encode it
    int64_t Imm = 0;

    if (Op.isImm())
      Imm = Op.getImm();
    else if (Op.isExpr()) {
      if (const auto *C = dyn_cast<MCConstantExpr>(Op.getExpr()))
        Imm = C->getValue();

    } else if (!Op.isExpr()) // Exprs will be replaced with a fixup value.
      llvm_unreachable("Must be immediate or expr");

    for (unsigned j = 0; j < 4; j++) {
      OS.write((uint8_t) ((Imm >> (8 * j)) & 0xff));
    }

    // Only one literal value allowed
    break;
  }
}

void SIMCCodeEmitter::getSOPPBrEncoding(const MCInst &MI, unsigned OpNo,
                                        APInt &Op,
                                        SmallVectorImpl<MCFixup> &Fixups,
                                        const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);

  if (MO.isExpr()) {
    const MCExpr *Expr = MO.getExpr();
    MCFixupKind Kind = (MCFixupKind)AMDGPU::fixup_si_sopp_br;
    Fixups.push_back(MCFixup::create(0, Expr, Kind, MI.getLoc()));
    Op = APInt::getNullValue(96);
  } else {
    getMachineOpValue(MI, MO, Op, Fixups, STI);
  }
}

void SIMCCodeEmitter::getSMEMOffsetEncoding(const MCInst &MI, unsigned OpNo,
                                            APInt &Op,
                                            SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const {
  auto Offset = MI.getOperand(OpNo).getImm();
  // VI only supports 20-bit unsigned offsets.
  assert(!AMDGPU::isVI(STI) || isUInt<20>(Offset));
  Op = Offset;
}

void SIMCCodeEmitter::getSDWASrcEncoding(const MCInst &MI, unsigned OpNo,
                                         APInt &Op,
                                         SmallVectorImpl<MCFixup> &Fixups,
                                         const MCSubtargetInfo &STI) const {
  using namespace AMDGPU::SDWA;

  uint64_t RegEnc = 0;

  const MCOperand &MO = MI.getOperand(OpNo);

  if (MO.isReg()) {
    unsigned Reg = MO.getReg();
    RegEnc |= MRI.getEncodingValue(Reg);
    RegEnc &= SDWA9EncValues::SRC_VGPR_MASK;
    if (AMDGPU::isSGPR(AMDGPU::mc2PseudoReg(Reg), &MRI)) {
      RegEnc |= SDWA9EncValues::SRC_SGPR_MASK;
    }
    Op = RegEnc;
    return;
  } else {
    const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
    uint32_t Enc = getLitEncoding(MO, Desc.OpInfo[OpNo], STI);
    if (Enc != ~0U && Enc != 255) {
      Op = Enc | SDWA9EncValues::SRC_SGPR_MASK;
      return;
    }
  }

  llvm_unreachable("Unsupported operand kind");
}

void SIMCCodeEmitter::getSDWAVopcDstEncoding(const MCInst &MI, unsigned OpNo,
                                             APInt &Op,
                                             SmallVectorImpl<MCFixup> &Fixups,
                                             const MCSubtargetInfo &STI) const {
  using namespace AMDGPU::SDWA;

  uint64_t RegEnc = 0;

  const MCOperand &MO = MI.getOperand(OpNo);

  unsigned Reg = MO.getReg();
  if (Reg != AMDGPU::VCC && Reg != AMDGPU::VCC_LO) {
    RegEnc |= MRI.getEncodingValue(Reg);
    RegEnc &= SDWA9EncValues::VOPC_DST_SGPR_MASK;
    RegEnc |= SDWA9EncValues::VOPC_DST_VCC_MASK;
  }
  Op = RegEnc;
}

void SIMCCodeEmitter::getAVOperandEncoding(const MCInst &MI, unsigned OpNo,
                                           APInt &Op,
                                           SmallVectorImpl<MCFixup> &Fixups,
                                           const MCSubtargetInfo &STI) const {
  unsigned Reg = MI.getOperand(OpNo).getReg();
  uint64_t Enc = MRI.getEncodingValue(Reg);

  // VGPR and AGPR have the same encoding, but SrcA and SrcB operands of mfma
  // instructions use acc[0:1] modifier bits to distinguish. These bits are
  // encoded as a virtual 9th bit of the register for these operands.
  if (MRI.getRegClass(AMDGPU::AGPR_32RegClassID).contains(Reg) ||
      MRI.getRegClass(AMDGPU::AReg_64RegClassID).contains(Reg) ||
      MRI.getRegClass(AMDGPU::AReg_96RegClassID).contains(Reg) ||
      MRI.getRegClass(AMDGPU::AReg_128RegClassID).contains(Reg) ||
      MRI.getRegClass(AMDGPU::AReg_160RegClassID).contains(Reg) ||
      MRI.getRegClass(AMDGPU::AReg_192RegClassID).contains(Reg) ||
      MRI.getRegClass(AMDGPU::AReg_224RegClassID).contains(Reg) ||
      MRI.getRegClass(AMDGPU::AReg_256RegClassID).contains(Reg) ||
      MRI.getRegClass(AMDGPU::AReg_512RegClassID).contains(Reg) ||
      MRI.getRegClass(AMDGPU::AGPR_LO16RegClassID).contains(Reg))
    Enc |= 512;

  Op = Enc;
}

static bool needsPCRel(const MCExpr *Expr) {
  switch (Expr->getKind()) {
  case MCExpr::SymbolRef: {
    auto *SE = cast<MCSymbolRefExpr>(Expr);
    MCSymbolRefExpr::VariantKind Kind = SE->getKind();
    return Kind != MCSymbolRefExpr::VK_AMDGPU_ABS32_LO &&
           Kind != MCSymbolRefExpr::VK_AMDGPU_ABS32_HI;
  }
  case MCExpr::Binary: {
    auto *BE = cast<MCBinaryExpr>(Expr);
    if (BE->getOpcode() == MCBinaryExpr::Sub)
      return false;
    return needsPCRel(BE->getLHS()) || needsPCRel(BE->getRHS());
  }
  case MCExpr::Unary:
    return needsPCRel(cast<MCUnaryExpr>(Expr)->getSubExpr());
  case MCExpr::Target:
  case MCExpr::Constant:
    return false;
  }
  llvm_unreachable("invalid kind");
}

void SIMCCodeEmitter::getMachineOpValue(const MCInst &MI,
                                        const MCOperand &MO, APInt &Op,
                                        SmallVectorImpl<MCFixup> &Fixups,
                                        const MCSubtargetInfo &STI) const {
  if (MO.isReg()){
    Op = MRI.getEncodingValue(MO.getReg());
    return;
  }
  unsigned OpNo = &MO - MI.begin();
  getMachineOpValueCommon(MI, MO, OpNo, Op, Fixups, STI);
}

void SIMCCodeEmitter::getMachineOpValueCommon(
    const MCInst &MI, const MCOperand &MO, unsigned OpNo, APInt &Op,
    SmallVectorImpl<MCFixup> &Fixups, const MCSubtargetInfo &STI) const {

  if (MO.isExpr() && MO.getExpr()->getKind() != MCExpr::Constant) {
    // FIXME: If this is expression is PCRel or not should not depend on what
    // the expression looks like. Given that this is just a general expression,
    // it should probably be FK_Data_4 and whatever is producing
    //
    //    s_add_u32 s2, s2, (extern_const_addrspace+16
    //
    // And expecting a PCRel should instead produce
    //
    // .Ltmp1:
    //   s_add_u32 s2, s2, (extern_const_addrspace+16)-.Ltmp1
    MCFixupKind Kind;
    if (needsPCRel(MO.getExpr()))
      Kind = FK_PCRel_4;
    else
      Kind = FK_Data_4;

    const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
    uint32_t Offset = Desc.getSize();
    assert(Offset == 4 || Offset == 8);

    Fixups.push_back(MCFixup::create(Offset, MO.getExpr(), Kind, MI.getLoc()));
  }

  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  if (AMDGPU::isSISrcOperand(Desc, OpNo)) {
    uint32_t Enc = getLitEncoding(MO, Desc.OpInfo[OpNo], STI);
    if (Enc != ~0U) {
      Op = Enc;
      return;
    }
  } else if (MO.isImm()) {
    Op = MO.getImm();
    return;
  }

  llvm_unreachable("Encoding of this operand type is not supported yet.");
}

#include "AMDGPUGenMCCodeEmitter.inc"
