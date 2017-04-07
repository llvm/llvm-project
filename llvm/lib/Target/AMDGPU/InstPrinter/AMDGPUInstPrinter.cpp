//===-- AMDGPUInstPrinter.cpp - AMDGPU MC Inst -> ASM ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// \file
//===----------------------------------------------------------------------===//

#include "AMDGPUInstPrinter.h"
#include "SIDefines.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "Utils/AMDGPUAsmUtils.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;
using namespace llvm::AMDGPU;

void AMDGPUInstPrinter::printInst(const MCInst *MI, raw_ostream &OS,
                                  StringRef Annot, const MCSubtargetInfo &STI) {
  OS.flush();
  printInstruction(MI, STI, OS);
  printAnnotation(OS, Annot);
}

void AMDGPUInstPrinter::printU4ImmOperand(const MCInst *MI, unsigned OpNo,
                                          const MCSubtargetInfo &STI,
                                          raw_ostream &O) {
  O << formatHex(MI->getOperand(OpNo).getImm() & 0xf);
}

void AMDGPUInstPrinter::printU8ImmOperand(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O) {
  O << formatHex(MI->getOperand(OpNo).getImm() & 0xff);
}

void AMDGPUInstPrinter::printU16ImmOperand(const MCInst *MI, unsigned OpNo,
                                           const MCSubtargetInfo &STI,
                                           raw_ostream &O) {
  // It's possible to end up with a 32-bit literal used with a 16-bit operand
  // with ignored high bits. Print as 32-bit anyway in that case.
  int64_t Imm = MI->getOperand(OpNo).getImm();
  if (isInt<16>(Imm) || isUInt<16>(Imm))
    O << formatHex(static_cast<uint64_t>(Imm & 0xffff));
  else
    printU32ImmOperand(MI, OpNo, STI, O);
}

void AMDGPUInstPrinter::printU4ImmDecOperand(const MCInst *MI, unsigned OpNo,
                                             raw_ostream &O) {
  O << formatDec(MI->getOperand(OpNo).getImm() & 0xf);
}

void AMDGPUInstPrinter::printU8ImmDecOperand(const MCInst *MI, unsigned OpNo,
                                             raw_ostream &O) {
  O << formatDec(MI->getOperand(OpNo).getImm() & 0xff);
}

void AMDGPUInstPrinter::printU16ImmDecOperand(const MCInst *MI, unsigned OpNo,
                                              raw_ostream &O) {
  O << formatDec(MI->getOperand(OpNo).getImm() & 0xffff);
}

void AMDGPUInstPrinter::printU32ImmOperand(const MCInst *MI, unsigned OpNo,
                                           const MCSubtargetInfo &STI,
                                           raw_ostream &O) {
  O << formatHex(MI->getOperand(OpNo).getImm() & 0xffffffff);
}

void AMDGPUInstPrinter::printNamedBit(const MCInst *MI, unsigned OpNo,
                                      raw_ostream &O, StringRef BitName) {
  if (MI->getOperand(OpNo).getImm()) {
    O << ' ' << BitName;
  }
}

void AMDGPUInstPrinter::printOffen(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "offen");
}

void AMDGPUInstPrinter::printIdxen(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "idxen");
}

void AMDGPUInstPrinter::printAddr64(const MCInst *MI, unsigned OpNo,
                                    raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "addr64");
}

void AMDGPUInstPrinter::printMBUFOffset(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  if (MI->getOperand(OpNo).getImm()) {
    O << " offset:";
    printU16ImmDecOperand(MI, OpNo, O);
  }
}

void AMDGPUInstPrinter::printOffset(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &STI,
                                    raw_ostream &O) {
  uint16_t Imm = MI->getOperand(OpNo).getImm();
  if (Imm != 0) {
    O << ((OpNo == 0)? "offset:" : " offset:");
    printU16ImmDecOperand(MI, OpNo, O);
  }
}

void AMDGPUInstPrinter::printOffset0(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  if (MI->getOperand(OpNo).getImm()) {
    O << " offset0:";
    printU8ImmDecOperand(MI, OpNo, O);
  }
}

void AMDGPUInstPrinter::printOffset1(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  if (MI->getOperand(OpNo).getImm()) {
    O << " offset1:";
    printU8ImmDecOperand(MI, OpNo, O);
  }
}

void AMDGPUInstPrinter::printSMRDOffset8(const MCInst *MI, unsigned OpNo,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  printU32ImmOperand(MI, OpNo, STI, O);
}

void AMDGPUInstPrinter::printSMRDOffset20(const MCInst *MI, unsigned OpNo,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  printU32ImmOperand(MI, OpNo, STI, O);
}

void AMDGPUInstPrinter::printSMRDLiteralOffset(const MCInst *MI, unsigned OpNo,
                                               const MCSubtargetInfo &STI,
                                               raw_ostream &O) {
  printU32ImmOperand(MI, OpNo, STI, O);
}

void AMDGPUInstPrinter::printGDS(const MCInst *MI, unsigned OpNo,
                                 const MCSubtargetInfo &STI, raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "gds");
}

void AMDGPUInstPrinter::printGLC(const MCInst *MI, unsigned OpNo,
                                 const MCSubtargetInfo &STI, raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "glc");
}

void AMDGPUInstPrinter::printSLC(const MCInst *MI, unsigned OpNo,
                                 const MCSubtargetInfo &STI, raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "slc");
}

void AMDGPUInstPrinter::printTFE(const MCInst *MI, unsigned OpNo,
                                 const MCSubtargetInfo &STI, raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "tfe");
}

void AMDGPUInstPrinter::printDMask(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI, raw_ostream &O) {
  if (MI->getOperand(OpNo).getImm()) {
    O << " dmask:";
    printU16ImmOperand(MI, OpNo, STI, O);
  }
}

void AMDGPUInstPrinter::printUNorm(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI, raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "unorm");
}

void AMDGPUInstPrinter::printDA(const MCInst *MI, unsigned OpNo,
                                const MCSubtargetInfo &STI, raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "da");
}

void AMDGPUInstPrinter::printR128(const MCInst *MI, unsigned OpNo,
                                  const MCSubtargetInfo &STI, raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "r128");
}

void AMDGPUInstPrinter::printLWE(const MCInst *MI, unsigned OpNo,
                                 const MCSubtargetInfo &STI, raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "lwe");
}

void AMDGPUInstPrinter::printExpCompr(const MCInst *MI, unsigned OpNo,
                                      const MCSubtargetInfo &STI,
                                      raw_ostream &O) {
  if (MI->getOperand(OpNo).getImm())
    O << " compr";
}

void AMDGPUInstPrinter::printExpVM(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI,
                                   raw_ostream &O) {
  if (MI->getOperand(OpNo).getImm())
    O << " vm";
}

void AMDGPUInstPrinter::printRegOperand(unsigned RegNo, raw_ostream &O,
                                        const MCRegisterInfo &MRI) {
  switch (RegNo) {
  case AMDGPU::VCC:
    O << "vcc";
    return;
  case AMDGPU::SCC:
    O << "scc";
    return;
  case AMDGPU::EXEC:
    O << "exec";
    return;
  case AMDGPU::M0:
    O << "m0";
    return;
  case AMDGPU::FLAT_SCR:
    O << "flat_scratch";
    return;
  case AMDGPU::VCC_LO:
    O << "vcc_lo";
    return;
  case AMDGPU::VCC_HI:
    O << "vcc_hi";
    return;
  case AMDGPU::TBA_LO:
    O << "tba_lo";
    return;
  case AMDGPU::TBA_HI:
    O << "tba_hi";
    return;
  case AMDGPU::TMA_LO:
    O << "tma_lo";
    return;
  case AMDGPU::TMA_HI:
    O << "tma_hi";
    return;
  case AMDGPU::EXEC_LO:
    O << "exec_lo";
    return;
  case AMDGPU::EXEC_HI:
    O << "exec_hi";
    return;
  case AMDGPU::FLAT_SCR_LO:
    O << "flat_scratch_lo";
    return;
  case AMDGPU::FLAT_SCR_HI:
    O << "flat_scratch_hi";
    return;
  default:
    break;
  }

  // The low 8 bits of the encoding value is the register index, for both VGPRs
  // and SGPRs.
  unsigned RegIdx = MRI.getEncodingValue(RegNo) & ((1 << 8) - 1);

  unsigned NumRegs;
  if (MRI.getRegClass(AMDGPU::VGPR_32RegClassID).contains(RegNo)) {
    O << 'v';
    NumRegs = 1;
  } else  if (MRI.getRegClass(AMDGPU::SGPR_32RegClassID).contains(RegNo)) {
    O << 's';
    NumRegs = 1;
  } else if (MRI.getRegClass(AMDGPU::VReg_64RegClassID).contains(RegNo)) {
    O <<'v';
    NumRegs = 2;
  } else  if (MRI.getRegClass(AMDGPU::SGPR_64RegClassID).contains(RegNo)) {
    O << 's';
    NumRegs = 2;
  } else if (MRI.getRegClass(AMDGPU::VReg_128RegClassID).contains(RegNo)) {
    O << 'v';
    NumRegs = 4;
  } else  if (MRI.getRegClass(AMDGPU::SGPR_128RegClassID).contains(RegNo)) {
    O << 's';
    NumRegs = 4;
  } else if (MRI.getRegClass(AMDGPU::VReg_96RegClassID).contains(RegNo)) {
    O << 'v';
    NumRegs = 3;
  } else if (MRI.getRegClass(AMDGPU::VReg_256RegClassID).contains(RegNo)) {
    O << 'v';
    NumRegs = 8;
  } else if (MRI.getRegClass(AMDGPU::SReg_256RegClassID).contains(RegNo)) {
    O << 's';
    NumRegs = 8;
  } else if (MRI.getRegClass(AMDGPU::VReg_512RegClassID).contains(RegNo)) {
    O << 'v';
    NumRegs = 16;
  } else if (MRI.getRegClass(AMDGPU::SReg_512RegClassID).contains(RegNo)) {
    O << 's';
    NumRegs = 16;
  } else if (MRI.getRegClass(AMDGPU::TTMP_64RegClassID).contains(RegNo)) {
    O << "ttmp";
    NumRegs = 2;
    // Trap temps start at offset 112. TODO: Get this from tablegen.
    RegIdx -= 112;
  } else if (MRI.getRegClass(AMDGPU::TTMP_128RegClassID).contains(RegNo)) {
    O << "ttmp";
    NumRegs = 4;
    // Trap temps start at offset 112. TODO: Get this from tablegen.
    RegIdx -= 112;
  } else {
    O << getRegisterName(RegNo);
    return;
  }

  if (NumRegs == 1) {
    O << RegIdx;
    return;
  }

  O << '[' << RegIdx << ':' << (RegIdx + NumRegs - 1) << ']';
}

void AMDGPUInstPrinter::printVOPDst(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &STI, raw_ostream &O) {
  if (MII.get(MI->getOpcode()).TSFlags & SIInstrFlags::VOP3)
    O << "_e64 ";
  else if (MII.get(MI->getOpcode()).TSFlags & SIInstrFlags::DPP)
    O << "_dpp ";
  else if (MII.get(MI->getOpcode()).TSFlags & SIInstrFlags::SDWA)
    O << "_sdwa ";
  else
    O << "_e32 ";

  printOperand(MI, OpNo, STI, O);
}

void AMDGPUInstPrinter::printImmediate16(uint32_t Imm,
                                         const MCSubtargetInfo &STI,
                                         raw_ostream &O) {
  int16_t SImm = static_cast<int16_t>(Imm);
  if (SImm >= -16 && SImm <= 64) {
    O << SImm;
    return;
  }

  if (Imm == 0x3C00)
    O<< "1.0";
  else if (Imm == 0xBC00)
    O<< "-1.0";
  else if (Imm == 0x3800)
    O<< "0.5";
  else if (Imm == 0xB800)
    O<< "-0.5";
  else if (Imm == 0x4000)
    O<< "2.0";
  else if (Imm == 0xC000)
    O<< "-2.0";
  else if (Imm == 0x4400)
    O<< "4.0";
  else if (Imm == 0xC400)
    O<< "-4.0";
  else if (Imm == 0x3118) {
    assert(STI.getFeatureBits()[AMDGPU::FeatureInv2PiInlineImm]);
    O << "0.15915494";
  } else
    O << formatHex(static_cast<uint64_t>(Imm));
}

void AMDGPUInstPrinter::printImmediateV216(uint32_t Imm,
                                           const MCSubtargetInfo &STI,
                                           raw_ostream &O) {
  uint16_t Lo16 = static_cast<uint16_t>(Imm);
  assert(Lo16 == static_cast<uint16_t>(Imm >> 16));
  printImmediate16(Lo16, STI, O);
}

void AMDGPUInstPrinter::printImmediate32(uint32_t Imm,
                                         const MCSubtargetInfo &STI,
                                         raw_ostream &O) {
  int32_t SImm = static_cast<int32_t>(Imm);
  if (SImm >= -16 && SImm <= 64) {
    O << SImm;
    return;
  }

  if (Imm == FloatToBits(0.0f))
    O << "0.0";
  else if (Imm == FloatToBits(1.0f))
    O << "1.0";
  else if (Imm == FloatToBits(-1.0f))
    O << "-1.0";
  else if (Imm == FloatToBits(0.5f))
    O << "0.5";
  else if (Imm == FloatToBits(-0.5f))
    O << "-0.5";
  else if (Imm == FloatToBits(2.0f))
    O << "2.0";
  else if (Imm == FloatToBits(-2.0f))
    O << "-2.0";
  else if (Imm == FloatToBits(4.0f))
    O << "4.0";
  else if (Imm == FloatToBits(-4.0f))
    O << "-4.0";
  else if (Imm == 0x3e22f983 &&
           STI.getFeatureBits()[AMDGPU::FeatureInv2PiInlineImm])
    O << "0.15915494";
  else
    O << formatHex(static_cast<uint64_t>(Imm));
}

void AMDGPUInstPrinter::printImmediate64(uint64_t Imm,
                                         const MCSubtargetInfo &STI,
                                         raw_ostream &O) {
  int64_t SImm = static_cast<int64_t>(Imm);
  if (SImm >= -16 && SImm <= 64) {
    O << SImm;
    return;
  }

  if (Imm == DoubleToBits(0.0))
    O << "0.0";
  else if (Imm == DoubleToBits(1.0))
    O << "1.0";
  else if (Imm == DoubleToBits(-1.0))
    O << "-1.0";
  else if (Imm == DoubleToBits(0.5))
    O << "0.5";
  else if (Imm == DoubleToBits(-0.5))
    O << "-0.5";
  else if (Imm == DoubleToBits(2.0))
    O << "2.0";
  else if (Imm == DoubleToBits(-2.0))
    O << "-2.0";
  else if (Imm == DoubleToBits(4.0))
    O << "4.0";
  else if (Imm == DoubleToBits(-4.0))
    O << "-4.0";
  else if (Imm == 0x3fc45f306dc9c882 &&
           STI.getFeatureBits()[AMDGPU::FeatureInv2PiInlineImm])
  O << "0.15915494";
  else {
    assert(isUInt<32>(Imm) || Imm == 0x3fc45f306dc9c882);

    // In rare situations, we will have a 32-bit literal in a 64-bit
    // operand. This is technically allowed for the encoding of s_mov_b64.
    O << formatHex(static_cast<uint64_t>(Imm));
  }
}

void AMDGPUInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  if (OpNo >= MI->getNumOperands()) {
    O << "/*Missing OP" << OpNo << "*/";
    return;
  }

  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    switch (Op.getReg()) {
    // This is the default predicate state, so we don't need to print it.
    case AMDGPU::PRED_SEL_OFF:
      break;

    default:
      printRegOperand(Op.getReg(), O, MRI);
      break;
    }
  } else if (Op.isImm()) {
    const MCInstrDesc &Desc = MII.get(MI->getOpcode());
    switch (Desc.OpInfo[OpNo].OperandType) {
    case AMDGPU::OPERAND_REG_IMM_INT32:
    case AMDGPU::OPERAND_REG_IMM_FP32:
    case AMDGPU::OPERAND_REG_INLINE_C_INT32:
    case AMDGPU::OPERAND_REG_INLINE_C_FP32:
    case MCOI::OPERAND_IMMEDIATE:
      printImmediate32(Op.getImm(), STI, O);
      break;
    case AMDGPU::OPERAND_REG_IMM_INT64:
    case AMDGPU::OPERAND_REG_IMM_FP64:
    case AMDGPU::OPERAND_REG_INLINE_C_INT64:
    case AMDGPU::OPERAND_REG_INLINE_C_FP64:
      printImmediate64(Op.getImm(), STI, O);
      break;
    case AMDGPU::OPERAND_REG_INLINE_C_INT16:
    case AMDGPU::OPERAND_REG_INLINE_C_FP16:
    case AMDGPU::OPERAND_REG_IMM_INT16:
    case AMDGPU::OPERAND_REG_IMM_FP16:
      printImmediate16(Op.getImm(), STI, O);
      break;
    case AMDGPU::OPERAND_REG_INLINE_C_V2FP16:
    case AMDGPU::OPERAND_REG_INLINE_C_V2INT16:
      printImmediateV216(Op.getImm(), STI, O);
      break;
    case MCOI::OPERAND_UNKNOWN:
    case MCOI::OPERAND_PCREL:
      O << formatDec(Op.getImm());
      break;
    case MCOI::OPERAND_REGISTER:
      // FIXME: This should be removed and handled somewhere else. Seems to come
      // from a disassembler bug.
      O << "/*invalid immediate*/";
      break;
    default:
      // We hit this for the immediate instruction bits that don't yet have a
      // custom printer.
      llvm_unreachable("unexpected immediate operand type");
    }
  } else if (Op.isFPImm()) {
    // We special case 0.0 because otherwise it will be printed as an integer.
    if (Op.getFPImm() == 0.0)
      O << "0.0";
    else {
      const MCInstrDesc &Desc = MII.get(MI->getOpcode());
      int RCID = Desc.OpInfo[OpNo].RegClass;
      unsigned RCBits = AMDGPU::getRegBitWidth(MRI.getRegClass(RCID));
      if (RCBits == 32)
        printImmediate32(FloatToBits(Op.getFPImm()), STI, O);
      else if (RCBits == 64)
        printImmediate64(DoubleToBits(Op.getFPImm()), STI, O);
      else
        llvm_unreachable("Invalid register class size");
    }
  } else if (Op.isExpr()) {
    const MCExpr *Exp = Op.getExpr();
    Exp->print(O, &MAI);
  } else {
    O << "/*INV_OP*/";
  }
}

void AMDGPUInstPrinter::printOperandAndFPInputMods(const MCInst *MI,
                                                   unsigned OpNo,
                                                   const MCSubtargetInfo &STI,
                                                   raw_ostream &O) {
  unsigned InputModifiers = MI->getOperand(OpNo).getImm();

  // Use 'neg(...)' instead of '-' to avoid ambiguity.
  // This is important for integer literals because
  // -1 is not the same value as neg(1).
  bool NegMnemo = false;

  if (InputModifiers & SISrcMods::NEG) {
    if (OpNo + 1 < MI->getNumOperands() &&
        (InputModifiers & SISrcMods::ABS) == 0) {
      const MCOperand &Op = MI->getOperand(OpNo + 1);
      NegMnemo = Op.isImm() || Op.isFPImm();
    }
    if (NegMnemo) {
      O << "neg(";
    } else {
      O << '-';
    }
  }

  if (InputModifiers & SISrcMods::ABS)
    O << '|';
  printOperand(MI, OpNo + 1, STI, O);
  if (InputModifiers & SISrcMods::ABS)
    O << '|';

  if (NegMnemo) {
    O << ')';
  }
}

void AMDGPUInstPrinter::printOperandAndIntInputMods(const MCInst *MI,
                                                    unsigned OpNo,
                                                    const MCSubtargetInfo &STI,
                                                    raw_ostream &O) {
  unsigned InputModifiers = MI->getOperand(OpNo).getImm();
  if (InputModifiers & SISrcMods::SEXT)
    O << "sext(";
  printOperand(MI, OpNo + 1, STI, O);
  if (InputModifiers & SISrcMods::SEXT)
    O << ')';
}

void AMDGPUInstPrinter::printDPPCtrl(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNo).getImm();
  if (Imm <= 0x0ff) {
    O << " quad_perm:[";
    O << formatDec(Imm & 0x3)         << ',';
    O << formatDec((Imm & 0xc)  >> 2) << ',';
    O << formatDec((Imm & 0x30) >> 4) << ',';
    O << formatDec((Imm & 0xc0) >> 6) << ']';
  } else if ((Imm >= 0x101) && (Imm <= 0x10f)) {
    O << " row_shl:";
    printU4ImmDecOperand(MI, OpNo, O);
  } else if ((Imm >= 0x111) && (Imm <= 0x11f)) {
    O << " row_shr:";
    printU4ImmDecOperand(MI, OpNo, O);
  } else if ((Imm >= 0x121) && (Imm <= 0x12f)) {
    O << " row_ror:";
    printU4ImmDecOperand(MI, OpNo, O);
  } else if (Imm == 0x130) {
    O << " wave_shl:1";
  } else if (Imm == 0x134) {
    O << " wave_rol:1";
  } else if (Imm == 0x138) {
    O << " wave_shr:1";
  } else if (Imm == 0x13c) {
    O << " wave_ror:1";
  } else if (Imm == 0x140) {
    O << " row_mirror";
  } else if (Imm == 0x141) {
    O << " row_half_mirror";
  } else if (Imm == 0x142) {
    O << " row_bcast:15";
  } else if (Imm == 0x143) {
    O << " row_bcast:31";
  } else {
    llvm_unreachable("Invalid dpp_ctrl value");
  }
}

void AMDGPUInstPrinter::printRowMask(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  O << " row_mask:";
  printU4ImmOperand(MI, OpNo, STI, O);
}

void AMDGPUInstPrinter::printBankMask(const MCInst *MI, unsigned OpNo,
                                      const MCSubtargetInfo &STI,
                                      raw_ostream &O) {
  O << " bank_mask:";
  printU4ImmOperand(MI, OpNo, STI, O);
}

void AMDGPUInstPrinter::printBoundCtrl(const MCInst *MI, unsigned OpNo,
                                       const MCSubtargetInfo &STI,
                                       raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNo).getImm();
  if (Imm) {
    O << " bound_ctrl:0"; // XXX - this syntax is used in sp3
  }
}

void AMDGPUInstPrinter::printSDWASel(const MCInst *MI, unsigned OpNo,
                                     raw_ostream &O) {
  using namespace llvm::AMDGPU::SDWA;

  unsigned Imm = MI->getOperand(OpNo).getImm();
  switch (Imm) {
  case SdwaSel::BYTE_0: O << "BYTE_0"; break;
  case SdwaSel::BYTE_1: O << "BYTE_1"; break;
  case SdwaSel::BYTE_2: O << "BYTE_2"; break;
  case SdwaSel::BYTE_3: O << "BYTE_3"; break;
  case SdwaSel::WORD_0: O << "WORD_0"; break;
  case SdwaSel::WORD_1: O << "WORD_1"; break;
  case SdwaSel::DWORD: O << "DWORD"; break;
  default: llvm_unreachable("Invalid SDWA data select operand");
  }
}

void AMDGPUInstPrinter::printSDWADstSel(const MCInst *MI, unsigned OpNo,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  O << "dst_sel:";
  printSDWASel(MI, OpNo, O);
}

void AMDGPUInstPrinter::printSDWASrc0Sel(const MCInst *MI, unsigned OpNo,
                                         const MCSubtargetInfo &STI,
                                         raw_ostream &O) {
  O << "src0_sel:";
  printSDWASel(MI, OpNo, O);
}

void AMDGPUInstPrinter::printSDWASrc1Sel(const MCInst *MI, unsigned OpNo,
                                         const MCSubtargetInfo &STI,
                                         raw_ostream &O) {
  O << "src1_sel:";
  printSDWASel(MI, OpNo, O);
}

void AMDGPUInstPrinter::printSDWADstUnused(const MCInst *MI, unsigned OpNo,
                                           const MCSubtargetInfo &STI,
                                           raw_ostream &O) {
  using namespace llvm::AMDGPU::SDWA;

  O << "dst_unused:";
  unsigned Imm = MI->getOperand(OpNo).getImm();
  switch (Imm) {
  case DstUnused::UNUSED_PAD: O << "UNUSED_PAD"; break;
  case DstUnused::UNUSED_SEXT: O << "UNUSED_SEXT"; break;
  case DstUnused::UNUSED_PRESERVE: O << "UNUSED_PRESERVE"; break;
  default: llvm_unreachable("Invalid SDWA dest_unused operand");
  }
}

template <unsigned N>
void AMDGPUInstPrinter::printExpSrcN(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  unsigned Opc = MI->getOpcode();
  int EnIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::en);
  unsigned En = MI->getOperand(EnIdx).getImm();

  int ComprIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::compr);

  // If compr is set, print as src0, src0, src1, src1
  if (MI->getOperand(ComprIdx).getImm()) {
    if (N == 1 || N == 2)
      --OpNo;
    else if (N == 3)
      OpNo -= 2;
  }

  if (En & (1 << N))
    printRegOperand(MI->getOperand(OpNo).getReg(), O, MRI);
  else
    O << "off";
}

void AMDGPUInstPrinter::printExpSrc0(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  printExpSrcN<0>(MI, OpNo, STI, O);
}

void AMDGPUInstPrinter::printExpSrc1(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  printExpSrcN<1>(MI, OpNo, STI, O);
}

void AMDGPUInstPrinter::printExpSrc2(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  printExpSrcN<2>(MI, OpNo, STI, O);
}

void AMDGPUInstPrinter::printExpSrc3(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  printExpSrcN<3>(MI, OpNo, STI, O);
}

void AMDGPUInstPrinter::printExpTgt(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &STI,
                                    raw_ostream &O) {
  // This is really a 6 bit field.
  uint32_t Tgt = MI->getOperand(OpNo).getImm() & ((1 << 6) - 1);

  if (Tgt <= 7)
    O << " mrt" << Tgt;
  else if (Tgt == 8)
    O << " mrtz";
  else if (Tgt == 9)
    O << " null";
  else if (Tgt >= 12 && Tgt <= 15)
    O << " pos" << Tgt - 12;
  else if (Tgt >= 32 && Tgt <= 63)
    O << " param" << Tgt - 32;
  else {
    // Reserved values 10, 11
    O << " invalid_target_" << Tgt;
  }
}

static bool allOpsDefaultValue(const int* Ops, int NumOps, int Mod) {
  int DefaultValue = (Mod == SISrcMods::OP_SEL_1);

  for (int I = 0; I < NumOps; ++I) {
    if (!!(Ops[I] & Mod) != DefaultValue)
      return false;
  }

  return true;
}

static void printPackedModifier(const MCInst *MI, StringRef Name, unsigned Mod,
                                raw_ostream &O) {
  unsigned Opc = MI->getOpcode();
  int NumOps = 0;
  int Ops[3];

  for (int OpName : { AMDGPU::OpName::src0_modifiers,
                      AMDGPU::OpName::src1_modifiers,
                      AMDGPU::OpName::src2_modifiers }) {
    int Idx = AMDGPU::getNamedOperandIdx(Opc, OpName);
    if (Idx == -1)
      break;

    Ops[NumOps++] = MI->getOperand(Idx).getImm();
  }

  if (allOpsDefaultValue(Ops, NumOps, Mod))
    return;

  O << Name;
  for (int I = 0; I < NumOps; ++I) {
    if (I != 0)
      O << ',';

    O << !!(Ops[I] & Mod);
  }

  O << ']';
}

void AMDGPUInstPrinter::printOpSel(const MCInst *MI, unsigned,
                                   const MCSubtargetInfo &STI,
                                   raw_ostream &O) {
  printPackedModifier(MI, " op_sel:[", SISrcMods::OP_SEL_0, O);
}

void AMDGPUInstPrinter::printOpSelHi(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  printPackedModifier(MI, " op_sel_hi:[", SISrcMods::OP_SEL_1, O);
}

void AMDGPUInstPrinter::printNegLo(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI,
                                   raw_ostream &O) {
  printPackedModifier(MI, " neg_lo:[", SISrcMods::NEG, O);
}

void AMDGPUInstPrinter::printNegHi(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI,
                                   raw_ostream &O) {
  printPackedModifier(MI, " neg_hi:[", SISrcMods::NEG_HI, O);
}

void AMDGPUInstPrinter::printInterpSlot(const MCInst *MI, unsigned OpNum,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNum).getImm();
  switch (Imm) {
  case 0:
    O << "p10";
    break;
  case 1:
    O << "p20";
    break;
  case 2:
    O << "p0";
    break;
  default:
    O << "invalid_param_" << Imm;
  }
}

void AMDGPUInstPrinter::printInterpAttr(const MCInst *MI, unsigned OpNum,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  unsigned Attr = MI->getOperand(OpNum).getImm();
  O << "attr" << Attr;
}

void AMDGPUInstPrinter::printInterpAttrChan(const MCInst *MI, unsigned OpNum,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  unsigned Chan = MI->getOperand(OpNum).getImm();
  O << '.' << "xyzw"[Chan & 0x3];
}

void AMDGPUInstPrinter::printVGPRIndexMode(const MCInst *MI, unsigned OpNo,
                                           const MCSubtargetInfo &STI,
                                           raw_ostream &O) {
  unsigned Val = MI->getOperand(OpNo).getImm();
  if (Val == 0) {
    O << " 0";
    return;
  }

  if (Val & VGPRIndexMode::DST_ENABLE)
    O << " dst";

  if (Val & VGPRIndexMode::SRC0_ENABLE)
    O << " src0";

  if (Val & VGPRIndexMode::SRC1_ENABLE)
    O << " src1";

  if (Val & VGPRIndexMode::SRC2_ENABLE)
    O << " src2";
}

void AMDGPUInstPrinter::printMemOperand(const MCInst *MI, unsigned OpNo,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  printOperand(MI, OpNo, STI, O);
  O  << ", ";
  printOperand(MI, OpNo + 1, STI, O);
}

void AMDGPUInstPrinter::printIfSet(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O, StringRef Asm,
                                   StringRef Default) {
  const MCOperand &Op = MI->getOperand(OpNo);
  assert(Op.isImm());
  if (Op.getImm() == 1) {
    O << Asm;
  } else {
    O << Default;
  }
}

void AMDGPUInstPrinter::printIfSet(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O, char Asm) {
  const MCOperand &Op = MI->getOperand(OpNo);
  assert(Op.isImm());
  if (Op.getImm() == 1)
    O << Asm;
}

void AMDGPUInstPrinter::printAbs(const MCInst *MI, unsigned OpNo,
                                 const MCSubtargetInfo &STI, raw_ostream &O) {
  printIfSet(MI, OpNo, O, '|');
}

void AMDGPUInstPrinter::printClamp(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI, raw_ostream &O) {
  printIfSet(MI, OpNo, O, "_SAT");
}

void AMDGPUInstPrinter::printClampSI(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  if (MI->getOperand(OpNo).getImm())
    O << " clamp";
}

void AMDGPUInstPrinter::printOModSI(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &STI,
                                    raw_ostream &O) {
  int Imm = MI->getOperand(OpNo).getImm();
  if (Imm == SIOutMods::MUL2)
    O << " mul:2";
  else if (Imm == SIOutMods::MUL4)
    O << " mul:4";
  else if (Imm == SIOutMods::DIV2)
    O << " div:2";
}

void AMDGPUInstPrinter::printLiteral(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  assert(Op.isImm() || Op.isExpr());
  if (Op.isImm()) {
    int64_t Imm = Op.getImm();
    O << Imm << '(' << BitsToFloat(Imm) << ')';
  }
  if (Op.isExpr()) {
    Op.getExpr()->print(O << '@', &MAI);
  }
}

void AMDGPUInstPrinter::printLast(const MCInst *MI, unsigned OpNo,
                                  const MCSubtargetInfo &STI, raw_ostream &O) {
  printIfSet(MI, OpNo, O, "*", " ");
}

void AMDGPUInstPrinter::printNeg(const MCInst *MI, unsigned OpNo,
                                 const MCSubtargetInfo &STI, raw_ostream &O) {
  printIfSet(MI, OpNo, O, '-');
}

void AMDGPUInstPrinter::printOMOD(const MCInst *MI, unsigned OpNo,
                                  const MCSubtargetInfo &STI, raw_ostream &O) {
  switch (MI->getOperand(OpNo).getImm()) {
  default: break;
  case 1:
    O << " * 2.0";
    break;
  case 2:
    O << " * 4.0";
    break;
  case 3:
    O << " / 2.0";
    break;
  }
}

void AMDGPUInstPrinter::printRel(const MCInst *MI, unsigned OpNo,
                                 const MCSubtargetInfo &STI, raw_ostream &O) {
  printIfSet(MI, OpNo, O, '+');
}

void AMDGPUInstPrinter::printUpdateExecMask(const MCInst *MI, unsigned OpNo,
                                            const MCSubtargetInfo &STI,
                                            raw_ostream &O) {
  printIfSet(MI, OpNo, O, "ExecMask,");
}

void AMDGPUInstPrinter::printUpdatePred(const MCInst *MI, unsigned OpNo,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  printIfSet(MI, OpNo, O, "Pred,");
}

void AMDGPUInstPrinter::printWrite(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI, raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.getImm() == 0) {
    O << " (MASKED)";
  }
}

void AMDGPUInstPrinter::printSel(const MCInst *MI, unsigned OpNo,
                                 raw_ostream &O) {
  const char * chans = "XYZW";
  int sel = MI->getOperand(OpNo).getImm();

  int chan = sel & 3;
  sel >>= 2;

  if (sel >= 512) {
    sel -= 512;
    int cb = sel >> 12;
    sel &= 4095;
    O << cb << '[' << sel << ']';
  } else if (sel >= 448) {
    sel -= 448;
    O << sel;
  } else if (sel >= 0){
    O << sel;
  }

  if (sel >= 0)
    O << '.' << chans[chan];
}

void AMDGPUInstPrinter::printBankSwizzle(const MCInst *MI, unsigned OpNo,
                                         const MCSubtargetInfo &STI,
                                         raw_ostream &O) {
  int BankSwizzle = MI->getOperand(OpNo).getImm();
  switch (BankSwizzle) {
  case 1:
    O << "BS:VEC_021/SCL_122";
    break;
  case 2:
    O << "BS:VEC_120/SCL_212";
    break;
  case 3:
    O << "BS:VEC_102/SCL_221";
    break;
  case 4:
    O << "BS:VEC_201";
    break;
  case 5:
    O << "BS:VEC_210";
    break;
  default:
    break;
  }
}

void AMDGPUInstPrinter::printRSel(const MCInst *MI, unsigned OpNo,
                                  const MCSubtargetInfo &STI, raw_ostream &O) {
  unsigned Sel = MI->getOperand(OpNo).getImm();
  switch (Sel) {
  case 0:
    O << 'X';
    break;
  case 1:
    O << 'Y';
    break;
  case 2:
    O << 'Z';
    break;
  case 3:
    O << 'W';
    break;
  case 4:
    O << '0';
    break;
  case 5:
    O << '1';
    break;
  case 7:
    O << '_';
    break;
  default:
    break;
  }
}

void AMDGPUInstPrinter::printCT(const MCInst *MI, unsigned OpNo,
                                const MCSubtargetInfo &STI, raw_ostream &O) {
  unsigned CT = MI->getOperand(OpNo).getImm();
  switch (CT) {
  case 0:
    O << 'U';
    break;
  case 1:
    O << 'N';
    break;
  default:
    break;
  }
}

void AMDGPUInstPrinter::printKCache(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &STI, raw_ostream &O) {
  int KCacheMode = MI->getOperand(OpNo).getImm();
  if (KCacheMode > 0) {
    int KCacheBank = MI->getOperand(OpNo - 2).getImm();
    O << "CB" << KCacheBank << ':';
    int KCacheAddr = MI->getOperand(OpNo + 2).getImm();
    int LineSize = (KCacheMode == 1) ? 16 : 32;
    O << KCacheAddr * 16 << '-' << KCacheAddr * 16 + LineSize;
  }
}

void AMDGPUInstPrinter::printSendMsg(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  using namespace llvm::AMDGPU::SendMsg;

  const unsigned SImm16 = MI->getOperand(OpNo).getImm();
  const unsigned Id = SImm16 & ID_MASK_;
  do {
    if (Id == ID_INTERRUPT) {
      if ((SImm16 & ~ID_MASK_) != 0) // Unused/unknown bits must be 0.
        break;
      O << "sendmsg(" << IdSymbolic[Id] << ')';
      return;
    }
    if (Id == ID_GS || Id == ID_GS_DONE) {
      if ((SImm16 & ~(ID_MASK_|OP_GS_MASK_|STREAM_ID_MASK_)) != 0) // Unused/unknown bits must be 0.
        break;
      const unsigned OpGs = (SImm16 & OP_GS_MASK_) >> OP_SHIFT_;
      const unsigned StreamId = (SImm16 & STREAM_ID_MASK_) >> STREAM_ID_SHIFT_;
      if (OpGs == OP_GS_NOP && Id != ID_GS_DONE) // NOP to be used for GS_DONE only.
        break;
      if (OpGs == OP_GS_NOP && StreamId != 0) // NOP does not use/define stream id bits.
        break;
      O << "sendmsg(" << IdSymbolic[Id] << ", " << OpGsSymbolic[OpGs];
      if (OpGs != OP_GS_NOP) {  O << ", " << StreamId; }
      O << ')';
      return;
    }
    if (Id == ID_SYSMSG) {
      if ((SImm16 & ~(ID_MASK_|OP_SYS_MASK_)) != 0) // Unused/unknown bits must be 0.
        break;
      const unsigned OpSys = (SImm16 & OP_SYS_MASK_) >> OP_SHIFT_;
      if (! (OP_SYS_FIRST_ <= OpSys && OpSys < OP_SYS_LAST_)) // Unused/unknown.
        break;
      O << "sendmsg(" << IdSymbolic[Id] << ", " << OpSysSymbolic[OpSys] << ')';
      return;
    }
  } while (false);
  O << SImm16; // Unknown simm16 code.
}

void AMDGPUInstPrinter::printWaitFlag(const MCInst *MI, unsigned OpNo,
                                      const MCSubtargetInfo &STI,
                                      raw_ostream &O) {
  AMDGPU::IsaInfo::IsaVersion ISA =
      AMDGPU::IsaInfo::getIsaVersion(STI.getFeatureBits());

  unsigned SImm16 = MI->getOperand(OpNo).getImm();
  unsigned Vmcnt, Expcnt, Lgkmcnt;
  decodeWaitcnt(ISA, SImm16, Vmcnt, Expcnt, Lgkmcnt);

  bool NeedSpace = false;

  if (Vmcnt != getVmcntBitMask(ISA)) {
    O << "vmcnt(" << Vmcnt << ')';
    NeedSpace = true;
  }

  if (Expcnt != getExpcntBitMask(ISA)) {
    if (NeedSpace)
      O << ' ';
    O << "expcnt(" << Expcnt << ')';
    NeedSpace = true;
  }

  if (Lgkmcnt != getLgkmcntBitMask(ISA)) {
    if (NeedSpace)
      O << ' ';
    O << "lgkmcnt(" << Lgkmcnt << ')';
  }
}

void AMDGPUInstPrinter::printHwreg(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI, raw_ostream &O) {
  using namespace llvm::AMDGPU::Hwreg;

  unsigned SImm16 = MI->getOperand(OpNo).getImm();
  const unsigned Id = (SImm16 & ID_MASK_) >> ID_SHIFT_;
  const unsigned Offset = (SImm16 & OFFSET_MASK_) >> OFFSET_SHIFT_;
  const unsigned Width = ((SImm16 & WIDTH_M1_MASK_) >> WIDTH_M1_SHIFT_) + 1;

  O << "hwreg(";
  if (ID_SYMBOLIC_FIRST_ <= Id && Id < ID_SYMBOLIC_LAST_) {
    O << IdSymbolic[Id];
  } else {
    O << Id;
  }
  if (Width != WIDTH_M1_DEFAULT_ + 1 || Offset != OFFSET_DEFAULT_) {
    O << ", " << Offset << ", " << Width;
  }
  O << ')';
}

#include "AMDGPUGenAsmWriter.inc"
