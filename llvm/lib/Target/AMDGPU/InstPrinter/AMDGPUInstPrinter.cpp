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
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "Utils/AMDGPUAsmUtils.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace llvm;

void AMDGPUInstPrinter::printInst(const MCInst *MI, raw_ostream &OS,
                                  StringRef Annot, const MCSubtargetInfo &STI) {
  OS.flush();
  printInstruction(MI, OS);

  printAnnotation(OS, Annot);
}

void AMDGPUInstPrinter::printU4ImmOperand(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O) {
  O << formatHex(MI->getOperand(OpNo).getImm() & 0xf);
}

void AMDGPUInstPrinter::printU8ImmOperand(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O) {
  O << formatHex(MI->getOperand(OpNo).getImm() & 0xff);
}

void AMDGPUInstPrinter::printU16ImmOperand(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O) {
  O << formatHex(MI->getOperand(OpNo).getImm() & 0xffff);
}

void AMDGPUInstPrinter::printU32ImmOperand(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O) {
  O << formatHex(MI->getOperand(OpNo).getImm() & 0xffffffff);
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

void AMDGPUInstPrinter::printNamedBit(const MCInst* MI, unsigned OpNo,
                                      raw_ostream& O, StringRef BitName) {
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
                                      raw_ostream &O) {
  uint16_t Imm = MI->getOperand(OpNo).getImm();
  if (Imm != 0) {
    O << " offset:";
    printU16ImmDecOperand(MI, OpNo, O);
  }
}

void AMDGPUInstPrinter::printOffset0(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  if (MI->getOperand(OpNo).getImm()) {
    O << " offset0:";
    printU8ImmDecOperand(MI, OpNo, O);
  }
}

void AMDGPUInstPrinter::printOffset1(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  if (MI->getOperand(OpNo).getImm()) {
    O << " offset1:";
    printU8ImmDecOperand(MI, OpNo, O);
  }
}

void AMDGPUInstPrinter::printSMRDOffset(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  printU32ImmOperand(MI, OpNo, O);
}

void AMDGPUInstPrinter::printSMRDLiteralOffset(const MCInst *MI, unsigned OpNo,
                                               raw_ostream &O) {
  printU32ImmOperand(MI, OpNo, O);
}

void AMDGPUInstPrinter::printGDS(const MCInst *MI, unsigned OpNo,
                                 raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "gds");
}

void AMDGPUInstPrinter::printGLC(const MCInst *MI, unsigned OpNo,
                                 raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "glc");
}

void AMDGPUInstPrinter::printSLC(const MCInst *MI, unsigned OpNo,
                                 raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "slc");
}

void AMDGPUInstPrinter::printTFE(const MCInst *MI, unsigned OpNo,
                                 raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "tfe");
}

void AMDGPUInstPrinter::printDMask(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O) {
  if (MI->getOperand(OpNo).getImm()) {
    O << " dmask:";
    printU16ImmOperand(MI, OpNo, O);
  }
}

void AMDGPUInstPrinter::printUNorm(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "unorm");
}

void AMDGPUInstPrinter::printDA(const MCInst *MI, unsigned OpNo,
                                raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "da");
}

void AMDGPUInstPrinter::printR128(const MCInst *MI, unsigned OpNo,
                                  raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "r128");
}

void AMDGPUInstPrinter::printLWE(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O) {
  printNamedBit(MI, OpNo, O, "lwe");
}

void AMDGPUInstPrinter::printRegOperand(unsigned reg, raw_ostream &O,
                                        const MCRegisterInfo &MRI) {
  switch (reg) {
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
  unsigned RegIdx = MRI.getEncodingValue(reg) & ((1 << 8) - 1);

  unsigned NumRegs;
  if (MRI.getRegClass(AMDGPU::VGPR_32RegClassID).contains(reg)) {
    O << 'v';
    NumRegs = 1;
  } else  if (MRI.getRegClass(AMDGPU::SGPR_32RegClassID).contains(reg)) {
    O << 's';
    NumRegs = 1;
  } else if (MRI.getRegClass(AMDGPU::VReg_64RegClassID).contains(reg)) {
    O <<'v';
    NumRegs = 2;
  } else  if (MRI.getRegClass(AMDGPU::SGPR_64RegClassID).contains(reg)) {
    O << 's';
    NumRegs = 2;
  } else if (MRI.getRegClass(AMDGPU::VReg_128RegClassID).contains(reg)) {
    O << 'v';
    NumRegs = 4;
  } else  if (MRI.getRegClass(AMDGPU::SGPR_128RegClassID).contains(reg)) {
    O << 's';
    NumRegs = 4;
  } else if (MRI.getRegClass(AMDGPU::VReg_96RegClassID).contains(reg)) {
    O << 'v';
    NumRegs = 3;
  } else if (MRI.getRegClass(AMDGPU::VReg_256RegClassID).contains(reg)) {
    O << 'v';
    NumRegs = 8;
  } else if (MRI.getRegClass(AMDGPU::SReg_256RegClassID).contains(reg)) {
    O << 's';
    NumRegs = 8;
  } else if (MRI.getRegClass(AMDGPU::VReg_512RegClassID).contains(reg)) {
    O << 'v';
    NumRegs = 16;
  } else if (MRI.getRegClass(AMDGPU::SReg_512RegClassID).contains(reg)) {
    O << 's';
    NumRegs = 16;
  } else if (MRI.getRegClass(AMDGPU::TTMP_64RegClassID).contains(reg)) {
    O << "ttmp";
    NumRegs = 2;
    RegIdx -= 112; // Trap temps start at offset 112. TODO: Get this from tablegen.
  } else if (MRI.getRegClass(AMDGPU::TTMP_128RegClassID).contains(reg)) {
    O << "ttmp";
    NumRegs = 4;
    RegIdx -= 112; // Trap temps start at offset 112. TODO: Get this from tablegen.
  } else {
    O << getRegisterName(reg);
    return;
  }

  if (NumRegs == 1) {
    O << RegIdx;
    return;
  }

  O << '[' << RegIdx << ':' << (RegIdx + NumRegs - 1) << ']';
}

void AMDGPUInstPrinter::printVOPDst(const MCInst *MI, unsigned OpNo,
                                    raw_ostream &O) {
  if (MII.get(MI->getOpcode()).TSFlags & SIInstrFlags::VOP3)
    O << "_e64 ";
  else if (MII.get(MI->getOpcode()).TSFlags & SIInstrFlags::DPP)
    O << "_dpp ";
  else if (MII.get(MI->getOpcode()).TSFlags & SIInstrFlags::SDWA)
    O << "_sdwa ";
  else
    O << "_e32 ";

  printOperand(MI, OpNo, O);
}

void AMDGPUInstPrinter::printImmediate32(uint32_t Imm, raw_ostream &O) {
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
  else
    O << formatHex(static_cast<uint64_t>(Imm));
}

void AMDGPUInstPrinter::printImmediate64(uint64_t Imm, raw_ostream &O) {
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
  else {
    assert(isUInt<32>(Imm));

    // In rare situations, we will have a 32-bit literal in a 64-bit
    // operand. This is technically allowed for the encoding of s_mov_b64.
    O << formatHex(static_cast<uint64_t>(Imm));
  }
}

void AMDGPUInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
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
    int RCID = Desc.OpInfo[OpNo].RegClass;
    if (RCID != -1) {
      const MCRegisterClass &ImmRC = MRI.getRegClass(RCID);
      if (ImmRC.getSize() == 4)
        printImmediate32(Op.getImm(), O);
      else if (ImmRC.getSize() == 8)
        printImmediate64(Op.getImm(), O);
      else
        llvm_unreachable("Invalid register class size");
    } else if (Desc.OpInfo[OpNo].OperandType == MCOI::OPERAND_IMMEDIATE) {
      printImmediate32(Op.getImm(), O);
    } else {
      // We hit this for the immediate instruction bits that don't yet have a
      // custom printer.
      // TODO: Eventually this should be unnecessary.
      O << formatDec(Op.getImm());
    }
  } else if (Op.isFPImm()) {
    // We special case 0.0 because otherwise it will be printed as an integer.
    if (Op.getFPImm() == 0.0)
      O << "0.0";
    else {
      const MCInstrDesc &Desc = MII.get(MI->getOpcode());
      const MCRegisterClass &ImmRC = MRI.getRegClass(Desc.OpInfo[OpNo].RegClass);

      if (ImmRC.getSize() == 4)
        printImmediate32(FloatToBits(Op.getFPImm()), O);
      else if (ImmRC.getSize() == 8)
        printImmediate64(DoubleToBits(Op.getFPImm()), O);
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
                                                      raw_ostream &O) {
  unsigned InputModifiers = MI->getOperand(OpNo).getImm();
  if (InputModifiers & SISrcMods::NEG)
    O << '-';
  if (InputModifiers & SISrcMods::ABS)
    O << '|';
  printOperand(MI, OpNo + 1, O);
  if (InputModifiers & SISrcMods::ABS)
    O << '|';
}

void AMDGPUInstPrinter::printOperandAndIntInputMods(const MCInst *MI,
                                                     unsigned OpNo,
                                                     raw_ostream &O) {
  unsigned InputModifiers = MI->getOperand(OpNo).getImm();
  if (InputModifiers & SISrcMods::SEXT)
    O << "sext(";
  printOperand(MI, OpNo + 1, O);
  if (InputModifiers & SISrcMods::SEXT)
    O << ')';
}


void AMDGPUInstPrinter::printDPPCtrl(const MCInst *MI, unsigned OpNo,
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
                                            raw_ostream &O) {
  O << " row_mask:";
  printU4ImmOperand(MI, OpNo, O);
}

void AMDGPUInstPrinter::printBankMask(const MCInst *MI, unsigned OpNo,
                                             raw_ostream &O) {
  O << " bank_mask:";
  printU4ImmOperand(MI, OpNo, O);
}

void AMDGPUInstPrinter::printBoundCtrl(const MCInst *MI, unsigned OpNo,
                                              raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNo).getImm();
  if (Imm) {
    O << " bound_ctrl:0"; // XXX - this syntax is used in sp3
  }
}

void AMDGPUInstPrinter::printSDWASel(const MCInst *MI, unsigned OpNo,
                                     raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNo).getImm();
  switch (Imm) {
  case 0: O << "BYTE_0"; break;
  case 1: O << "BYTE_1"; break;
  case 2: O << "BYTE_2"; break;
  case 3: O << "BYTE_3"; break;
  case 4: O << "WORD_0"; break;
  case 5: O << "WORD_1"; break;
  case 6: O << "DWORD"; break;
  default: llvm_unreachable("Invalid SDWA data select operand");
  }
}

void AMDGPUInstPrinter::printSDWADstSel(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  O << "dst_sel:";
  printSDWASel(MI, OpNo, O);
}

void AMDGPUInstPrinter::printSDWASrc0Sel(const MCInst *MI, unsigned OpNo,
                                         raw_ostream &O) {
  O << "src0_sel:";
  printSDWASel(MI, OpNo, O);
}

void AMDGPUInstPrinter::printSDWASrc1Sel(const MCInst *MI, unsigned OpNo,
                                         raw_ostream &O) {
  O << "src1_sel:";
  printSDWASel(MI, OpNo, O);
}

void AMDGPUInstPrinter::printSDWADstUnused(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O) {
  O << "dst_unused:";
  unsigned Imm = MI->getOperand(OpNo).getImm();
  switch (Imm) {
  case 0: O << "UNUSED_PAD"; break;
  case 1: O << "UNUSED_SEXT"; break;
  case 2: O << "UNUSED_PRESERVE"; break;
  default: llvm_unreachable("Invalid SDWA dest_unused operand");
  }
}

void AMDGPUInstPrinter::printInterpSlot(const MCInst *MI, unsigned OpNum,
                                        raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNum).getImm();

  if (Imm == 2) {
    O << "P0";
  } else if (Imm == 1) {
    O << "P20";
  } else if (Imm == 0) {
    O << "P10";
  } else {
    llvm_unreachable("Invalid interpolation parameter slot");
  }
}

void AMDGPUInstPrinter::printMemOperand(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  printOperand(MI, OpNo, O);
  O  << ", ";
  printOperand(MI, OpNo + 1, O);
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
                                 raw_ostream &O) {
  printIfSet(MI, OpNo, O, '|');
}

void AMDGPUInstPrinter::printClamp(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O) {
  printIfSet(MI, OpNo, O, "_SAT");
}

void AMDGPUInstPrinter::printClampSI(const MCInst *MI, unsigned OpNo,
                                     raw_ostream &O) {
  if (MI->getOperand(OpNo).getImm())
    O << " clamp";
}

void AMDGPUInstPrinter::printOModSI(const MCInst *MI, unsigned OpNo,
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
                                  raw_ostream &O) {
  printIfSet(MI, OpNo, O, "*", " ");
}

void AMDGPUInstPrinter::printNeg(const MCInst *MI, unsigned OpNo,
                                 raw_ostream &O) {
  printIfSet(MI, OpNo, O, '-');
}

void AMDGPUInstPrinter::printOMOD(const MCInst *MI, unsigned OpNo,
                                  raw_ostream &O) {
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
                                 raw_ostream &O) {
  printIfSet(MI, OpNo, O, '+');
}

void AMDGPUInstPrinter::printUpdateExecMask(const MCInst *MI, unsigned OpNo,
                                            raw_ostream &O) {
  printIfSet(MI, OpNo, O, "ExecMask,");
}

void AMDGPUInstPrinter::printUpdatePred(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  printIfSet(MI, OpNo, O, "Pred,");
}

void AMDGPUInstPrinter::printWrite(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &O) {
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
  return;
}

void AMDGPUInstPrinter::printRSel(const MCInst *MI, unsigned OpNo,
                                  raw_ostream &O) {
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
                                  raw_ostream &O) {
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
                                    raw_ostream &O) {
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
  } while (0);
  O << SImm16; // Unknown simm16 code.
}

void AMDGPUInstPrinter::printWaitFlag(const MCInst *MI, unsigned OpNo,
                                      raw_ostream &O) {
  unsigned SImm16 = MI->getOperand(OpNo).getImm();
  unsigned Vmcnt = SImm16 & 0xF;
  unsigned Expcnt = (SImm16 >> 4) & 0x7;
  unsigned Lgkmcnt = (SImm16 >> 8) & 0xF;

  bool NeedSpace = false;

  if (Vmcnt != 0xF) {
    O << "vmcnt(" << Vmcnt << ')';
    NeedSpace = true;
  }

  if (Expcnt != 0x7) {
    if (NeedSpace)
      O << ' ';
    O << "expcnt(" << Expcnt << ')';
    NeedSpace = true;
  }

  if (Lgkmcnt != 0xF) {
    if (NeedSpace)
      O << ' ';
    O << "lgkmcnt(" << Lgkmcnt << ')';
  }
}

void AMDGPUInstPrinter::printHwreg(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O) {
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
