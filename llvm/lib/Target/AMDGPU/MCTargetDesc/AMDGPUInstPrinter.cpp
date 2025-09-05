//===-- AMDGPUInstPrinter.cpp - AMDGPU MC Inst -> ASM ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// \file
//===----------------------------------------------------------------------===//

#include "AMDGPUInstPrinter.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"
#include "Utils/AMDGPUAsmUtils.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/TargetParser/TargetParser.h"

using namespace llvm;
using namespace llvm::AMDGPU;

void AMDGPUInstPrinter::printRegName(raw_ostream &OS, MCRegister Reg) {
  // FIXME: The current implementation of
  // AsmParser::parseRegisterOrRegisterNumber in MC implies we either emit this
  // as an integer or we provide a name which represents a physical register.
  // For CFI instructions we really want to emit a name for the DWARF register
  // instead, because there may be multiple DWARF registers corresponding to a
  // single physical register. One case where this problem manifests is with
  // wave32/wave64 where using the physical register name is ambiguous: if we
  // write e.g. `.cfi_undefined v0` we lose information about the wavefront
  // size which we need to encode the register in the final DWARF. Ideally we
  // would extend MC to support parsing DWARF register names so we could do
  // something like `.cfi_undefined dwarf_wave32_v0`. For now we just live with
  // non-pretty DWARF register names in assembly text.
  OS << Reg.id();
}

void AMDGPUInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                  StringRef Annot, const MCSubtargetInfo &STI,
                                  raw_ostream &OS) {
  printInstruction(MI, Address, STI, OS);
  printAnnotation(OS, Annot);
}

void AMDGPUInstPrinter::printU16ImmOperand(const MCInst *MI, unsigned OpNo,
                                           const MCSubtargetInfo &STI,
                                           raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isExpr()) {
    MAI.printExpr(O, *Op.getExpr());
    return;
  }

  // It's possible to end up with a 32-bit literal used with a 16-bit operand
  // with ignored high bits. Print as 32-bit anyway in that case.
  int64_t Imm = Op.getImm();
  if (isInt<16>(Imm) || isUInt<16>(Imm))
    O << formatHex(static_cast<uint64_t>(Imm & 0xffff));
  else
    printU32ImmOperand(MI, OpNo, STI, O);
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

void AMDGPUInstPrinter::printFP64ImmOperand(const MCInst *MI, unsigned OpNo,
                                            const MCSubtargetInfo &STI,
                                            raw_ostream &O) {
  // KIMM64
  const MCInstrDesc &Desc = MII.get(MI->getOpcode());
  uint64_t Imm = MI->getOperand(OpNo).getImm();
  printLiteral64(Desc, Imm, STI, O, /*IsFP=*/true);
}

void AMDGPUInstPrinter::printNamedBit(const MCInst *MI, unsigned OpNo,
                                      raw_ostream &O, StringRef BitName) {
  if (MI->getOperand(OpNo).getImm()) {
    O << ' ' << BitName;
  }
}

void AMDGPUInstPrinter::printOffset(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &STI,
                                    raw_ostream &O) {
  uint32_t Imm = MI->getOperand(OpNo).getImm();
  if (Imm != 0) {
    O << " offset:";

    // GFX12 uses a 24-bit signed offset for VBUFFER.
    const MCInstrDesc &Desc = MII.get(MI->getOpcode());
    bool IsVBuffer = Desc.TSFlags & (SIInstrFlags::MUBUF | SIInstrFlags::MTBUF);
    if (AMDGPU::isGFX12(STI) && IsVBuffer)
      O << formatDec(SignExtend32<24>(Imm));
    else
      printU16ImmDecOperand(MI, OpNo, O);
  }
}

void AMDGPUInstPrinter::printFlatOffset(const MCInst *MI, unsigned OpNo,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  uint32_t Imm = MI->getOperand(OpNo).getImm();
  if (Imm != 0) {
    O << " offset:";

    const MCInstrDesc &Desc = MII.get(MI->getOpcode());
    bool AllowNegative = (Desc.TSFlags & (SIInstrFlags::FlatGlobal |
                                          SIInstrFlags::FlatScratch)) ||
                         AMDGPU::isGFX12(STI);

    if (AllowNegative) // Signed offset
      O << formatDec(SignExtend32(Imm, AMDGPU::getNumFlatOffsetBits(STI)));
    else // Unsigned offset
      printU16ImmDecOperand(MI, OpNo, O);
  }
}

void AMDGPUInstPrinter::printSMRDOffset8(const MCInst *MI, unsigned OpNo,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  printU32ImmOperand(MI, OpNo, STI, O);
}

void AMDGPUInstPrinter::printSMEMOffset(const MCInst *MI, unsigned OpNo,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  O << formatHex(MI->getOperand(OpNo).getImm());
}

void AMDGPUInstPrinter::printSMRDLiteralOffset(const MCInst *MI, unsigned OpNo,
                                               const MCSubtargetInfo &STI,
                                               raw_ostream &O) {
  printU32ImmOperand(MI, OpNo, STI, O);
}

void AMDGPUInstPrinter::printCPol(const MCInst *MI, unsigned OpNo,
                                  const MCSubtargetInfo &STI, raw_ostream &O) {
  auto Imm = MI->getOperand(OpNo).getImm();

  if (AMDGPU::isGFX12Plus(STI)) {
    const int64_t TH = Imm & CPol::TH;
    const int64_t Scope = Imm & CPol::SCOPE;

    if (Imm & CPol::SCAL)
      O << " scale_offset";

    printTH(MI, TH, Scope, O);
    printScope(Scope, O);

    if (Imm & CPol::NV)
      O << " nv";

    return;
  }

  if (Imm & CPol::GLC)
    O << ((AMDGPU::isGFX940(STI) &&
           !(MII.get(MI->getOpcode()).TSFlags & SIInstrFlags::SMRD)) ? " sc0"
                                                                     : " glc");
  if (Imm & CPol::SLC)
    O << (AMDGPU::isGFX940(STI) ? " nt" : " slc");
  if ((Imm & CPol::DLC) && AMDGPU::isGFX10Plus(STI))
    O << " dlc";
  if ((Imm & CPol::SCC) && AMDGPU::isGFX90A(STI))
    O << (AMDGPU::isGFX940(STI) ? " sc1" : " scc");
  if (Imm & ~CPol::ALL_pregfx12)
    O << " /* unexpected cache policy bit */";
}

void AMDGPUInstPrinter::printTH(const MCInst *MI, int64_t TH, int64_t Scope,
                                raw_ostream &O) {
  // For th = 0 do not print this field
  if (TH == 0)
    return;

  const unsigned Opcode = MI->getOpcode();
  const MCInstrDesc &TID = MII.get(Opcode);
  unsigned THType = AMDGPU::getTemporalHintType(TID);
  bool IsStore = (THType == AMDGPU::CPol::TH_TYPE_STORE);

  O << " th:";

  if (THType == AMDGPU::CPol::TH_TYPE_ATOMIC) {
    O << "TH_ATOMIC_";
    if (TH & AMDGPU::CPol::TH_ATOMIC_CASCADE) {
      if (Scope >= AMDGPU::CPol::SCOPE_DEV)
        O << "CASCADE" << (TH & AMDGPU::CPol::TH_ATOMIC_NT ? "_NT" : "_RT");
      else
        O << formatHex(TH);
    } else if (TH & AMDGPU::CPol::TH_ATOMIC_NT)
      O << "NT" << (TH & AMDGPU::CPol::TH_ATOMIC_RETURN ? "_RETURN" : "");
    else if (TH & AMDGPU::CPol::TH_ATOMIC_RETURN)
      O << "RETURN";
    else
      O << formatHex(TH);
  } else {
    if (!IsStore && TH == AMDGPU::CPol::TH_RESERVED)
      O << formatHex(TH);
    else {
      O << (IsStore ? "TH_STORE_" : "TH_LOAD_");
      switch (TH) {
      case AMDGPU::CPol::TH_NT:
        O << "NT";
        break;
      case AMDGPU::CPol::TH_HT:
        O << "HT";
        break;
      case AMDGPU::CPol::TH_BYPASS: // or LU or WB
        O << (Scope == AMDGPU::CPol::SCOPE_SYS ? "BYPASS"
                                               : (IsStore ? "WB" : "LU"));
        break;
      case AMDGPU::CPol::TH_NT_RT:
        O << "NT_RT";
        break;
      case AMDGPU::CPol::TH_RT_NT:
        O << "RT_NT";
        break;
      case AMDGPU::CPol::TH_NT_HT:
        O << "NT_HT";
        break;
      case AMDGPU::CPol::TH_NT_WB:
        O << "NT_WB";
        break;
      default:
        llvm_unreachable("unexpected th value");
      }
    }
  }
}

void AMDGPUInstPrinter::printScope(int64_t Scope, raw_ostream &O) {
  if (Scope == CPol::SCOPE_CU)
    return;

  O << " scope:";

  if (Scope == CPol::SCOPE_SE)
    O << "SCOPE_SE";
  else if (Scope == CPol::SCOPE_DEV)
    O << "SCOPE_DEV";
  else if (Scope == CPol::SCOPE_SYS)
    O << "SCOPE_SYS";
  else
    llvm_unreachable("unexpected scope policy value");
}

void AMDGPUInstPrinter::printDim(const MCInst *MI, unsigned OpNo,
                                 const MCSubtargetInfo &STI, raw_ostream &O) {
  unsigned Dim = MI->getOperand(OpNo).getImm();
  O << " dim:SQ_RSRC_IMG_";

  const AMDGPU::MIMGDimInfo *DimInfo = AMDGPU::getMIMGDimInfoByEncoding(Dim);
  if (DimInfo)
    O << DimInfo->AsmSuffix;
  else
    O << Dim;
}

void AMDGPUInstPrinter::printR128A16(const MCInst *MI, unsigned OpNo,
                                  const MCSubtargetInfo &STI, raw_ostream &O) {
  if (STI.hasFeature(AMDGPU::FeatureR128A16))
    printNamedBit(MI, OpNo, O, "a16");
  else
    printNamedBit(MI, OpNo, O, "r128");
}

void AMDGPUInstPrinter::printFORMAT(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &STI,
                                    raw_ostream &O) {
}

void AMDGPUInstPrinter::printSymbolicFormat(const MCInst *MI,
                                            const MCSubtargetInfo &STI,
                                            raw_ostream &O) {
  using namespace llvm::AMDGPU::MTBUFFormat;

  int OpNo =
    AMDGPU::getNamedOperandIdx(MI->getOpcode(), AMDGPU::OpName::format);
  assert(OpNo != -1);

  unsigned Val = MI->getOperand(OpNo).getImm();
  if (AMDGPU::isGFX10Plus(STI)) {
    if (Val == UFMT_DEFAULT)
      return;
    if (isValidUnifiedFormat(Val, STI)) {
      O << " format:[" << getUnifiedFormatName(Val, STI) << ']';
    } else {
      O << " format:" << Val;
    }
  } else {
    if (Val == DFMT_NFMT_DEFAULT)
      return;
    if (isValidDfmtNfmt(Val, STI)) {
      unsigned Dfmt;
      unsigned Nfmt;
      decodeDfmtNfmt(Val, Dfmt, Nfmt);
      O << " format:[";
      if (Dfmt != DFMT_DEFAULT) {
        O << getDfmtName(Dfmt);
        if (Nfmt != NFMT_DEFAULT) {
          O << ',';
        }
      }
      if (Nfmt != NFMT_DEFAULT) {
        O << getNfmtName(Nfmt, STI);
      }
      O << ']';
    } else {
      O << " format:" << Val;
    }
  }
}

// \returns a low 256 vgpr representing a high vgpr \p Reg [v256..v1023] or
// \p Reg itself otherwise.
static MCPhysReg getRegForPrinting(MCPhysReg Reg, const MCRegisterInfo &MRI) {
  unsigned Enc = MRI.getEncodingValue(Reg);
  unsigned Idx = Enc & AMDGPU::HWEncoding::REG_IDX_MASK;
  if (Idx < 0x100)
    return Reg;

  const MCRegisterClass *RC = getVGPRPhysRegClass(Reg, MRI);
  return RC->getRegister(Idx % 0x100);
}

// Restore MSBs of a VGPR above 255 from the MCInstrAnalysis.
static MCPhysReg getRegFromMIA(MCPhysReg Reg, unsigned OpNo,
                               const MCInstrDesc &Desc,
                               const MCRegisterInfo &MRI,
                               const AMDGPUMCInstrAnalysis &MIA) {
  unsigned VgprMSBs = MIA.getVgprMSBs();
  if (!VgprMSBs)
    return Reg;

  unsigned Enc = MRI.getEncodingValue(Reg);
  if (!(Enc & AMDGPU::HWEncoding::IS_VGPR))
    return Reg;

  auto Ops = AMDGPU::getVGPRLoweringOperandTables(Desc);
  if (!Ops.first)
    return Reg;
  unsigned Opc = Desc.getOpcode();
  unsigned I;
  for (I = 0; I < 4; ++I) {
    if (Ops.first[I] != AMDGPU::OpName::NUM_OPERAND_NAMES &&
        (unsigned)AMDGPU::getNamedOperandIdx(Opc, Ops.first[I]) == OpNo)
      break;
    if (Ops.second && Ops.second[I] != AMDGPU::OpName::NUM_OPERAND_NAMES &&
        (unsigned)AMDGPU::getNamedOperandIdx(Opc, Ops.second[I]) == OpNo)
      break;
  }
  if (I == 4)
    return Reg;
  unsigned OpMSBs = (VgprMSBs >> (I * 2)) & 3;
  if (!OpMSBs)
    return Reg;
  if (MCRegister NewReg = AMDGPU::getVGPRWithMSBs(Reg, OpMSBs, MRI))
    return NewReg;
  return Reg;
}

void AMDGPUInstPrinter::printRegOperand(MCRegister Reg, raw_ostream &O,
                                        const MCRegisterInfo &MRI) {
#if !defined(NDEBUG)
  switch (Reg.id()) {
  case AMDGPU::FP_REG:
  case AMDGPU::SP_REG:
  case AMDGPU::PRIVATE_RSRC_REG:
    llvm_unreachable("pseudo-register should not ever be emitted");
  default:
    break;
  }
#endif

  unsigned PrintReg = getRegForPrinting(Reg, MRI);
  O << getRegisterName(PrintReg);

  if (PrintReg != Reg.id())
    O << " /*" << getRegisterName(Reg) << "*/";
}

void AMDGPUInstPrinter::printRegOperand(MCRegister Reg, unsigned Opc,
                                        unsigned OpNo, raw_ostream &O,
                                        const MCRegisterInfo &MRI) {
  if (MIA)
    Reg = getRegFromMIA(Reg, OpNo, MII.get(Opc), MRI,
                        *static_cast<const AMDGPUMCInstrAnalysis *>(MIA));
  printRegOperand(Reg, O, MRI);
}

void AMDGPUInstPrinter::printVOPDst(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &STI, raw_ostream &O) {
  auto Opcode = MI->getOpcode();
  auto Flags = MII.get(Opcode).TSFlags;
  if (OpNo == 0) {
    if (Flags & SIInstrFlags::VOP3 && Flags & SIInstrFlags::DPP)
      O << "_e64_dpp";
    else if (Flags & SIInstrFlags::VOP3) {
      if (!getVOP3IsSingle(Opcode))
        O << "_e64";
    } else if (Flags & SIInstrFlags::DPP)
      O << "_dpp";
    else if (Flags & SIInstrFlags::SDWA)
      O << "_sdwa";
    else if (((Flags & SIInstrFlags::VOP1) && !getVOP1IsSingle(Opcode)) ||
             ((Flags & SIInstrFlags::VOP2) && !getVOP2IsSingle(Opcode)))
      O << "_e32";
    O << " ";
  }

  printRegularOperand(MI, OpNo, STI, O);

  // Print default vcc/vcc_lo operand.
  switch (Opcode) {
  default: break;

  case AMDGPU::V_ADD_CO_CI_U32_e32_gfx10:
  case AMDGPU::V_SUB_CO_CI_U32_e32_gfx10:
  case AMDGPU::V_SUBREV_CO_CI_U32_e32_gfx10:
  case AMDGPU::V_ADD_CO_CI_U32_sdwa_gfx10:
  case AMDGPU::V_SUB_CO_CI_U32_sdwa_gfx10:
  case AMDGPU::V_SUBREV_CO_CI_U32_sdwa_gfx10:
  case AMDGPU::V_ADD_CO_CI_U32_dpp_gfx10:
  case AMDGPU::V_SUB_CO_CI_U32_dpp_gfx10:
  case AMDGPU::V_SUBREV_CO_CI_U32_dpp_gfx10:
  case AMDGPU::V_ADD_CO_CI_U32_dpp8_gfx10:
  case AMDGPU::V_SUB_CO_CI_U32_dpp8_gfx10:
  case AMDGPU::V_SUBREV_CO_CI_U32_dpp8_gfx10:
  case AMDGPU::V_ADD_CO_CI_U32_e32_gfx11:
  case AMDGPU::V_SUB_CO_CI_U32_e32_gfx11:
  case AMDGPU::V_SUBREV_CO_CI_U32_e32_gfx11:
  case AMDGPU::V_ADD_CO_CI_U32_dpp_gfx11:
  case AMDGPU::V_SUB_CO_CI_U32_dpp_gfx11:
  case AMDGPU::V_SUBREV_CO_CI_U32_dpp_gfx11:
  case AMDGPU::V_ADD_CO_CI_U32_dpp8_gfx11:
  case AMDGPU::V_SUB_CO_CI_U32_dpp8_gfx11:
  case AMDGPU::V_SUBREV_CO_CI_U32_dpp8_gfx11:
  case AMDGPU::V_ADD_CO_CI_U32_e32_gfx12:
  case AMDGPU::V_SUB_CO_CI_U32_e32_gfx12:
  case AMDGPU::V_SUBREV_CO_CI_U32_e32_gfx12:
  case AMDGPU::V_ADD_CO_CI_U32_dpp_gfx12:
  case AMDGPU::V_SUB_CO_CI_U32_dpp_gfx12:
  case AMDGPU::V_SUBREV_CO_CI_U32_dpp_gfx12:
  case AMDGPU::V_ADD_CO_CI_U32_dpp8_gfx12:
  case AMDGPU::V_SUB_CO_CI_U32_dpp8_gfx12:
  case AMDGPU::V_SUBREV_CO_CI_U32_dpp8_gfx12:
    printDefaultVccOperand(false, STI, O);
    break;
  }
}

void AMDGPUInstPrinter::printVINTRPDst(const MCInst *MI, unsigned OpNo,
                                       const MCSubtargetInfo &STI, raw_ostream &O) {
  if (AMDGPU::isSI(STI) || AMDGPU::isCI(STI))
    O << " ";
  else
    O << "_e32 ";

  printRegularOperand(MI, OpNo, STI, O);
}

void AMDGPUInstPrinter::printImmediateInt16(uint32_t Imm,
                                            const MCSubtargetInfo &STI,
                                            raw_ostream &O) {
  int32_t SImm = static_cast<int32_t>(Imm);
  if (isInlinableIntLiteral(SImm)) {
    O << SImm;
    return;
  }

  if (printImmediateFloat32(Imm, STI, O))
    return;

  O << formatHex(static_cast<uint64_t>(Imm & 0xffff));
}

static bool printImmediateFP16(uint32_t Imm, const MCSubtargetInfo &STI,
                               raw_ostream &O) {
  if (Imm == 0x3C00)
    O << "1.0";
  else if (Imm == 0xBC00)
    O << "-1.0";
  else if (Imm == 0x3800)
    O << "0.5";
  else if (Imm == 0xB800)
    O << "-0.5";
  else if (Imm == 0x4000)
    O << "2.0";
  else if (Imm == 0xC000)
    O << "-2.0";
  else if (Imm == 0x4400)
    O << "4.0";
  else if (Imm == 0xC400)
    O << "-4.0";
  else if (Imm == 0x3118 && STI.hasFeature(AMDGPU::FeatureInv2PiInlineImm))
    O << "0.15915494";
  else
    return false;

  return true;
}

static bool printImmediateBFloat16(uint32_t Imm, const MCSubtargetInfo &STI,
                                   raw_ostream &O) {
  if (Imm == 0x3F80)
    O << "1.0";
  else if (Imm == 0xBF80)
    O << "-1.0";
  else if (Imm == 0x3F00)
    O << "0.5";
  else if (Imm == 0xBF00)
    O << "-0.5";
  else if (Imm == 0x4000)
    O << "2.0";
  else if (Imm == 0xC000)
    O << "-2.0";
  else if (Imm == 0x4080)
    O << "4.0";
  else if (Imm == 0xC080)
    O << "-4.0";
  else if (Imm == 0x3E22 && STI.hasFeature(AMDGPU::FeatureInv2PiInlineImm))
    O << "0.15915494";
  else
    return false;

  return true;
}

void AMDGPUInstPrinter::printImmediateBF16(uint32_t Imm,
                                           const MCSubtargetInfo &STI,
                                           raw_ostream &O) {
  int16_t SImm = static_cast<int16_t>(Imm);
  if (isInlinableIntLiteral(SImm)) {
    O << SImm;
    return;
  }

  if (printImmediateBFloat16(static_cast<uint16_t>(Imm), STI, O))
    return;

  O << formatHex(static_cast<uint64_t>(Imm));
}

void AMDGPUInstPrinter::printImmediateF16(uint32_t Imm,
                                          const MCSubtargetInfo &STI,
                                          raw_ostream &O) {
  int16_t SImm = static_cast<int16_t>(Imm);
  if (isInlinableIntLiteral(SImm)) {
    O << SImm;
    return;
  }

  uint16_t HImm = static_cast<uint16_t>(Imm);
  if (printImmediateFP16(HImm, STI, O))
    return;

  uint64_t Imm16 = static_cast<uint16_t>(Imm);
  O << formatHex(Imm16);
}

void AMDGPUInstPrinter::printImmediateV216(uint32_t Imm, uint8_t OpType,
                                           const MCSubtargetInfo &STI,
                                           raw_ostream &O) {
  int32_t SImm = static_cast<int32_t>(Imm);
  if (isInlinableIntLiteral(SImm)) {
    O << SImm;
    return;
  }

  switch (OpType) {
  case AMDGPU::OPERAND_REG_IMM_V2INT16:
  case AMDGPU::OPERAND_REG_INLINE_C_V2INT16:
    if (printImmediateFloat32(Imm, STI, O))
      return;
    break;
  case AMDGPU::OPERAND_REG_IMM_V2FP16:
  case AMDGPU::OPERAND_REG_INLINE_C_V2FP16:
    if (isUInt<16>(Imm) &&
        printImmediateFP16(static_cast<uint16_t>(Imm), STI, O))
      return;
    break;
  case AMDGPU::OPERAND_REG_IMM_V2BF16:
  case AMDGPU::OPERAND_REG_INLINE_C_V2BF16:
    if (isUInt<16>(Imm) &&
        printImmediateBFloat16(static_cast<uint16_t>(Imm), STI, O))
      return;
    break;
  case AMDGPU::OPERAND_REG_IMM_NOINLINE_V2FP16:
    break;
  default:
    llvm_unreachable("bad operand type");
  }

  O << formatHex(static_cast<uint64_t>(Imm));
}

bool AMDGPUInstPrinter::printImmediateFloat32(uint32_t Imm,
                                              const MCSubtargetInfo &STI,
                                              raw_ostream &O) {
  if (Imm == llvm::bit_cast<uint32_t>(0.0f))
    O << "0.0";
  else if (Imm == llvm::bit_cast<uint32_t>(1.0f))
    O << "1.0";
  else if (Imm == llvm::bit_cast<uint32_t>(-1.0f))
    O << "-1.0";
  else if (Imm == llvm::bit_cast<uint32_t>(0.5f))
    O << "0.5";
  else if (Imm == llvm::bit_cast<uint32_t>(-0.5f))
    O << "-0.5";
  else if (Imm == llvm::bit_cast<uint32_t>(2.0f))
    O << "2.0";
  else if (Imm == llvm::bit_cast<uint32_t>(-2.0f))
    O << "-2.0";
  else if (Imm == llvm::bit_cast<uint32_t>(4.0f))
    O << "4.0";
  else if (Imm == llvm::bit_cast<uint32_t>(-4.0f))
    O << "-4.0";
  else if (Imm == 0x3e22f983 &&
           STI.hasFeature(AMDGPU::FeatureInv2PiInlineImm))
    O << "0.15915494";
  else
    return false;

  return true;
}

void AMDGPUInstPrinter::printImmediate32(uint32_t Imm,
                                         const MCSubtargetInfo &STI,
                                         raw_ostream &O) {
  int32_t SImm = static_cast<int32_t>(Imm);
  if (isInlinableIntLiteral(SImm)) {
    O << SImm;
    return;
  }

  if (printImmediateFloat32(Imm, STI, O))
    return;

  O << formatHex(static_cast<uint64_t>(Imm));
}

void AMDGPUInstPrinter::printImmediate64(const MCInstrDesc &Desc, uint64_t Imm,
                                         const MCSubtargetInfo &STI,
                                         raw_ostream &O, bool IsFP) {
  int64_t SImm = static_cast<int64_t>(Imm);
  if (SImm >= -16 && SImm <= 64) {
    O << SImm;
    return;
  }

  if (Imm == llvm::bit_cast<uint64_t>(0.0))
    O << "0.0";
  else if (Imm == llvm::bit_cast<uint64_t>(1.0))
    O << "1.0";
  else if (Imm == llvm::bit_cast<uint64_t>(-1.0))
    O << "-1.0";
  else if (Imm == llvm::bit_cast<uint64_t>(0.5))
    O << "0.5";
  else if (Imm == llvm::bit_cast<uint64_t>(-0.5))
    O << "-0.5";
  else if (Imm == llvm::bit_cast<uint64_t>(2.0))
    O << "2.0";
  else if (Imm == llvm::bit_cast<uint64_t>(-2.0))
    O << "-2.0";
  else if (Imm == llvm::bit_cast<uint64_t>(4.0))
    O << "4.0";
  else if (Imm == llvm::bit_cast<uint64_t>(-4.0))
    O << "-4.0";
  else if (Imm == 0x3fc45f306dc9c882 &&
           STI.hasFeature(AMDGPU::FeatureInv2PiInlineImm))
    O << "0.15915494309189532";
  else
    printLiteral64(Desc, Imm, STI, O, IsFP);
}

void AMDGPUInstPrinter::printLiteral64(const MCInstrDesc &Desc, uint64_t Imm,
                                       const MCSubtargetInfo &STI,
                                       raw_ostream &O, bool IsFP) {
  // This part needs to align with AMDGPUOperand::addLiteralImmOperand.
  bool CanUse64BitLiterals =
      STI.hasFeature(AMDGPU::Feature64BitLiterals) &&
      !(Desc.TSFlags & (SIInstrFlags::VOP3 | SIInstrFlags::VOP3P));
  if (IsFP) {
    if (CanUse64BitLiterals && Lo_32(Imm))
      O << "lit64(" << formatHex(static_cast<uint64_t>(Imm)) << ')';
    else
      O << formatHex(static_cast<uint64_t>(Hi_32(Imm)));
  } else {
    if (CanUse64BitLiterals && (!isInt<32>(Imm) || !isUInt<32>(Imm)))
      O << "lit64(" << formatHex(static_cast<uint64_t>(Imm)) << ')';
    else
      O << formatHex(static_cast<uint64_t>(Imm));
  }
}

void AMDGPUInstPrinter::printBLGP(const MCInst *MI, unsigned OpNo,
                                  const MCSubtargetInfo &STI,
                                  raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNo).getImm();
  if (!Imm)
    return;

  if (AMDGPU::isGFX940(STI)) {
    switch (MI->getOpcode()) {
    case AMDGPU::V_MFMA_F64_16X16X4F64_gfx940_acd:
    case AMDGPU::V_MFMA_F64_16X16X4F64_gfx940_vcd:
    case AMDGPU::V_MFMA_F64_4X4X4F64_gfx940_acd:
    case AMDGPU::V_MFMA_F64_4X4X4F64_gfx940_vcd:
      O << " neg:[" << (Imm & 1) << ',' << ((Imm >> 1) & 1) << ','
        << ((Imm >> 2) & 1) << ']';
      return;
    }
  }

  O << " blgp:" << Imm;
}

void AMDGPUInstPrinter::printDefaultVccOperand(bool FirstOperand,
                                               const MCSubtargetInfo &STI,
                                               raw_ostream &O) {
  if (!FirstOperand)
    O << ", ";
  printRegOperand(STI.hasFeature(AMDGPU::FeatureWavefrontSize32)
                      ? AMDGPU::VCC_LO
                      : AMDGPU::VCC,
                  O, MRI);
  if (FirstOperand)
    O << ", ";
}

bool AMDGPUInstPrinter::needsImpliedVcc(const MCInstrDesc &Desc,
                                        unsigned OpNo) const {
  return OpNo == 0 && (Desc.TSFlags & SIInstrFlags::DPP) &&
         (Desc.TSFlags & SIInstrFlags::VOPC) &&
         !isVOPCAsmOnly(Desc.getOpcode()) &&
         (Desc.hasImplicitDefOfPhysReg(AMDGPU::VCC) ||
          Desc.hasImplicitDefOfPhysReg(AMDGPU::VCC_LO));
}

// Print default vcc/vcc_lo operand of VOPC.
void AMDGPUInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  unsigned Opc = MI->getOpcode();
  const MCInstrDesc &Desc = MII.get(Opc);
  int ModIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0_modifiers);
  // 0, 1 and 2 are the first printed operands in different cases
  // If there are printed modifiers, printOperandAndFPInputMods or
  // printOperandAndIntInputMods will be called instead
  if ((OpNo == 0 ||
       (OpNo == 1 && (Desc.TSFlags & SIInstrFlags::DPP) && ModIdx != -1)) &&
      (Desc.TSFlags & SIInstrFlags::VOPC) && !isVOPCAsmOnly(Desc.getOpcode()) &&
      (Desc.hasImplicitDefOfPhysReg(AMDGPU::VCC) ||
       Desc.hasImplicitDefOfPhysReg(AMDGPU::VCC_LO)))
    printDefaultVccOperand(true, STI, O);

  printRegularOperand(MI, OpNo, STI, O);
}

// Print operands after vcc or modifier handling.
void AMDGPUInstPrinter::printRegularOperand(const MCInst *MI, unsigned OpNo,
                                            const MCSubtargetInfo &STI,
                                            raw_ostream &O) {
  const MCInstrDesc &Desc = MII.get(MI->getOpcode());

  if (OpNo >= MI->getNumOperands()) {
    O << "/*Missing OP" << OpNo << "*/";
    return;
  }

  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    printRegOperand(Op.getReg(), MI->getOpcode(), OpNo, O, MRI);

    // Check if operand register class contains register used.
    // Intention: print disassembler message when invalid code is decoded,
    // for example sgpr register used in VReg or VISrc(VReg or imm) operand.
    int RCID = Desc.operands()[OpNo].RegClass;
    if (RCID != -1) {
      const MCRegisterClass RC = MRI.getRegClass(RCID);
      auto Reg = mc2PseudoReg(Op.getReg());
      if (!RC.contains(Reg) && !isInlineValue(Reg)) {
        O << "/*Invalid register, operand has \'" << MRI.getRegClassName(&RC)
          << "\' register class*/";
      }
    }
  } else if (Op.isImm()) {
    const uint8_t OpTy = Desc.operands()[OpNo].OperandType;
    switch (OpTy) {
    case AMDGPU::OPERAND_REG_IMM_INT32:
    case AMDGPU::OPERAND_REG_IMM_FP32:
    case AMDGPU::OPERAND_REG_INLINE_C_INT32:
    case AMDGPU::OPERAND_REG_INLINE_C_FP32:
    case AMDGPU::OPERAND_REG_INLINE_AC_INT32:
    case AMDGPU::OPERAND_REG_INLINE_AC_FP32:
    case AMDGPU::OPERAND_REG_IMM_V2INT32:
    case AMDGPU::OPERAND_REG_IMM_V2FP32:
    case MCOI::OPERAND_IMMEDIATE:
    case AMDGPU::OPERAND_INLINE_SPLIT_BARRIER_INT32:
      printImmediate32(Op.getImm(), STI, O);
      break;
    case AMDGPU::OPERAND_REG_IMM_INT64:
    case AMDGPU::OPERAND_REG_INLINE_C_INT64:
      printImmediate64(Desc, Op.getImm(), STI, O, false);
      break;
    case AMDGPU::OPERAND_REG_IMM_FP64:
    case AMDGPU::OPERAND_REG_INLINE_C_FP64:
    case AMDGPU::OPERAND_REG_INLINE_AC_FP64:
      printImmediate64(Desc, Op.getImm(), STI, O, true);
      break;
    case AMDGPU::OPERAND_REG_INLINE_C_INT16:
    case AMDGPU::OPERAND_REG_IMM_INT16:
      printImmediateInt16(Op.getImm(), STI, O);
      break;
    case AMDGPU::OPERAND_REG_INLINE_C_FP16:
    case AMDGPU::OPERAND_REG_IMM_FP16:
      printImmediateF16(Op.getImm(), STI, O);
      break;
    case AMDGPU::OPERAND_REG_INLINE_C_BF16:
    case AMDGPU::OPERAND_REG_IMM_BF16:
      printImmediateBF16(Op.getImm(), STI, O);
      break;
    case AMDGPU::OPERAND_REG_IMM_V2INT16:
    case AMDGPU::OPERAND_REG_IMM_V2BF16:
    case AMDGPU::OPERAND_REG_IMM_V2FP16:
    case AMDGPU::OPERAND_REG_IMM_NOINLINE_V2FP16:
    case AMDGPU::OPERAND_REG_INLINE_C_V2INT16:
    case AMDGPU::OPERAND_REG_INLINE_C_V2BF16:
    case AMDGPU::OPERAND_REG_INLINE_C_V2FP16:
      printImmediateV216(Op.getImm(), OpTy, STI, O);
      break;
    case MCOI::OPERAND_UNKNOWN:
    case MCOI::OPERAND_PCREL:
      O << formatDec(Op.getImm());
      break;
    case MCOI::OPERAND_REGISTER:
      // Disassembler does not fail when operand should not allow immediate
      // operands but decodes them into 32bit immediate operand.
      printImmediate32(Op.getImm(), STI, O);
      O << "/*Invalid immediate*/";
      break;
    default:
      // We hit this for the immediate instruction bits that don't yet have a
      // custom printer.
      llvm_unreachable("unexpected immediate operand type");
    }
  } else if (Op.isExpr()) {
    const MCExpr *Exp = Op.getExpr();
    MAI.printExpr(O, *Exp);
  } else {
    O << "/*INV_OP*/";
  }

  // Print default vcc/vcc_lo operand of v_cndmask_b32_e32.
  switch (MI->getOpcode()) {
  default: break;

  case AMDGPU::V_CNDMASK_B32_e32_gfx10:
  case AMDGPU::V_ADD_CO_CI_U32_e32_gfx10:
  case AMDGPU::V_SUB_CO_CI_U32_e32_gfx10:
  case AMDGPU::V_SUBREV_CO_CI_U32_e32_gfx10:
  case AMDGPU::V_ADD_CO_CI_U32_dpp_gfx10:
  case AMDGPU::V_SUB_CO_CI_U32_dpp_gfx10:
  case AMDGPU::V_SUBREV_CO_CI_U32_dpp_gfx10:
  case AMDGPU::V_CNDMASK_B32_dpp8_gfx10:
  case AMDGPU::V_ADD_CO_CI_U32_dpp8_gfx10:
  case AMDGPU::V_SUB_CO_CI_U32_dpp8_gfx10:
  case AMDGPU::V_SUBREV_CO_CI_U32_dpp8_gfx10:
  case AMDGPU::V_CNDMASK_B32_e32_gfx11:
  case AMDGPU::V_ADD_CO_CI_U32_e32_gfx11:
  case AMDGPU::V_SUB_CO_CI_U32_e32_gfx11:
  case AMDGPU::V_SUBREV_CO_CI_U32_e32_gfx11:
  case AMDGPU::V_ADD_CO_CI_U32_dpp_gfx11:
  case AMDGPU::V_SUB_CO_CI_U32_dpp_gfx11:
  case AMDGPU::V_SUBREV_CO_CI_U32_dpp_gfx11:
  case AMDGPU::V_CNDMASK_B32_dpp8_gfx11:
  case AMDGPU::V_ADD_CO_CI_U32_dpp8_gfx11:
  case AMDGPU::V_SUB_CO_CI_U32_dpp8_gfx11:
  case AMDGPU::V_SUBREV_CO_CI_U32_dpp8_gfx11:
  case AMDGPU::V_CNDMASK_B32_e32_gfx12:
  case AMDGPU::V_ADD_CO_CI_U32_e32_gfx12:
  case AMDGPU::V_SUB_CO_CI_U32_e32_gfx12:
  case AMDGPU::V_SUBREV_CO_CI_U32_e32_gfx12:
  case AMDGPU::V_CNDMASK_B32_dpp_gfx12:
  case AMDGPU::V_ADD_CO_CI_U32_dpp_gfx12:
  case AMDGPU::V_SUB_CO_CI_U32_dpp_gfx12:
  case AMDGPU::V_SUBREV_CO_CI_U32_dpp_gfx12:
  case AMDGPU::V_CNDMASK_B32_dpp8_gfx12:
  case AMDGPU::V_ADD_CO_CI_U32_dpp8_gfx12:
  case AMDGPU::V_SUB_CO_CI_U32_dpp8_gfx12:
  case AMDGPU::V_SUBREV_CO_CI_U32_dpp8_gfx12:

  case AMDGPU::V_CNDMASK_B32_e32_gfx6_gfx7:
  case AMDGPU::V_CNDMASK_B32_e32_vi:
    if ((int)OpNo == AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                                AMDGPU::OpName::src1))
      printDefaultVccOperand(OpNo == 0, STI, O);
    break;
  }

  if (Desc.TSFlags & SIInstrFlags::MTBUF) {
    int SOffsetIdx =
      AMDGPU::getNamedOperandIdx(MI->getOpcode(), AMDGPU::OpName::soffset);
    assert(SOffsetIdx != -1);
    if ((int)OpNo == SOffsetIdx)
      printSymbolicFormat(MI, STI, O);
  }
}

void AMDGPUInstPrinter::printOperandAndFPInputMods(const MCInst *MI,
                                                   unsigned OpNo,
                                                   const MCSubtargetInfo &STI,
                                                   raw_ostream &O) {
  const MCInstrDesc &Desc = MII.get(MI->getOpcode());
  if (needsImpliedVcc(Desc, OpNo))
    printDefaultVccOperand(true, STI, O);

  unsigned InputModifiers = MI->getOperand(OpNo).getImm();

  // Use 'neg(...)' instead of '-' to avoid ambiguity.
  // This is important for integer literals because
  // -1 is not the same value as neg(1).
  bool NegMnemo = false;

  if (InputModifiers & SISrcMods::NEG) {
    if (OpNo + 1 < MI->getNumOperands() &&
        (InputModifiers & SISrcMods::ABS) == 0) {
      const MCOperand &Op = MI->getOperand(OpNo + 1);
      NegMnemo = Op.isImm();
    }
    if (NegMnemo) {
      O << "neg(";
    } else {
      O << '-';
    }
  }

  if (InputModifiers & SISrcMods::ABS)
    O << '|';
  printRegularOperand(MI, OpNo + 1, STI, O);
  if (InputModifiers & SISrcMods::ABS)
    O << '|';

  if (NegMnemo) {
    O << ')';
  }

  // Print default vcc/vcc_lo operand of VOP2b.
  switch (MI->getOpcode()) {
  default:
    break;

  case AMDGPU::V_CNDMASK_B32_sdwa_gfx10:
  case AMDGPU::V_CNDMASK_B32_dpp_gfx10:
  case AMDGPU::V_CNDMASK_B32_dpp_gfx11:
    if ((int)OpNo + 1 ==
        AMDGPU::getNamedOperandIdx(MI->getOpcode(), AMDGPU::OpName::src1))
      printDefaultVccOperand(OpNo == 0, STI, O);
    break;
  }
}

void AMDGPUInstPrinter::printOperandAndIntInputMods(const MCInst *MI,
                                                    unsigned OpNo,
                                                    const MCSubtargetInfo &STI,
                                                    raw_ostream &O) {
  const MCInstrDesc &Desc = MII.get(MI->getOpcode());
  if (needsImpliedVcc(Desc, OpNo))
    printDefaultVccOperand(true, STI, O);

  unsigned InputModifiers = MI->getOperand(OpNo).getImm();
  if (InputModifiers & SISrcMods::SEXT)
    O << "sext(";
  printRegularOperand(MI, OpNo + 1, STI, O);
  if (InputModifiers & SISrcMods::SEXT)
    O << ')';

  // Print default vcc/vcc_lo operand of VOP2b.
  switch (MI->getOpcode()) {
  default: break;

  case AMDGPU::V_ADD_CO_CI_U32_sdwa_gfx10:
  case AMDGPU::V_SUB_CO_CI_U32_sdwa_gfx10:
  case AMDGPU::V_SUBREV_CO_CI_U32_sdwa_gfx10:
    if ((int)OpNo + 1 == AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                                    AMDGPU::OpName::src1))
      printDefaultVccOperand(OpNo == 0, STI, O);
    break;
  }
}

void AMDGPUInstPrinter::printDPP8(const MCInst *MI, unsigned OpNo,
                                  const MCSubtargetInfo &STI,
                                  raw_ostream &O) {
  if (!AMDGPU::isGFX10Plus(STI))
    llvm_unreachable("dpp8 is not supported on ASICs earlier than GFX10");

  unsigned Imm = MI->getOperand(OpNo).getImm();
  O << "dpp8:[" << formatDec(Imm & 0x7);
  for (size_t i = 1; i < 8; ++i) {
    O << ',' << formatDec((Imm >> (3 * i)) & 0x7);
  }
  O << ']';
}

void AMDGPUInstPrinter::printDPPCtrl(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  using namespace AMDGPU::DPP;

  unsigned Imm = MI->getOperand(OpNo).getImm();
  const MCInstrDesc &Desc = MII.get(MI->getOpcode());

  if (!AMDGPU::isLegalDPALU_DPPControl(STI, Imm) &&
      AMDGPU::isDPALU_DPP(Desc, STI)) {
    O << " /* DP ALU dpp only supports "
      << (isGFX12(STI) ? "row_share" : "row_newbcast") << " */";
    return;
  }
  if (Imm <= DppCtrl::QUAD_PERM_LAST) {
    O << "quad_perm:[";
    O << formatDec(Imm & 0x3)         << ',';
    O << formatDec((Imm & 0xc)  >> 2) << ',';
    O << formatDec((Imm & 0x30) >> 4) << ',';
    O << formatDec((Imm & 0xc0) >> 6) << ']';
  } else if ((Imm >= DppCtrl::ROW_SHL_FIRST) &&
             (Imm <= DppCtrl::ROW_SHL_LAST)) {
    O << "row_shl:" << formatDec(Imm - DppCtrl::ROW_SHL0);
  } else if ((Imm >= DppCtrl::ROW_SHR_FIRST) &&
             (Imm <= DppCtrl::ROW_SHR_LAST)) {
    O << "row_shr:" << formatDec(Imm - DppCtrl::ROW_SHR0);
  } else if ((Imm >= DppCtrl::ROW_ROR_FIRST) &&
             (Imm <= DppCtrl::ROW_ROR_LAST)) {
    O << "row_ror:" << formatDec(Imm - DppCtrl::ROW_ROR0);
  } else if (Imm == DppCtrl::WAVE_SHL1) {
    if (AMDGPU::isGFX10Plus(STI)) {
      O << "/* wave_shl is not supported starting from GFX10 */";
      return;
    }
    O << "wave_shl:1";
  } else if (Imm == DppCtrl::WAVE_ROL1) {
    if (AMDGPU::isGFX10Plus(STI)) {
      O << "/* wave_rol is not supported starting from GFX10 */";
      return;
    }
    O << "wave_rol:1";
  } else if (Imm == DppCtrl::WAVE_SHR1) {
    if (AMDGPU::isGFX10Plus(STI)) {
      O << "/* wave_shr is not supported starting from GFX10 */";
      return;
    }
    O << "wave_shr:1";
  } else if (Imm == DppCtrl::WAVE_ROR1) {
    if (AMDGPU::isGFX10Plus(STI)) {
      O << "/* wave_ror is not supported starting from GFX10 */";
      return;
    }
    O << "wave_ror:1";
  } else if (Imm == DppCtrl::ROW_MIRROR) {
    O << "row_mirror";
  } else if (Imm == DppCtrl::ROW_HALF_MIRROR) {
    O << "row_half_mirror";
  } else if (Imm == DppCtrl::BCAST15) {
    if (AMDGPU::isGFX10Plus(STI)) {
      O << "/* row_bcast is not supported starting from GFX10 */";
      return;
    }
    O << "row_bcast:15";
  } else if (Imm == DppCtrl::BCAST31) {
    if (AMDGPU::isGFX10Plus(STI)) {
      O << "/* row_bcast is not supported starting from GFX10 */";
      return;
    }
    O << "row_bcast:31";
  } else if ((Imm >= DppCtrl::ROW_SHARE_FIRST) &&
             (Imm <= DppCtrl::ROW_SHARE_LAST)) {
    if (AMDGPU::isGFX90A(STI)) {
      O << "row_newbcast:";
    } else if (AMDGPU::isGFX10Plus(STI)) {
      O << "row_share:";
    } else {
      O << " /* row_newbcast/row_share is not supported on ASICs earlier "
           "than GFX90A/GFX10 */";
      return;
    }
    O << formatDec(Imm - DppCtrl::ROW_SHARE_FIRST);
  } else if ((Imm >= DppCtrl::ROW_XMASK_FIRST) &&
             (Imm <= DppCtrl::ROW_XMASK_LAST)) {
    if (!AMDGPU::isGFX10Plus(STI)) {
      O << "/* row_xmask is not supported on ASICs earlier than GFX10 */";
      return;
    }
    O << "row_xmask:" << formatDec(Imm - DppCtrl::ROW_XMASK_FIRST);
  } else {
    O << "/* Invalid dpp_ctrl value */";
  }
}

void AMDGPUInstPrinter::printDppBoundCtrl(const MCInst *MI, unsigned OpNo,
                                          const MCSubtargetInfo &STI,
                                          raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNo).getImm();
  if (Imm) {
    O << " bound_ctrl:1";
  }
}

void AMDGPUInstPrinter::printDppFI(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI, raw_ostream &O) {
  using namespace llvm::AMDGPU::DPP;
  unsigned Imm = MI->getOperand(OpNo).getImm();
  if (Imm == DPP_FI_1 || Imm == DPP8_FI_1) {
    O << " fi:1";
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

void AMDGPUInstPrinter::printExpSrcN(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI, raw_ostream &O,
                                     unsigned N) {
  unsigned Opc = MI->getOpcode();
  int EnIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::en);
  unsigned En = MI->getOperand(EnIdx).getImm();

  int ComprIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::compr);

  // If compr is set, print as src0, src0, src1, src1
  if (MI->getOperand(ComprIdx).getImm())
    OpNo = OpNo - N + N / 2;

  if (En & (1 << N))
    printRegOperand(MI->getOperand(OpNo).getReg(), Opc, OpNo, O, MRI);
  else
    O << "off";
}

void AMDGPUInstPrinter::printExpSrc0(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  printExpSrcN(MI, OpNo, STI, O, 0);
}

void AMDGPUInstPrinter::printExpSrc1(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  printExpSrcN(MI, OpNo, STI, O, 1);
}

void AMDGPUInstPrinter::printExpSrc2(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  printExpSrcN(MI, OpNo, STI, O, 2);
}

void AMDGPUInstPrinter::printExpSrc3(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  printExpSrcN(MI, OpNo, STI, O, 3);
}

void AMDGPUInstPrinter::printExpTgt(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &STI,
                                    raw_ostream &O) {
  using namespace llvm::AMDGPU::Exp;

  // This is really a 6 bit field.
  unsigned Id = MI->getOperand(OpNo).getImm() & ((1 << 6) - 1);

  int Index;
  StringRef TgtName;
  if (getTgtName(Id, TgtName, Index) && isSupportedTgtId(Id, STI)) {
    O << ' ' << TgtName;
    if (Index >= 0)
      O << Index;
  } else {
    O << " invalid_target_" << Id;
  }
}

static bool allOpsDefaultValue(const int* Ops, int NumOps, int Mod,
                               bool IsPacked, bool HasDstSel) {
  int DefaultValue = IsPacked && (Mod == SISrcMods::OP_SEL_1);

  for (int I = 0; I < NumOps; ++I) {
    if (!!(Ops[I] & Mod) != DefaultValue)
      return false;
  }

  if (HasDstSel && (Ops[0] & SISrcMods::DST_OP_SEL) != 0)
    return false;

  return true;
}

void AMDGPUInstPrinter::printPackedModifier(const MCInst *MI,
                                            StringRef Name,
                                            unsigned Mod,
                                            raw_ostream &O) {
  unsigned Opc = MI->getOpcode();
  int NumOps = 0;
  int Ops[3];

  std::pair<AMDGPU::OpName, AMDGPU::OpName> MOps[] = {
      {AMDGPU::OpName::src0_modifiers, AMDGPU::OpName::src0},
      {AMDGPU::OpName::src1_modifiers, AMDGPU::OpName::src1},
      {AMDGPU::OpName::src2_modifiers, AMDGPU::OpName::src2}};
  int DefaultValue = (Mod == SISrcMods::OP_SEL_1);

  for (auto [SrcMod, Src] : MOps) {
    if (!AMDGPU::hasNamedOperand(Opc, Src))
      break;

    int ModIdx = AMDGPU::getNamedOperandIdx(Opc, SrcMod);
    Ops[NumOps++] =
        (ModIdx != -1) ? MI->getOperand(ModIdx).getImm() : DefaultValue;
  }

  const bool HasDst =
      (AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vdst) != -1) ||
      (AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::sdst) != -1);

  // Print three values of neg/opsel for wmma instructions (prints 0 when there
  // is no src_modifier operand instead of not printing anything).
  if (MII.get(MI->getOpcode()).TSFlags & SIInstrFlags::IsSWMMAC ||
      MII.get(MI->getOpcode()).TSFlags & SIInstrFlags::IsWMMA) {
    NumOps = 0;
    int DefaultValue = Mod == SISrcMods::OP_SEL_1;
    for (AMDGPU::OpName OpName :
         {AMDGPU::OpName::src0_modifiers, AMDGPU::OpName::src1_modifiers,
          AMDGPU::OpName::src2_modifiers}) {
      int Idx = AMDGPU::getNamedOperandIdx(Opc, OpName);
      if (Idx != -1)
        Ops[NumOps++] = MI->getOperand(Idx).getImm();
      else
        Ops[NumOps++] = DefaultValue;
    }
  }

  const bool HasDstSel =
      HasDst && NumOps > 0 && Mod == SISrcMods::OP_SEL_0 &&
      MII.get(MI->getOpcode()).TSFlags & SIInstrFlags::VOP3_OPSEL;

  const bool IsPacked =
    MII.get(MI->getOpcode()).TSFlags & SIInstrFlags::IsPacked;

  if (allOpsDefaultValue(Ops, NumOps, Mod, IsPacked, HasDstSel))
    return;

  O << Name;
  for (int I = 0; I < NumOps; ++I) {
    if (I != 0)
      O << ',';

    O << !!(Ops[I] & Mod);
  }

  if (HasDstSel) {
    O << ',' << !!(Ops[0] & SISrcMods::DST_OP_SEL);
  }

  O << ']';
}

void AMDGPUInstPrinter::printOpSel(const MCInst *MI, unsigned,
                                   const MCSubtargetInfo &STI,
                                   raw_ostream &O) {
  unsigned Opc = MI->getOpcode();
  if (isCvt_F32_Fp8_Bf8_e64(Opc)) {
    auto SrcMod =
        AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0_modifiers);
    unsigned Mod = MI->getOperand(SrcMod).getImm();
    unsigned Index0 = !!(Mod & SISrcMods::OP_SEL_0);
    unsigned Index1 = !!(Mod & SISrcMods::OP_SEL_1);
    if (Index0 || Index1)
      O << " op_sel:[" << Index0 << ',' << Index1 << ']';
    return;
  }
  if (isPermlane16(Opc)) {
    auto FIN = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0_modifiers);
    auto BCN = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src1_modifiers);
    unsigned FI = !!(MI->getOperand(FIN).getImm() & SISrcMods::OP_SEL_0);
    unsigned BC = !!(MI->getOperand(BCN).getImm() & SISrcMods::OP_SEL_0);
    if (FI || BC)
      O << " op_sel:[" << FI << ',' << BC << ']';
    return;
  }

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

void AMDGPUInstPrinter::printIndexKey8bit(const MCInst *MI, unsigned OpNo,
                                          const MCSubtargetInfo &STI,
                                          raw_ostream &O) {
  auto Imm = MI->getOperand(OpNo).getImm() & 0x7;
  if (Imm == 0)
    return;

  O << " index_key:" << Imm;
}

void AMDGPUInstPrinter::printIndexKey16bit(const MCInst *MI, unsigned OpNo,
                                           const MCSubtargetInfo &STI,
                                           raw_ostream &O) {
  auto Imm = MI->getOperand(OpNo).getImm() & 0x7;
  if (Imm == 0)
    return;

  O << " index_key:" << Imm;
}

void AMDGPUInstPrinter::printIndexKey32bit(const MCInst *MI, unsigned OpNo,
                                           const MCSubtargetInfo &STI,
                                           raw_ostream &O) {
  auto Imm = MI->getOperand(OpNo).getImm() & 0x7;
  if (Imm == 0)
    return;

  O << " index_key:" << Imm;
}

void AMDGPUInstPrinter::printMatrixFMT(const MCInst *MI, unsigned OpNo,
                                       const MCSubtargetInfo &STI,
                                       raw_ostream &O, char AorB) {
  auto Imm = MI->getOperand(OpNo).getImm() & 0x7;
  if (Imm == 0)
    return;

  O << " matrix_" << AorB << "_fmt:";
  switch (Imm) {
  default:
    O << Imm;
    break;
  case WMMA::MatrixFMT::MATRIX_FMT_FP8:
    O << "MATRIX_FMT_FP8";
    break;
  case WMMA::MatrixFMT::MATRIX_FMT_BF8:
    O << "MATRIX_FMT_BF8";
    break;
  case WMMA::MatrixFMT::MATRIX_FMT_FP6:
    O << "MATRIX_FMT_FP6";
    break;
  case WMMA::MatrixFMT::MATRIX_FMT_BF6:
    O << "MATRIX_FMT_BF6";
    break;
  case WMMA::MatrixFMT::MATRIX_FMT_FP4:
    O << "MATRIX_FMT_FP4";
    break;
  }
}

void AMDGPUInstPrinter::printMatrixAFMT(const MCInst *MI, unsigned OpNo,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  printMatrixFMT(MI, OpNo, STI, O, 'a');
}

void AMDGPUInstPrinter::printMatrixBFMT(const MCInst *MI, unsigned OpNo,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  printMatrixFMT(MI, OpNo, STI, O, 'b');
}

void AMDGPUInstPrinter::printMatrixScale(const MCInst *MI, unsigned OpNo,
                                         const MCSubtargetInfo &STI,
                                         raw_ostream &O, char AorB) {
  auto Imm = MI->getOperand(OpNo).getImm() & 1;
  if (Imm == 0)
    return;

  O << " matrix_" << AorB << "_scale:";
  switch (Imm) {
  default:
    O << Imm;
    break;
  case WMMA::MatrixScale::MATRIX_SCALE_ROW0:
    O << "MATRIX_SCALE_ROW0";
    break;
  case WMMA::MatrixScale::MATRIX_SCALE_ROW1:
    O << "MATRIX_SCALE_ROW1";
    break;
  }
}

void AMDGPUInstPrinter::printMatrixAScale(const MCInst *MI, unsigned OpNo,
                                          const MCSubtargetInfo &STI,
                                          raw_ostream &O) {
  printMatrixScale(MI, OpNo, STI, O, 'a');
}

void AMDGPUInstPrinter::printMatrixBScale(const MCInst *MI, unsigned OpNo,
                                          const MCSubtargetInfo &STI,
                                          raw_ostream &O) {
  printMatrixScale(MI, OpNo, STI, O, 'b');
}

void AMDGPUInstPrinter::printMatrixScaleFmt(const MCInst *MI, unsigned OpNo,
                                            const MCSubtargetInfo &STI,
                                            raw_ostream &O, char AorB) {
  auto Imm = MI->getOperand(OpNo).getImm() & 3;
  if (Imm == 0)
    return;

  O << " matrix_" << AorB << "_scale_fmt:";
  switch (Imm) {
  default:
    O << Imm;
    break;
  case WMMA::MatrixScaleFmt::MATRIX_SCALE_FMT_E8:
    O << "MATRIX_SCALE_FMT_E8";
    break;
  case WMMA::MatrixScaleFmt::MATRIX_SCALE_FMT_E5M3:
    O << "MATRIX_SCALE_FMT_E5M3";
    break;
  case WMMA::MatrixScaleFmt::MATRIX_SCALE_FMT_E4M3:
    O << "MATRIX_SCALE_FMT_E4M3";
    break;
  }
}

void AMDGPUInstPrinter::printMatrixAScaleFmt(const MCInst *MI, unsigned OpNo,
                                             const MCSubtargetInfo &STI,
                                             raw_ostream &O) {
  printMatrixScaleFmt(MI, OpNo, STI, O, 'a');
}

void AMDGPUInstPrinter::printMatrixBScaleFmt(const MCInst *MI, unsigned OpNo,
                                             const MCSubtargetInfo &STI,
                                             raw_ostream &O) {
  printMatrixScaleFmt(MI, OpNo, STI, O, 'b');
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

void AMDGPUInstPrinter::printGPRIdxMode(const MCInst *MI, unsigned OpNo,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  using namespace llvm::AMDGPU::VGPRIndexMode;
  unsigned Val = MI->getOperand(OpNo).getImm();

  if ((Val & ~ENABLE_MASK) != 0) {
    O << formatHex(static_cast<uint64_t>(Val));
  } else {
    O << "gpr_idx(";
    bool NeedComma = false;
    for (unsigned ModeId = ID_MIN; ModeId <= ID_MAX; ++ModeId) {
      if (Val & (1 << ModeId)) {
        if (NeedComma)
          O << ',';
        O << IdSymbolic[ModeId];
        NeedComma = true;
      }
    }
    O << ')';
  }
}

void AMDGPUInstPrinter::printMemOperand(const MCInst *MI, unsigned OpNo,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  printRegularOperand(MI, OpNo, STI, O);
  O  << ", ";
  printRegularOperand(MI, OpNo + 1, STI, O);
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

void AMDGPUInstPrinter::printSendMsg(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  using namespace llvm::AMDGPU::SendMsg;

  const unsigned Imm16 = MI->getOperand(OpNo).getImm();

  uint16_t MsgId;
  uint16_t OpId;
  uint16_t StreamId;
  decodeMsg(Imm16, MsgId, OpId, StreamId, STI);

  StringRef MsgName = getMsgName(MsgId, STI);

  if (!MsgName.empty() && isValidMsgOp(MsgId, OpId, STI) &&
      isValidMsgStream(MsgId, OpId, StreamId, STI)) {
    O << "sendmsg(" << MsgName;
    if (msgRequiresOp(MsgId, STI)) {
      O << ", " << getMsgOpName(MsgId, OpId, STI);
      if (msgSupportsStream(MsgId, OpId, STI)) {
        O << ", " << StreamId;
      }
    }
    O << ')';
  } else if (encodeMsg(MsgId, OpId, StreamId) == Imm16) {
    O << "sendmsg(" << MsgId << ", " << OpId << ", " << StreamId << ')';
  } else {
    O << Imm16; // Unknown imm16 code.
  }
}

static void printSwizzleBitmask(const uint16_t AndMask,
                                const uint16_t OrMask,
                                const uint16_t XorMask,
                                raw_ostream &O) {
  using namespace llvm::AMDGPU::Swizzle;

  uint16_t Probe0 = ((0            & AndMask) | OrMask) ^ XorMask;
  uint16_t Probe1 = ((BITMASK_MASK & AndMask) | OrMask) ^ XorMask;

  O << "\"";

  for (unsigned Mask = 1 << (BITMASK_WIDTH - 1); Mask > 0; Mask >>= 1) {
    uint16_t p0 = Probe0 & Mask;
    uint16_t p1 = Probe1 & Mask;

    if (p0 == p1) {
      if (p0 == 0) {
        O << "0";
      } else {
        O << "1";
      }
    } else {
      if (p0 == 0) {
        O << "p";
      } else {
        O << "i";
      }
    }
  }

  O << "\"";
}

void AMDGPUInstPrinter::printSwizzle(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  using namespace llvm::AMDGPU::Swizzle;

  uint16_t Imm = MI->getOperand(OpNo).getImm();
  if (Imm == 0) {
    return;
  }

  O << " offset:";

  // Rotate and FFT modes
  if (Imm >= ROTATE_MODE_LO && AMDGPU::isGFX9Plus(STI)) {
    if (Imm >= FFT_MODE_LO) {
      O << "swizzle(" << IdSymbolic[ID_FFT] << ',' << (Imm & FFT_SWIZZLE_MASK)
        << ')';
    } else if (Imm >= ROTATE_MODE_LO) {
      O << "swizzle(" << IdSymbolic[ID_ROTATE] << ','
        << ((Imm >> ROTATE_DIR_SHIFT) & ROTATE_DIR_MASK) << ','
        << ((Imm >> ROTATE_SIZE_SHIFT) & ROTATE_SIZE_MASK) << ')';
    }
    return;
  }

  // Basic mode
  if ((Imm & QUAD_PERM_ENC_MASK) == QUAD_PERM_ENC) {
    O << "swizzle(" << IdSymbolic[ID_QUAD_PERM];
    for (unsigned I = 0; I < LANE_NUM; ++I) {
      O << ",";
      O << formatDec(Imm & LANE_MASK);
      Imm >>= LANE_SHIFT;
    }
    O << ")";

  } else if ((Imm & BITMASK_PERM_ENC_MASK) == BITMASK_PERM_ENC) {

    uint16_t AndMask = (Imm >> BITMASK_AND_SHIFT) & BITMASK_MASK;
    uint16_t OrMask  = (Imm >> BITMASK_OR_SHIFT)  & BITMASK_MASK;
    uint16_t XorMask = (Imm >> BITMASK_XOR_SHIFT) & BITMASK_MASK;

    if (AndMask == BITMASK_MAX && OrMask == 0 && llvm::popcount(XorMask) == 1) {

      O << "swizzle(" << IdSymbolic[ID_SWAP];
      O << ",";
      O << formatDec(XorMask);
      O << ")";

    } else if (AndMask == BITMASK_MAX && OrMask == 0 && XorMask > 0 &&
               isPowerOf2_64(XorMask + 1)) {

      O << "swizzle(" << IdSymbolic[ID_REVERSE];
      O << ",";
      O << formatDec(XorMask + 1);
      O << ")";

    } else {

      uint16_t GroupSize = BITMASK_MAX - AndMask + 1;
      if (GroupSize > 1 &&
          isPowerOf2_64(GroupSize) &&
          OrMask < GroupSize &&
          XorMask == 0) {

        O << "swizzle(" << IdSymbolic[ID_BROADCAST];
        O << ",";
        O << formatDec(GroupSize);
        O << ",";
        O << formatDec(OrMask);
        O << ")";

      } else {
        O << "swizzle(" << IdSymbolic[ID_BITMASK_PERM];
        O << ",";
        printSwizzleBitmask(AndMask, OrMask, XorMask, O);
        O << ")";
      }
    }
  } else {
    printU16ImmDecOperand(MI, OpNo, O);
  }
}

void AMDGPUInstPrinter::printSWaitCnt(const MCInst *MI, unsigned OpNo,
                                      const MCSubtargetInfo &STI,
                                      raw_ostream &O) {
  AMDGPU::IsaVersion ISA = AMDGPU::getIsaVersion(STI.getCPU());

  unsigned SImm16 = MI->getOperand(OpNo).getImm();
  unsigned Vmcnt, Expcnt, Lgkmcnt;
  decodeWaitcnt(ISA, SImm16, Vmcnt, Expcnt, Lgkmcnt);

  bool IsDefaultVmcnt = Vmcnt == getVmcntBitMask(ISA);
  bool IsDefaultExpcnt = Expcnt == getExpcntBitMask(ISA);
  bool IsDefaultLgkmcnt = Lgkmcnt == getLgkmcntBitMask(ISA);
  bool PrintAll = IsDefaultVmcnt && IsDefaultExpcnt && IsDefaultLgkmcnt;

  bool NeedSpace = false;

  if (!IsDefaultVmcnt || PrintAll) {
    O << "vmcnt(" << Vmcnt << ')';
    NeedSpace = true;
  }

  if (!IsDefaultExpcnt || PrintAll) {
    if (NeedSpace)
      O << ' ';
    O << "expcnt(" << Expcnt << ')';
    NeedSpace = true;
  }

  if (!IsDefaultLgkmcnt || PrintAll) {
    if (NeedSpace)
      O << ' ';
    O << "lgkmcnt(" << Lgkmcnt << ')';
  }
}

void AMDGPUInstPrinter::printDepCtr(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &STI,
                                    raw_ostream &O) {
  using namespace llvm::AMDGPU::DepCtr;

  uint64_t Imm16 = MI->getOperand(OpNo).getImm() & 0xffff;

  bool HasNonDefaultVal = false;
  if (isSymbolicDepCtrEncoding(Imm16, HasNonDefaultVal, STI)) {
    int Id = 0;
    StringRef Name;
    unsigned Val;
    bool IsDefault;
    bool NeedSpace = false;
    while (decodeDepCtr(Imm16, Id, Name, Val, IsDefault, STI)) {
      if (!IsDefault || !HasNonDefaultVal) {
        if (NeedSpace)
          O << ' ';
        O << Name << '(' << Val << ')';
        NeedSpace = true;
      }
    }
  } else {
    O << formatHex(Imm16);
  }
}

void AMDGPUInstPrinter::printSDelayALU(const MCInst *MI, unsigned OpNo,
                                       const MCSubtargetInfo &STI,
                                       raw_ostream &O) {
  const char *BadInstId = "/* invalid instid value */";
  static const std::array<const char *, 12> InstIds = {
      "NO_DEP",        "VALU_DEP_1",    "VALU_DEP_2",
      "VALU_DEP_3",    "VALU_DEP_4",    "TRANS32_DEP_1",
      "TRANS32_DEP_2", "TRANS32_DEP_3", "FMA_ACCUM_CYCLE_1",
      "SALU_CYCLE_1",  "SALU_CYCLE_2",  "SALU_CYCLE_3"};

  const char *BadInstSkip = "/* invalid instskip value */";
  static const std::array<const char *, 6> InstSkips = {
      "SAME", "NEXT", "SKIP_1", "SKIP_2", "SKIP_3", "SKIP_4"};

  unsigned SImm16 = MI->getOperand(OpNo).getImm();
  const char *Prefix = "";

  unsigned Value = SImm16 & 0xF;
  if (Value) {
    const char *Name = Value < InstIds.size() ? InstIds[Value] : BadInstId;
    O << Prefix << "instid0(" << Name << ')';
    Prefix = " | ";
  }

  Value = (SImm16 >> 4) & 7;
  if (Value) {
    const char *Name =
        Value < InstSkips.size() ? InstSkips[Value] : BadInstSkip;
    O << Prefix << "instskip(" << Name << ')';
    Prefix = " | ";
  }

  Value = (SImm16 >> 7) & 0xF;
  if (Value) {
    const char *Name = Value < InstIds.size() ? InstIds[Value] : BadInstId;
    O << Prefix << "instid1(" << Name << ')';
    Prefix = " | ";
  }

  if (!*Prefix)
    O << "0";
}

void AMDGPUInstPrinter::printHwreg(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI, raw_ostream &O) {
  using namespace llvm::AMDGPU::Hwreg;
  unsigned Val = MI->getOperand(OpNo).getImm();
  auto [Id, Offset, Width] = HwregEncoding::decode(Val);
  StringRef HwRegName = getHwreg(Id, STI);

  O << "hwreg(";
  if (!HwRegName.empty()) {
    O << HwRegName;
  } else {
    O << Id;
  }
  if (Width != HwregSize::Default || Offset != HwregOffset::Default)
    O << ", " << Offset << ", " << Width;
  O << ')';
}

void AMDGPUInstPrinter::printEndpgm(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &STI,
                                    raw_ostream &O) {
  uint16_t Imm = MI->getOperand(OpNo).getImm();
  if (Imm == 0) {
    return;
  }

  O << ' ' << formatDec(Imm);
}

void AMDGPUInstPrinter::printNamedInt(const MCInst *MI, unsigned OpNo,
                                      const MCSubtargetInfo &STI,
                                      raw_ostream &O, StringRef Prefix,
                                      bool PrintInHex, bool AlwaysPrint) {
  int64_t V = MI->getOperand(OpNo).getImm();
  if (AlwaysPrint || V != 0)
    O << ' ' << Prefix << ':' << (PrintInHex ? formatHex(V) : formatDec(V));
}

void AMDGPUInstPrinter::printBitOp3(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &STI,
                                    raw_ostream &O) {
  uint8_t Imm = MI->getOperand(OpNo).getImm();
  if (!Imm)
    return;

  O << " bitop3:";
  if (Imm <= 10)
    O << formatDec(Imm);
  else
    O << formatHex(static_cast<uint64_t>(Imm));
}

void AMDGPUInstPrinter::printScaleSel(const MCInst *MI, unsigned OpNo,
                                      const MCSubtargetInfo &STI,
                                      raw_ostream &O) {
  uint8_t Imm = MI->getOperand(OpNo).getImm();
  if (!Imm)
    return;

  O << " scale_sel:" << formatDec(Imm);
}

#include "AMDGPUGenAsmWriter.inc"
