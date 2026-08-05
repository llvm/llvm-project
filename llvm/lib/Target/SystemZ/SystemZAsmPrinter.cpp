//===-- SystemZAsmPrinter.cpp - SystemZ LLVM assembly printer -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Streams SystemZ assembly language and associated data, in the form of
// MCInsts and MCExprs respectively.
//
//===----------------------------------------------------------------------===//

#include "SystemZAsmPrinter.h"
#include "SystemZELFAsmPrinter.h"
#include "SystemZXPLINKAsmPrinter.h"
#include "MCTargetDesc/SystemZGNUInstPrinter.h"
#include "MCTargetDesc/SystemZHLASMInstPrinter.h"
#include "MCTargetDesc/SystemZMCAsmInfo.h"
#include "MCTargetDesc/SystemZMCTargetDesc.h"
#include "SystemZMCInstLower.h"
#include "SystemZSubtarget.h"
#include "TargetInfo/SystemZTargetInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

// Return an RI instruction like MI with opcode Opcode, but with the
// GR64 register operands turned into GR32s.
static MCInst lowerRILow(const MachineInstr *MI, unsigned Opcode) {
  if (MI->isCompare())
    return MCInstBuilder(Opcode)
      .addReg(SystemZMC::getRegAsGR32(MI->getOperand(0).getReg()))
      .addImm(MI->getOperand(1).getImm());
  else
    return MCInstBuilder(Opcode)
      .addReg(SystemZMC::getRegAsGR32(MI->getOperand(0).getReg()))
      .addReg(SystemZMC::getRegAsGR32(MI->getOperand(1).getReg()))
      .addImm(MI->getOperand(2).getImm());
}

// Return an RI instruction like MI with opcode Opcode, but with the
// GR64 register operands turned into GRH32s.
static MCInst lowerRIHigh(const MachineInstr *MI, unsigned Opcode) {
  if (MI->isCompare())
    return MCInstBuilder(Opcode)
      .addReg(SystemZMC::getRegAsGRH32(MI->getOperand(0).getReg()))
      .addImm(MI->getOperand(1).getImm());
  else
    return MCInstBuilder(Opcode)
      .addReg(SystemZMC::getRegAsGRH32(MI->getOperand(0).getReg()))
      .addReg(SystemZMC::getRegAsGRH32(MI->getOperand(1).getReg()))
      .addImm(MI->getOperand(2).getImm());
}

// Return an RI instruction like MI with opcode Opcode, but with the
// R2 register turned into a GR64.
static MCInst lowerRIEfLow(const MachineInstr *MI, unsigned Opcode) {
  return MCInstBuilder(Opcode)
    .addReg(MI->getOperand(0).getReg())
    .addReg(MI->getOperand(1).getReg())
    .addReg(SystemZMC::getRegAsGR64(MI->getOperand(2).getReg()))
    .addImm(MI->getOperand(3).getImm())
    .addImm(MI->getOperand(4).getImm())
    .addImm(MI->getOperand(5).getImm());
}

// MI is an instruction that accepts an optional alignment hint,
// and which was already lowered to LoweredMI.  If the alignment
// of the original memory operand is known, update LoweredMI to
// an instruction with the corresponding hint set.
static void lowerAlignmentHint(const MachineInstr *MI, MCInst &LoweredMI,
                               unsigned Opcode) {
  if (MI->memoperands_empty())
    return;

  Align Alignment = Align(16);
  for (const MachineMemOperand *MMO : MI->memoperands())
    if (MMO->getAlign() < Alignment)
      Alignment = MMO->getAlign();

  unsigned AlignmentHint = 0;
  if (Alignment >= Align(16))
    AlignmentHint = 4;
  else if (Alignment >= Align(8))
    AlignmentHint = 3;
  if (AlignmentHint == 0)
    return;

  LoweredMI.setOpcode(Opcode);
  LoweredMI.addOperand(MCOperand::createImm(AlignmentHint));
}

// MI loads the high part of a vector from memory.  Return an instruction
// that uses replicating vector load Opcode to do the same thing.
static MCInst lowerSubvectorLoad(const MachineInstr *MI, unsigned Opcode) {
  return MCInstBuilder(Opcode)
    .addReg(SystemZMC::getRegAsVR128(MI->getOperand(0).getReg()))
    .addReg(MI->getOperand(1).getReg())
    .addImm(MI->getOperand(2).getImm())
    .addReg(MI->getOperand(3).getReg());
}

// MI stores the high part of a vector to memory.  Return an instruction
// that uses elemental vector store Opcode to do the same thing.
static MCInst lowerSubvectorStore(const MachineInstr *MI, unsigned Opcode) {
  return MCInstBuilder(Opcode)
    .addReg(SystemZMC::getRegAsVR128(MI->getOperand(0).getReg()))
    .addReg(MI->getOperand(1).getReg())
    .addImm(MI->getOperand(2).getImm())
    .addReg(MI->getOperand(3).getReg())
    .addImm(0);
}

// MI extracts the first element of the source vector.
static MCInst lowerVecEltExtraction(const MachineInstr *MI, unsigned Opcode) {
  return MCInstBuilder(Opcode)
      .addReg(SystemZMC::getRegAsGR64(MI->getOperand(0).getReg()))
      .addReg(SystemZMC::getRegAsVR128(MI->getOperand(1).getReg()))
      .addReg(0)
      .addImm(0);
}

// MI inserts value into the first element of the destination vector.
static MCInst lowerVecEltInsertion(const MachineInstr *MI, unsigned Opcode) {
  return MCInstBuilder(Opcode)
      .addReg(SystemZMC::getRegAsVR128(MI->getOperand(0).getReg()))
      .addReg(SystemZMC::getRegAsVR128(MI->getOperand(0).getReg()))
      .addReg(MI->getOperand(1).getReg())
      .addReg(0)
      .addImm(0);
}

void SystemZAsmPrinter::emitInstruction(const MachineInstr *MI) {
  SystemZ_MC::verifyInstructionPredicates(MI->getOpcode(),
                                          getSubtargetInfo().getFeatureBits());

  SystemZMCInstLower Lower(MF->getContext(), *this);
  MCInst LoweredMI;
  switch (MI->getOpcode()) {
  case SystemZ::Return:
    LoweredMI = MCInstBuilder(SystemZ::BR)
      .addReg(SystemZ::R14D);
    break;

  case SystemZ::CondReturn:
    LoweredMI = MCInstBuilder(SystemZ::BCR)
      .addImm(MI->getOperand(0).getImm())
      .addImm(MI->getOperand(1).getImm())
      .addReg(SystemZ::R14D);
    break;

  case SystemZ::CRBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CGRBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CGRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CIBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CGIBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CGIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CLRBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CLRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CLGRBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CLGRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CLIBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CLIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CLGIBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CLGIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CallBR:
    LoweredMI = MCInstBuilder(SystemZ::BR)
      .addReg(MI->getOperand(0).getReg());
    break;

  case SystemZ::CallBCR:
    LoweredMI = MCInstBuilder(SystemZ::BCR)
      .addImm(MI->getOperand(0).getImm())
      .addImm(MI->getOperand(1).getImm())
      .addReg(MI->getOperand(2).getReg());
    break;

  case SystemZ::CRBCall:
    LoweredMI = MCInstBuilder(SystemZ::CRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::CGRBCall:
    LoweredMI = MCInstBuilder(SystemZ::CGRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::CIBCall:
    LoweredMI = MCInstBuilder(SystemZ::CIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::CGIBCall:
    LoweredMI = MCInstBuilder(SystemZ::CGIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::CLRBCall:
    LoweredMI = MCInstBuilder(SystemZ::CLRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::CLGRBCall:
    LoweredMI = MCInstBuilder(SystemZ::CLGRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::CLIBCall:
    LoweredMI = MCInstBuilder(SystemZ::CLIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::CLGIBCall:
    LoweredMI = MCInstBuilder(SystemZ::CLGIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::IILF64:
    LoweredMI = MCInstBuilder(SystemZ::IILF)
      .addReg(SystemZMC::getRegAsGR32(MI->getOperand(0).getReg()))
      .addImm(MI->getOperand(2).getImm());
    break;

  case SystemZ::IIHF64:
    LoweredMI = MCInstBuilder(SystemZ::IIHF)
      .addReg(SystemZMC::getRegAsGRH32(MI->getOperand(0).getReg()))
      .addImm(MI->getOperand(2).getImm());
    break;

  case SystemZ::RISBHH:
  case SystemZ::RISBHL:
    LoweredMI = lowerRIEfLow(MI, SystemZ::RISBHG);
    break;

  case SystemZ::RISBLH:
  case SystemZ::RISBLL:
    LoweredMI = lowerRIEfLow(MI, SystemZ::RISBLG);
    break;

  case SystemZ::VLVGP32:
    LoweredMI = MCInstBuilder(SystemZ::VLVGP)
      .addReg(MI->getOperand(0).getReg())
      .addReg(SystemZMC::getRegAsGR64(MI->getOperand(1).getReg()))
      .addReg(SystemZMC::getRegAsGR64(MI->getOperand(2).getReg()));
    break;

  case SystemZ::VLR16:
  case SystemZ::VLR32:
  case SystemZ::VLR64:
    LoweredMI = MCInstBuilder(SystemZ::VLR)
      .addReg(SystemZMC::getRegAsVR128(MI->getOperand(0).getReg()))
      .addReg(SystemZMC::getRegAsVR128(MI->getOperand(1).getReg()));
    break;

  case SystemZ::VL:
    Lower.lower(MI, LoweredMI);
    lowerAlignmentHint(MI, LoweredMI, SystemZ::VLAlign);
    break;

  case SystemZ::VST:
    Lower.lower(MI, LoweredMI);
    lowerAlignmentHint(MI, LoweredMI, SystemZ::VSTAlign);
    break;

  case SystemZ::VLM:
    Lower.lower(MI, LoweredMI);
    lowerAlignmentHint(MI, LoweredMI, SystemZ::VLMAlign);
    break;

  case SystemZ::VSTM:
    Lower.lower(MI, LoweredMI);
    lowerAlignmentHint(MI, LoweredMI, SystemZ::VSTMAlign);
    break;

  case SystemZ::VL16:
    LoweredMI = lowerSubvectorLoad(MI, SystemZ::VLREPH);
    break;

  case SystemZ::VL32:
    LoweredMI = lowerSubvectorLoad(MI, SystemZ::VLREPF);
    break;

  case SystemZ::VL64:
    LoweredMI = lowerSubvectorLoad(MI, SystemZ::VLREPG);
    break;

  case SystemZ::VST16:
    LoweredMI = lowerSubvectorStore(MI, SystemZ::VSTEH);
    break;

  case SystemZ::VST32:
    LoweredMI = lowerSubvectorStore(MI, SystemZ::VSTEF);
    break;

  case SystemZ::VST64:
    LoweredMI = lowerSubvectorStore(MI, SystemZ::VSTEG);
    break;

  case SystemZ::LFER:
    LoweredMI = lowerVecEltExtraction(MI, SystemZ::VLGVF);
    break;

  case SystemZ::LFER_16:
    LoweredMI = lowerVecEltExtraction(MI, SystemZ::VLGVH);
    break;

  case SystemZ::LEFR:
    LoweredMI = lowerVecEltInsertion(MI, SystemZ::VLVGF);
    break;

  case SystemZ::LEFR_16:
    LoweredMI = lowerVecEltInsertion(MI, SystemZ::VLVGH);
    break;

#define LOWER_LOW(NAME)                                                 \
  case SystemZ::NAME##64: LoweredMI = lowerRILow(MI, SystemZ::NAME); break

  LOWER_LOW(IILL);
  LOWER_LOW(IILH);
  LOWER_LOW(TMLL);
  LOWER_LOW(TMLH);
  LOWER_LOW(NILL);
  LOWER_LOW(NILH);
  LOWER_LOW(NILF);
  LOWER_LOW(OILL);
  LOWER_LOW(OILH);
  LOWER_LOW(OILF);
  LOWER_LOW(XILF);

#undef LOWER_LOW

#define LOWER_HIGH(NAME) \
  case SystemZ::NAME##64: LoweredMI = lowerRIHigh(MI, SystemZ::NAME); break

  LOWER_HIGH(IIHL);
  LOWER_HIGH(IIHH);
  LOWER_HIGH(TMHL);
  LOWER_HIGH(TMHH);
  LOWER_HIGH(NIHL);
  LOWER_HIGH(NIHH);
  LOWER_HIGH(NIHF);
  LOWER_HIGH(OIHL);
  LOWER_HIGH(OIHH);
  LOWER_HIGH(OIHF);
  LOWER_HIGH(XIHF);

#undef LOWER_HIGH

  case SystemZ::Serialize:
    if (MF->getSubtarget<SystemZSubtarget>().hasFastSerialization())
      LoweredMI = MCInstBuilder(SystemZ::BCRAsm)
        .addImm(14).addReg(SystemZ::R0D);
    else
      LoweredMI = MCInstBuilder(SystemZ::BCRAsm)
        .addImm(15).addReg(SystemZ::R0D);
    break;

  // We want to emit "j .+2" for traps, jumping to the relative immediate field
  // of the jump instruction, which is an illegal instruction. We cannot emit a
  // "." symbol, so create and emit a temp label before the instruction and use
  // that instead.
  case SystemZ::Trap: {
    MCSymbol *DotSym = OutContext.createTempSymbol();
    OutStreamer->emitLabel(DotSym);

    const MCSymbolRefExpr *Expr = MCSymbolRefExpr::create(DotSym, OutContext);
    const MCConstantExpr *ConstExpr = MCConstantExpr::create(2, OutContext);
    LoweredMI = MCInstBuilder(SystemZ::J)
      .addExpr(MCBinaryExpr::createAdd(Expr, ConstExpr, OutContext));
    }
    break;

  // Conditional traps will create a branch on condition instruction that jumps
  // to the relative immediate field of the jump instruction. (eg. "jo .+2")
  case SystemZ::CondTrap: {
    MCSymbol *DotSym = OutContext.createTempSymbol();
    OutStreamer->emitLabel(DotSym);

    const MCSymbolRefExpr *Expr = MCSymbolRefExpr::create(DotSym, OutContext);
    const MCConstantExpr *ConstExpr = MCConstantExpr::create(2, OutContext);
    LoweredMI = MCInstBuilder(SystemZ::BRC)
      .addImm(MI->getOperand(0).getImm())
      .addImm(MI->getOperand(1).getImm())
      .addExpr(MCBinaryExpr::createAdd(Expr, ConstExpr, OutContext));
    }
    break;

  case TargetOpcode::PATCHABLE_FUNCTION_EXIT:
    llvm_unreachable("PATCHABLE_FUNCTION_EXIT should never be emitted");

  case TargetOpcode::PATCHABLE_TAIL_CALL:
    // TODO: Define a trampoline `__xray_FunctionTailExit` and differentiate a
    // normal function exit from a tail exit.
    llvm_unreachable("Tail call is handled in the normal case. See comments "
                     "around this assert.");

  case SystemZ::EXRL_Pseudo: {
    unsigned TargetInsOpc = MI->getOperand(0).getImm();
    Register LenMinus1Reg = MI->getOperand(1).getReg();
    Register DestReg = MI->getOperand(2).getReg();
    int64_t DestDisp = MI->getOperand(3).getImm();
    Register SrcReg = MI->getOperand(4).getReg();
    int64_t SrcDisp = MI->getOperand(5).getImm();

    SystemZTargetStreamer *TS = getTargetStreamer();
    MCInst ET = MCInstBuilder(TargetInsOpc)
                    .addReg(DestReg)
                    .addImm(DestDisp)
                    .addImm(1)
                    .addReg(SrcReg)
                    .addImm(SrcDisp);
    SystemZTargetStreamer::MCInstSTIPair ET_STI(ET, &MF->getSubtarget());
    auto [It, Inserted] = TS->EXRLTargets2Sym.try_emplace(ET_STI);
    if (Inserted)
      It->second = OutContext.createTempSymbol();
    MCSymbol *DotSym = It->second;
    const MCSymbolRefExpr *Dot = MCSymbolRefExpr::create(DotSym, OutContext);
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(SystemZ::EXRL).addReg(LenMinus1Reg).addExpr(Dot));
    return;
  }

  case SystemZ::FENCE:
    OutStreamer->emitRawComment("FENCE");
    return;

  // EH_SjLj_Setup is a dummy terminator instruction of size 0.
  // It is used to handle the clobber register for builtin setjmp.
  case SystemZ::EH_SjLj_Setup:
    return;

  default:
    Lower.lower(MI, LoweredMI);
    break;
  }
  EmitToStreamer(*OutStreamer, LoweredMI);
}

static void printFormattedRegName(const MCAsmInfo *MAI, unsigned RegNo,
                                  raw_ostream &OS) {
  const char *RegName;
  if (MAI->getAssemblerDialect() == AD_HLASM) {
    RegName = SystemZHLASMInstPrinter::getRegisterName(RegNo);
    // Skip register prefix so that only register number is left
    assert(isalpha(RegName[0]) && isdigit(RegName[1]));
    OS << (RegName + 1);
  } else {
    RegName = SystemZGNUInstPrinter::getRegisterName(RegNo);
    OS << '%' << RegName;
  }
}

static void printReg(unsigned Reg, const MCAsmInfo *MAI, raw_ostream &OS) {
  if (!Reg)
    OS << '0';
  else
    printFormattedRegName(MAI, Reg, OS);
}

static void printOperand(const MCOperand &MCOp, const MCAsmInfo *MAI,
                         raw_ostream &OS) {
  if (MCOp.isReg())
    printReg(MCOp.getReg(), MAI, OS);
  else if (MCOp.isImm())
    OS << MCOp.getImm();
  else if (MCOp.isExpr())
    MAI->printExpr(OS, *MCOp.getExpr());
  else
    llvm_unreachable("Invalid operand");
}

static void printAddress(const MCAsmInfo *MAI, unsigned Base,
                         const MCOperand &DispMO, unsigned Index,
                         raw_ostream &OS) {
  printOperand(DispMO, MAI, OS);
  if (Base || Index) {
    OS << '(';
    if (Index) {
      printFormattedRegName(MAI, Index, OS);
      if (Base)
        OS << ',';
    }
    if (Base)
      printFormattedRegName(MAI, Base, OS);
    OS << ')';
  }
}

bool SystemZAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                        const char *ExtraCode,
                                        raw_ostream &OS) {
  const MCRegisterInfo &MRI = TM.getMCRegisterInfo();
  const MachineOperand &MO = MI->getOperand(OpNo);
  MCOperand MCOp;
  if (ExtraCode) {
    if ((ExtraCode[0] == 'N' || ExtraCode[0] == 'M') && !ExtraCode[1] &&
        MO.isReg() && SystemZ::GR128BitRegClass.contains(MO.getReg()))
      MCOp =
          MCOperand::createReg(MRI.getSubReg(MO.getReg(), SystemZ::subreg_l64));
    else
      return AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, OS);
  } else {
    SystemZMCInstLower Lower(MF->getContext(), *this);
    MCOp = Lower.lowerOperand(MO);
  }
  printOperand(MCOp, &MAI, OS);
  return false;
}

bool SystemZAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                              unsigned OpNo,
                                              const char *ExtraCode,
                                              raw_ostream &OS) {
  if (ExtraCode && ExtraCode[0] && !ExtraCode[1]) {
    switch (ExtraCode[0]) {
    case 'A':
      // Unlike EmitMachineNode(), EmitSpecialNode(INLINEASM) does not call
      // setMemRefs(), so MI->memoperands() is empty and the alignment
      // information is not available.
      return false;
    case 'O':
      OS << MI->getOperand(OpNo + 1).getImm();
      return false;
    case 'R':
      ::printReg(MI->getOperand(OpNo).getReg(), &MAI, OS);
      return false;
    }
  }
  printAddress(&MAI, MI->getOperand(OpNo).getReg(),
               MCOperand::createImm(MI->getOperand(OpNo + 1).getImm()),
               MI->getOperand(OpNo + 2).getReg(), OS);
  return false;
}

char SystemZAsmPrinter::ID = 0;

INITIALIZE_PASS(SystemZAsmPrinter, "systemz-asm-printer",
                "SystemZ Assembly Printer", false, false)

static AsmPrinter *createSystemZAsmPrinter(TargetMachine &TM,
                                           std::unique_ptr<MCStreamer> &&Streamer) {
  if (TM.getTargetTriple().isOSzOS())
    return new SystemZXPLINKAsmPrinter(TM, std::move(Streamer));
  return new SystemZELFAsmPrinter(TM, std::move(Streamer));
}

// Force static initialization.
extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeSystemZAsmPrinter() {
  TargetRegistry::RegisterAsmPrinter(getTheSystemZTarget(),
                                     createSystemZAsmPrinter);
}
