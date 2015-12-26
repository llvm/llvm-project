//==-- AArch64InstPrinter.cpp - Convert AArch64 MCInst to assembly syntax --==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints an AArch64 MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "AArch64InstPrinter.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#define GET_INSTRUCTION_NAME
#define PRINT_ALIAS_INSTR
#include "AArch64GenAsmWriter.inc"
#define GET_INSTRUCTION_NAME
#define PRINT_ALIAS_INSTR
#include "AArch64GenAsmWriter1.inc"

AArch64InstPrinter::AArch64InstPrinter(const MCAsmInfo &MAI,
                                       const MCInstrInfo &MII,
                                       const MCRegisterInfo &MRI)
    : MCInstPrinter(MAI, MII, MRI) {}

AArch64AppleInstPrinter::AArch64AppleInstPrinter(const MCAsmInfo &MAI,
                                                 const MCInstrInfo &MII,
                                                 const MCRegisterInfo &MRI)
    : AArch64InstPrinter(MAI, MII, MRI) {}

void AArch64InstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
  // This is for .cfi directives.
  OS << getRegisterName(RegNo);
}

void AArch64InstPrinter::printInst(const MCInst *MI, raw_ostream &O,
                                   StringRef Annot,
                                   const MCSubtargetInfo &STI) {
  // Check for special encodings and print the canonical alias instead.

  unsigned Opcode = MI->getOpcode();

  if (Opcode == AArch64::SYSxt)
    if (printSysAlias(MI, STI, O)) {
      printAnnotation(O, Annot);
      return;
    }

  // SBFM/UBFM should print to a nicer aliased form if possible.
  if (Opcode == AArch64::SBFMXri || Opcode == AArch64::SBFMWri ||
      Opcode == AArch64::UBFMXri || Opcode == AArch64::UBFMWri) {
    const MCOperand &Op0 = MI->getOperand(0);
    const MCOperand &Op1 = MI->getOperand(1);
    const MCOperand &Op2 = MI->getOperand(2);
    const MCOperand &Op3 = MI->getOperand(3);

    bool IsSigned = (Opcode == AArch64::SBFMXri || Opcode == AArch64::SBFMWri);
    bool Is64Bit = (Opcode == AArch64::SBFMXri || Opcode == AArch64::UBFMXri);
    if (Op2.isImm() && Op2.getImm() == 0 && Op3.isImm()) {
      const char *AsmMnemonic = nullptr;

      switch (Op3.getImm()) {
      default:
        break;
      case 7:
        if (IsSigned)
          AsmMnemonic = "sxtb";
        else if (!Is64Bit)
          AsmMnemonic = "uxtb";
        break;
      case 15:
        if (IsSigned)
          AsmMnemonic = "sxth";
        else if (!Is64Bit)
          AsmMnemonic = "uxth";
        break;
      case 31:
        // *xtw is only valid for signed 64-bit operations.
        if (Is64Bit && IsSigned)
          AsmMnemonic = "sxtw";
        break;
      }

      if (AsmMnemonic) {
        O << '\t' << AsmMnemonic << '\t' << getRegisterName(Op0.getReg())
          << ", " << getRegisterName(getWRegFromXReg(Op1.getReg()));
        printAnnotation(O, Annot);
        return;
      }
    }

    // All immediate shifts are aliases, implemented using the Bitfield
    // instruction. In all cases the immediate shift amount shift must be in
    // the range 0 to (reg.size -1).
    if (Op2.isImm() && Op3.isImm()) {
      const char *AsmMnemonic = nullptr;
      int shift = 0;
      int64_t immr = Op2.getImm();
      int64_t imms = Op3.getImm();
      if (Opcode == AArch64::UBFMWri && imms != 0x1F && ((imms + 1) == immr)) {
        AsmMnemonic = "lsl";
        shift = 31 - imms;
      } else if (Opcode == AArch64::UBFMXri && imms != 0x3f &&
                 ((imms + 1 == immr))) {
        AsmMnemonic = "lsl";
        shift = 63 - imms;
      } else if (Opcode == AArch64::UBFMWri && imms == 0x1f) {
        AsmMnemonic = "lsr";
        shift = immr;
      } else if (Opcode == AArch64::UBFMXri && imms == 0x3f) {
        AsmMnemonic = "lsr";
        shift = immr;
      } else if (Opcode == AArch64::SBFMWri && imms == 0x1f) {
        AsmMnemonic = "asr";
        shift = immr;
      } else if (Opcode == AArch64::SBFMXri && imms == 0x3f) {
        AsmMnemonic = "asr";
        shift = immr;
      }
      if (AsmMnemonic) {
        O << '\t' << AsmMnemonic << '\t' << getRegisterName(Op0.getReg())
          << ", " << getRegisterName(Op1.getReg()) << ", #" << shift;
        printAnnotation(O, Annot);
        return;
      }
    }

    // SBFIZ/UBFIZ aliases
    if (Op2.getImm() > Op3.getImm()) {
      O << '\t' << (IsSigned ? "sbfiz" : "ubfiz") << '\t'
        << getRegisterName(Op0.getReg()) << ", " << getRegisterName(Op1.getReg())
        << ", #" << (Is64Bit ? 64 : 32) - Op2.getImm() << ", #" << Op3.getImm() + 1;
      printAnnotation(O, Annot);
      return;
    }

    // Otherwise SBFX/UBFX is the preferred form
    O << '\t' << (IsSigned ? "sbfx" : "ubfx") << '\t'
      << getRegisterName(Op0.getReg()) << ", " << getRegisterName(Op1.getReg())
      << ", #" << Op2.getImm() << ", #" << Op3.getImm() - Op2.getImm() + 1;
    printAnnotation(O, Annot);
    return;
  }

  if (Opcode == AArch64::BFMXri || Opcode == AArch64::BFMWri) {
    const MCOperand &Op0 = MI->getOperand(0); // Op1 == Op0
    const MCOperand &Op2 = MI->getOperand(2);
    int ImmR = MI->getOperand(3).getImm();
    int ImmS = MI->getOperand(4).getImm();

    if ((Op2.getReg() == AArch64::WZR || Op2.getReg() == AArch64::XZR) &&
        (ImmR == 0 || ImmS < ImmR)) {
      // BFC takes precedence over its entire range, sligtly differently to BFI.
      int BitWidth = Opcode == AArch64::BFMXri ? 64 : 32;
      int LSB = (BitWidth - ImmR) % BitWidth;
      int Width = ImmS + 1;

      O << "\tbfc\t" << getRegisterName(Op0.getReg())
        << ", #" << LSB << ", #" << Width;
      printAnnotation(O, Annot);
      return;
    } else if (ImmS < ImmR) {
      // BFI alias
      int BitWidth = Opcode == AArch64::BFMXri ? 64 : 32;
      int LSB = (BitWidth - ImmR) % BitWidth;
      int Width = ImmS + 1;

      O << "\tbfi\t" << getRegisterName(Op0.getReg()) << ", "
        << getRegisterName(Op2.getReg()) << ", #" << LSB << ", #" << Width;
      printAnnotation(O, Annot);
      return;
    }

    int LSB = ImmR;
    int Width = ImmS - ImmR + 1;
    // Otherwise BFXIL the preferred form
    O << "\tbfxil\t"
      << getRegisterName(Op0.getReg()) << ", " << getRegisterName(Op2.getReg())
      << ", #" << LSB << ", #" << Width;
    printAnnotation(O, Annot);
    return;
  }

  // Symbolic operands for MOVZ, MOVN and MOVK already imply a shift
  // (e.g. :gottprel_g1: is always going to be "lsl #16") so it should not be
  // printed.
  if ((Opcode == AArch64::MOVZXi || Opcode == AArch64::MOVZWi ||
       Opcode == AArch64::MOVNXi || Opcode == AArch64::MOVNWi) &&
      MI->getOperand(1).isExpr()) {
    if (Opcode == AArch64::MOVZXi || Opcode == AArch64::MOVZWi)
      O << "\tmovz\t";
    else
      O << "\tmovn\t";

    O << getRegisterName(MI->getOperand(0).getReg()) << ", #";
    MI->getOperand(1).getExpr()->print(O, &MAI);
    return;
  }

  if ((Opcode == AArch64::MOVKXi || Opcode == AArch64::MOVKWi) &&
      MI->getOperand(2).isExpr()) {
    O << "\tmovk\t" << getRegisterName(MI->getOperand(0).getReg()) << ", #";
    MI->getOperand(2).getExpr()->print(O, &MAI);
    return;
  }

  if (!printAliasInstr(MI, STI, O))
    printInstruction(MI, STI, O);

  printAnnotation(O, Annot);
}

static bool isTblTbxInstruction(unsigned Opcode, StringRef &Layout,
                                bool &IsTbx) {
  switch (Opcode) {
  case AArch64::TBXv8i8One:
  case AArch64::TBXv8i8Two:
  case AArch64::TBXv8i8Three:
  case AArch64::TBXv8i8Four:
    IsTbx = true;
    Layout = ".8b";
    return true;
  case AArch64::TBLv8i8One:
  case AArch64::TBLv8i8Two:
  case AArch64::TBLv8i8Three:
  case AArch64::TBLv8i8Four:
    IsTbx = false;
    Layout = ".8b";
    return true;
  case AArch64::TBXv16i8One:
  case AArch64::TBXv16i8Two:
  case AArch64::TBXv16i8Three:
  case AArch64::TBXv16i8Four:
    IsTbx = true;
    Layout = ".16b";
    return true;
  case AArch64::TBLv16i8One:
  case AArch64::TBLv16i8Two:
  case AArch64::TBLv16i8Three:
  case AArch64::TBLv16i8Four:
    IsTbx = false;
    Layout = ".16b";
    return true;
  default:
    return false;
  }
}

struct LdStNInstrDesc {
  unsigned Opcode;
  const char *Mnemonic;
  const char *Layout;
  int ListOperand;
  bool HasLane;
  int NaturalOffset;
};

static const LdStNInstrDesc LdStNInstInfo[] = {
  { AArch64::LD1i8,             "ld1",  ".b",     1, true,  0  },
  { AArch64::LD1i16,            "ld1",  ".h",     1, true,  0  },
  { AArch64::LD1i32,            "ld1",  ".s",     1, true,  0  },
  { AArch64::LD1i64,            "ld1",  ".d",     1, true,  0  },
  { AArch64::LD1i8_POST,        "ld1",  ".b",     2, true,  1  },
  { AArch64::LD1i16_POST,       "ld1",  ".h",     2, true,  2  },
  { AArch64::LD1i32_POST,       "ld1",  ".s",     2, true,  4  },
  { AArch64::LD1i64_POST,       "ld1",  ".d",     2, true,  8  },
  { AArch64::LD1Rv16b,          "ld1r", ".16b",   0, false, 0  },
  { AArch64::LD1Rv8h,           "ld1r", ".8h",    0, false, 0  },
  { AArch64::LD1Rv4s,           "ld1r", ".4s",    0, false, 0  },
  { AArch64::LD1Rv2d,           "ld1r", ".2d",    0, false, 0  },
  { AArch64::LD1Rv8b,           "ld1r", ".8b",    0, false, 0  },
  { AArch64::LD1Rv4h,           "ld1r", ".4h",    0, false, 0  },
  { AArch64::LD1Rv2s,           "ld1r", ".2s",    0, false, 0  },
  { AArch64::LD1Rv1d,           "ld1r", ".1d",    0, false, 0  },
  { AArch64::LD1Rv16b_POST,     "ld1r", ".16b",   1, false, 1  },
  { AArch64::LD1Rv8h_POST,      "ld1r", ".8h",    1, false, 2  },
  { AArch64::LD1Rv4s_POST,      "ld1r", ".4s",    1, false, 4  },
  { AArch64::LD1Rv2d_POST,      "ld1r", ".2d",    1, false, 8  },
  { AArch64::LD1Rv8b_POST,      "ld1r", ".8b",    1, false, 1  },
  { AArch64::LD1Rv4h_POST,      "ld1r", ".4h",    1, false, 2  },
  { AArch64::LD1Rv2s_POST,      "ld1r", ".2s",    1, false, 4  },
  { AArch64::LD1Rv1d_POST,      "ld1r", ".1d",    1, false, 8  },
  { AArch64::LD1Onev16b,        "ld1",  ".16b",   0, false, 0  },
  { AArch64::LD1Onev8h,         "ld1",  ".8h",    0, false, 0  },
  { AArch64::LD1Onev4s,         "ld1",  ".4s",    0, false, 0  },
  { AArch64::LD1Onev2d,         "ld1",  ".2d",    0, false, 0  },
  { AArch64::LD1Onev8b,         "ld1",  ".8b",    0, false, 0  },
  { AArch64::LD1Onev4h,         "ld1",  ".4h",    0, false, 0  },
  { AArch64::LD1Onev2s,         "ld1",  ".2s",    0, false, 0  },
  { AArch64::LD1Onev1d,         "ld1",  ".1d",    0, false, 0  },
  { AArch64::LD1Onev16b_POST,   "ld1",  ".16b",   1, false, 16 },
  { AArch64::LD1Onev8h_POST,    "ld1",  ".8h",    1, false, 16 },
  { AArch64::LD1Onev4s_POST,    "ld1",  ".4s",    1, false, 16 },
  { AArch64::LD1Onev2d_POST,    "ld1",  ".2d",    1, false, 16 },
  { AArch64::LD1Onev8b_POST,    "ld1",  ".8b",    1, false, 8  },
  { AArch64::LD1Onev4h_POST,    "ld1",  ".4h",    1, false, 8  },
  { AArch64::LD1Onev2s_POST,    "ld1",  ".2s",    1, false, 8  },
  { AArch64::LD1Onev1d_POST,    "ld1",  ".1d",    1, false, 8  },
  { AArch64::LD1Twov16b,        "ld1",  ".16b",   0, false, 0  },
  { AArch64::LD1Twov8h,         "ld1",  ".8h",    0, false, 0  },
  { AArch64::LD1Twov4s,         "ld1",  ".4s",    0, false, 0  },
  { AArch64::LD1Twov2d,         "ld1",  ".2d",    0, false, 0  },
  { AArch64::LD1Twov8b,         "ld1",  ".8b",    0, false, 0  },
  { AArch64::LD1Twov4h,         "ld1",  ".4h",    0, false, 0  },
  { AArch64::LD1Twov2s,         "ld1",  ".2s",    0, false, 0  },
  { AArch64::LD1Twov1d,         "ld1",  ".1d",    0, false, 0  },
  { AArch64::LD1Twov16b_POST,   "ld1",  ".16b",   1, false, 32 },
  { AArch64::LD1Twov8h_POST,    "ld1",  ".8h",    1, false, 32 },
  { AArch64::LD1Twov4s_POST,    "ld1",  ".4s",    1, false, 32 },
  { AArch64::LD1Twov2d_POST,    "ld1",  ".2d",    1, false, 32 },
  { AArch64::LD1Twov8b_POST,    "ld1",  ".8b",    1, false, 16 },
  { AArch64::LD1Twov4h_POST,    "ld1",  ".4h",    1, false, 16 },
  { AArch64::LD1Twov2s_POST,    "ld1",  ".2s",    1, false, 16 },
  { AArch64::LD1Twov1d_POST,    "ld1",  ".1d",    1, false, 16 },
  { AArch64::LD1Threev16b,      "ld1",  ".16b",   0, false, 0  },
  { AArch64::LD1Threev8h,       "ld1",  ".8h",    0, false, 0  },
  { AArch64::LD1Threev4s,       "ld1",  ".4s",    0, false, 0  },
  { AArch64::LD1Threev2d,       "ld1",  ".2d",    0, false, 0  },
  { AArch64::LD1Threev8b,       "ld1",  ".8b",    0, false, 0  },
  { AArch64::LD1Threev4h,       "ld1",  ".4h",    0, false, 0  },
  { AArch64::LD1Threev2s,       "ld1",  ".2s",    0, false, 0  },
  { AArch64::LD1Threev1d,       "ld1",  ".1d",    0, false, 0  },
  { AArch64::LD1Threev16b_POST, "ld1",  ".16b",   1, false, 48 },
  { AArch64::LD1Threev8h_POST,  "ld1",  ".8h",    1, false, 48 },
  { AArch64::LD1Threev4s_POST,  "ld1",  ".4s",    1, false, 48 },
  { AArch64::LD1Threev2d_POST,  "ld1",  ".2d",    1, false, 48 },
  { AArch64::LD1Threev8b_POST,  "ld1",  ".8b",    1, false, 24 },
  { AArch64::LD1Threev4h_POST,  "ld1",  ".4h",    1, false, 24 },
  { AArch64::LD1Threev2s_POST,  "ld1",  ".2s",    1, false, 24 },
  { AArch64::LD1Threev1d_POST,  "ld1",  ".1d",    1, false, 24 },
  { AArch64::LD1Fourv16b,       "ld1",  ".16b",   0, false, 0  },
  { AArch64::LD1Fourv8h,        "ld1",  ".8h",    0, false, 0  },
  { AArch64::LD1Fourv4s,        "ld1",  ".4s",    0, false, 0  },
  { AArch64::LD1Fourv2d,        "ld1",  ".2d",    0, false, 0  },
  { AArch64::LD1Fourv8b,        "ld1",  ".8b",    0, false, 0  },
  { AArch64::LD1Fourv4h,        "ld1",  ".4h",    0, false, 0  },
  { AArch64::LD1Fourv2s,        "ld1",  ".2s",    0, false, 0  },
  { AArch64::LD1Fourv1d,        "ld1",  ".1d",    0, false, 0  },
  { AArch64::LD1Fourv16b_POST,  "ld1",  ".16b",   1, false, 64 },
  { AArch64::LD1Fourv8h_POST,   "ld1",  ".8h",    1, false, 64 },
  { AArch64::LD1Fourv4s_POST,   "ld1",  ".4s",    1, false, 64 },
  { AArch64::LD1Fourv2d_POST,   "ld1",  ".2d",    1, false, 64 },
  { AArch64::LD1Fourv8b_POST,   "ld1",  ".8b",    1, false, 32 },
  { AArch64::LD1Fourv4h_POST,   "ld1",  ".4h",    1, false, 32 },
  { AArch64::LD1Fourv2s_POST,   "ld1",  ".2s",    1, false, 32 },
  { AArch64::LD1Fourv1d_POST,   "ld1",  ".1d",    1, false, 32 },
  { AArch64::LD2i8,             "ld2",  ".b",     1, true,  0  },
  { AArch64::LD2i16,            "ld2",  ".h",     1, true,  0  },
  { AArch64::LD2i32,            "ld2",  ".s",     1, true,  0  },
  { AArch64::LD2i64,            "ld2",  ".d",     1, true,  0  },
  { AArch64::LD2i8_POST,        "ld2",  ".b",     2, true,  2  },
  { AArch64::LD2i16_POST,       "ld2",  ".h",     2, true,  4  },
  { AArch64::LD2i32_POST,       "ld2",  ".s",     2, true,  8  },
  { AArch64::LD2i64_POST,       "ld2",  ".d",     2, true,  16  },
  { AArch64::LD2Rv16b,          "ld2r", ".16b",   0, false, 0  },
  { AArch64::LD2Rv8h,           "ld2r", ".8h",    0, false, 0  },
  { AArch64::LD2Rv4s,           "ld2r", ".4s",    0, false, 0  },
  { AArch64::LD2Rv2d,           "ld2r", ".2d",    0, false, 0  },
  { AArch64::LD2Rv8b,           "ld2r", ".8b",    0, false, 0  },
  { AArch64::LD2Rv4h,           "ld2r", ".4h",    0, false, 0  },
  { AArch64::LD2Rv2s,           "ld2r", ".2s",    0, false, 0  },
  { AArch64::LD2Rv1d,           "ld2r", ".1d",    0, false, 0  },
  { AArch64::LD2Rv16b_POST,     "ld2r", ".16b",   1, false, 2  },
  { AArch64::LD2Rv8h_POST,      "ld2r", ".8h",    1, false, 4  },
  { AArch64::LD2Rv4s_POST,      "ld2r", ".4s",    1, false, 8  },
  { AArch64::LD2Rv2d_POST,      "ld2r", ".2d",    1, false, 16 },
  { AArch64::LD2Rv8b_POST,      "ld2r", ".8b",    1, false, 2  },
  { AArch64::LD2Rv4h_POST,      "ld2r", ".4h",    1, false, 4  },
  { AArch64::LD2Rv2s_POST,      "ld2r", ".2s",    1, false, 8  },
  { AArch64::LD2Rv1d_POST,      "ld2r", ".1d",    1, false, 16 },
  { AArch64::LD2Twov16b,        "ld2",  ".16b",   0, false, 0  },
  { AArch64::LD2Twov8h,         "ld2",  ".8h",    0, false, 0  },
  { AArch64::LD2Twov4s,         "ld2",  ".4s",    0, false, 0  },
  { AArch64::LD2Twov2d,         "ld2",  ".2d",    0, false, 0  },
  { AArch64::LD2Twov8b,         "ld2",  ".8b",    0, false, 0  },
  { AArch64::LD2Twov4h,         "ld2",  ".4h",    0, false, 0  },
  { AArch64::LD2Twov2s,         "ld2",  ".2s",    0, false, 0  },
  { AArch64::LD2Twov16b_POST,   "ld2",  ".16b",   1, false, 32 },
  { AArch64::LD2Twov8h_POST,    "ld2",  ".8h",    1, false, 32 },
  { AArch64::LD2Twov4s_POST,    "ld2",  ".4s",    1, false, 32 },
  { AArch64::LD2Twov2d_POST,    "ld2",  ".2d",    1, false, 32 },
  { AArch64::LD2Twov8b_POST,    "ld2",  ".8b",    1, false, 16 },
  { AArch64::LD2Twov4h_POST,    "ld2",  ".4h",    1, false, 16 },
  { AArch64::LD2Twov2s_POST,    "ld2",  ".2s",    1, false, 16 },
  { AArch64::LD3i8,             "ld3",  ".b",     1, true,  0  },
  { AArch64::LD3i16,            "ld3",  ".h",     1, true,  0  },
  { AArch64::LD3i32,            "ld3",  ".s",     1, true,  0  },
  { AArch64::LD3i64,            "ld3",  ".d",     1, true,  0  },
  { AArch64::LD3i8_POST,        "ld3",  ".b",     2, true,  3  },
  { AArch64::LD3i16_POST,       "ld3",  ".h",     2, true,  6  },
  { AArch64::LD3i32_POST,       "ld3",  ".s",     2, true,  12  },
  { AArch64::LD3i64_POST,       "ld3",  ".d",     2, true,  24  },
  { AArch64::LD3Rv16b,          "ld3r", ".16b",   0, false, 0  },
  { AArch64::LD3Rv8h,           "ld3r", ".8h",    0, false, 0  },
  { AArch64::LD3Rv4s,           "ld3r", ".4s",    0, false, 0  },
  { AArch64::LD3Rv2d,           "ld3r", ".2d",    0, false, 0  },
  { AArch64::LD3Rv8b,           "ld3r", ".8b",    0, false, 0  },
  { AArch64::LD3Rv4h,           "ld3r", ".4h",    0, false, 0  },
  { AArch64::LD3Rv2s,           "ld3r", ".2s",    0, false, 0  },
  { AArch64::LD3Rv1d,           "ld3r", ".1d",    0, false, 0  },
  { AArch64::LD3Rv16b_POST,     "ld3r", ".16b",   1, false, 3  },
  { AArch64::LD3Rv8h_POST,      "ld3r", ".8h",    1, false, 6  },
  { AArch64::LD3Rv4s_POST,      "ld3r", ".4s",    1, false, 12 },
  { AArch64::LD3Rv2d_POST,      "ld3r", ".2d",    1, false, 24 },
  { AArch64::LD3Rv8b_POST,      "ld3r", ".8b",    1, false, 3  },
  { AArch64::LD3Rv4h_POST,      "ld3r", ".4h",    1, false, 6  },
  { AArch64::LD3Rv2s_POST,      "ld3r", ".2s",    1, false, 12 },
  { AArch64::LD3Rv1d_POST,      "ld3r", ".1d",    1, false, 24 },
  { AArch64::LD3Threev16b,      "ld3",  ".16b",   0, false, 0  },
  { AArch64::LD3Threev8h,       "ld3",  ".8h",    0, false, 0  },
  { AArch64::LD3Threev4s,       "ld3",  ".4s",    0, false, 0  },
  { AArch64::LD3Threev2d,       "ld3",  ".2d",    0, false, 0  },
  { AArch64::LD3Threev8b,       "ld3",  ".8b",    0, false, 0  },
  { AArch64::LD3Threev4h,       "ld3",  ".4h",    0, false, 0  },
  { AArch64::LD3Threev2s,       "ld3",  ".2s",    0, false, 0  },
  { AArch64::LD3Threev16b_POST, "ld3",  ".16b",   1, false, 48 },
  { AArch64::LD3Threev8h_POST,  "ld3",  ".8h",    1, false, 48 },
  { AArch64::LD3Threev4s_POST,  "ld3",  ".4s",    1, false, 48 },
  { AArch64::LD3Threev2d_POST,  "ld3",  ".2d",    1, false, 48 },
  { AArch64::LD3Threev8b_POST,  "ld3",  ".8b",    1, false, 24 },
  { AArch64::LD3Threev4h_POST,  "ld3",  ".4h",    1, false, 24 },
  { AArch64::LD3Threev2s_POST,  "ld3",  ".2s",    1, false, 24 },
  { AArch64::LD4i8,             "ld4",  ".b",     1, true,  0  },
  { AArch64::LD4i16,            "ld4",  ".h",     1, true,  0  },
  { AArch64::LD4i32,            "ld4",  ".s",     1, true,  0  },
  { AArch64::LD4i64,            "ld4",  ".d",     1, true,  0  },
  { AArch64::LD4i8_POST,        "ld4",  ".b",     2, true,  4  },
  { AArch64::LD4i16_POST,       "ld4",  ".h",     2, true,  8  },
  { AArch64::LD4i32_POST,       "ld4",  ".s",     2, true,  16 },
  { AArch64::LD4i64_POST,       "ld4",  ".d",     2, true,  32 },
  { AArch64::LD4Rv16b,          "ld4r", ".16b",   0, false, 0  },
  { AArch64::LD4Rv8h,           "ld4r", ".8h",    0, false, 0  },
  { AArch64::LD4Rv4s,           "ld4r", ".4s",    0, false, 0  },
  { AArch64::LD4Rv2d,           "ld4r", ".2d",    0, false, 0  },
  { AArch64::LD4Rv8b,           "ld4r", ".8b",    0, false, 0  },
  { AArch64::LD4Rv4h,           "ld4r", ".4h",    0, false, 0  },
  { AArch64::LD4Rv2s,           "ld4r", ".2s",    0, false, 0  },
  { AArch64::LD4Rv1d,           "ld4r", ".1d",    0, false, 0  },
  { AArch64::LD4Rv16b_POST,     "ld4r", ".16b",   1, false, 4  },
  { AArch64::LD4Rv8h_POST,      "ld4r", ".8h",    1, false, 8  },
  { AArch64::LD4Rv4s_POST,      "ld4r", ".4s",    1, false, 16 },
  { AArch64::LD4Rv2d_POST,      "ld4r", ".2d",    1, false, 32 },
  { AArch64::LD4Rv8b_POST,      "ld4r", ".8b",    1, false, 4  },
  { AArch64::LD4Rv4h_POST,      "ld4r", ".4h",    1, false, 8  },
  { AArch64::LD4Rv2s_POST,      "ld4r", ".2s",    1, false, 16 },
  { AArch64::LD4Rv1d_POST,      "ld4r", ".1d",    1, false, 32 },
  { AArch64::LD4Fourv16b,       "ld4",  ".16b",   0, false, 0  },
  { AArch64::LD4Fourv8h,        "ld4",  ".8h",    0, false, 0  },
  { AArch64::LD4Fourv4s,        "ld4",  ".4s",    0, false, 0  },
  { AArch64::LD4Fourv2d,        "ld4",  ".2d",    0, false, 0  },
  { AArch64::LD4Fourv8b,        "ld4",  ".8b",    0, false, 0  },
  { AArch64::LD4Fourv4h,        "ld4",  ".4h",    0, false, 0  },
  { AArch64::LD4Fourv2s,        "ld4",  ".2s",    0, false, 0  },
  { AArch64::LD4Fourv16b_POST,  "ld4",  ".16b",   1, false, 64 },
  { AArch64::LD4Fourv8h_POST,   "ld4",  ".8h",    1, false, 64 },
  { AArch64::LD4Fourv4s_POST,   "ld4",  ".4s",    1, false, 64 },
  { AArch64::LD4Fourv2d_POST,   "ld4",  ".2d",    1, false, 64 },
  { AArch64::LD4Fourv8b_POST,   "ld4",  ".8b",    1, false, 32 },
  { AArch64::LD4Fourv4h_POST,   "ld4",  ".4h",    1, false, 32 },
  { AArch64::LD4Fourv2s_POST,   "ld4",  ".2s",    1, false, 32 },
  { AArch64::ST1i8,             "st1",  ".b",     0, true,  0  },
  { AArch64::ST1i16,            "st1",  ".h",     0, true,  0  },
  { AArch64::ST1i32,            "st1",  ".s",     0, true,  0  },
  { AArch64::ST1i64,            "st1",  ".d",     0, true,  0  },
  { AArch64::ST1i8_POST,        "st1",  ".b",     1, true,  1  },
  { AArch64::ST1i16_POST,       "st1",  ".h",     1, true,  2  },
  { AArch64::ST1i32_POST,       "st1",  ".s",     1, true,  4  },
  { AArch64::ST1i64_POST,       "st1",  ".d",     1, true,  8  },
  { AArch64::ST1Onev16b,        "st1",  ".16b",   0, false, 0  },
  { AArch64::ST1Onev8h,         "st1",  ".8h",    0, false, 0  },
  { AArch64::ST1Onev4s,         "st1",  ".4s",    0, false, 0  },
  { AArch64::ST1Onev2d,         "st1",  ".2d",    0, false, 0  },
  { AArch64::ST1Onev8b,         "st1",  ".8b",    0, false, 0  },
  { AArch64::ST1Onev4h,         "st1",  ".4h",    0, false, 0  },
  { AArch64::ST1Onev2s,         "st1",  ".2s",    0, false, 0  },
  { AArch64::ST1Onev1d,         "st1",  ".1d",    0, false, 0  },
  { AArch64::ST1Onev16b_POST,   "st1",  ".16b",   1, false, 16 },
  { AArch64::ST1Onev8h_POST,    "st1",  ".8h",    1, false, 16 },
  { AArch64::ST1Onev4s_POST,    "st1",  ".4s",    1, false, 16 },
  { AArch64::ST1Onev2d_POST,    "st1",  ".2d",    1, false, 16 },
  { AArch64::ST1Onev8b_POST,    "st1",  ".8b",    1, false, 8  },
  { AArch64::ST1Onev4h_POST,    "st1",  ".4h",    1, false, 8  },
  { AArch64::ST1Onev2s_POST,    "st1",  ".2s",    1, false, 8  },
  { AArch64::ST1Onev1d_POST,    "st1",  ".1d",    1, false, 8  },
  { AArch64::ST1Twov16b,        "st1",  ".16b",   0, false, 0  },
  { AArch64::ST1Twov8h,         "st1",  ".8h",    0, false, 0  },
  { AArch64::ST1Twov4s,         "st1",  ".4s",    0, false, 0  },
  { AArch64::ST1Twov2d,         "st1",  ".2d",    0, false, 0  },
  { AArch64::ST1Twov8b,         "st1",  ".8b",    0, false, 0  },
  { AArch64::ST1Twov4h,         "st1",  ".4h",    0, false, 0  },
  { AArch64::ST1Twov2s,         "st1",  ".2s",    0, false, 0  },
  { AArch64::ST1Twov1d,         "st1",  ".1d",    0, false, 0  },
  { AArch64::ST1Twov16b_POST,   "st1",  ".16b",   1, false, 32 },
  { AArch64::ST1Twov8h_POST,    "st1",  ".8h",    1, false, 32 },
  { AArch64::ST1Twov4s_POST,    "st1",  ".4s",    1, false, 32 },
  { AArch64::ST1Twov2d_POST,    "st1",  ".2d",    1, false, 32 },
  { AArch64::ST1Twov8b_POST,    "st1",  ".8b",    1, false, 16 },
  { AArch64::ST1Twov4h_POST,    "st1",  ".4h",    1, false, 16 },
  { AArch64::ST1Twov2s_POST,    "st1",  ".2s",    1, false, 16 },
  { AArch64::ST1Twov1d_POST,    "st1",  ".1d",    1, false, 16 },
  { AArch64::ST1Threev16b,      "st1",  ".16b",   0, false, 0  },
  { AArch64::ST1Threev8h,       "st1",  ".8h",    0, false, 0  },
  { AArch64::ST1Threev4s,       "st1",  ".4s",    0, false, 0  },
  { AArch64::ST1Threev2d,       "st1",  ".2d",    0, false, 0  },
  { AArch64::ST1Threev8b,       "st1",  ".8b",    0, false, 0  },
  { AArch64::ST1Threev4h,       "st1",  ".4h",    0, false, 0  },
  { AArch64::ST1Threev2s,       "st1",  ".2s",    0, false, 0  },
  { AArch64::ST1Threev1d,       "st1",  ".1d",    0, false, 0  },
  { AArch64::ST1Threev16b_POST, "st1",  ".16b",   1, false, 48 },
  { AArch64::ST1Threev8h_POST,  "st1",  ".8h",    1, false, 48 },
  { AArch64::ST1Threev4s_POST,  "st1",  ".4s",    1, false, 48 },
  { AArch64::ST1Threev2d_POST,  "st1",  ".2d",    1, false, 48 },
  { AArch64::ST1Threev8b_POST,  "st1",  ".8b",    1, false, 24 },
  { AArch64::ST1Threev4h_POST,  "st1",  ".4h",    1, false, 24 },
  { AArch64::ST1Threev2s_POST,  "st1",  ".2s",    1, false, 24 },
  { AArch64::ST1Threev1d_POST,  "st1",  ".1d",    1, false, 24 },
  { AArch64::ST1Fourv16b,       "st1",  ".16b",   0, false, 0  },
  { AArch64::ST1Fourv8h,        "st1",  ".8h",    0, false, 0  },
  { AArch64::ST1Fourv4s,        "st1",  ".4s",    0, false, 0  },
  { AArch64::ST1Fourv2d,        "st1",  ".2d",    0, false, 0  },
  { AArch64::ST1Fourv8b,        "st1",  ".8b",    0, false, 0  },
  { AArch64::ST1Fourv4h,        "st1",  ".4h",    0, false, 0  },
  { AArch64::ST1Fourv2s,        "st1",  ".2s",    0, false, 0  },
  { AArch64::ST1Fourv1d,        "st1",  ".1d",    0, false, 0  },
  { AArch64::ST1Fourv16b_POST,  "st1",  ".16b",   1, false, 64 },
  { AArch64::ST1Fourv8h_POST,   "st1",  ".8h",    1, false, 64 },
  { AArch64::ST1Fourv4s_POST,   "st1",  ".4s",    1, false, 64 },
  { AArch64::ST1Fourv2d_POST,   "st1",  ".2d",    1, false, 64 },
  { AArch64::ST1Fourv8b_POST,   "st1",  ".8b",    1, false, 32 },
  { AArch64::ST1Fourv4h_POST,   "st1",  ".4h",    1, false, 32 },
  { AArch64::ST1Fourv2s_POST,   "st1",  ".2s",    1, false, 32 },
  { AArch64::ST1Fourv1d_POST,   "st1",  ".1d",    1, false, 32 },
  { AArch64::ST2i8,             "st2",  ".b",     0, true,  0  },
  { AArch64::ST2i16,            "st2",  ".h",     0, true,  0  },
  { AArch64::ST2i32,            "st2",  ".s",     0, true,  0  },
  { AArch64::ST2i64,            "st2",  ".d",     0, true,  0  },
  { AArch64::ST2i8_POST,        "st2",  ".b",     1, true,  2  },
  { AArch64::ST2i16_POST,       "st2",  ".h",     1, true,  4  },
  { AArch64::ST2i32_POST,       "st2",  ".s",     1, true,  8  },
  { AArch64::ST2i64_POST,       "st2",  ".d",     1, true,  16 },
  { AArch64::ST2Twov16b,        "st2",  ".16b",   0, false, 0  },
  { AArch64::ST2Twov8h,         "st2",  ".8h",    0, false, 0  },
  { AArch64::ST2Twov4s,         "st2",  ".4s",    0, false, 0  },
  { AArch64::ST2Twov2d,         "st2",  ".2d",    0, false, 0  },
  { AArch64::ST2Twov8b,         "st2",  ".8b",    0, false, 0  },
  { AArch64::ST2Twov4h,         "st2",  ".4h",    0, false, 0  },
  { AArch64::ST2Twov2s,         "st2",  ".2s",    0, false, 0  },
  { AArch64::ST2Twov16b_POST,   "st2",  ".16b",   1, false, 32 },
  { AArch64::ST2Twov8h_POST,    "st2",  ".8h",    1, false, 32 },
  { AArch64::ST2Twov4s_POST,    "st2",  ".4s",    1, false, 32 },
  { AArch64::ST2Twov2d_POST,    "st2",  ".2d",    1, false, 32 },
  { AArch64::ST2Twov8b_POST,    "st2",  ".8b",    1, false, 16 },
  { AArch64::ST2Twov4h_POST,    "st2",  ".4h",    1, false, 16 },
  { AArch64::ST2Twov2s_POST,    "st2",  ".2s",    1, false, 16 },
  { AArch64::ST3i8,             "st3",  ".b",     0, true,  0  },
  { AArch64::ST3i16,            "st3",  ".h",     0, true,  0  },
  { AArch64::ST3i32,            "st3",  ".s",     0, true,  0  },
  { AArch64::ST3i64,            "st3",  ".d",     0, true,  0  },
  { AArch64::ST3i8_POST,        "st3",  ".b",     1, true,  3  },
  { AArch64::ST3i16_POST,       "st3",  ".h",     1, true,  6  },
  { AArch64::ST3i32_POST,       "st3",  ".s",     1, true,  12 },
  { AArch64::ST3i64_POST,       "st3",  ".d",     1, true,  24 },
  { AArch64::ST3Threev16b,      "st3",  ".16b",   0, false, 0  },
  { AArch64::ST3Threev8h,       "st3",  ".8h",    0, false, 0  },
  { AArch64::ST3Threev4s,       "st3",  ".4s",    0, false, 0  },
  { AArch64::ST3Threev2d,       "st3",  ".2d",    0, false, 0  },
  { AArch64::ST3Threev8b,       "st3",  ".8b",    0, false, 0  },
  { AArch64::ST3Threev4h,       "st3",  ".4h",    0, false, 0  },
  { AArch64::ST3Threev2s,       "st3",  ".2s",    0, false, 0  },
  { AArch64::ST3Threev16b_POST, "st3",  ".16b",   1, false, 48 },
  { AArch64::ST3Threev8h_POST,  "st3",  ".8h",    1, false, 48 },
  { AArch64::ST3Threev4s_POST,  "st3",  ".4s",    1, false, 48 },
  { AArch64::ST3Threev2d_POST,  "st3",  ".2d",    1, false, 48 },
  { AArch64::ST3Threev8b_POST,  "st3",  ".8b",    1, false, 24 },
  { AArch64::ST3Threev4h_POST,  "st3",  ".4h",    1, false, 24 },
  { AArch64::ST3Threev2s_POST,  "st3",  ".2s",    1, false, 24 },
  { AArch64::ST4i8,             "st4",  ".b",     0, true,  0  },
  { AArch64::ST4i16,            "st4",  ".h",     0, true,  0  },
  { AArch64::ST4i32,            "st4",  ".s",     0, true,  0  },
  { AArch64::ST4i64,            "st4",  ".d",     0, true,  0  },
  { AArch64::ST4i8_POST,        "st4",  ".b",     1, true,  4  },
  { AArch64::ST4i16_POST,       "st4",  ".h",     1, true,  8  },
  { AArch64::ST4i32_POST,       "st4",  ".s",     1, true,  16 },
  { AArch64::ST4i64_POST,       "st4",  ".d",     1, true,  32 },
  { AArch64::ST4Fourv16b,       "st4",  ".16b",   0, false, 0  },
  { AArch64::ST4Fourv8h,        "st4",  ".8h",    0, false, 0  },
  { AArch64::ST4Fourv4s,        "st4",  ".4s",    0, false, 0  },
  { AArch64::ST4Fourv2d,        "st4",  ".2d",    0, false, 0  },
  { AArch64::ST4Fourv8b,        "st4",  ".8b",    0, false, 0  },
  { AArch64::ST4Fourv4h,        "st4",  ".4h",    0, false, 0  },
  { AArch64::ST4Fourv2s,        "st4",  ".2s",    0, false, 0  },
  { AArch64::ST4Fourv16b_POST,  "st4",  ".16b",   1, false, 64 },
  { AArch64::ST4Fourv8h_POST,   "st4",  ".8h",    1, false, 64 },
  { AArch64::ST4Fourv4s_POST,   "st4",  ".4s",    1, false, 64 },
  { AArch64::ST4Fourv2d_POST,   "st4",  ".2d",    1, false, 64 },
  { AArch64::ST4Fourv8b_POST,   "st4",  ".8b",    1, false, 32 },
  { AArch64::ST4Fourv4h_POST,   "st4",  ".4h",    1, false, 32 },
  { AArch64::ST4Fourv2s_POST,   "st4",  ".2s",    1, false, 32 },
};

static const LdStNInstrDesc *getLdStNInstrDesc(unsigned Opcode) {
  unsigned Idx;
  for (Idx = 0; Idx != array_lengthof(LdStNInstInfo); ++Idx)
    if (LdStNInstInfo[Idx].Opcode == Opcode)
      return &LdStNInstInfo[Idx];

  return nullptr;
}

void AArch64AppleInstPrinter::printInst(const MCInst *MI, raw_ostream &O,
                                        StringRef Annot,
                                        const MCSubtargetInfo &STI) {
  unsigned Opcode = MI->getOpcode();
  StringRef Layout, Mnemonic;

  bool IsTbx;
  if (isTblTbxInstruction(MI->getOpcode(), Layout, IsTbx)) {
    O << "\t" << (IsTbx ? "tbx" : "tbl") << Layout << '\t'
      << getRegisterName(MI->getOperand(0).getReg(), AArch64::vreg) << ", ";

    unsigned ListOpNum = IsTbx ? 2 : 1;
    printVectorList(MI, ListOpNum, STI, O, "");

    O << ", "
      << getRegisterName(MI->getOperand(ListOpNum + 1).getReg(), AArch64::vreg);
    printAnnotation(O, Annot);
    return;
  }

  if (const LdStNInstrDesc *LdStDesc = getLdStNInstrDesc(Opcode)) {
    O << "\t" << LdStDesc->Mnemonic << LdStDesc->Layout << '\t';

    // Now onto the operands: first a vector list with possible lane
    // specifier. E.g. { v0 }[2]
    int OpNum = LdStDesc->ListOperand;
    printVectorList(MI, OpNum++, STI, O, "");

    if (LdStDesc->HasLane)
      O << '[' << MI->getOperand(OpNum++).getImm() << ']';

    // Next the address: [xN]
    unsigned AddrReg = MI->getOperand(OpNum++).getReg();
    O << ", [" << getRegisterName(AddrReg) << ']';

    // Finally, there might be a post-indexed offset.
    if (LdStDesc->NaturalOffset != 0) {
      unsigned Reg = MI->getOperand(OpNum++).getReg();
      if (Reg != AArch64::XZR)
        O << ", " << getRegisterName(Reg);
      else {
        assert(LdStDesc->NaturalOffset && "no offset on post-inc instruction?");
        O << ", #" << LdStDesc->NaturalOffset;
      }
    }

    printAnnotation(O, Annot);
    return;
  }

  AArch64InstPrinter::printInst(MI, O, Annot, STI);
}

bool AArch64InstPrinter::printSysAlias(const MCInst *MI,
                                       const MCSubtargetInfo &STI,
                                       raw_ostream &O) {
#ifndef NDEBUG
  unsigned Opcode = MI->getOpcode();
  assert(Opcode == AArch64::SYSxt && "Invalid opcode for SYS alias!");
#endif

  const char *Asm = nullptr;
  const MCOperand &Op1 = MI->getOperand(0);
  const MCOperand &Cn = MI->getOperand(1);
  const MCOperand &Cm = MI->getOperand(2);
  const MCOperand &Op2 = MI->getOperand(3);

  unsigned Op1Val = Op1.getImm();
  unsigned CnVal = Cn.getImm();
  unsigned CmVal = Cm.getImm();
  unsigned Op2Val = Op2.getImm();

  if (CnVal == 7) {
    switch (CmVal) {
    default:
      break;

    // IC aliases
    case 1:
      if (Op1Val == 0 && Op2Val == 0)
        Asm = "ic\tialluis";
      break;
    case 5:
      if (Op1Val == 0 && Op2Val == 0)
        Asm = "ic\tiallu";
      else if (Op1Val == 3 && Op2Val == 1)
        Asm = "ic\tivau";
      break;

    // DC aliases
    case 4:
      if (Op1Val == 3 && Op2Val == 1)
        Asm = "dc\tzva";
      break;
    case 6:
      if (Op1Val == 0 && Op2Val == 1)
        Asm = "dc\tivac";
      if (Op1Val == 0 && Op2Val == 2)
        Asm = "dc\tisw";
      break;
    case 10:
      if (Op1Val == 3 && Op2Val == 1)
        Asm = "dc\tcvac";
      else if (Op1Val == 0 && Op2Val == 2)
        Asm = "dc\tcsw";
      break;
    case 11:
      if (Op1Val == 3 && Op2Val == 1)
        Asm = "dc\tcvau";
      break;
    case 12:
      if (Op1Val == 3 && Op2Val == 1 &&
          (STI.getFeatureBits()[AArch64::HasV8_2aOps]))
        Asm = "dc\tcvap";
      break;
    case 14:
      if (Op1Val == 3 && Op2Val == 1)
        Asm = "dc\tcivac";
      else if (Op1Val == 0 && Op2Val == 2)
        Asm = "dc\tcisw";
      break;

    // AT aliases
    case 8:
      switch (Op1Val) {
      default:
        break;
      case 0:
        switch (Op2Val) {
        default:
          break;
        case 0: Asm = "at\ts1e1r"; break;
        case 1: Asm = "at\ts1e1w"; break;
        case 2: Asm = "at\ts1e0r"; break;
        case 3: Asm = "at\ts1e0w"; break;
        }
        break;
      case 4:
        switch (Op2Val) {
        default:
          break;
        case 0: Asm = "at\ts1e2r"; break;
        case 1: Asm = "at\ts1e2w"; break;
        case 4: Asm = "at\ts12e1r"; break;
        case 5: Asm = "at\ts12e1w"; break;
        case 6: Asm = "at\ts12e0r"; break;
        case 7: Asm = "at\ts12e0w"; break;
        }
        break;
      case 6:
        switch (Op2Val) {
        default:
          break;
        case 0: Asm = "at\ts1e3r"; break;
        case 1: Asm = "at\ts1e3w"; break;
        }
        break;
      }
      break;
    case 9:
      switch (Op1Val) {
      default:
        break;
      case 0:
        if (STI.getFeatureBits()[AArch64::HasV8_2aOps]) {
          switch (Op2Val) {
          default:
            break;
          case 0: Asm = "at\ts1e1rp"; break;
          case 1: Asm = "at\ts1e1wp"; break;
          }
        }
        break;
      }
    }
  } else if (CnVal == 8) {
    // TLBI aliases
    switch (CmVal) {
    default:
      break;
    case 3:
      switch (Op1Val) {
      default:
        break;
      case 0:
        switch (Op2Val) {
        default:
          break;
        case 0: Asm = "tlbi\tvmalle1is"; break;
        case 1: Asm = "tlbi\tvae1is"; break;
        case 2: Asm = "tlbi\taside1is"; break;
        case 3: Asm = "tlbi\tvaae1is"; break;
        case 5: Asm = "tlbi\tvale1is"; break;
        case 7: Asm = "tlbi\tvaale1is"; break;
        }
        break;
      case 4:
        switch (Op2Val) {
        default:
          break;
        case 0: Asm = "tlbi\talle2is"; break;
        case 1: Asm = "tlbi\tvae2is"; break;
        case 4: Asm = "tlbi\talle1is"; break;
        case 5: Asm = "tlbi\tvale2is"; break;
        case 6: Asm = "tlbi\tvmalls12e1is"; break;
        }
        break;
      case 6:
        switch (Op2Val) {
        default:
          break;
        case 0: Asm = "tlbi\talle3is"; break;
        case 1: Asm = "tlbi\tvae3is"; break;
        case 5: Asm = "tlbi\tvale3is"; break;
        }
        break;
      }
      break;
    case 0:
      switch (Op1Val) {
      default:
        break;
      case 4:
        switch (Op2Val) {
        default:
          break;
        case 1: Asm = "tlbi\tipas2e1is"; break;
        case 5: Asm = "tlbi\tipas2le1is"; break;
        }
        break;
      }
      break;
    case 4:
      switch (Op1Val) {
      default:
        break;
      case 4:
        switch (Op2Val) {
        default:
          break;
        case 1: Asm = "tlbi\tipas2e1"; break;
        case 5: Asm = "tlbi\tipas2le1"; break;
        }
        break;
      }
      break;
    case 7:
      switch (Op1Val) {
      default:
        break;
      case 0:
        switch (Op2Val) {
        default:
          break;
        case 0: Asm = "tlbi\tvmalle1"; break;
        case 1: Asm = "tlbi\tvae1"; break;
        case 2: Asm = "tlbi\taside1"; break;
        case 3: Asm = "tlbi\tvaae1"; break;
        case 5: Asm = "tlbi\tvale1"; break;
        case 7: Asm = "tlbi\tvaale1"; break;
        }
        break;
      case 4:
        switch (Op2Val) {
        default:
          break;
        case 0: Asm = "tlbi\talle2"; break;
        case 1: Asm = "tlbi\tvae2"; break;
        case 4: Asm = "tlbi\talle1"; break;
        case 5: Asm = "tlbi\tvale2"; break;
        case 6: Asm = "tlbi\tvmalls12e1"; break;
        }
        break;
      case 6:
        switch (Op2Val) {
        default:
          break;
        case 0: Asm = "tlbi\talle3"; break;
        case 1: Asm = "tlbi\tvae3";  break;
        case 5: Asm = "tlbi\tvale3"; break;
        }
        break;
      }
      break;
    }
  }

  if (Asm) {
    unsigned Reg = MI->getOperand(4).getReg();

    O << '\t' << Asm;
    if (StringRef(Asm).lower().find("all") == StringRef::npos)
      O << ", " << getRegisterName(Reg);
  }

  return Asm != nullptr;
}

void AArch64InstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                      const MCSubtargetInfo &STI,
                                      raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    unsigned Reg = Op.getReg();
    O << getRegisterName(Reg);
  } else if (Op.isImm()) {
    O << '#' << Op.getImm();
  } else {
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    Op.getExpr()->print(O, &MAI);
  }
}

void AArch64InstPrinter::printHexImm(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  O << format("#%#llx", Op.getImm());
}

void AArch64InstPrinter::printPostIncOperand(const MCInst *MI, unsigned OpNo,
                                             unsigned Imm, raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    unsigned Reg = Op.getReg();
    if (Reg == AArch64::XZR)
      O << "#" << Imm;
    else
      O << getRegisterName(Reg);
  } else
    llvm_unreachable("unknown operand kind in printPostIncOperand64");
}

void AArch64InstPrinter::printVRegOperand(const MCInst *MI, unsigned OpNo,
                                          const MCSubtargetInfo &STI,
                                          raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  assert(Op.isReg() && "Non-register vreg operand!");
  unsigned Reg = Op.getReg();
  O << getRegisterName(Reg, AArch64::vreg);
}

void AArch64InstPrinter::printSysCROperand(const MCInst *MI, unsigned OpNo,
                                           const MCSubtargetInfo &STI,
                                           raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  assert(Op.isImm() && "System instruction C[nm] operands must be immediates!");
  O << "c" << Op.getImm();
}

void AArch64InstPrinter::printAddSubImm(const MCInst *MI, unsigned OpNum,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  if (MO.isImm()) {
    unsigned Val = (MO.getImm() & 0xfff);
    assert(Val == MO.getImm() && "Add/sub immediate out of range!");
    unsigned Shift =
        AArch64_AM::getShiftValue(MI->getOperand(OpNum + 1).getImm());
    O << '#' << Val;
    if (Shift != 0)
      printShifter(MI, OpNum + 1, STI, O);

    if (CommentStream)
      *CommentStream << '=' << (Val << Shift) << '\n';
  } else {
    assert(MO.isExpr() && "Unexpected operand type!");
    MO.getExpr()->print(O, &MAI);
    printShifter(MI, OpNum + 1, STI, O);
  }
}

void AArch64InstPrinter::printLogicalImm32(const MCInst *MI, unsigned OpNum,
                                           const MCSubtargetInfo &STI,
                                           raw_ostream &O) {
  uint64_t Val = MI->getOperand(OpNum).getImm();
  O << "#0x";
  O.write_hex(AArch64_AM::decodeLogicalImmediate(Val, 32));
}

void AArch64InstPrinter::printLogicalImm64(const MCInst *MI, unsigned OpNum,
                                           const MCSubtargetInfo &STI,
                                           raw_ostream &O) {
  uint64_t Val = MI->getOperand(OpNum).getImm();
  O << "#0x";
  O.write_hex(AArch64_AM::decodeLogicalImmediate(Val, 64));
}

void AArch64InstPrinter::printShifter(const MCInst *MI, unsigned OpNum,
                                      const MCSubtargetInfo &STI,
                                      raw_ostream &O) {
  unsigned Val = MI->getOperand(OpNum).getImm();
  // LSL #0 should not be printed.
  if (AArch64_AM::getShiftType(Val) == AArch64_AM::LSL &&
      AArch64_AM::getShiftValue(Val) == 0)
    return;
  O << ", " << AArch64_AM::getShiftExtendName(AArch64_AM::getShiftType(Val))
    << " #" << AArch64_AM::getShiftValue(Val);
}

void AArch64InstPrinter::printShiftedRegister(const MCInst *MI, unsigned OpNum,
                                              const MCSubtargetInfo &STI,
                                              raw_ostream &O) {
  O << getRegisterName(MI->getOperand(OpNum).getReg());
  printShifter(MI, OpNum + 1, STI, O);
}

void AArch64InstPrinter::printExtendedRegister(const MCInst *MI, unsigned OpNum,
                                               const MCSubtargetInfo &STI,
                                               raw_ostream &O) {
  O << getRegisterName(MI->getOperand(OpNum).getReg());
  printArithExtend(MI, OpNum + 1, STI, O);
}

void AArch64InstPrinter::printArithExtend(const MCInst *MI, unsigned OpNum,
                                          const MCSubtargetInfo &STI,
                                          raw_ostream &O) {
  unsigned Val = MI->getOperand(OpNum).getImm();
  AArch64_AM::ShiftExtendType ExtType = AArch64_AM::getArithExtendType(Val);
  unsigned ShiftVal = AArch64_AM::getArithShiftValue(Val);

  // If the destination or first source register operand is [W]SP, print
  // UXTW/UXTX as LSL, and if the shift amount is also zero, print nothing at
  // all.
  if (ExtType == AArch64_AM::UXTW || ExtType == AArch64_AM::UXTX) {
    unsigned Dest = MI->getOperand(0).getReg();
    unsigned Src1 = MI->getOperand(1).getReg();
    if ( ((Dest == AArch64::SP || Src1 == AArch64::SP) &&
          ExtType == AArch64_AM::UXTX) ||
         ((Dest == AArch64::WSP || Src1 == AArch64::WSP) &&
          ExtType == AArch64_AM::UXTW) ) {
      if (ShiftVal != 0)
        O << ", lsl #" << ShiftVal;
      return;
    }
  }
  O << ", " << AArch64_AM::getShiftExtendName(ExtType);
  if (ShiftVal != 0)
    O << " #" << ShiftVal;
}

void AArch64InstPrinter::printMemExtend(const MCInst *MI, unsigned OpNum,
                                        raw_ostream &O, char SrcRegKind,
                                        unsigned Width) {
  unsigned SignExtend = MI->getOperand(OpNum).getImm();
  unsigned DoShift = MI->getOperand(OpNum + 1).getImm();

  // sxtw, sxtx, uxtw or lsl (== uxtx)
  bool IsLSL = !SignExtend && SrcRegKind == 'x';
  if (IsLSL)
    O << "lsl";
  else
    O << (SignExtend ? 's' : 'u') << "xt" << SrcRegKind;

  if (DoShift || IsLSL)
    O << " #" << Log2_32(Width / 8);
}

void AArch64InstPrinter::printCondCode(const MCInst *MI, unsigned OpNum,
                                       const MCSubtargetInfo &STI,
                                       raw_ostream &O) {
  AArch64CC::CondCode CC = (AArch64CC::CondCode)MI->getOperand(OpNum).getImm();
  O << AArch64CC::getCondCodeName(CC);
}

void AArch64InstPrinter::printInverseCondCode(const MCInst *MI, unsigned OpNum,
                                              const MCSubtargetInfo &STI,
                                              raw_ostream &O) {
  AArch64CC::CondCode CC = (AArch64CC::CondCode)MI->getOperand(OpNum).getImm();
  O << AArch64CC::getCondCodeName(AArch64CC::getInvertedCondCode(CC));
}

void AArch64InstPrinter::printAMNoIndex(const MCInst *MI, unsigned OpNum,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  O << '[' << getRegisterName(MI->getOperand(OpNum).getReg()) << ']';
}

template<int Scale>
void AArch64InstPrinter::printImmScale(const MCInst *MI, unsigned OpNum,
                                       const MCSubtargetInfo &STI,
                                       raw_ostream &O) {
  O << '#' << Scale * MI->getOperand(OpNum).getImm();
}

void AArch64InstPrinter::printUImm12Offset(const MCInst *MI, unsigned OpNum,
                                           unsigned Scale, raw_ostream &O) {
  const MCOperand MO = MI->getOperand(OpNum);
  if (MO.isImm()) {
    O << "#" << (MO.getImm() * Scale);
  } else {
    assert(MO.isExpr() && "Unexpected operand type!");
    MO.getExpr()->print(O, &MAI);
  }
}

void AArch64InstPrinter::printAMIndexedWB(const MCInst *MI, unsigned OpNum,
                                          unsigned Scale, raw_ostream &O) {
  const MCOperand MO1 = MI->getOperand(OpNum + 1);
  O << '[' << getRegisterName(MI->getOperand(OpNum).getReg());
  if (MO1.isImm()) {
      O << ", #" << (MO1.getImm() * Scale);
  } else {
    assert(MO1.isExpr() && "Unexpected operand type!");
    O << ", ";
    MO1.getExpr()->print(O, &MAI);
  }
  O << ']';
}

void AArch64InstPrinter::printPrefetchOp(const MCInst *MI, unsigned OpNum,
                                         const MCSubtargetInfo &STI,
                                         raw_ostream &O) {
  unsigned prfop = MI->getOperand(OpNum).getImm();
  bool Valid;
  StringRef Name =
      AArch64PRFM::PRFMMapper().toString(prfop, STI.getFeatureBits(), Valid);
  if (Valid)
    O << Name;
  else
    O << '#' << prfop;
}

void AArch64InstPrinter::printPSBHintOp(const MCInst *MI, unsigned OpNum,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  unsigned psbhintop = MI->getOperand(OpNum).getImm();
  bool Valid;
  StringRef Name =
      AArch64PSBHint::PSBHintMapper().toString(psbhintop, STI.getFeatureBits(), Valid);
  if (Valid)
    O << Name;
  else
    O << '#' << psbhintop;
}

void AArch64InstPrinter::printFPImmOperand(const MCInst *MI, unsigned OpNum,
                                           const MCSubtargetInfo &STI,
                                           raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  float FPImm =
      MO.isFPImm() ? MO.getFPImm() : AArch64_AM::getFPImmFloat(MO.getImm());

  // 8 decimal places are enough to perfectly represent permitted floats.
  O << format("#%.8f", FPImm);
}

static unsigned getNextVectorRegister(unsigned Reg, unsigned Stride = 1) {
  while (Stride--) {
    switch (Reg) {
    default:
      llvm_unreachable("Vector register expected!");
    case AArch64::Q0:  Reg = AArch64::Q1;  break;
    case AArch64::Q1:  Reg = AArch64::Q2;  break;
    case AArch64::Q2:  Reg = AArch64::Q3;  break;
    case AArch64::Q3:  Reg = AArch64::Q4;  break;
    case AArch64::Q4:  Reg = AArch64::Q5;  break;
    case AArch64::Q5:  Reg = AArch64::Q6;  break;
    case AArch64::Q6:  Reg = AArch64::Q7;  break;
    case AArch64::Q7:  Reg = AArch64::Q8;  break;
    case AArch64::Q8:  Reg = AArch64::Q9;  break;
    case AArch64::Q9:  Reg = AArch64::Q10; break;
    case AArch64::Q10: Reg = AArch64::Q11; break;
    case AArch64::Q11: Reg = AArch64::Q12; break;
    case AArch64::Q12: Reg = AArch64::Q13; break;
    case AArch64::Q13: Reg = AArch64::Q14; break;
    case AArch64::Q14: Reg = AArch64::Q15; break;
    case AArch64::Q15: Reg = AArch64::Q16; break;
    case AArch64::Q16: Reg = AArch64::Q17; break;
    case AArch64::Q17: Reg = AArch64::Q18; break;
    case AArch64::Q18: Reg = AArch64::Q19; break;
    case AArch64::Q19: Reg = AArch64::Q20; break;
    case AArch64::Q20: Reg = AArch64::Q21; break;
    case AArch64::Q21: Reg = AArch64::Q22; break;
    case AArch64::Q22: Reg = AArch64::Q23; break;
    case AArch64::Q23: Reg = AArch64::Q24; break;
    case AArch64::Q24: Reg = AArch64::Q25; break;
    case AArch64::Q25: Reg = AArch64::Q26; break;
    case AArch64::Q26: Reg = AArch64::Q27; break;
    case AArch64::Q27: Reg = AArch64::Q28; break;
    case AArch64::Q28: Reg = AArch64::Q29; break;
    case AArch64::Q29: Reg = AArch64::Q30; break;
    case AArch64::Q30: Reg = AArch64::Q31; break;
    // Vector lists can wrap around.
    case AArch64::Q31:
      Reg = AArch64::Q0;
      break;
    }
  }
  return Reg;
}

template<unsigned size>
void AArch64InstPrinter::printGPRSeqPairsClassOperand(const MCInst *MI,
                                                   unsigned OpNum,
                                                   const MCSubtargetInfo &STI,
                                                   raw_ostream &O) {
  static_assert(size == 64 || size == 32,
                "Template parameter must be either 32 or 64");
  unsigned Reg = MI->getOperand(OpNum).getReg();

  unsigned Sube = (size == 32) ? AArch64::sube32 : AArch64::sube64;
  unsigned Subo = (size == 32) ? AArch64::subo32 : AArch64::subo64;

  unsigned Even = MRI.getSubReg(Reg,  Sube);
  unsigned Odd = MRI.getSubReg(Reg,  Subo);
  O << getRegisterName(Even) << ", " << getRegisterName(Odd);
}

void AArch64InstPrinter::printVectorList(const MCInst *MI, unsigned OpNum,
                                         const MCSubtargetInfo &STI,
                                         raw_ostream &O,
                                         StringRef LayoutSuffix) {
  unsigned Reg = MI->getOperand(OpNum).getReg();

  O << "{ ";

  // Work out how many registers there are in the list (if there is an actual
  // list).
  unsigned NumRegs = 1;
  if (MRI.getRegClass(AArch64::DDRegClassID).contains(Reg) ||
      MRI.getRegClass(AArch64::QQRegClassID).contains(Reg))
    NumRegs = 2;
  else if (MRI.getRegClass(AArch64::DDDRegClassID).contains(Reg) ||
           MRI.getRegClass(AArch64::QQQRegClassID).contains(Reg))
    NumRegs = 3;
  else if (MRI.getRegClass(AArch64::DDDDRegClassID).contains(Reg) ||
           MRI.getRegClass(AArch64::QQQQRegClassID).contains(Reg))
    NumRegs = 4;

  // Now forget about the list and find out what the first register is.
  if (unsigned FirstReg = MRI.getSubReg(Reg, AArch64::dsub0))
    Reg = FirstReg;
  else if (unsigned FirstReg = MRI.getSubReg(Reg, AArch64::qsub0))
    Reg = FirstReg;

  // If it's a D-reg, we need to promote it to the equivalent Q-reg before
  // printing (otherwise getRegisterName fails).
  if (MRI.getRegClass(AArch64::FPR64RegClassID).contains(Reg)) {
    const MCRegisterClass &FPR128RC =
        MRI.getRegClass(AArch64::FPR128RegClassID);
    Reg = MRI.getMatchingSuperReg(Reg, AArch64::dsub, &FPR128RC);
  }

  for (unsigned i = 0; i < NumRegs; ++i, Reg = getNextVectorRegister(Reg)) {
    O << getRegisterName(Reg, AArch64::vreg) << LayoutSuffix;
    if (i + 1 != NumRegs)
      O << ", ";
  }

  O << " }";
}

void
AArch64InstPrinter::printImplicitlyTypedVectorList(const MCInst *MI,
                                                   unsigned OpNum,
                                                   const MCSubtargetInfo &STI,
                                                   raw_ostream &O) {
  printVectorList(MI, OpNum, STI, O, "");
}

template <unsigned NumLanes, char LaneKind>
void AArch64InstPrinter::printTypedVectorList(const MCInst *MI, unsigned OpNum,
                                              const MCSubtargetInfo &STI,
                                              raw_ostream &O) {
  std::string Suffix(".");
  if (NumLanes)
    Suffix += itostr(NumLanes) + LaneKind;
  else
    Suffix += LaneKind;

  printVectorList(MI, OpNum, STI, O, Suffix);
}

void AArch64InstPrinter::printVectorIndex(const MCInst *MI, unsigned OpNum,
                                          const MCSubtargetInfo &STI,
                                          raw_ostream &O) {
  O << "[" << MI->getOperand(OpNum).getImm() << "]";
}

void AArch64InstPrinter::printAlignedLabel(const MCInst *MI, unsigned OpNum,
                                           const MCSubtargetInfo &STI,
                                           raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNum);

  // If the label has already been resolved to an immediate offset (say, when
  // we're running the disassembler), just print the immediate.
  if (Op.isImm()) {
    O << "#" << (Op.getImm() * 4);
    return;
  }

  // If the branch target is simply an address then print it in hex.
  const MCConstantExpr *BranchTarget =
      dyn_cast<MCConstantExpr>(MI->getOperand(OpNum).getExpr());
  int64_t Address;
  if (BranchTarget && BranchTarget->evaluateAsAbsolute(Address)) {
    O << "0x";
    O.write_hex(Address);
  } else {
    // Otherwise, just print the expression.
    MI->getOperand(OpNum).getExpr()->print(O, &MAI);
  }
}

void AArch64InstPrinter::printAdrpLabel(const MCInst *MI, unsigned OpNum,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNum);

  // If the label has already been resolved to an immediate offset (say, when
  // we're running the disassembler), just print the immediate.
  if (Op.isImm()) {
    O << "#" << (Op.getImm() * (1 << 12));
    return;
  }

  // Otherwise, just print the expression.
  MI->getOperand(OpNum).getExpr()->print(O, &MAI);
}

void AArch64InstPrinter::printBarrierOption(const MCInst *MI, unsigned OpNo,
                                            const MCSubtargetInfo &STI,
                                            raw_ostream &O) {
  unsigned Val = MI->getOperand(OpNo).getImm();
  unsigned Opcode = MI->getOpcode();

  bool Valid;
  StringRef Name;
  if (Opcode == AArch64::ISB)
    Name = AArch64ISB::ISBMapper().toString(Val, STI.getFeatureBits(),
                                            Valid);
  else
    Name = AArch64DB::DBarrierMapper().toString(Val, STI.getFeatureBits(),
                                                Valid);
  if (Valid)
    O << Name;
  else
    O << "#" << Val;
}

void AArch64InstPrinter::printMRSSystemRegister(const MCInst *MI, unsigned OpNo,
                                                const MCSubtargetInfo &STI,
                                                raw_ostream &O) {
  unsigned Val = MI->getOperand(OpNo).getImm();

  auto Mapper = AArch64SysReg::MRSMapper();
  std::string Name = Mapper.toString(Val, STI.getFeatureBits());

  O << StringRef(Name).upper();
}

void AArch64InstPrinter::printMSRSystemRegister(const MCInst *MI, unsigned OpNo,
                                                const MCSubtargetInfo &STI,
                                                raw_ostream &O) {
  unsigned Val = MI->getOperand(OpNo).getImm();

  auto Mapper = AArch64SysReg::MSRMapper();
  std::string Name = Mapper.toString(Val, STI.getFeatureBits());

  O << StringRef(Name).upper();
}

void AArch64InstPrinter::printSystemPStateField(const MCInst *MI, unsigned OpNo,
                                                const MCSubtargetInfo &STI,
                                                raw_ostream &O) {
  unsigned Val = MI->getOperand(OpNo).getImm();

  bool Valid;
  StringRef Name =
      AArch64PState::PStateMapper().toString(Val, STI.getFeatureBits(), Valid);
  if (Valid)
    O << Name.upper();
  else
    O << "#" << Val;
}

void AArch64InstPrinter::printSIMDType10Operand(const MCInst *MI, unsigned OpNo,
                                                const MCSubtargetInfo &STI,
                                                raw_ostream &O) {
  unsigned RawVal = MI->getOperand(OpNo).getImm();
  uint64_t Val = AArch64_AM::decodeAdvSIMDModImmType10(RawVal);
  O << format("#%#016llx", Val);
}
