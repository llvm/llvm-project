//===-- PISAInstPrinter.cpp - Output PISA MCInsts as ASM ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISAInstPrinter.h"
#include "MCTargetDesc/PISARegEncoder.h"
#include "PISA.h"
#include "PISABaseInfo.h"
#include "PISAEnum.h"
#include "PISAInstrInfo.h"
#include "PISAMCExpr.h"
#include "PISAMCInstLower.h"
#include "PISAUtils.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/IR/PISAIntrinsicUtils.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;
using namespace llvm::PISA;

#define DEBUG_TYPE "asm-printer"

static cl::opt<bool>
    AlwaysPrintDefaultValue("pisa-always-print-default-value",
                            cl::desc("Always print default value"),
                            cl::init(false), cl::ReallyHidden);

// Include the auto-generated portion of the assembly writer.
#include "PISAGenAsmWriter.inc"

void PISAInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                StringRef Annot, const MCSubtargetInfo &STI,
                                raw_ostream &OS) {
  this->STI = &STI;

  printInstruction(MI, Address, OS);

  printAnnotation(OS, Annot);
}

void PISAInstPrinter::printRegName(raw_ostream &OS, MCRegister Reg) {
  if (MCRegister::isPhysicalRegister(Reg)) {
    OS << getRegisterName(Reg);
  } else {
    auto [Prefix, Idx] = RegEncoder::decodeVirtualRegister(Reg);
    OS << Prefix << Idx;
  }
}

void PISAInstPrinter::printImm1Opnd(const MCInst *MCI, unsigned OpNo,
                                    raw_ostream &OS) {
  const MCOperand &MCOp = MCI->getOperand(OpNo);
  OS << formatImm(MCOp.getImm());
}

void PISAInstPrinter::printImm8Opnd(const MCInst *MCI, unsigned OpNo,
                                    raw_ostream &OS) {
  const MCOperand &MCOp = MCI->getOperand(OpNo);
  OS << formatImm(MCOp.getImm());
}

void PISAInstPrinter::printImm16Opnd(const MCInst *MCI, unsigned OpNo,
                                     raw_ostream &OS) {
  const MCOperand &MCOp = MCI->getOperand(OpNo);
  OS << formatImm(MCOp.getImm());
}

void PISAInstPrinter::printImm32Opnd(const MCInst *MCI, unsigned OpNo,
                                     raw_ostream &OS) {
  const MCOperand &MCOp = MCI->getOperand(OpNo);
  OS << formatImm(MCOp.getImm());
}

void PISAInstPrinter::printImm64Opnd(const MCInst *MCI, unsigned OpNo,
                                     raw_ostream &OS) {
  const MCOperand &MCOp = MCI->getOperand(OpNo);
  OS << formatImm(MCOp.getImm());
}

void PISAInstPrinter::printImm128Opnd(const MCInst *MCI, unsigned OpNo,
                                      raw_ostream &OS) {
  const MCOperand &MCOp = MCI->getOperand(OpNo);
  OS << formatImm(MCOp.getImm());
}

void PISAInstPrinter::printFpImm16Opnd(const MCInst *MCI, unsigned OpNo,
                                       raw_ostream &OS) {
  const MCOperand &MCOp = MCI->getOperand(OpNo);
  OS << format_hex(MCOp.getHFPImm(), /*Width=*/6, /*Upper=*/true);
}

void PISAInstPrinter::printFpImm32Opnd(const MCInst *MCI, unsigned OpNo,
                                       raw_ostream &OS) {
  const MCOperand &MCOp = MCI->getOperand(OpNo);
  OS << format_hex(MCOp.getSFPImm(), /*Width=*/10, /*Upper=*/true);
}

void PISAInstPrinter::printFpImm64Opnd(const MCInst *MCI, unsigned OpNo,
                                       raw_ostream &OS) {
  const MCOperand &MCOp = MCI->getOperand(OpNo);
  OS << format_hex(MCOp.getDFPImm(), /*Width=*/18, /*Upper=*/true);
}

void PISAInstPrinter::printRegOpnd(unsigned RCID, const MCInst *MCI,
                                   unsigned OpNo, raw_ostream &OS) {
  const MCOperand &MCOp = MCI->getOperand(OpNo);
  printRegName(OS, MCOp.getReg());
  printSwizzle(MCI, OpNo, OS);
}

void PISAInstPrinter::printBrTargetOpnd(const MCInst *MCI, uint64_t Address,
                                        unsigned OpNo, raw_ostream &OS) {
  const MCOperand &MCOp = MCI->getOperand(OpNo);
  MAI.printExpr(OS, *MCOp.getExpr());
}

void PISAInstPrinter::printLocalVariableOpnd(const MCInst *MCI, unsigned OpNo,
                                             raw_ostream &OS) {
  assert(PISAMCInstLower::isVariableRef(*MCI, OpNo));
  printOperand(MCI, OpNo, OS);
}

void PISAInstPrinter::printGlobalVariableOpnd(const MCInst *MCI,
                                              uint64_t Address, unsigned OpNo,
                                              raw_ostream &OS) {
  const MCOperand &MCOp = MCI->getOperand(OpNo);
  MAI.printExpr(OS, *MCOp.getExpr());
}

void PISAInstPrinter::printMemScopeOpnd(const MCInst *MCI, unsigned OpNo,
                                        raw_ostream &OS) {
  static const char *MemScopeStrs[] = {
      ".system",    // pisa::MemoryScope::system
      ".gpu",       // pisa::MemoryScope::gpu
      ".workgroup", // pisa::MemoryScope::workgroup
      ".subgroup",  // pisa::MemoryScope::subgroup
  };
  const MCOperand &MCOp = MCI->getOperand(OpNo);
  const unsigned Scope = MCOp.getImm();
  assert(Scope <= pisa::MemoryScope::subgroup && "Invalid memory scope");
  OS << MemScopeStrs[Scope];
}

void PISAInstPrinter::printSwizzle(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O) {
  auto Swizzle = PISAMCInstLower::getSwizzle(*MI, OpNo);
  swizzleRepr(O, static_cast<unsigned>(Swizzle));
}

void PISAInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O, const char *Modifier) {
  assert((Modifier == 0 || Modifier[0] == 0) && "No modifiers supported");
  if (PISAMCInstLower::isVariableRef(*MI, OpNo)) {
    assert(MI->getOperand(OpNo).isImm() && "expecting imm operand");
    unsigned int FrameIndex = MI->getOperand(OpNo).getImm();
    O << "@R" << FrameIndex;
  } else if (OpNo < MI->getNumOperands()) {
    const MCOperand &Op = MI->getOperand(OpNo);
    if (Op.isReg()) {
      printRegName(O, Op.getReg());
      printSwizzle(MI, OpNo, O);
    } else if (Op.isImm())
      O << formatImm((int64_t)Op.getImm());
    else if (Op.isSFPImm())
      printFpImm32Opnd(MI, OpNo, O);
    else if (Op.isDFPImm())
      printFpImm64Opnd(MI, OpNo, O);
    else if (Op.isExpr())
      MAI.printExpr(O, *Op.getExpr());
    else
      llvm_unreachable("Unexpected operand type");
  }
}

void PISAInstPrinter::negateRepr(const MCInst *MI, unsigned OpNo,
                                 raw_ostream &O) {
  int64_t DoNegate = MI->getOperand(OpNo).getImm();
  if (DoNegate)
    O << "!";
}

void PISAInstPrinter::swizzleRepr(raw_ostream &O, unsigned SwizzleVal) {
  switch (static_cast<Swizzle>(SwizzleVal)) {
  case Swizzle::X:
    O << ".x";
    break;
  case Swizzle::Y:
    O << ".y";
    break;
  case Swizzle::Z:
    O << ".z";
    break;
  case Swizzle::W:
    O << ".w";
    break;
  case Swizzle::XYZW:
    O << ".xyzw";
    break;
  case Swizzle::XY:
    O << ".xy";
    break;
  case Swizzle::ZW:
    O << ".zw";
    break;
  case Swizzle::NONE:
    break;
  }
}

void PISAInstPrinter::printFunctionCallTargetOpnd(const MCInst *MI,
                                                  unsigned OpNo,
                                                  raw_ostream &O) {
  printOperand(MI, OpNo, O);

  // print function args if any
  O << " (";
  for (unsigned ArgIdx = OpNo + 1; ArgIdx < MI->getNumOperands(); ++ArgIdx) {
    printOperand(MI, ArgIdx, O);
    if (ArgIdx + 1 < MI->getNumOperands())
      O << ", ";
  }
  O << ");";
}

void PISAInstPrinter::printAddrOffsetImm(const MCInst *MI, unsigned OpNo,
                                         raw_ostream &O) {
  int64_t Val = MI->getOperand(OpNo).getImm();
  // print imm offset only when it's not zero
  if (Val > 0) {
    O << " + " << formatImm(Val);
  } else if (Val < 0) {
    if (Val == std::numeric_limits<int64_t>::min())
      O << " - " << llvm::format("%" PRIu64, Val);
    else
      O << " - " << formatImm(-Val);
  }
}

void PISAInstPrinter::printBfnOpcode(const MCInst *MI, unsigned OpNo,
                                     raw_ostream &O) {
  int64_t Val = MI->getOperand(OpNo).getImm();
  O << format_hex(Val & 0xFF, 4);
}

void PISAInstPrinter::printStringImm(const MCInst *MI, unsigned OpNo,
                                     raw_ostream &O) {
  const unsigned NumOps = MI->getNumOperands();
  unsigned StrStartIndex = OpNo;
  while (StrStartIndex < NumOps) {
    if (MI->getOperand(StrStartIndex).isReg())
      break;

    std::string Str = getPISAStringOperand(*MI, OpNo);
    if (StrStartIndex != OpNo)
      O << ' '; // Add a space if we're starting a new string/argument.
    O << '"';
    for (char C : Str) {
      if (C == '"')
        O.write('\\'); // Escape " characters (might break for complex UTF-8).
      O.write(C);
    }
    O << '"';

    unsigned NumOpsInString = (Str.size() / 4) + 1;
    StrStartIndex += NumOpsInString;
  }
}

void PISAInstPrinter::printMemOperand(const MCInst *MI, int OpNo,
                                      raw_ostream &OS,
                                      const char * /*Modifier*/) {
  // [ Base OP Offset ]
  OS << "[";
  printOperand(MI, OpNo, OS);
  const MCOperand &OffsetOp = MI->getOperand(OpNo + 1);
  if (OffsetOp.isImm()) {
    auto Val = OffsetOp.getImm();
    if (Val != 0) {
      if (Val > 0)
        OS << " + " << formatImm(Val);
      else if (Val == std::numeric_limits<int64_t>::min())
        OS << " - " << llvm::format("%" PRIu64, Val);
      else
        OS << " - " << formatImm(-Val);
    }
  } else {
    assert(OffsetOp.isReg() && "Register expected");
    OS << " + ";
    printRegName(OS, OffsetOp.getReg());
    printSwizzle(MI, (OpNo + 1), OS);
  }
  OS << "]";
}

void PISAInstPrinter::printMemSeqOperand(StringRef Pattern, const MCInst *MI,
                                         int OpNo, raw_ostream &OS,
                                         const char * /* Modifier */) {
  OS << "[";
  for (char C : Pattern) {
    switch (C) {
    case '{':
      OS << "{";
      break;
    case '}':
      OS << "}";
      break;
    case ',':
      OS << ", ";
      break;
    default:
      printOperand(MI, OpNo++, OS);
      break;
    }
  }
  OS << "]";
}

void PISAInstPrinter::printParamMemOperand(const MCInst *MCI, int OpNo,
                                           raw_ostream &OS) {
  const MCOperand &ArgIdxOp = MCI->getOperand(OpNo);
  const MCOperand &OffsetOp = MCI->getOperand(OpNo + 1);
  assert(ArgIdxOp.isImm());

  // Check for an extra expression operand carrying the kernel argument name.
  unsigned NameOpIdx = OpNo + 2;
  if (NameOpIdx < MCI->getNumOperands() &&
      MCI->getOperand(NameOpIdx).isExpr()) {
    const auto *SRE =
        cast<MCSymbolRefExpr>(MCI->getOperand(NameOpIdx).getExpr());
    OS << "[%" << SRE->getSymbol().getName();
  } else {
    OS << "[%arg" << ArgIdxOp.getImm();
  }

  if (OffsetOp.isImm()) {
    int64_t Offset = OffsetOp.getImm();
    if (Offset != 0)
      OS << llvm::format("%+" PRId64, Offset);
  } else {
    assert(OffsetOp.isReg() && "Register expected");
    OS << " + ";
    printRegName(OS, OffsetOp.getReg());
    printSwizzle(MCI, (OpNo + 1), OS);
  }
  OS << "]";
}

void PISAInstPrinter::printSymbolName(raw_ostream &OS, StringRef Name,
                                      const MCAsmInfo *MAI) {
  // This is a modification of MCSymbol::print() that prints non-printable
  // characters differently.
  if (!MAI || MAI->isValidUnquotedName(Name)) {
    OS << Name;
    return;
  }

  if (MAI && !MAI->supportsNameQuoting())
    report_fatal_error("Symbol name with unsupported characters");

  OS << '"';
  for (char C : Name) {
    if (C == '\n')
      OS << "\\n";
    else if (C == '"')
      OS << "\\\"";
    else if (isPrint(C))
      OS << C;
    else
      OS << '\\' << hexdigit(C >> 4) << hexdigit(C & 0x0F);
  }
  OS << '"';
}

void PISAInstPrinter::printEnumOption(EnumOptionClass OptClass,
                                      bool IsMandatoryField,
                                      unsigned DefaultVal, const MCInst *MCI,
                                      unsigned OpNo, raw_ostream &OS) {
  const MCOperand &MCOp = MCI->getOperand(OpNo);

  unsigned Val = MCOp.getImm();
  if (!IsMandatoryField && Val == DefaultVal)
    return;

  const EnumOptionEntry *Entry = lookupEnumOptionByValue(OptClass, Val);
  assert(Entry && "Enum value not found from lookup table");
  if (!Entry)
    return;

  StringRef OptStrRef = getEnumOptionEntryStr(Entry->OptStr);
  if (!OptStrRef.empty())
    OS << "." << OptStrRef;
}

PISAInstPrinter::printOpndFn
PISAInstPrinter::printBoolOptionOpnd(PISA::BoolOptionID OptID) {
  return [=](const MCInst *MCI, unsigned OpNo, raw_ostream &OS) -> void {
    const MCOperand &MCOp = MCI->getOperand(OpNo);
    if (MCOp.getImm()) {
      StringRef OptStr =
          getBoolOptionTableEntryStr(lookupBoolOptionByID(OptID)->OptStr);
      OS << "." << OptStr;
    }
  };
}
