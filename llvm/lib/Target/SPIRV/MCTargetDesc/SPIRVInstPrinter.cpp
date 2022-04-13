//===-- SPIRVInstPrinter.cpp - Output SPIR-V MCInsts as ASM -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class prints a SPIR-V MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "SPIRVInstPrinter.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

// Include the auto-generated portion of the assembly writer.
#include "SPIRVGenAsmWriter.inc"

void SPIRVInstPrinter::printRemainingVariableOps(const MCInst *MI,
                                                 unsigned StartIndex,
                                                 raw_ostream &O,
                                                 bool SkipFirstSpace,
                                                 bool SkipImmediates) {
  const unsigned NumOps = MI->getNumOperands();
  for (unsigned i = StartIndex; i < NumOps; ++i) {
    if (!SkipImmediates || !MI->getOperand(i).isImm()) {
      if (!SkipFirstSpace || i != StartIndex)
        O << ' ';
      printOperand(MI, i, O);
    }
  }
}

void SPIRVInstPrinter::printOpConstantVarOps(const MCInst *MI,
                                             unsigned StartIndex,
                                             raw_ostream &O) {
  O << ' ';
  if (MI->getNumOperands() - StartIndex == 2) { // Handle 64 bit literals.
    uint64_t Imm = MI->getOperand(StartIndex).getImm();
    Imm |= (MI->getOperand(StartIndex + 1).getImm() << 32);
    O << Imm;
  } else {
    printRemainingVariableOps(MI, StartIndex, O, true, false);
  }
}

void SPIRVInstPrinter::recordOpExtInstImport(const MCInst *MI) {
  llvm_unreachable("Unimplemented recordOpExtInstImport");
}

void SPIRVInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                 StringRef Annot, const MCSubtargetInfo &STI,
                                 raw_ostream &OS) {
  printInstruction(MI, Address, OS);
  printAnnotation(OS, Annot);
}

void SPIRVInstPrinter::printOpExtInst(const MCInst *MI, raw_ostream &O) {
  llvm_unreachable("Unimplemented printOpExtInst");
}

void SPIRVInstPrinter::printOpDecorate(const MCInst *MI, raw_ostream &O) {
  llvm_unreachable("Unimplemented printOpDecorate");
}

static void printExpr(const MCExpr *Expr, raw_ostream &O) {
#ifndef NDEBUG
  const MCSymbolRefExpr *SRE;

  if (const MCBinaryExpr *BE = dyn_cast<MCBinaryExpr>(Expr))
    SRE = cast<MCSymbolRefExpr>(BE->getLHS());
  else
    SRE = cast<MCSymbolRefExpr>(Expr);

  MCSymbolRefExpr::VariantKind Kind = SRE->getKind();

  assert(Kind == MCSymbolRefExpr::VK_None);
#endif
  O << *Expr;
}

void SPIRVInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                    raw_ostream &O, const char *Modifier) {
  assert((Modifier == 0 || Modifier[0] == 0) && "No modifiers supported");
  if (OpNo < MI->getNumOperands()) {
    const MCOperand &Op = MI->getOperand(OpNo);
    if (Op.isReg())
      O << '%' << (Register::virtReg2Index(Op.getReg()) + 1);
    else if (Op.isImm())
      O << formatImm((int64_t)Op.getImm());
    else if (Op.isDFPImm())
      O << formatImm((double)Op.getDFPImm());
    else if (Op.isExpr())
      printExpr(Op.getExpr(), O);
    else
      llvm_unreachable("Unexpected operand type");
  }
}

void SPIRVInstPrinter::printStringImm(const MCInst *MI, unsigned OpNo,
                                      raw_ostream &O) {
  llvm_unreachable("Unimplemented printStringImm");
}

void SPIRVInstPrinter::printExtInst(const MCInst *MI, unsigned OpNo,
                                    raw_ostream &O) {
  llvm_unreachable("Unimplemented printExtInst");
}

// Methods for printing textual names of SPIR-V enums.
#define GEN_INSTR_PRINTER_IMPL(EnumName)                                       \
  void SPIRVInstPrinter::print##EnumName(const MCInst *MI, unsigned OpNo,      \
                                         raw_ostream &O) {                     \
    llvm_unreachable("Unimplemented print" #EnumName);                         \
  }
GEN_INSTR_PRINTER_IMPL(Capability)
GEN_INSTR_PRINTER_IMPL(SourceLanguage)
GEN_INSTR_PRINTER_IMPL(ExecutionModel)
GEN_INSTR_PRINTER_IMPL(AddressingModel)
GEN_INSTR_PRINTER_IMPL(MemoryModel)
GEN_INSTR_PRINTER_IMPL(ExecutionMode)
GEN_INSTR_PRINTER_IMPL(StorageClass)
GEN_INSTR_PRINTER_IMPL(Dim)
GEN_INSTR_PRINTER_IMPL(SamplerAddressingMode)
GEN_INSTR_PRINTER_IMPL(SamplerFilterMode)
GEN_INSTR_PRINTER_IMPL(ImageFormat)
GEN_INSTR_PRINTER_IMPL(ImageChannelOrder)
GEN_INSTR_PRINTER_IMPL(ImageChannelDataType)
GEN_INSTR_PRINTER_IMPL(ImageOperand)
GEN_INSTR_PRINTER_IMPL(FPFastMathMode)
GEN_INSTR_PRINTER_IMPL(FPRoundingMode)
GEN_INSTR_PRINTER_IMPL(LinkageType)
GEN_INSTR_PRINTER_IMPL(AccessQualifier)
GEN_INSTR_PRINTER_IMPL(FunctionParameterAttribute)
GEN_INSTR_PRINTER_IMPL(Decoration)
GEN_INSTR_PRINTER_IMPL(BuiltIn)
GEN_INSTR_PRINTER_IMPL(SelectionControl)
GEN_INSTR_PRINTER_IMPL(LoopControl)
GEN_INSTR_PRINTER_IMPL(FunctionControl)
GEN_INSTR_PRINTER_IMPL(MemorySemantics)
GEN_INSTR_PRINTER_IMPL(MemoryOperand)
GEN_INSTR_PRINTER_IMPL(Scope)
GEN_INSTR_PRINTER_IMPL(GroupOperation)
GEN_INSTR_PRINTER_IMPL(KernelEnqueueFlags)
GEN_INSTR_PRINTER_IMPL(KernelProfilingInfo)
