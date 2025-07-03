//=== DWARFUnwindTablePrinter.cpp - Print the cfi-portions of .debug_frame ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFUnwindTablePrinter.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFExpressionPrinter.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <optional>

using namespace llvm;
using namespace dwarf;

static void printRegister(raw_ostream &OS, DIDumpOptions DumpOpts,
                          unsigned RegNum) {
  if (DumpOpts.GetNameForDWARFReg) {
    auto RegName = DumpOpts.GetNameForDWARFReg(RegNum, DumpOpts.IsEH);
    if (!RegName.empty()) {
      OS << RegName;
      return;
    }
  }
  OS << "reg" << RegNum;
}

void llvm::dwarf::printUnwindLocation(const UnwindLocation &UL, raw_ostream &OS,
                                      DIDumpOptions DumpOpts) {
  if (UL.getDereference())
    OS << '[';
  switch (UL.getLocation()) {
  case UnwindLocation::Unspecified:
    OS << "unspecified";
    break;
  case UnwindLocation::Undefined:
    OS << "undefined";
    break;
  case UnwindLocation::Same:
    OS << "same";
    break;
  case UnwindLocation::CFAPlusOffset:
    OS << "CFA";
    if (UL.getOffset() == 0)
      break;
    if (UL.getOffset() > 0)
      OS << "+";
    OS << UL.getOffset();
    break;
  case UnwindLocation::RegPlusOffset:
    printRegister(OS, DumpOpts, UL.getRegister());
    if (UL.getOffset() == 0 && !UL.hasAddressSpace())
      break;
    if (UL.getOffset() >= 0)
      OS << "+";
    OS << UL.getOffset();
    if (UL.hasAddressSpace())
      OS << " in addrspace" << UL.getAddressSpace();
    break;
  case UnwindLocation::DWARFExpr: {
    if (UL.getDWARFExpressionBytes()) {
      auto Expr = *UL.getDWARFExpressionBytes();
      printDwarfExpression(&Expr, OS, DumpOpts, nullptr);
    }
    break;
  }
  case UnwindLocation::Constant:
    OS << UL.getOffset();
    break;
  }
  if (UL.getDereference())
    OS << ']';
}

raw_ostream &llvm::dwarf::operator<<(raw_ostream &OS,
                                     const UnwindLocation &UL) {
  auto DumpOpts = DIDumpOptions();
  printUnwindLocation(UL, OS, DumpOpts);
  return OS;
}

void llvm::dwarf::printRegisterLocations(const RegisterLocations &RL,
                                         raw_ostream &OS,
                                         DIDumpOptions DumpOpts) {
  bool First = true;
  for (uint32_t Reg : RL.getRegisters()) {
    auto Loc = *RL.getRegisterLocation(Reg);
    if (First)
      First = false;
    else
      OS << ", ";
    printRegister(OS, DumpOpts, Reg);
    OS << '=';
    printUnwindLocation(Loc, OS, DumpOpts);
  }
}

raw_ostream &llvm::dwarf::operator<<(raw_ostream &OS,
                                     const RegisterLocations &RL) {
  auto DumpOpts = DIDumpOptions();
  printRegisterLocations(RL, OS, DumpOpts);
  return OS;
}

void llvm::dwarf::printUnwindRow(const UnwindRow &Row, raw_ostream &OS,
                                 DIDumpOptions DumpOpts, unsigned IndentLevel) {
  OS.indent(2 * IndentLevel);
  if (Row.hasAddress())
    OS << format("0x%" PRIx64 ": ", Row.getAddress());
  OS << "CFA=";
  printUnwindLocation(Row.getCFAValue(), OS, DumpOpts);
  if (Row.getRegisterLocations().hasLocations()) {
    OS << ": ";
    printRegisterLocations(Row.getRegisterLocations(), OS, DumpOpts);
  }
  OS << "\n";
}

raw_ostream &llvm::dwarf::operator<<(raw_ostream &OS, const UnwindRow &Row) {
  auto DumpOpts = DIDumpOptions();
  printUnwindRow(Row, OS, DumpOpts, 0);
  return OS;
}

void llvm::dwarf::printUnwindTable(const UnwindTable &Rows, raw_ostream &OS,
                                   DIDumpOptions DumpOpts,
                                   unsigned IndentLevel) {
  for (const UnwindRow &Row : Rows)
    printUnwindRow(Row, OS, DumpOpts, IndentLevel);
}

raw_ostream &llvm::dwarf::operator<<(raw_ostream &OS, const UnwindTable &Rows) {
  auto DumpOpts = DIDumpOptions();
  printUnwindTable(Rows, OS, DumpOpts, 0);
  return OS;
}
