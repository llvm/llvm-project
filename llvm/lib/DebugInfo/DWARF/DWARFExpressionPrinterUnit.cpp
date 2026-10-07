//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFExpressionPrinterImpl.h"
#include "llvm/DebugInfo/DWARF/DWARFExpressionPrinter.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Support/FormatVariadic.h"
#include <cassert>

using namespace llvm;

static void prettyPrintBaseTypeRef(DWARFUnit *U, raw_ostream &OS,
                                   DIDumpOptions DumpOpts,
                                   ArrayRef<uint64_t> Operands,
                                   unsigned Operand) {
  assert(Operand < Operands.size() && "operand out of bounds");
  auto Die = U->getDIEForOffset(U->getOffset() + Operands[Operand]);
  if (Die && Die.getTag() == dwarf::DW_TAG_base_type) {
    OS << " (";
    if (DumpOpts.Verbose)
      OS << formatv("{0:x8} -> ", Operands[Operand]);
    OS << formatv("{0:x8})", U->getOffset() + Operands[Operand]);
    if (auto Name = dwarf::toString(Die.find(dwarf::DW_AT_name)))
      OS << " \"" << *Name << "\"";
  } else {
    OS << formatv(" <invalid base_type ref: {0:x}>", Operands[Operand]);
  }
}

void llvm::printDwarfExpression(const DWARFExpression *E, raw_ostream &OS,
                                DIDumpOptions DumpOpts, DWARFUnit *U,
                                bool IsEH) {
  auto PrintBaseTypeRef = [U](raw_ostream &OS, DIDumpOptions DumpOpts,
                              ArrayRef<uint64_t> Operands, unsigned Operand) {
    prettyPrintBaseTypeRef(U, OS, DumpOpts, Operands, Operand);
  };
  detail::printDwarfExpression(
      E, OS, DumpOpts,
      U ? detail::BaseTypeRefPrinter(PrintBaseTypeRef) : nullptr, IsEH);
}

bool llvm::prettyPrintRegisterOp(DWARFUnit *U, raw_ostream &OS,
                                 DIDumpOptions DumpOpts, uint8_t Opcode,
                                 ArrayRef<uint64_t> Operands) {
  auto PrintBaseTypeRef = [U](raw_ostream &OS, DIDumpOptions DumpOpts,
                              ArrayRef<uint64_t> Operands, unsigned Operand) {
    prettyPrintBaseTypeRef(U, OS, DumpOpts, Operands, Operand);
  };
  return detail::prettyPrintRegisterOp(
      U ? detail::BaseTypeRefPrinter(PrintBaseTypeRef) : nullptr, OS, DumpOpts,
      Opcode, Operands);
}
