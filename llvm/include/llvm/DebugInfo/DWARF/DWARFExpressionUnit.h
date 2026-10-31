//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFEXPRESSIONUNIT_H
#define LLVM_DEBUGINFO_DWARF_DWARFEXPRESSIONUNIT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/DWARF/DWARFExpressionPrinter.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class DWARFUnit;
struct DIDumpOptions;
class raw_ostream;

/// Print a Dwarf expression.
/// \param E to be printed
/// \param OS to this stream
/// \param DumpOpts with these options
/// \param U within the context of this Dwarf unit, if any.
LLVM_ABI void printDwarfExpression(const DWARFExpression *E, raw_ostream &OS,
                                   DIDumpOptions DumpOpts, DWARFUnit *U,
                                   bool IsEH = false);

/// Pretty print a register opcode and operands.
/// \param U within the context of this Dwarf unit, if any.
/// \param OS to this stream
/// \param DumpOpts with these options
/// \param Opcode to print
/// \param Operands to the opcode
///
/// returns true if the Op was successfully printed
LLVM_ABI bool prettyPrintRegisterOp(DWARFUnit *U, raw_ostream &OS,
                                    DIDumpOptions DumpOpts, uint8_t Opcode,
                                    ArrayRef<uint64_t> Operands);

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFEXPRESSIONUNIT_H
