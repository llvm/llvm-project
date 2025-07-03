//===- DWARFUnwindTablePrinter.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFUNWINDTABLEPRINTER_H
#define LLVM_DEBUGINFO_DWARF_DWARFUNWINDTABLEPRINTER_H

#include "llvm/DebugInfo/DWARF/LowLevel/DWARFUnwindTable.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

struct DIDumpOptions;

namespace dwarf {

/// Print an unwind location expression as text and use the register information
/// if some is provided.
///
/// \param R the unwind location to print.
///
/// \param OS the stream to use for output.
///
/// \param MRI register information that helps emit register names insteead
/// of raw register numbers.
///
/// \param IsEH true if the DWARF Call Frame Information is from .eh_frame
/// instead of from .debug_frame. This is needed for register number
/// conversion because some register numbers differ between the two sections
/// for certain architectures like x86.
LLVM_ABI void printUnwindLocation(const UnwindLocation &R, raw_ostream &OS,
                                  DIDumpOptions DumpOpts);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const UnwindLocation &R);

/// Print all registers + locations that are currently defined in a register
/// locations.
///
/// \param RL the register locations to print.
///
/// \param OS the stream to use for output.
///
/// \param MRI register information that helps emit register names insteead
/// of raw register numbers.
///
/// \param IsEH true if the DWARF Call Frame Information is from .eh_frame
/// instead of from .debug_frame. This is needed for register number
/// conversion because some register numbers differ between the two sections
/// for certain architectures like x86.
LLVM_ABI void printRegisterLocations(const RegisterLocations &RL,
                                     raw_ostream &OS, DIDumpOptions DumpOpts);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const RegisterLocations &RL);

/// Print an UnwindRow to the stream.
///
/// \param Row the UnwindRow to print.
///
/// \param OS the stream to use for output.
///
/// \param MRI register information that helps emit register names insteead
/// of raw register numbers.
///
/// \param IsEH true if the DWARF Call Frame Information is from .eh_frame
/// instead of from .debug_frame. This is needed for register number
/// conversion because some register numbers differ between the two sections
/// for certain architectures like x86.
///
/// \param IndentLevel specify the indent level as an integer. The UnwindRow
/// will be output to the stream preceded by 2 * IndentLevel number of spaces.
LLVM_ABI void printUnwindRow(const UnwindRow &Row, raw_ostream &OS,
                             DIDumpOptions DumpOpts, unsigned IndentLevel = 0);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const UnwindRow &Row);

/// Print a UnwindTable to the stream.
///
/// \param Rows the UnwindTable to print.
///
/// \param OS the stream to use for output.
///
/// \param MRI register information that helps emit register names instead
/// of raw register numbers.
///
/// \param IsEH true if the DWARF Call Frame Information is from .eh_frame
/// instead of from .debug_frame. This is needed for register number
/// conversion because some register numbers differ between the two sections
/// for certain architectures like x86.
///
/// \param IndentLevel specify the indent level as an integer. The UnwindRow
/// will be output to the stream preceded by 2 * IndentLevel number of spaces.
LLVM_ABI void printUnwindTable(const UnwindTable &Rows, raw_ostream &OS,
                               DIDumpOptions DumpOpts,
                               unsigned IndentLevel = 0);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const UnwindTable &Rows);

} // end namespace dwarf

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFUNWINDTABLEPRINTER_H
