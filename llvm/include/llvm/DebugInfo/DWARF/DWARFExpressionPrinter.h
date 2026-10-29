//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFEXPRESSIONPRINTER_H
#define LLVM_DEBUGINFO_DWARF_DWARFEXPRESSIONPRINTER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/DebugInfo/DWARF/LowLevel/DWARFExpression.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

// DWARFUnit is intentionally not forward-declared here: the printers in this
// header must not depend on DWARFUnit, so that users of them do not link
// DWARFUnit and DWARFContext. Printers that resolve base type references
// through a DWARFUnit live in DWARFExpressionUnit.h.
struct DIDumpOptions;
class raw_ostream;

/// Print a Dwarf expression. Base type references are printed as raw offsets.
/// \param E to be printed
/// \param OS to this stream
/// \param DumpOpts with these options
LLVM_ABI void printDwarfExpression(const DWARFExpression *E, raw_ostream &OS,
                                   DIDumpOptions DumpOpts, bool IsEH = false);

/// Print the expression in a format intended to be compact and useful to a
/// user, but not perfectly unambiguous, or capable of representing every
/// valid DWARF expression. Returns true if the expression was sucessfully
/// printed.
///
/// \param E to be printed
/// \param OS to this stream
/// \param GetNameForDWARFReg side-effect-free callback to return a target
/// register name, or an empty string if none is available. The printer may try
/// another name-resolution method before reporting failure.
///
/// \returns true if the expression was successfully printed
LLVM_ABI bool printDwarfExpressionCompact(
    const DWARFExpression *E, raw_ostream &OS,
    std::function<StringRef(uint64_t RegNum, bool IsEH)> GetNameForDWARFReg =
        nullptr);

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFEXPRESSIONPRINTER_H
