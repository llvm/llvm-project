//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_DWARF_DWARFEXPRESSIONPRINTERIMPL_H
#define LLVM_LIB_DEBUGINFO_DWARF_DWARFEXPRESSIONPRINTERIMPL_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/DebugInfo/DWARF/LowLevel/DWARFExpression.h"

namespace llvm {

struct DIDumpOptions;
class raw_ostream;

namespace detail {

/// Prints the base type DIE that operand \p Operand of \p Operands refers to.
/// A callback, so that printing without a DWARFUnit does not link DWARFUnit.
using BaseTypeRefPrinter =
    function_ref<void(raw_ostream &OS, DIDumpOptions DumpOpts,
                      ArrayRef<uint64_t> Operands, unsigned Operand)>;

void printDwarfExpression(const DWARFExpression *E, raw_ostream &OS,
                          DIDumpOptions DumpOpts,
                          BaseTypeRefPrinter PrintBaseTypeRef, bool IsEH);

bool prettyPrintRegisterOp(BaseTypeRefPrinter PrintBaseTypeRef, raw_ostream &OS,
                           DIDumpOptions DumpOpts, uint8_t Opcode,
                           ArrayRef<uint64_t> Operands);

} // namespace detail
} // namespace llvm

#endif // LLVM_LIB_DEBUGINFO_DWARF_DWARFEXPRESSIONPRINTERIMPL_H
