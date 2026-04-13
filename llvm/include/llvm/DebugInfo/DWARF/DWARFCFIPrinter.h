//===- DWARFCFIPrinter.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFCFIPRINTER_H
#define LLVM_DEBUGINFO_DWARF_DWARFCFIPRINTER_H

#include "llvm/DebugInfo/DWARF/LowLevel/DWARFCFIProgram.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

struct DIDumpOptions;

namespace dwarf {

LLVM_ABI void printCFIProgram(const CFIProgram &P, raw_ostream &OS,
                              const DIDumpOptions &DumpOpts,
                              unsigned IndentLevel,
                              std::optional<uint64_t> Address);

} // end namespace dwarf

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFCFIPRINTER_H
