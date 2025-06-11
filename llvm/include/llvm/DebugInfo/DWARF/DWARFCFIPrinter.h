//===- DWARFCFIPrinter.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFCFIPRINTER_H
#define LLVM_DEBUGINFO_DWARF_DWARFCFIPRINTER_H

#include "llvm/DebugInfo/DWARF/DWARFCFIProgram.h"

namespace llvm {

struct DIDumpOptions;

namespace dwarf {

// This class is separate from CFIProgram to decouple CFIPrograms from the
// enclosing DWARF dies and type units, which allows using them in lower-level
// places without build dependencies.

class CFIPrinter {
public:
  static void print(const CFIProgram &P, raw_ostream &OS,
                    DIDumpOptions DumpOpts, unsigned IndentLevel,
                    std::optional<uint64_t> Address);

  static void printOperand(raw_ostream &OS, DIDumpOptions DumpOpts,
                           const CFIProgram &P,
                           const CFIProgram::Instruction &Instr,
                           unsigned OperandIdx, uint64_t Operand,
                           std::optional<uint64_t> &Address);
};

} // end namespace dwarf

} // end namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFCFIPRINTER_H
