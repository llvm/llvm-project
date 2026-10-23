//===- BytecodeProgram.h - Statement bytecode emission ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emits a table of statement ids per opcode and a loop that runs them, in
// place of a switch with one case per opcode. The tables are plain integers,
// so they need no relocations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_COMMON_BYTECODEPROGRAM_H
#define LLVM_UTILS_TABLEGEN_COMMON_BYTECODEPROGRAM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <deque>
#include <string>
#include <vector>

namespace llvm {

class raw_ostream;

class BytecodeProgram {
public:
  /// \p UnreachableStmt runs for an entry with no sequence. \p TerminatorStmt
  /// ends every sequence and must leave the interpreter loop.
  BytecodeProgram(unsigned NumEntries, StringRef UnreachableStmt,
                  StringRef TerminatorStmt);

  /// Select \p Stmts, then the terminator, for \p Entry. Identical sequences
  /// share one copy.
  void addSequence(unsigned Entry, ArrayRef<std::string> Stmts);

  /// Emit the offset and program tables, then the loop that runs the sequence
  /// of entry `EntryExpr`. Entry I of the offset table is followed by
  /// `// PrintEntryName(I)`.
  void emit(raw_ostream &O, StringRef OffsetsName, StringRef ProgramName,
            StringRef EntryExpr,
            function_ref<void(raw_ostream &, unsigned)> PrintEntryName);

private:
  // Offset 0 is the sequence {0}, which backs unset entries.
  static constexpr unsigned UnreachableCmd = 0;
  static constexpr unsigned TerminatorCmd = 1;

  MapVector<std::string, unsigned, StringMap<unsigned>> Statements;
  // Deque, so the ArrayRef keys of SequenceOffsets stay valid.
  std::deque<std::vector<unsigned>> Sequences;
  DenseMap<ArrayRef<unsigned>, unsigned> SequenceOffsets;
  std::vector<unsigned> EntryOffsets;
  unsigned ProgramSize = 1;
};

} // end namespace llvm

#endif // LLVM_UTILS_TABLEGEN_COMMON_BYTECODEPROGRAM_H
