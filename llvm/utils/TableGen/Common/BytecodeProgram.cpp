//===- BytecodeProgram.cpp - Statement bytecode emission ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Common/BytecodeProgram.h"
#include "Common/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

BytecodeProgram::BytecodeProgram(unsigned NumEntries, StringRef UnreachableStmt,
                                 StringRef TerminatorStmt)
    : EntryOffsets(NumEntries, 0) {
  Statements.insert({UnreachableStmt.str(), UnreachableCmd});
  Statements.insert({TerminatorStmt.str(), TerminatorCmd});
}

void BytecodeProgram::addSequence(unsigned Entry, ArrayRef<std::string> Stmts) {
  std::vector<unsigned> Seq;
  for (const std::string &Stmt : Stmts) {
    // The interpreter supplies the indentation and the line breaks.
    std::string Trimmed = StringRef(Stmt).trim().str();
    Seq.push_back(
        Statements.try_emplace(Trimmed, Statements.size()).first->second);
  }
  Seq.push_back(TerminatorCmd);

  auto It = SequenceOffsets.find(Seq);
  if (It == SequenceOffsets.end()) {
    // The key must point into the deque, not at the local.
    Sequences.push_back(std::move(Seq));
    It = SequenceOffsets.try_emplace(Sequences.back(), ProgramSize).first;
    ProgramSize += Sequences.back().size();
  }
  EntryOffsets[Entry] = It->second;
}

void BytecodeProgram::emit(
    raw_ostream &O, StringRef OffsetsName, StringRef ProgramName,
    StringRef EntryExpr,
    function_ref<void(raw_ostream &, unsigned)> PrintEntryName) {
  const char *OffsetType = getMinimalTypeForRange(ProgramSize - 1);

  O << "  static const " << OffsetType << ' ' << OffsetsName << "[] = {\n";
  for (auto [Entry, Offset] : enumerate(EntryOffsets)) {
    O << "    " << Offset << ",\t// ";
    PrintEntryName(O, Entry);
    O << '\n';
  }
  O << "  };\n";

  O << "  static const " << getMinimalTypeForRange(Statements.size() - 1) << ' '
    << ProgramName << "[] = {\n"
    << "    /* 0 */ " << UnreachableCmd << ",\n";
  unsigned Offset = 1;
  for (const std::vector<unsigned> &Seq : Sequences) {
    O << "    /* " << Offset << " */";
    for (unsigned Cmd : Seq)
      O << ' ' << Cmd << ',';
    O << '\n';
    Offset += Seq.size();
  }
  O << "  };\n";

  O << "  for (" << OffsetType << " Idx = " << OffsetsName << '[' << EntryExpr
    << "];; ++Idx) {\n"
    << "    switch (" << ProgramName << "[Idx]) {\n"
    << "    default: llvm_unreachable(\"Unexpected bytecode command.\");\n";
  for (const auto &[Stmt, Cmd] : Statements) {
    O << "    case " << Cmd << ":\n      " << Stmt << "\n";
    if (Cmd != TerminatorCmd)
      O << "      break;\n";
  }
  O << "    }\n  }\n";
}
