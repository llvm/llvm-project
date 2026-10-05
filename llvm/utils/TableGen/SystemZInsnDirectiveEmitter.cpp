//===- SystemZInsnDirectiveEmitter.cpp - Generate .insn match table -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits a match table for SystemZ .insn directives.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <string>

using namespace llvm;

namespace {

struct InsnMatchEntry {
  std::string Format;
  std::string Opcode;
  unsigned NumOptionalOperands;
  SmallVector<std::string, 8> OperandKinds;
};

static StringRef getFormatName(const Record &Def) {
  StringRef AsmString = Def.getValueAsString("AsmString");
  if (!AsmString.consume_front(".insn "))
    PrintFatalError(&Def, "expected .insn asm string");
  return AsmString.take_until([](char C) { return C == ','; });
}

static std::string getMatchClassKind(const Record &Def, const Init *Arg,
                                     unsigned OperandIndex) {

  // Obtain record of operand.
  const Record *OpRec;
  if (const DefInit *DefOp = dyn_cast<DefInit>(Arg))
    OpRec = DefOp->getDef();
  else if (const DagInit *DagOp = dyn_cast<DagInit>(Arg))
    OpRec = DagOp->getOperatorAsDef(Def.getLoc());
  else
    PrintFatalError(&Def, "Unexpected Init Type (neither def nor dag) in .insn "
                          "directive operand");

  // Get name of ParserMatchClass associated with operand.
  StringRef MC =
      OpRec->getValueAsDef("ParserMatchClass")->getValueAsString("Name");

  // Prepend MCK_ and return.
  return "MCK_" + std::string(MC);
}

static InsnMatchEntry buildInsnMatchEntry(const Record &Def, unsigned &MaxNum) {
  const DagInit *OutOperands = Def.getValueAsDag("OutOperandList");
  const DagInit *InOperands = Def.getValueAsDag("InOperandList");
  if (OutOperands->getNumArgs() > 0)
    PrintFatalError(&Def, ".insn directive may not have output operands");
  unsigned InOps = InOperands->getNumArgs();
  if (InOps == 0)
    PrintFatalError(&Def, ".insn directive missing encoding operand");

  InsnMatchEntry Entry;
  Entry.Format = getFormatName(Def).str();
  Entry.Opcode = ("SystemZ::" + Def.getName()).str();

  for (unsigned I = 0; I < InOps; ++I)
    Entry.OperandKinds.push_back(
        getMatchClassKind(Def, InOperands->getArg(I), I));

  Entry.NumOptionalOperands =
      (unsigned)Def.getValueAsInt("NumOptionalOperands");
  if (Entry.NumOptionalOperands > InOps)
    PrintFatalError(&Def, "NumOptionalOperands exceeds operand count");

  if (MaxNum < InOps)
    MaxNum = InOps;

  return Entry;
}

static void emitInsnDirectiveMatchTable(const RecordKeeper &RK,
                                        raw_ostream &OS) {
  emitSourceFileHeader("Match Table for SystemZ .insn directive operand types",
                       OS);
  unsigned MaxOperandCount = 0;
  // This will hold all .insn directive definitions (~100 plus margin).
  SmallVector<InsnMatchEntry, 128> Entries;
  // Collect all InstSystemZ records that have IsInsnDirective set to 1.
  for (const Record *Def : RK.getAllDerivedDefinitions("InstSystemZ"))
    if (Def->getValueAsBit("IsInsnDirective"))
      Entries.push_back(buildInsnMatchEntry(*Def, MaxOperandCount));

  // Sort entries by format name.
  llvm::sort(Entries, [](const InsnMatchEntry &LHS, const InsnMatchEntry &RHS) {
    return LHS.Format < RHS.Format;
  });

  OS << "constexpr unsigned MAX_INSN_OPERANDNUM = " << MaxOperandCount << ";\n";
  OS << "\n";
  OS << "struct InsnMatchEntry {\n";
  OS << "  StringRef Format;\n";
  OS << "  uint64_t Opcode;\n";
  OS << "  uint32_t NumOperands;\n";
  OS << "  uint32_t NumOptionalOperands;\n";
  OS << "  MatchClassKind OperandKinds[MAX_INSN_OPERANDNUM];\n";
  OS << "};\n";
  OS << "\n";
  OS << "struct CompareInsn {\n";
  OS << "  bool operator()(const InsnMatchEntry &LHS, StringRef RHS) {\n";
  OS << "    return LHS.Format < RHS;\n";
  OS << "  }\n";
  OS << "  bool operator()(StringRef LHS, const InsnMatchEntry &RHS) {\n";
  OS << "    return LHS < RHS.Format;\n";
  OS << "  }\n";
  OS << "  bool operator()(const InsnMatchEntry &LHS,\n";
  OS << "                  const InsnMatchEntry &RHS) {\n";
  OS << "    return LHS.Format < RHS.Format;\n";
  OS << "  }\n";
  OS << "};\n";
  OS << "\n";
  OS << "/* Format, Opcode, NumOperands, NumOptionalOperands, OperandKinds "
        "*/\n";
  OS << "static InsnMatchEntry InsnMatchTable[] = {\n";
  for (const InsnMatchEntry &Entry : Entries) {
    OS << "  {\"" << Entry.Format << "\", " << Entry.Opcode << ", "
       << Entry.OperandKinds.size() << ", " << Entry.NumOptionalOperands
       << ", {";
    for (unsigned I = 0; I < Entry.OperandKinds.size(); ++I) {
      if (I != 0)
        OS << ", ";
      OS << Entry.OperandKinds[I];
    }
    OS << "}},\n";
  }
  OS << "};\n";
}
} // namespace

static TableGen::Emitter::Opt X("gen-insn-directive-match-table",
                                emitInsnDirectiveMatchTable,
                                "Generate SystemZ .insn match table.");
