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
  SmallVector<std::string, 8> OperandKinds;
};

static StringRef getFormatName(const Record &Def) {
  StringRef AsmString = Def.getValueAsString("AsmString");
  if (!AsmString.consume_front(".insn "))
    PrintFatalError(&Def, "expected .insn asm string");
  return AsmString.ltrim(" \t").take_until([](char C) { return C == ','; });
}

static StringRef getMatchClassKind(const Record &Def, const Init *Arg,
                                   unsigned OperandIndex) {
  static const StringMap<StringRef> KindMap = {
      {"AnyReg", "MCK_AnyReg"},
      {"VR128", "MCK_VR128"},
      {"brtarget12", "MCK_PCRel12"},
      {"brtarget12bpp", "MCK_PCRel12"},
      {"brtarget16", "MCK_PCRel16"},
      {"brtarget16bpp", "MCK_PCRel16"},
      {"brtarget24bpp", "MCK_PCRel24"},
      {"brtarget32", "MCK_PCRel32"},
      {"uimm32", "MCK_U32Imm"},
      {"imm32zx4", "MCK_U4Imm"},
      {"imm32zx8", "MCK_U8Imm"},
      {"imm32sx8", "MCK_X8Imm"},
      {"imm32xx8", "MCK_X8Imm"},
      {"imm32zx12", "MCK_U12Imm"},
      {"imm32zx16", "MCK_U16Imm"},
      {"imm32sx16", "MCK_S16Imm"},
      {"imm32xx16", "MCK_X16Imm"},
      {"imm64zx16", "MCK_U16Imm"},
      {"imm64zx32", "MCK_U32Imm"},
      {"imm64xx32", "MCK_X32Imm"},
      {"imm64zx48", "MCK_U48Imm"},
      {"bdxaddr12only", "MCK_BDXAddr64Disp12"},
      {"bdxaddr20only", "MCK_BDXAddr64Disp20"},
      {"bdaddr12only", "MCK_BDAddr64Disp12"},
      {"bdaddr20only", "MCK_BDAddr64Disp20"},
      {"bdvaddr12only", "MCK_BDVAddr64Disp12"},
      {"bdladdr12onlylen4", "MCK_BDLAddr64Disp12Len4"},
      {"bdladdr12onlylen8", "MCK_BDLAddr64Disp12Len8"},
      {"bdraddr12only", "MCK_BDXAddr64Disp12"}};

  std::string ArgTextStorage = Arg->getAsString();
  StringRef ArgText(ArgTextStorage);
  if (!ArgText.empty() && ArgText.front() == '(') {
    ArgText = ArgText.drop_front();
    ArgText = ArgText.take_while([](char C) { return C != ' '; });
  }

  // Check registered mappings
  auto It = KindMap.find(ArgText);
  if (It != KindMap.end())
    return It->second;

  PrintFatalError(&Def, "unsupported operand kind in .insn directive operand " +
                            Twine(OperandIndex) + ": " + ArgText);
}

static InsnMatchEntry buildInsnMatchEntry(const Record &Def) {
  const DagInit *InOperands = Def.getValueAsDag("InOperandList");
  if (InOperands->getNumArgs() == 0)
    PrintFatalError(&Def, ".insn directive missing encoding operand");

  InsnMatchEntry Entry;
  Entry.Format = getFormatName(Def).str();
  Entry.Opcode = ("SystemZ::" + Def.getName()).str();

  for (unsigned I = 0; I < InOperands->getNumArgs(); ++I)
    Entry.OperandKinds.push_back(
        getMatchClassKind(Def, InOperands->getArg(I), I).str());

  return Entry;
}

static void emitInsnDirectiveMatchTable(const RecordKeeper &RK,
                                        raw_ostream &OS) {
  // This will hold all .insn directive definitions (~100 plus margin).
  SmallVector<InsnMatchEntry, 128> Entries;
  // All .insn directive instructions inherit from InsnDirectiveBase.
  for (const Record *Def : RK.getAllDerivedDefinitions("InsnDirectiveBase")) {
    Entries.push_back(buildInsnMatchEntry(*Def));
  }

  llvm::sort(Entries, [](const InsnMatchEntry &LHS, const InsnMatchEntry &RHS) {
    return LHS.Format < RHS.Format;
  });

  OS << "/* Format, Opcode, NumOperands, OperandKinds */\n";
  for (const InsnMatchEntry &Entry : Entries) {
    OS << "  {\"" << Entry.Format << "\", " << Entry.Opcode << ", "
       << Entry.OperandKinds.size() << ", {";
    for (unsigned I = 0; I < Entry.OperandKinds.size(); ++I) {
      if (I != 0)
        OS << ", ";
      OS << Entry.OperandKinds[I];
    }
    OS << "}},\n";
  }
}
} // namespace

static TableGen::Emitter::Opt X("gen-insn-directive-match-table",
                                emitInsnDirectiveMatchTable,
                                "Generate SystemZ .insn match table.");
