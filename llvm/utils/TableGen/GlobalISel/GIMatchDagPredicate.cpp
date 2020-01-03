//===- GIMatchDagPredicate.cpp - Represent a predicate to check -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GIMatchDagPredicate.h"

#include "llvm/TableGen/Record.h"

#include "GIMatchDagOperands.h"
#include "../CodeGenInstruction.h"

using namespace llvm;

void GIMatchDagPredicate::print(raw_ostream &OS) const {
  OS << "<<";
  printDescription(OS);
  OS << ">>:$" << Name;
}

void GIMatchDagPredicate::printDescription(raw_ostream &OS) const { OS << ""; }

GIMatchDagOpcodePredicate::GIMatchDagOpcodePredicate(
    GIMatchDagContext &Ctx, StringRef Name, const CodeGenInstruction &Instr)
    : GIMatchDagPredicate(GIMatchDagPredicateKind_Opcode, Name,
                          Ctx.makeMIPredicateOperandList()),
      Instr(Instr) {}

void GIMatchDagOpcodePredicate::printDescription(raw_ostream &OS) const {
  OS << "$mi.getOpcode() == " << Instr.TheDef->getName();
}

GIMatchDagOneOfOpcodesPredicate::GIMatchDagOneOfOpcodesPredicate(
    GIMatchDagContext &Ctx, StringRef Name)
    : GIMatchDagPredicate(GIMatchDagPredicateKind_OneOfOpcodes, Name,
                          Ctx.makeMIPredicateOperandList()) {}

void GIMatchDagOneOfOpcodesPredicate::printDescription(raw_ostream &OS) const {
  OS << "$mi.getOpcode() == oneof(";
  StringRef Separator = "";
  for (const CodeGenInstruction *Instr : Instrs) {
    OS << Separator << Instr->TheDef->getName();
    Separator = ",";
  }
  OS << ")";
}

GIMatchDagSameMOPredicate::GIMatchDagSameMOPredicate(GIMatchDagContext &Ctx,
                                                     StringRef Name)
    : GIMatchDagPredicate(GIMatchDagPredicateKind_SameMO, Name,
                          Ctx.makeTwoMOPredicateOperandList()) {}

void GIMatchDagSameMOPredicate::printDescription(raw_ostream &OS) const {
  OS << "$mi0 == $mi1";
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const GIMatchDagPredicate &N) {
  N.print(OS);
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS,
                              const GIMatchDagOpcodePredicate &N) {
  N.print(OS);
  return OS;
}
