//===- GIMatchDagInstr.cpp - A shared operand list for nodes --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GIMatchDagInstr.h"
#include "../CodeGenInstruction.h"
#include "GIMatchDag.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

void GIMatchDagInstr::print(raw_ostream &OS) const {
  OS << "(";
  if (const auto *Annotation = getOpcodeAnnotation())
    OS << Annotation->TheDef->getName();
  else
    OS << "<unknown>";
  OS << " ";
  OperandInfo.print(OS);
  OS << "):$" << Name;
  if (!UserAssignedNamesForOperands.empty()) {
    OS << " // ";
    SmallVector<std::pair<unsigned, StringRef>, 8> ToPrint;
    for (const auto &Assignment : UserAssignedNamesForOperands)
      ToPrint.emplace_back(Assignment.first, Assignment.second);
    llvm::sort(ToPrint);
    StringRef Separator = "";
    for (const auto &Assignment : ToPrint) {
      OS << Separator << "$" << Assignment.second << "=getOperand("
         << Assignment.first << ")";
      Separator = ", ";
    }
  }
}

void GIMatchDagInstr::setMatchRoot() {
  IsMatchRoot = true;
  Dag.addMatchRoot(this);
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const GIMatchDagInstr &N) {
  N.print(OS);
  return OS;
}
