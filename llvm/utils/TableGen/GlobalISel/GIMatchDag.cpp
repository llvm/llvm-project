//===- GIMatchDag.cpp - A DAG representation of a pattern to be matched ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GIMatchDag.h"

#include "llvm/Support/Format.h"
#include "llvm/TableGen/Record.h"
#include "../CodeGenInstruction.h"

using namespace llvm;

void GIMatchDag::writeDOTGraph(raw_ostream &OS, StringRef ID) const {
  const auto writePorts = [&](StringRef Prefix,
                              const GIMatchDagOperandList &Operands) {
    StringRef Separator = "";
    OS << "{";
    for (const auto &Op : enumerate(Operands)) {
      OS << Separator << "<" << Prefix << format("%d", Op.index()) << ">"
         << "#" << Op.index() << " $" << Op.value().getName();
      Separator = "|";
    }
    OS << "}";
  };

  OS << "digraph \"" << ID << "\" {\n"
     << "  rankdir=\"BT\"\n";
  for (const auto &N : InstrNodes) {
    OS << "  " << format("Node%p", &*N) << " [shape=record,label=\"{";
    writePorts("s", N->getOperandInfo());
    OS << "|" << N->getName();
    if (N->getOpcodeAnnotation())
      OS << "|" << N->getOpcodeAnnotation()->TheDef->getName();
    if (N->isMatchRoot())
      OS << "|Match starts here";
    OS << "|";
    SmallVector<std::pair<unsigned, StringRef>, 8> ToPrint;
    for (const auto &Assignment : N->user_assigned_operand_names())
      ToPrint.emplace_back(Assignment.first, Assignment.second);
    llvm::sort(ToPrint.begin(), ToPrint.end());
    StringRef Separator = "";
    for (const auto &Assignment : ToPrint) {
      OS << Separator << "$" << Assignment.second << "=getOperand("
         << Assignment.first << ")";
      Separator = ", ";
    }
    OS << format("|%p|", &N);
    writePorts("d", N->getOperandInfo());
    OS << "}\"";
    if (N->isMatchRoot())
      OS << ",color=red";
    OS << "]\n";
  }

  for (const auto &E : Edges) {
    const char *FromFmt = "Node%p:s%d:n";
    const char *ToFmt = "Node%p:d%d:s";
    if (E->getFromMO()->isDef() && !E->getToMO()->isDef())
      std::swap(FromFmt, ToFmt);
    auto From = format(FromFmt, E->getFromMI(), E->getFromMO()->getIdx());
    auto To = format(ToFmt, E->getToMI(), E->getToMO()->getIdx());
    if (E->getFromMO()->isDef() && !E->getToMO()->isDef())
      std::swap(From, To);

    OS << "  " << From << " -> " << To << " [label=\"$" << E->getName();
    if (E->getFromMO()->isDef() == E->getToMO()->isDef())
      OS << " INVALID EDGE!";
    OS << "\"";
    if (E->getFromMO()->isDef() == E->getToMO()->isDef())
      OS << ",color=red";
    else if (E->getFromMO()->isDef() && !E->getToMO()->isDef())
      OS << ",dir=back,arrowtail=crow";
    OS << "]\n";
  }

  for (const auto &N : PredicateNodes) {
    OS << "  " << format("Pred%p", &*N) << " [shape=record,label=\"{";
    writePorts("s", N->getOperandInfo());
    OS << "|" << N->getName() << "|";
    N->printDescription(OS);
    OS << format("|%p|", &N);
    writePorts("d", N->getOperandInfo());
    OS << "}\",style=dotted]\n";
  }

  for (const auto &E : PredicateDependencies) {
    const char *FromMIFmt = "Node%p:e";
    const char *FromMOFmt = "Node%p:s%d:n";
    const char *ToFmt = "Pred%p:d%d:s";
    auto To = format(ToFmt, E->getPredicate(), E->getPredicateOp()->getIdx());
    auto Style = "[style=dotted]";
    if (E->getRequiredMO()) {
      auto From =
          format(FromMOFmt, E->getRequiredMI(), E->getRequiredMO()->getIdx());
      OS << "  " << From << " -> " << To << " " << Style << "\n";
      continue;
    }
    auto From = format(FromMIFmt, E->getRequiredMI());
    OS << "  " << From << " -> " << To << " " << Style << "\n";
  }

  OS << "}\n";
}

LLVM_DUMP_METHOD void GIMatchDag::print(raw_ostream &OS) const {
  OS << "matchdag {\n";
  for (const auto &N : InstrNodes) {
    OS << "  ";
    N->print(OS);
    OS << "\n";
  }
  for (const auto &E : Edges) {
    OS << "  ";
    E->print(OS);
    OS << "\n";
  }

  for (const auto &P : PredicateNodes) {
    OS << "  ";
    P->print(OS);
    OS << "\n";
  }
  for (const auto &D : PredicateDependencies) {
    OS << "  ";
    D->print(OS);
    OS << "\n";
  }
  OS << "}\n";
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const GIMatchDag &G) {
  G.print(OS);
  return OS;
}
