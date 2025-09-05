//===- PseudoLoweringEmitter.cpp - PseudoLowering Generator -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Common/CodeGenInstruction.h"
#include "Common/CodeGenTarget.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TGTimer.h"
#include "llvm/TableGen/TableGenBackend.h"
using namespace llvm;

#define DEBUG_TYPE "pseudo-lowering"

namespace {
class PseudoLoweringEmitter {
  struct OpData {
    enum MapKind { Operand, Imm, Reg } Kind;
    union {
      unsigned OpNo;        // Operand number mapped to.
      uint64_t ImmVal;      // Integer immedate value.
      const Record *RegRec; // Physical register.
    };
  };
  struct PseudoExpansion {
    CodeGenInstruction Source; // The source pseudo instruction definition.
    CodeGenInstruction Dest;   // The destination instruction to lower to.
    IndexedMap<OpData> OperandMap;

    PseudoExpansion(CodeGenInstruction &s, CodeGenInstruction &d,
                    IndexedMap<OpData> &m)
        : Source(s), Dest(d), OperandMap(m) {}
  };

  const RecordKeeper &Records;

  // It's overkill to have an instance of the full CodeGenTarget object,
  // but it loads everything on demand, not in the constructor, so it's
  // lightweight in performance, so it works out OK.
  const CodeGenTarget Target;

  SmallVector<PseudoExpansion, 64> Expansions;

  void addOperandMapping(unsigned MIOpNo, unsigned NumOps, const Record *Rec,
                         const DagInit *Dag, unsigned DagIdx,
                         const Record *OpRec, IndexedMap<OpData> &OperandMap,
                         const StringMap<unsigned> &SourceOperands,
                         const CodeGenInstruction &SourceInsn);
  void evaluateExpansion(const Record *Pseudo);
  void emitLoweringEmitter(raw_ostream &o);

public:
  PseudoLoweringEmitter(const RecordKeeper &R) : Records(R), Target(R) {}

  /// run - Output the pseudo-lowerings.
  void run(raw_ostream &o);
};
} // End anonymous namespace

void PseudoLoweringEmitter::addOperandMapping(
    unsigned MIOpNo, unsigned NumOps, const Record *Rec, const DagInit *Dag,
    unsigned DagIdx, const Record *OpRec, IndexedMap<OpData> &OperandMap,
    const StringMap<unsigned> &SourceOperands,
    const CodeGenInstruction &SourceInsn) {
  const Init *DagArg = Dag->getArg(DagIdx);
  if (const DefInit *DI = dyn_cast<DefInit>(DagArg)) {
    // Physical register reference. Explicit check for the special case
    // "zero_reg" definition.
    if (DI->getDef()->isSubClassOf("Register") ||
        DI->getDef()->getName() == "zero_reg") {
      auto &Entry = OperandMap[MIOpNo];
      Entry.Kind = OpData::Reg;
      Entry.RegRec = DI->getDef();
      return;
    }

    if (DI->getDef() != OpRec)
      PrintFatalError(Rec, "In pseudo instruction '" + Rec->getName() +
                               "', operand type '" + DI->getDef()->getName() +
                               "' does not match expansion operand type '" +
                               OpRec->getName() + "'");

    StringMap<unsigned>::const_iterator SourceOp =
        SourceOperands.find(Dag->getArgNameStr(DagIdx));
    if (SourceOp == SourceOperands.end())
      PrintFatalError(Rec, "In pseudo instruction '" + Rec->getName() +
                               "', output operand '" +
                               Dag->getArgNameStr(DagIdx) +
                               "' has no matching source operand");
    const auto &SrcOpnd = SourceInsn.Operands[SourceOp->getValue()];
    if (NumOps != SrcOpnd.MINumOperands)
      PrintFatalError(
          Rec,
          "In pseudo instruction '" + Rec->getName() + "', output operand '" +
              OpRec->getName() +
              "' has a different number of sub operands than source operand '" +
              SrcOpnd.Rec->getName() + "'");

    // Source operand maps to destination operand. Do it for each corresponding
    // MachineInstr operand, not just the first.
    for (unsigned I = 0, E = NumOps; I != E; ++I) {
      auto &Entry = OperandMap[MIOpNo + I];
      Entry.Kind = OpData::Operand;
      Entry.OpNo = SrcOpnd.MIOperandNo + I;
    }

    LLVM_DEBUG(dbgs() << "    " << SourceOp->getValue() << " ==> " << DagIdx
                      << "\n");
  } else if (const auto *II = dyn_cast<IntInit>(DagArg)) {
    assert(NumOps == 1);
    auto &Entry = OperandMap[MIOpNo];
    Entry.Kind = OpData::Imm;
    Entry.ImmVal = II->getValue();
  } else if (const auto *BI = dyn_cast<BitsInit>(DagArg)) {
    assert(NumOps == 1);
    auto &Entry = OperandMap[MIOpNo];
    Entry.Kind = OpData::Imm;
    Entry.ImmVal = *BI->convertInitializerToInt();
  } else {
    llvm_unreachable("Unhandled pseudo-expansion argument type!");
  }
}

void PseudoLoweringEmitter::evaluateExpansion(const Record *Rec) {
  LLVM_DEBUG(dbgs() << "Pseudo definition: " << Rec->getName() << "\n");

  // Validate that the result pattern has the corrent number and types
  // of arguments for the instruction it references.
  const DagInit *Dag = Rec->getValueAsDag("ResultInst");
  assert(Dag && "Missing result instruction in pseudo expansion!");
  LLVM_DEBUG(dbgs() << "  Result: " << *Dag << "\n");

  const DefInit *OpDef = dyn_cast<DefInit>(Dag->getOperator());
  if (!OpDef)
    PrintFatalError(Rec, "In pseudo instruction '" + Rec->getName() +
                             "', result operator is not a record");
  const Record *Operator = OpDef->getDef();
  if (!Operator->isSubClassOf("Instruction"))
    PrintFatalError(Rec, "In pseudo instruction '" + Rec->getName() +
                             "', result operator '" + Operator->getName() +
                             "' is not an instruction");

  CodeGenInstruction Insn(Operator);

  if (Insn.isCodeGenOnly || Insn.isPseudo)
    PrintFatalError(Rec, "In pseudo instruction '" + Rec->getName() +
                             "', result operator '" + Operator->getName() +
                             "' cannot be a pseudo instruction");

  if (Insn.Operands.size() != Dag->getNumArgs())
    PrintFatalError(Rec, "In pseudo instruction '" + Rec->getName() +
                             "', result operator '" + Operator->getName() +
                             "' has the wrong number of operands");

  // If there are more operands that weren't in the DAG, they have to
  // be operands that have default values, or we have an error. Currently,
  // Operands that are a subclass of OperandWithDefaultOp have default values.

  // Validate that each result pattern argument has a matching (by name)
  // argument in the source instruction, in either the (outs) or (ins) list.
  // Also check that the type of the arguments match.
  //
  // Record the mapping of the source to result arguments for use by
  // the lowering emitter.
  CodeGenInstruction SourceInsn(Rec);
  StringMap<unsigned> SourceOperands;
  for (const auto &[Idx, SrcOp] : enumerate(SourceInsn.Operands))
    SourceOperands[SrcOp.Name] = Idx;

  unsigned NumMIOperands = 0;
  for (const auto &Op : Insn.Operands)
    NumMIOperands += Op.MINumOperands;
  IndexedMap<OpData> OperandMap;
  OperandMap.grow(NumMIOperands);

  // FIXME: This pass currently can only expand a pseudo to a single
  // instruction. The pseudo expansion really should take a list of dags, not
  // just a single dag, so we can do fancier things.
  LLVM_DEBUG(dbgs() << "  Operand mapping:\n");
  for (const auto &[Idx, DstOp] : enumerate(Insn.Operands)) {
    unsigned MIOpNo = DstOp.MIOperandNo;

    if (const auto *SubDag = dyn_cast<DagInit>(Dag->getArg(Idx))) {
      if (!DstOp.MIOperandInfo || DstOp.MIOperandInfo->getNumArgs() == 0)
        PrintFatalError(Rec, "In pseudo instruction '" + Rec->getName() +
                                 "', operand '" + DstOp.Rec->getName() +
                                 "' does not have suboperands");
      if (DstOp.MINumOperands != SubDag->getNumArgs()) {
        PrintFatalError(
            Rec, "In pseudo instruction '" + Rec->getName() + "', '" +
                     SubDag->getAsString() +
                     "' has wrong number of operands for operand type '" +
                     DstOp.Rec->getName() + "'");
      }
      for (unsigned I = 0, E = DstOp.MINumOperands; I != E; ++I) {
        auto *OpndRec = cast<DefInit>(DstOp.MIOperandInfo->getArg(I))->getDef();
        addOperandMapping(MIOpNo + I, 1, Rec, SubDag, I, OpndRec, OperandMap,
                          SourceOperands, SourceInsn);
      }
    } else {
      addOperandMapping(MIOpNo, DstOp.MINumOperands, Rec, Dag, Idx, DstOp.Rec,
                        OperandMap, SourceOperands, SourceInsn);
    }
  }

  Expansions.emplace_back(SourceInsn, Insn, OperandMap);
}

void PseudoLoweringEmitter::emitLoweringEmitter(raw_ostream &o) {
  // Emit file header.
  emitSourceFileHeader("Pseudo-instruction MC lowering Source Fragment", o);

  o << "bool " << Target.getName() + "AsmPrinter::\n"
    << "lowerPseudoInstExpansion(const MachineInstr *MI, MCInst &Inst) {\n";

  if (!Expansions.empty()) {
    o << "  Inst.clear();\n"
      << "  switch (MI->getOpcode()) {\n"
      << "  default: return false;\n";
    for (auto &Expansion : Expansions) {
      CodeGenInstruction &Source = Expansion.Source;
      CodeGenInstruction &Dest = Expansion.Dest;
      o << "  case " << Source.Namespace << "::" << Source.getName() << ": {\n"
        << "    MCOperand MCOp;\n"
        << "    Inst.setOpcode(" << Dest.Namespace << "::" << Dest.getName()
        << ");\n";

      // Copy the operands from the source instruction.
      // FIXME: Instruction operands with defaults values (predicates and cc_out
      //        in ARM, for example shouldn't need explicit values in the
      //        expansion DAG.
      for (const auto &DestOperand : Dest.Operands) {
        o << "    // Operand: " << DestOperand.Name << "\n";
        unsigned MIOpNo = DestOperand.MIOperandNo;
        for (unsigned i = 0, e = DestOperand.MINumOperands; i != e; ++i) {
          switch (Expansion.OperandMap[MIOpNo + i].Kind) {
          case OpData::Operand:
            o << "    lowerOperand(MI->getOperand("
              << Expansion.OperandMap[MIOpNo + i].OpNo << "), MCOp);\n"
              << "    Inst.addOperand(MCOp);\n";
            break;
          case OpData::Imm:
            o << "    Inst.addOperand(MCOperand::createImm("
              << Expansion.OperandMap[MIOpNo + i].ImmVal << "));\n";
            break;
          case OpData::Reg: {
            const Record *Reg = Expansion.OperandMap[MIOpNo + i].RegRec;
            o << "    Inst.addOperand(MCOperand::createReg(";
            // "zero_reg" is special.
            if (Reg->getName() == "zero_reg")
              o << "0";
            else
              o << Reg->getValueAsString("Namespace") << "::" << Reg->getName();
            o << "));\n";
            break;
          }
          }
        }
      }
      if (Dest.Operands.isVariadic) {
        unsigned LastOpNo = 0;
        for (const auto &Op : Source.Operands)
          LastOpNo += Op.MINumOperands;
        o << "    // variable_ops\n";
        o << "    for (unsigned i = " << LastOpNo
          << ", e = MI->getNumOperands(); i != e; ++i)\n"
          << "      if (lowerOperand(MI->getOperand(i), MCOp))\n"
          << "        Inst.addOperand(MCOp);\n";
      }
      o << "    break;\n"
        << "  }\n";
    }
    o << "  }\n  return true;";
  } else {
    o << "  return false;";
  }

  o << "\n}\n\n";
}

void PseudoLoweringEmitter::run(raw_ostream &OS) {
  StringRef Classes[] = {"PseudoInstExpansion", "Instruction"};

  // Process the pseudo expansion definitions, validating them as we do so.
  TGTimer &Timer = Records.getTimer();
  Timer.startTimer("Process definitions");
  for (const Record *Inst : Records.getAllDerivedDefinitions(Classes))
    evaluateExpansion(Inst);

  // Generate expansion code to lower the pseudo to an MCInst of the real
  // instruction.
  Timer.startTimer("Emit expansion code");
  emitLoweringEmitter(OS);
}

static TableGen::Emitter::OptClass<PseudoLoweringEmitter>
    X("gen-pseudo-lowering", "Generate pseudo instruction lowering");
