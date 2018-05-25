//===--------------------- PredicateExpander.cpp --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// Functionalities used by the Tablegen backends to expand machine predicates.
//
//===----------------------------------------------------------------------===//

#include "PredicateExpander.h"

namespace llvm {

void PredicateExpander::expandTrue(formatted_raw_ostream &OS) { OS << "true"; }
void PredicateExpander::expandFalse(formatted_raw_ostream &OS) {
  OS << "false";
}

void PredicateExpander::expandCheckImmOperand(formatted_raw_ostream &OS,
                                              int OpIndex, int ImmVal) {
  OS << "MI.getOperand(" << OpIndex << ").getImm() "
     << (shouldNegate() ? "!= " : "== ") << ImmVal;
}

void PredicateExpander::expandCheckImmOperand(formatted_raw_ostream &OS,
                                              int OpIndex, StringRef ImmVal) {
  OS << "MI.getOperand(" << OpIndex << ").getImm() "
     << (shouldNegate() ? "!= " : "== ") << ImmVal;
}

void PredicateExpander::expandCheckRegOperand(formatted_raw_ostream &OS,
                                              int OpIndex, const Record *Reg) {
  assert(Reg->isSubClassOf("Register") && "Expected a register Record!");

  OS << "MI" << (isByRef() ? "." : "->") << "getOperand(" << OpIndex
     << ").getReg() " << (shouldNegate() ? "!= " : "== ");
  const StringRef Str = Reg->getValueAsString("Namespace");
  if (!Str.empty())
    OS << Str << "::";
  OS << Reg->getName();
}

void PredicateExpander::expandCheckSameRegOperand(formatted_raw_ostream &OS,
                                                  int First, int Second) {
  OS << "MI" << (isByRef() ? "." : "->") << "getOperand(" << First
     << ").getReg() " << (shouldNegate() ? "!=" : "==") << " MI"
     << (isByRef() ? "." : "->") << "getOperand(" << Second << ").getReg()";
}

void PredicateExpander::expandCheckNumOperands(formatted_raw_ostream &OS,
                                               int NumOps) {
  OS << "MI" << (isByRef() ? "." : "->") << "getNumOperands() "
     << (shouldNegate() ? "!= " : "== ") << NumOps;
}

void PredicateExpander::expandCheckOpcode(formatted_raw_ostream &OS,
                                          const Record *Inst) {
  OS << "MI" << (isByRef() ? "." : "->") << "getOpcode() "
     << (shouldNegate() ? "!= " : "== ") << Inst->getValueAsString("Namespace")
     << "::" << Inst->getName();
}

void PredicateExpander::expandCheckOpcode(formatted_raw_ostream &OS,
                                          const RecVec &Opcodes) {
  assert(!Opcodes.empty() && "Expected at least one opcode to check!");
  bool First = true;

  if (Opcodes.size() == 1) {
    OS << "( ";
    expandCheckOpcode(OS, Opcodes[0]);
    OS << " )";
    return;
  }

  OS << '(';
  increaseIndentLevel();
  for (const Record *Rec : Opcodes) {
    OS << '\n';
    OS.PadToColumn(getIndentLevel() * 2);
    if (!First)
      OS << (shouldNegate() ? "&& " : "|| ");

    expandCheckOpcode(OS, Rec);
    First = false;
  }

  OS << '\n';
  decreaseIndentLevel();
  OS.PadToColumn(getIndentLevel() * 2);
  OS << ')';
}

void PredicateExpander::expandCheckPseudo(formatted_raw_ostream &OS,
                                          const RecVec &Opcodes) {
  if (shouldExpandForMC())
    expandFalse(OS);
  else
    expandCheckOpcode(OS, Opcodes);
}

void PredicateExpander::expandPredicateSequence(formatted_raw_ostream &OS,
                                                const RecVec &Sequence,
                                                bool IsCheckAll) {
  assert(!Sequence.empty() && "Found an invalid empty predicate set!");
  if (Sequence.size() == 1)
    return expandPredicate(OS, Sequence[0]);

  // Okay, there is more than one predicate in the set.
  bool First = true;
  OS << (shouldNegate() ? "!(" : "(");
  increaseIndentLevel();

  bool OldValue = shouldNegate();
  setNegatePredicate(false);
  for (const Record *Rec : Sequence) {
    OS << '\n';
    OS.PadToColumn(getIndentLevel() * 2);
    if (!First)
      OS << (IsCheckAll ? "&& " : "|| ");
    expandPredicate(OS, Rec);
    First = false;
  }
  OS << '\n';
  decreaseIndentLevel();
  OS.PadToColumn(getIndentLevel() * 2);
  OS << ')';
  setNegatePredicate(OldValue);
}

void PredicateExpander::expandTIIFunctionCall(formatted_raw_ostream &OS,
                                              StringRef TargetName,
                                              StringRef MethodName) {
  OS << (shouldNegate() ? "!" : "");
  if (shouldExpandForMC())
    OS << TargetName << "_MC::";
  else
    OS << TargetName << "Gen"
       << "InstrInfo::";
  OS << MethodName << (isByRef() ? "(MI)" : "(*MI)");
}

void PredicateExpander::expandCheckIsRegOperand(formatted_raw_ostream &OS,
                                                int OpIndex) {
  OS << (shouldNegate() ? "!" : "") << "MI" << (isByRef() ? "." : "->")
     << "getOperand(" << OpIndex << ").isReg() ";
}

void PredicateExpander::expandCheckIsImmOperand(formatted_raw_ostream &OS,
                                                int OpIndex) {
  OS << (shouldNegate() ? "!" : "") << "MI" << (isByRef() ? "." : "->")
     << "getOperand(" << OpIndex << ").isImm() ";
}

void PredicateExpander::expandCheckFunctionPredicate(formatted_raw_ostream &OS,
                                                     StringRef MCInstFn,
                                                     StringRef MachineInstrFn) {
  OS << (shouldExpandForMC() ? MCInstFn : MachineInstrFn)
     << (isByRef() ? "(MI)" : "(*MI)");
}

void PredicateExpander::expandCheckNonPortable(formatted_raw_ostream &OS,
                                               StringRef Code) {
  if (shouldExpandForMC())
    return expandFalse(OS);

  OS << '(' << Code << ')';
}

void PredicateExpander::expandPredicate(formatted_raw_ostream &OS,
                                        const Record *Rec) {
  OS.flush();
  unsigned ColNum = getIndentLevel() * 2;
  if (OS.getColumn() < ColNum)
    OS.PadToColumn(ColNum);

  if (Rec->isSubClassOf("MCTrue")) {
    if (shouldNegate())
      return expandFalse(OS);
    return expandTrue(OS);
  }

  if (Rec->isSubClassOf("MCFalse")) {
    if (shouldNegate())
      return expandTrue(OS);
    return expandFalse(OS);
  }

  if (Rec->isSubClassOf("CheckNot")) {
    flipNegatePredicate();
    expandPredicate(OS, Rec->getValueAsDef("Pred"));
    flipNegatePredicate();
    return;
  }

  if (Rec->isSubClassOf("CheckIsRegOperand"))
    return expandCheckIsRegOperand(OS, Rec->getValueAsInt("OpIndex"));

  if (Rec->isSubClassOf("CheckIsImmOperand"))
    return expandCheckIsImmOperand(OS, Rec->getValueAsInt("OpIndex"));

  if (Rec->isSubClassOf("CheckRegOperand"))
    return expandCheckRegOperand(OS, Rec->getValueAsInt("OpIndex"),
                                 Rec->getValueAsDef("Reg"));

  if (Rec->isSubClassOf("CheckImmOperand"))
    return expandCheckImmOperand(OS, Rec->getValueAsInt("OpIndex"),
                                 Rec->getValueAsInt("ImmVal"));

  if (Rec->isSubClassOf("CheckImmOperand_s"))
    return expandCheckImmOperand(OS, Rec->getValueAsInt("OpIndex"),
                                 Rec->getValueAsString("ImmVal"));

  if (Rec->isSubClassOf("CheckSameRegOperand"))
    return expandCheckSameRegOperand(OS, Rec->getValueAsInt("FirstIndex"),
                                     Rec->getValueAsInt("SecondIndex"));

  if (Rec->isSubClassOf("CheckNumOperands"))
    return expandCheckNumOperands(OS, Rec->getValueAsInt("NumOps"));

  if (Rec->isSubClassOf("CheckPseudo"))
    return expandCheckPseudo(OS, Rec->getValueAsListOfDefs("ValidOpcodes"));

  if (Rec->isSubClassOf("CheckOpcode"))
    return expandCheckOpcode(OS, Rec->getValueAsListOfDefs("ValidOpcodes"));

  if (Rec->isSubClassOf("CheckAll"))
    return expandPredicateSequence(OS, Rec->getValueAsListOfDefs("Predicates"),
                                   /* AllOf */ true);

  if (Rec->isSubClassOf("CheckAny"))
    return expandPredicateSequence(OS, Rec->getValueAsListOfDefs("Predicates"),
                                   /* AllOf */ false);

  if (Rec->isSubClassOf("CheckFunctionPredicate"))
    return expandCheckFunctionPredicate(
        OS, Rec->getValueAsString("MCInstFnName"),
        Rec->getValueAsString("MachineInstrFnName"));

  if (Rec->isSubClassOf("CheckNonPortable"))
    return expandCheckNonPortable(OS, Rec->getValueAsString("CodeBlock"));

  if (Rec->isSubClassOf("TIIPredicate"))
    return expandTIIFunctionCall(OS, Rec->getValueAsString("TargetName"),
                                 Rec->getValueAsString("FunctionName"));

  llvm_unreachable("No known rules to expand this MCInstPredicate");
}

} // namespace llvm
