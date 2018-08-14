//===--------------------- PredicateExpander.h ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// Functionalities used by the Tablegen backends to expand machine predicates.
///
/// See file llvm/Target/TargetInstrPredicate.td for a full list and description
/// of all the supported MCInstPredicate classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_PREDICATEEXPANDER_H
#define LLVM_UTILS_TABLEGEN_PREDICATEEXPANDER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

namespace llvm {

class raw_ostream;

class PredicateExpander {
  bool EmitCallsByRef;
  bool NegatePredicate;
  bool ExpandForMC;
  unsigned IndentLevel;

  PredicateExpander(const PredicateExpander &) = delete;
  PredicateExpander &operator=(const PredicateExpander &) = delete;

public:
  PredicateExpander()
      : EmitCallsByRef(true), NegatePredicate(false), ExpandForMC(false),
        IndentLevel(1U) {}
  bool isByRef() const { return EmitCallsByRef; }
  bool shouldNegate() const { return NegatePredicate; }
  bool shouldExpandForMC() const { return ExpandForMC; }
  unsigned getIndentLevel() const { return IndentLevel; }

  void setByRef(bool Value) { EmitCallsByRef = Value; }
  void flipNegatePredicate() { NegatePredicate = !NegatePredicate; }
  void setNegatePredicate(bool Value) { NegatePredicate = Value; }
  void setExpandForMC(bool Value) { ExpandForMC = Value; }
  void increaseIndentLevel() { ++IndentLevel; }
  void decreaseIndentLevel() { --IndentLevel; }
  void setIndentLevel(unsigned Level) { IndentLevel = Level; }

  using RecVec = std::vector<Record *>;
  void expandTrue(raw_ostream &OS);
  void expandFalse(raw_ostream &OS);
  void expandCheckImmOperand(raw_ostream &OS, int OpIndex, int ImmVal);
  void expandCheckImmOperand(raw_ostream &OS, int OpIndex, StringRef ImmVal);
  void expandCheckRegOperand(raw_ostream &OS, int OpIndex, const Record *Reg);
  void expandCheckSameRegOperand(raw_ostream &OS, int First, int Second);
  void expandCheckNumOperands(raw_ostream &OS, int NumOps);
  void expandCheckOpcode(raw_ostream &OS, const Record *Inst);

  void expandCheckPseudo(raw_ostream &OS, const RecVec &Opcodes);
  void expandCheckOpcode(raw_ostream &OS, const RecVec &Opcodes);
  void expandPredicateSequence(raw_ostream &OS, const RecVec &Sequence,
                               bool IsCheckAll);
  void expandTIIFunctionCall(raw_ostream &OS, StringRef TargetName,
                             StringRef MethodName);
  void expandCheckIsRegOperand(raw_ostream &OS, int OpIndex);
  void expandCheckIsImmOperand(raw_ostream &OS, int OpIndex);
  void expandCheckInvalidRegOperand(raw_ostream &OS, int OpIndex);
  void expandCheckFunctionPredicate(raw_ostream &OS, StringRef MCInstFn,
                                    StringRef MachineInstrFn);
  void expandCheckNonPortable(raw_ostream &OS, StringRef CodeBlock);
  void expandPredicate(raw_ostream &OS, const Record *Rec);
  void expandReturnStatement(raw_ostream &OS, const Record *Rec);
  void expandOpcodeSwitchCase(raw_ostream &OS, const Record *Rec);
  void expandOpcodeSwitchStatement(raw_ostream &OS, const RecVec &Cases,
                                   const Record *Default);
  void expandStatement(raw_ostream &OS, const Record *Rec);
};

} // namespace llvm

#endif
