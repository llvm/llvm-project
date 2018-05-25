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
#include "llvm/Support/FormattedStream.h"
#include "llvm/TableGen/Record.h"

namespace llvm {

class formatted_raw_ostream;

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
  void expandTrue(formatted_raw_ostream &OS);
  void expandFalse(formatted_raw_ostream &OS);
  void expandCheckImmOperand(formatted_raw_ostream &OS, int OpIndex,
                             int ImmVal);
  void expandCheckImmOperand(formatted_raw_ostream &OS, int OpIndex,
                             StringRef ImmVal);
  void expandCheckRegOperand(formatted_raw_ostream &OS, int OpIndex,
                             const Record *Reg);
  void expandCheckSameRegOperand(formatted_raw_ostream &OS, int First,
                                 int Second);
  void expandCheckNumOperands(formatted_raw_ostream &OS, int NumOps);
  void expandCheckOpcode(formatted_raw_ostream &OS, const Record *Inst);

  void expandCheckPseudo(formatted_raw_ostream &OS, const RecVec &Opcodes);
  void expandCheckOpcode(formatted_raw_ostream &OS, const RecVec &Opcodes);
  void expandPredicateSequence(formatted_raw_ostream &OS,
                               const RecVec &Sequence, bool IsCheckAll);
  void expandTIIFunctionCall(formatted_raw_ostream &OS, StringRef TargetName,
                             StringRef MethodName);
  void expandCheckIsRegOperand(formatted_raw_ostream &OS, int OpIndex);
  void expandCheckIsImmOperand(formatted_raw_ostream &OS, int OpIndex);
  void expandCheckFunctionPredicate(formatted_raw_ostream &OS,
                                    StringRef MCInstFn,
                                    StringRef MachineInstrFn);
  void expandCheckNonPortable(formatted_raw_ostream &OS, StringRef CodeBlock);
  void expandPredicate(formatted_raw_ostream &OS, const Record *Rec);
};

} // namespace llvm

#endif
