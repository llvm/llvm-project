//===--------------------- PredicateExpander.h ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Functionalities used by the Tablegen backends to expand machine predicates.
///
/// See file llvm/Target/TargetInstrPredicate.td for a full list and description
/// of all the supported MCInstPredicate classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_COMMON_PREDICATEEXPANDER_H
#define LLVM_UTILS_TABLEGEN_COMMON_PREDICATEEXPANDER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class Record;

class PredicateExpander {
  bool EmitCallsByRef;
  bool NegatePredicate;
  bool ExpandForMC;
  StringRef TargetName;

  PredicateExpander(const PredicateExpander &) = delete;
  PredicateExpander &operator=(const PredicateExpander &) = delete;

protected:
  indent Indent;

public:
  explicit PredicateExpander(StringRef Target, unsigned Indent = 1)
      : EmitCallsByRef(true), NegatePredicate(false), ExpandForMC(false),
        TargetName(Target), Indent(Indent, 2) {}
  bool isByRef() const { return EmitCallsByRef; }
  bool shouldNegate() const { return NegatePredicate; }
  bool shouldExpandForMC() const { return ExpandForMC; }
  indent &getIndent() { return Indent; }
  StringRef getTargetName() const { return TargetName; }

  void setByRef(bool Value) { EmitCallsByRef = Value; }
  void flipNegatePredicate() { NegatePredicate = !NegatePredicate; }
  void setNegatePredicate(bool Value) { NegatePredicate = Value; }
  void setExpandForMC(bool Value) { ExpandForMC = Value; }

  void expandTrue(raw_ostream &OS);
  void expandFalse(raw_ostream &OS);
  void expandCheckImmOperand(raw_ostream &OS, int OpIndex, int ImmVal,
                             StringRef FunctionMapper);
  void expandCheckImmOperand(raw_ostream &OS, int OpIndex, StringRef ImmVal,
                             StringRef FunctionMapperer);
  void expandCheckImmOperandSimple(raw_ostream &OS, int OpIndex,
                                   StringRef FunctionMapper);
  void expandCheckImmOperandLT(raw_ostream &OS, int OpIndex, int ImmVal,
                               StringRef FunctionMapper);
  void expandCheckImmOperandGT(raw_ostream &OS, int OpIndex, int ImmVal,
                               StringRef FunctionMapper);
  void expandCheckRegOperand(raw_ostream &OS, int OpIndex, const Record *Reg,
                             StringRef FunctionMapper);
  void expandCheckRegOperandSimple(raw_ostream &OS, int OpIndex,
                                   StringRef FunctionMapper);
  void expandCheckSameRegOperand(raw_ostream &OS, int First, int Second);
  void expandCheckNumOperands(raw_ostream &OS, int NumOps);
  void expandCheckOpcode(raw_ostream &OS, const Record *Inst);

  void expandCheckPseudo(raw_ostream &OS, ArrayRef<const Record *> Opcodes);
  void expandCheckOpcode(raw_ostream &OS, ArrayRef<const Record *> Opcodes);
  void expandPredicateSequence(raw_ostream &OS,
                               ArrayRef<const Record *> Sequence,
                               bool IsCheckAll);
  void expandTIIFunctionCall(raw_ostream &OS, StringRef MethodName);
  void expandCheckIsRegOperand(raw_ostream &OS, int OpIndex);
  void expandCheckIsVRegOperand(raw_ostream &OS, int OpIndex);
  void expandCheckIsImmOperand(raw_ostream &OS, int OpIndex);
  void expandCheckInvalidRegOperand(raw_ostream &OS, int OpIndex);
  void expandCheckFunctionPredicate(raw_ostream &OS, StringRef MCInstFn,
                                    StringRef MachineInstrFn);
  void expandCheckFunctionPredicateWithTII(raw_ostream &OS, StringRef MCInstFn,
                                           StringRef MachineInstrFn,
                                           StringRef TIIPtr);
  void expandCheckNonPortable(raw_ostream &OS, StringRef CodeBlock);
  void expandPredicate(raw_ostream &OS, const Record *Rec);
  void expandReturnStatement(raw_ostream &OS, const Record *Rec);
  void expandOpcodeSwitchCase(raw_ostream &OS, const Record *Rec);
  void expandOpcodeSwitchStatement(raw_ostream &OS,
                                   ArrayRef<const Record *> Cases,
                                   const Record *Default);
  void expandStatement(raw_ostream &OS, const Record *Rec);
};

// Forward declarations.
class STIPredicateFunction;
class OpcodeGroup;

class STIPredicateExpander : public PredicateExpander {
  StringRef ClassPrefix;
  bool ExpandDefinition;

  STIPredicateExpander(const PredicateExpander &) = delete;
  STIPredicateExpander &operator=(const PredicateExpander &) = delete;

  void expandHeader(raw_ostream &OS, const STIPredicateFunction &Fn);
  void expandPrologue(raw_ostream &OS, const STIPredicateFunction &Fn);
  void expandOpcodeGroup(raw_ostream &OS, const OpcodeGroup &Group,
                         bool ShouldUpdateOpcodeMask);
  void expandBody(raw_ostream &OS, const STIPredicateFunction &Fn);
  void expandEpilogue(raw_ostream &OS, const STIPredicateFunction &Fn);

public:
  explicit STIPredicateExpander(StringRef Target, unsigned Indent = 1)
      : PredicateExpander(Target, Indent), ExpandDefinition(false) {}

  bool shouldExpandDefinition() const { return ExpandDefinition; }
  StringRef getClassPrefix() const { return ClassPrefix; }
  void setClassPrefix(StringRef S) { ClassPrefix = S; }
  void setExpandDefinition(bool Value) { ExpandDefinition = Value; }

  void expandSTIPredicate(raw_ostream &OS, const STIPredicateFunction &Fn);
};

} // namespace llvm

#endif // LLVM_UTILS_TABLEGEN_COMMON_PREDICATEEXPANDER_H
