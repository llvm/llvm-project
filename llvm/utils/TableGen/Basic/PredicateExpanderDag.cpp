//===- PredicateExpanderDag.cpp - Composable predicate dag lowering -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PredicateExpanderDag.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

bool llvm::emitPredicateDag(
    const Record *Owner, const Init &Val, bool ParenIfBinOp, raw_ostream &OS,
    function_ref<bool(const Init &, raw_ostream &)> EmitLeaf) {
  if (const auto *D = dyn_cast<DagInit>(&Val)) {
    const auto *Op = dyn_cast<DefInit>(D->getOperator());
    if (!Op)
      PrintFatalError(Owner, "invalid predicate dag operator");
    StringRef OpName = Op->getDef()->getName();
    if (OpName == "not") {
      if (D->getNumArgs() != 1)
        PrintFatalError(Owner, "'not' takes exactly one operand");
      OS << "!(";
      bool Err = emitPredicateDag(Owner, *D->getArg(0), /*ParenIfBinOp=*/false,
                                  OS, EmitLeaf);
      OS << ')';
      return Err;
    }
    if (OpName == "any_of" || OpName == "all_of") {
      if (D->getNumArgs() == 0)
        PrintFatalError(Owner,
                        "'" + OpName + "' requires at least one operand");
      bool Paren = D->getNumArgs() > 1 && std::exchange(ParenIfBinOp, true);
      if (Paren)
        OS << '(';
      ListSeparator LS(OpName == "any_of" ? " || " : " && ");
      for (const Init *Arg : D->getArgs()) {
        OS << LS;
        if (emitPredicateDag(Owner, *Arg, ParenIfBinOp, OS, EmitLeaf))
          return true;
      }
      if (Paren)
        OS << ')';
      return false;
    }
    PrintFatalError(Owner, "unknown predicate dag operator '" + OpName +
                               "'; expected all_of/any_of/not");
  }

  // Any non-dag operand (or the base leaf) is emitted by the caller, which
  // diagnoses leaf-level errors specific to its consumer.
  return EmitLeaf(Val, OS);
}
