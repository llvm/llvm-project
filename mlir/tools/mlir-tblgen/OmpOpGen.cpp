//===- OmpOpGen.cpp - OpenMP dialect op specific generators ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OmpOpGen defines OpenMP dialect operation specific generators.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"

#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

/// Obtain the name of the OpenMP clause a given record inheriting
/// `OpenMP_Clause` refers to.
///
/// It supports direct and indirect `OpenMP_Clause` superclasses. Once the
/// `OpenMP_Clause` class the record is based on is found, the optional
/// "OpenMP_" prefix and "Skip" and "Clause" suffixes are removed to return only
/// the clause name, i.e. "OpenMP_CollapseClauseSkip" is returned as "Collapse".
static StringRef extractOmpClauseName(const Record *clause) {
  const Record *ompClause = clause->getRecords().getClass("OpenMP_Clause");
  assert(ompClause && "base OpenMP records expected to be defined");

  StringRef clauseClassName;
  SmallVector<Record *, 1> clauseSuperClasses;
  clause->getDirectSuperClasses(clauseSuperClasses);

  // Check if OpenMP_Clause is a direct superclass.
  for (const Record *superClass : clauseSuperClasses) {
    if (superClass == ompClause) {
      clauseClassName = clause->getName();
      break;
    }
  }

  // Support indirectly-inherited OpenMP_Clauses.
  if (clauseClassName.empty()) {
    for (auto [superClass, _] : clause->getSuperClasses()) {
      if (superClass->isSubClassOf(ompClause)) {
        clauseClassName = superClass->getName();
        break;
      }
    }
  }

  assert(!clauseClassName.empty() && "clause name must be found");

  // Keep only the OpenMP clause name itself for reporting purposes.
  StringRef prefix = "OpenMP_";
  StringRef suffixes[] = {"Skip", "Clause"};

  if (clauseClassName.starts_with(prefix))
    clauseClassName = clauseClassName.substr(prefix.size());

  for (StringRef suffix : suffixes) {
    if (clauseClassName.ends_with(suffix))
      clauseClassName =
          clauseClassName.substr(0, clauseClassName.size() - suffix.size());
  }

  return clauseClassName;
}

/// Check that the given argument, identified by its name and initialization
/// value, is present in the \c arguments `dag`.
static bool verifyArgument(DagInit *arguments, StringRef argName,
                           Init *argInit) {
  auto range = zip_equal(arguments->getArgNames(), arguments->getArgs());
  return llvm::any_of(
      range, [&](std::tuple<llvm::StringInit *const &, llvm::Init *const &> v) {
        return std::get<0>(v)->getAsUnquotedString() == argName &&
               std::get<1>(v) == argInit;
      });
}

/// Check that the given string record value, identified by its name \c value,
/// is either undefined or empty in both the given operation and clause record
/// or its contents for the clause record are contained in the operation record.
static bool verifyStringValue(StringRef value, const Record *op,
                              const Record *clause) {
  auto opValue = op->getValueAsOptionalString(value);
  auto clauseValue = clause->getValueAsOptionalString(value);

  bool opHasValue = opValue && !opValue->trim().empty();
  bool clauseHasValue = clauseValue && !clauseValue->trim().empty();

  if (!opHasValue)
    return !clauseHasValue;

  return !clauseHasValue || opValue->contains(clauseValue->trim());
}

/// Verify that all fields of the given clause not explicitly ignored are
/// present in the corresponding operation field.
///
/// Print warnings or errors where this is not the case.
static void verifyClause(const Record *op, const Record *clause) {
  StringRef clauseClassName = extractOmpClauseName(clause);

  if (!clause->getValueAsBit("ignoreArgs")) {
    DagInit *opArguments = op->getValueAsDag("arguments");
    DagInit *arguments = clause->getValueAsDag("arguments");

    for (auto [name, arg] :
         zip(arguments->getArgNames(), arguments->getArgs())) {
      if (!verifyArgument(opArguments, name->getAsUnquotedString(), arg))
        PrintWarning(
            op->getLoc(),
            "'" + clauseClassName + "' clause-defined argument '" +
                arg->getAsUnquotedString() + ":$" +
                name->getAsUnquotedString() +
                "' not present in operation. Consider `dag arguments = "
                "!con(clausesArgs, ...)` or explicitly skipping this field.");
    }
  }

  if (!clause->getValueAsBit("ignoreAsmFormat") &&
      !verifyStringValue("assemblyFormat", op, clause))
    PrintWarning(
        op->getLoc(),
        "'" + clauseClassName +
            "' clause-defined `assemblyFormat` not present in operation. "
            "Consider concatenating `clausesAssemblyFormat` or explicitly "
            "skipping this field.");

  if (!clause->getValueAsBit("ignoreDesc") &&
      !verifyStringValue("description", op, clause))
    PrintError(op->getLoc(),
               "'" + clauseClassName +
                   "' clause-defined `description` not present in operation. "
                   "Consider concatenating `clausesDescription` or explicitly "
                   "skipping this field.");

  if (!clause->getValueAsBit("ignoreExtraDecl") &&
      !verifyStringValue("extraClassDeclaration", op, clause))
    PrintWarning(
        op->getLoc(),
        "'" + clauseClassName +
            "' clause-defined `extraClassDeclaration` not present in "
            "operation. Consider concatenating `clausesExtraClassDeclaration` "
            "or explicitly skipping this field.");
}

/// Verify that all properties of `OpenMP_Clause`s of records deriving from
/// `OpenMP_Op`s have been inherited by the latter.
static bool verifyDecls(const RecordKeeper &recordKeeper, raw_ostream &) {
  for (const Record *op : recordKeeper.getAllDerivedDefinitions("OpenMP_Op")) {
    for (const Record *clause : op->getValueAsListOfDefs("clauseList"))
      verifyClause(op, clause);
  }

  return false;
}

// Registers the generator to mlir-tblgen.
static mlir::GenRegistration
    verifyOpenmpOps("verify-openmp-ops",
                    "Verify OpenMP operations (produce no output file)",
                    verifyDecls);
