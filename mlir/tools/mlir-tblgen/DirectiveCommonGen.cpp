//===========- DirectiveCommonGen.cpp - Directive common info generator -=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpenMPCommonGen generates utility information from the single OpenMP source
// of truth in llvm/lib/Frontend/OpenMP.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/DirectiveEmitter.h"
#include "llvm/TableGen/Record.h"

using llvm::Clause;
using llvm::ClauseVal;
using llvm::raw_ostream;
using llvm::RecordKeeper;
using llvm::Twine;

// LLVM has multiple places (Clang, Flang, MLIR) where information about
// the directives (OpenMP/OpenACC), and clauses are needed. It is good software
// engineering to keep the common information in a single place to avoid
// duplication, reduce engineering effort and prevent mistakes.
// Currently that common place is llvm/include/llvm/Frontend/OpenMP/OMP.td for
// OpenMP and llvm/include/llvm/Frontend/OpenACC/ACC.td for OpenACC.
// We plan to use this tablegen source to generate all the required
// declarations, functions etc.
//
// Some OpenMP/OpenACC clauses accept only a fixed set of values as inputs.
// These can be represented as a String Enum Attribute (StrEnumAttr) in MLIR
// ODS. The emitDecls function below currently generates these enumerations. The
// name of the enumeration is specified in the enumClauseValue field of
// Clause record in OMP.td. This name can be used to specify the type of the
// OpenMP operation's operand. The allowedClauseValues field provides the list
// of ClauseValues which are part of the enumeration.
static bool emitDecls(const RecordKeeper &recordKeeper, raw_ostream &os) {
  const auto &directiveLanguages =
      recordKeeper.getAllDerivedDefinitions("DirectiveLanguage");
  assert(directiveLanguages.size() != 0 && "DirectiveLanguage missing.");

  const auto &clauses = recordKeeper.getAllDerivedDefinitions("Clause");

  for (const auto &r : clauses) {
    Clause c{r};
    const auto &clauseVals = c.getClauseVals();
    if (clauseVals.size() <= 0)
      continue;

    const auto enumName = c.getEnumName();
    assert(enumName.size() != 0 && "enumClauseValue field not set.");

    std::vector<std::string> cvDefs;
    for (const auto &cv : clauseVals) {
      ClauseVal cval{cv};
      if (!cval.isUserVisible())
        continue;

      const auto name = cval.getFormattedName();
      std::string cvDef{(enumName + llvm::Twine(name)).str()};
      os << "def " << cvDef << " : StrEnumAttrCase<\"" << name << "\">;\n";
      cvDefs.push_back(cvDef);
    }

    os << "def " << enumName << ": StrEnumAttr<\n";
    os << "  \"Clause" << enumName << "\",\n";
    os << "  \"" << enumName << " Clause\",\n";
    os << "  [";
    for (unsigned int i = 0; i < cvDefs.size(); i++) {
      os << cvDefs[i];
      if (i != cvDefs.size() - 1)
        os << ",";
    }
    os << "]> {\n";
    os << "    let cppNamespace = \"::mlir::"
       << directiveLanguages[0]->getValueAsString("cppNamespace") << "\";\n";
    os << "}\n";
  }
  return false;
}

// Registers the generator to mlir-tblgen.
static mlir::GenRegistration genDirectiveDecls(
    "gen-directive-decl",
    "Generate declarations for directives (OpenMP/OpenACC etc.)",
    [](const RecordKeeper &records, raw_ostream &os) {
      return emitDecls(records, os);
    });
