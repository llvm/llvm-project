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

#include "mlir/TableGen/CodeGenHelpers.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

/// The code block defining the base mixin class for combining clause operand
/// structures.
static const char *const baseMixinClass = R"(
namespace detail {
template <typename... Mixins>
struct Clauses : public Mixins... {};
} // namespace detail
)";

/// The code block defining operation argument structures.
static const char *const operationArgStruct = R"(
using {0}Operands = detail::Clauses<{1}>;
)";

/// Remove multiple optional prefixes and suffixes from \c str.
///
/// Prefixes and suffixes are attempted to be removed once in the order they
/// appear in the \c prefixes and \c suffixes arguments. All prefixes are
/// processed before suffixes are. This means it will behave as shown in the
/// following example:
///   - str: "PrePreNameSuf1Suf2"
///   - prefixes: ["Pre"]
///   - suffixes: ["Suf1", "Suf2"]
///   - return: "PreNameSuf1"
static StringRef stripPrefixAndSuffix(StringRef str,
                                      llvm::ArrayRef<StringRef> prefixes,
                                      llvm::ArrayRef<StringRef> suffixes) {
  for (StringRef prefix : prefixes)
    if (str.starts_with(prefix))
      str = str.drop_front(prefix.size());

  for (StringRef suffix : suffixes)
    if (str.ends_with(suffix))
      str = str.drop_back(suffix.size());

  return str;
}

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
  SmallVector<const Record *, 1> clauseSuperClasses;
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
  return stripPrefixAndSuffix(clauseClassName, /*prefixes=*/{"OpenMP_"},
                              /*suffixes=*/{"Skip", "Clause"});
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

/// Check that the given string record value, identified by its \c opValueName,
/// is either undefined or empty in both the given operation and clause record
/// or its contents for the clause record are contained in the operation record.
/// Passing a non-empty \c clauseValueName enables checking values named
/// differently in the operation and clause records.
static bool verifyStringValue(const Record *op, const Record *clause,
                              StringRef opValueName,
                              StringRef clauseValueName = {}) {
  auto opValue = op->getValueAsOptionalString(opValueName);
  auto clauseValue = clause->getValueAsOptionalString(
      clauseValueName.empty() ? opValueName : clauseValueName);

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
      !verifyStringValue(op, clause, "assemblyFormat", "reqAssemblyFormat"))
    PrintWarning(
        op->getLoc(),
        "'" + clauseClassName +
            "' clause-defined `reqAssemblyFormat` not present in operation. "
            "Consider concatenating `clauses[{Req,Opt}]AssemblyFormat` or "
            "explicitly skipping this field.");

  if (!clause->getValueAsBit("ignoreAsmFormat") &&
      !verifyStringValue(op, clause, "assemblyFormat", "optAssemblyFormat"))
    PrintWarning(
        op->getLoc(),
        "'" + clauseClassName +
            "' clause-defined `optAssemblyFormat` not present in operation. "
            "Consider concatenating `clauses[{Req,Opt}]AssemblyFormat` or "
            "explicitly skipping this field.");

  if (!clause->getValueAsBit("ignoreDesc") &&
      !verifyStringValue(op, clause, "description"))
    PrintError(op->getLoc(),
               "'" + clauseClassName +
                   "' clause-defined `description` not present in operation. "
                   "Consider concatenating `clausesDescription` or explicitly "
                   "skipping this field.");

  if (!clause->getValueAsBit("ignoreExtraDecl") &&
      !verifyStringValue(op, clause, "extraClassDeclaration"))
    PrintWarning(
        op->getLoc(),
        "'" + clauseClassName +
            "' clause-defined `extraClassDeclaration` not present in "
            "operation. Consider concatenating `clausesExtraClassDeclaration` "
            "or explicitly skipping this field.");
}

/// Translate the type of an OpenMP clause's argument to its corresponding
/// representation for clause operand structures.
///
/// All kinds of values are represented as `mlir::Value` fields, whereas
/// attributes are represented based on their `storageType`.
///
/// \param[in] name The name of the argument.
/// \param[in] init The `DefInit` object representing the argument.
/// \param[out] nest Number of levels of array nesting associated with the
///                  type. Must be initially set to 0.
/// \param[out] rank Rank (number of dimensions, if an array type) of the base
///                  type. Must be initially set to 1.
///
/// \return the name of the base type to represent elements of the argument
///         type.
static StringRef translateArgumentType(ArrayRef<SMLoc> loc, StringInit *name,
                                       Init *init, int &nest, int &rank) {
  const Record *def = cast<DefInit>(init)->getDef();

  llvm::StringSet<> superClasses;
  for (auto [sc, _] : def->getSuperClasses())
    superClasses.insert(sc->getNameInitAsString());

  // Handle wrapper-style superclasses.
  if (superClasses.contains("OptionalAttr"))
    return translateArgumentType(
        loc, name, def->getValue("baseAttr")->getValue(), nest, rank);

  if (superClasses.contains("TypedArrayAttrBase"))
    return translateArgumentType(
        loc, name, def->getValue("elementAttr")->getValue(), ++nest, rank);

  // Handle ElementsAttrBase superclasses.
  if (superClasses.contains("ElementsAttrBase")) {
    // TODO: Obtain the rank from ranked types.
    ++nest;

    if (superClasses.contains("IntElementsAttrBase"))
      return "::llvm::APInt";
    if (superClasses.contains("FloatElementsAttr") ||
        superClasses.contains("RankedFloatElementsAttr"))
      return "::llvm::APFloat";
    if (superClasses.contains("DenseArrayAttrBase"))
      return stripPrefixAndSuffix(def->getValueAsString("returnType"),
                                  {"::llvm::ArrayRef<"}, {">"});

    // Decrease the nesting depth in the case where the base type cannot be
    // inferred, so that the bare storageType is used instead of a vector.
    --nest;
    PrintWarning(
        loc,
        "could not infer array-like attribute element type for argument '" +
            name->getAsUnquotedString() + "', will use bare `storageType`");
  }

  // Handle simple attribute and value types.
  [[maybe_unused]] bool isAttr = superClasses.contains("Attr");
  bool isValue = superClasses.contains("TypeConstraint");
  if (superClasses.contains("Variadic"))
    ++nest;

  if (isValue) {
    assert(!isAttr &&
           "argument can't be simultaneously a value and an attribute");
    return "::mlir::Value";
  }

  assert(isAttr && "argument must be an attribute if it's not a value");
  return nest > 0 ? "::mlir::Attribute"
                  : def->getValueAsString("storageType").trim();
}

/// Generate the structure that represents the arguments of the given \c clause
/// record of type \c OpenMP_Clause.
///
/// It will contain a field for each argument, using the same name translated to
/// camel case and the corresponding base type as returned by
/// translateArgumentType() optionally wrapped in one or more llvm::SmallVector.
///
/// An additional field containing a tuple of integers to hold the size of each
/// dimension will also be created for multi-rank types. This is not yet
/// supported.
static void genClauseOpsStruct(const Record *clause, raw_ostream &os) {
  if (clause->isAnonymous())
    return;

  StringRef clauseName = extractOmpClauseName(clause);
  os << "struct " << clauseName << "ClauseOps {\n";

  DagInit *arguments = clause->getValueAsDag("arguments");
  for (auto [name, arg] :
       zip_equal(arguments->getArgNames(), arguments->getArgs())) {
    int nest = 0, rank = 1;
    StringRef baseType =
        translateArgumentType(clause->getLoc(), name, arg, nest, rank);
    std::string fieldName =
        convertToCamelFromSnakeCase(name->getAsUnquotedString(),
                                    /*capitalizeFirst=*/false);

    os << formatv("  {0}{1}{2} {3};\n",
                  fmt_repeat("::llvm::SmallVector<", nest), baseType,
                  fmt_repeat(">", nest), fieldName);

    if (rank > 1) {
      assert(nest >= 1 && "must be nested if it's a ranked type");
      os << formatv("  {0}::std::tuple<{1}int>{2} {3}Dims;\n",
                    fmt_repeat("::llvm::SmallVector<", nest - 1),
                    fmt_repeat("int, ", rank - 1), fmt_repeat(">", nest - 1),
                    fieldName);
    }
  }

  os << "};\n";
}

/// Generate the structure that represents the clause-related arguments of the
/// given \c op record of type \c OpenMP_Op.
///
/// This structure will be defined in terms of the clause operand structures
/// associated to the clauses of the operation.
static void genOperandsDef(const Record *op, raw_ostream &os) {
  if (op->isAnonymous())
    return;

  SmallVector<std::string> clauseNames;
  for (const Record *clause : op->getValueAsListOfDefs("clauseList"))
    clauseNames.push_back((extractOmpClauseName(clause) + "ClauseOps").str());

  StringRef opName = stripPrefixAndSuffix(
      op->getName(), /*prefixes=*/{"OpenMP_"}, /*suffixes=*/{"Op"});
  os << formatv(operationArgStruct, opName, join(clauseNames, ", "));
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

/// Generate structures to represent clause-related operands, based on existing
/// `OpenMP_Clause` definitions and aggregate them into operation-specific
/// structures according to the `clauses` argument of each definition deriving
/// from `OpenMP_Op`.
static bool genClauseOps(const RecordKeeper &recordKeeper, raw_ostream &os) {
  mlir::tblgen::NamespaceEmitter ns(os, "mlir::omp");
  for (const Record *clause :
       recordKeeper.getAllDerivedDefinitions("OpenMP_Clause"))
    genClauseOpsStruct(clause, os);

  // Produce base mixin class.
  os << baseMixinClass;

  for (const Record *op : recordKeeper.getAllDerivedDefinitions("OpenMP_Op"))
    genOperandsDef(op, os);

  return false;
}

// Registers the generator to mlir-tblgen.
static mlir::GenRegistration
    verifyOpenmpOps("verify-openmp-ops",
                    "Verify OpenMP operations (produce no output file)",
                    verifyDecls);

static mlir::GenRegistration
    genOpenmpClauseOps("gen-openmp-clause-ops",
                       "Generate OpenMP clause operand structures",
                       genClauseOps);
