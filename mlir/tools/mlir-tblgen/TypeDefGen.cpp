//===- TypeDefGen.cpp - MLIR typeDef definitions generator ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TypeDefGen uses the description of typeDefs to generate C++ definitions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/LogicalResult.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/TypeDef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "mlir-tblgen-typedefgen"

using namespace mlir;
using namespace mlir::tblgen;

static llvm::cl::OptionCategory typedefGenCat("Options for -gen-typedef-*");
static llvm::cl::opt<std::string>
    selectedDialect("typedefs-dialect",
                    llvm::cl::desc("Gen types for this dialect"),
                    llvm::cl::cat(typedefGenCat), llvm::cl::CommaSeparated);

/// Find all the TypeDefs for the specified dialect. If no dialect specified and
/// can only find one dialect's types, use that.
static void findAllTypeDefs(const llvm::RecordKeeper &recordKeeper,
                            SmallVectorImpl<TypeDef> &typeDefs) {
  auto recDefs = recordKeeper.getAllDerivedDefinitions("TypeDef");
  auto defs = llvm::map_range(
      recDefs, [&](const llvm::Record *rec) { return TypeDef(rec); });
  if (defs.empty())
    return;

  StringRef dialectName;
  if (selectedDialect.getNumOccurrences() == 0) {
    if (defs.empty())
      return;

    llvm::SmallSet<Dialect, 4> dialects;
    for (const TypeDef &typeDef : defs)
      dialects.insert(typeDef.getDialect());
    if (dialects.size() != 1)
      llvm::PrintFatalError("TypeDefs belonging to more than one dialect. Must "
                            "select one via '--typedefs-dialect'");

    dialectName = (*dialects.begin()).getName();
  } else if (selectedDialect.getNumOccurrences() == 1) {
    dialectName = selectedDialect.getValue();
  } else {
    llvm::PrintFatalError("Cannot select multiple dialects for which to "
                          "generate types via '--typedefs-dialect'.");
  }

  for (const TypeDef &typeDef : defs)
    if (typeDef.getDialect().getName().equals(dialectName))
      typeDefs.push_back(typeDef);
}

namespace {

/// Pass an instance of this class to llvm::formatv() to emit a comma separated
/// list of parameters in the format by 'EmitFormat'.
class TypeParamCommaFormatter : public llvm::detail::format_adapter {
public:
  /// Choose the output format
  enum EmitFormat {
    /// Emit "parameter1Type parameter1Name, parameter2Type parameter2Name,
    /// [...]".
    TypeNamePairs,

    /// Emit "parameter1(parameter1), parameter2(parameter2), [...]".
    TypeNameInitializer,

    /// Emit "param1Name, param2Name, [...]".
    JustParams,
  };

  TypeParamCommaFormatter(EmitFormat emitFormat, ArrayRef<TypeParameter> params,
                          bool prependComma = true)
      : emitFormat(emitFormat), params(params), prependComma(prependComma) {}

  /// llvm::formatv will call this function when using an instance as a
  /// replacement value.
  void format(raw_ostream &os, StringRef options) override {
    if (params.size() && prependComma)
      os << ", ";

    switch (emitFormat) {
    case EmitFormat::TypeNamePairs:
      interleaveComma(params, os,
                      [&](const TypeParameter &p) { emitTypeNamePair(p, os); });
      break;
    case EmitFormat::TypeNameInitializer:
      interleaveComma(params, os, [&](const TypeParameter &p) {
        emitTypeNameInitializer(p, os);
      });
      break;
    case EmitFormat::JustParams:
      interleaveComma(params, os,
                      [&](const TypeParameter &p) { os << p.getName(); });
      break;
    }
  }

private:
  // Emit "paramType paramName".
  static void emitTypeNamePair(const TypeParameter &param, raw_ostream &os) {
    os << param.getCppType() << " " << param.getName();
  }
  // Emit "paramName(paramName)"
  void emitTypeNameInitializer(const TypeParameter &param, raw_ostream &os) {
    os << param.getName() << "(" << param.getName() << ")";
  }

  EmitFormat emitFormat;
  ArrayRef<TypeParameter> params;
  bool prependComma;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// GEN: TypeDef declarations
//===----------------------------------------------------------------------===//

/// Print this above all the other declarations. Contains type declarations used
/// later on.
static const char *const typeDefDeclHeader = R"(
namespace mlir {
class DialectAsmParser;
class DialectAsmPrinter;
} // namespace mlir
)";

/// The code block for the start of a typeDef class declaration -- singleton
/// case.
///
/// {0}: The name of the typeDef class.
static const char *const typeDefDeclSingletonBeginStr = R"(
  class {0}: public ::mlir::Type::TypeBase<{0}, ::mlir::Type, ::mlir::TypeStorage> {{
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

)";

/// The code block for the start of a typeDef class declaration -- parametric
/// case.
///
/// {0}: The name of the typeDef class.
/// {1}: The typeDef storage class namespace.
/// {2}: The storage class name.
/// {3}: The list of parameters with types.
static const char *const typeDefDeclParametricBeginStr = R"(
  namespace {1} {
    struct {2};
  }
  class {0}: public ::mlir::Type::TypeBase<{0}, ::mlir::Type,
                                        {1}::{2}> {{
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

)";

/// The snippet for print/parse.
static const char *const typeDefParsePrint = R"(
    static ::mlir::Type parse(::mlir::MLIRContext* ctxt, ::mlir::DialectAsmParser& parser);
    void print(::mlir::DialectAsmPrinter& printer) const;
)";

/// The code block for the verifyConstructionInvariants and getChecked.
///
/// {0}: List of parameters, parameters style.
static const char *const typeDefDeclVerifyStr = R"(
    static ::mlir::LogicalResult verifyConstructionInvariants(::mlir::Location loc{0});
    static ::mlir::Type getChecked(::mlir::Location loc{0});
)";

/// Generate the declaration for the given typeDef class.
static void emitTypeDefDecl(const TypeDef &typeDef, raw_ostream &os) {
  SmallVector<TypeParameter, 4> params;
  typeDef.getParameters(params);

  // Emit the beginning string template: either the singleton or parametric
  // template.
  if (typeDef.getNumParameters() == 0)
    os << formatv(typeDefDeclSingletonBeginStr, typeDef.getCppClassName(),
                  typeDef.getStorageNamespace(), typeDef.getStorageClassName());
  else
    os << formatv(typeDefDeclParametricBeginStr, typeDef.getCppClassName(),
                  typeDef.getStorageNamespace(), typeDef.getStorageClassName());

  // Emit the extra declarations first in case there's a type definition in
  // there.
  if (Optional<StringRef> extraDecl = typeDef.getExtraDecls())
    os << *extraDecl << "\n";

  TypeParamCommaFormatter emitTypeNamePairsAfterComma(
      TypeParamCommaFormatter::EmitFormat::TypeNamePairs, params);
  os << llvm::formatv("    static {0} get(::mlir::MLIRContext* ctxt{1});\n",
                      typeDef.getCppClassName(), emitTypeNamePairsAfterComma);

  // Emit the verify invariants declaration.
  if (typeDef.genVerifyInvariantsDecl())
    os << llvm::formatv(typeDefDeclVerifyStr, emitTypeNamePairsAfterComma);

  // Emit the mnenomic, if specified.
  if (auto mnenomic = typeDef.getMnemonic()) {
    os << "    static ::llvm::StringRef getMnemonic() { return \"" << mnenomic
       << "\"; }\n";

    // If mnemonic specified, emit print/parse declarations.
    os << typeDefParsePrint;
  }

  if (typeDef.genAccessors()) {
    SmallVector<TypeParameter, 4> parameters;
    typeDef.getParameters(parameters);

    for (TypeParameter &parameter : parameters) {
      SmallString<16> name = parameter.getName();
      name[0] = llvm::toUpper(name[0]);
      os << formatv("    {0} get{1}() const;\n", parameter.getCppType(), name);
    }
  }

  // End the typeDef decl.
  os << "  };\n";
}

/// Main entry point for decls.
static bool emitTypeDefDecls(const llvm::RecordKeeper &recordKeeper,
                             raw_ostream &os) {
  emitSourceFileHeader("TypeDef Declarations", os);

  SmallVector<TypeDef, 16> typeDefs;
  findAllTypeDefs(recordKeeper, typeDefs);

  IfDefScope scope("GET_TYPEDEF_CLASSES", os);

  // Output the common "header".
  os << typeDefDeclHeader;

  if (typeDefs.size() > 0) {
    NamespaceEmitter nsEmitter(os, typeDefs.begin()->getDialect());

    // Well known print/parse dispatch function declarations. These are called
    // from Dialect::parseType() and Dialect::printType() methods.
    os << "  ::mlir::Type generatedTypeParser(::mlir::MLIRContext* ctxt, "
          "::mlir::DialectAsmParser& parser, ::llvm::StringRef mnenomic);\n";
    os << "  ::mlir::LogicalResult generatedTypePrinter(::mlir::Type type, "
          "::mlir::DialectAsmPrinter& printer);\n";
    os << "\n";

    // Declare all the type classes first (in case they reference each other).
    for (const TypeDef &typeDef : typeDefs)
      os << "  class " << typeDef.getCppClassName() << ";\n";

    // Declare all the typedefs.
    for (const TypeDef &typeDef : typeDefs)
      emitTypeDefDecl(typeDef, os);
  }

  return false;
}

//===----------------------------------------------------------------------===//
// GEN: TypeDef list
//===----------------------------------------------------------------------===//

static void emitTypeDefList(SmallVectorImpl<TypeDef> &typeDefs,
                            raw_ostream &os) {
  IfDefScope scope("GET_TYPEDEF_LIST", os);
  for (auto *i = typeDefs.begin(); i != typeDefs.end(); i++) {
    os << i->getDialect().getCppNamespace() << "::" << i->getCppClassName();
    if (i < typeDefs.end() - 1)
      os << ",\n";
    else
      os << "\n";
  }
}

//===----------------------------------------------------------------------===//
// GEN: TypeDef definitions
//===----------------------------------------------------------------------===//

/// Beginning of storage class.
/// {0}: Storage class namespace.
/// {1}: Storage class c++ name.
/// {2}: Parameters parameters.
/// {3}: Parameter initialzer string.
/// {4}: Parameter name list.
/// {5}: Parameter types.
static const char *const typeDefStorageClassBegin = R"(
namespace {0} {{
  struct {1} : public ::mlir::TypeStorage {{
    {1} ({2})
      : {3} {{ }

    /// The hash key for this storage is a pair of the integer and type params.
    using KeyTy = std::tuple<{5}>;

    /// Define the comparison function for the key type.
    bool operator==(const KeyTy &key) const {{
      return key == KeyTy({4});
    }
)";

/// The storage class' constructor template.
/// {0}: storage class name.
static const char *const typeDefStorageClassConstructorBegin = R"(
    /// Define a construction method for creating a new instance of this storage.
    static {0} *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &key) {{
)";

/// The storage class' constructor return template.
/// {0}: storage class name.
/// {1}: list of parameters.
static const char *const typeDefStorageClassConstructorReturn = R"(
      return new (allocator.allocate<{0}>())
          {0}({1});
    }
)";

/// The code block for the getChecked definition.
///
/// {0}: List of parameters, parameters style.
/// {1}: C++ type class name.
/// {2}: Comma separated list of parameter names.
static const char *const typeDefDefGetCheckeStr = R"(
    ::mlir::Type {1}::getChecked(Location loc{0}) {{
      return Base::getChecked(loc{2});
    }
)";

/// Use tgfmt to emit custom allocation code for each parameter, if necessary.
static void emitParameterAllocationCode(TypeDef &typeDef, raw_ostream &os) {
  SmallVector<TypeParameter, 4> parameters;
  typeDef.getParameters(parameters);
  auto fmtCtxt = FmtContext().addSubst("_allocator", "allocator");
  for (TypeParameter &parameter : parameters) {
    auto allocCode = parameter.getAllocator();
    if (allocCode) {
      fmtCtxt.withSelf(parameter.getName());
      fmtCtxt.addSubst("_dst", parameter.getName());
      os << "      " << tgfmt(*allocCode, &fmtCtxt) << "\n";
    }
  }
}

/// Emit the storage class code for type 'typeDef'.
/// This includes (in-order):
///  1) typeDefStorageClassBegin, which includes:
///      - The class constructor.
///      - The KeyTy definition.
///      - The equality (==) operator.
///  2) The hashKey method.
///  3) The construct method.
///  4) The list of parameters as the storage class member variables.
static void emitStorageClass(TypeDef typeDef, raw_ostream &os) {
  SmallVector<TypeParameter, 4> parameters;
  typeDef.getParameters(parameters);

  // Initialize a bunch of variables to be used later on.
  auto parameterNames = map_range(
      parameters, [](TypeParameter parameter) { return parameter.getName(); });
  auto parameterTypes = map_range(parameters, [](TypeParameter parameter) {
    return parameter.getCppType();
  });
  auto parameterList = join(parameterNames, ", ");
  auto parameterTypeList = join(parameterTypes, ", ");

  // 1) Emit most of the storage class up until the hashKey body.
  os << formatv(typeDefStorageClassBegin, typeDef.getStorageNamespace(),
                typeDef.getStorageClassName(),
                TypeParamCommaFormatter(
                    TypeParamCommaFormatter::EmitFormat::TypeNamePairs,
                    parameters, /*prependComma=*/false),
                TypeParamCommaFormatter(
                    TypeParamCommaFormatter::EmitFormat::TypeNameInitializer,
                    parameters, /*prependComma=*/false),
                parameterList, parameterTypeList);

  // 2) Emit the haskKey method.
  os << "  static ::llvm::hash_code hashKey(const KeyTy &key) {\n";
  // Extract each parameter from the key.
  for (size_t i = 0, e = parameters.size(); i < e; ++i)
    os << llvm::formatv("      const auto &{0} = std::get<{1}>(key);\n",
                        parameters[i].getName(), i);
  // Then combine them all. This requires all the parameters types to have a
  // hash_value defined.
  os << llvm::formatv(
      "      return ::llvm::hash_combine({0});\n    }\n",
      TypeParamCommaFormatter(TypeParamCommaFormatter::EmitFormat::JustParams,
                              parameters, /* prependComma */ false));

  // 3) Emit the construct method.
  if (typeDef.hasStorageCustomConstructor())
    // If user wants to build the storage constructor themselves, declare it
    // here and then they can write the definition elsewhere.
    os << "    static " << typeDef.getStorageClassName()
       << " *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy "
          "&key);\n";
  else {
    // If not, autogenerate one.

    // First, unbox the parameters.
    os << formatv(typeDefStorageClassConstructorBegin,
                  typeDef.getStorageClassName());
    for (size_t i = 0; i < parameters.size(); ++i) {
      os << formatv("      auto {0} = std::get<{1}>(key);\n",
                    parameters[i].getName(), i);
    }
    // Second, reassign the parameter variables with allocation code, if it's
    // specified.
    emitParameterAllocationCode(typeDef, os);

    // Last, return an allocated copy.
    os << formatv(typeDefStorageClassConstructorReturn,
                  typeDef.getStorageClassName(), parameterList);
  }

  // 4) Emit the parameters as storage class members.
  for (auto parameter : parameters) {
    os << "      " << parameter.getCppType() << " " << parameter.getName()
       << ";\n";
  }
  os << "  };\n";

  os << "} // namespace " << typeDef.getStorageNamespace() << "\n";
}

/// Emit the parser and printer for a particular type, if they're specified.
void emitParserPrinter(TypeDef typeDef, raw_ostream &os) {
  // Emit the printer code, if specified.
  if (auto printerCode = typeDef.getPrinterCode()) {
    // Both the mnenomic and printerCode must be defined (for parity with
    // parserCode).
    os << "void " << typeDef.getCppClassName()
       << "::print(::mlir::DialectAsmPrinter& printer) const {\n";
    if (*printerCode == "") {
      // If no code specified, emit error.
      PrintFatalError(typeDef.getLoc(),
                      typeDef.getName() +
                          ": printer (if specified) must have non-empty code");
    }
    auto fmtCtxt = FmtContext().addSubst("_printer", "printer");
    os << tgfmt(*printerCode, &fmtCtxt) << "\n}\n";
  }

  // emit a parser, if specified.
  if (auto parserCode = typeDef.getParserCode()) {
    // The mnenomic must be defined so the dispatcher knows how to dispatch.
    os << "::mlir::Type " << typeDef.getCppClassName()
       << "::parse(::mlir::MLIRContext* ctxt, ::mlir::DialectAsmParser& "
          "parser) "
          "{\n";
    if (*parserCode == "") {
      // if no code specified, emit error.
      PrintFatalError(typeDef.getLoc(),
                      typeDef.getName() +
                          ": parser (if specified) must have non-empty code");
    }
    auto fmtCtxt =
        FmtContext().addSubst("_parser", "parser").addSubst("_ctxt", "ctxt");
    os << tgfmt(*parserCode, &fmtCtxt) << "\n}\n";
  }
}

/// Print all the typedef-specific definition code.
static void emitTypeDefDef(TypeDef typeDef, raw_ostream &os) {
  NamespaceEmitter ns(os, typeDef.getDialect());
  SmallVector<TypeParameter, 4> parameters;
  typeDef.getParameters(parameters);

  // Emit the storage class, if requested and necessary.
  if (typeDef.genStorageClass() && typeDef.getNumParameters() > 0)
    emitStorageClass(typeDef, os);

  os << llvm::formatv(
      "{0} {0}::get(::mlir::MLIRContext* ctxt{1}) {{\n"
      "  return Base::get(ctxt{2});\n}\n",
      typeDef.getCppClassName(),
      TypeParamCommaFormatter(
          TypeParamCommaFormatter::EmitFormat::TypeNamePairs, parameters),
      TypeParamCommaFormatter(TypeParamCommaFormatter::EmitFormat::JustParams,
                              parameters));

  // Emit the parameter accessors.
  if (typeDef.genAccessors())
    for (const TypeParameter &parameter : parameters) {
      SmallString<16> name = parameter.getName();
      name[0] = llvm::toUpper(name[0]);
      os << formatv("{0} {3}::get{1}() const { return getImpl()->{2}; }\n",
                    parameter.getCppType(), name, parameter.getName(),
                    typeDef.getCppClassName());
    }

  // Generate getChecked() method.
  if (typeDef.genVerifyInvariantsDecl()) {
    os << llvm::formatv(
        typeDefDefGetCheckeStr,
        TypeParamCommaFormatter(
            TypeParamCommaFormatter::EmitFormat::TypeNamePairs, parameters),
        typeDef.getCppClassName(),
        TypeParamCommaFormatter(TypeParamCommaFormatter::EmitFormat::JustParams,
                                parameters));
  }

  // If mnemonic is specified maybe print definitions for the parser and printer
  // code, if they're specified.
  if (typeDef.getMnemonic())
    emitParserPrinter(typeDef, os);
}

/// Emit the dialect printer/parser dispatcher. User's code should call these
/// functions from their dialect's print/parse methods.
static void emitParsePrintDispatch(SmallVectorImpl<TypeDef> &typeDefs,
                                   raw_ostream &os) {
  if (typeDefs.size() == 0)
    return;
  const Dialect &dialect = typeDefs.begin()->getDialect();
  NamespaceEmitter ns(os, dialect);

  // The parser dispatch is just a list of if-elses, matching on the mnemonic
  // and calling the class's parse function.
  os << "::mlir::Type generatedTypeParser(::mlir::MLIRContext* ctxt, "
        "::mlir::DialectAsmParser& parser, ::llvm::StringRef mnemonic) {\n";
  for (const TypeDef &typeDef : typeDefs)
    if (typeDef.getMnemonic())
      os << formatv("  if (mnemonic == {0}::{1}::getMnemonic()) return "
                    "{0}::{1}::parse(ctxt, parser);\n",
                    typeDef.getDialect().getCppNamespace(),
                    typeDef.getCppClassName());
  os << "  return ::mlir::Type();\n";
  os << "}\n\n";

  // The printer dispatch uses llvm::TypeSwitch to find and call the correct
  // printer.
  os << "::mlir::LogicalResult generatedTypePrinter(::mlir::Type type, "
        "::mlir::DialectAsmPrinter& printer) {\n"
     << "  ::mlir::LogicalResult found = ::mlir::success();\n"
     << "  ::llvm::TypeSwitch<::mlir::Type>(type)\n";
  for (auto typeDef : typeDefs)
    if (typeDef.getMnemonic())
      os << formatv("    .Case<{0}::{1}>([&](::mlir::Type t) {{ "
                    "t.dyn_cast<{0}::{1}>().print(printer); })\n",
                    typeDef.getDialect().getCppNamespace(),
                    typeDef.getCppClassName());
  os << "    .Default([&found](::mlir::Type) { found = ::mlir::failure(); "
        "});\n"
     << "  return found;\n"
     << "}\n\n";
}

/// Entry point for typedef definitions.
static bool emitTypeDefDefs(const llvm::RecordKeeper &recordKeeper,
                            raw_ostream &os) {
  emitSourceFileHeader("TypeDef Definitions", os);

  SmallVector<TypeDef, 16> typeDefs;
  findAllTypeDefs(recordKeeper, typeDefs);
  emitTypeDefList(typeDefs, os);

  IfDefScope scope("GET_TYPEDEF_CLASSES", os);
  emitParsePrintDispatch(typeDefs, os);
  for (auto typeDef : typeDefs)
    emitTypeDefDef(typeDef, os);

  return false;
}

//===----------------------------------------------------------------------===//
// GEN: TypeDef registration hooks
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genTypeDefDefs("gen-typedef-defs", "Generate TypeDef definitions",
                   [](const llvm::RecordKeeper &records, raw_ostream &os) {
                     return emitTypeDefDefs(records, os);
                   });

static mlir::GenRegistration
    genTypeDefDecls("gen-typedef-decls", "Generate TypeDef declarations",
                    [](const llvm::RecordKeeper &records, raw_ostream &os) {
                      return emitTypeDefDecls(records, os);
                    });
