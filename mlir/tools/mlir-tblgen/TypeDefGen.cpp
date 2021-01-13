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
    for (const TypeDef typeDef : defs)
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

  for (const TypeDef typeDef : defs)
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
    if (!params.empty() && prependComma)
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
/// {1}: The name of the type base class.
static const char *const typeDefDeclSingletonBeginStr = R"(
  class {0} : public ::mlir::Type::TypeBase<{0}, {1}, ::mlir::TypeStorage> {{
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

)";

/// The code block for the start of a typeDef class declaration -- parametric
/// case.
///
/// {0}: The name of the typeDef class.
/// {1}: The name of the type base class.
/// {2}: The typeDef storage class namespace.
/// {3}: The storage class name.
/// {4}: The list of parameters with types.
static const char *const typeDefDeclParametricBeginStr = R"(
  namespace {2} {
    struct {3};
  } // end namespace {2}
  class {0} : public ::mlir::Type::TypeBase<{0}, {1},
                                         {2}::{3}> {{
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

)";

/// The snippet for print/parse.
static const char *const typeDefParsePrint = R"(
    static ::mlir::Type parse(::mlir::MLIRContext *context,
                              ::mlir::DialectAsmParser &parser);
    void print(::mlir::DialectAsmPrinter &printer) const;
)";

/// The code block for the verifyConstructionInvariants and getChecked.
///
/// {0}: The name of the typeDef class.
/// {1}: List of parameters, parameters style.
static const char *const typeDefDeclVerifyStr = R"(
    static ::mlir::LogicalResult verifyConstructionInvariants(::mlir::Location loc{1});
)";

/// Emit the builders for the given type.
static void emitTypeBuilderDecls(const TypeDef &typeDef, raw_ostream &os,
                                 TypeParamCommaFormatter &paramTypes) {
  StringRef typeClass = typeDef.getCppClassName();
  bool genCheckedMethods = typeDef.genVerifyInvariantsDecl();
  if (!typeDef.skipDefaultBuilders()) {
    os << llvm::formatv(
        "    static {0} get(::mlir::MLIRContext *context{1});\n", typeClass,
        paramTypes);
    if (genCheckedMethods) {
      os << llvm::formatv(
          "    static {0} getChecked(::mlir::Location loc{1});\n", typeClass,
          paramTypes);
    }
  }

  // Generate the builders specified by the user.
  for (const TypeBuilder &builder : typeDef.getBuilders()) {
    std::string paramStr;
    llvm::raw_string_ostream paramOS(paramStr);
    llvm::interleaveComma(
        builder.getParameters(), paramOS,
        [&](const TypeBuilder::Parameter &param) {
          // Note: TypeBuilder parameters are guaranteed to have names.
          paramOS << param.getCppType() << " " << *param.getName();
          if (Optional<StringRef> defaultParamValue = param.getDefaultValue())
            paramOS << " = " << *defaultParamValue;
        });
    paramOS.flush();

    // Generate the `get` variant of the builder.
    os << "    static " << typeClass << " get(";
    if (!builder.hasInferredContextParameter()) {
      os << "::mlir::MLIRContext *context";
      if (!paramStr.empty())
        os << ", ";
    }
    os << paramStr << ");\n";

    // Generate the `getChecked` variant of the builder.
    if (genCheckedMethods) {
      os << "    static " << typeClass << " getChecked(::mlir::Location loc";
      if (!paramStr.empty())
        os << ", " << paramStr;
      os << ");\n";
    }
  }
}

/// Generate the declaration for the given typeDef class.
static void emitTypeDefDecl(const TypeDef &typeDef, raw_ostream &os) {
  SmallVector<TypeParameter, 4> params;
  typeDef.getParameters(params);

  // Emit the beginning string template: either the singleton or parametric
  // template.
  if (typeDef.getNumParameters() == 0)
    os << formatv(typeDefDeclSingletonBeginStr, typeDef.getCppClassName(),
                  typeDef.getCppBaseClassName());
  else
    os << formatv(typeDefDeclParametricBeginStr, typeDef.getCppClassName(),
                  typeDef.getCppBaseClassName(), typeDef.getStorageNamespace(),
                  typeDef.getStorageClassName());

  // Emit the extra declarations first in case there's a type definition in
  // there.
  if (Optional<StringRef> extraDecl = typeDef.getExtraDecls())
    os << *extraDecl << "\n";

  TypeParamCommaFormatter emitTypeNamePairsAfterComma(
      TypeParamCommaFormatter::EmitFormat::TypeNamePairs, params);
  if (!params.empty()) {
    emitTypeBuilderDecls(typeDef, os, emitTypeNamePairsAfterComma);

    // Emit the verify invariants declaration.
    if (typeDef.genVerifyInvariantsDecl())
      os << llvm::formatv(typeDefDeclVerifyStr, typeDef.getCppClassName(),
                          emitTypeNamePairsAfterComma);
  }

  // Emit the mnenomic, if specified.
  if (auto mnenomic = typeDef.getMnemonic()) {
    os << "    static ::llvm::StringRef getMnemonic() { return \"" << mnenomic
       << "\"; }\n";

    // If mnemonic specified, emit print/parse declarations.
    if (typeDef.getParserCode() || typeDef.getPrinterCode() || !params.empty())
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

  if (!typeDefs.empty()) {
    NamespaceEmitter nsEmitter(os, typeDefs.begin()->getDialect());

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
/// {3}: Parameter initializer string.
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
  if (typeDef.hasStorageCustomConstructor()) {
    // If user wants to build the storage constructor themselves, declare it
    // here and then they can write the definition elsewhere.
    os << "    static " << typeDef.getStorageClassName()
       << " *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy "
          "&key);\n";
  } else {
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
       << "::print(::mlir::DialectAsmPrinter &printer) const {\n";
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
       << "::parse(::mlir::MLIRContext *context, ::mlir::DialectAsmParser &"
          "parser) "
          "{\n";
    if (*parserCode == "") {
      // if no code specified, emit error.
      PrintFatalError(typeDef.getLoc(),
                      typeDef.getName() +
                          ": parser (if specified) must have non-empty code");
    }
    auto fmtCtxt =
        FmtContext().addSubst("_parser", "parser").addSubst("_ctxt", "context");
    os << tgfmt(*parserCode, &fmtCtxt) << "\n}\n";
  }
}

/// Emit the builders for the given type.
static void emitTypeBuilderDefs(const TypeDef &typeDef, raw_ostream &os,
                                ArrayRef<TypeParameter> typeDefParams) {
  bool genCheckedMethods = typeDef.genVerifyInvariantsDecl();
  StringRef typeClass = typeDef.getCppClassName();
  if (!typeDef.skipDefaultBuilders()) {
    os << llvm::formatv(
        "{0} {0}::get(::mlir::MLIRContext *context{1}) {{\n"
        "  return Base::get(context{2});\n}\n",
        typeClass,
        TypeParamCommaFormatter(
            TypeParamCommaFormatter::EmitFormat::TypeNamePairs, typeDefParams),
        TypeParamCommaFormatter(TypeParamCommaFormatter::EmitFormat::JustParams,
                                typeDefParams));
    if (genCheckedMethods) {
      os << llvm::formatv(
          "{0} {0}::getChecked(::mlir::Location loc{1}) {{\n"
          "  return Base::getChecked(loc{2});\n}\n",
          typeClass,
          TypeParamCommaFormatter(
              TypeParamCommaFormatter::EmitFormat::TypeNamePairs,
              typeDefParams),
          TypeParamCommaFormatter(
              TypeParamCommaFormatter::EmitFormat::JustParams, typeDefParams));
    }
  }

  // Generate the builders specified by the user.
  auto builderFmtCtx = FmtContext().addSubst("_ctxt", "context");
  auto checkedBuilderFmtCtx = FmtContext()
                                  .addSubst("_loc", "loc")
                                  .addSubst("_ctxt", "loc.getContext()");
  for (const TypeBuilder &builder : typeDef.getBuilders()) {
    Optional<StringRef> body = builder.getBody();
    Optional<StringRef> checkedBody =
        genCheckedMethods ? builder.getCheckedBody() : llvm::None;
    if (!body && !checkedBody)
      continue;
    std::string paramStr;
    llvm::raw_string_ostream paramOS(paramStr);
    llvm::interleaveComma(builder.getParameters(), paramOS,
                          [&](const TypeBuilder::Parameter &param) {
                            // Note: TypeBuilder parameters are guaranteed to
                            // have names.
                            paramOS << param.getCppType() << " "
                                    << *param.getName();
                          });
    paramOS.flush();

    // Emit the `get` variant of the builder.
    if (body) {
      os << llvm::formatv("{0} {0}::get(", typeClass);
      if (!builder.hasInferredContextParameter()) {
        os << "::mlir::MLIRContext *context";
        if (!paramStr.empty())
          os << ", ";
        os << llvm::formatv("{0}) {{\n  {1};\n}\n", paramStr,
                            tgfmt(*body, &builderFmtCtx).str());
      } else {
        os << llvm::formatv("{0}) {{\n  {1};\n}\n", paramStr, *body);
      }
    }

    // Emit the `getChecked` variant of the builder.
    if (checkedBody) {
      os << llvm::formatv("{0} {0}::getChecked(::mlir::Location loc",
                          typeClass);
      if (!paramStr.empty())
        os << ", " << paramStr;
      os << llvm::formatv(") {{\n  {0};\n}\n",
                          tgfmt(*checkedBody, &checkedBuilderFmtCtx));
    }
  }
}

/// Print all the typedef-specific definition code.
static void emitTypeDefDef(const TypeDef &typeDef, raw_ostream &os) {
  NamespaceEmitter ns(os, typeDef.getDialect());

  SmallVector<TypeParameter, 4> parameters;
  typeDef.getParameters(parameters);
  if (!parameters.empty()) {
    // Emit the storage class, if requested and necessary.
    if (typeDef.genStorageClass())
      emitStorageClass(typeDef, os);

    // Emit the builders for this type.
    emitTypeBuilderDefs(typeDef, os, parameters);

    // Generate accessor definitions only if we also generate the storage class.
    // Otherwise, let the user define the exact accessor definition.
    if (typeDef.genAccessors() && typeDef.genStorageClass()) {
      // Emit the parameter accessors.
      for (const TypeParameter &parameter : parameters) {
        SmallString<16> name = parameter.getName();
        name[0] = llvm::toUpper(name[0]);
        os << formatv("{0} {3}::get{1}() const { return getImpl()->{2}; }\n",
                      parameter.getCppType(), name, parameter.getName(),
                      typeDef.getCppClassName());
      }
    }
  }

  // If mnemonic is specified maybe print definitions for the parser and printer
  // code, if they're specified.
  if (typeDef.getMnemonic())
    emitParserPrinter(typeDef, os);
}

/// Emit the dialect printer/parser dispatcher. User's code should call these
/// functions from their dialect's print/parse methods.
static void emitParsePrintDispatch(ArrayRef<TypeDef> types, raw_ostream &os) {
  if (llvm::none_of(types, [](const TypeDef &type) {
        return type.getMnemonic().hasValue();
      })) {
    return;
  }

  // The parser dispatch is just a list of if-elses, matching on the
  // mnemonic and calling the class's parse function.
  os << "static ::mlir::Type generatedTypeParser(::mlir::MLIRContext *"
        "context, ::mlir::DialectAsmParser &parser, "
        "::llvm::StringRef mnemonic) {\n";
  for (const TypeDef &type : types) {
    if (type.getMnemonic()) {
      os << formatv("  if (mnemonic == {0}::{1}::getMnemonic()) return "
                    "{0}::{1}::",
                    type.getDialect().getCppNamespace(),
                    type.getCppClassName());

      // If the type has no parameters and no parser code, just invoke a normal
      // `get`.
      if (type.getNumParameters() == 0 && !type.getParserCode())
        os << "get(context);\n";
      else
        os << "parse(context, parser);\n";
    }
  }
  os << "  return ::mlir::Type();\n";
  os << "}\n\n";

  // The printer dispatch uses llvm::TypeSwitch to find and call the correct
  // printer.
  os << "static ::mlir::LogicalResult generatedTypePrinter(::mlir::Type "
        "type, "
        "::mlir::DialectAsmPrinter &printer) {\n"
     << "  return ::llvm::TypeSwitch<::mlir::Type, "
        "::mlir::LogicalResult>(type)\n";
  for (const TypeDef &type : types) {
    if (Optional<StringRef> mnemonic = type.getMnemonic()) {
      StringRef cppNamespace = type.getDialect().getCppNamespace();
      StringRef cppClassName = type.getCppClassName();
      os << formatv("    .Case<{0}::{1}>([&]({0}::{1} t) {{\n      ",
                    cppNamespace, cppClassName);

      // If the type has no parameters and no printer code, just print the
      // mnemonic.
      if (type.getNumParameters() == 0 && !type.getPrinterCode())
        os << formatv("printer << {0}::{1}::getMnemonic();", cppNamespace,
                      cppClassName);
      else
        os << "t.print(printer);";
      os << "\n      return ::mlir::success();\n    })\n";
    }
  }
  os << "    .Default([](::mlir::Type) { return ::mlir::failure(); });\n"
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
  for (const TypeDef &typeDef : typeDefs)
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
