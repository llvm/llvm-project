//===- AttrOrTypeDefGen.cpp - MLIR AttrOrType definitions generator -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/LogicalResult.h"
#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "mlir-tblgen-attrortypedefgen"

using namespace mlir;
using namespace mlir::tblgen;

/// Find all the AttrOrTypeDef for the specified dialect. If no dialect
/// specified and can only find one dialect's defs, use that.
static void collectAllDefs(StringRef selectedDialect,
                           std::vector<llvm::Record *> records,
                           SmallVectorImpl<AttrOrTypeDef> &resultDefs) {
  auto defs = llvm::map_range(
      records, [&](const llvm::Record *rec) { return AttrOrTypeDef(rec); });
  if (defs.empty())
    return;

  StringRef dialectName;
  if (selectedDialect.empty()) {
    if (defs.empty())
      return;

    Dialect dialect(nullptr);
    for (const AttrOrTypeDef &typeDef : defs) {
      if (!dialect) {
        dialect = typeDef.getDialect();
      } else if (dialect != typeDef.getDialect()) {
        llvm::PrintFatalError("defs belonging to more than one dialect. Must "
                              "select one via '--(attr|type)defs-dialect'");
      }
    }

    dialectName = dialect.getName();
  } else {
    dialectName = selectedDialect;
  }

  for (const AttrOrTypeDef &def : defs)
    if (def.getDialect().getName().equals(dialectName))
      resultDefs.push_back(def);
}

//===----------------------------------------------------------------------===//
// ParamCommaFormatter
//===----------------------------------------------------------------------===//

namespace {

/// Pass an instance of this class to llvm::formatv() to emit a comma separated
/// list of parameters in the format by 'EmitFormat'.
class ParamCommaFormatter : public llvm::detail::format_adapter {
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

  ParamCommaFormatter(EmitFormat emitFormat,
                      ArrayRef<AttrOrTypeParameter> params,
                      bool prependComma = true)
      : emitFormat(emitFormat), params(params), prependComma(prependComma) {}

  /// llvm::formatv will call this function when using an instance as a
  /// replacement value.
  void format(raw_ostream &os, StringRef options) override {
    if (!params.empty() && prependComma)
      os << ", ";

    switch (emitFormat) {
    case EmitFormat::TypeNamePairs:
      interleaveComma(params, os, [&](const AttrOrTypeParameter &p) {
        emitTypeNamePair(p, os);
      });
      break;
    case EmitFormat::TypeNameInitializer:
      interleaveComma(params, os, [&](const AttrOrTypeParameter &p) {
        emitTypeNameInitializer(p, os);
      });
      break;
    case EmitFormat::JustParams:
      interleaveComma(params, os,
                      [&](const AttrOrTypeParameter &p) { os << p.getName(); });
      break;
    }
  }

private:
  // Emit "paramType paramName".
  static void emitTypeNamePair(const AttrOrTypeParameter &param,
                               raw_ostream &os) {
    os << param.getCppType() << " " << param.getName();
  }
  // Emit "paramName(paramName)"
  void emitTypeNameInitializer(const AttrOrTypeParameter &param,
                               raw_ostream &os) {
    os << param.getName() << "(" << param.getName() << ")";
  }

  EmitFormat emitFormat;
  ArrayRef<AttrOrTypeParameter> params;
  bool prependComma;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// DefGenerator
//===----------------------------------------------------------------------===//

namespace {
/// This struct is the base generator used when processing tablegen interfaces.
class DefGenerator {
public:
  bool emitDecls(StringRef selectedDialect);
  bool emitDefs(StringRef selectedDialect);

protected:
  DefGenerator(std::vector<llvm::Record *> &&defs, raw_ostream &os)
      : defRecords(std::move(defs)), os(os), isAttrGenerator(false) {}

  /// Emit the declaration of a single def.
  void emitDefDecl(const AttrOrTypeDef &def);
  /// Emit the list of def type names.
  void emitTypeDefList(ArrayRef<AttrOrTypeDef> defs);
  /// Emit the code to dispatch between different defs during parsing/printing.
  void emitParsePrintDispatch(ArrayRef<AttrOrTypeDef> defs);
  /// Emit the definition of a single def.
  void emitDefDef(const AttrOrTypeDef &def);
  /// Emit the storage class for the given def.
  void emitStorageClass(const AttrOrTypeDef &def);
  /// Emit the parser/printer for the given def.
  void emitParsePrint(const AttrOrTypeDef &def);

  /// The set of def records to emit.
  std::vector<llvm::Record *> defRecords;
  /// The stream to emit to.
  raw_ostream &os;
  /// The prefix of the tablegen def name, e.g. Attr or Type.
  StringRef defTypePrefix;
  /// The C++ base value type of the def, e.g. Attribute or Type.
  StringRef valueType;
  /// Flag indicating if this generator is for Attributes. False if the
  /// generator is for types.
  bool isAttrGenerator;
};

/// A specialized generator for AttrDefs.
struct AttrDefGenerator : public DefGenerator {
  AttrDefGenerator(const llvm::RecordKeeper &records, raw_ostream &os)
      : DefGenerator(records.getAllDerivedDefinitions("AttrDef"), os) {
    isAttrGenerator = true;
    defTypePrefix = "Attr";
    valueType = "Attribute";
  }
};
/// A specialized generator for TypeDefs.
struct TypeDefGenerator : public DefGenerator {
  TypeDefGenerator(const llvm::RecordKeeper &records, raw_ostream &os)
      : DefGenerator(records.getAllDerivedDefinitions("TypeDef"), os) {
    defTypePrefix = "Type";
    valueType = "Type";
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// GEN: Declarations
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
/// {0}: The name of the def class.
/// {1}: The name of the type base class.
/// {2}: The name of the base value type, e.g. Attribute or Type.
/// {3}: The tablegen record type prefix, e.g. Attr or Type.
static const char *const defDeclSingletonBeginStr = R"(
  class {0} : public ::mlir::{2}::{3}Base<{0}, {1}, ::mlir::{2}Storage> {{
  public:
    /// Inherit some necessary constructors from '{3}Base'.
    using Base::Base;
)";

/// The code block for the start of a typeDef class declaration -- parametric
/// case.
///
/// {0}: The name of the typeDef class.
/// {1}: The name of the type base class.
/// {2}: The typeDef storage class namespace.
/// {3}: The storage class name.
/// {4}: The name of the base value type, e.g. Attribute or Type.
/// {5}: The tablegen record type prefix, e.g. Attr or Type.
static const char *const defDeclParametricBeginStr = R"(
  namespace {2} {
    struct {3};
  } // end namespace {2}
  class {0} : public ::mlir::{4}::{5}Base<{0}, {1},
                                         {2}::{3}> {{
  public:
    /// Inherit some necessary constructors from '{5}Base'.
    using Base::Base;

)";

/// The code snippet for print/parse of an Attribute/Type.
///
/// {0}: The name of the base value type, e.g. Attribute or Type.
/// {1}: Extra parser parameters.
static const char *const defDeclParsePrintStr = R"(
    static ::mlir::{0} parse(::mlir::MLIRContext *context,
                             ::mlir::DialectAsmParser &parser{1});
    void print(::mlir::DialectAsmPrinter &printer) const;
)";

/// The code block for the verify method declaration.
///
/// {0}: List of parameters, parameters style.
static const char *const defDeclVerifyStr = R"(
    using Base::getChecked;
    static ::mlir::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError{0});
)";

/// Emit the builders for the given def.
static void emitBuilderDecls(const AttrOrTypeDef &def, raw_ostream &os,
                             ParamCommaFormatter &paramTypes) {
  StringRef typeClass = def.getCppClassName();
  bool genCheckedMethods = def.genVerifyDecl();
  if (!def.skipDefaultBuilders()) {
    os << llvm::formatv(
        "    static {0} get(::mlir::MLIRContext *context{1});\n", typeClass,
        paramTypes);
    if (genCheckedMethods) {
      os << llvm::formatv("    static {0} "
                          "getChecked(llvm::function_ref<::mlir::"
                          "InFlightDiagnostic()> emitError, "
                          "::mlir::MLIRContext *context{1});\n",
                          typeClass, paramTypes);
    }
  }

  // Generate the builders specified by the user.
  for (const AttrOrTypeBuilder &builder : def.getBuilders()) {
    std::string paramStr;
    llvm::raw_string_ostream paramOS(paramStr);
    llvm::interleaveComma(
        builder.getParameters(), paramOS,
        [&](const AttrOrTypeBuilder::Parameter &param) {
          // Note: AttrOrTypeBuilder parameters are guaranteed to have names.
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
      os << "    static " << typeClass
         << " getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> "
            "emitError";
      if (!builder.hasInferredContextParameter())
        os << ", ::mlir::MLIRContext *context";
      if (!paramStr.empty())
        os << ", ";
      os << paramStr << ");\n";
    }
  }
}

void DefGenerator::emitDefDecl(const AttrOrTypeDef &def) {
  SmallVector<AttrOrTypeParameter, 4> params;
  def.getParameters(params);

  // Emit the beginning string template: either the singleton or parametric
  // template.
  if (def.getNumParameters() == 0) {
    os << formatv(defDeclSingletonBeginStr, def.getCppClassName(),
                  def.getCppBaseClassName(), valueType, defTypePrefix);
  } else {
    os << formatv(defDeclParametricBeginStr, def.getCppClassName(),
                  def.getCppBaseClassName(), def.getStorageNamespace(),
                  def.getStorageClassName(), valueType, defTypePrefix);
  }

  // Emit the extra declarations first in case there's a definition in there.
  if (Optional<StringRef> extraDecl = def.getExtraDecls())
    os << *extraDecl << "\n";

  ParamCommaFormatter emitTypeNamePairsAfterComma(
      ParamCommaFormatter::EmitFormat::TypeNamePairs, params);
  if (!params.empty()) {
    emitBuilderDecls(def, os, emitTypeNamePairsAfterComma);

    // Emit the verify invariants declaration.
    if (def.genVerifyDecl())
      os << llvm::formatv(defDeclVerifyStr, emitTypeNamePairsAfterComma);
  }

  // Emit the mnenomic, if specified.
  if (auto mnenomic = def.getMnemonic()) {
    os << "    static constexpr ::llvm::StringLiteral getMnemonic() {\n"
       << "      return ::llvm::StringLiteral(\"" << mnenomic << "\");\n"
       << "    }\n";

    // If mnemonic specified, emit print/parse declarations.
    if (def.getParserCode() || def.getPrinterCode() || !params.empty()) {
      os << llvm::formatv(defDeclParsePrintStr, valueType,
                          isAttrGenerator ? ", ::mlir::Type type" : "");
    }
  }

  if (def.genAccessors()) {
    SmallVector<AttrOrTypeParameter, 4> parameters;
    def.getParameters(parameters);

    for (AttrOrTypeParameter &parameter : parameters) {
      SmallString<16> name = parameter.getName();
      name[0] = llvm::toUpper(name[0]);
      os << formatv("    {0} get{1}() const;\n", parameter.getCppType(), name);
    }
  }

  // End the decl.
  os << "  };\n";
}

bool DefGenerator::emitDecls(StringRef selectedDialect) {
  emitSourceFileHeader((defTypePrefix + "Def Declarations").str(), os);
  IfDefScope scope("GET_" + defTypePrefix.upper() + "DEF_CLASSES", os);

  // Output the common "header".
  os << typeDefDeclHeader;

  SmallVector<AttrOrTypeDef, 16> defs;
  collectAllDefs(selectedDialect, defRecords, defs);
  if (defs.empty())
    return false;

  NamespaceEmitter nsEmitter(os, defs.front().getDialect());

  // Declare all the def classes first (in case they reference each other).
  for (const AttrOrTypeDef &def : defs)
    os << "  class " << def.getCppClassName() << ";\n";

  // Emit the declarations.
  for (const AttrOrTypeDef &def : defs)
    emitDefDecl(def);
  return false;
}

//===----------------------------------------------------------------------===//
// GEN: Def List
//===----------------------------------------------------------------------===//

void DefGenerator::emitTypeDefList(ArrayRef<AttrOrTypeDef> defs) {
  IfDefScope scope("GET_" + defTypePrefix.upper() + "DEF_LIST", os);
  auto interleaveFn = [&](const AttrOrTypeDef &def) {
    os << def.getDialect().getCppNamespace() << "::" << def.getCppClassName();
  };
  llvm::interleave(defs, os, interleaveFn, ",\n");
  os << "\n";
}

//===----------------------------------------------------------------------===//
// GEN: Definitions
//===----------------------------------------------------------------------===//

/// The code block used to start the auto-generated parser function.
///
/// {0}: The name of the base value type, e.g. Attribute or Type.
/// {1}: Additional parser parameters.
static const char *const defParserDispatchStartStr = R"(
static OptionalParseResult generated{0}Parser(::mlir::MLIRContext *context,
                                      ::mlir::DialectAsmParser &parser,
                                      ::llvm::StringRef mnemonic{1},
                                      ::mlir::{0} &value) {{
)";

/// The code block used to start the auto-generated printer function.
///
/// {0}: The name of the base value type, e.g. Attribute or Type.
static const char *const defPrinterDispatchStartStr = R"(
static ::mlir::LogicalResult generated{0}Printer(
                         ::mlir::{0} def, ::mlir::DialectAsmPrinter &printer) {{
  return ::llvm::TypeSwitch<::mlir::{0}, ::mlir::LogicalResult>(def)
)";

/// Beginning of storage class.
/// {0}: Storage class namespace.
/// {1}: Storage class c++ name.
/// {2}: Parameters parameters.
/// {3}: Parameter initializer string.
/// {4}: Parameter name list.
/// {5}: Parameter types.
/// {6}: The name of the base value type, e.g. Attribute or Type.
static const char *const defStorageClassBeginStr = R"(
namespace {0} {{
  struct {1} : public ::mlir::{6}Storage {{
    {1} ({2})
      : {3} {{ }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = std::tuple<{5}>;

    /// Define the comparison function for the key type.
    bool operator==(const KeyTy &key) const {{
      return key == KeyTy({4});
    }
)";

/// The storage class' constructor template.
///
/// {0}: storage class name.
/// {1}: The name of the base value type, e.g. Attribute or Type.
static const char *const defStorageClassConstructorBeginStr = R"(
    /// Define a construction method for creating a new instance of this
    /// storage.
    static {0} *construct(::mlir::{1}StorageAllocator &allocator,
                          const KeyTy &key) {{
)";

/// The storage class' constructor return template.
///
/// {0}: storage class name.
/// {1}: list of parameters.
static const char *const defStorageClassConstructorEndStr = R"(
      return new (allocator.allocate<{0}>())
          {0}({1});
    }
)";

/// Use tgfmt to emit custom allocation code for each parameter, if necessary.
static void emitStorageParameterAllocation(const AttrOrTypeDef &def,
                                           raw_ostream &os) {
  SmallVector<AttrOrTypeParameter> parameters;
  def.getParameters(parameters);
  FmtContext fmtCtxt = FmtContext().addSubst("_allocator", "allocator");
  for (AttrOrTypeParameter &parameter : parameters) {
    if (Optional<StringRef> allocCode = parameter.getAllocator()) {
      fmtCtxt.withSelf(parameter.getName());
      fmtCtxt.addSubst("_dst", parameter.getName());
      os << "      " << tgfmt(*allocCode, &fmtCtxt) << "\n";
    }
  }
}

/// Builds a code block that initializes the attribute storage of 'def'.
/// Attribute initialization is separated from Type initialization given that
/// the Attribute also needs to initialize its self-type, which has multiple
/// means of initialization.
static std::string buildAttributeStorageParamInitializer(
    const AttrOrTypeDef &def, ArrayRef<AttrOrTypeParameter> parameters) {
  std::string paramInitializer;
  llvm::raw_string_ostream paramOS(paramInitializer);
  paramOS << "::mlir::AttributeStorage(";

  // If this is an attribute, we need to check for value type initialization.
  Optional<size_t> selfParamIndex;
  for (auto it : llvm::enumerate(parameters)) {
    const auto *selfParam = dyn_cast<AttributeSelfTypeParameter>(&it.value());
    if (!selfParam)
      continue;
    if (selfParamIndex) {
      llvm::PrintFatalError(def.getLoc(),
                            "Only one attribute parameter can be marked as "
                            "AttributeSelfTypeParameter");
    }
    paramOS << selfParam->getName();
    selfParamIndex = it.index();
  }

  // If we didn't find a self param, but the def has a type builder we use that
  // to construct the type.
  if (!selfParamIndex) {
    const AttrDef &attrDef = cast<AttrDef>(def);
    if (Optional<StringRef> typeBuilder = attrDef.getTypeBuilder()) {
      FmtContext fmtContext;
      for (const AttrOrTypeParameter &param : parameters)
        fmtContext.addSubst(("_" + param.getName()).str(), param.getName());
      paramOS << tgfmt(*typeBuilder, &fmtContext);
    }
  }
  paramOS << ")";

  // Append the parameters to the initializer.
  for (auto it : llvm::enumerate(parameters))
    if (it.index() != selfParamIndex)
      paramOS << llvm::formatv(", {0}({0})", it.value().getName());

  return paramOS.str();
}

void DefGenerator::emitStorageClass(const AttrOrTypeDef &def) {
  SmallVector<AttrOrTypeParameter, 4> params;
  def.getParameters(params);

  // Collect the parameter types.
  auto parameterTypes =
      llvm::map_range(params, [](const AttrOrTypeParameter &parameter) {
        return parameter.getCppType();
      });
  std::string parameterTypeList = llvm::join(parameterTypes, ", ");

  // Collect the parameter initializer.
  std::string paramInitializer;
  if (isAttrGenerator) {
    paramInitializer = buildAttributeStorageParamInitializer(def, params);

  } else {
    llvm::raw_string_ostream initOS(paramInitializer);
    llvm::interleaveComma(params, initOS, [&](const AttrOrTypeParameter &it) {
      initOS << llvm::formatv("{0}({0})", it.getName());
    });
  }

  // Construct the parameter list that is used when a concrete instance of the
  // storage exists.
  auto nonStaticParameterNames = llvm::map_range(params, [](const auto &param) {
    return isa<AttributeSelfTypeParameter>(param) ? "getType()"
                                                  : param.getName();
  });

  // 1) Emit most of the storage class up until the hashKey body.
  os << formatv(
      defStorageClassBeginStr, def.getStorageNamespace(),
      def.getStorageClassName(),
      ParamCommaFormatter(ParamCommaFormatter::EmitFormat::TypeNamePairs,
                          params, /*prependComma=*/false),
      paramInitializer, llvm::join(nonStaticParameterNames, ", "),
      parameterTypeList, valueType);

  // 2) Emit the haskKey method.
  os << "  static ::llvm::hash_code hashKey(const KeyTy &key) {\n";

  // Extract each parameter from the key.
  os << "      return ::llvm::hash_combine(";
  llvm::interleaveComma(
      llvm::seq<unsigned>(0, params.size()), os,
      [&](unsigned it) { os << "std::get<" << it << ">(key)"; });
  os << ");\n    }\n";

  // 3) Emit the construct method.

  // If user wants to build the storage constructor themselves, declare it
  // here and then they can write the definition elsewhere.
  if (def.hasStorageCustomConstructor()) {
    os << llvm::formatv("    static {0} *construct(::mlir::{1}StorageAllocator "
                        "&allocator, const KeyTy &key);\n",
                        def.getStorageClassName(), valueType);

    // Otherwise, generate one.
  } else {
    // First, unbox the parameters.
    os << formatv(defStorageClassConstructorBeginStr, def.getStorageClassName(),
                  valueType);
    for (unsigned i = 0, e = params.size(); i < e; ++i) {
      os << formatv("      auto {0} = std::get<{1}>(key);\n",
                    params[i].getName(), i);
    }

    // Second, reassign the parameter variables with allocation code, if it's
    // specified.
    emitStorageParameterAllocation(def, os);

    // Last, return an allocated copy.
    auto parameterNames = llvm::map_range(
        params, [](const auto &param) { return param.getName(); });
    os << formatv(defStorageClassConstructorEndStr, def.getStorageClassName(),
                  llvm::join(parameterNames, ", "));
  }

  // 4) Emit the parameters as storage class members.
  for (const AttrOrTypeParameter &parameter : params) {
    // Attribute value types are not stored as fields in the storage.
    if (!isa<AttributeSelfTypeParameter>(parameter))
      os << "      " << parameter.getCppType() << " " << parameter.getName()
         << ";\n";
  }
  os << "  };\n";

  os << "} // namespace " << def.getStorageNamespace() << "\n";
}

void DefGenerator::emitParsePrint(const AttrOrTypeDef &def) {
  // Emit the printer code, if specified.
  if (Optional<StringRef> printerCode = def.getPrinterCode()) {
    // Both the mnenomic and printerCode must be defined (for parity with
    // parserCode).
    os << "void " << def.getCppClassName()
       << "::print(::mlir::DialectAsmPrinter &printer) const {\n";
    if (printerCode->empty()) {
      // If no code specified, emit error.
      PrintFatalError(def.getLoc(),
                      def.getName() +
                          ": printer (if specified) must have non-empty code");
    }
    FmtContext fmtCtxt = FmtContext().addSubst("_printer", "printer");
    os << tgfmt(*printerCode, &fmtCtxt) << "\n}\n";
  }

  // Emit the parser code, if specified.
  if (Optional<StringRef> parserCode = def.getParserCode()) {
    FmtContext fmtCtxt;
    fmtCtxt.addSubst("_parser", "parser").addSubst("_ctxt", "context");

    // The mnenomic must be defined so the dispatcher knows how to dispatch.
    os << llvm::formatv("::mlir::{0} {1}::parse(::mlir::MLIRContext *context, "
                        "::mlir::DialectAsmParser &parser",
                        valueType, def.getCppClassName());
    if (isAttrGenerator) {
      // Attributes also accept a type parameter instead of a context.
      os << ", ::mlir::Type type";
      fmtCtxt.addSubst("_type", "type");
    }
    os << ") {\n";

    if (parserCode->empty()) {
      PrintFatalError(def.getLoc(),
                      def.getName() +
                          ": parser (if specified) must have non-empty code");
    }
    os << tgfmt(*parserCode, &fmtCtxt) << "\n}\n";
  }
}

/// Replace all instances of 'from' to 'to' in `str` and return the new string.
static std::string replaceInStr(std::string str, StringRef from, StringRef to) {
  size_t pos = 0;
  while ((pos = str.find(from.data(), pos, from.size())) != std::string::npos)
    str.replace(pos, from.size(), to.data(), to.size());
  return str;
}

/// Emit the builders for the given def.
static void emitBuilderDefs(const AttrOrTypeDef &def, raw_ostream &os,
                            ArrayRef<AttrOrTypeParameter> params) {
  bool genCheckedMethods = def.genVerifyDecl();
  StringRef className = def.getCppClassName();
  if (!def.skipDefaultBuilders()) {
    os << llvm::formatv(
        "{0} {0}::get(::mlir::MLIRContext *context{1}) {{\n"
        "  return Base::get(context{2});\n}\n",
        className,
        ParamCommaFormatter(ParamCommaFormatter::EmitFormat::TypeNamePairs,
                            params),
        ParamCommaFormatter(ParamCommaFormatter::EmitFormat::JustParams,
                            params));
    if (genCheckedMethods) {
      os << llvm::formatv(
          "{0} {0}::getChecked("
          "llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, "
          "::mlir::MLIRContext *context{1}) {{\n"
          "  return Base::getChecked(emitError, context{2});\n}\n",
          className,
          ParamCommaFormatter(ParamCommaFormatter::EmitFormat::TypeNamePairs,
                              params),
          ParamCommaFormatter(ParamCommaFormatter::EmitFormat::JustParams,
                              params));
    }
  }

  auto builderFmtCtx =
      FmtContext().addSubst("_ctxt", "context").addSubst("_get", "Base::get");
  auto inferredCtxBuilderFmtCtx = FmtContext().addSubst("_get", "Base::get");
  auto checkedBuilderFmtCtx = FmtContext().addSubst("_ctxt", "context");

  // Generate the builders specified by the user.
  for (const AttrOrTypeBuilder &builder : def.getBuilders()) {
    Optional<StringRef> body = builder.getBody();
    if (!body)
      continue;
    std::string paramStr;
    llvm::raw_string_ostream paramOS(paramStr);
    llvm::interleaveComma(builder.getParameters(), paramOS,
                          [&](const AttrOrTypeBuilder::Parameter &param) {
                            // Note: AttrOrTypeBuilder parameters are guaranteed
                            // to have names.
                            paramOS << param.getCppType() << " "
                                    << *param.getName();
                          });
    paramOS.flush();

    // Emit the `get` variant of the builder.
    os << llvm::formatv("{0} {0}::get(", className);
    if (!builder.hasInferredContextParameter()) {
      os << "::mlir::MLIRContext *context";
      if (!paramStr.empty())
        os << ", ";
      os << llvm::formatv("{0}) {{\n  {1};\n}\n", paramStr,
                          tgfmt(*body, &builderFmtCtx).str());
    } else {
      os << llvm::formatv("{0}) {{\n  {1};\n}\n", paramStr,
                          tgfmt(*body, &inferredCtxBuilderFmtCtx).str());
    }

    // Emit the `getChecked` variant of the builder.
    if (genCheckedMethods) {
      os << llvm::formatv("{0} "
                          "{0}::getChecked(llvm::function_ref<::mlir::"
                          "InFlightDiagnostic()> emitErrorFn",
                          className);
      std::string checkedBody =
          replaceInStr(body->str(), "$_get(", "Base::getChecked(emitErrorFn, ");
      if (!builder.hasInferredContextParameter()) {
        os << ", ::mlir::MLIRContext *context";
        checkedBody = tgfmt(checkedBody, &checkedBuilderFmtCtx).str();
      }
      if (!paramStr.empty())
        os << ", ";
      os << llvm::formatv("{0}) {{\n  {1};\n}\n", paramStr, checkedBody);
    }
  }
}

/// Print all the def-specific definition code.
void DefGenerator::emitDefDef(const AttrOrTypeDef &def) {
  NamespaceEmitter ns(os, def.getDialect());

  SmallVector<AttrOrTypeParameter, 4> parameters;
  def.getParameters(parameters);
  if (!parameters.empty()) {
    // Emit the storage class, if requested and necessary.
    if (def.genStorageClass())
      emitStorageClass(def);

    // Emit the builders for this def.
    emitBuilderDefs(def, os, parameters);

    // Generate accessor definitions only if we also generate the storage class.
    // Otherwise, let the user define the exact accessor definition.
    if (def.genAccessors() && def.genStorageClass()) {
      for (const AttrOrTypeParameter &parameter : parameters) {
        StringRef paramStorageName = isa<AttributeSelfTypeParameter>(parameter)
                                         ? "getType()"
                                         : parameter.getName();

        SmallString<16> name = parameter.getName();
        name[0] = llvm::toUpper(name[0]);
        os << formatv("{0} {3}::get{1}() const {{ return getImpl()->{2}; }\n",
                      parameter.getCppType(), name, paramStorageName,
                      def.getCppClassName());
      }
    }
  }

  // If mnemonic is specified maybe print definitions for the parser and printer
  // code, if they're specified.
  if (def.getMnemonic())
    emitParsePrint(def);
}

/// Emit the dialect printer/parser dispatcher. User's code should call these
/// functions from their dialect's print/parse methods.
void DefGenerator::emitParsePrintDispatch(ArrayRef<AttrOrTypeDef> defs) {
  if (llvm::none_of(defs, [](const AttrOrTypeDef &def) {
        return def.getMnemonic().hasValue();
      })) {
    return;
  }

  // The parser dispatch is just a list of if-elses, matching on the mnemonic
  // and calling the def's parse function.
  os << llvm::formatv(defParserDispatchStartStr, valueType,
                      isAttrGenerator ? ", ::mlir::Type type" : "");
  for (const AttrOrTypeDef &def : defs) {
    if (def.getMnemonic()) {
      os << formatv("  if (mnemonic == {0}::{1}::getMnemonic()) { \n"
                    "    value = {0}::{1}::",
                    def.getDialect().getCppNamespace(), def.getCppClassName());

      // If the def has no parameters and no parser code, just invoke a normal
      // `get`.
      if (def.getNumParameters() == 0 && !def.getParserCode()) {
        os << "get(context);\n    return success(!!value);\n  }\n";
        continue;
      }

      os << "parse(context, parser" << (isAttrGenerator ? ", type" : "")
         << ");\n    return success(!!value);\n  }\n";
    }
  }
  os << "  return {};\n";
  os << "}\n\n";

  // The printer dispatch uses llvm::TypeSwitch to find and call the correct
  // printer.
  os << llvm::formatv(defPrinterDispatchStartStr, valueType);
  for (const AttrOrTypeDef &def : defs) {
    Optional<StringRef> mnemonic = def.getMnemonic();
    if (!mnemonic)
      continue;

    StringRef cppNamespace = def.getDialect().getCppNamespace();
    StringRef cppClassName = def.getCppClassName();
    os << formatv("    .Case<{0}::{1}>([&]({0}::{1} t) {{\n      ",
                  cppNamespace, cppClassName);

    // If the def has no parameters and no printer, just print the mnemonic.
    if (def.getNumParameters() == 0 && !def.getPrinterCode()) {
      os << formatv("printer << {0}::{1}::getMnemonic();", cppNamespace,
                    cppClassName);
    } else {
      os << "t.print(printer);";
    }
    os << "\n      return ::mlir::success();\n    })\n";
  }
  os << llvm::formatv(
      "    .Default([](::mlir::{0}) {{ return ::mlir::failure(); });\n}\n\n",
      valueType);
}

bool DefGenerator::emitDefs(StringRef selectedDialect) {
  emitSourceFileHeader((defTypePrefix + "Def Definitions").str(), os);

  SmallVector<AttrOrTypeDef, 16> defs;
  collectAllDefs(selectedDialect, defRecords, defs);
  if (defs.empty())
    return false;
  emitTypeDefList(defs);

  IfDefScope scope("GET_" + defTypePrefix.upper() + "DEF_CLASSES", os);
  emitParsePrintDispatch(defs);
  for (const AttrOrTypeDef &def : defs)
    emitDefDef(def);

  return false;
}

//===----------------------------------------------------------------------===//
// GEN: Registration hooks
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AttrDef

static llvm::cl::OptionCategory attrdefGenCat("Options for -gen-attrdef-*");
static llvm::cl::opt<std::string>
    attrDialect("attrdefs-dialect",
                llvm::cl::desc("Generate attributes for this dialect"),
                llvm::cl::cat(attrdefGenCat), llvm::cl::CommaSeparated);

static mlir::GenRegistration
    genAttrDefs("gen-attrdef-defs", "Generate AttrDef definitions",
                [](const llvm::RecordKeeper &records, raw_ostream &os) {
                  AttrDefGenerator generator(records, os);
                  return generator.emitDefs(attrDialect);
                });
static mlir::GenRegistration
    genAttrDecls("gen-attrdef-decls", "Generate AttrDef declarations",
                 [](const llvm::RecordKeeper &records, raw_ostream &os) {
                   AttrDefGenerator generator(records, os);
                   return generator.emitDecls(attrDialect);
                 });

//===----------------------------------------------------------------------===//
// TypeDef

static llvm::cl::OptionCategory typedefGenCat("Options for -gen-typedef-*");
static llvm::cl::opt<std::string>
    typeDialect("typedefs-dialect",
                llvm::cl::desc("Generate types for this dialect"),
                llvm::cl::cat(typedefGenCat), llvm::cl::CommaSeparated);

static mlir::GenRegistration
    genTypeDefs("gen-typedef-defs", "Generate TypeDef definitions",
                [](const llvm::RecordKeeper &records, raw_ostream &os) {
                  TypeDefGenerator generator(records, os);
                  return generator.emitDefs(typeDialect);
                });
static mlir::GenRegistration
    genTypeDecls("gen-typedef-decls", "Generate TypeDef declarations",
                 [](const llvm::RecordKeeper &records, raw_ostream &os) {
                   TypeDefGenerator generator(records, os);
                   return generator.emitDecls(typeDialect);
                 });
