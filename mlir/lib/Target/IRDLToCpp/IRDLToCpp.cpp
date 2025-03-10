//===- IRDLToCpp.cpp - Converts IRDL definitions to C++ -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/IRDLToCpp/IRDLToCpp.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "TemplatingUtils.h"

using namespace mlir;

constexpr char headerTemplateText[] =
#include "Templates/Header.txt"
    ;

constexpr char declarationMacroFlag[] = "GEN_DIALECT_DECL_HEADER";
constexpr char definitionMacroFlag[] = "GEN_DIALECT_DEF";

namespace {

struct DialectStrings {
  StringRef dialectName;
  StringRef dialectCppName;
  StringRef dialectCppShortName;
  StringRef dialectBaseTypeName;

  StringRef namespaceOpen;
  StringRef namespaceClose;
  StringRef namespacePath;
};

struct TypeStrings {
  StringRef typeName;
  std::string typeCppName;
};

struct OpStrings {
  StringRef opName;
  std::string opCppName;
  llvm::SmallVector<std::string> opResultNames;
  llvm::SmallVector<std::string> opOperandNames;
};

static std::string joinNameList(llvm::ArrayRef<std::string> names) {
  std::string nameArray;
  llvm::raw_string_ostream nameArrayStream(nameArray);
  nameArrayStream << "{\"" << llvm::join(names, "\", \"") << "\"}";

  return nameArray;
}

static std::string snakeToCamel(llvm::StringRef in) {
  std::string output{};
  output.reserve(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    if (in[i] == '_' && i + 1 < in.size()) {
      output += toupper(in[i + 1]);
      ++i;
    } else {
      output += in[i];
    }
  }
  return output;
}

static std::string snakeToPascal(llvm::StringRef in) {
  std::string output{static_cast<std::string::value_type>(toupper(in.front()))};
  output.reserve(in.size());
  for (size_t i = 1; i < in.size(); ++i) {
    if (in[i] == '_' && i + 1 < in.size()) {
      output += toupper(in[i + 1]);
      ++i;
    } else {
      output += in[i];
    }
  }
  return output;
}

static std::string typeToCppName(irdl::TypeOp type) {
  return llvm::formatv("{0}Type", snakeToPascal(type.getSymName()));
}

static std::string opToCppName(irdl::OperationOp op) {
  return llvm::formatv("{0}Op", snakeToPascal(op.getSymName()));
}

static TypeStrings getStrings(irdl::TypeOp type) {
  TypeStrings strings;
  strings.typeName = type.getSymName();
  strings.typeCppName = typeToCppName(type);
  return strings;
}

static OpStrings getStrings(irdl::OperationOp op) {
  auto operands = op.getOps<irdl::OperandsOp>();
  auto operandOp =
      operands.empty() ? std::optional<irdl::OperandsOp>{} : *operands.begin();

  auto results = op.getOps<irdl::ResultsOp>();
  auto resultOp =
      results.empty() ? std::optional<irdl::ResultsOp>{} : *results.begin();

  OpStrings strings;
  strings.opName = op.getSymName();
  strings.opCppName = opToCppName(op);

  if (operandOp) {
    strings.opOperandNames = llvm::SmallVector<std::string>(
        llvm::map_range(operandOp->getNames(), [](Attribute attr) {
          return llvm::formatv("{0}", cast<StringAttr>(attr));
        }));
  }

  if (resultOp) {
    strings.opResultNames = llvm::SmallVector<std::string>(
        llvm::map_range(resultOp->getNames(), [](Attribute attr) {
          return llvm::formatv("{0}", cast<StringAttr>(attr));
        }));
  }

  return strings;
}

static void fillDict(irdl::detail::dictionary &dict,
                     const TypeStrings &strings) {
  dict["TYPE_NAME"] = strings.typeName;
  dict["TYPE_CPP_NAME"] = strings.typeCppName;
}

static void fillDict(irdl::detail::dictionary &dict, const OpStrings &strings) {
  const auto operandCount = strings.opOperandNames.size();
  const auto resultCount = strings.opResultNames.size();

  dict["OP_NAME"] = strings.opName;
  dict["OP_CPP_NAME"] = strings.opCppName;
  dict["OP_OPERAND_COUNT"] = std::to_string(strings.opOperandNames.size());
  dict["OP_RESULT_COUNT"] = std::to_string(strings.opResultNames.size());
  dict["OP_OPERAND_INITIALIZER_LIST"] =
      operandCount ? joinNameList(strings.opOperandNames) : "{\"\"}";
  dict["OP_RESULT_INITIALIZER_LIST"] =
      resultCount ? joinNameList(strings.opResultNames) : "{\"\"}";
}

static void fillDict(irdl::detail::dictionary &dict,
                     const DialectStrings &strings) {
  dict["DIALECT_NAME"] = strings.dialectName;
  dict["DIALECT_BASE_TYPE_NAME"] = strings.dialectBaseTypeName;
  dict["DIALECT_CPP_NAME"] = strings.dialectCppName;
  dict["DIALECT_CPP_SHORT_NAME"] = strings.dialectCppShortName;
  dict["NAMESPACE_OPEN"] = strings.namespaceOpen;
  dict["NAMESPACE_CLOSE"] = strings.namespaceClose;
  dict["NAMESPACE_PATH"] = strings.namespacePath;
}

static LogicalResult
generateTypedefList(irdl::DialectOp &dialect,
                    llvm::SmallVector<std::string> &typeNames) {
  auto typeOps = dialect.getOps<irdl::TypeOp>();
  auto range = llvm::map_range(typeOps, typeToCppName);
  typeNames = llvm::SmallVector<std::string>(range);
  return success();
}

static LogicalResult generateOpList(irdl::DialectOp &dialect,
                                    llvm::SmallVector<std::string> &opNames) {
  auto operationOps = dialect.getOps<irdl::OperationOp>();
  auto range = llvm::map_range(operationOps, opToCppName);
  opNames = llvm::SmallVector<std::string>(range);
  return success();
}

} // namespace

static LogicalResult generateTypeInclude(irdl::TypeOp type, raw_ostream &output,
                                         irdl::detail::dictionary &dict) {
  static const auto typeDeclTemplate = irdl::detail::Template(
#include "Templates/TypeDecl.txt"
  );

  fillDict(dict, getStrings(type));
  typeDeclTemplate.render(output, dict);

  return success();
}

static LogicalResult generateOperationInclude(irdl::OperationOp op,
                                              raw_ostream &output,
                                              irdl::detail::dictionary &dict) {
  static const auto perOpDeclTemplate = irdl::detail::Template(
#include "Templates/PerOperationDecl.txt"
  );
  const auto opStrings = getStrings(op);
  fillDict(dict, opStrings);

  auto getters = std::string{};

  for (size_t i = 0; i < opStrings.opOperandNames.size(); ++i) {
    const auto &op = snakeToPascal(opStrings.opOperandNames[i]);
    getters += llvm::formatv(
        "::mlir::Value get{0}() { return getODSOperands({1}).front(); }\n  ",
        op, i);
  }

  dict["OP_GETTER_DECLS"] = getters;
  std::string buildDecls;
  llvm::raw_string_ostream stream{buildDecls};

  auto resultParams =
      llvm::join(llvm::map_range(opStrings.opResultNames,
                                 [](StringRef name) -> std::string {
                                   return llvm::formatv("::mlir::Type {0}, ",
                                                        snakeToCamel(name));
                                 }),
                 "");

  auto operandParams =
      llvm::join(llvm::map_range(opStrings.opOperandNames,
                                 [](StringRef name) -> std::string {
                                   return llvm::formatv("::mlir::Value {0}, ",
                                                        snakeToCamel(name));
                                 }),
                 "");

  stream << llvm::formatv(
      R"(static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, {0} {1} ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {{});)",
      resultParams, operandParams);
  dict["OP_BUILD_DECLS"] = buildDecls;

  perOpDeclTemplate.render(output, dict);
  return success();
}

static LogicalResult generateInclude(irdl::DialectOp dialect,
                                     raw_ostream &output,
                                     DialectStrings &dialectStrings) {
  static const auto dialectDeclTemplate = irdl::detail::Template(
#include "Templates/DialectDecl.txt"
  );
  static const auto typeHeaderDeclTemplate = irdl::detail::Template(
#include "Templates/TypeHeaderDecl.txt"
  );
  output << "#ifdef " << declarationMacroFlag << "\n#undef "
         << declarationMacroFlag << "\n";

  irdl::detail::dictionary dict;
  fillDict(dict, dialectStrings);

  dialectDeclTemplate.render(output, dict);
  typeHeaderDeclTemplate.render(output, dict);

  auto typeOps = dialect.getOps<irdl::TypeOp>();
  auto operationOps = dialect.getOps<irdl::OperationOp>();

  for (auto &&typeOp : typeOps) {
    if (failed(generateTypeInclude(typeOp, output, dict)))
      return failure();
  }

  llvm::SmallVector<std::string> opNames;
  if (failed(generateOpList(dialect, opNames)))
    return failure();

  auto classDeclarations =
      llvm::join(llvm::map_range(opNames,
                                 [](llvm::StringRef name) -> std::string {
                                   return llvm::formatv("class {0};", name);
                                 }),
                 "\n");
  const auto forwardDeclarations = llvm::formatv(
      "{1}\n{0}\n{2}", std::move(classDeclarations),
      dialectStrings.namespaceOpen, dialectStrings.namespaceClose);

  output << forwardDeclarations;
  for (auto &&operationOp : operationOps) {
    if (failed(generateOperationInclude(operationOp, output, dict)))
      return failure();
  }

  output << "#endif // " << declarationMacroFlag << "\n";

  return success();
}

static LogicalResult generateLib(irdl::DialectOp dialect, raw_ostream &output,
                                 DialectStrings &dialectStrings) {

  static const auto typeHeaderDefTemplate = mlir::irdl::detail::Template{
#include "Templates/TypeHeaderDef.txt"
  };
  static const auto typeDefTemplate = mlir::irdl::detail::Template{
#include "Templates/TypeDef.txt"
  };
  static const auto dialectDefTemplate = mlir::irdl::detail::Template{
#include "Templates/DialectDef.txt"
  };
  static const auto perOpDefTemplate = mlir::irdl::detail::Template{
#include "Templates/PerOperationDef.txt"
  };

  irdl::detail::dictionary dict;
  fillDict(dict, dialectStrings);

  output << "#ifdef " << definitionMacroFlag << "\n#undef "
         << definitionMacroFlag << "\n";

  typeHeaderDefTemplate.render(output, dict);

  // get typedef list
  llvm::SmallVector<std::string> typeNames;
  if (failed(generateTypedefList(dialect, typeNames)))
    return failure();

  dict["TYPE_LIST"] = llvm::join(
      llvm::map_range(typeNames,
                      [&dialectStrings](llvm::StringRef name) -> std::string {
                        return llvm::formatv(
                            "{0}::{1}", dialectStrings.namespacePath, name);
                      }),
      ",\n");

  auto typeCase = llvm::join(
      llvm::map_range(
          typeNames,
          [&](llvm::StringRef name) -> std::string {
            return llvm::formatv(
                R"(.Case({1}::{0}::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = {1}::{0}::get(parser.getContext());
      return ::mlir::success(!!value);
      }))",
                name, dialectStrings.namespacePath);
          }),
      "\n");

  dict["TYPE_PARSER"] = llvm::formatv(
      R"(static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    {0}    
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {{
      *mnemonic = keyword;
      return std::nullopt;
    });
})",
      std::move(typeCase));

  auto typePrintCase =
      llvm::join(llvm::map_range(typeNames,
                                 [&](llvm::StringRef name) -> std::string {
                                   return llvm::formatv(
                                       R"(.Case<{1}::{0}>([&](auto t) {
      printer << {1}::{0}::getMnemonic();
      return ::mlir::success();
    }))",
                                       name, dialectStrings.namespacePath);
                                 }),
                 "\n");
  dict["TYPE_PRINTER"] = llvm::formatv(
      R"(static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)
    {0}
    .Default([](auto) {{ return ::mlir::failure(); });
})",
      std::move(typePrintCase));

  dict["TYPE_DEFINES"] =
      llvm::join(llvm::map_range(typeNames,
                                 [&](StringRef name) -> std::string {
                                   return llvm::formatv(
                                       "MLIR_DEFINE_EXPLICIT_TYPE_ID({1}::{0})",
                                       name, dialectStrings.namespacePath);
                                 }),
                 "\n");

  typeDefTemplate.render(output, dict);

  // get op list
  auto operations = dialect.getOps<irdl::OperationOp>();
  llvm::SmallVector<std::string> opNames;
  if (failed(generateOpList(dialect, opNames)))
    return failure();

  const auto commaSeparatedOpList = llvm::join(
      llvm::map_range(opNames,
                      [&dialectStrings](llvm::StringRef name) -> std::string {
                        return llvm::formatv(
                            "{0}::{1}", dialectStrings.namespacePath, name);
                      }),
      ",\n");

  const auto perOpDefinitions = llvm::join(
      llvm::map_range(
          operations,
          [&dict, &perOpDefTemplate =
                      perOpDefTemplate](irdl::OperationOp op) -> std::string {
            auto opStrings = getStrings(op);
            fillDict(dict, opStrings);

            const auto operandCount = opStrings.opOperandNames.size();
            const auto operandNames =
                operandCount ? joinNameList(opStrings.opOperandNames)
                             : "{\"\"}";

            const auto resultNames = joinNameList(opStrings.opResultNames);

            auto resultTypes =
                llvm::join(llvm::map_range(opStrings.opResultNames,
                                           [](StringRef attr) -> std::string {
                                             return llvm::formatv(
                                                 "::mlir::Type {0}, ", attr);
                                           }),
                           "");
            auto operandTypes =
                llvm::join(llvm::map_range(opStrings.opOperandNames,
                                           [](StringRef attr) -> std::string {
                                             return llvm::formatv(
                                                 "::mlir::Value {0}, ", attr);
                                           }),
                           "");
            auto odsOperandAdder = llvm::join(
                llvm::map_range(opStrings.opOperandNames,
                                [](StringRef attr) -> std::string {
                                  return llvm::formatv(
                                      "  odsState.addOperands({0});", attr);
                                }),
                "\n");
            auto odsResultAdder = llvm::join(
                llvm::map_range(opStrings.opResultNames,
                                [](StringRef attr) -> std::string {
                                  return llvm::formatv(
                                      "  odsState.addTypes({0});", attr);
                                }),
                "\n");

            const auto buildDefinition = llvm::formatv(
                R"(
void {0}::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, {1} {2} ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {{
{3}
{4}
}
    )",
                opStrings.opCppName, std::move(resultTypes),
                std::move(operandTypes), std::move(odsOperandAdder),
                std::move(odsResultAdder));

            dict["OP_BUILD_DEFS"] = buildDefinition;

            std::string str;
            llvm::raw_string_ostream stream{str};
            perOpDefTemplate.render(stream, dict);
            return str;
          }),
      "\n");

  dict["OP_LIST"] = commaSeparatedOpList;
  dict["OP_CLASSES"] = perOpDefinitions;
  output << perOpDefinitions;
  dialectDefTemplate.render(output, dict);

  output << "#endif // " << definitionMacroFlag << "\n";
  return success();
}

static LogicalResult verifySupported(irdl::DialectOp dialect) {
  for (auto operation : dialect.getOps<irdl::OperationOp>()) {
    // scan operands of operation
    for (auto operands : operation.getOps<irdl::OperandsOp>()) {
      for (auto operand : operands.getOperands()) {
        if (!llvm::isa<irdl::AnyOp>(operand.getDefiningOp())) {
          return operands.emitError(
              "IRDL C++ translation only supports irdl.any "
              "constraint for types");
        }
      }
    }

    // scan results of operation
    for (auto results : operation.getOps<irdl::ResultsOp>()) {
      for (auto operand : results.getOperands()) {
        if (!llvm::isa<irdl::AnyOp>(operand.getDefiningOp())) {
          return results.emitError(
              "IRDL C++ translation only supports irdl.any "
              "constraint for types");
        }
      }
    }
  }

  return success();
}

LogicalResult irdl::translateIRDLDialectToCpp(irdl::DialectOp dialect,
                                              raw_ostream &output) {
  static const auto typeDefTempl = detail::Template(
#include "Templates/TypeDef.txt"
  );

  StringRef dialectName = dialect.getSymName();

  // TODO: deal with no more constraints than the verifier allows.
  if (dialectName.size() < 1)
    return dialect->emitError("dialect name must be more than one character");
  if (!llvm::isAlpha(dialectName[0]))
    return dialect->emitError("dialect name must start with a letter");
  if (!llvm::all_of(dialectName,
                    [](char c) { return llvm::isAlnum(c) || c == '_'; }))
    return dialect->emitError(
        "dialect name must only contain letters, numbers or underscores");

  if (failed(verifySupported(dialect)))
    return failure();

  llvm::SmallVector<llvm::SmallString<8>> namespaceAbsolutePath{{"mlir"},
                                                                dialectName};
  std::string namespaceOpen;
  std::string namespaceClose;
  std::string namespacePath;
  llvm::raw_string_ostream namespaceOpenStream(namespaceOpen);
  llvm::raw_string_ostream namespaceCloseStream(namespaceClose);
  llvm::raw_string_ostream namespacePathStream(namespacePath);
  for (auto &pathElement : namespaceAbsolutePath) {
    namespaceOpenStream << "namespace " << pathElement << " {\n";
    namespaceCloseStream << "} // namespace " << pathElement << "\n";
    namespacePathStream << "::" << pathElement;
  }

  std::string cppShortName = snakeToPascal(dialectName);
  std::string dialectBaseTypeName = llvm::formatv("{0}Type", cppShortName);
  std::string cppName = llvm::formatv("{0}Dialect", cppShortName);

  DialectStrings dialectStrings;
  dialectStrings.dialectName = dialectName;
  dialectStrings.dialectBaseTypeName = dialectBaseTypeName;
  dialectStrings.dialectCppName = cppName;
  dialectStrings.dialectCppShortName = cppShortName;
  dialectStrings.namespaceOpen = namespaceOpen;
  dialectStrings.namespaceClose = namespaceClose;
  dialectStrings.namespacePath = namespacePath;

  output << headerTemplateText;

  if (failed(generateInclude(dialect, output, dialectStrings)))
    return failure();

  if (failed(generateLib(dialect, output, dialectStrings)))
    return failure();

  return success();
}
