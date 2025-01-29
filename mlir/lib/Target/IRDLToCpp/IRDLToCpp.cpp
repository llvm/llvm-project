//===- IRDLToCpp.cpp - Converts IRDL definitions to C++ -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/IRDLToCpp/IRDLToCpp.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

constexpr char headerTemplateText[] =
#include "Templates/Header.txt"
    ;

// 0: Namespace open
// 1: Namespace close
// 2: Dialect C++ name
// 3: Dialect namespace
// 4: Dialect name
constexpr char dialectDeclTemplateText[] =
#include "Templates/DialectDecl.txt"
    ;

// 0: Namespace open
// 1: Namespace close
// 2: Dialect C++ name
// 3: Dialect namespace
constexpr char dialectDefTemplateText[] =
#include "Templates/DialectDef.txt"
    ;

constexpr char declarationMacroFlag[] = "GEN_DIALECT_DECL_HEADER";
constexpr char definitionMacroFlag[] = "GEN_DIALECT_DEF";

constexpr char typeHeaderDeclTemplateText[] =
#include "Templates/TypeHeaderDecl.txt"
    ;

constexpr char typeHeaderDefTemplateText[] =
#include "Templates/TypeHeaderDef.txt"
    ;

constexpr char typeDeclTemplateText[] =
#include "Templates/TypeDecl.txt"
    ;

constexpr char typeDefTemplateText[] =
#include "Templates/TypeDef.txt"
    ;
constexpr char perOpDeclTemplateText[] =
#include "Templates/PerOperationDecl.txt"
    ;

constexpr char perOpDefTemplateText[] =
#include "Templates/PerOperationDef.txt"
    ;

constexpr char opDefTemplateText[] =
#include "Templates/OperationDef.txt"
    ;

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

static std::string capitalize(StringRef str) {
  return llvm::formatv("{0}{1}", llvm::toUpper(str[0]),
                       str.slice(1, str.size()));
}

static TypeStrings getStrings(irdl::TypeOp type) {
  TypeStrings strings;
  strings.typeName = type.getSymName();
  strings.typeCppName = llvm::formatv("{0}Type", capitalize(strings.typeName));
  return strings;
}

static OpStrings getStrings(irdl::OperationOp op) {

  auto &block = op.getBody().getBlocks().front();

  auto operands = block.getOps<irdl::OperandsOp>();
  auto operandOp =
      operands.empty() ? std::optional<irdl::OperandsOp>{} : *operands.begin();

  auto resultsOp = *block.getOps<irdl::ResultsOp>().begin();

  constexpr auto getNames = [](auto op) {
    auto names = llvm::map_range(
        op.getNames(), [](auto &attr) { return mlir::cast<StringAttr>(attr); });
    return names;
  };

  const auto operandCount = operandOp ? operandOp->getNumOperands() : 0;

  const auto resultCount = resultsOp.getNumOperands();
  const auto resultAttrArray = getNames(resultsOp);

  OpStrings strings;
  strings.opName = op.getSymName();
  strings.opCppName = llvm::formatv("{0}Op", capitalize(strings.opName));
  if (operandOp)
    strings.opOperandNames = llvm::SmallVector<std::string>(
        llvm::map_range(operandOp->getNames(), [](Attribute attr) {
          return llvm::formatv("{0}", cast<StringAttr>(attr));
        }));
  strings.opResultNames = llvm::SmallVector<std::string>(
      llvm::map_range(resultsOp.getNames(), [](Attribute attr) {
        return llvm::formatv("{0}", cast<StringAttr>(attr));
      }));

  return strings;
}

static LogicalResult
generateTypedefList(mlir::Block &dialectBlock,
                    llvm::SmallVector<std::string> &typeNames) {
  auto typeOps = dialectBlock.getOps<irdl::TypeOp>();
  auto range = llvm::map_range(
      typeOps, [](auto &&type) { return getStrings(type).typeCppName; });
  typeNames = llvm::SmallVector<std::string>(range);
  return success();
}

static LogicalResult generateOpList(mlir::Block &dialectBlock,
                                    llvm::SmallVector<std::string> &typeNames) {
  auto typeOps = dialectBlock.getOps<irdl::OperationOp>();
  auto range = llvm::map_range(
      typeOps, [](auto &&type) { return getStrings(type).opCppName; });
  typeNames = llvm::SmallVector<std::string>(range);
  return success();
}

} // namespace

static LogicalResult generateTypeInclude(irdl::TypeOp type, raw_ostream &output,
                                         DialectStrings &dialectStrings) {

  const auto typeStrings = getStrings(type);

  output << llvm::formatv(
      typeDeclTemplateText, typeStrings.typeName, typeStrings.typeCppName,
      dialectStrings.dialectName, dialectStrings.dialectBaseTypeName,
      dialectStrings.namespaceOpen, dialectStrings.namespaceClose,
      dialectStrings.namespacePath);

  return success();
}

static LogicalResult generateOperationInclude(irdl::OperationOp op,
                                              raw_ostream &output,
                                              DialectStrings &dialectStrings) {

  const auto opStrings = getStrings(op);

  constexpr auto stringify = [](auto &&names) -> std::string {
    std::string nameArray;
    llvm::raw_string_ostream nameArrayStream(nameArray);
    nameArrayStream << "{\"" << llvm::join(names, "\", \"") << "\"}";

    return nameArray;
  };

  const auto operandCount = opStrings.opOperandNames.size();
  const auto operandNames =
      operandCount ? stringify(opStrings.opOperandNames) : "{}";

  const auto resultCount = opStrings.opResultNames.size();
  const auto resultNames = stringify(opStrings.opResultNames);

  const auto buildDeclaration = llvm::formatv(
      R"(static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, {0} {1} ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {{});)",
      llvm::join(llvm::map_range(opStrings.opResultNames,
                                 [](StringRef name) -> std::string {
                                   return llvm::formatv("::mlir::Type {0}, ",
                                                        name);
                                 }),
                 ""),
      llvm::join(llvm::map_range(opStrings.opOperandNames,
                                 [](StringRef name) -> std::string {
                                   return llvm::formatv("::mlir::Value {0}, ",
                                                        name);
                                 }),
                 ""));

  output << llvm::formatv(
      perOpDeclTemplateText, opStrings.opName, opStrings.opCppName,
      dialectStrings.dialectName, operandCount, operandNames, resultCount,
      resultNames, buildDeclaration, dialectStrings.namespaceOpen,
      dialectStrings.namespaceClose, dialectStrings.namespacePath);

  return success();
}

static LogicalResult generateInclude(irdl::DialectOp dialect,
                                     raw_ostream &output,
                                     DialectStrings &dialectStrings) {
  output << "#ifdef " << declarationMacroFlag << "\n#undef "
         << declarationMacroFlag << "\n";

  output << llvm::formatv(
      dialectDeclTemplateText, dialectStrings.namespaceOpen,
      dialectStrings.namespaceClose, dialectStrings.dialectCppName,
      dialectStrings.namespacePath, dialectStrings.dialectName);

  output << llvm::formatv(
      typeHeaderDeclTemplateText, dialectStrings.dialectBaseTypeName,
      dialectStrings.namespaceOpen, dialectStrings.namespaceClose,
      dialectStrings.namespacePath);

  auto &dialectBlock = *dialect.getRegion().getBlocks().begin();
  auto typeOps = dialectBlock.getOps<irdl::TypeOp>();
  auto operationOps = dialectBlock.getOps<irdl::OperationOp>();

  for (auto &&typeOp : typeOps) {
    if (failed(generateTypeInclude(typeOp, output, dialectStrings)))
      return failure();
  }

  llvm::SmallVector<std::string> opNames;
  if (failed(generateOpList(dialectBlock, opNames)))
    return failure();
  const auto forwardDeclarations = llvm::formatv(
      R"(
{1}
{0}
{2}
    )",
      llvm::join(llvm::map_range(opNames,
                                 [](llvm::StringRef name) -> std::string {
                                   return llvm::formatv("class {0};", name);
                                 }),
                 "\n"),
      dialectStrings.namespaceOpen, dialectStrings.namespaceClose);

  output << forwardDeclarations;
  for (auto &&operationOp : operationOps) {
    if (failed(generateOperationInclude(operationOp, output, dialectStrings)))
      return failure();
  }

  output << "#endif // " << declarationMacroFlag << "\n";

  return success();
}

static LogicalResult generateLib(irdl::DialectOp dialect, raw_ostream &output,
                                 DialectStrings &dialectStrings) {
  output << "#ifdef " << definitionMacroFlag << "\n#undef "
         << definitionMacroFlag << "\n";

  // type header
  output << llvm::formatv(
      typeHeaderDefTemplateText, dialectStrings.dialectBaseTypeName,
      dialectStrings.dialectCppName, dialectStrings.namespaceOpen,
      dialectStrings.namespaceClose);

  // get typedef list
  auto &dialectBlock = *dialect.getRegion().getBlocks().begin();
  llvm::SmallVector<std::string> typeNames;
  if (failed(generateTypedefList(dialectBlock, typeNames)))
    return failure();

  const auto commaSeparatedTypeList = llvm::join(
      llvm::map_range(typeNames,
                      [&dialectStrings](llvm::StringRef name) -> std::string {
                        return llvm::formatv(
                            "{0}::{1}", dialectStrings.namespacePath, name);
                      }),
      ",\n");

  const auto generatedTypeParser = llvm::formatv(
      R"(static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    {0}    
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {{
      *mnemonic = keyword;
      return std::nullopt;
    });
})",
      llvm::join(
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
          "\n"));

  const auto generatedTypePrinter = llvm::formatv(
      R"(static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)
    {0}
    .Default([](auto) {{ return ::mlir::failure(); });
})",
      llvm::join(llvm::map_range(typeNames,
                                 [&](llvm::StringRef name) -> std::string {
                                   return llvm::formatv(
                                       R"(.Case<{1}::{0}>([&](auto t) {
      printer << {1}::{0}::getMnemonic();
      return ::mlir::success();
    }))",
                                       name, dialectStrings.namespacePath);
                                 }),
                 "\n"));

  const auto typeIdDefinitions = llvm::join(llvm::map_range(typeNames, [&](StringRef name) -> std::string {
    return llvm::formatv("MLIR_DEFINE_EXPLICIT_TYPE_ID({1}::{0})", name, dialectStrings.namespacePath);
  }), "\n");

  output << llvm::formatv(
      typeDefTemplateText, commaSeparatedTypeList, generatedTypeParser,
      generatedTypePrinter, dialectStrings.dialectCppName, typeIdDefinitions,
      dialectStrings.namespaceOpen, dialectStrings.namespaceClose);

  // get op list
  auto operations = dialectBlock.getOps<irdl::OperationOp>();
  llvm::SmallVector<std::string> opNames;
  if (failed(generateOpList(dialectBlock, opNames)))
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
          [&dialectStrings](irdl::OperationOp op) -> std::string {
            auto opStrings = getStrings(op);

            constexpr auto stringify = [](auto &&names) -> std::string {
              std::string nameArray;
              llvm::raw_string_ostream nameArrayStream(nameArray);
              nameArrayStream << "{\"" << llvm::join(names, "\", \"") << "\"}";

              return nameArray;
            };

            const auto operandCount = opStrings.opOperandNames.size();
            const auto operandNames =
                operandCount ? stringify(opStrings.opOperandNames) : "{}";

            const auto resultCount = opStrings.opResultNames.size();
            const auto resultNames = stringify(opStrings.opResultNames);

            const auto buildDefinition = llvm::formatv(
                R"(
static void {0}::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, {1} {2} ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {{
{3}
{4}
}
    )",
                opStrings.opCppName,
                llvm::join(llvm::map_range(opStrings.opResultNames,
                                           [](StringRef attr) -> std::string {
                                             return llvm::formatv(
                                                 "::mlir::Type {0}, ", attr);
                                           }),
                           ""),
                llvm::join(llvm::map_range(opStrings.opOperandNames,
                                           [](StringRef attr) -> std::string {
                                             return llvm::formatv(
                                                 "::mlir::Value {0}, ", attr);
                                           }),
                           ""),
                llvm::join(llvm::map_range(
                               opStrings.opOperandNames,
                               [](StringRef attr) -> std::string {
                                 return llvm::formatv(
                                     "  odsBuilder.addOperands({0});", attr);
                               }),
                           "\n"),
                llvm::join(llvm::map_range(opStrings.opResultNames,
                                           [](StringRef attr) -> std::string {
                                             return llvm::formatv(
                                                 "  odsBuilder.addTypes({0});",
                                                 attr);
                                           }),
                           "\n"));
            return llvm::formatv(
                perOpDefTemplateText, opStrings.opName, opStrings.opCppName,
                dialectStrings.dialectName, operandCount, operandNames,
                resultCount, resultNames, buildDefinition,
                dialectStrings.namespaceOpen, dialectStrings.namespaceClose,
                dialectStrings.namespacePath);
          }),
      "\n");

  output << llvm::formatv(opDefTemplateText, commaSeparatedOpList,
                          perOpDefinitions);

  output << llvm::formatv(dialectDefTemplateText, dialectStrings.namespaceOpen,
                          dialectStrings.namespaceClose,
                          dialectStrings.dialectCppName,
                          dialectStrings.namespacePath,
                          commaSeparatedOpList,
                          commaSeparatedTypeList
                          );

  output << "#endif // " << definitionMacroFlag << "\n";
  return success();
}

LogicalResult irdl::translateIRDLDialectToCpp(irdl::DialectOp dialect,
                                              raw_ostream &output) {
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

  // TODO: allow more complex path.
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

  // TODO: allow control over C++ name.
  std::string cppShortName =
      llvm::formatv("{0}{1}", llvm::toUpper(dialectName[0]),
                    dialectName.slice(1, dialectName.size()));
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
