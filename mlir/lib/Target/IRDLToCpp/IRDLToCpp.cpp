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

constexpr char opDeclTemplateText[] =
#include "Templates/OperationDecl.txt"
    ;

namespace {

struct UsefulStrings {
  StringRef dialectName;
  StringRef dialectCppName;
  StringRef dialectCppShortName;
  StringRef dialectBaseTypeName;

  StringRef namespaceOpen;
  StringRef namespaceClose;
  StringRef namespacePath;
};

static std::string capitalize(StringRef str) {
  return llvm::formatv("{0}{1}", llvm::toUpper(str[0]),
                       str.slice(1, str.size()));
}

} // namespace

static LogicalResult generateTypeInclude(irdl::TypeOp type, raw_ostream &output,
                                         UsefulStrings &usefulStrings) {

  StringRef typeName = type.getSymName();
  std::string typeCppShortName = capitalize(typeName);
  std::string typeCppName = llvm::formatv("{0}Type", typeCppShortName);

  output << llvm::formatv(
      typeDeclTemplateText, typeName, typeCppName, usefulStrings.dialectName,
      usefulStrings.dialectBaseTypeName, usefulStrings.namespaceOpen,
      usefulStrings.namespaceClose, usefulStrings.namespacePath);

  return success();
}

static LogicalResult generateOperationInclude(irdl::OperationOp op,
                                              raw_ostream &output,
                                              UsefulStrings &usefulStrings) {

  StringRef opName = op.getSymName();
  std::string opCppShortName = capitalize(opName);
  std::string opCppName = llvm::formatv("{0}Op", opCppShortName);

  auto &&block = op.getBody().getBlocks().front();

  auto operandOp = ([&block]() -> std::optional<irdl::OperandsOp> {
    auto operands = block.getOps<irdl::OperandsOp>();
    if (operands.empty())
      return {};
    return *operands.begin();
  })();

  auto resultsOp = *block.getOps<irdl::ResultsOp>().begin();

  const auto operandCount = operandOp ? operandOp->getNumOperands() : 0;

  const auto operandNames = ([&operandOp]() -> std::string {
    if (!operandOp)
      return "{}";

    auto names = llvm::map_range(operandOp->getNames(), [](auto &attr) {
      return mlir::cast<StringAttr, Attribute>(attr);
    });

    std::string nameArray;
    llvm::raw_string_ostream nameArrayStream(nameArray);
    nameArrayStream << "{\"" << llvm::join(names, "\", \"") << "\"}";

    return nameArray;
  })();

  const unsigned resultCount = resultsOp.getNumOperands();

  output << llvm::formatv(
      opDeclTemplateText, opName, opCppName, usefulStrings.dialectName,
      operandCount, operandNames, resultCount, usefulStrings.namespaceOpen,
      usefulStrings.namespaceClose, usefulStrings.namespacePath);

  return success();
}

static LogicalResult generateInclude(irdl::DialectOp dialect,
                                     raw_ostream &output,
                                     UsefulStrings &usefulStrings) {
  output << "#ifdef " << declarationMacroFlag << "\n#undef "
         << declarationMacroFlag << "\n";

  output << llvm::formatv(
      dialectDeclTemplateText, usefulStrings.namespaceOpen,
      usefulStrings.namespaceClose, usefulStrings.dialectCppName,
      usefulStrings.namespacePath, usefulStrings.dialectName);

  output << llvm::formatv(
      typeHeaderDeclTemplateText, usefulStrings.dialectBaseTypeName,
      usefulStrings.namespaceOpen, usefulStrings.namespaceClose,
      usefulStrings.namespacePath);

  auto &&region = dialect->getRegion(0);
  auto &&block = region.getBlocks().front();
  for (auto &&op : block) {
    if (auto typeop = llvm::dyn_cast_or_null<irdl::TypeOp>(op)) {
      if (failed(generateTypeInclude(typeop, output, usefulStrings)))
        return failure();
    }
    if (auto operationOp = llvm::dyn_cast_or_null<irdl::OperationOp>(op)) {
      if (failed(generateOperationInclude(operationOp, output, usefulStrings)))
        return failure();
    }
  }

  output << "#endif // " << declarationMacroFlag << "\n";

  return success();
}

static LogicalResult generateLib(irdl::DialectOp dialect, raw_ostream &output,
                                 UsefulStrings &usefulStrings) {
  output << "#ifdef " << definitionMacroFlag << "\n#undef "
         << definitionMacroFlag << "\n";

  output << llvm::formatv(dialectDefTemplateText, usefulStrings.namespaceOpen,
                          usefulStrings.namespaceClose,
                          usefulStrings.dialectCppName,
                          usefulStrings.namespacePath);

  output << usefulStrings.namespaceOpen;
  output << llvm::formatv(typeHeaderDefTemplateText,
                          usefulStrings.dialectBaseTypeName,
                          usefulStrings.dialectCppName);
  output << usefulStrings.namespaceClose;

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

  UsefulStrings usefulStrings;
  usefulStrings.dialectName = dialectName;
  usefulStrings.dialectBaseTypeName = dialectBaseTypeName;
  usefulStrings.dialectCppName = cppName;
  usefulStrings.dialectCppShortName = cppShortName;
  usefulStrings.namespaceOpen = namespaceOpen;
  usefulStrings.namespaceClose = namespaceClose;
  usefulStrings.namespacePath = namespacePath;

  output << headerTemplateText;

  if (failed(generateInclude(dialect, output, usefulStrings)))
    return failure();

  if (failed(generateLib(dialect, output, usefulStrings)))
    return failure();

  return success();
}
