//===- IRDLToCpp.cpp - Converts IRDL definitions to C++ -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/IRDLToCpp/IRDLToCpp.h"
#include "mlir/Support/LLVM.h"
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
// 3: Dialect name
// 4: Dialect namespace
constexpr char dialectDeclTemplateText[] =
#include "Templates/DialectDecl.txt"
    ;

constexpr char declarationMacroFlag[] = "GEN_DIALECT_DECL_HEADER";
constexpr char definitionMacroFlag[] = "GEN_DIALECT_DEF";

LogicalResult irdl::translateIRDLDialectToCpp(irdl::DialectOp dialect,
                                              raw_ostream &output) {
  StringRef dialectName = dialect.getSymName();

  // TODO: allow more complex path.
  llvm::SmallVector<llvm::SmallString<8>> namespaceAbsolutePath{{"mlir"},
                                                                dialectName};

  std::string namespaceOpen;
  std::string namespaceClose;
  std::string namespacePathString;
  llvm::raw_string_ostream namespaceOpenStream(namespaceOpen);
  llvm::raw_string_ostream namespaceCloseStream(namespaceClose);
  llvm::raw_string_ostream namespacePathStream(namespacePathString);
  for (auto &pathElement : namespaceAbsolutePath) {
    namespaceOpenStream << "namespace " << pathElement << " {\n";
    namespaceCloseStream << "} // namespace " << pathElement << "\n";
    namespacePathStream << "::" << pathElement;
  }

  // TODO: allow control over C++ name.
  std::string cppName;
  llvm::raw_string_ostream cppNameStream(cppName);
  cppNameStream << llvm::toUpper(dialectName[0])
                << dialectName.slice(1, dialectName.size()) << "Dialect";

  output << headerTemplateText;

  output << "#ifdef " << declarationMacroFlag << "\n#undef "
         << declarationMacroFlag << "\n";

  output << llvm::formatv(dialectDeclTemplateText, namespaceOpen,
                          namespaceClose, cppName, dialectName,
                          namespacePathString);

  output << "#endif\n";

  return success();
}
