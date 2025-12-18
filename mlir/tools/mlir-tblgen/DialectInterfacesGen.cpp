//===- DialectInterfacesGen.cpp - MLIR dialect interface utility generator ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DialectInterfaceGen generates definitions for Dialect interfaces.
//
//===----------------------------------------------------------------------===//

#include "CppGenUtilities.h"
#include "DocGenUtilities.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Interfaces.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/CodeGenHelpers.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace mlir;
using llvm::Record;
using llvm::RecordKeeper;
using mlir::tblgen::DialectInterface;
using mlir::tblgen::InterfaceMethod;

/// Emit a string corresponding to a C++ type, followed by a space if necessary.
static raw_ostream &emitCPPType(StringRef type, raw_ostream &os) {
  type = type.trim();
  os << type;
  if (type.back() != '&' && type.back() != '*')
    os << " ";
  return os;
}

/// Emit the method name and argument list for the given method.
static void emitMethodNameAndArgs(const InterfaceMethod &method, StringRef name,
                                  raw_ostream &os) {
  os << name << '(';
  llvm::interleaveComma(method.getArguments(), os,
                        [&](const InterfaceMethod::Argument &arg) {
                          os << arg.type << " " << arg.name;
                        });
  os << ") const";
}

/// Get an array of all Dialect Interface definitions
static std::vector<const Record *>
getAllInterfaceDefinitions(const RecordKeeper &records) {
  std::vector<const Record *> defs =
      records.getAllDerivedDefinitions("DialectInterface");

  llvm::erase_if(defs, [&](const Record *def) {
    // Ignore interfaces defined outside of the top-level file.
    return llvm::SrcMgr.FindBufferContainingLoc(def->getLoc()[0]) !=
           llvm::SrcMgr.getMainFileID();
  });
  return defs;
}

namespace {
/// This struct is the generator used when processing tablegen dialect
/// interfaces.
class DialectInterfaceGenerator {
public:
  DialectInterfaceGenerator(const RecordKeeper &records, raw_ostream &os)
      : defs(getAllInterfaceDefinitions(records)), os(os) {}

  bool emitInterfaceDecls();

protected:
  void emitInterfaceDecl(const DialectInterface &interface);

  /// The set of interface records to emit.
  std::vector<const Record *> defs;
  // The stream to emit to.
  raw_ostream &os;
};
} // namespace

//===----------------------------------------------------------------------===//
// GEN: Interface declarations
//===----------------------------------------------------------------------===//

static void emitInterfaceMethodDoc(const InterfaceMethod &method,
                                   raw_ostream &os, StringRef prefix = "") {
  if (std::optional<StringRef> description = method.getDescription())
    tblgen::emitDescriptionComment(*description, os, prefix);
  else
    os << "\n";
}

static void emitInterfaceMethodsDef(const DialectInterface &interface,
                                    raw_ostream &os) {

  raw_indented_ostream ios(os);
  ios.indent(2);

  for (auto &method : interface.getMethods()) {
    emitInterfaceMethodDoc(method, ios);
    ios << "virtual ";
    emitCPPType(method.getReturnType(), ios);
    emitMethodNameAndArgs(method, method.getName(), ios);

    if (method.isDeclaration()) {
      ios << ";\n";
      continue;
    }

    // Otherwise it's a normal interface method
    ios << " {";

    if (auto body = method.getBody()) {
      ios << "\n";
      ios.indent(4);
      ios << body << "\n";
      ios.indent(2);
    }
    os << "}\n";
  }
}

static void emitInterfaceAliasDeclarations(const DialectInterface &interface,
                                           raw_ostream &os) {
  raw_indented_ostream ios(os);
  ios.indent(2);

  for (auto &alias : interface.getAliasDeclarations()) {
    ios << "using " << alias.getKey() << " = " << alias.getValue() << ";\n";
  }
}

void DialectInterfaceGenerator::emitInterfaceDecl(
    const DialectInterface &interface) {
  llvm::NamespaceEmitter ns(os, interface.getCppNamespace());

  tblgen::emitSummaryAndDescComments(os, "",
                                     interface.getDescription().value_or(""));

  // Emit the main interface class declaration.
  os << llvm::formatv(
      "class {0} : public ::mlir::DialectInterface::Base<{0}> {\n"
      "public:\n"
      "  {0}(::mlir::Dialect *dialect) : Base(dialect) {{}\n",
      interface.getName());

  emitInterfaceAliasDeclarations(interface, os);

  emitInterfaceMethodsDef(interface, os);

  os << "};\n";
}

bool DialectInterfaceGenerator::emitInterfaceDecls() {

  llvm::emitSourceFileHeader("Dialect Interface Declarations", os);

  // Sort according to ID, so defs are emitted in the order in which they appear
  // in the Tablegen file.
  std::vector<const Record *> sortedDefs(defs);
  llvm::sort(sortedDefs, [](const Record *lhs, const Record *rhs) {
    return lhs->getID() < rhs->getID();
  });

  for (const Record *def : sortedDefs)
    emitInterfaceDecl(DialectInterface(def));

  return false;
}

//===----------------------------------------------------------------------===//
// GEN: Interface registration hooks
//===----------------------------------------------------------------------===//

static mlir::GenRegistration genDecls(
    "gen-dialect-interface-decls", "Generate dialect interface declarations.",
    [](const RecordKeeper &records, raw_ostream &os) {
      return DialectInterfaceGenerator(records, os).emitInterfaceDecls();
    });
