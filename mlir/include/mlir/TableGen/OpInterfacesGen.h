//===- OpInterfacesGen.h - MLIR operation generator helpers -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helpers used in the op interface generators.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRTBLGEN_OPINTERFACESGEN_H_
#define MLIR_TOOLS_MLIRTBLGEN_OPINTERFACESGEN_H_

#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Interfaces.h"

#include "llvm/TableGen/Record.h"

#include <vector>

namespace mlir {
namespace tblgen {

template <typename GeneratorT>
struct InterfaceGenRegistration {
  InterfaceGenRegistration(StringRef genArg, StringRef genDesc)
      : genDeclArg(("gen-" + genArg + "-interface-decls").str()),
        genDefArg(("gen-" + genArg + "-interface-defs").str()),
        genDocArg(("gen-" + genArg + "-interface-docs").str()),
        genDeclDesc(("Generate " + genDesc + " interface declarations").str()),
        genDefDesc(("Generate " + genDesc + " interface definitions").str()),
        genDocDesc(("Generate " + genDesc + " interface documentation").str()),
        genDecls(genDeclArg, genDeclDesc,
                 [](const llvm::RecordKeeper &records, raw_ostream &os) {
                   return GeneratorT(records, os).emitInterfaceDecls();
                 }),
        genDefs(genDefArg, genDefDesc,
                [](const llvm::RecordKeeper &records, raw_ostream &os) {
                  return GeneratorT(records, os).emitInterfaceDefs();
                }),
        genDocs(genDocArg, genDocDesc,
                [](const llvm::RecordKeeper &records, raw_ostream &os) {
                  return GeneratorT(records, os).emitInterfaceDocs();
                }) {}

  std::string genDeclArg, genDefArg, genDocArg;
  std::string genDeclDesc, genDefDesc, genDocDesc;
  mlir::GenRegistration genDecls, genDefs, genDocs;
};

/// This struct is the base generator used when processing tablegen interfaces.
class InterfaceGenerator {
public:
  bool emitInterfaceDefs();
  bool emitInterfaceDecls();
  bool emitInterfaceDocs();

protected:
  InterfaceGenerator(std::vector<llvm::Record *> &&defs, raw_ostream &os)
      : defs(std::move(defs)), os(os) {}

  void emitConceptDecl(const Interface &interface);
  void emitModelDecl(const Interface &interface);
  void emitModelMethodsDef(const Interface &interface);
  void emitTraitDecl(const Interface &interface, StringRef interfaceName,
                     StringRef interfaceTraitsName);
  void emitInterfaceDecl(const Interface &interface);

  /// The set of interface records to emit.
  std::vector<llvm::Record *> defs;
  // The stream to emit to.
  raw_ostream &os;
  /// The C++ value type of the interface, e.g. Operation*.
  StringRef valueType;
  /// The C++ base interface type.
  StringRef interfaceBaseType;
  /// The name of the typename for the value template.
  StringRef valueTemplate;
  /// The name of the substituion variable for the value.
  StringRef substVar;
  /// The format context to use for methods.
  tblgen::FmtContext nonStaticMethodFmt;
  tblgen::FmtContext traitMethodFmt;
  tblgen::FmtContext extraDeclsFmt;
};

std::vector<llvm::Record *>
getAllInterfaceDefinitions(const llvm::RecordKeeper &recordKeeper,
                           StringRef name);

/// A specialized generator for attribute interfaces.
struct AttrInterfaceGenerator : public InterfaceGenerator {
  AttrInterfaceGenerator(const llvm::RecordKeeper &records, raw_ostream &os)
      : InterfaceGenerator(getAllInterfaceDefinitions(records, "Attr"), os) {
    valueType = "::mlir::Attribute";
    interfaceBaseType = "AttributeInterface";
    valueTemplate = "ConcreteAttr";
    substVar = "_attr";
    StringRef castCode = "(::llvm::cast<ConcreteAttr>(tablegen_opaque_val))";
    nonStaticMethodFmt.addSubst(substVar, castCode).withSelf(castCode);
    traitMethodFmt.addSubst(substVar,
                            "(*static_cast<const ConcreteAttr *>(this))");
    extraDeclsFmt.addSubst(substVar, "(*this)");
  }
};

/// A specialized generator for operation interfaces.
struct OpInterfaceGenerator : public InterfaceGenerator {
  OpInterfaceGenerator(const llvm::RecordKeeper &records, raw_ostream &os)
      : InterfaceGenerator(getAllInterfaceDefinitions(records, "Op"), os) {
    valueType = "::mlir::Operation *";
    interfaceBaseType = "OpInterface";
    valueTemplate = "ConcreteOp";
    substVar = "_op";
    StringRef castCode = "(llvm::cast<ConcreteOp>(tablegen_opaque_val))";
    nonStaticMethodFmt.addSubst("_this", "impl")
        .addSubst(substVar, castCode)
        .withSelf(castCode);
    traitMethodFmt.addSubst(substVar, "(*static_cast<ConcreteOp *>(this))");
    extraDeclsFmt.addSubst(substVar, "(*this)");
  }
};

/// A specialized generator for type interfaces.
struct TypeInterfaceGenerator : public InterfaceGenerator {
  TypeInterfaceGenerator(const llvm::RecordKeeper &records, raw_ostream &os)
      : InterfaceGenerator(getAllInterfaceDefinitions(records, "Type"), os) {
    valueType = "::mlir::Type";
    interfaceBaseType = "TypeInterface";
    valueTemplate = "ConcreteType";
    substVar = "_type";
    StringRef castCode = "(::llvm::cast<ConcreteType>(tablegen_opaque_val))";
    nonStaticMethodFmt.addSubst(substVar, castCode).withSelf(castCode);
    traitMethodFmt.addSubst(substVar,
                            "(*static_cast<const ConcreteType *>(this))");
    extraDeclsFmt.addSubst(substVar, "(*this)");
  }
};

} // namespace tblgen
} // namespace mlir

#endif //  MLIR_TOOLS_MLIRTBLGEN_OPINTERFACESGEN_H_
