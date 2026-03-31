//===- OpInterfacesGen.cpp - MLIR op interface utility generator ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpInterfacesGen generates definitions for operation interfaces.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Generators/OpInterfacesGen.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/Generators/CppGenUtilities.h"
#include "mlir/TableGen/Generators/DocGenUtilities.h"
#include "mlir/TableGen/Interfaces.h"
#include "llvm/ADT/SmallVector.h"
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
using mlir::tblgen::Interface;
using mlir::tblgen::InterfaceMethod;
using mlir::tblgen::OpInterface;

/// Emit a string corresponding to a C++ type, followed by a space if necessary.
static raw_ostream &emitCPPType(StringRef type, raw_ostream &os) {
  type = type.trim();
  os << type;
  if (type.back() != '&' && type.back() != '*')
    os << " ";
  return os;
}

/// Emit the method name and argument list for the given method. If 'addThisArg'
/// is true, then an argument is added to the beginning of the argument list for
/// the concrete value.
static void emitMethodNameAndArgs(const InterfaceMethod &method, StringRef name,
                                  raw_ostream &os, StringRef valueType,
                                  bool addThisArg, bool addConst) {
  os << name << '(';
  if (addThisArg) {
    if (addConst)
      os << "const ";
    os << "const Concept *impl, ";
    emitCPPType(valueType, os)
        << "tablegen_opaque_val" << (method.arg_empty() ? "" : ", ");
  }
  llvm::interleaveComma(method.getArguments(), os,
                        [&](const InterfaceMethod::Argument &arg) {
                          os << arg.type << " " << arg.name;
                        });
  os << ')';
  if (addConst)
    os << " const";
}

std::vector<const Record *>
mlir::tblgen::getAllInterfaceDefinitions(const RecordKeeper &records,
                                         StringRef name) {
  std::vector<const Record *> defs =
      records.getAllDerivedDefinitions((name + "Interface").str());

  std::string declareName = ("Declare" + name + "InterfaceMethods").str();
  llvm::erase_if(defs, [&](const Record *def) {
    if (def->isSubClassOf(declareName))
      return true;
    return llvm::SrcMgr.FindBufferContainingLoc(def->getLoc()[0]) !=
           llvm::SrcMgr.getMainFileID();
  });
  return defs;
}

mlir::tblgen::AttrInterfaceGenerator::AttrInterfaceGenerator(
    const RecordKeeper &records, raw_ostream &os)
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

mlir::tblgen::OpInterfaceGenerator::OpInterfaceGenerator(
    const RecordKeeper &records, raw_ostream &os)
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

mlir::tblgen::TypeInterfaceGenerator::TypeInterfaceGenerator(
    const RecordKeeper &records, raw_ostream &os)
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

//===----------------------------------------------------------------------===//
// GEN: Interface definitions
//===----------------------------------------------------------------------===//

static void emitInterfaceMethodDoc(const InterfaceMethod &method,
                                   raw_ostream &os, StringRef prefix = "") {
  if (std::optional<StringRef> description = method.getDescription())
    tblgen::emitDescriptionComment(*description, os, prefix);
}
static void emitInterfaceDefMethods(StringRef interfaceQualName,
                                    const Interface &interface,
                                    StringRef valueType, const Twine &implValue,
                                    raw_ostream &os, bool isOpInterface) {
  for (auto &method : interface.getMethods()) {
    emitInterfaceMethodDoc(method, os);
    emitCPPType(method.getReturnType(), os);
    os << interfaceQualName << "::";
    emitMethodNameAndArgs(method, method.getName(), os, valueType,
                          /*addThisArg=*/false,
                          /*addConst=*/!isOpInterface);

    // Forward to the method on the concrete operation type.
    os << " {\n      return " << implValue << "->" << method.getUniqueName()
       << '(';
    if (!method.isStatic()) {
      os << implValue << ", ";
      os << (isOpInterface ? "getOperation()" : "*this");
      os << (method.arg_empty() ? "" : ", ");
    }
    llvm::interleaveComma(
        method.getArguments(), os,
        [&](const InterfaceMethod::Argument &arg) { os << arg.name; });
    os << ");\n  }\n";
  }
}

static void emitInterfaceDef(const Interface &interface, StringRef valueType,
                             raw_ostream &os) {
  std::string interfaceQualNameStr = interface.getFullyQualifiedName();
  StringRef interfaceQualName = interfaceQualNameStr;
  interfaceQualName.consume_front("::");

  bool isOpInterface = isa<OpInterface>(interface);
  emitInterfaceDefMethods(interfaceQualName, interface, valueType, "getImpl()",
                          os, isOpInterface);

  for (auto &base : interface.getBaseInterfaces()) {
    emitInterfaceDefMethods(interfaceQualName, base, valueType,
                            "getImpl()->impl" + base.getName(), os,
                            isOpInterface);
  }
}

bool mlir::tblgen::InterfaceGenerator::emitInterfaceDefs() {
  llvm::emitSourceFileHeader("Interface Definitions", os);

  for (const auto *def : defs)
    emitInterfaceDef(Interface(def), valueType, os);
  return false;
}

//===----------------------------------------------------------------------===//
// GEN: Interface declarations
//===----------------------------------------------------------------------===//

void mlir::tblgen::InterfaceGenerator::emitConceptDecl(
    const Interface &interface) {
  os << "  struct Concept {\n";

  os << "    /// The methods defined by the interface.\n";
  for (auto &method : interface.getMethods()) {
    os << "    ";
    emitCPPType(method.getReturnType(), os);
    os << "(*" << method.getUniqueName() << ")(";
    if (!method.isStatic()) {
      os << "const Concept *impl, ";
      emitCPPType(valueType, os) << (method.arg_empty() ? "" : ", ");
    }
    llvm::interleaveComma(
        method.getArguments(), os,
        [&](const InterfaceMethod::Argument &arg) { os << arg.type; });
    os << ");\n";
  }

  auto baseInterfaces = interface.getBaseInterfaces();
  if (!baseInterfaces.empty()) {
    os << "    /// The base classes of this interface.\n";
    for (const auto &base : interface.getBaseInterfaces()) {
      os << "    const " << base.getFullyQualifiedName() << "::Concept *impl"
         << base.getName() << " = nullptr;\n";
    }

    os << "\n    void initializeInterfaceConcept(::mlir::detail::InterfaceMap "
          "&interfaceMap) {\n";
    std::string interfaceQualName = interface.getFullyQualifiedName();
    for (const auto &base : interface.getBaseInterfaces()) {
      StringRef baseName = base.getName();
      std::string baseQualName = base.getFullyQualifiedName();
      os << "      impl" << baseName << " = interfaceMap.lookup<"
         << baseQualName << ">();\n"
         << "      assert(impl" << baseName << " && \"`" << interfaceQualName
         << "` expected its base interface `" << baseQualName
         << "` to be registered\");\n";
    }
    os << "    }\n";
  }

  os << "  };\n";
}

void mlir::tblgen::InterfaceGenerator::emitModelDecl(
    const Interface &interface) {
  for (const char *modelClass : {"Model", "FallbackModel"}) {
    os << "  template<typename " << valueTemplate << ">\n";
    os << "  class " << modelClass << " : public Concept {\n  public:\n";
    os << "    using Interface = " << interface.getFullyQualifiedName()
       << ";\n";
    os << "    " << modelClass << "() : Concept{";
    llvm::interleaveComma(
        interface.getMethods(), os,
        [&](const InterfaceMethod &method) { os << method.getUniqueName(); });
    os << "} {}\n\n";

    for (auto &method : interface.getMethods()) {
      emitCPPType(method.getReturnType(), os << "    static inline ");
      emitMethodNameAndArgs(method, method.getUniqueName(), os, valueType,
                            /*addThisArg=*/!method.isStatic(),
                            /*addConst=*/false);
      os << ";\n";
    }
    os << "  };\n";
  }

  os << "  template<typename ConcreteModel, typename " << valueTemplate
     << ">\n";
  os << "  class ExternalModel : public FallbackModel<ConcreteModel> {\n";
  os << "  public:\n";
  os << "    using ConcreteEntity = " << valueTemplate << ";\n";

  for (auto &method : interface.getMethods()) {
    if (!method.getDefaultImplementation())
      continue;
    os << "    ";
    if (method.isStatic())
      os << "static ";
    emitCPPType(method.getReturnType(), os);
    os << method.getUniqueName() << "(";
    if (!method.isStatic()) {
      emitCPPType(valueType, os);
      os << "tablegen_opaque_val";
      if (!method.arg_empty())
        os << ", ";
    }
    llvm::interleaveComma(method.getArguments(), os,
                          [&](const InterfaceMethod::Argument &arg) {
                            emitCPPType(arg.type, os);
                            os << arg.name;
                          });
    os << ")";
    if (!method.isStatic())
      os << " const";
    os << ";\n";
  }
  os << "  };\n";
}

void mlir::tblgen::InterfaceGenerator::emitModelMethodsDef(
    const Interface &interface) {
  llvm::NamespaceEmitter ns(os, interface.getCppNamespace());
  for (auto &method : interface.getMethods()) {
    os << "template<typename " << valueTemplate << ">\n";
    emitCPPType(method.getReturnType(), os);
    os << "detail::" << interface.getName() << "InterfaceTraits::Model<"
       << valueTemplate << ">::";
    emitMethodNameAndArgs(method, method.getUniqueName(), os, valueType,
                          /*addThisArg=*/!method.isStatic(),
                          /*addConst=*/false);
    os << " {\n  ";

    if (std::optional<StringRef> body = method.getBody()) {
      if (method.isStatic())
        os << body->trim();
      else
        os << tblgen::tgfmt(body->trim(), &nonStaticMethodFmt);
      os << "\n}\n";
      continue;
    }

    if (method.isStatic())
      os << "return " << valueTemplate << "::";
    else
      os << tblgen::tgfmt("return $_self.", &nonStaticMethodFmt);

    os << method.getName() << '(';
    llvm::interleaveComma(
        method.getArguments(), os,
        [&](const InterfaceMethod::Argument &arg) { os << arg.name; });
    os << ");\n}\n";
  }

  for (auto &method : interface.getMethods()) {
    os << "template<typename " << valueTemplate << ">\n";
    emitCPPType(method.getReturnType(), os);
    os << "detail::" << interface.getName() << "InterfaceTraits::FallbackModel<"
       << valueTemplate << ">::";
    emitMethodNameAndArgs(method, method.getUniqueName(), os, valueType,
                          /*addThisArg=*/!method.isStatic(),
                          /*addConst=*/false);
    os << " {\n  ";

    if (method.isStatic())
      os << "return " << valueTemplate << "::";
    else
      os << "return static_cast<const " << valueTemplate << " *>(impl)->";

    os << method.getUniqueName() << '(';
    if (!method.isStatic())
      os << "tablegen_opaque_val" << (method.arg_empty() ? "" : ", ");
    llvm::interleaveComma(
        method.getArguments(), os,
        [&](const InterfaceMethod::Argument &arg) { os << arg.name; });
    os << ");\n}\n";
  }

  for (auto &method : interface.getMethods()) {
    if (!method.getDefaultImplementation())
      continue;
    os << "template<typename ConcreteModel, typename " << valueTemplate
       << ">\n";
    emitCPPType(method.getReturnType(), os);
    os << "detail::" << interface.getName()
       << "InterfaceTraits::ExternalModel<ConcreteModel, " << valueTemplate
       << ">::";

    os << method.getUniqueName() << "(";
    if (!method.isStatic()) {
      emitCPPType(valueType, os);
      os << "tablegen_opaque_val";
      if (!method.arg_empty())
        os << ", ";
    }
    llvm::interleaveComma(method.getArguments(), os,
                          [&](const InterfaceMethod::Argument &arg) {
                            emitCPPType(arg.type, os);
                            os << arg.name;
                          });
    os << ")";
    if (!method.isStatic())
      os << " const";

    os << " {\n";

    tblgen::FmtContext ctx;
    os << tblgen::tgfmt(method.getDefaultImplementation()->trim(),
                        method.isStatic() ? &ctx : &nonStaticMethodFmt);
    os << "\n}\n";
  }
}

void mlir::tblgen::InterfaceGenerator::emitInterfaceTraitDecl(
    const Interface &interface) {
  auto cppNamespace = (interface.getCppNamespace() + "::detail").str();
  llvm::NamespaceEmitter ns(os, cppNamespace);

  StringRef interfaceName = interface.getName();
  auto interfaceTraitsName = (interfaceName + "InterfaceTraits").str();
  os << llvm::formatv("  template <typename {3}>\n"
                      "  struct {0}Trait : public ::mlir::{2}<{0},"
                      " detail::{1}>::Trait<{3}> {{\n",
                      interfaceName, interfaceTraitsName, interfaceBaseType,
                      valueTemplate);

  bool isOpInterface = isa<OpInterface>(interface);
  for (auto &method : interface.getMethods()) {
    if (method.getName() == "verifyTrait")
      PrintFatalError(
          formatv("'verifyTrait' method cannot be specified as interface "
                  "method for '{0}'; use the 'verify' field instead",
                  interfaceName));
    auto defaultImpl = method.getDefaultImplementation();
    if (!defaultImpl)
      continue;

    emitInterfaceMethodDoc(method, os, "    ");
    os << "    " << (method.isStatic() ? "static " : "");
    emitCPPType(method.getReturnType(), os);
    emitMethodNameAndArgs(method, method.getName(), os, valueType,
                          /*addThisArg=*/false,
                          /*addConst=*/!isOpInterface && !method.isStatic());
    os << " {\n      " << tblgen::tgfmt(defaultImpl->trim(), &traitMethodFmt)
       << "\n    }\n";
  }

  if (auto verify = interface.getVerify()) {
    assert(isa<OpInterface>(interface) && "only OpInterface supports 'verify'");

    tblgen::FmtContext verifyCtx;
    verifyCtx.addSubst("_op", "op");
    os << llvm::formatv(
              "    static ::llvm::LogicalResult {0}(::mlir::Operation *op) ",
              (interface.verifyWithRegions() ? "verifyRegionTrait"
                                             : "verifyTrait"))
       << "{\n      " << tblgen::tgfmt(verify->trim(), &verifyCtx)
       << "\n    }\n";
  }
  if (auto extraTraitDecls = interface.getExtraTraitClassDeclaration())
    os << tblgen::tgfmt(*extraTraitDecls, &traitMethodFmt) << "\n";
  if (auto extraTraitDecls = interface.getExtraSharedClassDeclaration())
    os << tblgen::tgfmt(*extraTraitDecls, &traitMethodFmt) << "\n";

  os << "  };\n";
}

static void emitInterfaceDeclMethods(const Interface &interface,
                                     raw_ostream &os, StringRef valueType,
                                     bool isOpInterface,
                                     tblgen::FmtContext &extraDeclsFmt) {
  for (auto &method : interface.getMethods()) {
    emitInterfaceMethodDoc(method, os, "  ");
    emitCPPType(method.getReturnType(), os << "  ");
    emitMethodNameAndArgs(method, method.getName(), os, valueType,
                          /*addThisArg=*/false,
                          /*addConst=*/!isOpInterface);
    os << ";\n";
  }

  if (std::optional<StringRef> extraDecls =
          interface.getExtraClassDeclaration())
    os << extraDecls->rtrim() << "\n";
  if (std::optional<StringRef> extraDecls =
          interface.getExtraSharedClassDeclaration())
    os << tblgen::tgfmt(extraDecls->rtrim(), &extraDeclsFmt) << "\n";
}

void mlir::tblgen::InterfaceGenerator::forwardDeclareInterface(
    const Interface &interface) {
  llvm::NamespaceEmitter ns(os, interface.getCppNamespace());

  tblgen::emitSummaryAndDescComments(os, "",
                                     interface.getDescription().value_or(""));

  StringRef interfaceName = interface.getName();
  os << "class " << interfaceName << ";\n";
}

void mlir::tblgen::InterfaceGenerator::emitInterfaceDecl(
    const Interface &interface) {
  llvm::NamespaceEmitter ns(os, interface.getCppNamespace());

  StringRef interfaceName = interface.getName();
  auto interfaceTraitsName = (interfaceName + "InterfaceTraits").str();

  tblgen::emitSummaryAndDescComments(os, "",
                                     interface.getDescription().value_or(""));

  os << "namespace detail {\n"
     << "struct " << interfaceTraitsName << " {\n";
  emitConceptDecl(interface);
  emitModelDecl(interface);
  os << "};\n";

  os << "template <typename " << valueTemplate << ">\n";
  os << "struct " << interface.getName() << "Trait;\n";

  os << "\n} // namespace detail\n";

  os << llvm::formatv("class {0} : public ::mlir::{3}<{1}, detail::{2}> {\n"
                      "public:\n"
                      "  using ::mlir::{3}<{1}, detail::{2}>::{3};\n",
                      interfaceName, interfaceName, interfaceTraitsName,
                      interfaceBaseType);

  os << llvm::formatv("  template <typename {1}>\n"
                      "  struct Trait : public detail::{0}Trait<{1}> {{};\n",
                      interfaceName, valueTemplate);

  bool isOpInterface = isa<OpInterface>(interface);
  emitInterfaceDeclMethods(interface, os, valueType, isOpInterface,
                           extraDeclsFmt);

  for (auto &base : interface.getBaseInterfaces()) {
    std::string baseQualName = base.getFullyQualifiedName();
    os << "  //"
          "===---------------------------------------------------------------"
          "-===//\n"
       << "  // Inherited from " << baseQualName << "\n"
       << "  //"
          "===---------------------------------------------------------------"
          "-===//\n\n";

    os << "  operator " << baseQualName << " () const {\n"
       << "    if (!*this) return nullptr;\n"
       << "    return " << baseQualName << "(*this, getImpl()->impl"
       << base.getName() << ");\n"
       << "  }\n\n";

    emitInterfaceDeclMethods(base, os, valueType, isOpInterface, extraDeclsFmt);
  }

  if (std::optional<StringRef> extraClassOf = interface.getExtraClassOf()) {
    auto extraClassOfFmt = tblgen::FmtContext();
    extraClassOfFmt.addSubst(substVar, "odsInterfaceInstance");
    os << "  static bool classof(" << valueType << " base) {\n"
       << "    auto* interface = getInterfaceFor(base);\n"
       << "    if (!interface)\n"
          "      return false;\n"
          "    "
       << interfaceName << " odsInterfaceInstance(base, interface);\n"
       << "    " << tblgen::tgfmt(extraClassOf->trim(), &extraClassOfFmt)
       << "\n  }\n";
  }

  os << "};\n";
}

bool mlir::tblgen::InterfaceGenerator::emitInterfaceDecls() {
  llvm::emitSourceFileHeader("Interface Declarations", os);
  std::vector<const Record *> sortedDefs(defs);
  llvm::sort(sortedDefs, [](const Record *lhs, const Record *rhs) {
    return lhs->getID() < rhs->getID();
  });
  for (const Record *def : sortedDefs)
    forwardDeclareInterface(Interface(def));
  for (const Record *def : sortedDefs)
    emitInterfaceDecl(Interface(def));
  for (const Record *def : sortedDefs)
    emitInterfaceTraitDecl(Interface(def));
  for (const Record *def : sortedDefs)
    emitModelMethodsDef(Interface(def));

  return false;
}

//===----------------------------------------------------------------------===//
// GEN: Interface documentation
//===----------------------------------------------------------------------===//

static void emitInterfaceDoc(const Record &interfaceDef, raw_ostream &os) {
  Interface interface(&interfaceDef);

  os << "\n## " << interface.getName() << " (`" << interfaceDef.getName()
     << "`)\n";
  if (auto description = interface.getDescription())
    mlir::tblgen::emitDescription(*description, os);

  os << "\n### Methods:\n";
  for (const auto &method : interface.getMethods()) {
    os << "\n#### `" << method.getName() << "`\n\n```c++\n";

    if (method.isStatic())
      os << "static ";
    emitCPPType(method.getReturnType(), os) << method.getName() << '(';
    llvm::interleaveComma(method.getArguments(), os,
                          [&](const InterfaceMethod::Argument &arg) {
                            emitCPPType(arg.type, os) << arg.name;
                          });
    os << ");\n```\n";

    if (auto description = method.getDescription())
      mlir::tblgen::emitDescription(*description, os);

    if (!method.getBody())
      os << "\nNOTE: This method *must* be implemented by the user.";

    os << "\n";
  }
}

bool mlir::tblgen::InterfaceGenerator::emitInterfaceDocs() {
  os << "<!-- Autogenerated by mlir-tblgen; don't manually edit -->\n";
  os << "\n# " << interfaceBaseType << " definitions\n";

  for (const auto *def : defs)
    emitInterfaceDoc(*def, os);
  return false;
}
