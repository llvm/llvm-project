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

#include "CppGenUtilities.h"
#include "DocGenUtilities.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Interfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
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
static void emitMethodNameAndArgs(const InterfaceMethod &method,
                                  raw_ostream &os, StringRef valueType,
                                  bool addThisArg, bool addConst) {
  os << method.getName() << '(';
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

/// Get an array of all OpInterface definitions but exclude those subclassing
/// "DeclareOpInterfaceMethods".
static std::vector<const Record *>
getAllInterfaceDefinitions(const RecordKeeper &records, StringRef name) {
  std::vector<const Record *> defs =
      records.getAllDerivedDefinitions((name + "Interface").str());

  std::string declareName = ("Declare" + name + "InterfaceMethods").str();
  llvm::erase_if(defs, [&](const Record *def) {
    // Ignore any "declare methods" interfaces.
    if (def->isSubClassOf(declareName))
      return true;
    // Ignore interfaces defined outside of the top-level file.
    return llvm::SrcMgr.FindBufferContainingLoc(def->getLoc()[0]) !=
           llvm::SrcMgr.getMainFileID();
  });
  return defs;
}

namespace {
/// This struct is the base generator used when processing tablegen interfaces.
class InterfaceGenerator {
public:
  bool emitInterfaceDefs();
  bool emitInterfaceDecls();
  bool emitInterfaceDocs();

protected:
  InterfaceGenerator(std::vector<const Record *> &&defs, raw_ostream &os)
      : defs(std::move(defs)), os(os) {}

  void emitConceptDecl(const Interface &interface);
  void emitModelDecl(const Interface &interface);
  void emitModelMethodsDef(const Interface &interface);
  void forwardDeclareInterface(const Interface &interface);
  void emitInterfaceDecl(const Interface &interface);
  void emitInterfaceTraitDecl(const Interface &interface);

  /// The set of interface records to emit.
  std::vector<const Record *> defs;
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

/// A specialized generator for attribute interfaces.
struct AttrInterfaceGenerator : public InterfaceGenerator {
  AttrInterfaceGenerator(const RecordKeeper &records, raw_ostream &os)
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
  OpInterfaceGenerator(const RecordKeeper &records, raw_ostream &os)
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
  TypeInterfaceGenerator(const RecordKeeper &records, raw_ostream &os)
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
} // namespace

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
    emitMethodNameAndArgs(method, os, valueType, /*addThisArg=*/false,
                          /*addConst=*/!isOpInterface);

    // Forward to the method on the concrete operation type.
    os << " {\n      return " << implValue << "->" << method.getName() << '(';
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

  // Insert the method definitions.
  bool isOpInterface = isa<OpInterface>(interface);
  emitInterfaceDefMethods(interfaceQualName, interface, valueType, "getImpl()",
                          os, isOpInterface);

  // Insert the method definitions for base classes.
  for (auto &base : interface.getBaseInterfaces()) {
    emitInterfaceDefMethods(interfaceQualName, base, valueType,
                            "getImpl()->impl" + base.getName(), os,
                            isOpInterface);
  }
}

bool InterfaceGenerator::emitInterfaceDefs() {
  llvm::emitSourceFileHeader("Interface Definitions", os);

  for (const auto *def : defs)
    emitInterfaceDef(Interface(def), valueType, os);
  return false;
}

//===----------------------------------------------------------------------===//
// GEN: Interface declarations
//===----------------------------------------------------------------------===//

void InterfaceGenerator::emitConceptDecl(const Interface &interface) {
  os << "  struct Concept {\n";

  // Insert each of the pure virtual concept methods.
  os << "    /// The methods defined by the interface.\n";
  for (auto &method : interface.getMethods()) {
    os << "    ";
    emitCPPType(method.getReturnType(), os);
    os << "(*" << method.getName() << ")(";
    if (!method.isStatic()) {
      os << "const Concept *impl, ";
      emitCPPType(valueType, os) << (method.arg_empty() ? "" : ", ");
    }
    llvm::interleaveComma(
        method.getArguments(), os,
        [&](const InterfaceMethod::Argument &arg) { os << arg.type; });
    os << ");\n";
  }

  // Insert a field containing a concept for each of the base interfaces.
  auto baseInterfaces = interface.getBaseInterfaces();
  if (!baseInterfaces.empty()) {
    os << "    /// The base classes of this interface.\n";
    for (const auto &base : interface.getBaseInterfaces()) {
      os << "    const " << base.getFullyQualifiedName() << "::Concept *impl"
         << base.getName() << " = nullptr;\n";
    }

    // Define an "initialize" method that allows for the initialization of the
    // base class concepts.
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

void InterfaceGenerator::emitModelDecl(const Interface &interface) {
  // Emit the basic model and the fallback model.
  for (const char *modelClass : {"Model", "FallbackModel"}) {
    os << "  template<typename " << valueTemplate << ">\n";
    os << "  class " << modelClass << " : public Concept {\n  public:\n";
    os << "    using Interface = " << interface.getFullyQualifiedName()
       << ";\n";
    os << "    " << modelClass << "() : Concept{";
    llvm::interleaveComma(
        interface.getMethods(), os,
        [&](const InterfaceMethod &method) { os << method.getName(); });
    os << "} {}\n\n";

    // Insert each of the virtual method overrides.
    for (auto &method : interface.getMethods()) {
      emitCPPType(method.getReturnType(), os << "    static inline ");
      emitMethodNameAndArgs(method, os, valueType,
                            /*addThisArg=*/!method.isStatic(),
                            /*addConst=*/false);
      os << ";\n";
    }
    os << "  };\n";
  }

  // Emit the template for the external model.
  os << "  template<typename ConcreteModel, typename " << valueTemplate
     << ">\n";
  os << "  class ExternalModel : public FallbackModel<ConcreteModel> {\n";
  os << "  public:\n";
  os << "    using ConcreteEntity = " << valueTemplate << ";\n";

  // Emit declarations for methods that have default implementations. Other
  // methods are expected to be implemented by the concrete derived model.
  for (auto &method : interface.getMethods()) {
    if (!method.getDefaultImplementation())
      continue;
    os << "    ";
    if (method.isStatic())
      os << "static ";
    emitCPPType(method.getReturnType(), os);
    os << method.getName() << "(";
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

void InterfaceGenerator::emitModelMethodsDef(const Interface &interface) {
  llvm::SmallVector<StringRef, 2> namespaces;
  llvm::SplitString(interface.getCppNamespace(), namespaces, "::");
  for (StringRef ns : namespaces)
    os << "namespace " << ns << " {\n";

  for (auto &method : interface.getMethods()) {
    os << "template<typename " << valueTemplate << ">\n";
    emitCPPType(method.getReturnType(), os);
    os << "detail::" << interface.getName() << "InterfaceTraits::Model<"
       << valueTemplate << ">::";
    emitMethodNameAndArgs(method, os, valueType,
                          /*addThisArg=*/!method.isStatic(),
                          /*addConst=*/false);
    os << " {\n  ";

    // Check for a provided body to the function.
    if (std::optional<StringRef> body = method.getBody()) {
      if (method.isStatic())
        os << body->trim();
      else
        os << tblgen::tgfmt(body->trim(), &nonStaticMethodFmt);
      os << "\n}\n";
      continue;
    }

    // Forward to the method on the concrete operation type.
    if (method.isStatic())
      os << "return " << valueTemplate << "::";
    else
      os << tblgen::tgfmt("return $_self.", &nonStaticMethodFmt);

    // Add the arguments to the call.
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
    emitMethodNameAndArgs(method, os, valueType,
                          /*addThisArg=*/!method.isStatic(),
                          /*addConst=*/false);
    os << " {\n  ";

    // Forward to the method on the concrete Model implementation.
    if (method.isStatic())
      os << "return " << valueTemplate << "::";
    else
      os << "return static_cast<const " << valueTemplate << " *>(impl)->";

    // Add the arguments to the call.
    os << method.getName() << '(';
    if (!method.isStatic())
      os << "tablegen_opaque_val" << (method.arg_empty() ? "" : ", ");
    llvm::interleaveComma(
        method.getArguments(), os,
        [&](const InterfaceMethod::Argument &arg) { os << arg.name; });
    os << ");\n}\n";
  }

  // Emit default implementations for the external model.
  for (auto &method : interface.getMethods()) {
    if (!method.getDefaultImplementation())
      continue;
    os << "template<typename ConcreteModel, typename " << valueTemplate
       << ">\n";
    emitCPPType(method.getReturnType(), os);
    os << "detail::" << interface.getName()
       << "InterfaceTraits::ExternalModel<ConcreteModel, " << valueTemplate
       << ">::";

    os << method.getName() << "(";
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

    // Use the empty context for static methods.
    tblgen::FmtContext ctx;
    os << tblgen::tgfmt(method.getDefaultImplementation()->trim(),
                        method.isStatic() ? &ctx : &nonStaticMethodFmt);
    os << "\n}\n";
  }

  for (StringRef ns : llvm::reverse(namespaces))
    os << "} // namespace " << ns << "\n";
}

void InterfaceGenerator::emitInterfaceTraitDecl(const Interface &interface) {
  llvm::SmallVector<StringRef, 2> namespaces;
  llvm::SplitString(interface.getCppNamespace(), namespaces, "::");
  for (StringRef ns : namespaces)
    os << "namespace " << ns << " {\n";

  os << "namespace detail {\n";

  StringRef interfaceName = interface.getName();
  auto interfaceTraitsName = (interfaceName + "InterfaceTraits").str();
  os << llvm::formatv("  template <typename {3}>\n"
                      "  struct {0}Trait : public ::mlir::{2}<{0},"
                      " detail::{1}>::Trait<{3}> {{\n",
                      interfaceName, interfaceTraitsName, interfaceBaseType,
                      valueTemplate);

  // Insert the default implementation for any methods.
  bool isOpInterface = isa<OpInterface>(interface);
  for (auto &method : interface.getMethods()) {
    // Flag interface methods named verifyTrait.
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
    emitMethodNameAndArgs(method, os, valueType, /*addThisArg=*/false,
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
  os << "}// namespace detail\n";

  for (StringRef ns : llvm::reverse(namespaces))
    os << "} // namespace " << ns << "\n";
}

static void emitInterfaceDeclMethods(const Interface &interface,
                                     raw_ostream &os, StringRef valueType,
                                     bool isOpInterface,
                                     tblgen::FmtContext &extraDeclsFmt) {
  for (auto &method : interface.getMethods()) {
    emitInterfaceMethodDoc(method, os, "  ");
    emitCPPType(method.getReturnType(), os << "  ");
    emitMethodNameAndArgs(method, os, valueType, /*addThisArg=*/false,
                          /*addConst=*/!isOpInterface);
    os << ";\n";
  }

  // Emit any extra declarations.
  if (std::optional<StringRef> extraDecls =
          interface.getExtraClassDeclaration())
    os << extraDecls->rtrim() << "\n";
  if (std::optional<StringRef> extraDecls =
          interface.getExtraSharedClassDeclaration())
    os << tblgen::tgfmt(extraDecls->rtrim(), &extraDeclsFmt) << "\n";
}

void InterfaceGenerator::forwardDeclareInterface(const Interface &interface) {
  llvm::SmallVector<StringRef, 2> namespaces;
  llvm::SplitString(interface.getCppNamespace(), namespaces, "::");
  for (StringRef ns : namespaces)
    os << "namespace " << ns << " {\n";

  // Emit a forward declaration of the interface class so that it becomes usable
  // in the signature of its methods.
  std::string comments = tblgen::emitSummaryAndDescComments(
      "", interface.getDescription().value_or(""));
  if (!comments.empty()) {
    os << comments << "\n";
  }

  StringRef interfaceName = interface.getName();
  os << "class " << interfaceName << ";\n";

  for (StringRef ns : llvm::reverse(namespaces))
    os << "} // namespace " << ns << "\n";
}

void InterfaceGenerator::emitInterfaceDecl(const Interface &interface) {
  llvm::SmallVector<StringRef, 2> namespaces;
  llvm::SplitString(interface.getCppNamespace(), namespaces, "::");
  for (StringRef ns : namespaces)
    os << "namespace " << ns << " {\n";

  StringRef interfaceName = interface.getName();
  auto interfaceTraitsName = (interfaceName + "InterfaceTraits").str();

  // Emit a forward declaration of the interface class so that it becomes usable
  // in the signature of its methods.
  std::string comments = tblgen::emitSummaryAndDescComments(
      "", interface.getDescription().value_or(""));
  if (!comments.empty()) {
    os << comments << "\n";
  }

  // Emit the traits struct containing the concept and model declarations.
  os << "namespace detail {\n"
     << "struct " << interfaceTraitsName << " {\n";
  emitConceptDecl(interface);
  emitModelDecl(interface);
  os << "};\n";

  // Emit the derived trait for the interface.
  os << "template <typename " << valueTemplate << ">\n";
  os << "struct " << interface.getName() << "Trait;\n";

  os << "\n} // namespace detail\n";

  // Emit the main interface class declaration.
  os << llvm::formatv("class {0} : public ::mlir::{3}<{1}, detail::{2}> {\n"
                      "public:\n"
                      "  using ::mlir::{3}<{1}, detail::{2}>::{3};\n",
                      interfaceName, interfaceName, interfaceTraitsName,
                      interfaceBaseType);

  // Emit a utility wrapper trait class.
  os << llvm::formatv("  template <typename {1}>\n"
                      "  struct Trait : public detail::{0}Trait<{1}> {{};\n",
                      interfaceName, valueTemplate);

  // Insert the method declarations.
  bool isOpInterface = isa<OpInterface>(interface);
  emitInterfaceDeclMethods(interface, os, valueType, isOpInterface,
                           extraDeclsFmt);

  // Insert the method declarations for base classes.
  for (auto &base : interface.getBaseInterfaces()) {
    std::string baseQualName = base.getFullyQualifiedName();
    os << "  //"
          "===---------------------------------------------------------------"
          "-===//\n"
       << "  // Inherited from " << baseQualName << "\n"
       << "  //"
          "===---------------------------------------------------------------"
          "-===//\n\n";

    // Allow implicit conversion to the base interface.
    os << "  operator " << baseQualName << " () const {\n"
       << "    if (!*this) return nullptr;\n"
       << "    return " << baseQualName << "(*this, getImpl()->impl"
       << base.getName() << ");\n"
       << "  }\n\n";

    // Inherit the base interface's methods.
    emitInterfaceDeclMethods(base, os, valueType, isOpInterface, extraDeclsFmt);
  }

  // Emit classof code if necessary.
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

  for (StringRef ns : llvm::reverse(namespaces))
    os << "} // namespace " << ns << "\n";
}

bool InterfaceGenerator::emitInterfaceDecls() {
  llvm::emitSourceFileHeader("Interface Declarations", os);
  // Sort according to ID, so defs are emitted in the order in which they appear
  // in the Tablegen file.
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

  // Emit the interface name followed by the description.
  os << "\n## " << interface.getName() << " (`" << interfaceDef.getName()
     << "`)\n";
  if (auto description = interface.getDescription())
    mlir::tblgen::emitDescription(*description, os);

  // Emit the methods required by the interface.
  os << "\n### Methods:\n";
  for (const auto &method : interface.getMethods()) {
    // Emit the method name.
    os << "\n#### `" << method.getName() << "`\n\n```c++\n";

    // Emit the method signature.
    if (method.isStatic())
      os << "static ";
    emitCPPType(method.getReturnType(), os) << method.getName() << '(';
    llvm::interleaveComma(method.getArguments(), os,
                          [&](const InterfaceMethod::Argument &arg) {
                            emitCPPType(arg.type, os) << arg.name;
                          });
    os << ");\n```\n";

    // Emit the description.
    if (auto description = method.getDescription())
      mlir::tblgen::emitDescription(*description, os);

    // If the body is not provided, this method must be provided by the user.
    if (!method.getBody())
      os << "\nNOTE: This method *must* be implemented by the user.";

    os << "\n";
  }
}

bool InterfaceGenerator::emitInterfaceDocs() {
  os << "<!-- Autogenerated by mlir-tblgen; don't manually edit -->\n";
  os << "\n# " << interfaceBaseType << " definitions\n";

  for (const auto *def : defs)
    emitInterfaceDoc(*def, os);
  return false;
}

//===----------------------------------------------------------------------===//
// GEN: Interface registration hooks
//===----------------------------------------------------------------------===//

namespace {
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
                 [](const RecordKeeper &records, raw_ostream &os) {
                   return GeneratorT(records, os).emitInterfaceDecls();
                 }),
        genDefs(genDefArg, genDefDesc,
                [](const RecordKeeper &records, raw_ostream &os) {
                  return GeneratorT(records, os).emitInterfaceDefs();
                }),
        genDocs(genDocArg, genDocDesc,
                [](const RecordKeeper &records, raw_ostream &os) {
                  return GeneratorT(records, os).emitInterfaceDocs();
                }) {}

  std::string genDeclArg, genDefArg, genDocArg;
  std::string genDeclDesc, genDefDesc, genDocDesc;
  mlir::GenRegistration genDecls, genDefs, genDocs;
};
} // namespace

static InterfaceGenRegistration<AttrInterfaceGenerator> attrGen("attr",
                                                                "attribute");
static InterfaceGenRegistration<OpInterfaceGenerator> opGen("op", "op");
static InterfaceGenRegistration<TypeInterfaceGenerator> typeGen("type", "type");
