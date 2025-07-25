//===- Interfaces.cpp - Interface classes ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Interfaces.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;
using llvm::DagInit;
using llvm::DefInit;
using llvm::Init;
using llvm::ListInit;
using llvm::Record;
using llvm::StringInit;

//===----------------------------------------------------------------------===//
// InterfaceMethod
//===----------------------------------------------------------------------===//

InterfaceMethod::InterfaceMethod(const Record *def) : def(def) {
  const DagInit *args = def->getValueAsDag("arguments");
  for (unsigned i = 0, e = args->getNumArgs(); i != e; ++i) {
    arguments.push_back({cast<StringInit>(args->getArg(i))->getValue(),
                         args->getArgNameStr(i)});
  }
}

StringRef InterfaceMethod::getReturnType() const {
  return def->getValueAsString("returnType");
}

// Return the name of this method.
StringRef InterfaceMethod::getName() const {
  return def->getValueAsString("name");
}

// Return if this method is static.
bool InterfaceMethod::isStatic() const {
  return def->isSubClassOf("StaticInterfaceMethod");
}

// Return the body for this method if it has one.
std::optional<StringRef> InterfaceMethod::getBody() const {
  // Trim leading and trailing spaces from the default implementation.
  auto value = def->getValueAsString("body").trim();
  return value.empty() ? std::optional<StringRef>() : value;
}

// Return the default implementation for this method if it has one.
std::optional<StringRef> InterfaceMethod::getDefaultImplementation() const {
  // Trim leading and trailing spaces from the default implementation.
  auto value = def->getValueAsString("defaultBody").trim();
  return value.empty() ? std::optional<StringRef>() : value;
}

// Return the description of this method if it has one.
std::optional<StringRef> InterfaceMethod::getDescription() const {
  auto value = def->getValueAsString("description");
  return value.empty() ? std::optional<StringRef>() : value;
}

ArrayRef<InterfaceMethod::Argument> InterfaceMethod::getArguments() const {
  return arguments;
}

bool InterfaceMethod::arg_empty() const { return arguments.empty(); }

//===----------------------------------------------------------------------===//
// Interface
//===----------------------------------------------------------------------===//

Interface::Interface(const Record *def) : def(def) {
  assert(def->isSubClassOf("Interface") &&
         "must be subclass of TableGen 'Interface' class");

  // Initialize the interface methods.
  auto *listInit = dyn_cast<ListInit>(def->getValueInit("methods"));
  for (const Init *init : listInit->getElements())
    methods.emplace_back(cast<DefInit>(init)->getDef());

  // Initialize the interface base classes.
  auto *basesInit = dyn_cast<ListInit>(def->getValueInit("baseInterfaces"));
  // Chained inheritance will produce duplicates in the base interface set.
  StringSet<> basesAdded;
  llvm::unique_function<void(Interface)> addBaseInterfaceFn =
      [&](const Interface &baseInterface) {
        // Inherit any base interfaces.
        for (const auto &baseBaseInterface : baseInterface.getBaseInterfaces())
          addBaseInterfaceFn(baseBaseInterface);

        // Add the base interface.
        if (basesAdded.contains(baseInterface.getName()))
          return;
        baseInterfaces.push_back(std::make_unique<Interface>(baseInterface));
        basesAdded.insert(baseInterface.getName());
      };
  for (const Init *init : basesInit->getElements())
    addBaseInterfaceFn(Interface(cast<DefInit>(init)->getDef()));
}

// Return the name of this interface.
StringRef Interface::getName() const {
  return def->getValueAsString("cppInterfaceName");
}

// Returns this interface's name prefixed with namespaces.
std::string Interface::getFullyQualifiedName() const {
  StringRef cppNamespace = getCppNamespace();
  StringRef name = getName();
  if (cppNamespace.empty())
    return name.str();
  return (cppNamespace + "::" + name).str();
}

// Return the C++ namespace of this interface.
StringRef Interface::getCppNamespace() const {
  return def->getValueAsString("cppNamespace");
}

// Return the methods of this interface.
ArrayRef<InterfaceMethod> Interface::getMethods() const { return methods; }

// Return the description of this method if it has one.
std::optional<StringRef> Interface::getDescription() const {
  auto value = def->getValueAsString("description");
  return value.empty() ? std::optional<StringRef>() : value;
}

// Return the interfaces extra class declaration code.
std::optional<StringRef> Interface::getExtraClassDeclaration() const {
  auto value = def->getValueAsString("extraClassDeclaration");
  return value.empty() ? std::optional<StringRef>() : value;
}

// Return the traits extra class declaration code.
std::optional<StringRef> Interface::getExtraTraitClassDeclaration() const {
  auto value = def->getValueAsString("extraTraitClassDeclaration");
  return value.empty() ? std::optional<StringRef>() : value;
}

// Return the shared extra class declaration code.
std::optional<StringRef> Interface::getExtraSharedClassDeclaration() const {
  auto value = def->getValueAsString("extraSharedClassDeclaration");
  return value.empty() ? std::optional<StringRef>() : value;
}

std::optional<StringRef> Interface::getExtraClassOf() const {
  auto value = def->getValueAsString("extraClassOf");
  return value.empty() ? std::optional<StringRef>() : value;
}

// Return the body for this method if it has one.
std::optional<StringRef> Interface::getVerify() const {
  // Only OpInterface supports the verify method.
  if (!isa<OpInterface>(this))
    return std::nullopt;
  auto value = def->getValueAsString("verify");
  return value.empty() ? std::optional<StringRef>() : value;
}

bool Interface::verifyWithRegions() const {
  return def->getValueAsBit("verifyWithRegions");
}

//===----------------------------------------------------------------------===//
// AttrInterface
//===----------------------------------------------------------------------===//

bool AttrInterface::classof(const Interface *interface) {
  return interface->getDef().isSubClassOf("AttrInterface");
}

//===----------------------------------------------------------------------===//
// OpInterface
//===----------------------------------------------------------------------===//

bool OpInterface::classof(const Interface *interface) {
  return interface->getDef().isSubClassOf("OpInterface");
}

//===----------------------------------------------------------------------===//
// TypeInterface
//===----------------------------------------------------------------------===//

bool TypeInterface::classof(const Interface *interface) {
  return interface->getDef().isSubClassOf("TypeInterface");
}
