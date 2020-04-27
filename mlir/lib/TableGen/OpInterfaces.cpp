//===- OpInterfaces.cpp - OpInterfaces class ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpInterfaces wrapper to simplify using TableGen OpInterfaces.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/OpInterfaces.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

OpInterfaceMethod::OpInterfaceMethod(const llvm::Record *def) : def(def) {
  llvm::DagInit *args = def->getValueAsDag("arguments");
  for (unsigned i = 0, e = args->getNumArgs(); i != e; ++i) {
    arguments.push_back(
        {llvm::cast<llvm::StringInit>(args->getArg(i))->getValue(),
         args->getArgNameStr(i)});
  }
}

StringRef OpInterfaceMethod::getReturnType() const {
  return def->getValueAsString("returnType");
}

// Return the name of this method.
StringRef OpInterfaceMethod::getName() const {
  return def->getValueAsString("name");
}

// Return if this method is static.
bool OpInterfaceMethod::isStatic() const {
  return def->isSubClassOf("StaticInterfaceMethod");
}

// Return the body for this method if it has one.
llvm::Optional<StringRef> OpInterfaceMethod::getBody() const {
  auto value = def->getValueAsString("body");
  return value.empty() ? llvm::Optional<StringRef>() : value;
}

// Return the default implementation for this method if it has one.
llvm::Optional<StringRef> OpInterfaceMethod::getDefaultImplementation() const {
  auto value = def->getValueAsString("defaultBody");
  return value.empty() ? llvm::Optional<StringRef>() : value;
}

// Return the description of this method if it has one.
llvm::Optional<StringRef> OpInterfaceMethod::getDescription() const {
  auto value = def->getValueAsString("description");
  return value.empty() ? llvm::Optional<StringRef>() : value;
}

ArrayRef<OpInterfaceMethod::Argument> OpInterfaceMethod::getArguments() const {
  return arguments;
}

bool OpInterfaceMethod::arg_empty() const { return arguments.empty(); }

OpInterface::OpInterface(const llvm::Record *def) : def(def) {
  auto *listInit = dyn_cast<llvm::ListInit>(def->getValueInit("methods"));
  for (llvm::Init *init : listInit->getValues())
    methods.emplace_back(cast<llvm::DefInit>(init)->getDef());
}

// Return the name of this interface.
StringRef OpInterface::getName() const {
  return def->getValueAsString("cppClassName");
}

// Return the methods of this interface.
ArrayRef<OpInterfaceMethod> OpInterface::getMethods() const { return methods; }

// Return the description of this method if it has one.
llvm::Optional<StringRef> OpInterface::getDescription() const {
  auto value = def->getValueAsString("description");
  return value.empty() ? llvm::Optional<StringRef>() : value;
}

// Return the interfaces extra class declaration code.
llvm::Optional<StringRef> OpInterface::getExtraClassDeclaration() const {
  auto value = def->getValueAsString("extraClassDeclaration");
  return value.empty() ? llvm::Optional<StringRef>() : value;
}

// Return the traits extra class declaration code.
llvm::Optional<StringRef> OpInterface::getExtraTraitClassDeclaration() const {
  auto value = def->getValueAsString("extraTraitClassDeclaration");
  return value.empty() ? llvm::Optional<StringRef>() : value;
}

// Return the body for this method if it has one.
llvm::Optional<StringRef> OpInterface::getVerify() const {
  auto value = def->getValueAsString("verify");
  return value.empty() ? llvm::Optional<StringRef>() : value;
}
