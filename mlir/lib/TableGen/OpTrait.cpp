//===- OpTrait.cpp - OpTrait class ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpTrait wrapper to simplify using TableGen Record defining a MLIR OpTrait.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/OpTrait.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/Predicate.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

OpTrait OpTrait::create(const llvm::Init *init) {
  auto def = cast<llvm::DefInit>(init)->getDef();
  if (def->isSubClassOf("PredOpTrait"))
    return OpTrait(Kind::Pred, def);
  if (def->isSubClassOf("GenInternalOpTrait"))
    return OpTrait(Kind::Internal, def);
  if (def->isSubClassOf("OpInterfaceTrait"))
    return OpTrait(Kind::Interface, def);
  assert(def->isSubClassOf("NativeOpTrait"));
  return OpTrait(Kind::Native, def);
}

OpTrait::OpTrait(Kind kind, const llvm::Record *def) : def(def), kind(kind) {}

llvm::StringRef NativeOpTrait::getTrait() const {
  return def->getValueAsString("trait");
}

llvm::StringRef InternalOpTrait::getTrait() const {
  return def->getValueAsString("trait");
}

std::string PredOpTrait::getPredTemplate() const {
  auto pred = Pred(def->getValueInit("predicate"));
  return pred.getCondition();
}

llvm::StringRef PredOpTrait::getDescription() const {
  return def->getValueAsString("description");
}

OpInterface InterfaceOpTrait::getOpInterface() const {
  return OpInterface(def);
}

std::string InterfaceOpTrait::getTrait() const {
  llvm::StringRef trait = def->getValueAsString("trait");
  llvm::StringRef cppNamespace = def->getValueAsString("cppNamespace");
  return cppNamespace.empty() ? trait.str()
                              : (cppNamespace + "::" + trait).str();
}

bool InterfaceOpTrait::shouldDeclareMethods() const {
  return def->isSubClassOf("DeclareOpInterfaceMethods");
}

std::vector<StringRef> InterfaceOpTrait::getAlwaysDeclaredMethods() const {
  return def->getValueAsListOfStrings("alwaysOverriddenMethods");
}
