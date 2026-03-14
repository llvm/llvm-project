//===- Trait.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Trait wrapper to simplify using TableGen Record defining a MLIR Trait.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Trait.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/Predicate.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// Trait
//===----------------------------------------------------------------------===//

Trait Trait::create(const llvm::Init *init) {
  auto *def = cast<llvm::DefInit>(init)->getDef();
  if (def->isSubClassOf("PredTrait"))
    return Trait(Kind::Pred, def);
  if (def->isSubClassOf("GenInternalTrait"))
    return Trait(Kind::Internal, def);
  if (def->isSubClassOf("InterfaceTrait"))
    return Trait(Kind::Interface, def);
  assert(def->isSubClassOf("NativeTrait"));
  return Trait(Kind::Native, def);
}

Trait::Trait(Kind kind, const llvm::Record *def) : def(def), kind(kind) {}

//===----------------------------------------------------------------------===//
// NativeTrait
//===----------------------------------------------------------------------===//

std::string NativeTrait::getFullyQualifiedTraitName() const {
  llvm::StringRef trait = def->getValueAsString("trait");
  llvm::StringRef cppNamespace = def->getValueAsString("cppNamespace");
  return cppNamespace.empty() ? trait.str()
                              : (cppNamespace + "::" + trait).str();
}

bool NativeTrait::isStructuralOpTrait() const {
  return def->isSubClassOf("StructuralOpTrait");
}

StringRef NativeTrait::getExtraConcreteClassDeclaration() const {
  return def->getValueAsString("extraConcreteClassDeclaration");
}

StringRef NativeTrait::getExtraConcreteClassDefinition() const {
  return def->getValueAsString("extraConcreteClassDefinition");
}

//===----------------------------------------------------------------------===//
// InternalTrait
//===----------------------------------------------------------------------===//

llvm::StringRef InternalTrait::getFullyQualifiedTraitName() const {
  return def->getValueAsString("trait");
}

//===----------------------------------------------------------------------===//
// PredTrait
//===----------------------------------------------------------------------===//

std::string PredTrait::getPredTemplate() const {
  auto pred = Pred(def->getValueInit("predicate"));
  return pred.getCondition();
}

llvm::StringRef PredTrait::getSummary() const {
  return def->getValueAsString("summary");
}

//===----------------------------------------------------------------------===//
// InterfaceTrait
//===----------------------------------------------------------------------===//

Interface InterfaceTrait::getInterface() const { return Interface(def); }

std::string InterfaceTrait::getFullyQualifiedTraitName() const {
  llvm::StringRef trait = def->getValueAsString("trait");
  llvm::StringRef cppNamespace = def->getValueAsString("cppNamespace");
  return cppNamespace.empty() ? trait.str()
                              : (cppNamespace + "::" + trait).str();
}

bool InterfaceTrait::shouldDeclareMethods() const {
  return def->isSubClassOf("DeclareInterfaceMethods");
}

std::vector<StringRef> InterfaceTrait::getAlwaysDeclaredMethods() const {
  return def->getValueAsListOfStrings("alwaysOverriddenMethods");
}
