//===- SideEffects.cpp - SideEffect classes -------------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/SideEffects.h"
#include "llvm/ADT/Twine.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// SideEffect
//===----------------------------------------------------------------------===//

StringRef SideEffect::getName() const {
  return def->getValueAsString("effect");
}

StringRef SideEffect::getBaseEffectName() const {
  return def->getValueAsString("baseEffectName");
}

std::string SideEffect::getInterfaceTrait() const {
  StringRef trait = def->getValueAsString("interfaceTrait");
  StringRef cppNamespace = def->getValueAsString("cppNamespace");
  return cppNamespace.empty() ? trait.str()
                              : (cppNamespace + "::" + trait).str();
}

StringRef SideEffect::getResource() const {
  return def->getValueAsString("resource");
}

bool SideEffect::classof(const Operator::VariableDecorator *var) {
  return var->getDef().isSubClassOf("SideEffect");
}

//===----------------------------------------------------------------------===//
// SideEffectsTrait
//===----------------------------------------------------------------------===//

Operator::var_decorator_range SideEffectTrait::getEffects() const {
  auto *listInit = dyn_cast<llvm::ListInit>(def->getValueInit("effects"));
  return {listInit->begin(), listInit->end()};
}

StringRef SideEffectTrait::getBaseEffectName() const {
  return def->getValueAsString("baseEffectName");
}

bool SideEffectTrait::classof(const OpTrait *t) {
  return t->getDef().isSubClassOf("SideEffectsTraitBase");
}
