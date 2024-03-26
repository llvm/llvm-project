//===- PDLPatternMatch.cpp - Base classes for PDL pattern match
//------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/RegionKindInterface.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// PDLValue
//===----------------------------------------------------------------------===//

void PDLValue::print(raw_ostream &os) const {
  if (!value) {
    os << "<NULL-PDLValue>";
    return;
  }
  switch (kind) {
  case Kind::Attribute:
    os << cast<Attribute>();
    break;
  case Kind::Operation:
    os << *cast<Operation *>();
    break;
  case Kind::Type:
    os << cast<Type>();
    break;
  case Kind::TypeRange:
    llvm::interleaveComma(cast<TypeRange>(), os);
    break;
  case Kind::Value:
    os << cast<Value>();
    break;
  case Kind::ValueRange:
    llvm::interleaveComma(cast<ValueRange>(), os);
    break;
  }
}

void PDLValue::print(raw_ostream &os, Kind kind) {
  switch (kind) {
  case Kind::Attribute:
    os << "Attribute";
    break;
  case Kind::Operation:
    os << "Operation";
    break;
  case Kind::Type:
    os << "Type";
    break;
  case Kind::TypeRange:
    os << "TypeRange";
    break;
  case Kind::Value:
    os << "Value";
    break;
  case Kind::ValueRange:
    os << "ValueRange";
    break;
  }
}

//===----------------------------------------------------------------------===//
// PDLPatternModule
//===----------------------------------------------------------------------===//

void PDLPatternModule::mergeIn(PDLPatternModule &&other) {
  // Ignore the other module if it has no patterns.
  if (!other.pdlModule)
    return;

  // Steal the functions and config of the other module.
  for (auto &it : other.constraintFunctions)
    registerConstraintFunction(it.first(), std::move(it.second));
  for (auto &it : other.rewriteFunctions)
    registerRewriteFunction(it.first(), std::move(it.second));
  for (auto &it : other.configs)
    configs.emplace_back(std::move(it));
  for (auto &it : other.configMap)
    configMap.insert(it);

  // Steal the other state if we have no patterns.
  if (!pdlModule) {
    pdlModule = std::move(other.pdlModule);
    return;
  }

  // Merge the pattern operations from the other module into this one.
  Block *block = pdlModule->getBody();
  block->getOperations().splice(block->end(),
                                other.pdlModule->getBody()->getOperations());
}

void PDLPatternModule::attachConfigToPatterns(ModuleOp module,
                                              PDLPatternConfigSet &configSet) {
  // Attach the configuration to the symbols within the module. We only add
  // to symbols to avoid hardcoding any specific operation names here (given
  // that we don't depend on any PDL dialect). We can't use
  // cast<SymbolOpInterface> here because patterns may be optional symbols.
  module->walk([&](Operation *op) {
    if (op->hasTrait<SymbolOpInterface::Trait>())
      configMap[op] = &configSet;
  });
}

//===----------------------------------------------------------------------===//
// Function Registry

void PDLPatternModule::registerConstraintFunction(
    StringRef name, PDLConstraintFunction constraintFn) {
  // TODO: Is it possible to diagnose when `name` is already registered to
  // a function that is not equivalent to `constraintFn`?
  // Allow existing mappings in the case multiple patterns depend on the same
  // constraint.
  constraintFunctions.try_emplace(name, std::move(constraintFn));
}

void PDLPatternModule::registerRewriteFunction(StringRef name,
                                               PDLRewriteFunction rewriteFn) {
  // TODO: Is it possible to diagnose when `name` is already registered to
  // a function that is not equivalent to `rewriteFn`?
  // Allow existing mappings in the case multiple patterns depend on the same
  // rewrite.
  rewriteFunctions.try_emplace(name, std::move(rewriteFn));
}
