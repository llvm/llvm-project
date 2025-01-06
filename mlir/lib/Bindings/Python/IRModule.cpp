//===- IRModule.cpp - IR pybind module ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IRModule.h"

#include <optional>
#include <vector>

#include "Globals.h"
#include "NanobindUtils.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir-c/Bindings/Python/Interop.h" // This is expected after nanobind.

namespace nb = nanobind;
using namespace mlir;
using namespace mlir::python;

// -----------------------------------------------------------------------------
// PyGlobals
// -----------------------------------------------------------------------------

PyGlobals *PyGlobals::instance = nullptr;

PyGlobals::PyGlobals() {
  assert(!instance && "PyGlobals already constructed");
  instance = this;
  // The default search path include {mlir.}dialects, where {mlir.} is the
  // package prefix configured at compile time.
  dialectSearchPrefixes.emplace_back(MAKE_MLIR_PYTHON_QUALNAME("dialects"));
}

PyGlobals::~PyGlobals() { instance = nullptr; }

bool PyGlobals::loadDialectModule(llvm::StringRef dialectNamespace) {
  if (loadedDialectModules.contains(dialectNamespace))
    return true;
  // Since re-entrancy is possible, make a copy of the search prefixes.
  std::vector<std::string> localSearchPrefixes = dialectSearchPrefixes;
  nb::object loaded = nb::none();
  for (std::string moduleName : localSearchPrefixes) {
    moduleName.push_back('.');
    moduleName.append(dialectNamespace.data(), dialectNamespace.size());

    try {
      loaded = nb::module_::import_(moduleName.c_str());
    } catch (nb::python_error &e) {
      if (e.matches(PyExc_ModuleNotFoundError)) {
        continue;
      }
      throw;
    }
    break;
  }

  if (loaded.is_none())
    return false;
  // Note: Iterator cannot be shared from prior to loading, since re-entrancy
  // may have occurred, which may do anything.
  loadedDialectModules.insert(dialectNamespace);
  return true;
}

void PyGlobals::registerAttributeBuilder(const std::string &attributeKind,
                                         nb::callable pyFunc, bool replace) {
  nb::object &found = attributeBuilderMap[attributeKind];
  if (found && !replace) {
    throw std::runtime_error((llvm::Twine("Attribute builder for '") +
                              attributeKind +
                              "' is already registered with func: " +
                              nb::cast<std::string>(nb::str(found)))
                                 .str());
  }
  found = std::move(pyFunc);
}

void PyGlobals::registerTypeCaster(MlirTypeID mlirTypeID,
                                   nb::callable typeCaster, bool replace) {
  nb::object &found = typeCasterMap[mlirTypeID];
  if (found && !replace)
    throw std::runtime_error("Type caster is already registered with caster: " +
                             nb::cast<std::string>(nb::str(found)));
  found = std::move(typeCaster);
}

void PyGlobals::registerValueCaster(MlirTypeID mlirTypeID,
                                    nb::callable valueCaster, bool replace) {
  nb::object &found = valueCasterMap[mlirTypeID];
  if (found && !replace)
    throw std::runtime_error("Value caster is already registered: " +
                             nb::cast<std::string>(nb::repr(found)));
  found = std::move(valueCaster);
}

void PyGlobals::registerDialectImpl(const std::string &dialectNamespace,
                                    nb::object pyClass) {
  nb::object &found = dialectClassMap[dialectNamespace];
  if (found) {
    throw std::runtime_error((llvm::Twine("Dialect namespace '") +
                              dialectNamespace + "' is already registered.")
                                 .str());
  }
  found = std::move(pyClass);
}

void PyGlobals::registerOperationImpl(const std::string &operationName,
                                      nb::object pyClass, bool replace) {
  nb::object &found = operationClassMap[operationName];
  if (found && !replace) {
    throw std::runtime_error((llvm::Twine("Operation '") + operationName +
                              "' is already registered.")
                                 .str());
  }
  found = std::move(pyClass);
}

std::optional<nb::callable>
PyGlobals::lookupAttributeBuilder(const std::string &attributeKind) {
  const auto foundIt = attributeBuilderMap.find(attributeKind);
  if (foundIt != attributeBuilderMap.end()) {
    assert(foundIt->second && "attribute builder is defined");
    return foundIt->second;
  }
  return std::nullopt;
}

std::optional<nb::callable> PyGlobals::lookupTypeCaster(MlirTypeID mlirTypeID,
                                                        MlirDialect dialect) {
  // Try to load dialect module.
  (void)loadDialectModule(unwrap(mlirDialectGetNamespace(dialect)));
  const auto foundIt = typeCasterMap.find(mlirTypeID);
  if (foundIt != typeCasterMap.end()) {
    assert(foundIt->second && "type caster is defined");
    return foundIt->second;
  }
  return std::nullopt;
}

std::optional<nb::callable> PyGlobals::lookupValueCaster(MlirTypeID mlirTypeID,
                                                         MlirDialect dialect) {
  // Try to load dialect module.
  (void)loadDialectModule(unwrap(mlirDialectGetNamespace(dialect)));
  const auto foundIt = valueCasterMap.find(mlirTypeID);
  if (foundIt != valueCasterMap.end()) {
    assert(foundIt->second && "value caster is defined");
    return foundIt->second;
  }
  return std::nullopt;
}

std::optional<nb::object>
PyGlobals::lookupDialectClass(const std::string &dialectNamespace) {
  // Make sure dialect module is loaded.
  if (!loadDialectModule(dialectNamespace))
    return std::nullopt;
  const auto foundIt = dialectClassMap.find(dialectNamespace);
  if (foundIt != dialectClassMap.end()) {
    assert(foundIt->second && "dialect class is defined");
    return foundIt->second;
  }
  // Not found and loading did not yield a registration.
  return std::nullopt;
}

std::optional<nb::object>
PyGlobals::lookupOperationClass(llvm::StringRef operationName) {
  // Make sure dialect module is loaded.
  auto split = operationName.split('.');
  llvm::StringRef dialectNamespace = split.first;
  if (!loadDialectModule(dialectNamespace))
    return std::nullopt;

  auto foundIt = operationClassMap.find(operationName);
  if (foundIt != operationClassMap.end()) {
    assert(foundIt->second && "OpView is defined");
    return foundIt->second;
  }
  // Not found and loading did not yield a registration.
  return std::nullopt;
}
