//===- IRModule.cpp - IR pybind module ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IRModule.h"
#include "Globals.h"
#include "PybindUtils.h"

#include <optional>
#include <vector>

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Support.h"

namespace py = pybind11;
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

void PyGlobals::loadDialectModule(llvm::StringRef dialectNamespace) {
  if (loadedDialectModulesCache.contains(dialectNamespace))
    return;
  // Since re-entrancy is possible, make a copy of the search prefixes.
  std::vector<std::string> localSearchPrefixes = dialectSearchPrefixes;
  py::object loaded;
  for (std::string moduleName : localSearchPrefixes) {
    moduleName.push_back('.');
    moduleName.append(dialectNamespace.data(), dialectNamespace.size());

    try {
      loaded = py::module::import(moduleName.c_str());
    } catch (py::error_already_set &e) {
      if (e.matches(PyExc_ModuleNotFoundError)) {
        continue;
      }
      throw;
    }
    break;
  }

  // Note: Iterator cannot be shared from prior to loading, since re-entrancy
  // may have occurred, which may do anything.
  loadedDialectModulesCache.insert(dialectNamespace);
}

void PyGlobals::registerAttributeBuilder(const std::string &attributeKind,
                                         py::function pyFunc) {
  py::object &found = attributeBuilderMap[attributeKind];
  if (found) {
    throw std::runtime_error((llvm::Twine("Attribute builder for '") +
                              attributeKind + "' is already registered")
                                 .str());
  }
  found = std::move(pyFunc);
}

void PyGlobals::registerTypeCaster(MlirTypeID mlirTypeID,
                                   pybind11::function typeCaster,
                                   bool replace) {
  pybind11::object &found = typeCasterMap[mlirTypeID];
  if (found && !found.is_none() && !replace)
    throw std::runtime_error("Type caster is already registered");
  found = std::move(typeCaster);
}

void PyGlobals::registerDialectImpl(const std::string &dialectNamespace,
                                    py::object pyClass) {
  py::object &found = dialectClassMap[dialectNamespace];
  if (found) {
    throw std::runtime_error((llvm::Twine("Dialect namespace '") +
                              dialectNamespace + "' is already registered.")
                                 .str());
  }
  found = std::move(pyClass);
}

void PyGlobals::registerOperationImpl(const std::string &operationName,
                                      py::object pyClass) {
  py::object &found = operationClassMap[operationName];
  if (found) {
    throw std::runtime_error((llvm::Twine("Operation '") + operationName +
                              "' is already registered.")
                                 .str());
  }
  found = std::move(pyClass);
}

std::optional<py::function>
PyGlobals::lookupAttributeBuilder(const std::string &attributeKind) {
  // Fast match against the class map first (common case).
  const auto foundIt = attributeBuilderMap.find(attributeKind);
  if (foundIt != attributeBuilderMap.end()) {
    if (foundIt->second.is_none())
      return std::nullopt;
    assert(foundIt->second && "py::function is defined");
    return foundIt->second;
  }

  // Not found and loading did not yield a registration. Negative cache.
  attributeBuilderMap[attributeKind] = py::none();
  return std::nullopt;
}

std::optional<py::function> PyGlobals::lookupTypeCaster(MlirTypeID mlirTypeID,
                                                        MlirDialect dialect) {
  {
    // Fast match against the class map first (common case).
    const auto foundIt = typeCasterMapCache.find(mlirTypeID);
    if (foundIt != typeCasterMapCache.end()) {
      if (foundIt->second.is_none())
        return std::nullopt;
      assert(foundIt->second && "py::function is defined");
      return foundIt->second;
    }
  }

  // Not found. Load the dialect namespace.
  loadDialectModule(unwrap(mlirDialectGetNamespace(dialect)));

  // Attempt to find from the canonical map and cache.
  {
    const auto foundIt = typeCasterMap.find(mlirTypeID);
    if (foundIt != typeCasterMap.end()) {
      if (foundIt->second.is_none())
        return std::nullopt;
      assert(foundIt->second && "py::object is defined");
      // Positive cache.
      typeCasterMapCache[mlirTypeID] = foundIt->second;
      return foundIt->second;
    }
    // Negative cache.
    typeCasterMap[mlirTypeID] = py::none();
    return std::nullopt;
  }
}

std::optional<py::object>
PyGlobals::lookupDialectClass(const std::string &dialectNamespace) {
  loadDialectModule(dialectNamespace);
  // Fast match against the class map first (common case).
  const auto foundIt = dialectClassMap.find(dialectNamespace);
  if (foundIt != dialectClassMap.end()) {
    if (foundIt->second.is_none())
      return std::nullopt;
    assert(foundIt->second && "py::object is defined");
    return foundIt->second;
  }

  // Not found and loading did not yield a registration. Negative cache.
  dialectClassMap[dialectNamespace] = py::none();
  return std::nullopt;
}

std::optional<pybind11::object>
PyGlobals::lookupOperationClass(llvm::StringRef operationName) {
  {
    auto foundIt = operationClassMapCache.find(operationName);
    if (foundIt != operationClassMapCache.end()) {
      if (foundIt->second.is_none())
        return std::nullopt;
      assert(foundIt->second && "py::object is defined");
      return foundIt->second;
    }
  }

  // Not found. Load the dialect namespace.
  auto split = operationName.split('.');
  llvm::StringRef dialectNamespace = split.first;
  loadDialectModule(dialectNamespace);

  // Attempt to find from the canonical map and cache.
  {
    auto foundIt = operationClassMap.find(operationName);
    if (foundIt != operationClassMap.end()) {
      if (foundIt->second.is_none())
        return std::nullopt;
      assert(foundIt->second && "py::object is defined");
      // Positive cache.
      operationClassMapCache[operationName] = foundIt->second;
      return foundIt->second;
    }
    // Negative cache.
    operationClassMap[operationName] = py::none();
    return std::nullopt;
  }
}

void PyGlobals::clearImportCache() {
  loadedDialectModulesCache.clear();
  operationClassMapCache.clear();
  typeCasterMapCache.clear();
}
