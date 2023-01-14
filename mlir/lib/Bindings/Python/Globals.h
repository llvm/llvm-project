//===- Globals.h - MLIR Python extension globals --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BINDINGS_PYTHON_GLOBALS_H
#define MLIR_BINDINGS_PYTHON_GLOBALS_H

#include <string>
#include <vector>
#include <optional>

#include "PybindUtils.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

namespace mlir {
namespace python {

/// Globals that are always accessible once the extension has been initialized.
class PyGlobals {
public:
  PyGlobals();
  ~PyGlobals();

  /// Most code should get the globals via this static accessor.
  static PyGlobals &get() {
    assert(instance && "PyGlobals is null");
    return *instance;
  }

  /// Get and set the list of parent modules to search for dialect
  /// implementation classes.
  std::vector<std::string> &getDialectSearchPrefixes() {
    return dialectSearchPrefixes;
  }
  void setDialectSearchPrefixes(std::vector<std::string> newValues) {
    dialectSearchPrefixes.swap(newValues);
  }

  /// Clears positive and negative caches regarding what implementations are
  /// available. Future lookups will do more expensive existence checks.
  void clearImportCache();

  /// Loads a python module corresponding to the given dialect namespace.
  /// No-ops if the module has already been loaded or is not found. Raises
  /// an error on any evaluation issues.
  /// Note that this returns void because it is expected that the module
  /// contains calls to decorators and helpers that register the salient
  /// entities.
  void loadDialectModule(llvm::StringRef dialectNamespace);

  /// Decorator for registering a custom Dialect class. The class object must
  /// have a DIALECT_NAMESPACE attribute.
  pybind11::object registerDialectDecorator(pybind11::object pyClass);

  /// Adds a user-friendly Attribute builder.
  /// Raises an exception if the mapping already exists.
  /// This is intended to be called by implementation code.
  void registerAttributeBuilder(const std::string &attributeKind,
                                pybind11::function pyFunc);

  /// Adds a concrete implementation dialect class.
  /// Raises an exception if the mapping already exists.
  /// This is intended to be called by implementation code.
  void registerDialectImpl(const std::string &dialectNamespace,
                           pybind11::object pyClass);

  /// Adds a concrete implementation operation class.
  /// Raises an exception if the mapping already exists.
  /// This is intended to be called by implementation code.
  void registerOperationImpl(const std::string &operationName,
                             pybind11::object pyClass,
                             pybind11::object rawOpViewClass);

  /// Returns the custom Attribute builder for Attribute kind.
  std::optional<pybind11::function>
  lookupAttributeBuilder(const std::string &attributeKind);

  /// Looks up a registered dialect class by namespace. Note that this may
  /// trigger loading of the defining module and can arbitrarily re-enter.
  std::optional<pybind11::object>
  lookupDialectClass(const std::string &dialectNamespace);

  /// Looks up a registered raw OpView class by operation name. Note that this
  /// may trigger a load of the dialect, which can arbitrarily re-enter.
  std::optional<pybind11::object>
  lookupRawOpViewClass(llvm::StringRef operationName);

private:
  static PyGlobals *instance;
  /// Module name prefixes to search under for dialect implementation modules.
  std::vector<std::string> dialectSearchPrefixes;
  /// Map of dialect namespace to external dialect class object.
  llvm::StringMap<pybind11::object> dialectClassMap;
  /// Map of full operation name to external operation class object.
  llvm::StringMap<pybind11::object> operationClassMap;
  /// Map of operation name to custom subclass that directly initializes
  /// the OpView base class (bypassing the user class constructor).
  llvm::StringMap<pybind11::object> rawOpViewClassMap;
  /// Map of attribute ODS name to custom builder.
  llvm::StringMap<pybind11::object> attributeBuilderMap;

  /// Set of dialect namespaces that we have attempted to import implementation
  /// modules for.
  llvm::StringSet<> loadedDialectModulesCache;
  /// Cache of operation name to custom OpView subclass that directly
  /// initializes the OpView base class (or an undefined object for negative
  /// lookup). This is maintained on loopup as a shadow of rawOpViewClassMap
  /// in order for repeat lookups of the OpView classes to only incur the cost
  /// of one hashtable lookup.
  llvm::StringMap<pybind11::object> rawOpViewClassMapCache;
};

} // namespace python
} // namespace mlir

#endif // MLIR_BINDINGS_PYTHON_GLOBALS_H
