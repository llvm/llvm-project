//===- Globals.h - MLIR Python extension globals --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BINDINGS_PYTHON_GLOBALS_H
#define MLIR_BINDINGS_PYTHON_GLOBALS_H

#include "PybindUtils.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/Support.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <optional>
#include <string>
#include <vector>

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

  /// Loads a python module corresponding to the given dialect namespace.
  /// No-ops if the module has already been loaded or is not found. Raises
  /// an error on any evaluation issues.
  /// Note that this returns void because it is expected that the module
  /// contains calls to decorators and helpers that register the salient
  /// entities. Returns true if dialect is successfully loaded.
  bool loadDialectModule(llvm::StringRef dialectNamespace);

  /// Adds a user-friendly Attribute builder.
  /// Raises an exception if the mapping already exists and replace == false.
  /// This is intended to be called by implementation code.
  void registerAttributeBuilder(const std::string &attributeKind,
                                pybind11::function pyFunc,
                                bool replace = false);

  /// Adds a user-friendly type caster. Raises an exception if the mapping
  /// already exists and replace == false. This is intended to be called by
  /// implementation code.
  void registerTypeCaster(MlirTypeID mlirTypeID, pybind11::function typeCaster,
                          bool replace = false);

  /// Adds a user-friendly value caster. Raises an exception if the mapping
  /// already exists and replace == false. This is intended to be called by
  /// implementation code.
  void registerValueCaster(MlirTypeID mlirTypeID,
                           pybind11::function valueCaster,
                           bool replace = false);

  /// Adds a concrete implementation dialect class.
  /// Raises an exception if the mapping already exists.
  /// This is intended to be called by implementation code.
  void registerDialectImpl(const std::string &dialectNamespace,
                           pybind11::object pyClass);

  /// Adds a concrete implementation operation class.
  /// Raises an exception if the mapping already exists and replace == false.
  /// This is intended to be called by implementation code.
  void registerOperationImpl(const std::string &operationName,
                             pybind11::object pyClass, bool replace = false);

  /// Returns the custom Attribute builder for Attribute kind.
  std::optional<pybind11::function>
  lookupAttributeBuilder(const std::string &attributeKind);

  /// Returns the custom type caster for MlirTypeID mlirTypeID.
  std::optional<pybind11::function> lookupTypeCaster(MlirTypeID mlirTypeID,
                                                     MlirDialect dialect);

  /// Returns the custom value caster for MlirTypeID mlirTypeID.
  std::optional<pybind11::function> lookupValueCaster(MlirTypeID mlirTypeID,
                                                      MlirDialect dialect);

  /// Looks up a registered dialect class by namespace. Note that this may
  /// trigger loading of the defining module and can arbitrarily re-enter.
  std::optional<pybind11::object>
  lookupDialectClass(const std::string &dialectNamespace);

  /// Looks up a registered operation class (deriving from OpView) by operation
  /// name. Note that this may trigger a load of the dialect, which can
  /// arbitrarily re-enter.
  std::optional<pybind11::object>
  lookupOperationClass(llvm::StringRef operationName);

private:
  static PyGlobals *instance;
  /// Module name prefixes to search under for dialect implementation modules.
  std::vector<std::string> dialectSearchPrefixes;
  /// Map of dialect namespace to external dialect class object.
  llvm::StringMap<pybind11::object> dialectClassMap;
  /// Map of full operation name to external operation class object.
  llvm::StringMap<pybind11::object> operationClassMap;
  /// Map of attribute ODS name to custom builder.
  llvm::StringMap<pybind11::object> attributeBuilderMap;
  /// Map of MlirTypeID to custom type caster.
  llvm::DenseMap<MlirTypeID, pybind11::object> typeCasterMap;
  /// Map of MlirTypeID to custom value caster.
  llvm::DenseMap<MlirTypeID, pybind11::object> valueCasterMap;
  /// Set of dialect namespaces that we have attempted to import implementation
  /// modules for.
  llvm::StringSet<> loadedDialectModules;
};

} // namespace python
} // namespace mlir

#endif // MLIR_BINDINGS_PYTHON_GLOBALS_H
