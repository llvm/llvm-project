//===- Globals.h - MLIR Python extension globals --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BINDINGS_PYTHON_GLOBALS_H
#define MLIR_BINDINGS_PYTHON_GLOBALS_H

#include <optional>
#include <regex>
#include <string>
#include <unordered_set>
#include <vector>

#include "NanobindUtils.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/Support.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Regex.h"

namespace mlir {
namespace python {

/// Globals that are always accessible once the extension has been initialized.
/// Methods of this class are thread-safe.
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
  std::vector<std::string> getDialectSearchPrefixes() {
    nanobind::ft_lock_guard lock(mutex);
    return dialectSearchPrefixes;
  }
  void setDialectSearchPrefixes(std::vector<std::string> newValues) {
    nanobind::ft_lock_guard lock(mutex);
    dialectSearchPrefixes.swap(newValues);
  }
  void addDialectSearchPrefix(std::string value) {
    nanobind::ft_lock_guard lock(mutex);
    dialectSearchPrefixes.push_back(std::move(value));
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
                                nanobind::callable pyFunc,
                                bool replace = false);

  /// Adds a user-friendly type caster. Raises an exception if the mapping
  /// already exists and replace == false. This is intended to be called by
  /// implementation code.
  void registerTypeCaster(MlirTypeID mlirTypeID, nanobind::callable typeCaster,
                          bool replace = false);

  /// Adds a user-friendly value caster. Raises an exception if the mapping
  /// already exists and replace == false. This is intended to be called by
  /// implementation code.
  void registerValueCaster(MlirTypeID mlirTypeID,
                           nanobind::callable valueCaster,
                           bool replace = false);

  /// Adds a concrete implementation dialect class.
  /// Raises an exception if the mapping already exists.
  /// This is intended to be called by implementation code.
  void registerDialectImpl(const std::string &dialectNamespace,
                           nanobind::object pyClass);

  /// Adds a concrete implementation operation class.
  /// Raises an exception if the mapping already exists and replace == false.
  /// This is intended to be called by implementation code.
  void registerOperationImpl(const std::string &operationName,
                             nanobind::object pyClass, bool replace = false);

  /// Returns the custom Attribute builder for Attribute kind.
  std::optional<nanobind::callable>
  lookupAttributeBuilder(const std::string &attributeKind);

  /// Returns the custom type caster for MlirTypeID mlirTypeID.
  std::optional<nanobind::callable> lookupTypeCaster(MlirTypeID mlirTypeID,
                                                     MlirDialect dialect);

  /// Returns the custom value caster for MlirTypeID mlirTypeID.
  std::optional<nanobind::callable> lookupValueCaster(MlirTypeID mlirTypeID,
                                                      MlirDialect dialect);

  /// Looks up a registered dialect class by namespace. Note that this may
  /// trigger loading of the defining module and can arbitrarily re-enter.
  std::optional<nanobind::object>
  lookupDialectClass(const std::string &dialectNamespace);

  /// Looks up a registered operation class (deriving from OpView) by operation
  /// name. Note that this may trigger a load of the dialect, which can
  /// arbitrarily re-enter.
  std::optional<nanobind::object>
  lookupOperationClass(llvm::StringRef operationName);

  class TracebackLoc {
  public:
    bool locTracebacksEnabled();

    void setLocTracebacksEnabled(bool value);

    size_t locTracebackFramesLimit();

    void setLocTracebackFramesLimit(size_t value);

    void registerTracebackFileInclusion(const std::string &file);

    void registerTracebackFileExclusion(const std::string &file);

    bool isUserTracebackFilename(llvm::StringRef file);

    static constexpr size_t kMaxFrames = 512;

  private:
    nanobind::ft_mutex mutex;
    bool locTracebackEnabled_ = false;
    size_t locTracebackFramesLimit_ = 10;
    std::unordered_set<std::string> userTracebackIncludeFiles;
    std::unordered_set<std::string> userTracebackExcludeFiles;
    std::regex userTracebackIncludeRegex;
    bool rebuildUserTracebackIncludeRegex = false;
    std::regex userTracebackExcludeRegex;
    bool rebuildUserTracebackExcludeRegex = false;
    llvm::StringMap<bool> isUserTracebackFilenameCache;
  };

  TracebackLoc &getTracebackLoc() { return tracebackLoc; }

  class TypeIDAllocator {
  public:
    TypeIDAllocator() : allocator(mlirTypeIDAllocatorCreate()) {}
    ~TypeIDAllocator() {
      if (allocator.ptr)
        mlirTypeIDAllocatorDestroy(allocator);
    }
    TypeIDAllocator(const TypeIDAllocator &) = delete;
    TypeIDAllocator(TypeIDAllocator &&other) : allocator(other.allocator) {
      other.allocator.ptr = nullptr;
    }

    MlirTypeIDAllocator get() { return allocator; }
    MlirTypeID allocate() {
      return mlirTypeIDAllocatorAllocateTypeID(allocator);
    }

  private:
    MlirTypeIDAllocator allocator;
  };

  MlirTypeID allocateTypeID() { return typeIDAllocator.allocate(); }

private:
  static PyGlobals *instance;

  nanobind::ft_mutex mutex;

  /// Module name prefixes to search under for dialect implementation modules.
  std::vector<std::string> dialectSearchPrefixes;
  /// Map of dialect namespace to external dialect class object.
  llvm::StringMap<nanobind::object> dialectClassMap;
  /// Map of full operation name to external operation class object.
  llvm::StringMap<nanobind::object> operationClassMap;
  /// Map of attribute ODS name to custom builder.
  llvm::StringMap<nanobind::callable> attributeBuilderMap;
  /// Map of MlirTypeID to custom type caster.
  llvm::DenseMap<MlirTypeID, nanobind::callable> typeCasterMap;
  /// Map of MlirTypeID to custom value caster.
  llvm::DenseMap<MlirTypeID, nanobind::callable> valueCasterMap;
  /// Set of dialect namespaces that we have attempted to import implementation
  /// modules for.
  llvm::StringSet<> loadedDialectModules;

  TracebackLoc tracebackLoc;
  TypeIDAllocator typeIDAllocator;
};

} // namespace python
} // namespace mlir

#endif // MLIR_BINDINGS_PYTHON_GLOBALS_H
