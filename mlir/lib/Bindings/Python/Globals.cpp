//===- IRModule.cpp - IR pybind module ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Bindings/Python/IRCore.h"

#include <cstring>
#include <optional>
#include <sstream>
#include <string_view>
#include <vector>

#include "mlir/Bindings/Python/Globals.h"
// clang-format off
#include "mlir/Bindings/Python/NanobindUtils.h"
#include "mlir-c/Bindings/Python/Interop.h"
// clang-format on
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/Nanobind.h"

namespace nb = nanobind;
using namespace mlir;

/// Local helper adapted from llvm::Regex::escape.
static std::string escapeRegex(std::string_view String) {
  static constexpr char RegexMetachars[] = "()^$|*+?.[]\\{}";
  std::string RegexStr;
  for (char C : String) {
    if (std::strchr(RegexMetachars, C))
      RegexStr += '\\';
    RegexStr += C;
  }
  return RegexStr;
}

// -----------------------------------------------------------------------------
// PyGlobals
// -----------------------------------------------------------------------------

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
PyGlobals *PyGlobals::instance = nullptr;

PyGlobals::PyGlobals() {
  assert(!instance && "PyGlobals already constructed");
  instance = this;
  // The default search path include {mlir.}dialects, where {mlir.} is the
  // package prefix configured at compile time.
  dialectSearchPrefixes.emplace_back(MAKE_MLIR_PYTHON_QUALNAME("dialects"));
}

PyGlobals::~PyGlobals() { instance = nullptr; }

PyGlobals &PyGlobals::get() {
  assert(instance && "PyGlobals is null");
  return *instance;
}

bool PyGlobals::loadDialectModule(std::string_view dialectNamespace) {
  {
    nb::ft_lock_guard lock(mutex);
    std::string dialectNamespaceStr(dialectNamespace);
    if (loadedDialectModules.find(dialectNamespaceStr) !=
        loadedDialectModules.end())
      return true;
  }
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
  nb::ft_lock_guard lock(mutex);
  loadedDialectModules.insert(std::string(dialectNamespace));
  return true;
}

void PyGlobals::registerAttributeBuilder(const std::string &attributeKind,
                                         nb::callable pyFunc, bool replace) {
  nb::ft_lock_guard lock(mutex);
  nb::object &found = attributeBuilderMap[attributeKind];
  if (found && !replace) {
    throw std::runtime_error(
        nanobind::detail::join("Attribute builder for '", attributeKind,
                               "' is already registered with func: ",
                               nb::cast<std::string>(nb::str(found))));
  }
  found = std::move(pyFunc);
}

void PyGlobals::registerTypeCaster(MlirTypeID mlirTypeID,
                                   nb::callable typeCaster, bool replace) {
  nb::ft_lock_guard lock(mutex);
  nb::object &found = typeCasterMap[mlirTypeID];
  if (found && !replace)
    throw std::runtime_error("Type caster is already registered with caster: " +
                             nb::cast<std::string>(nb::str(found)));
  found = std::move(typeCaster);
}

void PyGlobals::registerValueCaster(MlirTypeID mlirTypeID,
                                    nb::callable valueCaster, bool replace) {
  nb::ft_lock_guard lock(mutex);
  nb::object &found = valueCasterMap[mlirTypeID];
  if (found && !replace)
    throw std::runtime_error("Value caster is already registered: " +
                             nb::cast<std::string>(nb::repr(found)));
  found = std::move(valueCaster);
}

void PyGlobals::registerDialectImpl(const std::string &dialectNamespace,
                                    nb::object pyClass) {
  nb::ft_lock_guard lock(mutex);
  nb::object &found = dialectClassMap[dialectNamespace];
  if (found) {
    throw std::runtime_error(nanobind::detail::join(
        "Dialect namespace '", dialectNamespace, "' is already registered."));
  }
  found = std::move(pyClass);
}

void PyGlobals::registerOperationImpl(const std::string &operationName,
                                      nb::object pyClass, bool replace) {
  nb::ft_lock_guard lock(mutex);
  nb::object &found = operationClassMap[operationName];
  if (found && !replace) {
    throw std::runtime_error(nanobind::detail::join(
        "Operation '", operationName, "' is already registered."));
  }
  found = std::move(pyClass);
}

void PyGlobals::registerOpAdaptorImpl(const std::string &operationName,
                                      nb::object pyClass, bool replace) {
  nb::ft_lock_guard lock(mutex);
  nb::object &found = opAdaptorClassMap[operationName];
  if (found && !replace) {
    throw std::runtime_error(nanobind::detail::join(
        "Operation adaptor of '", operationName, "' is already registered."));
  }
  found = std::move(pyClass);
}

std::optional<nb::callable>
PyGlobals::lookupAttributeBuilder(const std::string &attributeKind) {
  nb::ft_lock_guard lock(mutex);
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
  MlirStringRef ns = mlirDialectGetNamespace(dialect);
  (void)loadDialectModule(std::string_view(ns.data, ns.length));
  nb::ft_lock_guard lock(mutex);
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
  MlirStringRef ns = mlirDialectGetNamespace(dialect);
  (void)loadDialectModule(std::string_view(ns.data, ns.length));
  nb::ft_lock_guard lock(mutex);
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
  (void)loadDialectModule(dialectNamespace);

  nb::ft_lock_guard lock(mutex);
  const auto foundIt = dialectClassMap.find(dialectNamespace);
  if (foundIt != dialectClassMap.end()) {
    assert(foundIt->second && "dialect class is defined");
    return foundIt->second;
  }
  // Not found and loading did not yield a registration.
  return std::nullopt;
}

std::optional<nb::object>
PyGlobals::lookupOperationClass(std::string_view operationName) {
  // Make sure dialect module is loaded.
  std::string_view dialectNamespace =
      operationName.substr(0, operationName.find('.'));
  (void)loadDialectModule(dialectNamespace);

  nb::ft_lock_guard lock(mutex);
  std::string operationNameStr(operationName);
  auto foundIt = operationClassMap.find(operationNameStr);
  if (foundIt != operationClassMap.end()) {
    assert(foundIt->second && "OpView is defined");
    return foundIt->second;
  }
  // Not found and loading did not yield a registration.
  return std::nullopt;
}

std::optional<nb::object>
PyGlobals::lookupOpAdaptorClass(std::string_view operationName) {
  // Make sure dialect module is loaded.
  std::string_view dialectNamespace =
      operationName.substr(0, operationName.find('.'));
  (void)loadDialectModule(dialectNamespace);

  nb::ft_lock_guard lock(mutex);
  std::string operationNameStr(operationName);
  auto foundIt = opAdaptorClassMap.find(operationNameStr);
  if (foundIt != opAdaptorClassMap.end()) {
    assert(foundIt->second && "OpAdaptor is defined");
    return foundIt->second;
  }
  // Not found and loading did not yield a registration.
  return std::nullopt;
}

bool PyGlobals::TracebackLoc::locTracebacksEnabled() {
  nanobind::ft_lock_guard lock(mutex);
  return locTracebackEnabled_;
}

void PyGlobals::TracebackLoc::setLocTracebacksEnabled(bool value) {
  nanobind::ft_lock_guard lock(mutex);
  locTracebackEnabled_ = value;
}

size_t PyGlobals::TracebackLoc::locTracebackFramesLimit() {
  nanobind::ft_lock_guard lock(mutex);
  return locTracebackFramesLimit_;
}

void PyGlobals::TracebackLoc::setLocTracebackFramesLimit(size_t value) {
  nanobind::ft_lock_guard lock(mutex);
  locTracebackFramesLimit_ = std::min(value, kMaxFrames);
}

void PyGlobals::TracebackLoc::registerTracebackFileInclusion(
    const std::string &file) {
  nanobind::ft_lock_guard lock(mutex);
  auto reg = "^" + escapeRegex(file);
  if (userTracebackIncludeFiles.insert(reg).second)
    rebuildUserTracebackIncludeRegex = true;
  if (userTracebackExcludeFiles.count(reg)) {
    if (userTracebackExcludeFiles.erase(reg))
      rebuildUserTracebackExcludeRegex = true;
  }
}

void PyGlobals::TracebackLoc::registerTracebackFileExclusion(
    const std::string &file) {
  nanobind::ft_lock_guard lock(mutex);
  auto reg = "^" + escapeRegex(file);
  if (userTracebackExcludeFiles.insert(reg).second)
    rebuildUserTracebackExcludeRegex = true;
  if (userTracebackIncludeFiles.count(reg)) {
    if (userTracebackIncludeFiles.erase(reg))
      rebuildUserTracebackIncludeRegex = true;
  }
}

bool PyGlobals::TracebackLoc::isUserTracebackFilename(
    const std::string_view file) {
  nanobind::ft_lock_guard lock(mutex);
  auto joinWithPipe = [](const std::unordered_set<std::string> &set) {
    std::ostringstream os;
    for (auto it = set.begin(); it != set.end(); ++it) {
      if (it != set.begin())
        os << "|";
      os << *it;
    }
    return os.str();
  };
  if (rebuildUserTracebackIncludeRegex) {
    userTracebackIncludeRegex.assign(joinWithPipe(userTracebackIncludeFiles));
    rebuildUserTracebackIncludeRegex = false;
    isUserTracebackFilenameCache.clear();
  }
  if (rebuildUserTracebackExcludeRegex) {
    userTracebackExcludeRegex.assign(joinWithPipe(userTracebackExcludeFiles));
    rebuildUserTracebackExcludeRegex = false;
    isUserTracebackFilenameCache.clear();
  }
  std::string fileStr(file);
  const auto foundIt = isUserTracebackFilenameCache.find(fileStr);
  if (foundIt == isUserTracebackFilenameCache.end()) {
    bool include = std::regex_search(fileStr, userTracebackIncludeRegex);
    bool exclude = std::regex_search(fileStr, userTracebackExcludeRegex);
    isUserTracebackFilenameCache[fileStr] = include || !exclude;
  }
  return isUserTracebackFilenameCache[fileStr];
}
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir
