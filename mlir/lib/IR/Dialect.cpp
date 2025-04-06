//===- Dialect.cpp - Dialect implementation -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Regex.h"
#include <memory>

#define DEBUG_TYPE "dialect"

using namespace mlir;
using namespace detail;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

Dialect::Dialect(StringRef name, MLIRContext *context, TypeID id)
    : name(name), dialectID(id), context(context) {
  assert(isValidNamespace(name) && "invalid dialect namespace");
}

Dialect::~Dialect() = default;

/// Verify an attribute from this dialect on the argument at 'argIndex' for
/// the region at 'regionIndex' on the given operation. Returns failure if
/// the verification failed, success otherwise. This hook may optionally be
/// invoked from any operation containing a region.
LogicalResult Dialect::verifyRegionArgAttribute(Operation *, unsigned, unsigned,
                                                NamedAttribute) {
  return success();
}

/// Verify an attribute from this dialect on the result at 'resultIndex' for
/// the region at 'regionIndex' on the given operation. Returns failure if
/// the verification failed, success otherwise. This hook may optionally be
/// invoked from any operation containing a region.
LogicalResult Dialect::verifyRegionResultAttribute(Operation *, unsigned,
                                                   unsigned, NamedAttribute) {
  return success();
}

/// Parse an attribute registered to this dialect.
Attribute Dialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  parser.emitError(parser.getNameLoc())
      << "dialect '" << getNamespace()
      << "' provides no attribute parsing hook";
  return Attribute();
}

/// Parse a type registered to this dialect.
Type Dialect::parseType(DialectAsmParser &parser) const {
  // If this dialect allows unknown types, then represent this with OpaqueType.
  if (allowsUnknownTypes()) {
    StringAttr ns = StringAttr::get(getContext(), getNamespace());
    return OpaqueType::get(ns, parser.getFullSymbolSpec());
  }

  parser.emitError(parser.getNameLoc())
      << "dialect '" << getNamespace() << "' provides no type parsing hook";
  return Type();
}

std::optional<Dialect::ParseOpHook>
Dialect::getParseOperationHook(StringRef opName) const {
  return std::nullopt;
}

llvm::unique_function<void(Operation *, OpAsmPrinter &printer)>
Dialect::getOperationPrinter(Operation *op) const {
  assert(op->getDialect() == this &&
         "Dialect hook invoked on non-dialect owned operation");
  return nullptr;
}

/// Utility function that returns if the given string is a valid dialect
/// namespace
bool Dialect::isValidNamespace(StringRef str) {
  llvm::Regex dialectNameRegex("^[a-zA-Z_][a-zA-Z_0-9\\$]*$");
  return dialectNameRegex.match(str);
}

/// Register a set of dialect interfaces with this dialect instance.
void Dialect::addInterface(std::unique_ptr<DialectInterface> interface) {
  // Handle the case where the models resolve a promised interface.
  handleAdditionOfUndefinedPromisedInterface(getTypeID(), interface->getID());

  auto it = registeredInterfaces.try_emplace(interface->getID(),
                                             std::move(interface));
  (void)it;
  LLVM_DEBUG({
    if (!it.second) {
      llvm::dbgs() << "[" DEBUG_TYPE
                      "] repeated interface registration for dialect "
                   << getNamespace();
    }
  });
}

//===----------------------------------------------------------------------===//
// Dialect Interface
//===----------------------------------------------------------------------===//

DialectInterface::~DialectInterface() = default;

MLIRContext *DialectInterface::getContext() const {
  return dialect->getContext();
}

DialectInterfaceCollectionBase::DialectInterfaceCollectionBase(
    MLIRContext *ctx, TypeID interfaceKind, StringRef interfaceName) {
  for (auto *dialect : ctx->getLoadedDialects()) {
#ifndef NDEBUG
    dialect->handleUseOfUndefinedPromisedInterface(
        dialect->getTypeID(), interfaceKind, interfaceName);
#endif
    if (auto *interface = dialect->getRegisteredInterface(interfaceKind)) {
      interfaces.insert(interface);
      orderedInterfaces.push_back(interface);
    }
  }
}

DialectInterfaceCollectionBase::~DialectInterfaceCollectionBase() = default;

/// Get the interface for the dialect of given operation, or null if one
/// is not registered.
const DialectInterface *
DialectInterfaceCollectionBase::getInterfaceFor(Operation *op) const {
  return getInterfaceFor(op->getDialect());
}

//===----------------------------------------------------------------------===//
// DialectExtension
//===----------------------------------------------------------------------===//

DialectExtensionBase::~DialectExtensionBase() = default;

void dialect_extension_detail::handleUseOfUndefinedPromisedInterface(
    Dialect &dialect, TypeID interfaceRequestorID, TypeID interfaceID,
    StringRef interfaceName) {
  dialect.handleUseOfUndefinedPromisedInterface(interfaceRequestorID,
                                                interfaceID, interfaceName);
}

void dialect_extension_detail::handleAdditionOfUndefinedPromisedInterface(
    Dialect &dialect, TypeID interfaceRequestorID, TypeID interfaceID) {
  dialect.handleAdditionOfUndefinedPromisedInterface(interfaceRequestorID,
                                                     interfaceID);
}

bool dialect_extension_detail::hasPromisedInterface(Dialect &dialect,
                                                    TypeID interfaceRequestorID,
                                                    TypeID interfaceID) {
  return dialect.hasPromisedInterface(interfaceRequestorID, interfaceID);
}

//===----------------------------------------------------------------------===//
// DialectRegistry
//===----------------------------------------------------------------------===//

namespace {
template <typename Fn>
void applyExtensionsFn(
    Fn &&applyExtension,
    const llvm::MapVector<TypeID, std::unique_ptr<DialectExtensionBase>>
        &extensions) {
  // Note: Additional extensions may be added while applying an extension.
  // The iterators will be invalidated if extensions are added so we'll keep
  // a copy of the extensions for ourselves.

  const auto extractExtension =
      [](const auto &entry) -> DialectExtensionBase * {
    return entry.second.get();
  };

  auto startIt = extensions.begin(), endIt = extensions.end();
  size_t count = 0;
  while (startIt != endIt) {
    count += endIt - startIt;

    // Grab the subset of extensions we'll apply in this iteration.
    const auto subset =
        llvm::map_to_vector(llvm::make_range(startIt, endIt), extractExtension);

    for (const auto *ext : subset)
      applyExtension(*ext);

    // Book-keep for the next iteration.
    startIt = extensions.begin() + count;
    endIt = extensions.end();
  }
}
} // namespace

DialectRegistry::DialectRegistry() { insert<BuiltinDialect>(); }

DialectAllocatorFunctionRef
DialectRegistry::getDialectAllocator(StringRef name) const {
  auto it = registry.find(name);
  if (it == registry.end())
    return nullptr;
  return it->second.second;
}

void DialectRegistry::insert(TypeID typeID, StringRef name,
                             const DialectAllocatorFunction &ctor) {
  auto inserted = registry.insert(
      std::make_pair(std::string(name), std::make_pair(typeID, ctor)));
  if (!inserted.second && inserted.first->second.first != typeID) {
    llvm::report_fatal_error(
        "Trying to register different dialects for the same namespace: " +
        name);
  }
}

void DialectRegistry::insertDynamic(
    StringRef name, const DynamicDialectPopulationFunction &ctor) {
  // This TypeID marks dynamic dialects. We cannot give a TypeID for the
  // dialect yet, since the TypeID of a dynamic dialect is defined at its
  // construction.
  TypeID typeID = TypeID::get<void>();

  // Create the dialect, and then call ctor, which allocates its components.
  auto constructor = [nameStr = name.str(), ctor](MLIRContext *ctx) {
    auto *dynDialect = ctx->getOrLoadDynamicDialect(
        nameStr, [ctx, ctor](DynamicDialect *dialect) { ctor(ctx, dialect); });
    assert(dynDialect && "Dynamic dialect creation unexpectedly failed");
    return dynDialect;
  };

  insert(typeID, name, constructor);
}

void DialectRegistry::applyExtensions(Dialect *dialect) const {
  MLIRContext *ctx = dialect->getContext();
  StringRef dialectName = dialect->getNamespace();

  // Functor used to try to apply the given extension.
  auto applyExtension = [&](const DialectExtensionBase &extension) {
    ArrayRef<StringRef> dialectNames = extension.getRequiredDialects();
    // An empty set is equivalent to always invoke.
    if (dialectNames.empty()) {
      extension.apply(ctx, dialect);
      return;
    }

    // Handle the simple case of a single dialect name. In this case, the
    // required dialect should be the current dialect.
    if (dialectNames.size() == 1) {
      if (dialectNames.front() == dialectName)
        extension.apply(ctx, dialect);
      return;
    }

    // Otherwise, check to see if this extension requires this dialect.
    const StringRef *nameIt = llvm::find(dialectNames, dialectName);
    if (nameIt == dialectNames.end())
      return;

    // If it does, ensure that all of the other required dialects have been
    // loaded.
    SmallVector<Dialect *> requiredDialects;
    requiredDialects.reserve(dialectNames.size());
    for (auto it = dialectNames.begin(), e = dialectNames.end(); it != e;
         ++it) {
      // The current dialect is known to be loaded.
      if (it == nameIt) {
        requiredDialects.push_back(dialect);
        continue;
      }
      // Otherwise, check if it is loaded.
      Dialect *loadedDialect = ctx->getLoadedDialect(*it);
      if (!loadedDialect)
        return;
      requiredDialects.push_back(loadedDialect);
    }
    extension.apply(ctx, requiredDialects);
  };

  applyExtensionsFn(applyExtension, extensions);
}

void DialectRegistry::applyExtensions(MLIRContext *ctx) const {
  // Functor used to try to apply the given extension.
  auto applyExtension = [&](const DialectExtensionBase &extension) {
    ArrayRef<StringRef> dialectNames = extension.getRequiredDialects();
    if (dialectNames.empty()) {
      auto loadedDialects = ctx->getLoadedDialects();
      extension.apply(ctx, loadedDialects);
      return;
    }

    // Check to see if all of the dialects for this extension are loaded.
    SmallVector<Dialect *> requiredDialects;
    requiredDialects.reserve(dialectNames.size());
    for (StringRef dialectName : dialectNames) {
      Dialect *loadedDialect = ctx->getLoadedDialect(dialectName);
      if (!loadedDialect)
        return;
      requiredDialects.push_back(loadedDialect);
    }
    extension.apply(ctx, requiredDialects);
  };

  applyExtensionsFn(applyExtension, extensions);
}

bool DialectRegistry::isSubsetOf(const DialectRegistry &rhs) const {
  // Check that all extension keys are present in 'rhs'.
  const auto hasExtension = [&](const auto &key) {
    return rhs.extensions.contains(key);
  };
  if (!llvm::all_of(make_first_range(extensions), hasExtension))
    return false;

  // Check that the current dialects fully overlap with the dialects in 'rhs'.
  return llvm::all_of(
      registry, [&](const auto &it) { return rhs.registry.count(it.first); });
}
