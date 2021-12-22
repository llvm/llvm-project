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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Regex.h"

#define DEBUG_TYPE "dialect"

using namespace mlir;
using namespace detail;

//===----------------------------------------------------------------------===//
// DialectRegistry
//===----------------------------------------------------------------------===//

DialectRegistry::DialectRegistry() { insert<BuiltinDialect>(); }

void DialectRegistry::addDialectInterface(
    StringRef dialectName, TypeID interfaceTypeID,
    DialectInterfaceAllocatorFunction allocator) {
  assert(allocator && "unexpected null interface allocation function");
  auto it = registry.find(dialectName.str());
  assert(it != registry.end() &&
         "adding an interface for an unregistered dialect");

  // Bail out if the interface with the given ID is already in the registry for
  // the given dialect. We expect a small number (dozens) of interfaces so a
  // linear search is fine here.
  auto &ifaces = interfaces[it->second.first];
  for (const auto &kvp : ifaces.dialectInterfaces) {
    if (kvp.first == interfaceTypeID) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE
                    "] repeated interface registration for dialect "
                 << dialectName);
      return;
    }
  }

  ifaces.dialectInterfaces.emplace_back(interfaceTypeID, allocator);
}

void DialectRegistry::addObjectInterface(
    StringRef dialectName, TypeID objectID, TypeID interfaceTypeID,
    ObjectInterfaceAllocatorFunction allocator) {
  assert(allocator && "unexpected null interface allocation function");

  auto it = registry.find(dialectName.str());
  assert(it != registry.end() &&
         "adding an interface for an op from an unregistered dialect");

  auto dialectID = it->second.first;
  auto &ifaces = interfaces[dialectID];

  for (const auto &info : ifaces.objectInterfaces) {
    if (std::get<0>(info) == objectID && std::get<1>(info) == interfaceTypeID) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE
                    "] repeated interface object interface registration");
      return;
    }
  }

  ifaces.objectInterfaces.emplace_back(objectID, interfaceTypeID, allocator);
}

DialectAllocatorFunctionRef
DialectRegistry::getDialectAllocator(StringRef name) const {
  auto it = registry.find(name.str());
  if (it == registry.end())
    return nullptr;
  return it->second.second;
}

void DialectRegistry::insert(TypeID typeID, StringRef name,
                             DialectAllocatorFunction ctor) {
  auto inserted = registry.insert(
      std::make_pair(std::string(name), std::make_pair(typeID, ctor)));
  if (!inserted.second && inserted.first->second.first != typeID) {
    llvm::report_fatal_error(
        "Trying to register different dialects for the same namespace: " +
        name);
  }
}

void DialectRegistry::registerDelayedInterfaces(Dialect *dialect) const {
  auto it = interfaces.find(dialect->getTypeID());
  if (it == interfaces.end())
    return;

  // Add an interface if it is not already present.
  for (const auto &kvp : it->getSecond().dialectInterfaces) {
    if (dialect->getRegisteredInterface(kvp.first))
      continue;
    dialect->addInterface(kvp.second(dialect));
  }

  // Add attribute, operation and type interfaces.
  for (const auto &info : it->getSecond().objectInterfaces)
    std::get<2>(info)(dialect->getContext());
}

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

Optional<Dialect::ParseOpHook>
Dialect::getParseOperationHook(StringRef opName) const {
  return None;
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
  auto it = registeredInterfaces.try_emplace(interface->getID(),
                                             std::move(interface));
  (void)it;
  assert(it.second && "interface kind has already been registered");
}

//===----------------------------------------------------------------------===//
// Dialect Interface
//===----------------------------------------------------------------------===//

DialectInterface::~DialectInterface() = default;

DialectInterfaceCollectionBase::DialectInterfaceCollectionBase(
    MLIRContext *ctx, TypeID interfaceKind) {
  for (auto *dialect : ctx->getLoadedDialects()) {
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
