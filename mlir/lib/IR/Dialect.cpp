//===- Dialect.cpp - Dialect implementation -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectHooks.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Regex.h"

using namespace mlir;
using namespace detail;

DialectAsmParser::~DialectAsmParser() {}

//===----------------------------------------------------------------------===//
// Dialect Registration
//===----------------------------------------------------------------------===//

/// Registry for all dialect allocation functions.
static llvm::ManagedStatic<llvm::MapVector<TypeID, DialectAllocatorFunction>>
    dialectRegistry;

/// Registry for functions that set dialect hooks.
static llvm::ManagedStatic<llvm::MapVector<TypeID, DialectHooksSetter>>
    dialectHooksRegistry;

void Dialect::registerDialectAllocator(
    TypeID typeID, const DialectAllocatorFunction &function) {
  assert(function &&
         "Attempting to register an empty dialect initialize function");
  dialectRegistry->insert({typeID, function});
}

/// Registers a function to set specific hooks for a specific dialect, typically
/// used through the DialectHooksRegistration template.
void DialectHooks::registerDialectHooksSetter(
    TypeID typeID, const DialectHooksSetter &function) {
  assert(
      function &&
      "Attempting to register an empty dialect hooks initialization function");

  dialectHooksRegistry->insert({typeID, function});
}

/// Registers all dialects and hooks from the global registries with the
/// specified MLIRContext.
void mlir::registerAllDialects(MLIRContext *context) {
  for (const auto &it : *dialectRegistry)
    it.second(context);
  for (const auto &it : *dialectHooksRegistry)
    it.second(context);
}

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

Dialect::Dialect(StringRef name, MLIRContext *context, TypeID id)
    : name(name), dialectID(id), context(context) {
  assert(isValidNamespace(name) && "invalid dialect namespace");
}

Dialect::~Dialect() {}

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
    auto ns = Identifier::get(getNamespace(), getContext());
    return OpaqueType::get(ns, parser.getFullSymbolSpec(), getContext());
  }

  parser.emitError(parser.getNameLoc())
      << "dialect '" << getNamespace() << "' provides no type parsing hook";
  return Type();
}

/// Utility function that returns if the given string is a valid dialect
/// namespace.
bool Dialect::isValidNamespace(StringRef str) {
  if (str.empty())
    return true;
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

DialectInterface::~DialectInterface() {}

DialectInterfaceCollectionBase::DialectInterfaceCollectionBase(
    MLIRContext *ctx, TypeID interfaceKind) {
  for (auto *dialect : ctx->getRegisteredDialects()) {
    if (auto *interface = dialect->getRegisteredInterface(interfaceKind)) {
      interfaces.insert(interface);
      orderedInterfaces.push_back(interface);
    }
  }
}

DialectInterfaceCollectionBase::~DialectInterfaceCollectionBase() {}

/// Get the interface for the dialect of given operation, or null if one
/// is not registered.
const DialectInterface *
DialectInterfaceCollectionBase::getInterfaceFor(Operation *op) const {
  return getInterfaceFor(op->getDialect());
}
