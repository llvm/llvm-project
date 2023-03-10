//===- IRDLLoading.cpp - IRDL dialect loading --------------------- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manages the loading of MLIR objects from IRDL operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IRDLLoading.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/SMLoc.h"

using namespace mlir;
using namespace mlir::irdl;

/// Define and load an operation represented by a `irdl.operation`
/// operation.
static WalkResult loadOperation(OperationOp op, ExtensibleDialect *dialect) {
  // IRDL does not support defining custom parsers or printers.
  auto parser = [](OpAsmParser &parser, OperationState &result) {
    return failure();
  };
  auto printer = [](Operation *op, OpAsmPrinter &printer, StringRef) {
    printer.printGenericOp(op);
  };

  auto verifier = [](Operation *op) { return success(); };

  // IRDL does not support defining regions.
  auto regionVerifier = [](Operation *op) { return success(); };

  auto opDef = DynamicOpDefinition::get(
      op.getName(), dialect, std::move(verifier), std::move(regionVerifier),
      std::move(parser), std::move(printer));
  dialect->registerDynamicOp(std::move(opDef));

  return WalkResult::advance();
}

/// Load all dialects in the given module, without loading any operation, type
/// or attribute definitions.
static DenseMap<DialectOp, ExtensibleDialect *> loadEmptyDialects(ModuleOp op) {
  DenseMap<DialectOp, ExtensibleDialect *> dialects;
  op.walk([&](DialectOp dialectOp) {
    MLIRContext *ctx = dialectOp.getContext();
    StringRef dialectName = dialectOp.getName();

    DynamicDialect *dialect = ctx->getOrLoadDynamicDialect(
        dialectName, [](DynamicDialect *dialect) {});

    dialects.insert({dialectOp, dialect});
  });
  return dialects;
}

/// Preallocate type definitions objects with empty verifiers.
/// This in particular allocates a TypeID for each type definition.
static DenseMap<TypeOp, std::unique_ptr<DynamicTypeDefinition>>
preallocateTypeDefs(ModuleOp op,
                    DenseMap<DialectOp, ExtensibleDialect *> dialects) {
  DenseMap<TypeOp, std::unique_ptr<DynamicTypeDefinition>> typeDefs;
  op.walk([&](TypeOp typeOp) {
    ExtensibleDialect *dialect = dialects[typeOp.getParentOp()];
    auto typeDef = DynamicTypeDefinition::get(
        typeOp.getName(), dialect,
        [](function_ref<InFlightDiagnostic()>, ArrayRef<Attribute>) {
          return success();
        });
    typeDefs.try_emplace(typeOp, std::move(typeDef));
  });
  return typeDefs;
}

/// Preallocate attribute definitions objects with empty verifiers.
/// This in particular allocates a TypeID for each attribute definition.
static DenseMap<AttributeOp, std::unique_ptr<DynamicAttrDefinition>>
preallocateAttrDefs(ModuleOp op,
                    DenseMap<DialectOp, ExtensibleDialect *> dialects) {
  DenseMap<AttributeOp, std::unique_ptr<DynamicAttrDefinition>> attrDefs;
  op.walk([&](AttributeOp attrOp) {
    ExtensibleDialect *dialect = dialects[attrOp.getParentOp()];
    auto attrDef = DynamicAttrDefinition::get(
        attrOp.getName(), dialect,
        [](function_ref<InFlightDiagnostic()>, ArrayRef<Attribute>) {
          return success();
        });
    attrDefs.try_emplace(attrOp, std::move(attrDef));
  });
  return attrDefs;
}

LogicalResult mlir::irdl::loadDialects(ModuleOp op) {
  // Preallocate all dialects, and type and attribute definitions.
  // In particular, this allocates TypeIDs so type and attributes can have
  // verifiers that refer to each other.
  DenseMap<DialectOp, ExtensibleDialect *> dialects = loadEmptyDialects(op);
  DenseMap<TypeOp, std::unique_ptr<DynamicTypeDefinition>> types =
      preallocateTypeDefs(op, dialects);
  DenseMap<AttributeOp, std::unique_ptr<DynamicAttrDefinition>> attrs =
      preallocateAttrDefs(op, dialects);

  // Define and load all operations.
  WalkResult res = op.walk([&](OperationOp opOp) {
    return loadOperation(opOp, dialects[opOp.getParentOp()]);
  });
  if (res.wasInterrupted())
    return failure();

  // Load all types in their dialects.
  for (auto &pair : types) {
    ExtensibleDialect *dialect = dialects[pair.first.getParentOp()];
    dialect->registerDynamicType(std::move(pair.second));
  }

  // Load all attributes in their dialects.
  for (auto &pair : attrs) {
    ExtensibleDialect *dialect = dialects[pair.first.getParentOp()];
    dialect->registerDynamicAttr(std::move(pair.second));
  }

  return success();
}
