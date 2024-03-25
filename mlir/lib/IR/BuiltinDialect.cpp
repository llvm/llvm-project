//===- BuiltinDialect.cpp - MLIR Builtin Dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Builtin dialect that contains all of the attributes,
// operations, and types that are necessary for the validity of the IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinDialect.h"
#include "BuiltinDialectBytecode.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// TableGen'erated dialect
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// BuiltinBlobManagerInterface
//===----------------------------------------------------------------------===//

using BuiltinBlobManagerInterface =
    ResourceBlobManagerDialectInterfaceBase<DenseResourceElementsHandle>;

//===----------------------------------------------------------------------===//
// BuiltinOpAsmDialectInterface
//===----------------------------------------------------------------------===//

namespace {
struct BuiltinOpAsmDialectInterface : public OpAsmDialectInterface {
  BuiltinOpAsmDialectInterface(Dialect *dialect,
                               BuiltinBlobManagerInterface &mgr)
      : OpAsmDialectInterface(dialect), blobManager(mgr) {}

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (llvm::isa<AffineMapAttr>(attr)) {
      os << "map";
      return AliasResult::OverridableAlias;
    }
    if (llvm::isa<IntegerSetAttr>(attr)) {
      os << "set";
      return AliasResult::OverridableAlias;
    }
    if (llvm::isa<LocationAttr>(attr)) {
      os << "loc";
      return AliasResult::OverridableAlias;
    }
    if (auto distinct = llvm::dyn_cast<DistinctAttr>(attr))
      if (!llvm::isa<UnitAttr>(distinct.getReferencedAttr())) {
        os << "distinct";
        return AliasResult::OverridableAlias;
      }
    return AliasResult::NoAlias;
  }

  AliasResult getAlias(Type type, raw_ostream &os) const final {
    if (auto tupleType = llvm::dyn_cast<TupleType>(type)) {
      if (tupleType.size() > 16) {
        os << "tuple";
        return AliasResult::OverridableAlias;
      }
    }
    return AliasResult::NoAlias;
  }

  //===------------------------------------------------------------------===//
  // Resources
  //===------------------------------------------------------------------===//

  std::string
  getResourceKey(const AsmDialectResourceHandle &handle) const override {
    return cast<DenseResourceElementsHandle>(handle).getKey().str();
  }
  FailureOr<AsmDialectResourceHandle>
  declareResource(StringRef key) const final {
    return blobManager.insert(key);
  }
  LogicalResult parseResource(AsmParsedResourceEntry &entry) const final {
    FailureOr<AsmResourceBlob> blob = entry.parseAsBlob();
    if (failed(blob))
      return failure();

    // Update the blob for this entry.
    blobManager.update(entry.getKey(), std::move(*blob));
    return success();
  }
  void
  buildResources(Operation *op,
                 const SetVector<AsmDialectResourceHandle> &referencedResources,
                 AsmResourceBuilder &provider) const final {
    blobManager.buildResources(provider, referencedResources.getArrayRef());
  }

private:
  /// The blob manager for the dialect.
  BuiltinBlobManagerInterface &blobManager;
};
} // namespace

void BuiltinDialect::initialize() {
  registerTypes();
  registerAttributes();
  registerLocationAttributes();
  addOperations<
#define GET_OP_LIST
#include "mlir/IR/BuiltinOps.cpp.inc"
      >();

  auto &blobInterface = addInterface<BuiltinBlobManagerInterface>();
  addInterface<BuiltinOpAsmDialectInterface>(blobInterface);
  builtin_dialect_detail::addBytecodeInterface(this);
}

//===----------------------------------------------------------------------===//
// ModuleOp
//===----------------------------------------------------------------------===//

void ModuleOp::build(OpBuilder &builder, OperationState &state,
                     std::optional<StringRef> name) {
  state.addRegion()->emplaceBlock();
  if (name) {
    state.attributes.push_back(builder.getNamedAttr(
        mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(*name)));
  }
}

/// Construct a module from the given context.
ModuleOp ModuleOp::create(Location loc, std::optional<StringRef> name) {
  OpBuilder builder(loc->getContext());
  return builder.create<ModuleOp>(loc, name);
}

DataLayoutSpecInterface ModuleOp::getDataLayoutSpec() {
  // Take the first and only (if present) attribute that implements the
  // interface. This needs a linear search, but is called only once per data
  // layout object construction that is used for repeated queries.
  for (NamedAttribute attr : getOperation()->getAttrs())
    if (auto spec = llvm::dyn_cast<DataLayoutSpecInterface>(attr.getValue()))
      return spec;
  return {};
}

LogicalResult ModuleOp::verify() {
  // Check that none of the attributes are non-dialect attributes, except for
  // the symbol related attributes.
  for (auto attr : (*this)->getAttrs()) {
    if (!attr.getName().strref().contains('.') &&
        !llvm::is_contained(
            ArrayRef<StringRef>{mlir::SymbolTable::getSymbolAttrName(),
                                mlir::SymbolTable::getVisibilityAttrName()},
            attr.getName().strref()))
      return emitOpError() << "can only contain attributes with "
                              "dialect-prefixed names, found: '"
                           << attr.getName().getValue() << "'";
  }

  // Check that there is at most one data layout spec attribute.
  StringRef layoutSpecAttrName;
  DataLayoutSpecInterface layoutSpec;
  for (const NamedAttribute &na : (*this)->getAttrs()) {
    if (auto spec = llvm::dyn_cast<DataLayoutSpecInterface>(na.getValue())) {
      if (layoutSpec) {
        InFlightDiagnostic diag =
            emitOpError() << "expects at most one data layout attribute";
        diag.attachNote() << "'" << layoutSpecAttrName
                          << "' is a data layout attribute";
        diag.attachNote() << "'" << na.getName().getValue()
                          << "' is a data layout attribute";
      }
      layoutSpecAttrName = na.getName().strref();
      layoutSpec = spec;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// UnrealizedConversionCastOp
//===----------------------------------------------------------------------===//

LogicalResult
UnrealizedConversionCastOp::fold(FoldAdaptor adaptor,
                                 SmallVectorImpl<OpFoldResult> &foldResults) {
  OperandRange operands = getInputs();
  ResultRange results = getOutputs();

  if (operands.getType() == results.getType()) {
    foldResults.append(operands.begin(), operands.end());
    return success();
  }

  if (operands.empty())
    return failure();

  // Check that the input is a cast with results that all feed into this
  // operation, and operand types that directly match the result types of this
  // operation.
  Value firstInput = operands.front();
  auto inputOp = firstInput.getDefiningOp<UnrealizedConversionCastOp>();
  if (!inputOp || inputOp.getResults() != operands ||
      inputOp.getOperandTypes() != results.getTypes())
    return failure();

  // If everything matches up, we can fold the passthrough.
  foldResults.append(inputOp->operand_begin(), inputOp->operand_end());
  return success();
}

LogicalResult UnrealizedConversionCastOp::verify() {
  // TODO: The verifier of external models is not called. This op verifier can
  // be removed when that is fixed.
  if (getNumResults() == 0)
    return emitOpError() << "expected at least one result for cast operation";
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/IR/BuiltinOps.cpp.inc"
