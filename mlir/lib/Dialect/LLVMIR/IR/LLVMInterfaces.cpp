//===- LLVMInterfaces.cpp - LLVM Interfaces ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines op interfaces for the LLVM dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::LLVM;

/// Verifies that all elements of `array` are instances of `Attr`.
template <class AttrT>
static LogicalResult isArrayOf(Operation *op, ArrayAttr array) {
  for (Attribute iter : array)
    if (!isa<AttrT>(iter))
      return op->emitOpError("expected op to return array of ")
             << AttrT::getMnemonic() << " attributes";
  return success();
}

//===----------------------------------------------------------------------===//
// AccessGroupOpInterface
//===----------------------------------------------------------------------===//

LogicalResult mlir::LLVM::detail::verifyAccessGroupOpInterface(Operation *op) {
  auto iface = cast<AccessGroupOpInterface>(op);
  ArrayAttr accessGroups = iface.getAccessGroupsOrNull();
  if (!accessGroups)
    return success();

  return isArrayOf<AccessGroupAttr>(op, accessGroups);
}

//===----------------------------------------------------------------------===//
// AliasAnalysisOpInterface
//===----------------------------------------------------------------------===//

LogicalResult
mlir::LLVM::detail::verifyAliasAnalysisOpInterface(Operation *op) {
  auto iface = cast<AliasAnalysisOpInterface>(op);

  if (auto aliasScopes = iface.getAliasScopesOrNull())
    if (failed(isArrayOf<AliasScopeAttr>(op, aliasScopes)))
      return failure();

  if (auto noAliasScopes = iface.getNoAliasScopesOrNull())
    if (failed(isArrayOf<AliasScopeAttr>(op, noAliasScopes)))
      return failure();

  ArrayAttr tags = iface.getTBAATagsOrNull();
  if (!tags)
    return success();

  return isArrayOf<TBAATagAttr>(op, tags);
}

//===----------------------------------------------------------------------===//
// TBAAStructOpInterface
//===----------------------------------------------------------------------===//

LogicalResult mlir::LLVM::detail::verifyTBAAStructOpInterface(Operation *op) {
  auto iface = cast<TBAAStructOpInterface>(op);
  ArrayAttr fields = iface.getTBAAStructOrNull();
  if (!fields)
    return success();

  if (failed(isArrayOf<TBAAStructFieldAttr>(op, fields)))
    return failure();

  // Field offsets must be non-decreasing, matching LLVM's verifier rule.
  // Compare unsigned: LLVM zero-extends the offsets before comparing.
  std::optional<uint64_t> prevOffset;
  for (Attribute field : fields) {
    uint64_t offset = cast<TBAAStructFieldAttr>(field).getOffset();
    if (prevOffset && offset < *prevOffset)
      return op->emitOpError(
          "tbaa.struct field offsets must be non-decreasing");
    prevOffset = offset;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DereferenceableOpInterface
//===----------------------------------------------------------------------===//

LogicalResult
mlir::LLVM::detail::verifyDereferenceableOpInterface(Operation *op) {
  auto iface = cast<DereferenceableOpInterface>(op);

  if (auto derefAttr = iface.getDereferenceableOrNull())
    if (op->getNumResults() != 1 ||
        !mlir::isa<LLVMPointerType>(op->getResult(0).getType()))
      return op->emitOpError(
          "expected op to return a single LLVM pointer type");

  return success();
}

//===----------------------------------------------------------------------===//
// RangeOpInterface
//===----------------------------------------------------------------------===//

LogicalResult mlir::LLVM::detail::verifyRangeOpInterface(Operation *op) {
  auto iface = cast<RangeOpInterface>(op);

  ConstantRangeAttr range = iface.getRangeOrNull();
  if (!range)
    return success();
  if (op->getNumResults() != 1)
    return op->emitOpError("expected op to have a single result");
  auto intType =
      dyn_cast<IntegerType>(getElementTypeOrSelf(op->getResult(0).getType()));
  if (!intType || intType.getWidth() != range.getLower().getBitWidth())
    return op->emitOpError(
        "expected range bitwidth to match the integer (element) result type");
  if (range.getLower() == range.getUpper())
    return op->emitOpError("expected range to be neither empty nor full");
  return success();
}

#include "mlir/Dialect/LLVMIR/LLVMInterfaces.cpp.inc"
