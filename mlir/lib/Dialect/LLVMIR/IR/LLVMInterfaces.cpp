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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace mlir::LLVM;

/// Verifies the given array attribute contains symbol references and checks the
/// referenced symbol types using the provided verification function.
static LogicalResult
verifySymbolRefs(Operation *op, StringRef name, ArrayAttr symbolRefs,
                 llvm::function_ref<LogicalResult(Operation *, SymbolRefAttr)>
                     verifySymbolType) {
  assert(symbolRefs && "expected a non-null attribute");

  // Verify that the attribute is a symbol ref array attribute,
  // because this constraint is not verified for all attribute
  // names processed here (e.g. 'tbaa'). This verification
  // is redundant in some cases.
  if (!llvm::all_of(symbolRefs, [](Attribute attr) {
        return attr && llvm::isa<SymbolRefAttr>(attr);
      }))
    return op->emitOpError() << name
                             << " attribute failed to satisfy constraint: "
                                "symbol ref array attribute";

  for (SymbolRefAttr symbolRef : symbolRefs.getAsRange<SymbolRefAttr>()) {
    StringAttr metadataName = symbolRef.getRootReference();
    StringAttr symbolName = symbolRef.getLeafReference();
    // We want @metadata::@symbol, not just @symbol
    if (metadataName == symbolName) {
      return op->emitOpError() << "expected '" << symbolRef
                               << "' to specify a fully qualified reference";
    }
    auto metadataOp = SymbolTable::lookupNearestSymbolFrom<LLVM::MetadataOp>(
        op->getParentOp(), metadataName);
    if (!metadataOp)
      return op->emitOpError()
             << "expected '" << symbolRef << "' to reference a metadata op";
    Operation *symbolOp =
        SymbolTable::lookupNearestSymbolFrom(metadataOp, symbolName);
    if (!symbolOp)
      return op->emitOpError()
             << "expected '" << symbolRef << "' to be a valid reference";
    if (failed(verifySymbolType(symbolOp, symbolRef))) {
      return failure();
    }
  }

  return success();
}

/// Verifies the given array attribute contains symbol references that point to
/// metadata operations of the given type.
template <typename OpTy>
LogicalResult verifySymbolRefsPointTo(Operation *op, StringRef name,
                                      ArrayAttr symbolRefs) {
  if (!symbolRefs)
    return success();

  auto verifySymbolType = [op](Operation *symbolOp,
                               SymbolRefAttr symbolRef) -> LogicalResult {
    if (!isa<OpTy>(symbolOp)) {
      return op->emitOpError()
             << "expected '" << symbolRef << "' to resolve to a "
             << OpTy::getOperationName();
    }
    return success();
  };
  return verifySymbolRefs(op, name, symbolRefs, verifySymbolType);
}

//===----------------------------------------------------------------------===//
// AccessGroupOpInterface
//===----------------------------------------------------------------------===//

LogicalResult mlir::LLVM::detail::verifyAccessGroupOpInterface(Operation *op) {
  auto iface = cast<AccessGroupOpInterface>(op);
  if (failed(verifySymbolRefsPointTo<LLVM::AccessGroupMetadataOp>(
          iface, "access groups", iface.getAccessGroupsOrNull())))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// AliasAnalysisOpInterface
//===----------------------------------------------------------------------===//

LogicalResult
mlir::LLVM::detail::verifyAliasAnalysisOpInterface(Operation *op) {
  auto iface = cast<AliasAnalysisOpInterface>(op);
  if (failed(verifySymbolRefsPointTo<LLVM::AliasScopeMetadataOp>(
          iface, "alias scopes", iface.getAliasScopesOrNull())))
    return failure();
  if (failed(verifySymbolRefsPointTo<LLVM::AliasScopeMetadataOp>(
          iface, "noalias scopes", iface.getNoAliasScopesOrNull())))
    return failure();
  if (failed(verifySymbolRefsPointTo<LLVM::TBAATagOp>(
          iface, "tbaa tags", iface.getTBAATagsOrNull())))
    return failure();
  return success();
}

#include "mlir/Dialect/LLVMIR/LLVMInterfaces.cpp.inc"
