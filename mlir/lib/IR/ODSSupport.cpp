//===- ODSSupport.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains out-of-line implementations of the support types that
// Operation and related classes build on top of.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/ODSSupport.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

using namespace mlir;

LogicalResult
mlir::convertFromAttribute(int64_t &storage, Attribute attr,
                           function_ref<InFlightDiagnostic()> emitError) {
  auto valueAttr = dyn_cast<IntegerAttr>(attr);
  if (!valueAttr) {
    emitError() << "expected IntegerAttr for key `value`";
    return failure();
  }
  storage = valueAttr.getValue().getSExtValue();
  return success();
}
Attribute mlir::convertToAttribute(MLIRContext *ctx, int64_t storage) {
  return IntegerAttr::get(IntegerType::get(ctx, 64), storage);
}

template <typename DenseArrayTy, typename T>
LogicalResult
convertDenseArrayFromAttr(MutableArrayRef<T> storage, Attribute attr,
                          function_ref<InFlightDiagnostic()> emitError,
                          StringRef denseArrayTyStr) {
  auto valueAttr = dyn_cast<DenseArrayTy>(attr);
  if (!valueAttr) {
    emitError() << "expected " << denseArrayTyStr << " for key `value`";
    return failure();
  }
  if (valueAttr.size() != static_cast<int64_t>(storage.size())) {
    emitError() << "size mismatch in attribute conversion: " << valueAttr.size()
                << " vs " << storage.size();
    return failure();
  }
  llvm::copy(valueAttr.asArrayRef(), storage.begin());
  return success();
}
LogicalResult
mlir::convertFromAttribute(MutableArrayRef<int64_t> storage, Attribute attr,
                           function_ref<InFlightDiagnostic()> emitError) {
  return convertDenseArrayFromAttr<DenseI64ArrayAttr>(storage, attr, emitError,
                                                      "DenseI64ArrayAttr");
}
LogicalResult
mlir::convertFromAttribute(MutableArrayRef<int32_t> storage, Attribute attr,
                           function_ref<InFlightDiagnostic()> emitError) {
  return convertDenseArrayFromAttr<DenseI32ArrayAttr>(storage, attr, emitError,
                                                      "DenseI32ArrayAttr");
}

Attribute mlir::convertToAttribute(MLIRContext *ctx,
                                   ArrayRef<int64_t> storage) {
  return DenseI64ArrayAttr::get(ctx, storage);
}
