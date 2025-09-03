//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRefAttrs.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

LogicalResult mlir::memref::AliasScopeAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, Attribute id,
    StringAttr description) {
  (void)description;
  if (!isa<StringAttr, DistinctAttr>(id))
    return emitError()
           << "id of an alias scope must be a StringAttr or a DistrinctAttr";

  return success();
}
