//===- ODSSupport.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a number of support method for ODS generated code.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_ODSSUPPORT_H
#define MLIR_IR_ODSSUPPORT_H

#include "mlir/IR/Attributes.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// Support for properties
//===----------------------------------------------------------------------===//

/// Convert an IntegerAttr attribute to an int64_t, or return an error if the
/// attribute isn't an IntegerAttr. If the optional diagnostic is provided an
/// error message is also emitted.
LogicalResult convertFromAttribute(int64_t &storage, Attribute attr,
                                   InFlightDiagnostic *diag);

/// Convert the provided int64_t to an IntegerAttr attribute.
Attribute convertToAttribute(MLIRContext *ctx, int64_t storage);

/// Convert a DenseI64ArrayAttr to the provided storage. It is expected that the
/// storage has the same size as the array. An error is returned if the
/// attribute isn't a DenseI64ArrayAttr or it does not have the same size. If
/// the optional diagnostic is provided an error message is also emitted.
LogicalResult convertFromAttribute(MutableArrayRef<int64_t> storage,
                                   Attribute attr, InFlightDiagnostic *diag);

/// Convert the provided ArrayRef<int64_t> to a DenseI64ArrayAttr attribute.
Attribute convertToAttribute(MLIRContext *ctx, ArrayRef<int64_t> storage);

} // namespace mlir

#endif // MLIR_IR_ODSSUPPORT_H