//===- ToEmitCInterface.h - Conversion to EmitC iface ---*- C++ -*-===========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_CONVERTTOEMITC_TOEMITCINTERFACE_H
#define MLIR_CONVERSION_CONVERTTOEMITC_TOEMITCINTERFACE_H

#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
class ConversionTarget;
class TypeConverter;
class MLIRContext;
class Operation;
class RewritePatternSet;
class AnalysisManager;
/// Recursively walk the IR and collect all dialects implementing the interface,
/// and populate the conversion patterns.
void populateConversionTargetFromOperation(Operation *op,
                                           ConversionTarget &target,
                                           TypeConverter &typeConverter,
                                           RewritePatternSet &patterns);

} // namespace mlir

#include "mlir/Conversion/ConvertToEmitC/ConvertToEmitCPatternInterface.h.inc"

#endif // MLIR_CONVERSION_CONVERTTOEMITC_TOEMITCINTERFACE_H
