//===- ToLLVMInterface.h - Conversion to LLVM iface ---*- C++ -*-=============//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_CONVERTTOLLVM_TOLLVMINTERFACE_H
#define MLIR_CONVERSION_CONVERTTOLLVM_TOLLVMINTERFACE_H

#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
class ConversionTarget;
class LLVMTypeConverter;
class MLIRContext;
class Operation;
class RewritePatternSet;
class AnalysisManager;

/// Recursively walk the IR and collect all dialects implementing the interface,
/// and populate the conversion patterns.
void populateConversionTargetFromOperation(Operation *op,
                                           ConversionTarget &target,
                                           LLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns);

/// Helper function for populating LLVM conversion patterns. If `op` implements
/// the `ConvertToLLVMOpInterface` interface, then the LLVM conversion pattern
/// attributes provided by the interface will be used to configure the
/// conversion target, type converter, and the pattern set.
void populateOpConvertToLLVMConversionPatterns(Operation *op,
                                               ConversionTarget &target,
                                               LLVMTypeConverter &typeConverter,
                                               RewritePatternSet &patterns);
} // namespace mlir

#include "mlir/Conversion/ConvertToLLVM/ToLLVMAttrInterface.h.inc"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMOpInterface.h.inc"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMDialectInterface.h.inc"

#endif // MLIR_CONVERSION_CONVERTTOLLVM_TOLLVMINTERFACE_H
