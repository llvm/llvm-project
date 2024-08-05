//===- ToSPIRVInterface.h - Conversion to SPIRV iface -*- C++ -*-=============//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_CONVERTTOSPIRV_TOSPIRVINTERFACE_H
#define MLIR_CONVERSION_CONVERTTOSPIRV_TOSPIRVINTERFACE_H

#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {
class ConversionTarget;
class SPIRVTypeConverter;
class MLIRContext;
class Operation;
class RewritePatternSet;

/// Base class for dialect interfaces providing translation to SPIR-V.
/// Dialects that can be translated should provide an implementation of this
/// interface for the supported operations. The interface may be implemented in
/// a separate library to avoid the "main" dialect library depending on SPIR-V
/// IR. The interface can be attached using the delayed registration mechanism
/// available in DialectRegistry.
class ConvertToSPIRVPatternInterface
    : public DialectInterface::Base<ConvertToSPIRVPatternInterface> {
public:
  ConvertToSPIRVPatternInterface(Dialect *dialect) : Base(dialect) {}

  /// Hook for derived dialect interface to load the dialects they
  /// target. The SPIRVDialect is implicitly already loaded, but this
  /// method allows to load other intermediate dialects used in the
  /// conversion.
  virtual void loadDependentDialects(MLIRContext *context) const {}

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  virtual void populateConvertToSPIRVConversionPatterns(
      ConversionTarget &target, SPIRVTypeConverter &typeConverter,
      RewritePatternSet &patterns) const = 0;
};

/// Recursively walk the IR and collect all dialects implementing the interface,
/// and populate the conversion patterns.
void populateConversionTargetFromOperation(Operation *op,
                                           ConversionTarget &target,
                                           SPIRVTypeConverter &typeConverter,
                                           RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_CONVERTTOSPIRV_TOSPIRVINTERFACE_H
