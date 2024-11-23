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

/// Base class for dialect interfaces providing translation to LLVM IR.
/// Dialects that can be translated should provide an implementation of this
/// interface for the supported operations. The interface may be implemented in
/// a separate library to avoid the "main" dialect library depending on LLVM IR.
/// The interface can be attached using the delayed registration mechanism
/// available in DialectRegistry.
class ConvertToLLVMPatternInterface
    : public DialectInterface::Base<ConvertToLLVMPatternInterface> {
public:
  ConvertToLLVMPatternInterface(Dialect *dialect) : Base(dialect) {}

  /// Hook for derived dialect interface to load the dialects they
  /// target. The LLVMDialect is implicitly already loaded, but this
  /// method allows to load other intermediate dialects used in the
  /// conversion, or target dialects like NVVM for example.
  virtual void loadDependentDialects(MLIRContext *context) const {}

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  virtual void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const = 0;
};

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

/// Base class for creating the internal implementation of `convert-to-llvm`
/// passes.
class ConvertToLLVMPassInterface {
public:
  ConvertToLLVMPassInterface(MLIRContext *context,
                             ArrayRef<std::string> filterDialects);
  virtual ~ConvertToLLVMPassInterface() = default;

  /// Get the dependent dialects used by `convert-to-llvm`.
  static void getDependentDialects(DialectRegistry &registry);

  /// Initialize the internal state of the `convert-to-llvm` pass
  /// implementation. This method is invoked by `ConvertToLLVMPass::initialize`.
  /// This method returns whether the initialization process failed.
  virtual LogicalResult initialize() = 0;

  /// Transform `op` to LLVM with the conversions available in the pass. The
  /// analysis manager can be used to query analyzes like `DataLayoutAnalysis`
  /// to further configure the conversion process. This method is invoked by
  /// `ConvertToLLVMPass::runOnOperation`. This method returns whether the
  /// transformation process failed.
  virtual LogicalResult transform(Operation *op,
                                  AnalysisManager manager) const = 0;

protected:
  /// Visit the `ConvertToLLVMPatternInterface` dialect interfaces and call
  /// `visitor` with each of the interfaces. If `filterDialects` is non-empty,
  /// then `visitor` is invoked only with the dialects in the `filterDialects`
  /// list.
  LogicalResult visitInterfaces(
      llvm::function_ref<void(ConvertToLLVMPatternInterface *)> visitor);
  MLIRContext *context;
  /// List of dialects names to use as filters.
  ArrayRef<std::string> filterDialects;
};
} // namespace mlir

#include "mlir/Conversion/ConvertToLLVM/ToLLVMAttrInterface.h.inc"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMOpInterface.h.inc"

#endif // MLIR_CONVERSION_CONVERTTOLLVM_TOLLVMINTERFACE_H
