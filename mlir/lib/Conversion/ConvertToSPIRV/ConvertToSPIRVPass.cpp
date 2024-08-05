//===- ConvertToSPIRVPass.cpp - MLIR SPIR-V Conversion --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ConvertToSPIRV/ConvertToSPIRVPass.h"
#include "mlir/Conversion/ConvertToSPIRV/ToSPIRVInterface.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

#define DEBUG_TYPE "convert-to-spirv"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOSPIRVPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
/// This DialectExtension can be attached to the context, which will invoke the
/// `apply()` method for every loaded dialect. If a dialect implements the
/// `ConvertToSPIRVPatternInterface` interface, we load dependent dialects
/// through the interface. This extension is loaded in the context before
/// starting a pass pipeline that involves dialect conversion to SPIR-V.
class LoadDependentDialectExtension : public DialectExtensionBase {
public:
  LoadDependentDialectExtension() : DialectExtensionBase(/*dialectNames=*/{}) {}

  void apply(MLIRContext *context,
             MutableArrayRef<Dialect *> dialects) const final {
    LLVM_DEBUG(llvm::dbgs() << "Convert to SPIR-V extension load\n");
    for (Dialect *dialect : dialects) {
      auto *iface = dyn_cast<ConvertToSPIRVPatternInterface>(dialect);
      if (!iface)
        continue;
      LLVM_DEBUG(llvm::dbgs()
                 << "Convert to SPIR-V found dialect interface for "
                 << dialect->getNamespace() << "\n");
      iface->loadDependentDialects(context);
    }
  }

  /// Return a copy of this extension.
  std::unique_ptr<DialectExtensionBase> clone() const final {
    return std::make_unique<LoadDependentDialectExtension>(*this);
  }
};

/// This is a generic pass to convert to SPIR-V, it uses the
/// `ConvertToSPIRVPatternInterface` dialect interface to delegate to dialects
/// the injection of conversion patterns.
class ConvertToSPIRVPass
    : public impl::ConvertToSPIRVPassBase<ConvertToSPIRVPass> {
  std::shared_ptr<const SmallVector<ConvertToSPIRVPatternInterface *>>
      interfaces;

public:
  using impl::ConvertToSPIRVPassBase<
      ConvertToSPIRVPass>::ConvertToSPIRVPassBase;
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<spirv::SPIRVDialect>();
    registry.addExtensions<LoadDependentDialectExtension>();
  }

  LogicalResult initialize(MLIRContext *context) final {
    auto interfaces =
        std::make_shared<SmallVector<ConvertToSPIRVPatternInterface *>>();
    if (!filterDialects.empty()) {
      // Test mode: Populate only patterns from the specified dialects. Produce
      // an error if the dialect is not loaded or does not implement the
      // interface.
      for (std::string &dialectName : filterDialects) {
        Dialect *dialect = context->getLoadedDialect(dialectName);
        if (!dialect)
          return emitError(UnknownLoc::get(context))
                 << "dialect not loaded: " << dialectName << "\n";
        auto *iface = dyn_cast<ConvertToSPIRVPatternInterface>(dialect);
        if (!iface)
          return emitError(UnknownLoc::get(context))
                 << "dialect does not implement "
                    "ConvertToSPIRVPatternInterface: "
                 << dialectName << "\n";
        interfaces->push_back(iface);
      }
    } else {
      // Normal mode: Populate all patterns from all dialects that implement the
      // interface.
      for (Dialect *dialect : context->getLoadedDialects()) {
        // First time we encounter this dialect: if it implements the interface,
        // let's populate patterns !
        auto *iface = dyn_cast<ConvertToSPIRVPatternInterface>(dialect);
        if (!iface)
          continue;
        interfaces->push_back(iface);
      }
    }

    this->interfaces = interfaces;
    return success();
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    Operation *op = getOperation();

    // Unroll vectors in function signatures to native size.
    if (runSignatureConversion && failed(spirv::unrollVectorsInSignatures(op)))
      return signalPassFailure();

    // Unroll vectors in function bodies to native size.
    if (runVectorUnrolling && failed(spirv::unrollVectorsInFuncBodies(op)))
      return signalPassFailure();

    // Lookup the target.
    spirv::TargetEnvAttr targetAttr = spirv::lookupTargetEnvOrDefault(op);
    // Create and configure the conversion infrastructure.
    std::unique_ptr<ConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);
    SPIRVTypeConverter typeConverter(targetAttr);
    RewritePatternSet patterns(context);

    // Configure the conversion with dialect interfaces.
    for (ConvertToSPIRVPatternInterface *iface : *interfaces)
      iface->populateConvertToSPIRVConversionPatterns(*target, typeConverter,
                                                      patterns);

    // TODO: Incorporate SCF to SPIR-V into the interface.
    ScfToSPIRVContext scfToSPIRVContext;
    populateSCFToSPIRVPatterns(typeConverter, scfToSPIRVContext, patterns);

    // Apply the conversion.
    if (failed(applyPartialConversion(op, *target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::registerConvertToSPIRVDependentDialectLoading(
    DialectRegistry &registry) {
  registry.addExtensions<LoadDependentDialectExtension>();
}
