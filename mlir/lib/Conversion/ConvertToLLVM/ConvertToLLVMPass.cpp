//===- ConvertToLLVMPass.cpp - MLIR LLVM Conversion -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

#define DEBUG_TYPE "convert-to-llvm"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// This DialectExtension can be attached to the context, which will invoke the
/// `apply()` method for every loaded dialect. If a dialect implements the
/// `ConvertToLLVMPatternInterface` interface, we load dependent dialects
/// through the interface. This extension is loaded in the context before
/// starting a pass pipeline that involves dialect conversion to LLVM.
class LoadDependentDialectExtension : public DialectExtensionBase {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoadDependentDialectExtension)

  LoadDependentDialectExtension() : DialectExtensionBase(/*dialectNames=*/{}) {}

  void apply(MLIRContext *context,
             MutableArrayRef<Dialect *> dialects) const final {
    LLVM_DEBUG(llvm::dbgs() << "Convert to LLVM extension load\n");
    for (Dialect *dialect : dialects) {
      auto *iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect);
      if (!iface)
        continue;
      LLVM_DEBUG(llvm::dbgs() << "Convert to LLVM found dialect interface for "
                              << dialect->getNamespace() << "\n");
      iface->loadDependentDialects(context);
    }
  }

  /// Return a copy of this extension.
  std::unique_ptr<DialectExtensionBase> clone() const final {
    return std::make_unique<LoadDependentDialectExtension>(*this);
  }
};

/// This is a generic pass to convert to LLVM, it uses the
/// `ConvertToLLVMPatternInterface` dialect interface to delegate to dialects
/// the injection of conversion patterns.
class ConvertToLLVMPass
    : public impl::ConvertToLLVMPassBase<ConvertToLLVMPass> {
  std::shared_ptr<const SmallVector<ConvertToLLVMPatternInterface *>>
      interfaces;
  std::shared_ptr<const FrozenRewritePatternSet> patterns;
  std::shared_ptr<const ConversionTarget> target;
  std::shared_ptr<const LLVMTypeConverter> typeConverter;

public:
  using impl::ConvertToLLVMPassBase<ConvertToLLVMPass>::ConvertToLLVMPassBase;
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect>();
    registry.addExtensions<LoadDependentDialectExtension>();
  }

  LogicalResult initialize(MLIRContext *context) final {
    std::shared_ptr<SmallVector<ConvertToLLVMPatternInterface *>> interfaces;
    std::shared_ptr<ConversionTarget> target;
    std::shared_ptr<LLVMTypeConverter> typeConverter;
    RewritePatternSet tempPatterns(context);

    // Only collect the interfaces if `useConversionAttrs=true` as everything
    // else must be initialized in `runOnOperation`.
    if (useConversionAttrs) {
      interfaces =
          std::make_shared<SmallVector<ConvertToLLVMPatternInterface *>>();
    } else {
      target = std::make_shared<ConversionTarget>(*context);
      target->addLegalDialect<LLVM::LLVMDialect>();
      typeConverter = std::make_shared<LLVMTypeConverter>(context);
    }

    if (!filterDialects.empty()) {
      // Test mode: Populate only patterns from the specified dialects. Produce
      // an error if the dialect is not loaded or does not implement the
      // interface.
      for (std::string &dialectName : filterDialects) {
        Dialect *dialect = context->getLoadedDialect(dialectName);
        if (!dialect)
          return emitError(UnknownLoc::get(context))
                 << "dialect not loaded: " << dialectName << "\n";
        auto *iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect);
        if (!iface)
          return emitError(UnknownLoc::get(context))
                 << "dialect does not implement ConvertToLLVMPatternInterface: "
                 << dialectName << "\n";
        if (useConversionAttrs) {
          interfaces->push_back(iface);
          continue;
        }
        iface->populateConvertToLLVMConversionPatterns(*target, *typeConverter,
                                                       tempPatterns);
      }
    } else {
      // Normal mode: Populate all patterns from all dialects that implement the
      // interface.
      for (Dialect *dialect : context->getLoadedDialects()) {
        // First time we encounter this dialect: if it implements the interface,
        // let's populate patterns !
        auto *iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect);
        if (!iface)
          continue;
        if (useConversionAttrs) {
          interfaces->push_back(iface);
          continue;
        }
        iface->populateConvertToLLVMConversionPatterns(*target, *typeConverter,
                                                       tempPatterns);
      }
    }

    if (useConversionAttrs) {
      this->interfaces = interfaces;
    } else {
      this->patterns =
          std::make_unique<FrozenRewritePatternSet>(std::move(tempPatterns));
      this->target = target;
      this->typeConverter = typeConverter;
    }
    return success();
  }

  void runOnOperation() final {
    // Fast path:
    if (!useConversionAttrs) {
      if (failed(applyPartialConversion(getOperation(), *target, *patterns)))
        signalPassFailure();
      return;
    }
    // Slow path with conversion attributes.
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    LLVMTypeConverter typeConverter(context);

    // Configure the conversion with dialect level interfaces.
    for (ConvertToLLVMPatternInterface *iface : *interfaces)
      iface->populateConvertToLLVMConversionPatterns(target, typeConverter,
                                                     patterns);

    // Configure the conversion attribute interfaces.
    populateOpConvertToLLVMConversionPatterns(getOperation(), target,
                                              typeConverter, patterns);

    // Apply the conversion.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::registerConvertToLLVMDependentDialectLoading(
    DialectRegistry &registry) {
  registry.addExtensions<LoadDependentDialectExtension>();
}

std::unique_ptr<Pass> mlir::createConvertToLLVMPass() {
  return std::make_unique<ConvertToLLVMPass>();
}
