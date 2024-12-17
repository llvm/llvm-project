//===- ConvertToLLVMPass.cpp - MLIR LLVM Conversion -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataLayoutAnalysis.h"
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

//===----------------------------------------------------------------------===//
// StaticConvertToLLVM
//===----------------------------------------------------------------------===//

/// Static implementation of the `convert-to-llvm` pass. This version only looks
/// at dialect interfaces to configure the conversion process.
struct StaticConvertToLLVM : public ConvertToLLVMPassInterface {
  /// Pattern set with conversions to LLVM.
  std::shared_ptr<const FrozenRewritePatternSet> patterns;
  /// The conversion target.
  std::shared_ptr<const ConversionTarget> target;
  /// The LLVM type converter.
  std::shared_ptr<const LLVMTypeConverter> typeConverter;
  using ConvertToLLVMPassInterface::ConvertToLLVMPassInterface;

  /// Configure the conversion to LLVM at pass initialization.
  LogicalResult initialize() final {
    auto target = std::make_shared<ConversionTarget>(*context);
    auto typeConverter = std::make_shared<LLVMTypeConverter>(context);
    RewritePatternSet tempPatterns(context);
    target->addLegalDialect<LLVM::LLVMDialect>();
    // Populate the patterns with the dialect interface.
    if (failed(visitInterfaces([&](ConvertToLLVMPatternInterface *iface) {
          iface->populateConvertToLLVMConversionPatterns(
              *target, *typeConverter, tempPatterns);
        })))
      return failure();
    this->patterns =
        std::make_unique<FrozenRewritePatternSet>(std::move(tempPatterns));
    this->target = target;
    this->typeConverter = typeConverter;
    return success();
  }

  /// Apply the conversion driver.
  LogicalResult transform(Operation *op, AnalysisManager manager) const final {
    if (failed(applyPartialConversion(op, *target, *patterns)))
      return failure();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// DynamicConvertToLLVM
//===----------------------------------------------------------------------===//

/// Dynamic implementation of the `convert-to-llvm` pass. This version inspects
/// the IR to configure the conversion to LLVM.
struct DynamicConvertToLLVM : public ConvertToLLVMPassInterface {
  /// A list of all the `ConvertToLLVMPatternInterface` dialect interfaces used
  /// to partially configure the conversion process.
  std::shared_ptr<const SmallVector<ConvertToLLVMPatternInterface *>>
      interfaces;
  using ConvertToLLVMPassInterface::ConvertToLLVMPassInterface;

  /// Collect the dialect interfaces used to configure the conversion process.
  LogicalResult initialize() final {
    auto interfaces =
        std::make_shared<SmallVector<ConvertToLLVMPatternInterface *>>();
    // Collect the interfaces.
    if (failed(visitInterfaces([&](ConvertToLLVMPatternInterface *iface) {
          interfaces->push_back(iface);
        })))
      return failure();
    this->interfaces = interfaces;
    return success();
  }

  /// Configure the conversion process and apply the conversion driver.
  LogicalResult transform(Operation *op, AnalysisManager manager) const final {
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    // Get the data layout analysis.
    const auto &dlAnalysis = manager.getAnalysis<DataLayoutAnalysis>();
    LLVMTypeConverter typeConverter(context, &dlAnalysis);

    // Configure the conversion with dialect level interfaces.
    for (ConvertToLLVMPatternInterface *iface : *interfaces)
      iface->populateConvertToLLVMConversionPatterns(target, typeConverter,
                                                     patterns);

    // Configure the conversion attribute interfaces.
    populateOpConvertToLLVMConversionPatterns(op, target, typeConverter,
                                              patterns);

    // Apply the conversion.
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      return failure();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertToLLVMPass
//===----------------------------------------------------------------------===//

/// This is a generic pass to convert to LLVM, it uses the
/// `ConvertToLLVMPatternInterface` dialect interface to delegate to dialects
/// the injection of conversion patterns.
class ConvertToLLVMPass
    : public impl::ConvertToLLVMPassBase<ConvertToLLVMPass> {
  std::shared_ptr<const ConvertToLLVMPassInterface> impl;

public:
  using impl::ConvertToLLVMPassBase<ConvertToLLVMPass>::ConvertToLLVMPassBase;
  void getDependentDialects(DialectRegistry &registry) const final {
    ConvertToLLVMPassInterface::getDependentDialects(registry);
  }

  LogicalResult initialize(MLIRContext *context) final {
    std::shared_ptr<ConvertToLLVMPassInterface> impl;
    // Choose the pass implementation.
    if (useDynamic)
      impl = std::make_shared<DynamicConvertToLLVM>(context, filterDialects);
    else
      impl = std::make_shared<StaticConvertToLLVM>(context, filterDialects);
    if (failed(impl->initialize()))
      return failure();
    this->impl = impl;
    return success();
  }

  void runOnOperation() final {
    if (failed(impl->transform(getOperation(), getAnalysisManager())))
      return signalPassFailure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// ConvertToLLVMPassInterface
//===----------------------------------------------------------------------===//

ConvertToLLVMPassInterface::ConvertToLLVMPassInterface(
    MLIRContext *context, ArrayRef<std::string> filterDialects)
    : context(context), filterDialects(filterDialects) {}

void ConvertToLLVMPassInterface::getDependentDialects(
    DialectRegistry &registry) {
  registry.insert<LLVM::LLVMDialect>();
  registry.addExtensions<LoadDependentDialectExtension>();
}

LogicalResult ConvertToLLVMPassInterface::visitInterfaces(
    llvm::function_ref<void(ConvertToLLVMPatternInterface *)> visitor) {
  if (!filterDialects.empty()) {
    // Test mode: Populate only patterns from the specified dialects. Produce
    // an error if the dialect is not loaded or does not implement the
    // interface.
    for (StringRef dialectName : filterDialects) {
      Dialect *dialect = context->getLoadedDialect(dialectName);
      if (!dialect)
        return emitError(UnknownLoc::get(context))
               << "dialect not loaded: " << dialectName << "\n";
      auto *iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect);
      if (!iface)
        return emitError(UnknownLoc::get(context))
               << "dialect does not implement ConvertToLLVMPatternInterface: "
               << dialectName << "\n";
      visitor(iface);
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
      visitor(iface);
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

void mlir::registerConvertToLLVMDependentDialectLoading(
    DialectRegistry &registry) {
  registry.addExtensions<LoadDependentDialectExtension>();
}

std::unique_ptr<Pass> mlir::createConvertToLLVMPass() {
  return std::make_unique<ConvertToLLVMPass>();
}
