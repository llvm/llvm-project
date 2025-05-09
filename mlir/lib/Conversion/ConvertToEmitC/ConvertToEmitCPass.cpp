//===- ConvertToEmitCPass.cpp - Conversion to EmitC pass --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ConvertToEmitC/ConvertToEmitCPass.h"

#include "mlir/Conversion/ConvertToEmitC/ToEmitCInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#include <memory>

#define DEBUG_TYPE "convert-to-emitc"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOEMITC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
/// Base class for creating the internal implementation of `convert-to-emitc`
/// passes.
class ConvertToEmitCPassInterface {
public:
  ConvertToEmitCPassInterface(MLIRContext *context,
                              ArrayRef<std::string> filterDialects);
  virtual ~ConvertToEmitCPassInterface() = default;

  /// Get the dependent dialects used by `convert-to-emitc`.
  static void getDependentDialects(DialectRegistry &registry);

  /// Initialize the internal state of the `convert-to-emitc` pass
  /// implementation. This method is invoked by `ConvertToEmitC::initialize`.
  /// This method returns whether the initialization process failed.
  virtual LogicalResult initialize() = 0;

  /// Transform `op` to the EmitC dialect with the conversions available in the
  /// pass. The analysis manager can be used to query analyzes like
  /// `DataLayoutAnalysis` to further configure the conversion process. This
  /// method is invoked by `ConvertToEmitC::runOnOperation`. This method returns
  /// whether the transformation process failed.
  virtual LogicalResult transform(Operation *op,
                                  AnalysisManager manager) const = 0;

protected:
  /// Visit the `ConvertToEmitCPatternInterface` dialect interfaces and call
  /// `visitor` with each of the interfaces. If `filterDialects` is non-empty,
  /// then `visitor` is invoked only with the dialects in the `filterDialects`
  /// list.
  LogicalResult visitInterfaces(
      llvm::function_ref<void(ConvertToEmitCPatternInterface *)> visitor);
  MLIRContext *context;
  /// List of dialects names to use as filters.
  ArrayRef<std::string> filterDialects;
};

/// This DialectExtension can be attached to the context, which will invoke the
/// `apply()` method for every loaded dialect. If a dialect implements the
/// `ConvertToEmitCPatternInterface` interface, we load dependent dialects
/// through the interface. This extension is loaded in the context before
/// starting a pass pipeline that involves dialect conversion to the EmitC
/// dialect.
class LoadDependentDialectExtension : public DialectExtensionBase {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoadDependentDialectExtension)

  LoadDependentDialectExtension() : DialectExtensionBase(/*dialectNames=*/{}) {}

  void apply(MLIRContext *context,
             MutableArrayRef<Dialect *> dialects) const final {
    LLVM_DEBUG(llvm::dbgs() << "Convert to EmitC extension load\n");
    for (Dialect *dialect : dialects) {
      auto *iface = dyn_cast<ConvertToEmitCPatternInterface>(dialect);
      if (!iface)
        continue;
      LLVM_DEBUG(llvm::dbgs() << "Convert to EmitC found dialect interface for "
                              << dialect->getNamespace() << "\n");
    }
  }

  /// Return a copy of this extension.
  std::unique_ptr<DialectExtensionBase> clone() const final {
    return std::make_unique<LoadDependentDialectExtension>(*this);
  }
};

//===----------------------------------------------------------------------===//
// StaticConvertToEmitC
//===----------------------------------------------------------------------===//

/// Static implementation of the `convert-to-emitc` pass. This version only
/// looks at dialect interfaces to configure the conversion process.
struct StaticConvertToEmitC : public ConvertToEmitCPassInterface {
  /// Pattern set with conversions to the EmitC dialect.
  std::shared_ptr<const FrozenRewritePatternSet> patterns;
  /// The conversion target.
  std::shared_ptr<const ConversionTarget> target;
  /// The type converter.
  std::shared_ptr<const TypeConverter> typeConverter;
  using ConvertToEmitCPassInterface::ConvertToEmitCPassInterface;

  /// Configure the conversion to EmitC at pass initialization.
  LogicalResult initialize() final {
    auto target = std::make_shared<ConversionTarget>(*context);
    auto typeConverter = std::make_shared<TypeConverter>();

    // Add fallback identity converison.
    typeConverter->addConversion([](Type type) -> std::optional<Type> {
      if (emitc::isSupportedEmitCType(type))
        return type;
      return std::nullopt;
    });

    RewritePatternSet tempPatterns(context);
    target->addLegalDialect<emitc::EmitCDialect>();
    // Populate the patterns with the dialect interface.
    if (failed(visitInterfaces([&](ConvertToEmitCPatternInterface *iface) {
          iface->populateConvertToEmitCConversionPatterns(
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
// ConvertToEmitC
//===----------------------------------------------------------------------===//

/// This is a generic pass to convert to the EmitC dialect. It uses the
/// `ConvertToEmitCPatternInterface` dialect interface to delegate the injection
/// of conversion patterns to dialects.
class ConvertToEmitC : public impl::ConvertToEmitCBase<ConvertToEmitC> {
  std::shared_ptr<const ConvertToEmitCPassInterface> impl;

public:
  using impl::ConvertToEmitCBase<ConvertToEmitC>::ConvertToEmitCBase;
  void getDependentDialects(DialectRegistry &registry) const final {
    ConvertToEmitCPassInterface::getDependentDialects(registry);
  }

  LogicalResult initialize(MLIRContext *context) final {
    std::shared_ptr<ConvertToEmitCPassInterface> impl;
    impl = std::make_shared<StaticConvertToEmitC>(context, filterDialects);
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
// ConvertToEmitCPassInterface
//===----------------------------------------------------------------------===//

ConvertToEmitCPassInterface::ConvertToEmitCPassInterface(
    MLIRContext *context, ArrayRef<std::string> filterDialects)
    : context(context), filterDialects(filterDialects) {}

void ConvertToEmitCPassInterface::getDependentDialects(
    DialectRegistry &registry) {
  registry.insert<emitc::EmitCDialect>();
  registry.addExtensions<LoadDependentDialectExtension>();
}

LogicalResult ConvertToEmitCPassInterface::visitInterfaces(
    llvm::function_ref<void(ConvertToEmitCPatternInterface *)> visitor) {
  if (!filterDialects.empty()) {
    // Test mode: Populate only patterns from the specified dialects. Produce
    // an error if the dialect is not loaded or does not implement the
    // interface.
    for (StringRef dialectName : filterDialects) {
      Dialect *dialect = context->getLoadedDialect(dialectName);
      if (!dialect)
        return emitError(UnknownLoc::get(context))
               << "dialect not loaded: " << dialectName << "\n";
      auto *iface = dyn_cast<ConvertToEmitCPatternInterface>(dialect);
      if (!iface)
        return emitError(UnknownLoc::get(context))
               << "dialect does not implement ConvertToEmitCPatternInterface: "
               << dialectName << "\n";
      visitor(iface);
    }
  } else {
    // Normal mode: Populate all patterns from all dialects that implement the
    // interface.
    for (Dialect *dialect : context->getLoadedDialects()) {
      auto *iface = dyn_cast<ConvertToEmitCPatternInterface>(dialect);
      if (!iface)
        continue;
      visitor(iface);
    }
  }
  return success();
}
