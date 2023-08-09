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
  LoadDependentDialectExtension() : DialectExtensionBase(/*dialectNames=*/{}) {}

  void apply(MLIRContext *context,
             MutableArrayRef<Dialect *> dialects) const final {
    LLVM_DEBUG(llvm::dbgs() << "Convert to LLVM extension load\n");
    for (Dialect *dialect : dialects) {
      auto iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect);
      if (!iface)
        continue;
      LLVM_DEBUG(llvm::dbgs() << "Convert to LLVM found dialect interface for "
                              << dialect->getNamespace() << "\n");
      iface->loadDependentDialects(context);
    }
  }

  /// Return a copy of this extension.
  virtual std::unique_ptr<DialectExtensionBase> clone() const final {
    return std::make_unique<LoadDependentDialectExtension>(*this);
  }
};

/// This is a generic pass to convert to LLVM, it uses the
/// `ConvertToLLVMPatternInterface` dialect interface to delegate to dialects
/// the injection of conversion patterns.
class ConvertToLLVMPass
    : public impl::ConvertToLLVMPassBase<ConvertToLLVMPass> {
  std::shared_ptr<const FrozenRewritePatternSet> patterns;
  std::shared_ptr<const ConversionTarget> target;

public:
  using impl::ConvertToLLVMPassBase<ConvertToLLVMPass>::ConvertToLLVMPassBase;
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect>();
    registry.addExtensions<LoadDependentDialectExtension>();
  }

  ConvertToLLVMPass(const ConvertToLLVMPass &other)
      : ConvertToLLVMPassBase(other), patterns(other.patterns),
        target(other.target) {}

  LogicalResult initialize(MLIRContext *context) final {
    RewritePatternSet tempPatterns(context);
    auto target = std::make_shared<ConversionTarget>(*context);
    target->addLegalDialect<LLVM::LLVMDialect>();
    for (Dialect *dialect : context->getLoadedDialects()) {
      // First time we encounter this dialect: if it implements the interface,
      // let's populate patterns !
      auto iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect);
      if (!iface)
        continue;
      iface->populateConvertToLLVMConversionPatterns(*target, tempPatterns);
    }
    patterns =
        std::make_unique<FrozenRewritePatternSet>(std::move(tempPatterns));
    this->target = target;
    return success();
  }

  void runOnOperation() final {
    if (failed(applyPartialConversion(getOperation(), *target, *patterns)))
      signalPassFailure();
  }
};

} // namespace
