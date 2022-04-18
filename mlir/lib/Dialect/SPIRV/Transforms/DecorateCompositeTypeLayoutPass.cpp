//===- DecorateCompositeTypeLayoutPass.cpp - Decorate composite type ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to decorate the composite types used by
// composite objects in the StorageBuffer, PhysicalStorageBuffer, Uniform, and
// PushConstant storage classes with layout information. See SPIR-V spec
// "2.16.2. Validation Rules for Shader Capabilities" for more details.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/Utils/LayoutUtils.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class SPIRVGlobalVariableOpLayoutInfoDecoration
    : public OpRewritePattern<spirv::GlobalVariableOp> {
public:
  using OpRewritePattern<spirv::GlobalVariableOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::GlobalVariableOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<NamedAttribute, 4> globalVarAttrs;

    auto ptrType = op.type().cast<spirv::PointerType>();
    auto structType = VulkanLayoutUtils::decorateType(
        ptrType.getPointeeType().cast<spirv::StructType>());

    if (!structType)
      return failure();

    auto decoratedType =
        spirv::PointerType::get(structType, ptrType.getStorageClass());

    // Save all named attributes except "type" attribute.
    for (const auto &attr : op->getAttrs()) {
      if (attr.getName() == "type")
        continue;
      globalVarAttrs.push_back(attr);
    }

    rewriter.replaceOpWithNewOp<spirv::GlobalVariableOp>(
        op, TypeAttr::get(decoratedType), globalVarAttrs);
    return success();
  }
};

class SPIRVAddressOfOpLayoutInfoDecoration
    : public OpRewritePattern<spirv::AddressOfOp> {
public:
  using OpRewritePattern<spirv::AddressOfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::AddressOfOp op,
                                PatternRewriter &rewriter) const override {
    auto spirvModule = op->getParentOfType<spirv::ModuleOp>();
    auto varName = op.variableAttr();
    auto varOp = spirvModule.lookupSymbol<spirv::GlobalVariableOp>(varName);

    rewriter.replaceOpWithNewOp<spirv::AddressOfOp>(
        op, varOp.type(), SymbolRefAttr::get(varName.getAttr()));
    return success();
  }
};

template <typename OpT>
class SPIRVPassThroughConversion : public OpConversionPattern<OpT> {
public:
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.updateRootInPlace(op,
                               [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};
} // namespace

static void populateSPIRVLayoutInfoPatterns(RewritePatternSet &patterns) {
  patterns.add<SPIRVGlobalVariableOpLayoutInfoDecoration,
               SPIRVAddressOfOpLayoutInfoDecoration,
               SPIRVPassThroughConversion<spirv::AccessChainOp>,
               SPIRVPassThroughConversion<spirv::LoadOp>,
               SPIRVPassThroughConversion<spirv::StoreOp>>(
      patterns.getContext());
}

namespace {
class DecorateSPIRVCompositeTypeLayoutPass
    : public SPIRVCompositeTypeLayoutBase<
          DecorateSPIRVCompositeTypeLayoutPass> {
  void runOnOperation() override;
};
} // namespace

void DecorateSPIRVCompositeTypeLayoutPass::runOnOperation() {
  auto module = getOperation();
  RewritePatternSet patterns(module.getContext());
  populateSPIRVLayoutInfoPatterns(patterns);
  ConversionTarget target(*(module.getContext()));
  target.addLegalDialect<spirv::SPIRVDialect>();
  target.addLegalOp<func::FuncOp>();
  target.addDynamicallyLegalOp<spirv::GlobalVariableOp>(
      [](spirv::GlobalVariableOp op) {
        return VulkanLayoutUtils::isLegalType(op.type());
      });

  // Change the type for the direct users.
  target.addDynamicallyLegalOp<spirv::AddressOfOp>([](spirv::AddressOfOp op) {
    return VulkanLayoutUtils::isLegalType(op.pointer().getType());
  });

  // Change the type for the indirect users.
  target.addDynamicallyLegalOp<spirv::AccessChainOp, spirv::LoadOp,
                               spirv::StoreOp>([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      auto addrOp = operand.getDefiningOp<spirv::AddressOfOp>();
      if (addrOp && !VulkanLayoutUtils::isLegalType(addrOp.pointer().getType()))
        return false;
    }
    return true;
  });

  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  for (auto spirvModule : module.getOps<spirv::ModuleOp>())
    if (failed(applyFullConversion(spirvModule, target, frozenPatterns)))
      signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::spirv::createDecorateSPIRVCompositeTypeLayoutPass() {
  return std::make_unique<DecorateSPIRVCompositeTypeLayoutPass>();
}
