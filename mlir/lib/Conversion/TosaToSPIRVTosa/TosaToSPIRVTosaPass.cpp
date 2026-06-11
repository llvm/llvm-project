//===- TosaToSPIRVTosaPass.cpp - Lower TOSA to SPIR-V Graph/TOSA ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers TOSA IR to the SPIR-V Graph/TOSA representation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToSPIRVTosa/TosaToSPIRVTosa.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"

#include <algorithm>

namespace mlir {
#define GEN_PASS_DEF_TOSATOSPIRVTOSA
#include "mlir/Conversion/Passes.h.inc"

namespace tosa {

spirv::VerCapExtAttr getDefaultVerCapExtAttr(MLIRContext *context) {
  return spirv::VerCapExtAttr::get(
      spirv::Version::V_1_5,
      {
          spirv::Capability::VulkanMemoryModel,
          spirv::Capability::Shader,
          spirv::Capability::Int8,
          spirv::Capability::Int16,
          spirv::Capability::Int64,
          spirv::Capability::Float16,
          spirv::Capability::BFloat16TypeKHR,
          spirv::Capability::Float8EXT,
          spirv::Capability::TensorsARM,
          spirv::Capability::GraphARM,
          spirv::Capability::ReplicatedCompositesEXT,
      },
      {
          spirv::Extension::SPV_ARM_tensors,
          spirv::Extension::SPV_ARM_graph,
          spirv::Extension::SPV_KHR_vulkan_memory_model,
          spirv::Extension::SPV_EXT_replicated_composites,
          spirv::Extension::SPV_KHR_bfloat16,
          spirv::Extension::SPV_EXT_float8,
          spirv::Extension::SPV_KHR_non_semantic_info,
      },
      context);
}

spirv::TargetEnvAttr constructTargetEnvAttrWithCapExtDefaults(
    MLIRContext *context, spirv::ResourceLimitsAttr limits,
    spirv::ClientAPI clientAPI, spirv::Vendor vendorID,
    spirv::DeviceType deviceType, uint32_t deviceID) {
  if (!limits)
    limits = spirv::getDefaultResourceLimits(context);

  return spirv::TargetEnvAttr::get(getDefaultVerCapExtAttr(context), limits,
                                   clientAPI, vendorID, deviceType, deviceID);
}

namespace {

LogicalResult verifyGraphTargetEnv(Operation *op,
                                   spirv::TargetEnvAttr targetAttr) {
  spirv::TargetEnv targetEnv(targetAttr);
  if (targetEnv.allows(spirv::Capability::GraphARM) &&
      targetEnv.allows(spirv::Extension::SPV_ARM_graph) &&
      targetEnv.allows(spirv::Extension::SPV_ARM_tensors)) {
    return success();
  }

  return op->emitOpError()
         << "requires GraphARM capability and SPV_ARM_graph/SPV_ARM_tensors "
            "extensions in spirv.target_env";
}

LogicalResult verifyNoUnsupportedFuncOps(Operation *op) {
  WalkResult result = op->walk([](Operation *op) -> WalkResult {
    if (isa<func::CallOp, func::CallIndirectOp>(op)) {
      op->emitOpError()
          << "is not supported in TOSA to SPIR-V Graph conversion; inline "
             "calls before running this pass";
      return WalkResult::interrupt();
    }
    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      if (funcOp->getParentOfType<func::FuncOp>()) {
        funcOp.emitOpError()
            << "nesting is not supported in TOSA to SPIR-V Graph conversion";
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  return failure(result.wasInterrupted());
}

LogicalResult verifyGraphConstantIdAttrs(Operation *op) {
  WalkResult result = op->walk([](Operation *op) -> WalkResult {
    if (!isa<tosa::ConstOp, tosa::ConstShapeOp>(op))
      return WalkResult::advance();

    auto graphConstantId =
        op->getAttrOfType<IntegerAttr>(graphARMGraphConstantIdAttrName);
    if (!graphConstantId)
      return WalkResult::advance();

    if (graphConstantId.getType().isSignlessInteger(32))
      return WalkResult::advance();

    op->emitOpError() << "requires `" << graphARMGraphConstantIdAttrName
                      << "` to be a signless i32 integer attribute";
    return WalkResult::interrupt();
  });

  return failure(result.wasInterrupted());
}

struct TosaToSPIRVTosa final : impl::TosaToSPIRVTosaBase<TosaToSPIRVTosa> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    Operation *op = getOperation();

    spirv::TargetEnvAttr targetAttr = spirv::lookupTargetEnv(op);
    if (!targetAttr) {
      targetAttr = constructTargetEnvAttrWithCapExtDefaults(context);
    }

    if (failed(verifyGraphTargetEnv(op, targetAttr)) ||
        failed(verifyNoUnsupportedFuncOps(op)) ||
        failed(verifyGraphConstantIdAttrs(op))) {
      signalPassFailure();
      return;
    }

    std::unique_ptr<ConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);

    target->addIllegalDialect<tosa::TosaDialect>();
    target->addIllegalOp<func::CallOp, func::CallIndirectOp>();

    SPIRVTypeConverter typeConverter(targetAttr);
    typeConverter.addConversion([this](IntegerType integerType) {
      return this->convertIntegerType(integerType);
    });
    typeConverter.addConversion([this](TensorType tensorType) {
      return this->convertTensorType(tensorType);
    });
    typeConverter.addConversion([this](tosa::shapeType shapeType) {
      return this->convertShapeType(shapeType);
    });

    populateTosaToSPIRVTosaConversionPatterns(typeConverter, patterns,
                                              targetAttr);
    populateTosaToSPIRVTosaOpsConversionPatterns(typeConverter, patterns);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    if (failed(applyPartialConversion(op, *target, frozenPatterns))) {
      signalPassFailure();
    }
  }

private:
  IntegerType convertIntegerType(IntegerType integerType) {
    if (integerType.getWidth() == 48) {
      return IntegerType::get(&getContext(), 64, integerType.getSignedness());
    }

    if (integerType.getWidth() == 4) {
      return IntegerType::get(&getContext(), 8, integerType.getSignedness());
    }

    return integerType;
  }

  std::optional<SmallVector<int64_t>> convertShape(ArrayRef<int64_t> shape) {
    // Scalar ARM tensors are not supported, so convert them to
    // tensors with shape [1].
    if (shape.empty())
      return SmallVector<int64_t>({1});

    if (llvm::is_contained(shape, 0))
      return std::nullopt;

    bool isPartiallyDynamic =
        llvm::is_contained(shape, ShapedType::kDynamic) &&
        llvm::any_of(shape, [](int64_t dim) { return dim > 0; });
    // Partially shaped ARM tensors are not supported, so convert them to
    // unshaped tensors.
    if (isPartiallyDynamic)
      return SmallVector<int64_t>(shape.size(), ShapedType::kDynamic);
    return SmallVector<int64_t>(shape);
  }

  std::optional<spirv::TensorArmType> convertTensorType(TensorType tensorType) {
    Type elementType = getElementTypeOrSelf(tensorType);
    if (elementType.isIndex())
      elementType = IntegerType::get(&getContext(), 32);
    if (auto integerType = dyn_cast<IntegerType>(elementType))
      elementType = convertIntegerType(integerType);

    SmallVector<int64_t> shape;
    if (tensorType.hasRank()) {
      std::optional<SmallVector<int64_t>> convertedShape =
          convertShape(tensorType.getShape());
      if (!convertedShape)
        return std::nullopt;
      shape = std::move(*convertedShape);
    }

    return spirv::TensorArmType::get(shape, elementType);
  }

  spirv::TensorArmType convertShapeType(tosa::shapeType shapeType) {
    const int64_t rank = std::max(shapeType.getRank(), 1);
    return spirv::TensorArmType::get({rank},
                                     IntegerType::get(&getContext(), 32));
  }
};
} // namespace

std::unique_ptr<Pass> createTosaToSPIRVTosa() {
  return std::make_unique<TosaToSPIRVTosa>();
}

} // namespace tosa
} // namespace mlir
