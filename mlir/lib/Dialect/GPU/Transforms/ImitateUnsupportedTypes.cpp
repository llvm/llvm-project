//===- ImitateUnsupportedTypes.cpp - Unsupported Type Imitation ----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass imitates (bitcast/reinterpret_cast) unsupported types
/// with supported types of same bitwidth. The imitation is done
/// by bitcasting the unspported types to the supported types of same bitwidth.
/// Therefore, the source type and destination type must have the same bitwidth.
/// The imitation is done by using the following operations: arith.bitcast.
///
/// The imitation is often needed when the GPU target (dialect/IR) does not
/// support a certain type but the underlying architecture does. Take SPIR-V for
/// example, it does not support bf16, but an underlying architecture (e.g.,
/// intel pvc gpu) that uses SPIR-V for code-generation does.
/// Therefore, bf16 is neither a valid data type to pass to gpu kernel, nor to
/// be used inside the kernel. To use bf16 data type in a SPIR-V kernel (as a
/// kernel parameter or inside the kernel), bf16 have to be bitcasted (similar
/// to C++ reinterpret_cast) to a supported type (e.g., i16 for Intel GPUs). The
/// SPIR-V kernel can then use the imitated type (i16) in the computation.
/// However, i16 is not the same as bf16 (integer vs float), so the computation
/// can not readily use the imitated type (i16).
///
/// Therefore, this transformation pass is intended to be used in conjuction
/// with other transformation passes such as `EmulateUnsupportedFloats` and
/// `ExtendUnsupportedTypes` that extend the bitwidth of bf16 to f32 and
/// vice-versa.
///
/// Finally, usually, there are instructions available in the target
/// (dialect/IR) that can take advantage of these generated patterns
/// (bf16->i16->f32, f32->bf16->i16), and convert them to the supported
/// types.
/// For example, Intel provides SPIR-V extension ops that can
/// take imitated bf16 (i16) and convert them to f32 and vice-versa.
/// https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/INTEL/SPV_INTEL_bfloat16_conversion.asciidoc
/// https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvintelconvertbf16tof-spirvintelconvertbf16tofop
/// https://mlir.llvm.org/docs/Dialects/SPIR-V/#spirvintelconvertftobf16-spirvintelconvertftobf16op
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <type_traits>
#include <variant>

using namespace mlir;
using namespace mlir::gpu;

namespace mlir {
#define GEN_PASS_DEF_GPUIMITATEUNSUPPORTEDTYPES
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

APFloat bitcastAPIntToAPFloat(const APInt &intValue,
                              const llvm::fltSemantics &semantics) {
  // Get the bit width of the APInt.
  unsigned intBitWidth = intValue.getBitWidth();
  // Get the total bit size required for the APFloat based on the semantics.
  unsigned floatBitWidth = APFloat::getSizeInBits(semantics);
  // Ensure the bit widths match for a direct bitcast.
  assert(intBitWidth == floatBitWidth &&
         "Bitwidth of APInt and APFloat must match for bitcast");

  // Get the raw bit representation of the APInt as a byte vector.
  auto intWords = intValue.getRawData();
  // Create an APFloat with the specified semantics and the raw integer bits.
  APFloat floatValue(semantics, APInt(intBitWidth, *intWords));
  return floatValue;
}

// Get FloatAttr from IntegerAttr.
FloatAttr getFloatAttrFromIntegerAttr(IntegerAttr intAttr, Type dstType,
                                      ConversionPatternRewriter &rewriter) {
  APInt intVal = intAttr.getValue();
  auto floatVal = bitcastAPIntToAPFloat(
      intVal, cast<FloatType>(dstType).getFloatSemantics());
  return rewriter.getFloatAttr(dstType, floatVal);
}
// Get IntegerAttr from FloatAttr.
IntegerAttr getIntegerAttrFromFloatAttr(FloatAttr floatAttr, Type dstType,
                                        ConversionPatternRewriter &rewriter) {
  APFloat floatVal = floatAttr.getValue();
  APInt intVal = floatVal.bitcastToAPInt();
  return rewriter.getIntegerAttr(dstType, intVal);
}

//===----------------------------------------------------------------------===//
// Convertion patterns
//===----------------------------------------------------------------------===//
namespace {

//===----------------------------------------------------------------------===//
// FunctionOp conversion pattern
//===----------------------------------------------------------------------===//
template <typename FuncLikeOp>
struct ConvertFuncOp final : public OpConversionPattern<FuncLikeOp> {
  ConvertFuncOp(MLIRContext *context, TypeConverter &typeConverter,
                ArrayRef<Type> sourceTypes, ArrayRef<Type> targetTypes,
                DenseMap<StringAttr, FunctionType> &convertedFuncTypes)
      : OpConversionPattern<FuncLikeOp>(context),
        typeConverter(typeConverter), // Store the reference
        sourceTypes(sourceTypes), targetTypes(targetTypes),
        convertedFuncTypes(convertedFuncTypes) {}
  using OpConversionPattern<FuncLikeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FuncLikeOp op, typename FuncLikeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle functions a gpu.module
    if (!op->template getParentOfType<gpu::GPUModuleOp>())
      return failure();
    FunctionType oldFuncType = op.getFunctionType();

    // Convert function signature
    TypeConverter::SignatureConversion signatureConverter(
        oldFuncType.getNumInputs());
    for (const auto &argType :
         llvm::enumerate(op.getFunctionType().getInputs())) {
      auto convertedType = typeConverter.convertType(argType.value());
      if (!convertedType)
        return failure();
      signatureConverter.addInputs(argType.index(), convertedType);
    }
    SmallVector<Type, 4> newResultTypes;
    for (const auto &resultType : llvm::enumerate(oldFuncType.getResults())) {
      auto convertedType = typeConverter.convertType(resultType.value());
      if (!convertedType)
        return failure();
      newResultTypes.push_back(convertedType);
    }

    // Convert function signature
    FunctionType newFuncType = rewriter.getFunctionType(
        signatureConverter.getConvertedTypes(), newResultTypes);

    if (!newFuncType)
      return rewriter.notifyMatchFailure(op, "could not convert function "
                                             "type");

    // Create new GPU function with converted type
    auto newFuncOp =
        rewriter.create<FuncLikeOp>(op.getLoc(), op.getName(), newFuncType);

    newFuncOp.setVisibility(op.getVisibility());
    // Copy attributes
    for (auto attr : op->getAttrs()) {
      // Skip the function_type attribute since it is already set by
      // the newFuncType and we don't want to overwrite it.
      if (attr.getName() != op.getFunctionTypeAttrName() &&
          attr.getName() != SymbolTable::getSymbolAttrName())
        newFuncOp->setAttr(attr.getName(), attr.getValue());
    }

    newFuncOp.getRegion().getBlocks().clear();
    // Inline region approach
    rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    // Convert block argument types using the type converter
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), typeConverter,
                                           &signatureConverter))) {
      return rewriter.notifyMatchFailure(op, "could not convert region "
                                             "types");
    }

    if (!op.use_empty()) {
      op.emitError("Cannot erase func: still has uses");
    }
    for (Operation *user : op->getUsers()) {
      user->emitRemark() << "User of function " << op.getName();
    }
    rewriter.eraseOp(op);
    // Add the converted function type to the map
    newFuncOp.getNameAttr().getValue();
    convertedFuncTypes[newFuncOp.getNameAttr()] = newFuncType;
    return success();
  }

private:
  TypeConverter &typeConverter; // Store a reference
  ArrayRef<Type> sourceTypes;
  ArrayRef<Type> targetTypes;
  DenseMap<StringAttr, FunctionType> &convertedFuncTypes;
};

//===----------------------------------------------------------------------===//
// CallOp conversion pattern
//===----------------------------------------------------------------------===//
struct ConvertCallOp : OpConversionPattern<func::CallOp> {
  ConvertCallOp(MLIRContext *context, TypeConverter &typeConverter,
                const DenseMap<StringAttr, FunctionType> &convertedFuncTypes)
      : OpConversionPattern(context), convertedFuncTypes(convertedFuncTypes) {}

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto callee = op.getCalleeAttr();

    auto it = convertedFuncTypes.find(
        StringAttr::get(callee.getContext(), callee.getValue()));
    if (it == convertedFuncTypes.end())
      return rewriter.notifyMatchFailure(
          op, "Callee signature not converted. Perhaps the callee is not in "
              "the same gpu module as the caller.");

    auto newResultTypes = it->second.getResults();
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, callee.getValue(), newResultTypes, adaptor.getOperands());

    return success();
  }

private:
  const DenseMap<StringAttr, FunctionType> &convertedFuncTypes;
};

//===----------------------------------------------------------------------===//
// GPULaunchFuncOp conversion pattern
//===----------------------------------------------------------------------===//
struct ConvertGPULaunchFuncOp : OpConversionPattern<gpu::LaunchFuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(gpu::LaunchFuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<KernelDim3> clusterSizeOpernads =
        op.hasClusterSize()
            ? std::optional<gpu::KernelDim3>(op.getClusterSizeOperandValues())
            : std::nullopt;

    // Create the new launch_func.
    auto newOp = rewriter.create<gpu::LaunchFuncOp>(
        op.getLoc(), adaptor.getKernel(), op.getGridSizeOperandValues(),
        op.getBlockSizeOperandValues(), op.getDynamicSharedMemorySize(),
        adaptor.getKernelOperands(), op.getAsyncObject(), clusterSizeOpernads);

    // Copy block size and grid size attributes
    newOp->setAttrs(op->getAttrs());
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ArithConstantOp conversion pattern
//===----------------------------------------------------------------------===//
struct ConvertArithConstantOp : OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  ConvertArithConstantOp(MLIRContext *context, TypeConverter &typeConverter,
                         ArrayRef<Type> sourceTypes, ArrayRef<Type> targetTypes)
      : OpConversionPattern(context),
        typeConverter(typeConverter), // Store the reference.
        sourceTypes(sourceTypes), targetTypes(targetTypes) {}

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type srcType = op.getType();
    Type dstType = typeConverter.convertType(srcType);
    if (!dstType || dstType == srcType)
      return failure();

    Attribute value = op.getValue();
    Value newConstOp = nullptr;

    // When source is IntegerAttr.
    if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
      APInt intVal = intAttr.getValue();
      if (isa<FloatType>(dstType)) {
        auto newAttr = getFloatAttrFromIntegerAttr(intAttr, dstType, rewriter);
        newConstOp =
            rewriter.create<arith::ConstantOp>(op.getLoc(), dstType, newAttr);
      } else if (isa<IntegerType>(dstType)) {
        auto newAttr = rewriter.getIntegerAttr(dstType, intVal);
        newConstOp =
            rewriter.create<arith::ConstantOp>(op.getLoc(), dstType, newAttr);
      } else {
        return rewriter.notifyMatchFailure(
            op, "expected integer or float target type for constant");
      }
    }

    // When source is FloatAttr.
    else if (auto floatAttr = dyn_cast<FloatAttr>(value)) {
      if (llvm::isa<IntegerType>(dstType)) {
        auto newAttr =
            getIntegerAttrFromFloatAttr(floatAttr, dstType, rewriter);
        newConstOp =
            rewriter.create<arith::ConstantOp>(op.getLoc(), dstType, newAttr);
      } else if (llvm::isa<FloatType>(dstType)) {
        auto newAttr = rewriter.getFloatAttr(dstType, floatAttr.getValue());
        newConstOp =
            rewriter.create<arith::ConstantOp>(op.getLoc(), dstType, newAttr);
      } else {
        return rewriter.notifyMatchFailure(
            op, "expected integer or float target type for constant");
      }
    }
    // Handle DenseElementsAttr.
    else if (auto denseAttr = dyn_cast<DenseElementsAttr>(value)) {
      Type newEltType;
      if (auto shapedType = dyn_cast<ShapedType>(dstType))
        newEltType = shapedType.getElementType();
      else
        return rewriter.notifyMatchFailure(
            op, "expected shaped type for dense constant");

      SmallVector<Attribute> newValues;
      for (Attribute attr : denseAttr.getValues<Attribute>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
          if (llvm::isa<FloatType>(newEltType)) {
            auto newAttr =
                getFloatAttrFromIntegerAttr(intAttr, newEltType, rewriter);
            newValues.push_back(newAttr);
          } else if (llvm::isa<IntegerType>(newEltType)) {
            newValues.push_back(
                rewriter.getIntegerAttr(newEltType, intAttr.getValue()));
          } else {
            return rewriter.notifyMatchFailure(
                op, "unsupported target element type in dense constant");
          }
        } else if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
          if (llvm::isa<IntegerType>(newEltType)) {
            auto newAttr =
                getIntegerAttrFromFloatAttr(floatAttr, newEltType, rewriter);
            newValues.push_back(newAttr);
          } else if (llvm::isa<FloatType>(newEltType))
            newValues.push_back(
                rewriter.getFloatAttr(newEltType, floatAttr.getValue()));
          else
            return rewriter.notifyMatchFailure(
                op, "unsupported target element type in dense constant");
        } else {
          return rewriter.notifyMatchFailure(
              op, "unsupported target element type in dense constant");
        }
      }

      auto newAttr =
          DenseElementsAttr::get(cast<ShapedType>(dstType), newValues);
      newConstOp =
          rewriter.create<arith::ConstantOp>(op.getLoc(), dstType, newAttr);
    }
    if (!newConstOp)
      return rewriter.notifyMatchFailure(
          op, "unsupported constant type for source to target conversion");

    auto bitcastOp =
        rewriter.create<arith::BitcastOp>(op.getLoc(), srcType, newConstOp);
    rewriter.replaceOp(op, bitcastOp.getResult());
    return success();
  }

private:
  TypeConverter &typeConverter; // Store a reference.
  ArrayRef<Type> sourceTypes;
  ArrayRef<Type> targetTypes;
};

//===----------------------------------------------------------------------===//
// GenericOp conversion pattern
//===----------------------------------------------------------------------===//
struct ConvertOpWithSourceType final : ConversionPattern {
  ConvertOpWithSourceType(MLIRContext *context,
                          const TypeConverter &typeConverter,
                          ArrayRef<Type> sourceTypes,
                          ArrayRef<Type> targetTypes)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag{}, 1, context),
        sourceTypes(sourceTypes), targetTypes(targetTypes) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type, 4> newResultTypes;
    for (Type t : op->getResultTypes()) {
      Type converted = typeConverter->convertType(t);
      if (!converted)
        return failure();
      newResultTypes.push_back(converted);
    }

    // Clone the op manually with the converted result types
    OperationState state(op->getLoc(), op->getName().getStringRef());
    state.addOperands(operands);
    state.addTypes(newResultTypes);
    state.addAttributes(op->getAttrs());

    for ([[maybe_unused]] auto &region : op->getRegions())
      state.regions.emplace_back();

    Operation *newOp = rewriter.create(state);
    // Transfer regions and convert them
    for (auto [oldRegion, newRegion] :
         llvm::zip(op->getRegions(), newOp->getRegions())) {
      if (!oldRegion.empty()) {
        newRegion.takeBody(oldRegion);
        if (failed(rewriter.convertRegionTypes(&newRegion, *typeConverter))) {
          return rewriter.notifyMatchFailure(op,
                                             "region type conversion failed");
        }
      }
    }

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }

private:
  ArrayRef<Type> sourceTypes;
  ArrayRef<Type> targetTypes;
};

} // namespace

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

void mlir::populateImitateUnsupportedTypesTypeConverter(
    TypeConverter &typeConverter, ArrayRef<Type> sourceTypes,
    ArrayRef<Type> targetTypes) {
  auto srcTypes = SmallVector<Type>(sourceTypes);
  auto tgtTypes = SmallVector<Type>(targetTypes);

  assert(sourceTypes.size() == targetTypes.size() &&
         "Source and target types must have same size");

  typeConverter.addConversion([srcTypes, tgtTypes](Type type) -> Type {
    if (type.isIntOrIndexOrFloat()) {
      for (auto [src, tgt] : llvm::zip_equal(srcTypes, tgtTypes)) {
        if (type == src)
          return tgt;
      }
    } else if (auto memref = llvm::dyn_cast<MemRefType>(type)) {
      Type elemType = memref.getElementType();
      for (auto [src, tgt] : llvm::zip_equal(srcTypes, tgtTypes)) {
        if (elemType == src)
          return MemRefType::get(memref.getShape(), tgt, memref.getLayout(),
                                 memref.getMemorySpace());
      }
    } else if (auto vec = llvm::dyn_cast<VectorType>(type)) {
      Type elemType = vec.getElementType();
      for (auto [src, tgt] : llvm::zip_equal(srcTypes, tgtTypes)) {
        if (elemType == src)
          return VectorType::get(vec.getShape(), tgt);
      }
    }
    return type;
  });

  auto materializeCast = [](OpBuilder &builder, Type resultType,
                            ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1 && "Expected single input");
    Type inputType = inputs[0].getType();
    if ((resultType.isIntOrIndexOrFloat() || isa<VectorType>(resultType) ||
         isa<MemRefType>(resultType)) &&
        (inputType.isIntOrIndexOrFloat() || isa<VectorType>(inputType) ||
         isa<MemRefType>(inputType))) {
      return builder.create<arith::BitcastOp>(loc, resultType, inputs[0])
          .getResult();
    }
    return nullptr;
  };

  typeConverter.addSourceMaterialization(materializeCast);
  typeConverter.addTargetMaterialization(materializeCast);
}

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void mlir::populateImitateUnsupportedTypesConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    ArrayRef<Type> sourceTypes, ArrayRef<Type> targetTypes,
    DenseMap<StringAttr, FunctionType> &convertedFuncTypes) {
  auto ctx = patterns.getContext();
  auto srcTypes = SmallVector<Type>(sourceTypes);
  auto tgtTypes = SmallVector<Type>(targetTypes);
  assert(srcTypes.size() == tgtTypes.size() &&
         "Source and target types must have same size");

  patterns.add<ConvertOpWithSourceType>(ctx, typeConverter, srcTypes, tgtTypes);
  patterns.add<ConvertFuncOp<gpu::GPUFuncOp>, ConvertFuncOp<func::FuncOp>>(
      ctx, typeConverter, srcTypes, tgtTypes, convertedFuncTypes);
  patterns.add<ConvertCallOp>(ctx, typeConverter, convertedFuncTypes);
  patterns.add<ConvertArithConstantOp>(ctx, typeConverter, srcTypes, tgtTypes);
  patterns.add<ConvertGPULaunchFuncOp>(ctx);
}

//===----------------------------------------------------------------------===//
// Conversion Legality configuration
//===----------------------------------------------------------------------===//

void mlir::configureImitateUnsupportedTypesLegality(
    ConversionTarget &target, TypeConverter &typeConverter) {
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<math::MathDialect>();
  // Make Memref, func dialect legal for all ops in host code
  target.addDynamicallyLegalDialect<memref::MemRefDialect>([&](Operation *op) {
    if (op->getParentOfType<gpu::GPUModuleOp>())
      return typeConverter.isLegal(op);
    else
      return true;
  });

  target.addDynamicallyLegalDialect<gpu::GPUDialect>([&](Operation *op) {
    if (op->getParentOfType<gpu::GPUModuleOp>())
      return typeConverter.isLegal(op);
    return true;
  });

  target.addDynamicallyLegalDialect<func::FuncDialect>([&](Operation *op) {
    if (op->getParentOfType<gpu::GPUModuleOp>())
      return typeConverter.isLegal(op);
    else
      return true;
  });

  target.addLegalOp<gpu::GPUModuleOp>();
  // Manually mark arithmetic-performing vector instructions.
  target.addLegalOp<vector::ContractionOp, vector::ReductionOp,
                    vector::MultiDimReductionOp, vector::FMAOp,
                    vector::OuterProductOp, vector::MatmulOp, vector::ScanOp,
                    vector::SplatOp>();
  target.addDynamicallyLegalOp<arith::ConstantOp>([&](arith::ConstantOp op) {
    return typeConverter.isLegal(op.getType());
  });
  target.addDynamicallyLegalOp<gpu::GPUFuncOp>([&](gpu::GPUFuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType());
  });
  target.addDynamicallyLegalOp<gpu::LaunchFuncOp>(
      [&](gpu::LaunchFuncOp op) { return typeConverter.isLegal(op); });
  // Only convert functions and function calls in gpu.module
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    if (op->getParentOfType<gpu::GPUModuleOp>())
      return typeConverter.isSignatureLegal(op.getFunctionType());
    return true;
  });
  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    if (op->getParentOfType<gpu::GPUModuleOp>())
      return typeConverter.isSignatureLegal(op.getCalleeType());
    return true;
  });

  // Mark unknown ops that are inside gpu.module, and one of its's operand is
  // a memref type as dynamically legal.
  target.markUnknownOpDynamicallyLegal([&typeConverter](Operation *op) -> bool {
    // Check if the operation is inside a gpu.module.
    if (op->getParentOfType<gpu::GPUModuleOp>()) {
      return typeConverter.isLegal(op);
    }
    return true; // If not in gpu.module, mark it as legal.
  });
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {

struct GpuImitateUnsupportedTypesPass
    : public impl::GpuImitateUnsupportedTypesBase<
          GpuImitateUnsupportedTypesPass> {
  using Base::Base;

  SmallVector<Type> sourceTypes;
  SmallVector<Type> targetTypes;
  TypeConverter typeConverter;

  LogicalResult initialize(MLIRContext *ctx) override {
    // Parse source types.
    for (StringRef sourceTypeStr : sourceTypeStrs) {
      std::optional<Type> maybeSourceType =
          arith::parseIntOrFloatType(ctx, sourceTypeStr);

      if (!maybeSourceType) {
        emitError(UnknownLoc::get(ctx),
                  "could not map source type '" + sourceTypeStr +
                      "' to a known integer or floating-point type.");
        return failure();
      }
      sourceTypes.push_back(*maybeSourceType);
    }
    if (sourceTypes.empty()) {
      (void)emitOptionalWarning(std::nullopt, "no source types "
                                              "specified, type "
                                              "imitation will do "
                                              "nothing");
    }

    // Parse target types.
    for (StringRef targetTypeStr : targetTypeStrs) {
      std::optional<Type> maybeTargetType =
          arith::parseIntOrFloatType(ctx, targetTypeStr);

      if (!maybeTargetType) {
        emitError(UnknownLoc::get(ctx),
                  "could not map target type '" + targetTypeStr +
                      "' to a known integer or floating-point type");
        return failure();
      }
      targetTypes.push_back(*maybeTargetType);

      if (llvm::is_contained(sourceTypes, *maybeTargetType)) {
        emitError(UnknownLoc::get(ctx),
                  "target type cannot be an unsupported source type");
        return failure();
      }
    }
    if (targetTypes.empty()) {
      (void)emitOptionalWarning(
          std::nullopt,
          "no target types specified, type imitation will do nothing");
    }

    if (sourceTypes.size() != targetTypes.size()) {
      emitError(UnknownLoc::get(ctx),
                "source and target types must have the same size");
      return failure();
    }
    // Set up the type converter.
    populateImitateUnsupportedTypesTypeConverter(typeConverter, sourceTypes,
                                                 targetTypes);
    return success();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    Operation *op = getOperation();

    // Populate the conversion patterns.
    RewritePatternSet patterns(ctx);
    DenseMap<StringAttr, FunctionType> convertedFuncTypes;
    populateImitateUnsupportedTypesConversionPatterns(
        patterns, typeConverter, sourceTypes, targetTypes, convertedFuncTypes);

    // Set up conversion target and configure the legality of the conversion.
    ConversionTarget target(*ctx);
    configureImitateUnsupportedTypesLegality(target, typeConverter);

    // Apply the conversion.
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
