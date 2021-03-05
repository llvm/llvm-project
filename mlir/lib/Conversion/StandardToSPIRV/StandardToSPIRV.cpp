//===- StandardToSPIRV.cpp - Standard to SPIR-V Patterns ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert standard dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/SPIRV/Utils/LayoutUtils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "std-to-spirv-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Returns true if the given `type` is a boolean scalar or vector type.
static bool isBoolScalarOrVector(Type type) {
  if (type.isInteger(1))
    return true;
  if (auto vecType = type.dyn_cast<VectorType>())
    return vecType.getElementType().isInteger(1);
  return false;
}

/// Converts the given `srcAttr` into a boolean attribute if it holds an
/// integral value. Returns null attribute if conversion fails.
static BoolAttr convertBoolAttr(Attribute srcAttr, Builder builder) {
  if (auto boolAttr = srcAttr.dyn_cast<BoolAttr>())
    return boolAttr;
  if (auto intAttr = srcAttr.dyn_cast<IntegerAttr>())
    return builder.getBoolAttr(intAttr.getValue().getBoolValue());
  return BoolAttr();
}

/// Converts the given `srcAttr` to a new attribute of the given `dstType`.
/// Returns null attribute if conversion fails.
static IntegerAttr convertIntegerAttr(IntegerAttr srcAttr, IntegerType dstType,
                                      Builder builder) {
  // If the source number uses less active bits than the target bitwidth, then
  // it should be safe to convert.
  if (srcAttr.getValue().isIntN(dstType.getWidth()))
    return builder.getIntegerAttr(dstType, srcAttr.getInt());

  // XXX: Try again by interpreting the source number as a signed value.
  // Although integers in the standard dialect are signless, they can represent
  // a signed number. It's the operation decides how to interpret. This is
  // dangerous, but it seems there is no good way of handling this if we still
  // want to change the bitwidth. Emit a message at least.
  if (srcAttr.getValue().isSignedIntN(dstType.getWidth())) {
    auto dstAttr = builder.getIntegerAttr(dstType, srcAttr.getInt());
    LLVM_DEBUG(llvm::dbgs() << "attribute '" << srcAttr << "' converted to '"
                            << dstAttr << "' for type '" << dstType << "'\n");
    return dstAttr;
  }

  LLVM_DEBUG(llvm::dbgs() << "attribute '" << srcAttr
                          << "' illegal: cannot fit into target type '"
                          << dstType << "'\n");
  return IntegerAttr();
}

/// Converts the given `srcAttr` to a new attribute of the given `dstType`.
/// Returns null attribute if `dstType` is not 32-bit or conversion fails.
static FloatAttr convertFloatAttr(FloatAttr srcAttr, FloatType dstType,
                                  Builder builder) {
  // Only support converting to float for now.
  if (!dstType.isF32())
    return FloatAttr();

  // Try to convert the source floating-point number to single precision.
  APFloat dstVal = srcAttr.getValue();
  bool losesInfo = false;
  APFloat::opStatus status =
      dstVal.convert(APFloat::IEEEsingle(), APFloat::rmTowardZero, &losesInfo);
  if (status != APFloat::opOK || losesInfo) {
    LLVM_DEBUG(llvm::dbgs()
               << srcAttr << " illegal: cannot fit into converted type '"
               << dstType << "'\n");
    return FloatAttr();
  }

  return builder.getF32FloatAttr(dstVal.convertToFloat());
}

/// Returns signed remainder for `lhs` and `rhs` and lets the result follow
/// the sign of `signOperand`.
///
/// Note that this is needed for Vulkan. Per the Vulkan's SPIR-V environment
/// spec, "for the OpSRem and OpSMod instructions, if either operand is negative
/// the result is undefined."  So we cannot directly use spv.SRem/spv.SMod
/// if either operand can be negative. Emulate it via spv.UMod.
static Value emulateSignedRemainder(Location loc, Value lhs, Value rhs,
                                    Value signOperand, OpBuilder &builder) {
  assert(lhs.getType() == rhs.getType());
  assert(lhs == signOperand || rhs == signOperand);

  Type type = lhs.getType();

  // Calculate the remainder with spv.UMod.
  Value lhsAbs = builder.create<spirv::GLSLSAbsOp>(loc, type, lhs);
  Value rhsAbs = builder.create<spirv::GLSLSAbsOp>(loc, type, rhs);
  Value abs = builder.create<spirv::UModOp>(loc, lhsAbs, rhsAbs);

  // Fix the sign.
  Value isPositive;
  if (lhs == signOperand)
    isPositive = builder.create<spirv::IEqualOp>(loc, lhs, lhsAbs);
  else
    isPositive = builder.create<spirv::IEqualOp>(loc, rhs, rhsAbs);
  Value absNegate = builder.create<spirv::SNegateOp>(loc, type, abs);
  return builder.create<spirv::SelectOp>(loc, type, isPositive, abs, absNegate);
}

/// Returns the offset of the value in `targetBits` representation.
///
/// `srcIdx` is an index into a 1-D array with each element having `sourceBits`.
/// It's assumed to be non-negative.
///
/// When accessing an element in the array treating as having elements of
/// `targetBits`, multiple values are loaded in the same time. The method
/// returns the offset where the `srcIdx` locates in the value. For example, if
/// `sourceBits` equals to 8 and `targetBits` equals to 32, the x-th element is
/// located at (x % 4) * 8. Because there are four elements in one i32, and one
/// element has 8 bits.
static Value getOffsetForBitwidth(Location loc, Value srcIdx, int sourceBits,
                                  int targetBits, OpBuilder &builder) {
  assert(targetBits % sourceBits == 0);
  IntegerType targetType = builder.getIntegerType(targetBits);
  IntegerAttr idxAttr =
      builder.getIntegerAttr(targetType, targetBits / sourceBits);
  auto idx = builder.create<spirv::ConstantOp>(loc, targetType, idxAttr);
  IntegerAttr srcBitsAttr = builder.getIntegerAttr(targetType, sourceBits);
  auto srcBitsValue =
      builder.create<spirv::ConstantOp>(loc, targetType, srcBitsAttr);
  auto m = builder.create<spirv::UModOp>(loc, srcIdx, idx);
  return builder.create<spirv::IMulOp>(loc, targetType, m, srcBitsValue);
}

/// Returns an adjusted spirv::AccessChainOp. Based on the
/// extension/capabilities, certain integer bitwidths `sourceBits` might not be
/// supported. During conversion if a memref of an unsupported type is used,
/// load/stores to this memref need to be modified to use a supported higher
/// bitwidth `targetBits` and extracting the required bits. For an accessing a
/// 1D array (spv.array or spv.rt_array), the last index is modified to load the
/// bits needed. The extraction of the actual bits needed are handled
/// separately. Note that this only works for a 1-D tensor.
static Value adjustAccessChainForBitwidth(SPIRVTypeConverter &typeConverter,
                                          spirv::AccessChainOp op,
                                          int sourceBits, int targetBits,
                                          OpBuilder &builder) {
  assert(targetBits % sourceBits == 0);
  const auto loc = op.getLoc();
  IntegerType targetType = builder.getIntegerType(targetBits);
  IntegerAttr attr =
      builder.getIntegerAttr(targetType, targetBits / sourceBits);
  auto idx = builder.create<spirv::ConstantOp>(loc, targetType, attr);
  auto lastDim = op->getOperand(op.getNumOperands() - 1);
  auto indices = llvm::to_vector<4>(op.indices());
  // There are two elements if this is a 1-D tensor.
  assert(indices.size() == 2);
  indices.back() = builder.create<spirv::SDivOp>(loc, lastDim, idx);
  Type t = typeConverter.convertType(op.component_ptr().getType());
  return builder.create<spirv::AccessChainOp>(loc, t, op.base_ptr(), indices);
}

/// Returns the shifted `targetBits`-bit value with the given offset.
static Value shiftValue(Location loc, Value value, Value offset, Value mask,
                        int targetBits, OpBuilder &builder) {
  Type targetType = builder.getIntegerType(targetBits);
  Value result = builder.create<spirv::BitwiseAndOp>(loc, value, mask);
  return builder.create<spirv::ShiftLeftLogicalOp>(loc, targetType, result,
                                                   offset);
}

/// Returns true if the allocations of type `t` can be lowered to SPIR-V.
static bool isAllocationSupported(MemRefType t) {
  // Currently only support workgroup local memory allocations with static
  // shape and int or float or vector of int or float element type.
  if (!(t.hasStaticShape() &&
        SPIRVTypeConverter::getMemorySpaceForStorageClass(
            spirv::StorageClass::Workgroup) == t.getMemorySpaceAsInt()))
    return false;
  Type elementType = t.getElementType();
  if (auto vecType = elementType.dyn_cast<VectorType>())
    elementType = vecType.getElementType();
  return elementType.isIntOrFloat();
}

/// Returns the scope to use for atomic operations use for emulating store
/// operations of unsupported integer bitwidths, based on the memref
/// type. Returns None on failure.
static Optional<spirv::Scope> getAtomicOpScope(MemRefType t) {
  Optional<spirv::StorageClass> storageClass =
      SPIRVTypeConverter::getStorageClassForMemorySpace(
          t.getMemorySpaceAsInt());
  if (!storageClass)
    return {};
  switch (*storageClass) {
  case spirv::StorageClass::StorageBuffer:
    return spirv::Scope::Device;
  case spirv::StorageClass::Workgroup:
    return spirv::Scope::Workgroup;
  default: {
  }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

// Note that DRR cannot be used for the patterns in this file: we may need to
// convert type along the way, which requires ConversionPattern. DRR generates
// normal RewritePattern.

namespace {

/// Converts an allocation operation to SPIR-V. Currently only supports lowering
/// to Workgroup memory when the size is constant.  Note that this pattern needs
/// to be applied in a pass that runs at least at spv.module scope since it wil
/// ladd global variables into the spv.module.
class AllocOpPattern final : public OpConversionPattern<AllocOp> {
public:
  using OpConversionPattern<AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AllocOp operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType allocType = operation.getType();
    if (!isAllocationSupported(allocType))
      return operation.emitError("unhandled allocation type");

    // Get the SPIR-V type for the allocation.
    Type spirvType = getTypeConverter()->convertType(allocType);

    // Insert spv.GlobalVariable for this allocation.
    Operation *parent =
        SymbolTable::getNearestSymbolTable(operation->getParentOp());
    if (!parent)
      return failure();
    Location loc = operation.getLoc();
    spirv::GlobalVariableOp varOp;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block &entryBlock = *parent->getRegion(0).begin();
      rewriter.setInsertionPointToStart(&entryBlock);
      auto varOps = entryBlock.getOps<spirv::GlobalVariableOp>();
      std::string varName =
          std::string("__workgroup_mem__") +
          std::to_string(std::distance(varOps.begin(), varOps.end()));
      varOp = rewriter.create<spirv::GlobalVariableOp>(loc, spirvType, varName,
                                                       /*initializer=*/nullptr);
    }

    // Get pointer to global variable at the current scope.
    rewriter.replaceOpWithNewOp<spirv::AddressOfOp>(operation, varOp);
    return success();
  }
};

/// Removed a deallocation if it is a supported allocation. Currently only
/// removes deallocation if the memory space is workgroup memory.
class DeallocOpPattern final : public OpConversionPattern<DeallocOp> {
public:
  using OpConversionPattern<DeallocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DeallocOp operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType deallocType = operation.memref().getType().cast<MemRefType>();
    if (!isAllocationSupported(deallocType))
      return operation.emitError("unhandled deallocation type");
    rewriter.eraseOp(operation);
    return success();
  }
};

/// Converts unary and binary standard operations to SPIR-V operations.
template <typename StdOp, typename SPIRVOp>
class UnaryAndBinaryOpPattern final : public OpConversionPattern<StdOp> {
public:
  using OpConversionPattern<StdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StdOp operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() <= 2);
    auto dstType = this->getTypeConverter()->convertType(operation.getType());
    if (!dstType)
      return failure();
    if (SPIRVOp::template hasTrait<OpTrait::spirv::UnsignedOp>() &&
        dstType != operation.getType()) {
      return operation.emitError(
          "bitwidth emulation is not implemented yet on unsigned op");
    }
    rewriter.template replaceOpWithNewOp<SPIRVOp>(operation, dstType, operands);
    return success();
  }
};

/// Converts std.remi_signed to SPIR-V ops.
///
/// This cannot be merged into the template unary/binary pattern due to
/// Vulkan restrictions over spv.SRem and spv.SMod.
class SignedRemIOpPattern final : public OpConversionPattern<SignedRemIOp> {
public:
  using OpConversionPattern<SignedRemIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SignedRemIOp remOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts bitwise standard operations to SPIR-V operations. This is a special
/// pattern other than the BinaryOpPatternPattern because if the operands are
/// boolean values, SPIR-V uses different operations (`SPIRVLogicalOp`). For
/// non-boolean operands, SPIR-V should use `SPIRVBitwiseOp`.
template <typename StdOp, typename SPIRVLogicalOp, typename SPIRVBitwiseOp>
class BitwiseOpPattern final : public OpConversionPattern<StdOp> {
public:
  using OpConversionPattern<StdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StdOp operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() == 2);
    auto dstType =
        this->getTypeConverter()->convertType(operation.getResult().getType());
    if (!dstType)
      return failure();
    if (isBoolScalarOrVector(operands.front().getType())) {
      rewriter.template replaceOpWithNewOp<SPIRVLogicalOp>(operation, dstType,
                                                           operands);
    } else {
      rewriter.template replaceOpWithNewOp<SPIRVBitwiseOp>(operation, dstType,
                                                           operands);
    }
    return success();
  }
};

/// Converts composite std.constant operation to spv.Constant.
class ConstantCompositeOpPattern final
    : public OpConversionPattern<ConstantOp> {
public:
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp constOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts scalar std.constant operation to spv.Constant.
class ConstantScalarOpPattern final : public OpConversionPattern<ConstantOp> {
public:
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp constOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts floating-point comparison operations to SPIR-V ops.
class CmpFOpPattern final : public OpConversionPattern<CmpFOp> {
public:
  using OpConversionPattern<CmpFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CmpFOp cmpFOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts floating point NaN check to SPIR-V ops. This pattern requires
/// Kernel capability.
class CmpFOpNanKernelPattern final : public OpConversionPattern<CmpFOp> {
public:
  using OpConversionPattern<CmpFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CmpFOp cmpFOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts floating point NaN check to SPIR-V ops. This pattern does not
/// require additional capability.
class CmpFOpNanNonePattern final : public OpConversionPattern<CmpFOp> {
public:
  using OpConversionPattern<CmpFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CmpFOp cmpFOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts integer compare operation on i1 type operands to SPIR-V ops.
class BoolCmpIOpPattern final : public OpConversionPattern<CmpIOp> {
public:
  using OpConversionPattern<CmpIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CmpIOp cmpIOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts integer compare operation to SPIR-V ops.
class CmpIOpPattern final : public OpConversionPattern<CmpIOp> {
public:
  using OpConversionPattern<CmpIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CmpIOp cmpIOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts std.load to spv.Load.
class IntLoadOpPattern final : public OpConversionPattern<LoadOp> {
public:
  using OpConversionPattern<LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LoadOp loadOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts std.load to spv.Load.
class LoadOpPattern final : public OpConversionPattern<LoadOp> {
public:
  using OpConversionPattern<LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LoadOp loadOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts std.return to spv.Return.
class ReturnOpPattern final : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp returnOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts std.select to spv.Select.
class SelectOpPattern final : public OpConversionPattern<SelectOp> {
public:
  using OpConversionPattern<SelectOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SelectOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts std.store to spv.Store on integers.
class IntStoreOpPattern final : public OpConversionPattern<StoreOp> {
public:
  using OpConversionPattern<StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StoreOp storeOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts std.store to spv.Store.
class StoreOpPattern final : public OpConversionPattern<StoreOp> {
public:
  using OpConversionPattern<StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StoreOp storeOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts std.zexti to spv.Select if the type of source is i1 or vector of
/// i1.
class ZeroExtendI1Pattern final : public OpConversionPattern<ZeroExtendIOp> {
public:
  using OpConversionPattern<ZeroExtendIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZeroExtendIOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = operands.front().getType();
    if (!isBoolScalarOrVector(srcType))
      return failure();

    auto dstType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    Location loc = op.getLoc();
    Value zero = spirv::ConstantOp::getZero(dstType, loc, rewriter);
    Value one = spirv::ConstantOp::getOne(dstType, loc, rewriter);
    rewriter.template replaceOpWithNewOp<spirv::SelectOp>(
        op, dstType, operands.front(), one, zero);
    return success();
  }
};

/// Converts std.trunci to spv.Select if the type of result is i1 or vector of
/// i1.
class TruncI1Pattern final : public OpConversionPattern<TruncateIOp> {
public:
  using OpConversionPattern<TruncateIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TruncateIOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!isBoolScalarOrVector(dstType))
      return failure();

    Location loc = op.getLoc();
    auto srcType = operands.front().getType();
    // Check if (x & 1) == 1.
    Value mask = spirv::ConstantOp::getOne(srcType, loc, rewriter);
    Value maskedSrc =
        rewriter.create<spirv::BitwiseAndOp>(loc, srcType, operands[0], mask);
    Value isOne = rewriter.create<spirv::IEqualOp>(loc, maskedSrc, mask);

    Value zero = spirv::ConstantOp::getZero(dstType, loc, rewriter);
    Value one = spirv::ConstantOp::getOne(dstType, loc, rewriter);
    rewriter.replaceOpWithNewOp<spirv::SelectOp>(op, dstType, isOne, one, zero);
    return success();
  }
};

/// Converts std.uitofp to spv.Select if the type of source is i1 or vector of
/// i1.
class UIToFPI1Pattern final : public OpConversionPattern<UIToFPOp> {
public:
  using OpConversionPattern<UIToFPOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UIToFPOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = operands.front().getType();
    if (!isBoolScalarOrVector(srcType))
      return failure();

    auto dstType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    Location loc = op.getLoc();
    Value zero = spirv::ConstantOp::getZero(dstType, loc, rewriter);
    Value one = spirv::ConstantOp::getOne(dstType, loc, rewriter);
    rewriter.template replaceOpWithNewOp<spirv::SelectOp>(
        op, dstType, operands.front(), one, zero);
    return success();
  }
};

/// Converts type-casting standard operations to SPIR-V operations.
template <typename StdOp, typename SPIRVOp>
class TypeCastingOpPattern final : public OpConversionPattern<StdOp> {
public:
  using OpConversionPattern<StdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StdOp operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() == 1);
    auto srcType = operands.front().getType();
    auto dstType =
        this->getTypeConverter()->convertType(operation.getResult().getType());
    if (isBoolScalarOrVector(srcType) || isBoolScalarOrVector(dstType))
      return failure();
    if (dstType == srcType) {
      // Due to type conversion, we are seeing the same source and target type.
      // Then we can just erase this operation by forwarding its operand.
      rewriter.replaceOp(operation, operands.front());
    } else {
      rewriter.template replaceOpWithNewOp<SPIRVOp>(operation, dstType,
                                                    operands);
    }
    return success();
  }
};

/// Converts std.xor to SPIR-V operations.
class XOrOpPattern final : public OpConversionPattern<XOrOp> {
public:
  using OpConversionPattern<XOrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(XOrOp xorOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

//===----------------------------------------------------------------------===//
// SignedRemIOpPattern
//===----------------------------------------------------------------------===//

LogicalResult SignedRemIOpPattern::matchAndRewrite(
    SignedRemIOp remOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  Value result = emulateSignedRemainder(remOp.getLoc(), operands[0],
                                        operands[1], operands[0], rewriter);
  rewriter.replaceOp(remOp, result);

  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp with composite type.
//===----------------------------------------------------------------------===//

LogicalResult ConstantCompositeOpPattern::matchAndRewrite(
    ConstantOp constOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto srcType = constOp.getType().dyn_cast<ShapedType>();
  if (!srcType)
    return failure();

  // std.constant should only have vector or tenor types.
  assert((srcType.isa<VectorType, RankedTensorType>()));

  auto dstType = getTypeConverter()->convertType(srcType);
  if (!dstType)
    return failure();

  auto dstElementsAttr = constOp.value().dyn_cast<DenseElementsAttr>();
  ShapedType dstAttrType = dstElementsAttr.getType();
  if (!dstElementsAttr)
    return failure();

  // If the composite type has more than one dimensions, perform linearization.
  if (srcType.getRank() > 1) {
    if (srcType.isa<RankedTensorType>()) {
      dstAttrType = RankedTensorType::get(srcType.getNumElements(),
                                          srcType.getElementType());
      dstElementsAttr = dstElementsAttr.reshape(dstAttrType);
    } else {
      // TODO: add support for large vectors.
      return failure();
    }
  }

  Type srcElemType = srcType.getElementType();
  Type dstElemType;
  // Tensor types are converted to SPIR-V array types; vector types are
  // converted to SPIR-V vector/array types.
  if (auto arrayType = dstType.dyn_cast<spirv::ArrayType>())
    dstElemType = arrayType.getElementType();
  else
    dstElemType = dstType.cast<VectorType>().getElementType();

  // If the source and destination element types are different, perform
  // attribute conversion.
  if (srcElemType != dstElemType) {
    SmallVector<Attribute, 8> elements;
    if (srcElemType.isa<FloatType>()) {
      for (Attribute srcAttr : dstElementsAttr.getAttributeValues()) {
        FloatAttr dstAttr = convertFloatAttr(
            srcAttr.cast<FloatAttr>(), dstElemType.cast<FloatType>(), rewriter);
        if (!dstAttr)
          return failure();
        elements.push_back(dstAttr);
      }
    } else if (srcElemType.isInteger(1)) {
      return failure();
    } else {
      for (Attribute srcAttr : dstElementsAttr.getAttributeValues()) {
        IntegerAttr dstAttr =
            convertIntegerAttr(srcAttr.cast<IntegerAttr>(),
                               dstElemType.cast<IntegerType>(), rewriter);
        if (!dstAttr)
          return failure();
        elements.push_back(dstAttr);
      }
    }

    // Unfortunately, we cannot use dialect-specific types for element
    // attributes; element attributes only works with builtin types. So we need
    // to prepare another converted builtin types for the destination elements
    // attribute.
    if (dstAttrType.isa<RankedTensorType>())
      dstAttrType = RankedTensorType::get(dstAttrType.getShape(), dstElemType);
    else
      dstAttrType = VectorType::get(dstAttrType.getShape(), dstElemType);

    dstElementsAttr = DenseElementsAttr::get(dstAttrType, elements);
  }

  rewriter.replaceOpWithNewOp<spirv::ConstantOp>(constOp, dstType,
                                                 dstElementsAttr);
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp with scalar type.
//===----------------------------------------------------------------------===//

LogicalResult ConstantScalarOpPattern::matchAndRewrite(
    ConstantOp constOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  Type srcType = constOp.getType();
  if (!srcType.isIntOrIndexOrFloat())
    return failure();

  Type dstType = getTypeConverter()->convertType(srcType);
  if (!dstType)
    return failure();

  // Floating-point types.
  if (srcType.isa<FloatType>()) {
    auto srcAttr = constOp.value().cast<FloatAttr>();
    auto dstAttr = srcAttr;

    // Floating-point types not supported in the target environment are all
    // converted to float type.
    if (srcType != dstType) {
      dstAttr = convertFloatAttr(srcAttr, dstType.cast<FloatType>(), rewriter);
      if (!dstAttr)
        return failure();
    }

    rewriter.replaceOpWithNewOp<spirv::ConstantOp>(constOp, dstType, dstAttr);
    return success();
  }

  // Bool type.
  if (srcType.isInteger(1)) {
    // std.constant can use 0/1 instead of true/false for i1 values. We need to
    // handle that here.
    auto dstAttr = convertBoolAttr(constOp.value(), rewriter);
    if (!dstAttr)
      return failure();
    rewriter.replaceOpWithNewOp<spirv::ConstantOp>(constOp, dstType, dstAttr);
    return success();
  }

  // IndexType or IntegerType. Index values are converted to 32-bit integer
  // values when converting to SPIR-V.
  auto srcAttr = constOp.value().cast<IntegerAttr>();
  auto dstAttr =
      convertIntegerAttr(srcAttr, dstType.cast<IntegerType>(), rewriter);
  if (!dstAttr)
    return failure();
  rewriter.replaceOpWithNewOp<spirv::ConstantOp>(constOp, dstType, dstAttr);
  return success();
}

//===----------------------------------------------------------------------===//
// CmpFOp
//===----------------------------------------------------------------------===//

LogicalResult
CmpFOpPattern::matchAndRewrite(CmpFOp cmpFOp, ArrayRef<Value> operands,
                               ConversionPatternRewriter &rewriter) const {
  CmpFOpAdaptor cmpFOpOperands(operands);

  switch (cmpFOp.getPredicate()) {
#define DISPATCH(cmpPredicate, spirvOp)                                        \
  case cmpPredicate:                                                           \
    rewriter.replaceOpWithNewOp<spirvOp>(cmpFOp, cmpFOp.getResult().getType(), \
                                         cmpFOpOperands.lhs(),                 \
                                         cmpFOpOperands.rhs());                \
    return success();

    // Ordered.
    DISPATCH(CmpFPredicate::OEQ, spirv::FOrdEqualOp);
    DISPATCH(CmpFPredicate::OGT, spirv::FOrdGreaterThanOp);
    DISPATCH(CmpFPredicate::OGE, spirv::FOrdGreaterThanEqualOp);
    DISPATCH(CmpFPredicate::OLT, spirv::FOrdLessThanOp);
    DISPATCH(CmpFPredicate::OLE, spirv::FOrdLessThanEqualOp);
    DISPATCH(CmpFPredicate::ONE, spirv::FOrdNotEqualOp);
    // Unordered.
    DISPATCH(CmpFPredicate::UEQ, spirv::FUnordEqualOp);
    DISPATCH(CmpFPredicate::UGT, spirv::FUnordGreaterThanOp);
    DISPATCH(CmpFPredicate::UGE, spirv::FUnordGreaterThanEqualOp);
    DISPATCH(CmpFPredicate::ULT, spirv::FUnordLessThanOp);
    DISPATCH(CmpFPredicate::ULE, spirv::FUnordLessThanEqualOp);
    DISPATCH(CmpFPredicate::UNE, spirv::FUnordNotEqualOp);

#undef DISPATCH

  default:
    break;
  }
  return failure();
}

LogicalResult CmpFOpNanKernelPattern::matchAndRewrite(
    CmpFOp cmpFOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  CmpFOpAdaptor cmpFOpOperands(operands);

  if (cmpFOp.getPredicate() == CmpFPredicate::ORD) {
    rewriter.replaceOpWithNewOp<spirv::OrderedOp>(cmpFOp, cmpFOpOperands.lhs(),
                                                  cmpFOpOperands.rhs());
    return success();
  }

  if (cmpFOp.getPredicate() == CmpFPredicate::UNO) {
    rewriter.replaceOpWithNewOp<spirv::UnorderedOp>(
        cmpFOp, cmpFOpOperands.lhs(), cmpFOpOperands.rhs());
    return success();
  }

  return failure();
}

LogicalResult CmpFOpNanNonePattern::matchAndRewrite(
    CmpFOp cmpFOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (cmpFOp.getPredicate() != CmpFPredicate::ORD &&
      cmpFOp.getPredicate() != CmpFPredicate::UNO)
    return failure();

  CmpFOpAdaptor cmpFOpOperands(operands);
  Location loc = cmpFOp.getLoc();

  Value lhsIsNan = rewriter.create<spirv::IsNanOp>(loc, cmpFOpOperands.lhs());
  Value rhsIsNan = rewriter.create<spirv::IsNanOp>(loc, cmpFOpOperands.rhs());

  Value replace = rewriter.create<spirv::LogicalOrOp>(loc, lhsIsNan, rhsIsNan);
  if (cmpFOp.getPredicate() == CmpFPredicate::ORD)
    replace = rewriter.create<spirv::LogicalNotOp>(loc, replace);

  rewriter.replaceOp(cmpFOp, replace);
  return success();
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

LogicalResult
BoolCmpIOpPattern::matchAndRewrite(CmpIOp cmpIOp, ArrayRef<Value> operands,
                                   ConversionPatternRewriter &rewriter) const {
  CmpIOpAdaptor cmpIOpOperands(operands);

  Type operandType = cmpIOp.lhs().getType();
  if (!isBoolScalarOrVector(operandType))
    return failure();

  switch (cmpIOp.getPredicate()) {
#define DISPATCH(cmpPredicate, spirvOp)                                        \
  case cmpPredicate:                                                           \
    rewriter.replaceOpWithNewOp<spirvOp>(cmpIOp, cmpIOp.getResult().getType(), \
                                         cmpIOpOperands.lhs(),                 \
                                         cmpIOpOperands.rhs());                \
    return success();

    DISPATCH(CmpIPredicate::eq, spirv::LogicalEqualOp);
    DISPATCH(CmpIPredicate::ne, spirv::LogicalNotEqualOp);

#undef DISPATCH
  default:;
  }
  return failure();
}

LogicalResult
CmpIOpPattern::matchAndRewrite(CmpIOp cmpIOp, ArrayRef<Value> operands,
                               ConversionPatternRewriter &rewriter) const {
  CmpIOpAdaptor cmpIOpOperands(operands);

  Type operandType = cmpIOp.lhs().getType();
  if (isBoolScalarOrVector(operandType))
    return failure();

  switch (cmpIOp.getPredicate()) {
#define DISPATCH(cmpPredicate, spirvOp)                                        \
  case cmpPredicate:                                                           \
    if (spirvOp::template hasTrait<OpTrait::spirv::UnsignedOp>() &&            \
        operandType != this->getTypeConverter()->convertType(operandType)) {   \
      return cmpIOp.emitError(                                                 \
          "bitwidth emulation is not implemented yet on unsigned op");         \
    }                                                                          \
    rewriter.replaceOpWithNewOp<spirvOp>(cmpIOp, cmpIOp.getResult().getType(), \
                                         cmpIOpOperands.lhs(),                 \
                                         cmpIOpOperands.rhs());                \
    return success();

    DISPATCH(CmpIPredicate::eq, spirv::IEqualOp);
    DISPATCH(CmpIPredicate::ne, spirv::INotEqualOp);
    DISPATCH(CmpIPredicate::slt, spirv::SLessThanOp);
    DISPATCH(CmpIPredicate::sle, spirv::SLessThanEqualOp);
    DISPATCH(CmpIPredicate::sgt, spirv::SGreaterThanOp);
    DISPATCH(CmpIPredicate::sge, spirv::SGreaterThanEqualOp);
    DISPATCH(CmpIPredicate::ult, spirv::ULessThanOp);
    DISPATCH(CmpIPredicate::ule, spirv::ULessThanEqualOp);
    DISPATCH(CmpIPredicate::ugt, spirv::UGreaterThanOp);
    DISPATCH(CmpIPredicate::uge, spirv::UGreaterThanEqualOp);

#undef DISPATCH
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LogicalResult
IntLoadOpPattern::matchAndRewrite(LoadOp loadOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const {
  LoadOpAdaptor loadOperands(operands);
  auto loc = loadOp.getLoc();
  auto memrefType = loadOp.memref().getType().cast<MemRefType>();
  if (!memrefType.getElementType().isSignlessInteger())
    return failure();

  auto &typeConverter = *getTypeConverter<SPIRVTypeConverter>();
  spirv::AccessChainOp accessChainOp =
      spirv::getElementPtr(typeConverter, memrefType, loadOperands.memref(),
                           loadOperands.indices(), loc, rewriter);

  int srcBits = memrefType.getElementType().getIntOrFloatBitWidth();
  auto dstType = typeConverter.convertType(memrefType)
                     .cast<spirv::PointerType>()
                     .getPointeeType()
                     .cast<spirv::StructType>()
                     .getElementType(0)
                     .cast<spirv::ArrayType>()
                     .getElementType();
  int dstBits = dstType.getIntOrFloatBitWidth();
  assert(dstBits % srcBits == 0);

  // If the rewrited load op has the same bit width, use the loading value
  // directly.
  if (srcBits == dstBits) {
    rewriter.replaceOpWithNewOp<spirv::LoadOp>(loadOp,
                                               accessChainOp.getResult());
    return success();
  }

  // Assume that getElementPtr() works linearizely. If it's a scalar, the method
  // still returns a linearized accessing. If the accessing is not linearized,
  // there will be offset issues.
  assert(accessChainOp.indices().size() == 2);
  Value adjustedPtr = adjustAccessChainForBitwidth(typeConverter, accessChainOp,
                                                   srcBits, dstBits, rewriter);
  Value spvLoadOp = rewriter.create<spirv::LoadOp>(
      loc, dstType, adjustedPtr,
      loadOp->getAttrOfType<IntegerAttr>(
          spirv::attributeName<spirv::MemoryAccess>()),
      loadOp->getAttrOfType<IntegerAttr>("alignment"));

  // Shift the bits to the rightmost.
  // ____XXXX________ -> ____________XXXX
  Value lastDim = accessChainOp->getOperand(accessChainOp.getNumOperands() - 1);
  Value offset = getOffsetForBitwidth(loc, lastDim, srcBits, dstBits, rewriter);
  Value result = rewriter.create<spirv::ShiftRightArithmeticOp>(
      loc, spvLoadOp.getType(), spvLoadOp, offset);

  // Apply the mask to extract corresponding bits.
  Value mask = rewriter.create<spirv::ConstantOp>(
      loc, dstType, rewriter.getIntegerAttr(dstType, (1 << srcBits) - 1));
  result = rewriter.create<spirv::BitwiseAndOp>(loc, dstType, result, mask);

  // Apply sign extension on the loading value unconditionally. The signedness
  // semantic is carried in the operator itself, we relies other pattern to
  // handle the casting.
  IntegerAttr shiftValueAttr =
      rewriter.getIntegerAttr(dstType, dstBits - srcBits);
  Value shiftValue =
      rewriter.create<spirv::ConstantOp>(loc, dstType, shiftValueAttr);
  result = rewriter.create<spirv::ShiftLeftLogicalOp>(loc, dstType, result,
                                                      shiftValue);
  result = rewriter.create<spirv::ShiftRightArithmeticOp>(loc, dstType, result,
                                                          shiftValue);
  rewriter.replaceOp(loadOp, result);

  assert(accessChainOp.use_empty());
  rewriter.eraseOp(accessChainOp);

  return success();
}

LogicalResult
LoadOpPattern::matchAndRewrite(LoadOp loadOp, ArrayRef<Value> operands,
                               ConversionPatternRewriter &rewriter) const {
  LoadOpAdaptor loadOperands(operands);
  auto memrefType = loadOp.memref().getType().cast<MemRefType>();
  if (memrefType.getElementType().isSignlessInteger())
    return failure();
  auto loadPtr = spirv::getElementPtr(
      *getTypeConverter<SPIRVTypeConverter>(), memrefType,
      loadOperands.memref(), loadOperands.indices(), loadOp.getLoc(), rewriter);
  rewriter.replaceOpWithNewOp<spirv::LoadOp>(loadOp, loadPtr);
  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult
ReturnOpPattern::matchAndRewrite(ReturnOp returnOp, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const {
  if (returnOp.getNumOperands() > 1)
    return failure();

  if (returnOp.getNumOperands() == 1) {
    rewriter.replaceOpWithNewOp<spirv::ReturnValueOp>(returnOp, operands[0]);
  } else {
    rewriter.replaceOpWithNewOp<spirv::ReturnOp>(returnOp);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

LogicalResult
SelectOpPattern::matchAndRewrite(SelectOp op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const {
  SelectOpAdaptor selectOperands(operands);
  rewriter.replaceOpWithNewOp<spirv::SelectOp>(op, selectOperands.condition(),
                                               selectOperands.true_value(),
                                               selectOperands.false_value());
  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult
IntStoreOpPattern::matchAndRewrite(StoreOp storeOp, ArrayRef<Value> operands,
                                   ConversionPatternRewriter &rewriter) const {
  StoreOpAdaptor storeOperands(operands);
  auto memrefType = storeOp.memref().getType().cast<MemRefType>();
  if (!memrefType.getElementType().isSignlessInteger())
    return failure();

  auto loc = storeOp.getLoc();
  auto &typeConverter = *getTypeConverter<SPIRVTypeConverter>();
  spirv::AccessChainOp accessChainOp =
      spirv::getElementPtr(typeConverter, memrefType, storeOperands.memref(),
                           storeOperands.indices(), loc, rewriter);
  int srcBits = memrefType.getElementType().getIntOrFloatBitWidth();
  auto dstType = typeConverter.convertType(memrefType)
                     .cast<spirv::PointerType>()
                     .getPointeeType()
                     .cast<spirv::StructType>()
                     .getElementType(0)
                     .cast<spirv::ArrayType>()
                     .getElementType();
  int dstBits = dstType.getIntOrFloatBitWidth();
  assert(dstBits % srcBits == 0);

  if (srcBits == dstBits) {
    rewriter.replaceOpWithNewOp<spirv::StoreOp>(
        storeOp, accessChainOp.getResult(), storeOperands.value());
    return success();
  }

  // Since there are multi threads in the processing, the emulation will be done
  // with atomic operations. E.g., if the storing value is i8, rewrite the
  // StoreOp to
  // 1) load a 32-bit integer
  // 2) clear 8 bits in the loading value
  // 3) store 32-bit value back
  // 4) load a 32-bit integer
  // 5) modify 8 bits in the loading value
  // 6) store 32-bit value back
  // The step 1 to step 3 are done by AtomicAnd as one atomic step, and the step
  // 4 to step 6 are done by AtomicOr as another atomic step.
  assert(accessChainOp.indices().size() == 2);
  Value lastDim = accessChainOp->getOperand(accessChainOp.getNumOperands() - 1);
  Value offset = getOffsetForBitwidth(loc, lastDim, srcBits, dstBits, rewriter);

  // Create a mask to clear the destination. E.g., if it is the second i8 in
  // i32, 0xFFFF00FF is created.
  Value mask = rewriter.create<spirv::ConstantOp>(
      loc, dstType, rewriter.getIntegerAttr(dstType, (1 << srcBits) - 1));
  Value clearBitsMask =
      rewriter.create<spirv::ShiftLeftLogicalOp>(loc, dstType, mask, offset);
  clearBitsMask = rewriter.create<spirv::NotOp>(loc, dstType, clearBitsMask);

  Value storeVal =
      shiftValue(loc, storeOperands.value(), offset, mask, dstBits, rewriter);
  Value adjustedPtr = adjustAccessChainForBitwidth(typeConverter, accessChainOp,
                                                   srcBits, dstBits, rewriter);
  Optional<spirv::Scope> scope = getAtomicOpScope(memrefType);
  if (!scope)
    return failure();
  Value result = rewriter.create<spirv::AtomicAndOp>(
      loc, dstType, adjustedPtr, *scope, spirv::MemorySemantics::AcquireRelease,
      clearBitsMask);
  result = rewriter.create<spirv::AtomicOrOp>(
      loc, dstType, adjustedPtr, *scope, spirv::MemorySemantics::AcquireRelease,
      storeVal);

  // The AtomicOrOp has no side effect. Since it is already inserted, we can
  // just remove the original StoreOp. Note that rewriter.replaceOp()
  // doesn't work because it only accepts that the numbers of result are the
  // same.
  rewriter.eraseOp(storeOp);

  assert(accessChainOp.use_empty());
  rewriter.eraseOp(accessChainOp);

  return success();
}

LogicalResult
StoreOpPattern::matchAndRewrite(StoreOp storeOp, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const {
  StoreOpAdaptor storeOperands(operands);
  auto memrefType = storeOp.memref().getType().cast<MemRefType>();
  if (memrefType.getElementType().isSignlessInteger())
    return failure();
  auto storePtr =
      spirv::getElementPtr(*getTypeConverter<SPIRVTypeConverter>(), memrefType,
                           storeOperands.memref(), storeOperands.indices(),
                           storeOp.getLoc(), rewriter);
  rewriter.replaceOpWithNewOp<spirv::StoreOp>(storeOp, storePtr,
                                              storeOperands.value());
  return success();
}

//===----------------------------------------------------------------------===//
// XorOp
//===----------------------------------------------------------------------===//

LogicalResult
XOrOpPattern::matchAndRewrite(XOrOp xorOp, ArrayRef<Value> operands,
                              ConversionPatternRewriter &rewriter) const {
  assert(operands.size() == 2);

  if (isBoolScalarOrVector(operands.front().getType()))
    return failure();

  auto dstType = getTypeConverter()->convertType(xorOp.getType());
  if (!dstType)
    return failure();
  rewriter.replaceOpWithNewOp<spirv::BitwiseXorOp>(xorOp, dstType, operands);

  return success();
}

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

namespace mlir {
void populateStandardToSPIRVPatterns(MLIRContext *context,
                                     SPIRVTypeConverter &typeConverter,
                                     OwningRewritePatternList &patterns) {
  patterns.insert<
      // Math dialect operations.
      // TODO: Move to separate pass.
      UnaryAndBinaryOpPattern<math::CosOp, spirv::GLSLCosOp>,
      UnaryAndBinaryOpPattern<math::ExpOp, spirv::GLSLExpOp>,
      UnaryAndBinaryOpPattern<math::LogOp, spirv::GLSLLogOp>,
      UnaryAndBinaryOpPattern<math::RsqrtOp, spirv::GLSLInverseSqrtOp>,
      UnaryAndBinaryOpPattern<math::SinOp, spirv::GLSLSinOp>,
      UnaryAndBinaryOpPattern<math::SqrtOp, spirv::GLSLSqrtOp>,
      UnaryAndBinaryOpPattern<math::TanhOp, spirv::GLSLTanhOp>,
      // Unary and binary patterns
      BitwiseOpPattern<AndOp, spirv::LogicalAndOp, spirv::BitwiseAndOp>,
      BitwiseOpPattern<OrOp, spirv::LogicalOrOp, spirv::BitwiseOrOp>,
      UnaryAndBinaryOpPattern<AbsFOp, spirv::GLSLFAbsOp>,
      UnaryAndBinaryOpPattern<AddFOp, spirv::FAddOp>,
      UnaryAndBinaryOpPattern<AddIOp, spirv::IAddOp>,
      UnaryAndBinaryOpPattern<CeilFOp, spirv::GLSLCeilOp>,
      UnaryAndBinaryOpPattern<DivFOp, spirv::FDivOp>,
      UnaryAndBinaryOpPattern<FloorFOp, spirv::GLSLFloorOp>,
      UnaryAndBinaryOpPattern<MulFOp, spirv::FMulOp>,
      UnaryAndBinaryOpPattern<MulIOp, spirv::IMulOp>,
      UnaryAndBinaryOpPattern<NegFOp, spirv::FNegateOp>,
      UnaryAndBinaryOpPattern<RemFOp, spirv::FRemOp>,
      UnaryAndBinaryOpPattern<ShiftLeftOp, spirv::ShiftLeftLogicalOp>,
      UnaryAndBinaryOpPattern<SignedDivIOp, spirv::SDivOp>,
      UnaryAndBinaryOpPattern<SignedShiftRightOp,
                              spirv::ShiftRightArithmeticOp>,
      UnaryAndBinaryOpPattern<SubIOp, spirv::ISubOp>,
      UnaryAndBinaryOpPattern<SubFOp, spirv::FSubOp>,
      UnaryAndBinaryOpPattern<UnsignedDivIOp, spirv::UDivOp>,
      UnaryAndBinaryOpPattern<UnsignedRemIOp, spirv::UModOp>,
      UnaryAndBinaryOpPattern<UnsignedShiftRightOp, spirv::ShiftRightLogicalOp>,
      SignedRemIOpPattern, XOrOpPattern,

      // Comparison patterns
      BoolCmpIOpPattern, CmpFOpPattern, CmpFOpNanNonePattern, CmpIOpPattern,

      // Constant patterns
      ConstantCompositeOpPattern, ConstantScalarOpPattern,

      // Memory patterns
      AllocOpPattern, DeallocOpPattern, IntLoadOpPattern, IntStoreOpPattern,
      LoadOpPattern, StoreOpPattern,

      ReturnOpPattern, SelectOpPattern,

      // Type cast patterns
      UIToFPI1Pattern, ZeroExtendI1Pattern, TruncI1Pattern,
      TypeCastingOpPattern<IndexCastOp, spirv::SConvertOp>,
      TypeCastingOpPattern<SIToFPOp, spirv::ConvertSToFOp>,
      TypeCastingOpPattern<UIToFPOp, spirv::ConvertUToFOp>,
      TypeCastingOpPattern<SignExtendIOp, spirv::SConvertOp>,
      TypeCastingOpPattern<ZeroExtendIOp, spirv::UConvertOp>,
      TypeCastingOpPattern<TruncateIOp, spirv::SConvertOp>,
      TypeCastingOpPattern<FPToSIOp, spirv::ConvertFToSOp>,
      TypeCastingOpPattern<FPExtOp, spirv::FConvertOp>,
      TypeCastingOpPattern<FPTruncOp, spirv::FConvertOp>>(typeConverter,
                                                          context);

  // Give CmpFOpNanKernelPattern a higher benefit so it can prevail when Kernel
  // capability is available.
  patterns.insert<CmpFOpNanKernelPattern>(typeConverter, context,
                                          /*benefit=*/2);
}
} // namespace mlir
