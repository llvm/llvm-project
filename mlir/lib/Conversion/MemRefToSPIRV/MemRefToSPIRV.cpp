//===- MemRefToSPIRV.cpp - MemRef to SPIR-V Patterns ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert MemRef dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <optional>

#define DEBUG_TYPE "memref-to-spirv-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

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
  Type type = srcIdx.getType();
  IntegerAttr idxAttr = builder.getIntegerAttr(type, targetBits / sourceBits);
  auto idx = builder.create<spirv::ConstantOp>(loc, type, idxAttr);
  IntegerAttr srcBitsAttr = builder.getIntegerAttr(type, sourceBits);
  auto srcBitsValue = builder.create<spirv::ConstantOp>(loc, type, srcBitsAttr);
  auto m = builder.create<spirv::UModOp>(loc, srcIdx, idx);
  return builder.create<spirv::IMulOp>(loc, type, m, srcBitsValue);
}

/// Returns an adjusted spirv::AccessChainOp. Based on the
/// extension/capabilities, certain integer bitwidths `sourceBits` might not be
/// supported. During conversion if a memref of an unsupported type is used,
/// load/stores to this memref need to be modified to use a supported higher
/// bitwidth `targetBits` and extracting the required bits. For an accessing a
/// 1D array (spirv.array or spirv.rtarray), the last index is modified to load
/// the bits needed. The extraction of the actual bits needed are handled
/// separately. Note that this only works for a 1-D tensor.
static Value
adjustAccessChainForBitwidth(const SPIRVTypeConverter &typeConverter,
                             spirv::AccessChainOp op, int sourceBits,
                             int targetBits, OpBuilder &builder) {
  assert(targetBits % sourceBits == 0);
  const auto loc = op.getLoc();
  Value lastDim = op->getOperand(op.getNumOperands() - 1);
  Type type = lastDim.getType();
  IntegerAttr attr = builder.getIntegerAttr(type, targetBits / sourceBits);
  auto idx = builder.create<spirv::ConstantOp>(loc, type, attr);
  auto indices = llvm::to_vector<4>(op.getIndices());
  // There are two elements if this is a 1-D tensor.
  assert(indices.size() == 2);
  indices.back() = builder.create<spirv::SDivOp>(loc, lastDim, idx);
  Type t = typeConverter.convertType(op.getComponentPtr().getType());
  return builder.create<spirv::AccessChainOp>(loc, t, op.getBasePtr(), indices);
}

/// Casts the given `srcBool` into an integer of `dstType`.
static Value castBoolToIntN(Location loc, Value srcBool, Type dstType,
                            OpBuilder &builder) {
  assert(srcBool.getType().isInteger(1));
  if (dstType.isInteger(1))
    return srcBool;
  Value zero = spirv::ConstantOp::getZero(dstType, loc, builder);
  Value one = spirv::ConstantOp::getOne(dstType, loc, builder);
  return builder.create<spirv::SelectOp>(loc, dstType, srcBool, one, zero);
}

/// Returns the `targetBits`-bit value shifted by the given `offset`, and cast
/// to the type destination type, and masked.
static Value shiftValue(Location loc, Value value, Value offset, Value mask,
                        OpBuilder &builder) {
  IntegerType dstType = cast<IntegerType>(mask.getType());
  int targetBits = static_cast<int>(dstType.getWidth());
  int valueBits = value.getType().getIntOrFloatBitWidth();
  assert(valueBits <= targetBits);

  if (valueBits == 1) {
    value = castBoolToIntN(loc, value, dstType, builder);
  } else {
    if (valueBits < targetBits) {
      value = builder.create<spirv::UConvertOp>(
          loc, builder.getIntegerType(targetBits), value);
    }

    value = builder.create<spirv::BitwiseAndOp>(loc, value, mask);
  }
  return builder.create<spirv::ShiftLeftLogicalOp>(loc, value.getType(), value,
                                                   offset);
}

/// Returns true if the allocations of memref `type` generated from `allocOp`
/// can be lowered to SPIR-V.
static bool isAllocationSupported(Operation *allocOp, MemRefType type) {
  if (isa<memref::AllocOp, memref::DeallocOp>(allocOp)) {
    auto sc = dyn_cast_or_null<spirv::StorageClassAttr>(type.getMemorySpace());
    if (!sc || sc.getValue() != spirv::StorageClass::Workgroup)
      return false;
  } else if (isa<memref::AllocaOp>(allocOp)) {
    auto sc = dyn_cast_or_null<spirv::StorageClassAttr>(type.getMemorySpace());
    if (!sc || sc.getValue() != spirv::StorageClass::Function)
      return false;
  } else {
    return false;
  }

  // Currently only support static shape and int or float or vector of int or
  // float element type.
  if (!type.hasStaticShape())
    return false;

  Type elementType = type.getElementType();
  if (auto vecType = dyn_cast<VectorType>(elementType))
    elementType = vecType.getElementType();
  return elementType.isIntOrFloat();
}

/// Returns the scope to use for atomic operations use for emulating store
/// operations of unsupported integer bitwidths, based on the memref
/// type. Returns std::nullopt on failure.
static std::optional<spirv::Scope> getAtomicOpScope(MemRefType type) {
  auto sc = dyn_cast_or_null<spirv::StorageClassAttr>(type.getMemorySpace());
  switch (sc.getValue()) {
  case spirv::StorageClass::StorageBuffer:
    return spirv::Scope::Device;
  case spirv::StorageClass::Workgroup:
    return spirv::Scope::Workgroup;
  default:
    break;
  }
  return {};
}

/// Casts the given `srcInt` into a boolean value.
static Value castIntNToBool(Location loc, Value srcInt, OpBuilder &builder) {
  if (srcInt.getType().isInteger(1))
    return srcInt;

  auto one = spirv::ConstantOp::getOne(srcInt.getType(), loc, builder);
  return builder.create<spirv::IEqualOp>(loc, srcInt, one);
}

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

// Note that DRR cannot be used for the patterns in this file: we may need to
// convert type along the way, which requires ConversionPattern. DRR generates
// normal RewritePattern.

namespace {

/// Converts memref.alloca to SPIR-V Function variables.
class AllocaOpPattern final : public OpConversionPattern<memref::AllocaOp> {
public:
  using OpConversionPattern<memref::AllocaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocaOp allocaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts an allocation operation to SPIR-V. Currently only supports lowering
/// to Workgroup memory when the size is constant.  Note that this pattern needs
/// to be applied in a pass that runs at least at spirv.module scope since it
/// wil ladd global variables into the spirv.module.
class AllocOpPattern final : public OpConversionPattern<memref::AllocOp> {
public:
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp operation, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts memref.automic_rmw operations to SPIR-V atomic operations.
class AtomicRMWOpPattern final
    : public OpConversionPattern<memref::AtomicRMWOp> {
public:
  using OpConversionPattern<memref::AtomicRMWOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AtomicRMWOp atomicOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Removed a deallocation if it is a supported allocation. Currently only
/// removes deallocation if the memory space is workgroup memory.
class DeallocOpPattern final : public OpConversionPattern<memref::DeallocOp> {
public:
  using OpConversionPattern<memref::DeallocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp operation, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts memref.load to spirv.Load + spirv.AccessChain on integers.
class IntLoadOpPattern final : public OpConversionPattern<memref::LoadOp> {
public:
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts memref.load to spirv.Load + spirv.AccessChain.
class LoadOpPattern final : public OpConversionPattern<memref::LoadOp> {
public:
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts memref.store to spirv.Store on integers.
class IntStoreOpPattern final : public OpConversionPattern<memref::StoreOp> {
public:
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts memref.memory_space_cast to the appropriate spirv cast operations.
class MemorySpaceCastOpPattern final
    : public OpConversionPattern<memref::MemorySpaceCastOp> {
public:
  using OpConversionPattern<memref::MemorySpaceCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::MemorySpaceCastOp addrCastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts memref.store to spirv.Store.
class StoreOpPattern final : public OpConversionPattern<memref::StoreOp> {
public:
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ReinterpretCastPattern final
    : public OpConversionPattern<memref::ReinterpretCastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class CastPattern final : public OpConversionPattern<memref::CastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSource();
    Type srcType = src.getType();

    const TypeConverter *converter = getTypeConverter();
    Type dstType = converter->convertType(op.getType());
    if (srcType != dstType)
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "types doesn't match: " << srcType << " and " << dstType;
      });

    rewriter.replaceOp(op, src);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

LogicalResult
AllocaOpPattern::matchAndRewrite(memref::AllocaOp allocaOp, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  MemRefType allocType = allocaOp.getType();
  if (!isAllocationSupported(allocaOp, allocType))
    return rewriter.notifyMatchFailure(allocaOp, "unhandled allocation type");

  // Get the SPIR-V type for the allocation.
  Type spirvType = getTypeConverter()->convertType(allocType);
  if (!spirvType)
    return rewriter.notifyMatchFailure(allocaOp, "type conversion failed");

  rewriter.replaceOpWithNewOp<spirv::VariableOp>(allocaOp, spirvType,
                                                 spirv::StorageClass::Function,
                                                 /*initializer=*/nullptr);
  return success();
}

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

LogicalResult
AllocOpPattern::matchAndRewrite(memref::AllocOp operation, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  MemRefType allocType = operation.getType();
  if (!isAllocationSupported(operation, allocType))
    return rewriter.notifyMatchFailure(operation, "unhandled allocation type");

  // Get the SPIR-V type for the allocation.
  Type spirvType = getTypeConverter()->convertType(allocType);
  if (!spirvType)
    return rewriter.notifyMatchFailure(operation, "type conversion failed");

  // Insert spirv.GlobalVariable for this allocation.
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

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

LogicalResult
AtomicRMWOpPattern::matchAndRewrite(memref::AtomicRMWOp atomicOp,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  if (isa<FloatType>(atomicOp.getType()))
    return rewriter.notifyMatchFailure(atomicOp,
                                       "unimplemented floating-point case");

  auto memrefType = cast<MemRefType>(atomicOp.getMemref().getType());
  std::optional<spirv::Scope> scope = getAtomicOpScope(memrefType);
  if (!scope)
    return rewriter.notifyMatchFailure(atomicOp,
                                       "unsupported memref memory space");

  auto &typeConverter = *getTypeConverter<SPIRVTypeConverter>();
  Type resultType = typeConverter.convertType(atomicOp.getType());
  if (!resultType)
    return rewriter.notifyMatchFailure(atomicOp,
                                       "failed to convert result type");

  auto loc = atomicOp.getLoc();
  Value ptr =
      spirv::getElementPtr(typeConverter, memrefType, adaptor.getMemref(),
                           adaptor.getIndices(), loc, rewriter);

  if (!ptr)
    return failure();

#define ATOMIC_CASE(kind, spirvOp)                                             \
  case arith::AtomicRMWKind::kind:                                             \
    rewriter.replaceOpWithNewOp<spirv::spirvOp>(                               \
        atomicOp, resultType, ptr, *scope,                                     \
        spirv::MemorySemantics::AcquireRelease, adaptor.getValue());           \
    break

  switch (atomicOp.getKind()) {
    ATOMIC_CASE(addi, AtomicIAddOp);
    ATOMIC_CASE(maxs, AtomicSMaxOp);
    ATOMIC_CASE(maxu, AtomicUMaxOp);
    ATOMIC_CASE(mins, AtomicSMinOp);
    ATOMIC_CASE(minu, AtomicUMinOp);
    ATOMIC_CASE(ori, AtomicOrOp);
    ATOMIC_CASE(andi, AtomicAndOp);
  default:
    return rewriter.notifyMatchFailure(atomicOp, "unimplemented atomic kind");
  }

#undef ATOMIC_CASE

  return success();
}

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//

LogicalResult
DeallocOpPattern::matchAndRewrite(memref::DeallocOp operation,
                                  OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  MemRefType deallocType = cast<MemRefType>(operation.getMemref().getType());
  if (!isAllocationSupported(operation, deallocType))
    return rewriter.notifyMatchFailure(operation, "unhandled allocation type");
  rewriter.eraseOp(operation);
  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

struct MemoryRequirements {
  spirv::MemoryAccessAttr memoryAccess;
  IntegerAttr alignment;
};

/// Given an accessed SPIR-V pointer, calculates its alignment requirements, if
/// any.
static FailureOr<MemoryRequirements>
calculateMemoryRequirements(Value accessedPtr, bool isNontemporal) {
  MLIRContext *ctx = accessedPtr.getContext();

  auto memoryAccess = spirv::MemoryAccess::None;
  if (isNontemporal) {
    memoryAccess = spirv::MemoryAccess::Nontemporal;
  }

  auto ptrType = cast<spirv::PointerType>(accessedPtr.getType());
  if (ptrType.getStorageClass() != spirv::StorageClass::PhysicalStorageBuffer) {
    if (memoryAccess == spirv::MemoryAccess::None) {
      return MemoryRequirements{spirv::MemoryAccessAttr{}, IntegerAttr{}};
    }
    return MemoryRequirements{spirv::MemoryAccessAttr::get(ctx, memoryAccess),
                              IntegerAttr{}};
  }

  // PhysicalStorageBuffers require the `Aligned` attribute.
  auto pointeeType = dyn_cast<spirv::ScalarType>(ptrType.getPointeeType());
  if (!pointeeType)
    return failure();

  // For scalar types, the alignment is determined by their size.
  std::optional<int64_t> sizeInBytes = pointeeType.getSizeInBytes();
  if (!sizeInBytes.has_value())
    return failure();

  memoryAccess = memoryAccess | spirv::MemoryAccess::Aligned;
  auto memAccessAttr = spirv::MemoryAccessAttr::get(ctx, memoryAccess);
  auto alignment = IntegerAttr::get(IntegerType::get(ctx, 32), *sizeInBytes);
  return MemoryRequirements{memAccessAttr, alignment};
}

/// Given an accessed SPIR-V pointer and the original memref load/store
/// `memAccess` op, calculates the alignment requirements, if any. Takes into
/// account the alignment attributes applied to the load/store op.
template <class LoadOrStoreOp>
static FailureOr<MemoryRequirements>
calculateMemoryRequirements(Value accessedPtr, LoadOrStoreOp loadOrStoreOp) {
  static_assert(
      llvm::is_one_of<LoadOrStoreOp, memref::LoadOp, memref::StoreOp>::value,
      "Must be called on either memref::LoadOp or memref::StoreOp");

  Operation *memrefAccessOp = loadOrStoreOp.getOperation();
  auto memrefMemAccess = memrefAccessOp->getAttrOfType<spirv::MemoryAccessAttr>(
      spirv::attributeName<spirv::MemoryAccess>());
  auto memrefAlignment =
      memrefAccessOp->getAttrOfType<IntegerAttr>("alignment");
  if (memrefMemAccess && memrefAlignment)
    return MemoryRequirements{memrefMemAccess, memrefAlignment};

  return calculateMemoryRequirements(accessedPtr,
                                     loadOrStoreOp.getNontemporal());
}

LogicalResult
IntLoadOpPattern::matchAndRewrite(memref::LoadOp loadOp, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  auto loc = loadOp.getLoc();
  auto memrefType = cast<MemRefType>(loadOp.getMemref().getType());
  if (!memrefType.getElementType().isSignlessInteger())
    return failure();

  const auto &typeConverter = *getTypeConverter<SPIRVTypeConverter>();
  Value accessChain =
      spirv::getElementPtr(typeConverter, memrefType, adaptor.getMemref(),
                           adaptor.getIndices(), loc, rewriter);

  if (!accessChain)
    return failure();

  int srcBits = memrefType.getElementType().getIntOrFloatBitWidth();
  bool isBool = srcBits == 1;
  if (isBool)
    srcBits = typeConverter.getOptions().boolNumBits;

  auto pointerType = typeConverter.convertType<spirv::PointerType>(memrefType);
  if (!pointerType)
    return rewriter.notifyMatchFailure(loadOp, "failed to convert memref type");

  Type pointeeType = pointerType.getPointeeType();
  Type dstType;
  if (typeConverter.allows(spirv::Capability::Kernel)) {
    if (auto arrayType = dyn_cast<spirv::ArrayType>(pointeeType))
      dstType = arrayType.getElementType();
    else
      dstType = pointeeType;
  } else {
    // For Vulkan we need to extract element from wrapping struct and array.
    Type structElemType =
        cast<spirv::StructType>(pointeeType).getElementType(0);
    if (auto arrayType = dyn_cast<spirv::ArrayType>(structElemType))
      dstType = arrayType.getElementType();
    else
      dstType = cast<spirv::RuntimeArrayType>(structElemType).getElementType();
  }
  int dstBits = dstType.getIntOrFloatBitWidth();
  assert(dstBits % srcBits == 0);

  // If the rewritten load op has the same bit width, use the loading value
  // directly.
  if (srcBits == dstBits) {
    auto memoryRequirements = calculateMemoryRequirements(accessChain, loadOp);
    if (failed(memoryRequirements))
      return rewriter.notifyMatchFailure(
          loadOp, "failed to determine memory requirements");

    auto [memoryAccess, alignment] = *memoryRequirements;
    Value loadVal = rewriter.create<spirv::LoadOp>(loc, accessChain,
                                                   memoryAccess, alignment);
    if (isBool)
      loadVal = castIntNToBool(loc, loadVal, rewriter);
    rewriter.replaceOp(loadOp, loadVal);
    return success();
  }

  // Bitcasting is currently unsupported for Kernel capability /
  // spirv.PtrAccessChain.
  if (typeConverter.allows(spirv::Capability::Kernel))
    return failure();

  auto accessChainOp = accessChain.getDefiningOp<spirv::AccessChainOp>();
  if (!accessChainOp)
    return failure();

  // Assume that getElementPtr() works linearizely. If it's a scalar, the method
  // still returns a linearized accessing. If the accessing is not linearized,
  // there will be offset issues.
  assert(accessChainOp.getIndices().size() == 2);
  Value adjustedPtr = adjustAccessChainForBitwidth(typeConverter, accessChainOp,
                                                   srcBits, dstBits, rewriter);
  auto memoryRequirements = calculateMemoryRequirements(adjustedPtr, loadOp);
  if (failed(memoryRequirements))
    return rewriter.notifyMatchFailure(
        loadOp, "failed to determine memory requirements");

  auto [memoryAccess, alignment] = *memoryRequirements;
  Value spvLoadOp = rewriter.create<spirv::LoadOp>(loc, dstType, adjustedPtr,
                                                   memoryAccess, alignment);

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
LoadOpPattern::matchAndRewrite(memref::LoadOp loadOp, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  auto memrefType = cast<MemRefType>(loadOp.getMemref().getType());
  if (memrefType.getElementType().isSignlessInteger())
    return failure();
  Value loadPtr = spirv::getElementPtr(
      *getTypeConverter<SPIRVTypeConverter>(), memrefType, adaptor.getMemref(),
      adaptor.getIndices(), loadOp.getLoc(), rewriter);

  if (!loadPtr)
    return failure();

  auto memoryRequirements = calculateMemoryRequirements(loadPtr, loadOp);
  if (failed(memoryRequirements))
    return rewriter.notifyMatchFailure(
        loadOp, "failed to determine memory requirements");

  auto [memoryAccess, alignment] = *memoryRequirements;
  rewriter.replaceOpWithNewOp<spirv::LoadOp>(loadOp, loadPtr, memoryAccess,
                                             alignment);
  return success();
}

LogicalResult
IntStoreOpPattern::matchAndRewrite(memref::StoreOp storeOp, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  auto memrefType = cast<MemRefType>(storeOp.getMemref().getType());
  if (!memrefType.getElementType().isSignlessInteger())
    return rewriter.notifyMatchFailure(storeOp,
                                       "element type is not a signless int");

  auto loc = storeOp.getLoc();
  auto &typeConverter = *getTypeConverter<SPIRVTypeConverter>();
  Value accessChain =
      spirv::getElementPtr(typeConverter, memrefType, adaptor.getMemref(),
                           adaptor.getIndices(), loc, rewriter);

  if (!accessChain)
    return rewriter.notifyMatchFailure(
        storeOp, "failed to convert element pointer type");

  int srcBits = memrefType.getElementType().getIntOrFloatBitWidth();

  bool isBool = srcBits == 1;
  if (isBool)
    srcBits = typeConverter.getOptions().boolNumBits;

  auto pointerType = typeConverter.convertType<spirv::PointerType>(memrefType);
  if (!pointerType)
    return rewriter.notifyMatchFailure(storeOp,
                                       "failed to convert memref type");

  Type pointeeType = pointerType.getPointeeType();
  IntegerType dstType;
  if (typeConverter.allows(spirv::Capability::Kernel)) {
    if (auto arrayType = dyn_cast<spirv::ArrayType>(pointeeType))
      dstType = dyn_cast<IntegerType>(arrayType.getElementType());
    else
      dstType = dyn_cast<IntegerType>(pointeeType);
  } else {
    // For Vulkan we need to extract element from wrapping struct and array.
    Type structElemType =
        cast<spirv::StructType>(pointeeType).getElementType(0);
    if (auto arrayType = dyn_cast<spirv::ArrayType>(structElemType))
      dstType = dyn_cast<IntegerType>(arrayType.getElementType());
    else
      dstType = dyn_cast<IntegerType>(
          cast<spirv::RuntimeArrayType>(structElemType).getElementType());
  }

  if (!dstType)
    return rewriter.notifyMatchFailure(
        storeOp, "failed to determine destination element type");

  int dstBits = static_cast<int>(dstType.getWidth());
  assert(dstBits % srcBits == 0);

  if (srcBits == dstBits) {
    auto memoryRequirements = calculateMemoryRequirements(accessChain, storeOp);
    if (failed(memoryRequirements))
      return rewriter.notifyMatchFailure(
          storeOp, "failed to determine memory requirements");

    auto [memoryAccess, alignment] = *memoryRequirements;
    Value storeVal = adaptor.getValue();
    if (isBool)
      storeVal = castBoolToIntN(loc, storeVal, dstType, rewriter);
    rewriter.replaceOpWithNewOp<spirv::StoreOp>(storeOp, accessChain, storeVal,
                                                memoryAccess, alignment);
    return success();
  }

  // Bitcasting is currently unsupported for Kernel capability /
  // spirv.PtrAccessChain.
  if (typeConverter.allows(spirv::Capability::Kernel))
    return failure();

  auto accessChainOp = accessChain.getDefiningOp<spirv::AccessChainOp>();
  if (!accessChainOp)
    return failure();

  // Since there are multiple threads in the processing, the emulation will be
  // done with atomic operations. E.g., if the stored value is i8, rewrite the
  // StoreOp to:
  // 1) load a 32-bit integer
  // 2) clear 8 bits in the loaded value
  // 3) set 8 bits in the loaded value
  // 4) store 32-bit value back
  //
  // Step 2 is done with AtomicAnd, and step 3 is done with AtomicOr (of the
  // loaded 32-bit value and the shifted 8-bit store value) as another atomic
  // step.
  assert(accessChainOp.getIndices().size() == 2);
  Value lastDim = accessChainOp->getOperand(accessChainOp.getNumOperands() - 1);
  Value offset = getOffsetForBitwidth(loc, lastDim, srcBits, dstBits, rewriter);

  // Create a mask to clear the destination. E.g., if it is the second i8 in
  // i32, 0xFFFF00FF is created.
  Value mask = rewriter.create<spirv::ConstantOp>(
      loc, dstType, rewriter.getIntegerAttr(dstType, (1 << srcBits) - 1));
  Value clearBitsMask =
      rewriter.create<spirv::ShiftLeftLogicalOp>(loc, dstType, mask, offset);
  clearBitsMask = rewriter.create<spirv::NotOp>(loc, dstType, clearBitsMask);

  Value storeVal = shiftValue(loc, adaptor.getValue(), offset, mask, rewriter);
  Value adjustedPtr = adjustAccessChainForBitwidth(typeConverter, accessChainOp,
                                                   srcBits, dstBits, rewriter);
  std::optional<spirv::Scope> scope = getAtomicOpScope(memrefType);
  if (!scope)
    return rewriter.notifyMatchFailure(storeOp, "atomic scope not available");

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

//===----------------------------------------------------------------------===//
// MemorySpaceCastOp
//===----------------------------------------------------------------------===//

LogicalResult MemorySpaceCastOpPattern::matchAndRewrite(
    memref::MemorySpaceCastOp addrCastOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = addrCastOp.getLoc();
  auto &typeConverter = *getTypeConverter<SPIRVTypeConverter>();
  if (!typeConverter.allows(spirv::Capability::Kernel))
    return rewriter.notifyMatchFailure(
        loc, "address space casts require kernel capability");

  auto sourceType = dyn_cast<MemRefType>(addrCastOp.getSource().getType());
  if (!sourceType)
    return rewriter.notifyMatchFailure(
        loc, "SPIR-V lowering requires ranked memref types");
  auto resultType = cast<MemRefType>(addrCastOp.getResult().getType());

  auto sourceStorageClassAttr =
      dyn_cast_or_null<spirv::StorageClassAttr>(sourceType.getMemorySpace());
  if (!sourceStorageClassAttr)
    return rewriter.notifyMatchFailure(loc, [sourceType](Diagnostic &diag) {
      diag << "source address space " << sourceType.getMemorySpace()
           << " must be a SPIR-V storage class";
    });
  auto resultStorageClassAttr =
      dyn_cast_or_null<spirv::StorageClassAttr>(resultType.getMemorySpace());
  if (!resultStorageClassAttr)
    return rewriter.notifyMatchFailure(loc, [resultType](Diagnostic &diag) {
      diag << "result address space " << resultType.getMemorySpace()
           << " must be a SPIR-V storage class";
    });

  spirv::StorageClass sourceSc = sourceStorageClassAttr.getValue();
  spirv::StorageClass resultSc = resultStorageClassAttr.getValue();

  Value result = adaptor.getSource();
  Type resultPtrType = typeConverter.convertType(resultType);
  if (!resultPtrType)
    return rewriter.notifyMatchFailure(addrCastOp,
                                       "failed to convert memref type");

  Type genericPtrType = resultPtrType;
  // SPIR-V doesn't have a general address space cast operation. Instead, it has
  // conversions to and from generic pointers. To implement the general case,
  // we use specific-to-generic conversions when the source class is not
  // generic. Then when the result storage class is not generic, we convert the
  // generic pointer (either the input on ar intermediate result) to that
  // class. This also means that we'll need the intermediate generic pointer
  // type if neither the source or destination have it.
  if (sourceSc != spirv::StorageClass::Generic &&
      resultSc != spirv::StorageClass::Generic) {
    Type intermediateType =
        MemRefType::get(sourceType.getShape(), sourceType.getElementType(),
                        sourceType.getLayout(),
                        rewriter.getAttr<spirv::StorageClassAttr>(
                            spirv::StorageClass::Generic));
    genericPtrType = typeConverter.convertType(intermediateType);
  }
  if (sourceSc != spirv::StorageClass::Generic) {
    result =
        rewriter.create<spirv::PtrCastToGenericOp>(loc, genericPtrType, result);
  }
  if (resultSc != spirv::StorageClass::Generic) {
    result =
        rewriter.create<spirv::GenericCastToPtrOp>(loc, resultPtrType, result);
  }
  rewriter.replaceOp(addrCastOp, result);
  return success();
}

LogicalResult
StoreOpPattern::matchAndRewrite(memref::StoreOp storeOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  auto memrefType = cast<MemRefType>(storeOp.getMemref().getType());
  if (memrefType.getElementType().isSignlessInteger())
    return rewriter.notifyMatchFailure(storeOp, "signless int");
  auto storePtr = spirv::getElementPtr(
      *getTypeConverter<SPIRVTypeConverter>(), memrefType, adaptor.getMemref(),
      adaptor.getIndices(), storeOp.getLoc(), rewriter);

  if (!storePtr)
    return rewriter.notifyMatchFailure(storeOp, "type conversion failed");

  auto memoryRequirements = calculateMemoryRequirements(storePtr, storeOp);
  if (failed(memoryRequirements))
    return rewriter.notifyMatchFailure(
        storeOp, "failed to determine memory requirements");

  auto [memoryAccess, alignment] = *memoryRequirements;
  rewriter.replaceOpWithNewOp<spirv::StoreOp>(
      storeOp, storePtr, adaptor.getValue(), memoryAccess, alignment);
  return success();
}

LogicalResult ReinterpretCastPattern::matchAndRewrite(
    memref::ReinterpretCastOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value src = adaptor.getSource();
  auto srcType = dyn_cast<spirv::PointerType>(src.getType());

  if (!srcType)
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << "invalid src type " << src.getType();
    });

  const TypeConverter *converter = getTypeConverter();

  auto dstType = converter->convertType<spirv::PointerType>(op.getType());
  if (dstType != srcType)
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << "invalid dst type " << op.getType();
    });

  OpFoldResult offset =
      getMixedValues(adaptor.getStaticOffsets(), adaptor.getOffsets(), rewriter)
          .front();
  if (isConstantIntValue(offset, 0)) {
    rewriter.replaceOp(op, src);
    return success();
  }

  Type intType = converter->convertType(rewriter.getIndexType());
  if (!intType)
    return rewriter.notifyMatchFailure(op, "failed to convert index type");

  Location loc = op.getLoc();
  auto offsetValue = [&]() -> Value {
    if (auto val = dyn_cast<Value>(offset))
      return val;

    int64_t attrVal = cast<IntegerAttr>(offset.get<Attribute>()).getInt();
    Attribute attr = rewriter.getIntegerAttr(intType, attrVal);
    return rewriter.create<spirv::ConstantOp>(loc, intType, attr);
  }();

  rewriter.replaceOpWithNewOp<spirv::InBoundsPtrAccessChainOp>(
      op, src, offsetValue, std::nullopt);
  return success();
}

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

namespace mlir {
void populateMemRefToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                   RewritePatternSet &patterns) {
  patterns.add<AllocaOpPattern, AllocOpPattern, AtomicRMWOpPattern,
               DeallocOpPattern, IntLoadOpPattern, IntStoreOpPattern,
               LoadOpPattern, MemorySpaceCastOpPattern, StoreOpPattern,
               ReinterpretCastPattern, CastPattern>(typeConverter,
                                                    patterns.getContext());
}
} // namespace mlir
