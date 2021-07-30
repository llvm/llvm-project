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

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "llvm/Support/Debug.h"

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
class AllocOpPattern final : public OpConversionPattern<memref::AllocOp> {
public:
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Removed a deallocation if it is a supported allocation. Currently only
/// removes deallocation if the memory space is workgroup memory.
class DeallocOpPattern final : public OpConversionPattern<memref::DeallocOp> {
public:
  using OpConversionPattern<memref::DeallocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts memref.load to spv.Load.
class IntLoadOpPattern final : public OpConversionPattern<memref::LoadOp> {
public:
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp loadOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts memref.load to spv.Load.
class LoadOpPattern final : public OpConversionPattern<memref::LoadOp> {
public:
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp loadOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts memref.store to spv.Store on integers.
class IntStoreOpPattern final : public OpConversionPattern<memref::StoreOp> {
public:
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp storeOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts memref.store to spv.Store.
class StoreOpPattern final : public OpConversionPattern<memref::StoreOp> {
public:
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp storeOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

LogicalResult
AllocOpPattern::matchAndRewrite(memref::AllocOp operation,
                                ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const {
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

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//

LogicalResult
DeallocOpPattern::matchAndRewrite(memref::DeallocOp operation,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const {
  MemRefType deallocType = operation.memref().getType().cast<MemRefType>();
  if (!isAllocationSupported(deallocType))
    return operation.emitError("unhandled deallocation type");
  rewriter.eraseOp(operation);
  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LogicalResult
IntLoadOpPattern::matchAndRewrite(memref::LoadOp loadOp,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const {
  memref::LoadOpAdaptor loadOperands(operands);
  auto loc = loadOp.getLoc();
  auto memrefType = loadOp.memref().getType().cast<MemRefType>();
  if (!memrefType.getElementType().isSignlessInteger())
    return failure();

  auto &typeConverter = *getTypeConverter<SPIRVTypeConverter>();
  spirv::AccessChainOp accessChainOp =
      spirv::getElementPtr(typeConverter, memrefType, loadOperands.memref(),
                           loadOperands.indices(), loc, rewriter);

  if (!accessChainOp)
    return failure();

  int srcBits = memrefType.getElementType().getIntOrFloatBitWidth();
  bool isBool = srcBits == 1;
  if (isBool)
    srcBits = typeConverter.getOptions().boolNumBits;
  Type pointeeType = typeConverter.convertType(memrefType)
                         .cast<spirv::PointerType>()
                         .getPointeeType();
  Type structElemType = pointeeType.cast<spirv::StructType>().getElementType(0);
  Type dstType;
  if (auto arrayType = structElemType.dyn_cast<spirv::ArrayType>())
    dstType = arrayType.getElementType();
  else
    dstType = structElemType.cast<spirv::RuntimeArrayType>().getElementType();

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
      loadOp->getAttrOfType<spirv::MemoryAccessAttr>(
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

  if (isBool) {
    dstType = typeConverter.convertType(loadOp.getType());
    mask = spirv::ConstantOp::getOne(result.getType(), loc, rewriter);
    Value isOne = rewriter.create<spirv::IEqualOp>(loc, result, mask);
    result = castBoolToIntN(loc, isOne, dstType, rewriter);
  } else if (result.getType().getIntOrFloatBitWidth() !=
             static_cast<unsigned>(dstBits)) {
    result = rewriter.create<spirv::SConvertOp>(loc, dstType, result);
  }
  rewriter.replaceOp(loadOp, result);

  assert(accessChainOp.use_empty());
  rewriter.eraseOp(accessChainOp);

  return success();
}

LogicalResult
LoadOpPattern::matchAndRewrite(memref::LoadOp loadOp, ArrayRef<Value> operands,
                               ConversionPatternRewriter &rewriter) const {
  memref::LoadOpAdaptor loadOperands(operands);
  auto memrefType = loadOp.memref().getType().cast<MemRefType>();
  if (memrefType.getElementType().isSignlessInteger())
    return failure();
  auto loadPtr = spirv::getElementPtr(
      *getTypeConverter<SPIRVTypeConverter>(), memrefType,
      loadOperands.memref(), loadOperands.indices(), loadOp.getLoc(), rewriter);

  if (!loadPtr)
    return failure();

  rewriter.replaceOpWithNewOp<spirv::LoadOp>(loadOp, loadPtr);
  return success();
}

LogicalResult
IntStoreOpPattern::matchAndRewrite(memref::StoreOp storeOp,
                                   ArrayRef<Value> operands,
                                   ConversionPatternRewriter &rewriter) const {
  memref::StoreOpAdaptor storeOperands(operands);
  auto memrefType = storeOp.memref().getType().cast<MemRefType>();
  if (!memrefType.getElementType().isSignlessInteger())
    return failure();

  auto loc = storeOp.getLoc();
  auto &typeConverter = *getTypeConverter<SPIRVTypeConverter>();
  spirv::AccessChainOp accessChainOp =
      spirv::getElementPtr(typeConverter, memrefType, storeOperands.memref(),
                           storeOperands.indices(), loc, rewriter);

  if (!accessChainOp)
    return failure();

  int srcBits = memrefType.getElementType().getIntOrFloatBitWidth();

  bool isBool = srcBits == 1;
  if (isBool)
    srcBits = typeConverter.getOptions().boolNumBits;

  Type pointeeType = typeConverter.convertType(memrefType)
                         .cast<spirv::PointerType>()
                         .getPointeeType();
  Type structElemType = pointeeType.cast<spirv::StructType>().getElementType(0);
  Type dstType;
  if (auto arrayType = structElemType.dyn_cast<spirv::ArrayType>())
    dstType = arrayType.getElementType();
  else
    dstType = structElemType.cast<spirv::RuntimeArrayType>().getElementType();

  int dstBits = dstType.getIntOrFloatBitWidth();
  assert(dstBits % srcBits == 0);

  if (srcBits == dstBits) {
    Value storeVal = storeOperands.value();
    if (isBool)
      storeVal = castBoolToIntN(loc, storeVal, dstType, rewriter);
    rewriter.replaceOpWithNewOp<spirv::StoreOp>(
        storeOp, accessChainOp.getResult(), storeVal);
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

  Value storeVal = storeOperands.value();
  if (isBool)
    storeVal = castBoolToIntN(loc, storeVal, dstType, rewriter);
  storeVal = shiftValue(loc, storeVal, offset, mask, dstBits, rewriter);
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
StoreOpPattern::matchAndRewrite(memref::StoreOp storeOp,
                                ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const {
  memref::StoreOpAdaptor storeOperands(operands);
  auto memrefType = storeOp.memref().getType().cast<MemRefType>();
  if (memrefType.getElementType().isSignlessInteger())
    return failure();
  auto storePtr =
      spirv::getElementPtr(*getTypeConverter<SPIRVTypeConverter>(), memrefType,
                           storeOperands.memref(), storeOperands.indices(),
                           storeOp.getLoc(), rewriter);

  if (!storePtr)
    return failure();

  rewriter.replaceOpWithNewOp<spirv::StoreOp>(storeOp, storePtr,
                                              storeOperands.value());
  return success();
}

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

namespace mlir {
void populateMemRefToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                   RewritePatternSet &patterns) {
  patterns.add<AllocOpPattern, DeallocOpPattern, IntLoadOpPattern,
               IntStoreOpPattern, LoadOpPattern, StoreOpPattern>(
      typeConverter, patterns.getContext());
}
} // namespace mlir
