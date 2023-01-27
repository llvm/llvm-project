//===- MemRefToLLVM.cpp - MemRef to LLVM dialect conversion ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/AllocLikeConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_FINALIZEMEMREFTOLLVMCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

bool isStaticStrideOrOffset(int64_t strideOrOffset) {
  return !ShapedType::isDynamic(strideOrOffset);
}

LLVM::LLVMFuncOp getFreeFn(LLVMTypeConverter *typeConverter, ModuleOp module) {
  bool useGenericFn = typeConverter->getOptions().useGenericFunctions;

  if (useGenericFn)
    return LLVM::lookupOrCreateGenericFreeFn(
        module, typeConverter->useOpaquePointers());

  return LLVM::lookupOrCreateFreeFn(module, typeConverter->useOpaquePointers());
}

struct AllocOpLowering : public AllocLikeOpLLVMLowering {
  AllocOpLowering(LLVMTypeConverter &converter)
      : AllocLikeOpLLVMLowering(memref::AllocOp::getOperationName(),
                                converter) {}
  std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          Operation *op) const override {
    return allocateBufferManuallyAlign(
        rewriter, loc, sizeBytes, op,
        getAlignment(rewriter, loc, cast<memref::AllocOp>(op)));
  }
};

struct AlignedAllocOpLowering : public AllocLikeOpLLVMLowering {
  AlignedAllocOpLowering(LLVMTypeConverter &converter)
      : AllocLikeOpLLVMLowering(memref::AllocOp::getOperationName(),
                                converter) {}
  std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          Operation *op) const override {
    Value ptr = allocateBufferAutoAlign(
        rewriter, loc, sizeBytes, op, &defaultLayout,
        alignedAllocationGetAlignment(rewriter, loc, cast<memref::AllocOp>(op),
                                      &defaultLayout));
    return std::make_tuple(ptr, ptr);
  }

private:
  /// Default layout to use in absence of the corresponding analysis.
  DataLayout defaultLayout;
};

struct AllocaOpLowering : public AllocLikeOpLLVMLowering {
  AllocaOpLowering(LLVMTypeConverter &converter)
      : AllocLikeOpLLVMLowering(memref::AllocaOp::getOperationName(),
                                converter) {}

  /// Allocates the underlying buffer using the right call. `allocatedBytePtr`
  /// is set to null for stack allocations. `accessAlignment` is set if
  /// alignment is needed post allocation (for eg. in conjunction with malloc).
  std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          Operation *op) const override {

    // With alloca, one gets a pointer to the element type right away.
    // For stack allocations.
    auto allocaOp = cast<memref::AllocaOp>(op);
    auto elementType =
        typeConverter->convertType(allocaOp.getType().getElementType());
    unsigned addrSpace =
        *getTypeConverter()->getMemRefAddressSpace(allocaOp.getType());
    auto elementPtrType =
        getTypeConverter()->getPointerType(elementType, addrSpace);

    auto allocatedElementPtr = rewriter.create<LLVM::AllocaOp>(
        loc, elementPtrType, elementType, sizeBytes,
        allocaOp.getAlignment().value_or(0));

    return std::make_tuple(allocatedElementPtr, allocatedElementPtr);
  }
};

/// The base class for lowering realloc op, to support the implementation of
/// realloc via allocation methods that may or may not support alignment.
/// A derived class should provide an implementation of allocateBuffer using
/// the underline allocation methods.
struct ReallocOpLoweringBase : public AllocationOpLLVMLowering {
  using OpAdaptor = typename memref::ReallocOp::Adaptor;

  ReallocOpLoweringBase(LLVMTypeConverter &converter)
      : AllocationOpLLVMLowering(memref::ReallocOp::getOperationName(),
                                 converter) {}

  /// Allocates the new buffer. Returns the allocated pointer and the
  /// aligned pointer.
  virtual std::tuple<Value, Value>
  allocateBuffer(ConversionPatternRewriter &rewriter, Location loc,
                 Value sizeBytes, memref::ReallocOp op) const = 0;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    return matchAndRewrite(cast<memref::ReallocOp>(op),
                           OpAdaptor(operands, op->getAttrDictionary()),
                           rewriter);
  }

  // A `realloc` is converted as follows:
  //   If new_size > old_size
  //     1. allocates a new buffer
  //     2. copies the content of the old buffer to the new buffer
  //     3. release the old buffer
  //     3. updates the buffer pointers in the memref descriptor
  //   Update the size in the memref descriptor
  // Alignment request is handled by allocating `alignment` more bytes than
  // requested and shifting the aligned pointer relative to the allocated
  // memory.
  LogicalResult matchAndRewrite(memref::ReallocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    OpBuilder::InsertionGuard guard(rewriter);
    Location loc = op.getLoc();

    auto computeNumElements =
        [&](MemRefType type, function_ref<Value()> getDynamicSize) -> Value {
      // Compute number of elements.
      int64_t size = type.getShape()[0];
      Value numElements = ((size == ShapedType::kDynamic)
                               ? getDynamicSize()
                               : createIndexConstant(rewriter, loc, size));
      Type indexType = getIndexType();
      if (numElements.getType() != indexType)
        numElements = typeConverter->materializeTargetConversion(
            rewriter, loc, indexType, numElements);
      return numElements;
    };

    MemRefDescriptor desc(adaptor.getSource());
    Value oldDesc = desc;

    // Split the block right before the current op into two blocks.
    Block *currentBlock = rewriter.getInsertionBlock();
    Block *block =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    // Add a block argument by creating an empty block with the argument type
    // and then merging the block into the empty block.
    Block *endBlock = rewriter.createBlock(
        block->getParent(), Region::iterator(block), oldDesc.getType(), loc);
    rewriter.mergeBlocks(block, endBlock, {});
    // Add a new block for the true branch of the conditional statement we will
    // add.
    Block *trueBlock = rewriter.createBlock(
        currentBlock->getParent(), std::next(Region::iterator(currentBlock)));

    rewriter.setInsertionPointToEnd(currentBlock);
    Value src = op.getSource();
    auto srcType = src.getType().dyn_cast<MemRefType>();
    Value srcNumElements = computeNumElements(
        srcType, [&]() -> Value { return desc.size(rewriter, loc, 0); });
    auto dstType = op.getType().cast<MemRefType>();
    Value dstNumElements = computeNumElements(
        dstType, [&]() -> Value { return op.getDynamicResultSize(); });
    Value cond = rewriter.create<LLVM::ICmpOp>(
        loc, IntegerType::get(rewriter.getContext(), 1),
        LLVM::ICmpPredicate::ugt, dstNumElements, srcNumElements);
    rewriter.create<LLVM::CondBrOp>(loc, cond, trueBlock, ArrayRef<Value>(),
                                    endBlock, ValueRange{oldDesc});

    rewriter.setInsertionPointToStart(trueBlock);
    Value sizeInBytes = getSizeInBytes(loc, dstType.getElementType(), rewriter);
    // Compute total byte size.
    auto dstByteSize =
        rewriter.create<LLVM::MulOp>(loc, dstNumElements, sizeInBytes);
    // Since the src and dst memref are guarantee to have the same
    // element type by the verifier, it is safe here to reuse the
    // type size computed from dst memref.
    auto srcByteSize =
        rewriter.create<LLVM::MulOp>(loc, srcNumElements, sizeInBytes);
    // Allocate a new buffer.
    auto [dstRawPtr, dstAlignedPtr] =
        allocateBuffer(rewriter, loc, dstByteSize, op);
    // Copy the data from the old buffer to the new buffer.
    Value srcAlignedPtr = desc.alignedPtr(rewriter, loc);
    Value isVolatile =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getBoolAttr(false));
    auto toVoidPtr = [&](Value ptr) -> Value {
      if (getTypeConverter()->useOpaquePointers())
        return ptr;
      return rewriter.create<LLVM::BitcastOp>(loc, getVoidPtrType(), ptr);
    };
    rewriter.create<LLVM::MemcpyOp>(loc, toVoidPtr(dstAlignedPtr),
                                    toVoidPtr(srcAlignedPtr), srcByteSize,
                                    isVolatile);
    // Deallocate the old buffer.
    LLVM::LLVMFuncOp freeFunc =
        getFreeFn(getTypeConverter(), op->getParentOfType<ModuleOp>());
    rewriter.create<LLVM::CallOp>(loc, freeFunc,
                                  toVoidPtr(desc.allocatedPtr(rewriter, loc)));
    // Replace the old buffer addresses in the MemRefDescriptor with the new
    // buffer addresses.
    desc.setAllocatedPtr(rewriter, loc, dstRawPtr);
    desc.setAlignedPtr(rewriter, loc, dstAlignedPtr);
    rewriter.create<LLVM::BrOp>(loc, Value(desc), endBlock);

    rewriter.setInsertionPoint(op);
    // Update the memref size.
    MemRefDescriptor newDesc(endBlock->getArgument(0));
    newDesc.setSize(rewriter, loc, 0, dstNumElements);
    rewriter.replaceOp(op, {newDesc});
    return success();
  }

private:
  using ConvertToLLVMPattern::matchAndRewrite;
};

struct ReallocOpLowering : public ReallocOpLoweringBase {
  ReallocOpLowering(LLVMTypeConverter &converter)
      : ReallocOpLoweringBase(converter) {}
  std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          memref::ReallocOp op) const override {
    return allocateBufferManuallyAlign(rewriter, loc, sizeBytes, op,
                                       getAlignment(rewriter, loc, op));
  }
};

struct AlignedReallocOpLowering : public ReallocOpLoweringBase {
  AlignedReallocOpLowering(LLVMTypeConverter &converter)
      : ReallocOpLoweringBase(converter) {}
  std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          memref::ReallocOp op) const override {
    Value ptr = allocateBufferAutoAlign(
        rewriter, loc, sizeBytes, op, &defaultLayout,
        alignedAllocationGetAlignment(rewriter, loc, op, &defaultLayout));
    return std::make_tuple(ptr, ptr);
  }

private:
  /// Default layout to use in absence of the corresponding analysis.
  DataLayout defaultLayout;
};

struct AllocaScopeOpLowering
    : public ConvertOpToLLVMPattern<memref::AllocaScopeOp> {
  using ConvertOpToLLVMPattern<memref::AllocaScopeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::AllocaScopeOp allocaScopeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    Location loc = allocaScopeOp.getLoc();

    // Split the current block before the AllocaScopeOp to create the inlining
    // point.
    auto *currentBlock = rewriter.getInsertionBlock();
    auto *remainingOpsBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *continueBlock;
    if (allocaScopeOp.getNumResults() == 0) {
      continueBlock = remainingOpsBlock;
    } else {
      continueBlock = rewriter.createBlock(
          remainingOpsBlock, allocaScopeOp.getResultTypes(),
          SmallVector<Location>(allocaScopeOp->getNumResults(),
                                allocaScopeOp.getLoc()));
      rewriter.create<LLVM::BrOp>(loc, ValueRange(), remainingOpsBlock);
    }

    // Inline body region.
    Block *beforeBody = &allocaScopeOp.getBodyRegion().front();
    Block *afterBody = &allocaScopeOp.getBodyRegion().back();
    rewriter.inlineRegionBefore(allocaScopeOp.getBodyRegion(), continueBlock);

    // Save stack and then branch into the body of the region.
    rewriter.setInsertionPointToEnd(currentBlock);
    auto stackSaveOp =
        rewriter.create<LLVM::StackSaveOp>(loc, getVoidPtrType());
    rewriter.create<LLVM::BrOp>(loc, ValueRange(), beforeBody);

    // Replace the alloca_scope return with a branch that jumps out of the body.
    // Stack restore before leaving the body region.
    rewriter.setInsertionPointToEnd(afterBody);
    auto returnOp =
        cast<memref::AllocaScopeReturnOp>(afterBody->getTerminator());
    auto branchOp = rewriter.replaceOpWithNewOp<LLVM::BrOp>(
        returnOp, returnOp.getResults(), continueBlock);

    // Insert stack restore before jumping out the body of the region.
    rewriter.setInsertionPoint(branchOp);
    rewriter.create<LLVM::StackRestoreOp>(loc, stackSaveOp);

    // Replace the op with values return from the body region.
    rewriter.replaceOp(allocaScopeOp, continueBlock->getArguments());

    return success();
  }
};

struct AssumeAlignmentOpLowering
    : public ConvertOpToLLVMPattern<memref::AssumeAlignmentOp> {
  using ConvertOpToLLVMPattern<
      memref::AssumeAlignmentOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::AssumeAlignmentOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value memref = adaptor.getMemref();
    unsigned alignment = op.getAlignment();
    auto loc = op.getLoc();

    MemRefDescriptor memRefDescriptor(memref);
    Value ptr = memRefDescriptor.alignedPtr(rewriter, memref.getLoc());

    // Emit llvm.assume(memref.alignedPtr & (alignment - 1) == 0). Notice that
    // the asserted memref.alignedPtr isn't used anywhere else, as the real
    // users like load/store/views always re-extract memref.alignedPtr as they
    // get lowered.
    //
    // This relies on LLVM's CSE optimization (potentially after SROA), since
    // after CSE all memref.alignedPtr instances get de-duplicated into the same
    // pointer SSA value.
    auto intPtrType =
        getIntPtrType(memRefDescriptor.getElementPtrType().getAddressSpace());
    Value zero = createIndexAttrConstant(rewriter, loc, intPtrType, 0);
    Value mask =
        createIndexAttrConstant(rewriter, loc, intPtrType, alignment - 1);
    Value ptrValue = rewriter.create<LLVM::PtrToIntOp>(loc, intPtrType, ptr);
    rewriter.create<LLVM::AssumeOp>(
        loc, rewriter.create<LLVM::ICmpOp>(
                 loc, LLVM::ICmpPredicate::eq,
                 rewriter.create<LLVM::AndOp>(loc, ptrValue, mask), zero));

    rewriter.eraseOp(op);
    return success();
  }
};

// A `dealloc` is converted into a call to `free` on the underlying data buffer.
// The memref descriptor being an SSA value, there is no need to clean it up
// in any way.
struct DeallocOpLowering : public ConvertOpToLLVMPattern<memref::DeallocOp> {
  using ConvertOpToLLVMPattern<memref::DeallocOp>::ConvertOpToLLVMPattern;

  explicit DeallocOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<memref::DeallocOp>(converter) {}

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Insert the `free` declaration if it is not already present.
    LLVM::LLVMFuncOp freeFunc =
        getFreeFn(getTypeConverter(), op->getParentOfType<ModuleOp>());
    MemRefDescriptor memref(adaptor.getMemref());
    Value allocatedPtr = memref.allocatedPtr(rewriter, op.getLoc());
    Value casted = allocatedPtr;
    if (!getTypeConverter()->useOpaquePointers())
      casted = rewriter.create<LLVM::BitcastOp>(op.getLoc(), getVoidPtrType(),
                                                allocatedPtr);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, freeFunc, casted);
    return success();
  }
};

// A `dim` is converted to a constant for static sizes and to an access to the
// size stored in the memref descriptor for dynamic sizes.
struct DimOpLowering : public ConvertOpToLLVMPattern<memref::DimOp> {
  using ConvertOpToLLVMPattern<memref::DimOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::DimOp dimOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type operandType = dimOp.getSource().getType();
    if (operandType.isa<UnrankedMemRefType>()) {
      FailureOr<Value> extractedSize = extractSizeOfUnrankedMemRef(
          operandType, dimOp, adaptor.getOperands(), rewriter);
      if (failed(extractedSize))
        return failure();
      rewriter.replaceOp(dimOp, {*extractedSize});
      return success();
    }
    if (operandType.isa<MemRefType>()) {
      rewriter.replaceOp(
          dimOp, {extractSizeOfRankedMemRef(operandType, dimOp,
                                            adaptor.getOperands(), rewriter)});
      return success();
    }
    llvm_unreachable("expected MemRefType or UnrankedMemRefType");
  }

private:
  FailureOr<Value>
  extractSizeOfUnrankedMemRef(Type operandType, memref::DimOp dimOp,
                              OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    Location loc = dimOp.getLoc();

    auto unrankedMemRefType = operandType.cast<UnrankedMemRefType>();
    auto scalarMemRefType =
        MemRefType::get({}, unrankedMemRefType.getElementType());
    FailureOr<unsigned> maybeAddressSpace =
        getTypeConverter()->getMemRefAddressSpace(unrankedMemRefType);
    if (failed(maybeAddressSpace)) {
      dimOp.emitOpError("memref memory space must be convertible to an integer "
                        "address space");
      return failure();
    }
    unsigned addressSpace = *maybeAddressSpace;

    // Extract pointer to the underlying ranked descriptor and bitcast it to a
    // memref<element_type> descriptor pointer to minimize the number of GEP
    // operations.
    UnrankedMemRefDescriptor unrankedDesc(adaptor.getSource());
    Value underlyingRankedDesc = unrankedDesc.memRefDescPtr(rewriter, loc);

    Type elementType = typeConverter->convertType(scalarMemRefType);
    Value scalarMemRefDescPtr;
    if (getTypeConverter()->useOpaquePointers())
      scalarMemRefDescPtr = underlyingRankedDesc;
    else
      scalarMemRefDescPtr = rewriter.create<LLVM::BitcastOp>(
          loc, LLVM::LLVMPointerType::get(elementType, addressSpace),
          underlyingRankedDesc);

    // Get pointer to offset field of memref<element_type> descriptor.
    Type indexPtrTy = getTypeConverter()->getPointerType(
        getTypeConverter()->getIndexType(), addressSpace);
    Value offsetPtr = rewriter.create<LLVM::GEPOp>(
        loc, indexPtrTy, elementType, scalarMemRefDescPtr,
        ArrayRef<LLVM::GEPArg>{0, 2});

    // The size value that we have to extract can be obtained using GEPop with
    // `dimOp.index() + 1` index argument.
    Value idxPlusOne = rewriter.create<LLVM::AddOp>(
        loc, createIndexConstant(rewriter, loc, 1), adaptor.getIndex());
    Value sizePtr = rewriter.create<LLVM::GEPOp>(
        loc, indexPtrTy, getTypeConverter()->getIndexType(), offsetPtr,
        idxPlusOne);
    return rewriter
        .create<LLVM::LoadOp>(loc, getTypeConverter()->getIndexType(), sizePtr)
        .getResult();
  }

  std::optional<int64_t> getConstantDimIndex(memref::DimOp dimOp) const {
    if (auto idx = dimOp.getConstantIndex())
      return idx;

    if (auto constantOp = dimOp.getIndex().getDefiningOp<LLVM::ConstantOp>())
      return constantOp.getValue()
          .cast<IntegerAttr>()
          .getValue()
          .getSExtValue();

    return std::nullopt;
  }

  Value extractSizeOfRankedMemRef(Type operandType, memref::DimOp dimOp,
                                  OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    Location loc = dimOp.getLoc();

    // Take advantage if index is constant.
    MemRefType memRefType = operandType.cast<MemRefType>();
    if (std::optional<int64_t> index = getConstantDimIndex(dimOp)) {
      int64_t i = *index;
      if (i >= 0 && i < memRefType.getRank()) {
        if (memRefType.isDynamicDim(i)) {
          // extract dynamic size from the memref descriptor.
          MemRefDescriptor descriptor(adaptor.getSource());
          return descriptor.size(rewriter, loc, i);
        }
        // Use constant for static size.
        int64_t dimSize = memRefType.getDimSize(i);
        return createIndexConstant(rewriter, loc, dimSize);
      }
    }
    Value index = adaptor.getIndex();
    int64_t rank = memRefType.getRank();
    MemRefDescriptor memrefDescriptor(adaptor.getSource());
    return memrefDescriptor.size(rewriter, loc, index, rank);
  }
};

/// Common base for load and store operations on MemRefs. Restricts the match
/// to supported MemRef types. Provides functionality to emit code accessing a
/// specific element of the underlying data buffer.
template <typename Derived>
struct LoadStoreOpLowering : public ConvertOpToLLVMPattern<Derived> {
  using ConvertOpToLLVMPattern<Derived>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<Derived>::isConvertibleAndHasIdentityMaps;
  using Base = LoadStoreOpLowering<Derived>;

  LogicalResult match(Derived op) const override {
    MemRefType type = op.getMemRefType();
    return isConvertibleAndHasIdentityMaps(type) ? success() : failure();
  }
};

/// Wrap a llvm.cmpxchg operation in a while loop so that the operation can be
/// retried until it succeeds in atomically storing a new value into memory.
///
///      +---------------------------------+
///      |   <code before the AtomicRMWOp> |
///      |   <compute initial %loaded>     |
///      |   cf.br loop(%loaded)              |
///      +---------------------------------+
///             |
///  -------|   |
///  |      v   v
///  |   +--------------------------------+
///  |   | loop(%loaded):                 |
///  |   |   <body contents>              |
///  |   |   %pair = cmpxchg              |
///  |   |   %ok = %pair[0]               |
///  |   |   %new = %pair[1]              |
///  |   |   cf.cond_br %ok, end, loop(%new) |
///  |   +--------------------------------+
///  |          |        |
///  |-----------        |
///                      v
///      +--------------------------------+
///      | end:                           |
///      |   <code after the AtomicRMWOp> |
///      +--------------------------------+
///
struct GenericAtomicRMWOpLowering
    : public LoadStoreOpLowering<memref::GenericAtomicRMWOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(memref::GenericAtomicRMWOp atomicOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = atomicOp.getLoc();
    Type valueType = typeConverter->convertType(atomicOp.getResult().getType());

    // Split the block into initial, loop, and ending parts.
    auto *initBlock = rewriter.getInsertionBlock();
    auto *loopBlock = rewriter.createBlock(
        initBlock->getParent(), std::next(Region::iterator(initBlock)),
        valueType, loc);
    auto *endBlock = rewriter.createBlock(
        loopBlock->getParent(), std::next(Region::iterator(loopBlock)));

    // Operations range to be moved to `endBlock`.
    auto opsToMoveStart = atomicOp->getIterator();
    auto opsToMoveEnd = initBlock->back().getIterator();

    // Compute the loaded value and branch to the loop block.
    rewriter.setInsertionPointToEnd(initBlock);
    auto memRefType = atomicOp.getMemref().getType().cast<MemRefType>();
    auto dataPtr = getStridedElementPtr(loc, memRefType, adaptor.getMemref(),
                                        adaptor.getIndices(), rewriter);
    Value init = rewriter.create<LLVM::LoadOp>(
        loc, typeConverter->convertType(memRefType.getElementType()), dataPtr);
    rewriter.create<LLVM::BrOp>(loc, init, loopBlock);

    // Prepare the body of the loop block.
    rewriter.setInsertionPointToStart(loopBlock);

    // Clone the GenericAtomicRMWOp region and extract the result.
    auto loopArgument = loopBlock->getArgument(0);
    IRMapping mapping;
    mapping.map(atomicOp.getCurrentValue(), loopArgument);
    Block &entryBlock = atomicOp.body().front();
    for (auto &nestedOp : entryBlock.without_terminator()) {
      Operation *clone = rewriter.clone(nestedOp, mapping);
      mapping.map(nestedOp.getResults(), clone->getResults());
    }
    Value result = mapping.lookup(entryBlock.getTerminator()->getOperand(0));

    // Prepare the epilog of the loop block.
    // Append the cmpxchg op to the end of the loop block.
    auto successOrdering = LLVM::AtomicOrdering::acq_rel;
    auto failureOrdering = LLVM::AtomicOrdering::monotonic;
    auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
        loc, dataPtr, loopArgument, result, successOrdering, failureOrdering);
    // Extract the %new_loaded and %ok values from the pair.
    Value newLoaded = rewriter.create<LLVM::ExtractValueOp>(loc, cmpxchg, 0);
    Value ok = rewriter.create<LLVM::ExtractValueOp>(loc, cmpxchg, 1);

    // Conditionally branch to the end or back to the loop depending on %ok.
    rewriter.create<LLVM::CondBrOp>(loc, ok, endBlock, ArrayRef<Value>(),
                                    loopBlock, newLoaded);

    rewriter.setInsertionPointToEnd(endBlock);
    moveOpsRange(atomicOp.getResult(), newLoaded, std::next(opsToMoveStart),
                 std::next(opsToMoveEnd), rewriter);

    // The 'result' of the atomic_rmw op is the newly loaded value.
    rewriter.replaceOp(atomicOp, {newLoaded});

    return success();
  }

private:
  // Clones a segment of ops [start, end) and erases the original.
  void moveOpsRange(ValueRange oldResult, ValueRange newResult,
                    Block::iterator start, Block::iterator end,
                    ConversionPatternRewriter &rewriter) const {
    IRMapping mapping;
    mapping.map(oldResult, newResult);
    SmallVector<Operation *, 2> opsToErase;
    for (auto it = start; it != end; ++it) {
      rewriter.clone(*it, mapping);
      opsToErase.push_back(&*it);
    }
    for (auto *it : opsToErase)
      rewriter.eraseOp(it);
  }
};

/// Returns the LLVM type of the global variable given the memref type `type`.
static Type convertGlobalMemrefTypeToLLVM(MemRefType type,
                                          LLVMTypeConverter &typeConverter) {
  // LLVM type for a global memref will be a multi-dimension array. For
  // declarations or uninitialized global memrefs, we can potentially flatten
  // this to a 1D array. However, for memref.global's with an initial value,
  // we do not intend to flatten the ElementsAttribute when going from std ->
  // LLVM dialect, so the LLVM type needs to me a multi-dimension array.
  Type elementType = typeConverter.convertType(type.getElementType());
  Type arrayTy = elementType;
  // Shape has the outermost dim at index 0, so need to walk it backwards
  for (int64_t dim : llvm::reverse(type.getShape()))
    arrayTy = LLVM::LLVMArrayType::get(arrayTy, dim);
  return arrayTy;
}

/// GlobalMemrefOp is lowered to a LLVM Global Variable.
struct GlobalMemrefOpLowering
    : public ConvertOpToLLVMPattern<memref::GlobalOp> {
  using ConvertOpToLLVMPattern<memref::GlobalOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::GlobalOp global, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = global.getType();
    if (!isConvertibleAndHasIdentityMaps(type))
      return failure();

    Type arrayTy = convertGlobalMemrefTypeToLLVM(type, *getTypeConverter());

    LLVM::Linkage linkage =
        global.isPublic() ? LLVM::Linkage::External : LLVM::Linkage::Private;

    Attribute initialValue = nullptr;
    if (!global.isExternal() && !global.isUninitialized()) {
      auto elementsAttr = global.getInitialValue()->cast<ElementsAttr>();
      initialValue = elementsAttr;

      // For scalar memrefs, the global variable created is of the element type,
      // so unpack the elements attribute to extract the value.
      if (type.getRank() == 0)
        initialValue = elementsAttr.getSplatValue<Attribute>();
    }

    uint64_t alignment = global.getAlignment().value_or(0);
    FailureOr<unsigned> addressSpace =
        getTypeConverter()->getMemRefAddressSpace(type);
    if (failed(addressSpace))
      return global.emitOpError(
          "memory space cannot be converted to an integer address space");
    auto newGlobal = rewriter.replaceOpWithNewOp<LLVM::GlobalOp>(
        global, arrayTy, global.getConstant(), linkage, global.getSymName(),
        initialValue, alignment, *addressSpace);
    if (!global.isExternal() && global.isUninitialized()) {
      Block *blk = new Block();
      newGlobal.getInitializerRegion().push_back(blk);
      rewriter.setInsertionPointToStart(blk);
      Value undef[] = {
          rewriter.create<LLVM::UndefOp>(global.getLoc(), arrayTy)};
      rewriter.create<LLVM::ReturnOp>(global.getLoc(), undef);
    }
    return success();
  }
};

/// GetGlobalMemrefOp is lowered into a Memref descriptor with the pointer to
/// the first element stashed into the descriptor. This reuses
/// `AllocLikeOpLowering` to reuse the Memref descriptor construction.
struct GetGlobalMemrefOpLowering : public AllocLikeOpLLVMLowering {
  GetGlobalMemrefOpLowering(LLVMTypeConverter &converter)
      : AllocLikeOpLLVMLowering(memref::GetGlobalOp::getOperationName(),
                                converter) {}

  /// Buffer "allocation" for memref.get_global op is getting the address of
  /// the global variable referenced.
  std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          Operation *op) const override {
    auto getGlobalOp = cast<memref::GetGlobalOp>(op);
    MemRefType type = getGlobalOp.getResult().getType().cast<MemRefType>();

    // This is called after a type conversion, which would have failed if this
    // call fails.
    unsigned memSpace = *getTypeConverter()->getMemRefAddressSpace(type);

    Type arrayTy = convertGlobalMemrefTypeToLLVM(type, *getTypeConverter());
    Type resTy = getTypeConverter()->getPointerType(arrayTy, memSpace);
    auto addressOf =
        rewriter.create<LLVM::AddressOfOp>(loc, resTy, getGlobalOp.getName());

    // Get the address of the first element in the array by creating a GEP with
    // the address of the GV as the base, and (rank + 1) number of 0 indices.
    Type elementType = typeConverter->convertType(type.getElementType());
    Type elementPtrType =
        getTypeConverter()->getPointerType(elementType, memSpace);

    auto gep = rewriter.create<LLVM::GEPOp>(
        loc, elementPtrType, arrayTy, addressOf,
        SmallVector<LLVM::GEPArg>(type.getRank() + 1, 0));

    // We do not expect the memref obtained using `memref.get_global` to be
    // ever deallocated. Set the allocated pointer to be known bad value to
    // help debug if that ever happens.
    auto intPtrType = getIntPtrType(memSpace);
    Value deadBeefConst =
        createIndexAttrConstant(rewriter, op->getLoc(), intPtrType, 0xdeadbeef);
    auto deadBeefPtr =
        rewriter.create<LLVM::IntToPtrOp>(loc, elementPtrType, deadBeefConst);

    // Both allocated and aligned pointers are same. We could potentially stash
    // a nullptr for the allocated pointer since we do not expect any dealloc.
    return std::make_tuple(deadBeefPtr, gep);
  }
};

// Load operation is lowered to obtaining a pointer to the indexed element
// and loading it.
struct LoadOpLowering : public LoadStoreOpLowering<memref::LoadOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(memref::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = loadOp.getMemRefType();

    Value dataPtr =
        getStridedElementPtr(loadOp.getLoc(), type, adaptor.getMemref(),
                             adaptor.getIndices(), rewriter);
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        loadOp, typeConverter->convertType(type.getElementType()), dataPtr, 0,
        false, loadOp.getNontemporal());
    return success();
  }
};

// Store operation is lowered to obtaining a pointer to the indexed element,
// and storing the given value to it.
struct StoreOpLowering : public LoadStoreOpLowering<memref::StoreOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = op.getMemRefType();

    Value dataPtr = getStridedElementPtr(op.getLoc(), type, adaptor.getMemref(),
                                         adaptor.getIndices(), rewriter);
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValue(), dataPtr,
                                               0, false, op.getNontemporal());
    return success();
  }
};

// The prefetch operation is lowered in a way similar to the load operation
// except that the llvm.prefetch operation is used for replacement.
struct PrefetchOpLowering : public LoadStoreOpLowering<memref::PrefetchOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(memref::PrefetchOp prefetchOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = prefetchOp.getMemRefType();
    auto loc = prefetchOp.getLoc();

    Value dataPtr = getStridedElementPtr(loc, type, adaptor.getMemref(),
                                         adaptor.getIndices(), rewriter);

    // Replace with llvm.prefetch.
    auto llvmI32Type = typeConverter->convertType(rewriter.getIntegerType(32));
    auto isWrite = rewriter.create<LLVM::ConstantOp>(loc, llvmI32Type,
                                                     prefetchOp.getIsWrite());
    auto localityHint = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32Type, prefetchOp.getLocalityHint());
    auto isData = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32Type, prefetchOp.getIsDataCache());

    rewriter.replaceOpWithNewOp<LLVM::Prefetch>(prefetchOp, dataPtr, isWrite,
                                                localityHint, isData);
    return success();
  }
};

struct RankOpLowering : public ConvertOpToLLVMPattern<memref::RankOp> {
  using ConvertOpToLLVMPattern<memref::RankOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::RankOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type operandType = op.getMemref().getType();
    if (auto unrankedMemRefType = operandType.dyn_cast<UnrankedMemRefType>()) {
      UnrankedMemRefDescriptor desc(adaptor.getMemref());
      rewriter.replaceOp(op, {desc.rank(rewriter, loc)});
      return success();
    }
    if (auto rankedMemRefType = operandType.dyn_cast<MemRefType>()) {
      rewriter.replaceOp(
          op, {createIndexConstant(rewriter, loc, rankedMemRefType.getRank())});
      return success();
    }
    return failure();
  }
};

struct MemRefCastOpLowering : public ConvertOpToLLVMPattern<memref::CastOp> {
  using ConvertOpToLLVMPattern<memref::CastOp>::ConvertOpToLLVMPattern;

  LogicalResult match(memref::CastOp memRefCastOp) const override {
    Type srcType = memRefCastOp.getOperand().getType();
    Type dstType = memRefCastOp.getType();

    // memref::CastOp reduce to bitcast in the ranked MemRef case and can be
    // used for type erasure. For now they must preserve underlying element type
    // and require source and result type to have the same rank. Therefore,
    // perform a sanity check that the underlying structs are the same. Once op
    // semantics are relaxed we can revisit.
    if (srcType.isa<MemRefType>() && dstType.isa<MemRefType>())
      return success(typeConverter->convertType(srcType) ==
                     typeConverter->convertType(dstType));

    // At least one of the operands is unranked type
    assert(srcType.isa<UnrankedMemRefType>() ||
           dstType.isa<UnrankedMemRefType>());

    // Unranked to unranked cast is disallowed
    return !(srcType.isa<UnrankedMemRefType>() &&
             dstType.isa<UnrankedMemRefType>())
               ? success()
               : failure();
  }

  void rewrite(memref::CastOp memRefCastOp, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    auto srcType = memRefCastOp.getOperand().getType();
    auto dstType = memRefCastOp.getType();
    auto targetStructType = typeConverter->convertType(memRefCastOp.getType());
    auto loc = memRefCastOp.getLoc();

    // For ranked/ranked case, just keep the original descriptor.
    if (srcType.isa<MemRefType>() && dstType.isa<MemRefType>())
      return rewriter.replaceOp(memRefCastOp, {adaptor.getSource()});

    if (srcType.isa<MemRefType>() && dstType.isa<UnrankedMemRefType>()) {
      // Casting ranked to unranked memref type
      // Set the rank in the destination from the memref type
      // Allocate space on the stack and copy the src memref descriptor
      // Set the ptr in the destination to the stack space
      auto srcMemRefType = srcType.cast<MemRefType>();
      int64_t rank = srcMemRefType.getRank();
      // ptr = AllocaOp sizeof(MemRefDescriptor)
      auto ptr = getTypeConverter()->promoteOneMemRefDescriptor(
          loc, adaptor.getSource(), rewriter);

      // voidptr = BitCastOp srcType* to void*
      Value voidPtr;
      if (getTypeConverter()->useOpaquePointers())
        voidPtr = ptr;
      else
        voidPtr = rewriter.create<LLVM::BitcastOp>(loc, getVoidPtrType(), ptr);

      // rank = ConstantOp srcRank
      auto rankVal = rewriter.create<LLVM::ConstantOp>(
          loc, getIndexType(), rewriter.getIndexAttr(rank));
      // undef = UndefOp
      UnrankedMemRefDescriptor memRefDesc =
          UnrankedMemRefDescriptor::undef(rewriter, loc, targetStructType);
      // d1 = InsertValueOp undef, rank, 0
      memRefDesc.setRank(rewriter, loc, rankVal);
      // d2 = InsertValueOp d1, voidptr, 1
      memRefDesc.setMemRefDescPtr(rewriter, loc, voidPtr);
      rewriter.replaceOp(memRefCastOp, (Value)memRefDesc);

    } else if (srcType.isa<UnrankedMemRefType>() && dstType.isa<MemRefType>()) {
      // Casting from unranked type to ranked.
      // The operation is assumed to be doing a correct cast. If the destination
      // type mismatches the unranked the type, it is undefined behavior.
      UnrankedMemRefDescriptor memRefDesc(adaptor.getSource());
      // ptr = ExtractValueOp src, 1
      auto ptr = memRefDesc.memRefDescPtr(rewriter, loc);
      // castPtr = BitCastOp i8* to structTy*
      Value castPtr;
      if (getTypeConverter()->useOpaquePointers())
        castPtr = ptr;
      else
        castPtr = rewriter.create<LLVM::BitcastOp>(
            loc, LLVM::LLVMPointerType::get(targetStructType), ptr);

      // struct = LoadOp castPtr
      auto loadOp =
          rewriter.create<LLVM::LoadOp>(loc, targetStructType, castPtr);
      rewriter.replaceOp(memRefCastOp, loadOp.getResult());
    } else {
      llvm_unreachable("Unsupported unranked memref to unranked memref cast");
    }
  }
};

/// Pattern to lower a `memref.copy` to llvm.
///
/// For memrefs with identity layouts, the copy is lowered to the llvm
/// `memcpy` intrinsic. For non-identity layouts, the copy is lowered to a call
/// to the generic `MemrefCopyFn`.
struct MemRefCopyOpLowering : public ConvertOpToLLVMPattern<memref::CopyOp> {
  using ConvertOpToLLVMPattern<memref::CopyOp>::ConvertOpToLLVMPattern;

  LogicalResult
  lowerToMemCopyIntrinsic(memref::CopyOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcType = op.getSource().getType().dyn_cast<MemRefType>();

    MemRefDescriptor srcDesc(adaptor.getSource());

    // Compute number of elements.
    Value numElements = rewriter.create<LLVM::ConstantOp>(
        loc, getIndexType(), rewriter.getIndexAttr(1));
    for (int pos = 0; pos < srcType.getRank(); ++pos) {
      auto size = srcDesc.size(rewriter, loc, pos);
      numElements = rewriter.create<LLVM::MulOp>(loc, numElements, size);
    }

    // Get element size.
    auto sizeInBytes = getSizeInBytes(loc, srcType.getElementType(), rewriter);
    // Compute total.
    Value totalSize =
        rewriter.create<LLVM::MulOp>(loc, numElements, sizeInBytes);

    Type elementType = typeConverter->convertType(srcType.getElementType());

    Value srcBasePtr = srcDesc.alignedPtr(rewriter, loc);
    Value srcOffset = srcDesc.offset(rewriter, loc);
    Value srcPtr = rewriter.create<LLVM::GEPOp>(
        loc, srcBasePtr.getType(), elementType, srcBasePtr, srcOffset);
    MemRefDescriptor targetDesc(adaptor.getTarget());
    Value targetBasePtr = targetDesc.alignedPtr(rewriter, loc);
    Value targetOffset = targetDesc.offset(rewriter, loc);
    Value targetPtr = rewriter.create<LLVM::GEPOp>(
        loc, targetBasePtr.getType(), elementType, targetBasePtr, targetOffset);
    Value isVolatile =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getBoolAttr(false));
    rewriter.create<LLVM::MemcpyOp>(loc, targetPtr, srcPtr, totalSize,
                                    isVolatile);
    rewriter.eraseOp(op);

    return success();
  }

  LogicalResult
  lowerToMemCopyFunctionCall(memref::CopyOp op, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcType = op.getSource().getType().cast<BaseMemRefType>();
    auto targetType = op.getTarget().getType().cast<BaseMemRefType>();

    // First make sure we have an unranked memref descriptor representation.
    auto makeUnranked = [&, this](Value ranked, BaseMemRefType type) {
      auto rank = rewriter.create<LLVM::ConstantOp>(loc, getIndexType(),
                                                    type.getRank());
      auto *typeConverter = getTypeConverter();
      auto ptr =
          typeConverter->promoteOneMemRefDescriptor(loc, ranked, rewriter);

      Value voidPtr;
      if (getTypeConverter()->useOpaquePointers())
        voidPtr = ptr;
      else
        voidPtr = rewriter.create<LLVM::BitcastOp>(loc, getVoidPtrType(), ptr);

      auto unrankedType =
          UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace());
      return UnrankedMemRefDescriptor::pack(rewriter, loc, *typeConverter,
                                            unrankedType,
                                            ValueRange{rank, voidPtr});
    };

    // Save stack position before promoting descriptors
    auto stackSaveOp =
        rewriter.create<LLVM::StackSaveOp>(loc, getVoidPtrType());

    Value unrankedSource = srcType.hasRank()
                               ? makeUnranked(adaptor.getSource(), srcType)
                               : adaptor.getSource();
    Value unrankedTarget = targetType.hasRank()
                               ? makeUnranked(adaptor.getTarget(), targetType)
                               : adaptor.getTarget();

    // Now promote the unranked descriptors to the stack.
    auto one = rewriter.create<LLVM::ConstantOp>(loc, getIndexType(),
                                                 rewriter.getIndexAttr(1));
    auto promote = [&](Value desc) {
      Type ptrType = getTypeConverter()->getPointerType(desc.getType());
      auto allocated =
          rewriter.create<LLVM::AllocaOp>(loc, ptrType, desc.getType(), one);
      rewriter.create<LLVM::StoreOp>(loc, desc, allocated);
      return allocated;
    };

    auto sourcePtr = promote(unrankedSource);
    auto targetPtr = promote(unrankedTarget);

    unsigned typeSize =
        mlir::DataLayout::closest(op).getTypeSize(srcType.getElementType());
    auto elemSize = rewriter.create<LLVM::ConstantOp>(
        loc, getIndexType(), rewriter.getIndexAttr(typeSize));
    auto copyFn = LLVM::lookupOrCreateMemRefCopyFn(
        op->getParentOfType<ModuleOp>(), getIndexType(), sourcePtr.getType());
    rewriter.create<LLVM::CallOp>(loc, copyFn,
                                  ValueRange{elemSize, sourcePtr, targetPtr});

    // Restore stack used for descriptors
    rewriter.create<LLVM::StackRestoreOp>(loc, stackSaveOp);

    rewriter.eraseOp(op);

    return success();
  }

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = op.getSource().getType().cast<BaseMemRefType>();
    auto targetType = op.getTarget().getType().cast<BaseMemRefType>();

    auto isStaticShapeAndContiguousRowMajor = [](MemRefType type) {
      if (!type.hasStaticShape())
        return false;

      SmallVector<int64_t> strides;
      int64_t offset;
      if (failed(getStridesAndOffset(type, strides, offset)))
        return false;

      int64_t runningStride = 1;
      for (unsigned i = strides.size(); i > 0; --i) {
        if (strides[i - 1] != runningStride)
          return false;
        runningStride *= type.getDimSize(i - 1);
      }
      return true;
    };

    auto isContiguousMemrefType = [&](BaseMemRefType type) {
      auto memrefType = type.dyn_cast<mlir::MemRefType>();
      // We can use memcpy for memrefs if they have an identity layout or are
      // contiguous with an arbitrary offset. Ignore empty memrefs, which is a
      // special case handled by memrefCopy.
      return memrefType &&
             (memrefType.getLayout().isIdentity() ||
              (memrefType.hasStaticShape() && memrefType.getNumElements() > 0 &&
               isStaticShapeAndContiguousRowMajor(memrefType)));
    };

    if (isContiguousMemrefType(srcType) && isContiguousMemrefType(targetType))
      return lowerToMemCopyIntrinsic(op, adaptor, rewriter);

    return lowerToMemCopyFunctionCall(op, adaptor, rewriter);
  }
};

struct MemorySpaceCastOpLowering
    : public ConvertOpToLLVMPattern<memref::MemorySpaceCastOp> {
  using ConvertOpToLLVMPattern<
      memref::MemorySpaceCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::MemorySpaceCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type resultType = op.getDest().getType();
    if (auto resultTypeR = resultType.dyn_cast<MemRefType>()) {
      auto resultDescType =
          typeConverter->convertType(resultTypeR).cast<LLVM::LLVMStructType>();
      Type newPtrType = resultDescType.getBody()[0];

      SmallVector<Value> descVals;
      MemRefDescriptor::unpack(rewriter, loc, adaptor.getSource(), resultTypeR,
                               descVals);
      descVals[0] =
          rewriter.create<LLVM::AddrSpaceCastOp>(loc, newPtrType, descVals[0]);
      descVals[1] =
          rewriter.create<LLVM::AddrSpaceCastOp>(loc, newPtrType, descVals[1]);
      Value result = MemRefDescriptor::pack(rewriter, loc, *getTypeConverter(),
                                            resultTypeR, descVals);
      rewriter.replaceOp(op, result);
      return success();
    }
    if (auto resultTypeU = resultType.dyn_cast<UnrankedMemRefType>()) {
      // Since the type converter won't be doing this for us, get the address
      // space.
      auto sourceType = op.getSource().getType().cast<UnrankedMemRefType>();
      FailureOr<unsigned> maybeSourceAddrSpace =
          getTypeConverter()->getMemRefAddressSpace(sourceType);
      if (failed(maybeSourceAddrSpace))
        return rewriter.notifyMatchFailure(loc,
                                           "non-integer source address space");
      unsigned sourceAddrSpace = *maybeSourceAddrSpace;
      FailureOr<unsigned> maybeResultAddrSpace =
          getTypeConverter()->getMemRefAddressSpace(resultTypeU);
      if (failed(maybeResultAddrSpace))
        return rewriter.notifyMatchFailure(loc,
                                           "non-integer result address space");
      unsigned resultAddrSpace = *maybeResultAddrSpace;

      UnrankedMemRefDescriptor sourceDesc(adaptor.getSource());
      Value rank = sourceDesc.rank(rewriter, loc);
      Value sourceUnderlyingDesc = sourceDesc.memRefDescPtr(rewriter, loc);

      // Create and allocate storage for new memref descriptor.
      auto result = UnrankedMemRefDescriptor::undef(
          rewriter, loc, typeConverter->convertType(resultTypeU));
      result.setRank(rewriter, loc, rank);
      SmallVector<Value, 1> sizes;
      UnrankedMemRefDescriptor::computeSizes(rewriter, loc, *getTypeConverter(),
                                             result, resultAddrSpace, sizes);
      Value resultUnderlyingSize = sizes.front();
      Value resultUnderlyingDesc = rewriter.create<LLVM::AllocaOp>(
          loc, getVoidPtrType(), rewriter.getI8Type(), resultUnderlyingSize);
      result.setMemRefDescPtr(rewriter, loc, resultUnderlyingDesc);

      // Copy pointers, performing address space casts.
      Type llvmElementType =
          typeConverter->convertType(sourceType.getElementType());
      LLVM::LLVMPointerType sourceElemPtrType =
          getTypeConverter()->getPointerType(llvmElementType, sourceAddrSpace);
      auto resultElemPtrType =
          getTypeConverter()->getPointerType(llvmElementType, resultAddrSpace);

      Value allocatedPtr = sourceDesc.allocatedPtr(
          rewriter, loc, sourceUnderlyingDesc, sourceElemPtrType);
      Value alignedPtr =
          sourceDesc.alignedPtr(rewriter, loc, *getTypeConverter(),
                                sourceUnderlyingDesc, sourceElemPtrType);
      allocatedPtr = rewriter.create<LLVM::AddrSpaceCastOp>(
          loc, resultElemPtrType, allocatedPtr);
      alignedPtr = rewriter.create<LLVM::AddrSpaceCastOp>(
          loc, resultElemPtrType, alignedPtr);

      result.setAllocatedPtr(rewriter, loc, resultUnderlyingDesc,
                             resultElemPtrType, allocatedPtr);
      result.setAlignedPtr(rewriter, loc, *getTypeConverter(),
                           resultUnderlyingDesc, resultElemPtrType, alignedPtr);

      // Copy all the index-valued operands.
      Value sourceIndexVals =
          sourceDesc.offsetBasePtr(rewriter, loc, *getTypeConverter(),
                                   sourceUnderlyingDesc, sourceElemPtrType);
      Value resultIndexVals =
          result.offsetBasePtr(rewriter, loc, *getTypeConverter(),
                               resultUnderlyingDesc, resultElemPtrType);

      int64_t bytesToSkip =
          2 *
          ceilDiv(getTypeConverter()->getPointerBitwidth(resultAddrSpace), 8);
      Value bytesToSkipConst = rewriter.create<LLVM::ConstantOp>(
          loc, getIndexType(), rewriter.getIndexAttr(bytesToSkip));
      Value copySize = rewriter.create<LLVM::SubOp>(
          loc, getIndexType(), resultUnderlyingSize, bytesToSkipConst);
      Type llvmBool = typeConverter->convertType(rewriter.getI1Type());
      Value nonVolatile = rewriter.create<LLVM::ConstantOp>(
          loc, llvmBool, rewriter.getBoolAttr(false));
      rewriter.create<LLVM::MemcpyOp>(loc, resultIndexVals, sourceIndexVals,
                                      copySize, nonVolatile);

      rewriter.replaceOp(op, ValueRange{result});
      return success();
    }
    return rewriter.notifyMatchFailure(loc, "unexpected memref type");
  }
};

/// Extracts allocated, aligned pointers and offset from a ranked or unranked
/// memref type. In unranked case, the fields are extracted from the underlying
/// ranked descriptor.
static void extractPointersAndOffset(Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     LLVMTypeConverter &typeConverter,
                                     Value originalOperand,
                                     Value convertedOperand,
                                     Value *allocatedPtr, Value *alignedPtr,
                                     Value *offset = nullptr) {
  Type operandType = originalOperand.getType();
  if (operandType.isa<MemRefType>()) {
    MemRefDescriptor desc(convertedOperand);
    *allocatedPtr = desc.allocatedPtr(rewriter, loc);
    *alignedPtr = desc.alignedPtr(rewriter, loc);
    if (offset != nullptr)
      *offset = desc.offset(rewriter, loc);
    return;
  }

  // These will all cause assert()s on unconvertible types.
  unsigned memorySpace = *typeConverter.getMemRefAddressSpace(
      operandType.cast<UnrankedMemRefType>());
  Type elementType = operandType.cast<UnrankedMemRefType>().getElementType();
  Type llvmElementType = typeConverter.convertType(elementType);
  LLVM::LLVMPointerType elementPtrType =
      typeConverter.getPointerType(llvmElementType, memorySpace);

  // Extract pointer to the underlying ranked memref descriptor and cast it to
  // ElemType**.
  UnrankedMemRefDescriptor unrankedDesc(convertedOperand);
  Value underlyingDescPtr = unrankedDesc.memRefDescPtr(rewriter, loc);

  *allocatedPtr = UnrankedMemRefDescriptor::allocatedPtr(
      rewriter, loc, underlyingDescPtr, elementPtrType);
  *alignedPtr = UnrankedMemRefDescriptor::alignedPtr(
      rewriter, loc, typeConverter, underlyingDescPtr, elementPtrType);
  if (offset != nullptr) {
    *offset = UnrankedMemRefDescriptor::offset(
        rewriter, loc, typeConverter, underlyingDescPtr, elementPtrType);
  }
}

struct MemRefReinterpretCastOpLowering
    : public ConvertOpToLLVMPattern<memref::ReinterpretCastOp> {
  using ConvertOpToLLVMPattern<
      memref::ReinterpretCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type srcType = castOp.getSource().getType();

    Value descriptor;
    if (failed(convertSourceMemRefToDescriptor(rewriter, srcType, castOp,
                                               adaptor, &descriptor)))
      return failure();
    rewriter.replaceOp(castOp, {descriptor});
    return success();
  }

private:
  LogicalResult convertSourceMemRefToDescriptor(
      ConversionPatternRewriter &rewriter, Type srcType,
      memref::ReinterpretCastOp castOp,
      memref::ReinterpretCastOp::Adaptor adaptor, Value *descriptor) const {
    MemRefType targetMemRefType =
        castOp.getResult().getType().cast<MemRefType>();
    auto llvmTargetDescriptorTy = typeConverter->convertType(targetMemRefType)
                                      .dyn_cast_or_null<LLVM::LLVMStructType>();
    if (!llvmTargetDescriptorTy)
      return failure();

    // Create descriptor.
    Location loc = castOp.getLoc();
    auto desc = MemRefDescriptor::undef(rewriter, loc, llvmTargetDescriptorTy);

    // Set allocated and aligned pointers.
    Value allocatedPtr, alignedPtr;
    extractPointersAndOffset(loc, rewriter, *getTypeConverter(),
                             castOp.getSource(), adaptor.getSource(),
                             &allocatedPtr, &alignedPtr);
    desc.setAllocatedPtr(rewriter, loc, allocatedPtr);
    desc.setAlignedPtr(rewriter, loc, alignedPtr);

    // Set offset.
    if (castOp.isDynamicOffset(0))
      desc.setOffset(rewriter, loc, adaptor.getOffsets()[0]);
    else
      desc.setConstantOffset(rewriter, loc, castOp.getStaticOffset(0));

    // Set sizes and strides.
    unsigned dynSizeId = 0;
    unsigned dynStrideId = 0;
    for (unsigned i = 0, e = targetMemRefType.getRank(); i < e; ++i) {
      if (castOp.isDynamicSize(i))
        desc.setSize(rewriter, loc, i, adaptor.getSizes()[dynSizeId++]);
      else
        desc.setConstantSize(rewriter, loc, i, castOp.getStaticSize(i));

      if (castOp.isDynamicStride(i))
        desc.setStride(rewriter, loc, i, adaptor.getStrides()[dynStrideId++]);
      else
        desc.setConstantStride(rewriter, loc, i, castOp.getStaticStride(i));
    }
    *descriptor = desc;
    return success();
  }
};

struct MemRefReshapeOpLowering
    : public ConvertOpToLLVMPattern<memref::ReshapeOp> {
  using ConvertOpToLLVMPattern<memref::ReshapeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::ReshapeOp reshapeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type srcType = reshapeOp.getSource().getType();

    Value descriptor;
    if (failed(convertSourceMemRefToDescriptor(rewriter, srcType, reshapeOp,
                                               adaptor, &descriptor)))
      return failure();
    rewriter.replaceOp(reshapeOp, {descriptor});
    return success();
  }

private:
  LogicalResult
  convertSourceMemRefToDescriptor(ConversionPatternRewriter &rewriter,
                                  Type srcType, memref::ReshapeOp reshapeOp,
                                  memref::ReshapeOp::Adaptor adaptor,
                                  Value *descriptor) const {
    auto shapeMemRefType = reshapeOp.getShape().getType().cast<MemRefType>();
    if (shapeMemRefType.hasStaticShape()) {
      MemRefType targetMemRefType =
          reshapeOp.getResult().getType().cast<MemRefType>();
      auto llvmTargetDescriptorTy =
          typeConverter->convertType(targetMemRefType)
              .dyn_cast_or_null<LLVM::LLVMStructType>();
      if (!llvmTargetDescriptorTy)
        return failure();

      // Create descriptor.
      Location loc = reshapeOp.getLoc();
      auto desc =
          MemRefDescriptor::undef(rewriter, loc, llvmTargetDescriptorTy);

      // Set allocated and aligned pointers.
      Value allocatedPtr, alignedPtr;
      extractPointersAndOffset(loc, rewriter, *getTypeConverter(),
                               reshapeOp.getSource(), adaptor.getSource(),
                               &allocatedPtr, &alignedPtr);
      desc.setAllocatedPtr(rewriter, loc, allocatedPtr);
      desc.setAlignedPtr(rewriter, loc, alignedPtr);

      // Extract the offset and strides from the type.
      int64_t offset;
      SmallVector<int64_t> strides;
      if (failed(getStridesAndOffset(targetMemRefType, strides, offset)))
        return rewriter.notifyMatchFailure(
            reshapeOp, "failed to get stride and offset exprs");

      if (!isStaticStrideOrOffset(offset))
        return rewriter.notifyMatchFailure(reshapeOp,
                                           "dynamic offset is unsupported");

      desc.setConstantOffset(rewriter, loc, offset);

      assert(targetMemRefType.getLayout().isIdentity() &&
             "Identity layout map is a precondition of a valid reshape op");

      Value stride = nullptr;
      int64_t targetRank = targetMemRefType.getRank();
      for (auto i : llvm::reverse(llvm::seq<int64_t>(0, targetRank))) {
        if (!ShapedType::isDynamic(strides[i])) {
          // If the stride for this dimension is dynamic, then use the product
          // of the sizes of the inner dimensions.
          stride = createIndexConstant(rewriter, loc, strides[i]);
        } else if (!stride) {
          // `stride` is null only in the first iteration of the loop.  However,
          // since the target memref has an identity layout, we can safely set
          // the innermost stride to 1.
          stride = createIndexConstant(rewriter, loc, 1);
        }

        Value dimSize;
        int64_t size = targetMemRefType.getDimSize(i);
        // If the size of this dimension is dynamic, then load it at runtime
        // from the shape operand.
        if (!ShapedType::isDynamic(size)) {
          dimSize = createIndexConstant(rewriter, loc, size);
        } else {
          Value shapeOp = reshapeOp.getShape();
          Value index = createIndexConstant(rewriter, loc, i);
          dimSize = rewriter.create<memref::LoadOp>(loc, shapeOp, index);
          Type indexType = getIndexType();
          if (dimSize.getType() != indexType)
            dimSize = typeConverter->materializeTargetConversion(
                rewriter, loc, indexType, dimSize);
          assert(dimSize && "Invalid memref element type");
        }

        desc.setSize(rewriter, loc, i, dimSize);
        desc.setStride(rewriter, loc, i, stride);

        // Prepare the stride value for the next dimension.
        stride = rewriter.create<LLVM::MulOp>(loc, stride, dimSize);
      }

      *descriptor = desc;
      return success();
    }

    // The shape is a rank-1 tensor with unknown length.
    Location loc = reshapeOp.getLoc();
    MemRefDescriptor shapeDesc(adaptor.getShape());
    Value resultRank = shapeDesc.size(rewriter, loc, 0);

    // Extract address space and element type.
    auto targetType =
        reshapeOp.getResult().getType().cast<UnrankedMemRefType>();
    unsigned addressSpace =
        *getTypeConverter()->getMemRefAddressSpace(targetType);
    Type elementType = targetType.getElementType();

    // Create the unranked memref descriptor that holds the ranked one. The
    // inner descriptor is allocated on stack.
    auto targetDesc = UnrankedMemRefDescriptor::undef(
        rewriter, loc, typeConverter->convertType(targetType));
    targetDesc.setRank(rewriter, loc, resultRank);
    SmallVector<Value, 4> sizes;
    UnrankedMemRefDescriptor::computeSizes(rewriter, loc, *getTypeConverter(),
                                           targetDesc, addressSpace, sizes);
    Value underlyingDescPtr = rewriter.create<LLVM::AllocaOp>(
        loc, getVoidPtrType(), IntegerType::get(getContext(), 8),
        sizes.front());
    targetDesc.setMemRefDescPtr(rewriter, loc, underlyingDescPtr);

    // Extract pointers and offset from the source memref.
    Value allocatedPtr, alignedPtr, offset;
    extractPointersAndOffset(loc, rewriter, *getTypeConverter(),
                             reshapeOp.getSource(), adaptor.getSource(),
                             &allocatedPtr, &alignedPtr, &offset);

    // Set pointers and offset.
    Type llvmElementType = typeConverter->convertType(elementType);
    LLVM::LLVMPointerType elementPtrType =
        getTypeConverter()->getPointerType(llvmElementType, addressSpace);

    UnrankedMemRefDescriptor::setAllocatedPtr(rewriter, loc, underlyingDescPtr,
                                              elementPtrType, allocatedPtr);
    UnrankedMemRefDescriptor::setAlignedPtr(rewriter, loc, *getTypeConverter(),
                                            underlyingDescPtr, elementPtrType,
                                            alignedPtr);
    UnrankedMemRefDescriptor::setOffset(rewriter, loc, *getTypeConverter(),
                                        underlyingDescPtr, elementPtrType,
                                        offset);

    // Use the offset pointer as base for further addressing. Copy over the new
    // shape and compute strides. For this, we create a loop from rank-1 to 0.
    Value targetSizesBase = UnrankedMemRefDescriptor::sizeBasePtr(
        rewriter, loc, *getTypeConverter(), underlyingDescPtr, elementPtrType);
    Value targetStridesBase = UnrankedMemRefDescriptor::strideBasePtr(
        rewriter, loc, *getTypeConverter(), targetSizesBase, resultRank);
    Value shapeOperandPtr = shapeDesc.alignedPtr(rewriter, loc);
    Value oneIndex = createIndexConstant(rewriter, loc, 1);
    Value resultRankMinusOne =
        rewriter.create<LLVM::SubOp>(loc, resultRank, oneIndex);

    Block *initBlock = rewriter.getInsertionBlock();
    Type indexType = getTypeConverter()->getIndexType();
    Block::iterator remainingOpsIt = std::next(rewriter.getInsertionPoint());

    Block *condBlock = rewriter.createBlock(initBlock->getParent(), {},
                                            {indexType, indexType}, {loc, loc});

    // Move the remaining initBlock ops to condBlock.
    Block *remainingBlock = rewriter.splitBlock(initBlock, remainingOpsIt);
    rewriter.mergeBlocks(remainingBlock, condBlock, ValueRange());

    rewriter.setInsertionPointToEnd(initBlock);
    rewriter.create<LLVM::BrOp>(loc, ValueRange({resultRankMinusOne, oneIndex}),
                                condBlock);
    rewriter.setInsertionPointToStart(condBlock);
    Value indexArg = condBlock->getArgument(0);
    Value strideArg = condBlock->getArgument(1);

    Value zeroIndex = createIndexConstant(rewriter, loc, 0);
    Value pred = rewriter.create<LLVM::ICmpOp>(
        loc, IntegerType::get(rewriter.getContext(), 1),
        LLVM::ICmpPredicate::sge, indexArg, zeroIndex);

    Block *bodyBlock =
        rewriter.splitBlock(condBlock, rewriter.getInsertionPoint());
    rewriter.setInsertionPointToStart(bodyBlock);

    // Copy size from shape to descriptor.
    Type llvmIndexPtrType = getTypeConverter()->getPointerType(indexType);
    Value sizeLoadGep = rewriter.create<LLVM::GEPOp>(
        loc, llvmIndexPtrType,
        typeConverter->convertType(shapeMemRefType.getElementType()),
        shapeOperandPtr, indexArg);
    Value size = rewriter.create<LLVM::LoadOp>(loc, indexType, sizeLoadGep);
    UnrankedMemRefDescriptor::setSize(rewriter, loc, *getTypeConverter(),
                                      targetSizesBase, indexArg, size);

    // Write stride value and compute next one.
    UnrankedMemRefDescriptor::setStride(rewriter, loc, *getTypeConverter(),
                                        targetStridesBase, indexArg, strideArg);
    Value nextStride = rewriter.create<LLVM::MulOp>(loc, strideArg, size);

    // Decrement loop counter and branch back.
    Value decrement = rewriter.create<LLVM::SubOp>(loc, indexArg, oneIndex);
    rewriter.create<LLVM::BrOp>(loc, ValueRange({decrement, nextStride}),
                                condBlock);

    Block *remainder =
        rewriter.splitBlock(bodyBlock, rewriter.getInsertionPoint());

    // Hook up the cond exit to the remainder.
    rewriter.setInsertionPointToEnd(condBlock);
    rewriter.create<LLVM::CondBrOp>(loc, pred, bodyBlock, std::nullopt,
                                    remainder, std::nullopt);

    // Reset position to beginning of new remainder block.
    rewriter.setInsertionPointToStart(remainder);

    *descriptor = targetDesc;
    return success();
  }
};

/// RessociatingReshapeOp must be expanded before we reach this stage.
/// Report that information.
template <typename ReshapeOp>
class ReassociatingReshapeOpConversion
    : public ConvertOpToLLVMPattern<ReshapeOp> {
public:
  using ConvertOpToLLVMPattern<ReshapeOp>::ConvertOpToLLVMPattern;
  using ReshapeOpAdaptor = typename ReshapeOp::Adaptor;

  LogicalResult
  matchAndRewrite(ReshapeOp reshapeOp, typename ReshapeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriter.notifyMatchFailure(
        reshapeOp,
        "reassociation operations should have been expanded beforehand");
  }
};

/// Subviews must be expanded before we reach this stage.
/// Report that information.
struct SubViewOpLowering : public ConvertOpToLLVMPattern<memref::SubViewOp> {
  using ConvertOpToLLVMPattern<memref::SubViewOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::SubViewOp subViewOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriter.notifyMatchFailure(
        subViewOp, "subview operations should have been expanded beforehand");
  }
};

/// Conversion pattern that transforms a transpose op into:
///   1. A function entry `alloca` operation to allocate a ViewDescriptor.
///   2. A load of the ViewDescriptor from the pointer allocated in 1.
///   3. Updates to the ViewDescriptor to introduce the data ptr, offset, size
///      and stride. Size and stride are permutations of the original values.
///   4. A store of the resulting ViewDescriptor to the alloca'ed pointer.
/// The transpose op is replaced by the alloca'ed pointer.
class TransposeOpLowering : public ConvertOpToLLVMPattern<memref::TransposeOp> {
public:
  using ConvertOpToLLVMPattern<memref::TransposeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::TransposeOp transposeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = transposeOp.getLoc();
    MemRefDescriptor viewMemRef(adaptor.getIn());

    // No permutation, early exit.
    if (transposeOp.getPermutation().isIdentity())
      return rewriter.replaceOp(transposeOp, {viewMemRef}), success();

    auto targetMemRef = MemRefDescriptor::undef(
        rewriter, loc, typeConverter->convertType(transposeOp.getShapedType()));

    // Copy the base and aligned pointers from the old descriptor to the new
    // one.
    targetMemRef.setAllocatedPtr(rewriter, loc,
                                 viewMemRef.allocatedPtr(rewriter, loc));
    targetMemRef.setAlignedPtr(rewriter, loc,
                               viewMemRef.alignedPtr(rewriter, loc));

    // Copy the offset pointer from the old descriptor to the new one.
    targetMemRef.setOffset(rewriter, loc, viewMemRef.offset(rewriter, loc));

    // Iterate over the dimensions and apply size/stride permutation.
    for (const auto &en :
         llvm::enumerate(transposeOp.getPermutation().getResults())) {
      int sourcePos = en.index();
      int targetPos = en.value().cast<AffineDimExpr>().getPosition();
      targetMemRef.setSize(rewriter, loc, targetPos,
                           viewMemRef.size(rewriter, loc, sourcePos));
      targetMemRef.setStride(rewriter, loc, targetPos,
                             viewMemRef.stride(rewriter, loc, sourcePos));
    }

    rewriter.replaceOp(transposeOp, {targetMemRef});
    return success();
  }
};

/// Conversion pattern that transforms an op into:
///   1. An `llvm.mlir.undef` operation to create a memref descriptor
///   2. Updates to the descriptor to introduce the data ptr, offset, size
///      and stride.
/// The view op is replaced by the descriptor.
struct ViewOpLowering : public ConvertOpToLLVMPattern<memref::ViewOp> {
  using ConvertOpToLLVMPattern<memref::ViewOp>::ConvertOpToLLVMPattern;

  // Build and return the value for the idx^th shape dimension, either by
  // returning the constant shape dimension or counting the proper dynamic size.
  Value getSize(ConversionPatternRewriter &rewriter, Location loc,
                ArrayRef<int64_t> shape, ValueRange dynamicSizes,
                unsigned idx) const {
    assert(idx < shape.size());
    if (!ShapedType::isDynamic(shape[idx]))
      return createIndexConstant(rewriter, loc, shape[idx]);
    // Count the number of dynamic dims in range [0, idx]
    unsigned nDynamic =
        llvm::count_if(shape.take_front(idx), ShapedType::isDynamic);
    return dynamicSizes[nDynamic];
  }

  // Build and return the idx^th stride, either by returning the constant stride
  // or by computing the dynamic stride from the current `runningStride` and
  // `nextSize`. The caller should keep a running stride and update it with the
  // result returned by this function.
  Value getStride(ConversionPatternRewriter &rewriter, Location loc,
                  ArrayRef<int64_t> strides, Value nextSize,
                  Value runningStride, unsigned idx) const {
    assert(idx < strides.size());
    if (!ShapedType::isDynamic(strides[idx]))
      return createIndexConstant(rewriter, loc, strides[idx]);
    if (nextSize)
      return runningStride
                 ? rewriter.create<LLVM::MulOp>(loc, runningStride, nextSize)
                 : nextSize;
    assert(!runningStride);
    return createIndexConstant(rewriter, loc, 1);
  }

  LogicalResult
  matchAndRewrite(memref::ViewOp viewOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = viewOp.getLoc();

    auto viewMemRefType = viewOp.getType();
    auto targetElementTy =
        typeConverter->convertType(viewMemRefType.getElementType());
    auto targetDescTy = typeConverter->convertType(viewMemRefType);
    if (!targetDescTy || !targetElementTy ||
        !LLVM::isCompatibleType(targetElementTy) ||
        !LLVM::isCompatibleType(targetDescTy))
      return viewOp.emitWarning("Target descriptor type not converted to LLVM"),
             failure();

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides = getStridesAndOffset(viewMemRefType, strides, offset);
    if (failed(successStrides))
      return viewOp.emitWarning("cannot cast to non-strided shape"), failure();
    assert(offset == 0 && "expected offset to be 0");

    // Target memref must be contiguous in memory (innermost stride is 1), or
    // empty (special case when at least one of the memref dimensions is 0).
    if (!strides.empty() && (strides.back() != 1 && strides.back() != 0))
      return viewOp.emitWarning("cannot cast to non-contiguous shape"),
             failure();

    // Create the descriptor.
    MemRefDescriptor sourceMemRef(adaptor.getSource());
    auto targetMemRef = MemRefDescriptor::undef(rewriter, loc, targetDescTy);

    // Field 1: Copy the allocated pointer, used for malloc/free.
    Value allocatedPtr = sourceMemRef.allocatedPtr(rewriter, loc);
    auto srcMemRefType = viewOp.getSource().getType().cast<MemRefType>();
    unsigned sourceMemorySpace =
        *getTypeConverter()->getMemRefAddressSpace(srcMemRefType);
    Value bitcastPtr;
    if (getTypeConverter()->useOpaquePointers())
      bitcastPtr = allocatedPtr;
    else
      bitcastPtr = rewriter.create<LLVM::BitcastOp>(
          loc, LLVM::LLVMPointerType::get(targetElementTy, sourceMemorySpace),
          allocatedPtr);

    targetMemRef.setAllocatedPtr(rewriter, loc, bitcastPtr);

    // Field 2: Copy the actual aligned pointer to payload.
    Value alignedPtr = sourceMemRef.alignedPtr(rewriter, loc);
    alignedPtr = rewriter.create<LLVM::GEPOp>(
        loc, alignedPtr.getType(),
        typeConverter->convertType(srcMemRefType.getElementType()), alignedPtr,
        adaptor.getByteShift());

    if (getTypeConverter()->useOpaquePointers()) {
      bitcastPtr = alignedPtr;
    } else {
      bitcastPtr = rewriter.create<LLVM::BitcastOp>(
          loc, LLVM::LLVMPointerType::get(targetElementTy, sourceMemorySpace),
          alignedPtr);
    }

    targetMemRef.setAlignedPtr(rewriter, loc, bitcastPtr);

    // Field 3: The offset in the resulting type must be 0. This is because of
    // the type change: an offset on srcType* may not be expressible as an
    // offset on dstType*.
    targetMemRef.setOffset(rewriter, loc,
                           createIndexConstant(rewriter, loc, offset));

    // Early exit for 0-D corner case.
    if (viewMemRefType.getRank() == 0)
      return rewriter.replaceOp(viewOp, {targetMemRef}), success();

    // Fields 4 and 5: Update sizes and strides.
    Value stride = nullptr, nextSize = nullptr;
    for (int i = viewMemRefType.getRank() - 1; i >= 0; --i) {
      // Update size.
      Value size = getSize(rewriter, loc, viewMemRefType.getShape(),
                           adaptor.getSizes(), i);
      targetMemRef.setSize(rewriter, loc, i, size);
      // Update stride.
      stride = getStride(rewriter, loc, strides, nextSize, stride, i);
      targetMemRef.setStride(rewriter, loc, i, stride);
      nextSize = size;
    }

    rewriter.replaceOp(viewOp, {targetMemRef});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AtomicRMWOpLowering
//===----------------------------------------------------------------------===//

/// Try to match the kind of a memref.atomic_rmw to determine whether to use a
/// lowering to llvm.atomicrmw or fallback to llvm.cmpxchg.
static std::optional<LLVM::AtomicBinOp>
matchSimpleAtomicOp(memref::AtomicRMWOp atomicOp) {
  switch (atomicOp.getKind()) {
  case arith::AtomicRMWKind::addf:
    return LLVM::AtomicBinOp::fadd;
  case arith::AtomicRMWKind::addi:
    return LLVM::AtomicBinOp::add;
  case arith::AtomicRMWKind::assign:
    return LLVM::AtomicBinOp::xchg;
  case arith::AtomicRMWKind::maxs:
    return LLVM::AtomicBinOp::max;
  case arith::AtomicRMWKind::maxu:
    return LLVM::AtomicBinOp::umax;
  case arith::AtomicRMWKind::mins:
    return LLVM::AtomicBinOp::min;
  case arith::AtomicRMWKind::minu:
    return LLVM::AtomicBinOp::umin;
  case arith::AtomicRMWKind::ori:
    return LLVM::AtomicBinOp::_or;
  case arith::AtomicRMWKind::andi:
    return LLVM::AtomicBinOp::_and;
  default:
    return std::nullopt;
  }
  llvm_unreachable("Invalid AtomicRMWKind");
}

struct AtomicRMWOpLowering : public LoadStoreOpLowering<memref::AtomicRMWOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(memref::AtomicRMWOp atomicOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(match(atomicOp)))
      return failure();
    auto maybeKind = matchSimpleAtomicOp(atomicOp);
    if (!maybeKind)
      return failure();
    auto memRefType = atomicOp.getMemRefType();
    auto dataPtr =
        getStridedElementPtr(atomicOp.getLoc(), memRefType, adaptor.getMemref(),
                             adaptor.getIndices(), rewriter);
    rewriter.replaceOpWithNewOp<LLVM::AtomicRMWOp>(
        atomicOp, *maybeKind, dataPtr, adaptor.getValue(),
        LLVM::AtomicOrdering::acq_rel);
    return success();
  }
};

/// Unpack the pointer returned by a memref.extract_aligned_pointer_as_index.
class ConvertExtractAlignedPointerAsIndex
    : public ConvertOpToLLVMPattern<memref::ExtractAlignedPointerAsIndexOp> {
public:
  using ConvertOpToLLVMPattern<
      memref::ExtractAlignedPointerAsIndexOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::ExtractAlignedPointerAsIndexOp extractOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefDescriptor desc(adaptor.getSource());
    rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(
        extractOp, getTypeConverter()->getIndexType(),
        desc.alignedPtr(rewriter, extractOp->getLoc()));
    return success();
  }
};

/// Materialize the MemRef descriptor represented by the results of
/// ExtractStridedMetadataOp.
class ExtractStridedMetadataOpLowering
    : public ConvertOpToLLVMPattern<memref::ExtractStridedMetadataOp> {
public:
  using ConvertOpToLLVMPattern<
      memref::ExtractStridedMetadataOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::ExtractStridedMetadataOp extractStridedMetadataOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!LLVM::isCompatibleType(adaptor.getOperands().front().getType()))
      return failure();

    // Create the descriptor.
    MemRefDescriptor sourceMemRef(adaptor.getSource());
    Location loc = extractStridedMetadataOp.getLoc();
    Value source = extractStridedMetadataOp.getSource();

    auto sourceMemRefType = source.getType().cast<MemRefType>();
    int64_t rank = sourceMemRefType.getRank();
    SmallVector<Value> results;
    results.reserve(2 + rank * 2);

    // Base buffer.
    Value baseBuffer = sourceMemRef.allocatedPtr(rewriter, loc);
    Value alignedBuffer = sourceMemRef.alignedPtr(rewriter, loc);
    MemRefDescriptor dstMemRef = MemRefDescriptor::fromStaticShape(
        rewriter, loc, *getTypeConverter(),
        extractStridedMetadataOp.getBaseBuffer().getType().cast<MemRefType>(),
        baseBuffer, alignedBuffer);
    results.push_back((Value)dstMemRef);

    // Offset.
    results.push_back(sourceMemRef.offset(rewriter, loc));

    // Sizes.
    for (unsigned i = 0; i < rank; ++i)
      results.push_back(sourceMemRef.size(rewriter, loc, i));
    // Strides.
    for (unsigned i = 0; i < rank; ++i)
      results.push_back(sourceMemRef.stride(rewriter, loc, i));

    rewriter.replaceOp(extractStridedMetadataOp, results);
    return success();
  }
};

} // namespace

void mlir::populateFinalizeMemRefToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
      AllocaOpLowering,
      AllocaScopeOpLowering,
      AtomicRMWOpLowering,
      AssumeAlignmentOpLowering,
      ConvertExtractAlignedPointerAsIndex,
      DimOpLowering,
      ExtractStridedMetadataOpLowering,
      GenericAtomicRMWOpLowering,
      GlobalMemrefOpLowering,
      GetGlobalMemrefOpLowering,
      LoadOpLowering,
      MemRefCastOpLowering,
      MemRefCopyOpLowering,
      MemorySpaceCastOpLowering,
      MemRefReinterpretCastOpLowering,
      MemRefReshapeOpLowering,
      PrefetchOpLowering,
      RankOpLowering,
      ReassociatingReshapeOpConversion<memref::ExpandShapeOp>,
      ReassociatingReshapeOpConversion<memref::CollapseShapeOp>,
      StoreOpLowering,
      SubViewOpLowering,
      TransposeOpLowering,
      ViewOpLowering>(converter);
  // clang-format on
  auto allocLowering = converter.getOptions().allocLowering;
  if (allocLowering == LowerToLLVMOptions::AllocLowering::AlignedAlloc)
    patterns.add<AlignedAllocOpLowering, AlignedReallocOpLowering,
                 DeallocOpLowering>(converter);
  else if (allocLowering == LowerToLLVMOptions::AllocLowering::Malloc)
    patterns.add<AllocOpLowering, ReallocOpLowering, DeallocOpLowering>(
        converter);
}

namespace {
struct FinalizeMemRefToLLVMConversionPass
    : public impl::FinalizeMemRefToLLVMConversionPassBase<
          FinalizeMemRefToLLVMConversionPass> {
  using FinalizeMemRefToLLVMConversionPassBase::
      FinalizeMemRefToLLVMConversionPassBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
    LowerToLLVMOptions options(&getContext(),
                               dataLayoutAnalysis.getAtOrAbove(op));
    options.allocLowering =
        (useAlignedAlloc ? LowerToLLVMOptions::AllocLowering::AlignedAlloc
                         : LowerToLLVMOptions::AllocLowering::Malloc);

    options.useGenericFunctions = useGenericFunctions;
    options.useOpaquePointers = useOpaquePointers;

    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    LLVMTypeConverter typeConverter(&getContext(), options,
                                    &dataLayoutAnalysis);
    RewritePatternSet patterns(&getContext());
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    LLVMConversionTarget target(getContext());
    target.addLegalOp<func::FuncOp>();
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
