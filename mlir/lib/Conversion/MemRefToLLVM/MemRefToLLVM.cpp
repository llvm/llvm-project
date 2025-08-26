//===- MemRefToLLVM.cpp - MemRef to LLVM dialect conversion ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/MathExtras.h"

#include <optional>

#define DEBUG_TYPE "memref-to-llvm"

namespace mlir {
#define GEN_PASS_DEF_FINALIZEMEMREFTOLLVMCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

static constexpr LLVM::GEPNoWrapFlags kNoWrapFlags =
    LLVM::GEPNoWrapFlags::inbounds | LLVM::GEPNoWrapFlags::nuw;

namespace {

static bool isStaticStrideOrOffset(int64_t strideOrOffset) {
  return ShapedType::isStatic(strideOrOffset);
}

static FailureOr<LLVM::LLVMFuncOp>
getFreeFn(OpBuilder &b, const LLVMTypeConverter *typeConverter, ModuleOp module,
          SymbolTableCollection *symbolTables) {
  bool useGenericFn = typeConverter->getOptions().useGenericFunctions;

  if (useGenericFn)
    return LLVM::lookupOrCreateGenericFreeFn(b, module, symbolTables);

  return LLVM::lookupOrCreateFreeFn(b, module, symbolTables);
}

static FailureOr<LLVM::LLVMFuncOp>
getNotalignedAllocFn(OpBuilder &b, const LLVMTypeConverter *typeConverter,
                     Operation *module, Type indexType,
                     SymbolTableCollection *symbolTables) {
  bool useGenericFn = typeConverter->getOptions().useGenericFunctions;
  if (useGenericFn)
    return LLVM::lookupOrCreateGenericAllocFn(b, module, indexType,
                                              symbolTables);

  return LLVM::lookupOrCreateMallocFn(b, module, indexType, symbolTables);
}

static FailureOr<LLVM::LLVMFuncOp>
getAlignedAllocFn(OpBuilder &b, const LLVMTypeConverter *typeConverter,
                  Operation *module, Type indexType,
                  SymbolTableCollection *symbolTables) {
  bool useGenericFn = typeConverter->getOptions().useGenericFunctions;

  if (useGenericFn)
    return LLVM::lookupOrCreateGenericAlignedAllocFn(b, module, indexType,
                                                     symbolTables);

  return LLVM::lookupOrCreateAlignedAllocFn(b, module, indexType, symbolTables);
}

/// Computes the aligned value for 'input' as follows:
///   bumped = input + alignement - 1
///   aligned = bumped - bumped % alignment
static Value createAligned(ConversionPatternRewriter &rewriter, Location loc,
                           Value input, Value alignment) {
  Value one = LLVM::ConstantOp::create(rewriter, loc, alignment.getType(),
                                       rewriter.getIndexAttr(1));
  Value bump = LLVM::SubOp::create(rewriter, loc, alignment, one);
  Value bumped = LLVM::AddOp::create(rewriter, loc, input, bump);
  Value mod = LLVM::URemOp::create(rewriter, loc, bumped, alignment);
  return LLVM::SubOp::create(rewriter, loc, bumped, mod);
}

/// Computes the byte size for the MemRef element type.
static unsigned getMemRefEltSizeInBytes(const LLVMTypeConverter *typeConverter,
                                        MemRefType memRefType, Operation *op,
                                        const DataLayout *defaultLayout) {
  const DataLayout *layout = defaultLayout;
  if (const DataLayoutAnalysis *analysis =
          typeConverter->getDataLayoutAnalysis()) {
    layout = &analysis->getAbove(op);
  }
  Type elementType = memRefType.getElementType();
  if (auto memRefElementType = dyn_cast<MemRefType>(elementType))
    return typeConverter->getMemRefDescriptorSize(memRefElementType, *layout);
  if (auto memRefElementType = dyn_cast<UnrankedMemRefType>(elementType))
    return typeConverter->getUnrankedMemRefDescriptorSize(memRefElementType,
                                                          *layout);
  return layout->getTypeSize(elementType);
}

static Value castAllocFuncResult(ConversionPatternRewriter &rewriter,
                                 Location loc, Value allocatedPtr,
                                 MemRefType memRefType, Type elementPtrType,
                                 const LLVMTypeConverter &typeConverter) {
  auto allocatedPtrTy = cast<LLVM::LLVMPointerType>(allocatedPtr.getType());
  FailureOr<unsigned> maybeMemrefAddrSpace =
      typeConverter.getMemRefAddressSpace(memRefType);
  assert(succeeded(maybeMemrefAddrSpace) && "unsupported address space");
  unsigned memrefAddrSpace = *maybeMemrefAddrSpace;
  if (allocatedPtrTy.getAddressSpace() != memrefAddrSpace)
    allocatedPtr = LLVM::AddrSpaceCastOp::create(
        rewriter, loc,
        LLVM::LLVMPointerType::get(rewriter.getContext(), memrefAddrSpace),
        allocatedPtr);
  return allocatedPtr;
}

class AllocOpLowering : public ConvertOpToLLVMPattern<memref::AllocOp> {
  SymbolTableCollection *symbolTables = nullptr;

public:
  explicit AllocOpLowering(const LLVMTypeConverter &typeConverter,
                           SymbolTableCollection *symbolTables = nullptr,
                           PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<memref::AllocOp>(typeConverter, benefit),
        symbolTables(symbolTables) {}

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MemRefType memRefType = op.getType();
    if (!isConvertibleAndHasIdentityMaps(memRefType))
      return rewriter.notifyMatchFailure(op, "incompatible memref type");

    // Get or insert alloc function into the module.
    FailureOr<LLVM::LLVMFuncOp> allocFuncOp =
        getNotalignedAllocFn(rewriter, getTypeConverter(),
                             op->getParentWithTrait<OpTrait::SymbolTable>(),
                             getIndexType(), symbolTables);
    if (failed(allocFuncOp))
      return failure();

    // Get actual sizes of the memref as values: static sizes are constant
    // values and dynamic sizes are passed to 'alloc' as operands.  In case of
    // zero-dimensional memref, assume a scalar (size 1).
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value sizeBytes;

    this->getMemRefDescriptorSizes(loc, memRefType, adaptor.getOperands(),
                                   rewriter, sizes, strides, sizeBytes, true);

    Value alignment = getAlignment(rewriter, loc, op);
    if (alignment) {
      // Adjust the allocation size to consider alignment.
      sizeBytes = LLVM::AddOp::create(rewriter, loc, sizeBytes, alignment);
    }

    // Allocate the underlying buffer.
    Type elementPtrType = this->getElementPtrType(memRefType);
    assert(elementPtrType && "could not compute element ptr type");
    auto results =
        LLVM::CallOp::create(rewriter, loc, allocFuncOp.value(), sizeBytes);

    Value allocatedPtr =
        castAllocFuncResult(rewriter, loc, results.getResult(), memRefType,
                            elementPtrType, *getTypeConverter());
    Value alignedPtr = allocatedPtr;
    if (alignment) {
      // Compute the aligned pointer.
      Value allocatedInt =
          LLVM::PtrToIntOp::create(rewriter, loc, getIndexType(), allocatedPtr);
      Value alignmentInt =
          createAligned(rewriter, loc, allocatedInt, alignment);
      alignedPtr =
          LLVM::IntToPtrOp::create(rewriter, loc, elementPtrType, alignmentInt);
    }

    // Create the MemRef descriptor.
    auto memRefDescriptor = this->createMemRefDescriptor(
        loc, memRefType, allocatedPtr, alignedPtr, sizes, strides, rewriter);

    // Return the final value of the descriptor.
    rewriter.replaceOp(op, {memRefDescriptor});
    return success();
  }

  /// Computes the alignment for the given memory allocation op.
  template <typename OpType>
  Value getAlignment(ConversionPatternRewriter &rewriter, Location loc,
                     OpType op) const {
    MemRefType memRefType = op.getType();
    Value alignment;
    if (auto alignmentAttr = op.getAlignment()) {
      Type indexType = getIndexType();
      alignment =
          createIndexAttrConstant(rewriter, loc, indexType, *alignmentAttr);
    } else if (!memRefType.getElementType().isSignlessIntOrIndexOrFloat()) {
      // In the case where no alignment is specified, we may want to override
      // `malloc's` behavior. `malloc` typically aligns at the size of the
      // biggest scalar on a target HW. For non-scalars, use the natural
      // alignment of the LLVM type given by the LLVM DataLayout.
      alignment = getSizeInBytes(loc, memRefType.getElementType(), rewriter);
    }
    return alignment;
  }
};

class AlignedAllocOpLowering : public ConvertOpToLLVMPattern<memref::AllocOp> {
  SymbolTableCollection *symbolTables = nullptr;

public:
  explicit AlignedAllocOpLowering(const LLVMTypeConverter &typeConverter,
                                  SymbolTableCollection *symbolTables = nullptr,
                                  PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<memref::AllocOp>(typeConverter, benefit),
        symbolTables(symbolTables) {}

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MemRefType memRefType = op.getType();
    if (!isConvertibleAndHasIdentityMaps(memRefType))
      return rewriter.notifyMatchFailure(op, "incompatible memref type");

    // Get or insert alloc function into module.
    FailureOr<LLVM::LLVMFuncOp> allocFuncOp =
        getAlignedAllocFn(rewriter, getTypeConverter(),
                          op->getParentWithTrait<OpTrait::SymbolTable>(),
                          getIndexType(), symbolTables);
    if (failed(allocFuncOp))
      return failure();

    // Get actual sizes of the memref as values: static sizes are constant
    // values and dynamic sizes are passed to 'alloc' as operands.  In case of
    // zero-dimensional memref, assume a scalar (size 1).
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value sizeBytes;

    this->getMemRefDescriptorSizes(loc, memRefType, adaptor.getOperands(),
                                   rewriter, sizes, strides, sizeBytes, !false);

    int64_t alignment = alignedAllocationGetAlignment(op, &defaultLayout);

    Value allocAlignment =
        createIndexAttrConstant(rewriter, loc, getIndexType(), alignment);

    // Function aligned_alloc requires size to be a multiple of alignment; we
    // pad the size to the next multiple if necessary.
    if (!isMemRefSizeMultipleOf(memRefType, alignment, op, &defaultLayout))
      sizeBytes = createAligned(rewriter, loc, sizeBytes, allocAlignment);

    Type elementPtrType = this->getElementPtrType(memRefType);
    auto results =
        LLVM::CallOp::create(rewriter, loc, allocFuncOp.value(),
                             ValueRange({allocAlignment, sizeBytes}));

    Value ptr =
        castAllocFuncResult(rewriter, loc, results.getResult(), memRefType,
                            elementPtrType, *getTypeConverter());

    // Create the MemRef descriptor.
    auto memRefDescriptor = this->createMemRefDescriptor(
        loc, memRefType, ptr, ptr, sizes, strides, rewriter);

    // Return the final value of the descriptor.
    rewriter.replaceOp(op, {memRefDescriptor});
    return success();
  }

  /// The minimum alignment to use with aligned_alloc (has to be a power of 2).
  static constexpr uint64_t kMinAlignedAllocAlignment = 16UL;

  /// Computes the alignment for aligned_alloc used to allocate the buffer for
  /// the memory allocation op.
  ///
  /// Aligned_alloc requires the allocation size to be a power of two, and the
  /// allocation size to be a multiple of the alignment.
  int64_t alignedAllocationGetAlignment(memref::AllocOp op,
                                        const DataLayout *defaultLayout) const {
    if (std::optional<uint64_t> alignment = op.getAlignment())
      return *alignment;

    // Whenever we don't have alignment set, we will use an alignment
    // consistent with the element type; since the allocation size has to be a
    // power of two, we will bump to the next power of two if it isn't.
    unsigned eltSizeBytes = getMemRefEltSizeInBytes(
        getTypeConverter(), op.getType(), op, defaultLayout);
    return std::max(kMinAlignedAllocAlignment,
                    llvm::PowerOf2Ceil(eltSizeBytes));
  }

  /// Returns true if the memref size in bytes is known to be a multiple of
  /// factor.
  bool isMemRefSizeMultipleOf(MemRefType type, uint64_t factor, Operation *op,
                              const DataLayout *defaultLayout) const {
    uint64_t sizeDivisor =
        getMemRefEltSizeInBytes(getTypeConverter(), type, op, defaultLayout);
    for (unsigned i = 0, e = type.getRank(); i < e; i++) {
      if (type.isDynamicDim(i))
        continue;
      sizeDivisor = sizeDivisor * type.getDimSize(i);
    }
    return sizeDivisor % factor == 0;
  }

private:
  /// Default layout to use in absence of the corresponding analysis.
  DataLayout defaultLayout;
};

struct AllocaOpLowering : public ConvertOpToLLVMPattern<memref::AllocaOp> {
  using ConvertOpToLLVMPattern<memref::AllocaOp>::ConvertOpToLLVMPattern;

  /// Allocates the underlying buffer using the right call. `allocatedBytePtr`
  /// is set to null for stack allocations. `accessAlignment` is set if
  /// alignment is needed post allocation (for eg. in conjunction with malloc).
  LogicalResult
  matchAndRewrite(memref::AllocaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MemRefType memRefType = op.getType();
    if (!isConvertibleAndHasIdentityMaps(memRefType))
      return rewriter.notifyMatchFailure(op, "incompatible memref type");

    // Get actual sizes of the memref as values: static sizes are constant
    // values and dynamic sizes are passed to 'alloc' as operands.  In case of
    // zero-dimensional memref, assume a scalar (size 1).
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value size;

    this->getMemRefDescriptorSizes(loc, memRefType, adaptor.getOperands(),
                                   rewriter, sizes, strides, size, !true);

    // With alloca, one gets a pointer to the element type right away.
    // For stack allocations.
    auto elementType =
        typeConverter->convertType(op.getType().getElementType());
    FailureOr<unsigned> maybeAddressSpace =
        getTypeConverter()->getMemRefAddressSpace(op.getType());
    assert(succeeded(maybeAddressSpace) && "unsupported address space");
    unsigned addrSpace = *maybeAddressSpace;
    auto elementPtrType =
        LLVM::LLVMPointerType::get(rewriter.getContext(), addrSpace);

    auto allocatedElementPtr =
        LLVM::AllocaOp::create(rewriter, loc, elementPtrType, elementType, size,
                               op.getAlignment().value_or(0));

    // Create the MemRef descriptor.
    auto memRefDescriptor = this->createMemRefDescriptor(
        loc, memRefType, allocatedElementPtr, allocatedElementPtr, sizes,
        strides, rewriter);

    // Return the final value of the descriptor.
    rewriter.replaceOp(op, {memRefDescriptor});
    return success();
  }
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
      LLVM::BrOp::create(rewriter, loc, ValueRange(), remainingOpsBlock);
    }

    // Inline body region.
    Block *beforeBody = &allocaScopeOp.getBodyRegion().front();
    Block *afterBody = &allocaScopeOp.getBodyRegion().back();
    rewriter.inlineRegionBefore(allocaScopeOp.getBodyRegion(), continueBlock);

    // Save stack and then branch into the body of the region.
    rewriter.setInsertionPointToEnd(currentBlock);
    auto stackSaveOp = LLVM::StackSaveOp::create(rewriter, loc, getPtrType());
    LLVM::BrOp::create(rewriter, loc, ValueRange(), beforeBody);

    // Replace the alloca_scope return with a branch that jumps out of the body.
    // Stack restore before leaving the body region.
    rewriter.setInsertionPointToEnd(afterBody);
    auto returnOp =
        cast<memref::AllocaScopeReturnOp>(afterBody->getTerminator());
    auto branchOp = rewriter.replaceOpWithNewOp<LLVM::BrOp>(
        returnOp, returnOp.getResults(), continueBlock);

    // Insert stack restore before jumping out the body of the region.
    rewriter.setInsertionPoint(branchOp);
    LLVM::StackRestoreOp::create(rewriter, loc, stackSaveOp);

    // Replace the op with values return from the body region.
    rewriter.replaceOp(allocaScopeOp, continueBlock->getArguments());

    return success();
  }
};

struct AssumeAlignmentOpLowering
    : public ConvertOpToLLVMPattern<memref::AssumeAlignmentOp> {
  using ConvertOpToLLVMPattern<
      memref::AssumeAlignmentOp>::ConvertOpToLLVMPattern;
  explicit AssumeAlignmentOpLowering(const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<memref::AssumeAlignmentOp>(converter) {}

  LogicalResult
  matchAndRewrite(memref::AssumeAlignmentOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value memref = adaptor.getMemref();
    unsigned alignment = op.getAlignment();
    auto loc = op.getLoc();

    auto srcMemRefType = cast<MemRefType>(op.getMemref().getType());
    Value ptr = getStridedElementPtr(rewriter, loc, srcMemRefType, memref,
                                     /*indices=*/{});

    // Emit llvm.assume(true) ["align"(memref, alignment)].
    // This is more direct than ptrtoint-based checks, is explicitly supported,
    // and works with non-integral address spaces.
    Value trueCond =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(true));
    Value alignmentConst =
        createIndexAttrConstant(rewriter, loc, getIndexType(), alignment);
    LLVM::AssumeOp::create(rewriter, loc, trueCond, LLVM::AssumeAlignTag(), ptr,
                           alignmentConst);
    rewriter.replaceOp(op, memref);
    return success();
  }
};

// A `dealloc` is converted into a call to `free` on the underlying data buffer.
// The memref descriptor being an SSA value, there is no need to clean it up
// in any way.
class DeallocOpLowering : public ConvertOpToLLVMPattern<memref::DeallocOp> {
  SymbolTableCollection *symbolTables = nullptr;

public:
  explicit DeallocOpLowering(const LLVMTypeConverter &typeConverter,
                             SymbolTableCollection *symbolTables = nullptr,
                             PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<memref::DeallocOp>(typeConverter, benefit),
        symbolTables(symbolTables) {}

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Insert the `free` declaration if it is not already present.
    FailureOr<LLVM::LLVMFuncOp> freeFunc =
        getFreeFn(rewriter, getTypeConverter(), op->getParentOfType<ModuleOp>(),
                  symbolTables);
    if (failed(freeFunc))
      return failure();
    Value allocatedPtr;
    if (auto unrankedTy =
            llvm::dyn_cast<UnrankedMemRefType>(op.getMemref().getType())) {
      auto elementPtrTy = LLVM::LLVMPointerType::get(
          rewriter.getContext(), unrankedTy.getMemorySpaceAsInt());
      allocatedPtr = UnrankedMemRefDescriptor::allocatedPtr(
          rewriter, op.getLoc(),
          UnrankedMemRefDescriptor(adaptor.getMemref())
              .memRefDescPtr(rewriter, op.getLoc()),
          elementPtrTy);
    } else {
      allocatedPtr = MemRefDescriptor(adaptor.getMemref())
                         .allocatedPtr(rewriter, op.getLoc());
    }
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, freeFunc.value(),
                                              allocatedPtr);
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
    if (isa<UnrankedMemRefType>(operandType)) {
      FailureOr<Value> extractedSize = extractSizeOfUnrankedMemRef(
          operandType, dimOp, adaptor.getOperands(), rewriter);
      if (failed(extractedSize))
        return failure();
      rewriter.replaceOp(dimOp, {*extractedSize});
      return success();
    }
    if (isa<MemRefType>(operandType)) {
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

    auto unrankedMemRefType = cast<UnrankedMemRefType>(operandType);
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

    // Get pointer to offset field of memref<element_type> descriptor.
    auto indexPtrTy =
        LLVM::LLVMPointerType::get(rewriter.getContext(), addressSpace);
    Value offsetPtr =
        LLVM::GEPOp::create(rewriter, loc, indexPtrTy, elementType,
                            underlyingRankedDesc, ArrayRef<LLVM::GEPArg>{0, 2});

    // The size value that we have to extract can be obtained using GEPop with
    // `dimOp.index() + 1` index argument.
    Value idxPlusOne = LLVM::AddOp::create(
        rewriter, loc,
        createIndexAttrConstant(rewriter, loc, getIndexType(), 1),
        adaptor.getIndex());
    Value sizePtr = LLVM::GEPOp::create(rewriter, loc, indexPtrTy,
                                        getTypeConverter()->getIndexType(),
                                        offsetPtr, idxPlusOne);
    return LLVM::LoadOp::create(rewriter, loc,
                                getTypeConverter()->getIndexType(), sizePtr)
        .getResult();
  }

  std::optional<int64_t> getConstantDimIndex(memref::DimOp dimOp) const {
    if (auto idx = dimOp.getConstantIndex())
      return idx;

    if (auto constantOp = dimOp.getIndex().getDefiningOp<LLVM::ConstantOp>())
      return cast<IntegerAttr>(constantOp.getValue()).getValue().getSExtValue();

    return std::nullopt;
  }

  Value extractSizeOfRankedMemRef(Type operandType, memref::DimOp dimOp,
                                  OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    Location loc = dimOp.getLoc();

    // Take advantage if index is constant.
    MemRefType memRefType = cast<MemRefType>(operandType);
    Type indexType = getIndexType();
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
        return createIndexAttrConstant(rewriter, loc, indexType, dimSize);
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
    auto *loopBlock = rewriter.splitBlock(initBlock, Block::iterator(atomicOp));
    loopBlock->addArgument(valueType, loc);

    auto *endBlock =
        rewriter.splitBlock(loopBlock, Block::iterator(atomicOp)++);

    // Compute the loaded value and branch to the loop block.
    rewriter.setInsertionPointToEnd(initBlock);
    auto memRefType = cast<MemRefType>(atomicOp.getMemref().getType());
    auto dataPtr = getStridedElementPtr(
        rewriter, loc, memRefType, adaptor.getMemref(), adaptor.getIndices());
    Value init = LLVM::LoadOp::create(
        rewriter, loc, typeConverter->convertType(memRefType.getElementType()),
        dataPtr);
    LLVM::BrOp::create(rewriter, loc, init, loopBlock);

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
    auto cmpxchg =
        LLVM::AtomicCmpXchgOp::create(rewriter, loc, dataPtr, loopArgument,
                                      result, successOrdering, failureOrdering);
    // Extract the %new_loaded and %ok values from the pair.
    Value newLoaded = LLVM::ExtractValueOp::create(rewriter, loc, cmpxchg, 0);
    Value ok = LLVM::ExtractValueOp::create(rewriter, loc, cmpxchg, 1);

    // Conditionally branch to the end or back to the loop depending on %ok.
    LLVM::CondBrOp::create(rewriter, loc, ok, endBlock, ArrayRef<Value>(),
                           loopBlock, newLoaded);

    rewriter.setInsertionPointToEnd(endBlock);

    // The 'result' of the atomic_rmw op is the newly loaded value.
    rewriter.replaceOp(atomicOp, {newLoaded});

    return success();
  }
};

/// Returns the LLVM type of the global variable given the memref type `type`.
static Type
convertGlobalMemrefTypeToLLVM(MemRefType type,
                              const LLVMTypeConverter &typeConverter) {
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
class GlobalMemrefOpLowering : public ConvertOpToLLVMPattern<memref::GlobalOp> {
  SymbolTableCollection *symbolTables = nullptr;

public:
  explicit GlobalMemrefOpLowering(const LLVMTypeConverter &typeConverter,
                                  SymbolTableCollection *symbolTables = nullptr,
                                  PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<memref::GlobalOp>(typeConverter, benefit),
        symbolTables(symbolTables) {}

  LogicalResult
  matchAndRewrite(memref::GlobalOp global, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = global.getType();
    if (!isConvertibleAndHasIdentityMaps(type))
      return failure();

    Type arrayTy = convertGlobalMemrefTypeToLLVM(type, *getTypeConverter());

    LLVM::Linkage linkage =
        global.isPublic() ? LLVM::Linkage::External : LLVM::Linkage::Private;
    bool isExternal = global.isExternal();
    bool isUninitialized = global.isUninitialized();

    Attribute initialValue = nullptr;
    if (!isExternal && !isUninitialized) {
      auto elementsAttr = llvm::cast<ElementsAttr>(*global.getInitialValue());
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

    // Remove old operation from symbol table.
    SymbolTable *symbolTable = nullptr;
    if (symbolTables) {
      Operation *symbolTableOp =
          global->getParentWithTrait<OpTrait::SymbolTable>();
      symbolTable = &symbolTables->getSymbolTable(symbolTableOp);
      symbolTable->remove(global);
    }

    // Create new operation.
    auto newGlobal = rewriter.replaceOpWithNewOp<LLVM::GlobalOp>(
        global, arrayTy, global.getConstant(), linkage, global.getSymName(),
        initialValue, alignment, *addressSpace);

    // Insert new operation into symbol table.
    if (symbolTable)
      symbolTable->insert(newGlobal, rewriter.getInsertionPoint());

    if (!isExternal && isUninitialized) {
      rewriter.createBlock(&newGlobal.getInitializerRegion());
      Value undef[] = {
          LLVM::UndefOp::create(rewriter, newGlobal.getLoc(), arrayTy)};
      LLVM::ReturnOp::create(rewriter, newGlobal.getLoc(), undef);
    }
    return success();
  }
};

/// GetGlobalMemrefOp is lowered into a Memref descriptor with the pointer to
/// the first element stashed into the descriptor. This reuses
/// `AllocLikeOpLowering` to reuse the Memref descriptor construction.
struct GetGlobalMemrefOpLowering
    : public ConvertOpToLLVMPattern<memref::GetGlobalOp> {
  using ConvertOpToLLVMPattern<memref::GetGlobalOp>::ConvertOpToLLVMPattern;

  /// Buffer "allocation" for memref.get_global op is getting the address of
  /// the global variable referenced.
  LogicalResult
  matchAndRewrite(memref::GetGlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MemRefType memRefType = op.getType();
    if (!isConvertibleAndHasIdentityMaps(memRefType))
      return rewriter.notifyMatchFailure(op, "incompatible memref type");

    // Get actual sizes of the memref as values: static sizes are constant
    // values and dynamic sizes are passed to 'alloc' as operands.  In case of
    // zero-dimensional memref, assume a scalar (size 1).
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value sizeBytes;

    this->getMemRefDescriptorSizes(loc, memRefType, adaptor.getOperands(),
                                   rewriter, sizes, strides, sizeBytes, !false);

    MemRefType type = cast<MemRefType>(op.getResult().getType());

    // This is called after a type conversion, which would have failed if this
    // call fails.
    FailureOr<unsigned> maybeAddressSpace =
        getTypeConverter()->getMemRefAddressSpace(type);
    assert(succeeded(maybeAddressSpace) && "unsupported address space");
    unsigned memSpace = *maybeAddressSpace;

    Type arrayTy = convertGlobalMemrefTypeToLLVM(type, *getTypeConverter());
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), memSpace);
    auto addressOf =
        LLVM::AddressOfOp::create(rewriter, loc, ptrTy, op.getName());

    // Get the address of the first element in the array by creating a GEP with
    // the address of the GV as the base, and (rank + 1) number of 0 indices.
    auto gep =
        LLVM::GEPOp::create(rewriter, loc, ptrTy, arrayTy, addressOf,
                            SmallVector<LLVM::GEPArg>(type.getRank() + 1, 0));

    // We do not expect the memref obtained using `memref.get_global` to be
    // ever deallocated. Set the allocated pointer to be known bad value to
    // help debug if that ever happens.
    auto intPtrType = getIntPtrType(memSpace);
    Value deadBeefConst =
        createIndexAttrConstant(rewriter, op->getLoc(), intPtrType, 0xdeadbeef);
    auto deadBeefPtr =
        LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, deadBeefConst);

    // Both allocated and aligned pointers are same. We could potentially stash
    // a nullptr for the allocated pointer since we do not expect any dealloc.
    // Create the MemRef descriptor.
    auto memRefDescriptor = this->createMemRefDescriptor(
        loc, memRefType, deadBeefPtr, gep, sizes, strides, rewriter);

    // Return the final value of the descriptor.
    rewriter.replaceOp(op, {memRefDescriptor});
    return success();
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

    // Per memref.load spec, the indices must be in-bounds:
    // 0 <= idx < dim_size, and additionally all offsets are non-negative,
    // hence inbounds and nuw are used when lowering to llvm.getelementptr.
    Value dataPtr = getStridedElementPtr(rewriter, loadOp.getLoc(), type,
                                         adaptor.getMemref(),
                                         adaptor.getIndices(), kNoWrapFlags);
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        loadOp, typeConverter->convertType(type.getElementType()), dataPtr,
        loadOp.getAlignment().value_or(0), false, loadOp.getNontemporal());
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

    // Per memref.store spec, the indices must be in-bounds:
    // 0 <= idx < dim_size, and additionally all offsets are non-negative,
    // hence inbounds and nuw are used when lowering to llvm.getelementptr.
    Value dataPtr =
        getStridedElementPtr(rewriter, op.getLoc(), type, adaptor.getMemref(),
                             adaptor.getIndices(), kNoWrapFlags);
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValue(), dataPtr,
                                               op.getAlignment().value_or(0),
                                               false, op.getNontemporal());
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

    Value dataPtr = getStridedElementPtr(
        rewriter, loc, type, adaptor.getMemref(), adaptor.getIndices());

    // Replace with llvm.prefetch.
    IntegerAttr isWrite = rewriter.getI32IntegerAttr(prefetchOp.getIsWrite());
    IntegerAttr localityHint = prefetchOp.getLocalityHintAttr();
    IntegerAttr isData =
        rewriter.getI32IntegerAttr(prefetchOp.getIsDataCache());
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
    if (isa<UnrankedMemRefType>(operandType)) {
      UnrankedMemRefDescriptor desc(adaptor.getMemref());
      rewriter.replaceOp(op, {desc.rank(rewriter, loc)});
      return success();
    }
    if (auto rankedMemRefType = dyn_cast<MemRefType>(operandType)) {
      Type indexType = getIndexType();
      rewriter.replaceOp(op,
                         {createIndexAttrConstant(rewriter, loc, indexType,
                                                  rankedMemRefType.getRank())});
      return success();
    }
    return failure();
  }
};

struct MemRefCastOpLowering : public ConvertOpToLLVMPattern<memref::CastOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::CastOp memRefCastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type srcType = memRefCastOp.getOperand().getType();
    Type dstType = memRefCastOp.getType();

    // memref::CastOp reduce to bitcast in the ranked MemRef case and can be
    // used for type erasure. For now they must preserve underlying element type
    // and require source and result type to have the same rank. Therefore,
    // perform a sanity check that the underlying structs are the same. Once op
    // semantics are relaxed we can revisit.
    if (isa<MemRefType>(srcType) && isa<MemRefType>(dstType))
      if (typeConverter->convertType(srcType) !=
          typeConverter->convertType(dstType))
        return failure();

    // Unranked to unranked cast is disallowed
    if (isa<UnrankedMemRefType>(srcType) && isa<UnrankedMemRefType>(dstType))
      return failure();

    auto targetStructType = typeConverter->convertType(memRefCastOp.getType());
    auto loc = memRefCastOp.getLoc();

    // For ranked/ranked case, just keep the original descriptor.
    if (isa<MemRefType>(srcType) && isa<MemRefType>(dstType)) {
      rewriter.replaceOp(memRefCastOp, {adaptor.getSource()});
      return success();
    }

    if (isa<MemRefType>(srcType) && isa<UnrankedMemRefType>(dstType)) {
      // Casting ranked to unranked memref type
      // Set the rank in the destination from the memref type
      // Allocate space on the stack and copy the src memref descriptor
      // Set the ptr in the destination to the stack space
      auto srcMemRefType = cast<MemRefType>(srcType);
      int64_t rank = srcMemRefType.getRank();
      // ptr = AllocaOp sizeof(MemRefDescriptor)
      auto ptr = getTypeConverter()->promoteOneMemRefDescriptor(
          loc, adaptor.getSource(), rewriter);

      // rank = ConstantOp srcRank
      auto rankVal = LLVM::ConstantOp::create(rewriter, loc, getIndexType(),
                                              rewriter.getIndexAttr(rank));
      // poison = PoisonOp
      UnrankedMemRefDescriptor memRefDesc =
          UnrankedMemRefDescriptor::poison(rewriter, loc, targetStructType);
      // d1 = InsertValueOp poison, rank, 0
      memRefDesc.setRank(rewriter, loc, rankVal);
      // d2 = InsertValueOp d1, ptr, 1
      memRefDesc.setMemRefDescPtr(rewriter, loc, ptr);
      rewriter.replaceOp(memRefCastOp, (Value)memRefDesc);

    } else if (isa<UnrankedMemRefType>(srcType) && isa<MemRefType>(dstType)) {
      // Casting from unranked type to ranked.
      // The operation is assumed to be doing a correct cast. If the destination
      // type mismatches the unranked the type, it is undefined behavior.
      UnrankedMemRefDescriptor memRefDesc(adaptor.getSource());
      // ptr = ExtractValueOp src, 1
      auto ptr = memRefDesc.memRefDescPtr(rewriter, loc);

      // struct = LoadOp ptr
      auto loadOp = LLVM::LoadOp::create(rewriter, loc, targetStructType, ptr);
      rewriter.replaceOp(memRefCastOp, loadOp.getResult());
    } else {
      llvm_unreachable("Unsupported unranked memref to unranked memref cast");
    }

    return success();
  }
};

/// Pattern to lower a `memref.copy` to llvm.
///
/// For memrefs with identity layouts, the copy is lowered to the llvm
/// `memcpy` intrinsic. For non-identity layouts, the copy is lowered to a call
/// to the generic `MemrefCopyFn`.
class MemRefCopyOpLowering : public ConvertOpToLLVMPattern<memref::CopyOp> {
  SymbolTableCollection *symbolTables = nullptr;

public:
  explicit MemRefCopyOpLowering(const LLVMTypeConverter &typeConverter,
                                SymbolTableCollection *symbolTables = nullptr,
                                PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<memref::CopyOp>(typeConverter, benefit),
        symbolTables(symbolTables) {}

  LogicalResult
  lowerToMemCopyIntrinsic(memref::CopyOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcType = dyn_cast<MemRefType>(op.getSource().getType());

    MemRefDescriptor srcDesc(adaptor.getSource());

    // Compute number of elements.
    Value numElements = LLVM::ConstantOp::create(rewriter, loc, getIndexType(),
                                                 rewriter.getIndexAttr(1));
    for (int pos = 0; pos < srcType.getRank(); ++pos) {
      auto size = srcDesc.size(rewriter, loc, pos);
      numElements = LLVM::MulOp::create(rewriter, loc, numElements, size);
    }

    // Get element size.
    auto sizeInBytes = getSizeInBytes(loc, srcType.getElementType(), rewriter);
    // Compute total.
    Value totalSize =
        LLVM::MulOp::create(rewriter, loc, numElements, sizeInBytes);

    Type elementType = typeConverter->convertType(srcType.getElementType());

    Value srcBasePtr = srcDesc.alignedPtr(rewriter, loc);
    Value srcOffset = srcDesc.offset(rewriter, loc);
    Value srcPtr = LLVM::GEPOp::create(rewriter, loc, srcBasePtr.getType(),
                                       elementType, srcBasePtr, srcOffset);
    MemRefDescriptor targetDesc(adaptor.getTarget());
    Value targetBasePtr = targetDesc.alignedPtr(rewriter, loc);
    Value targetOffset = targetDesc.offset(rewriter, loc);
    Value targetPtr =
        LLVM::GEPOp::create(rewriter, loc, targetBasePtr.getType(), elementType,
                            targetBasePtr, targetOffset);
    LLVM::MemcpyOp::create(rewriter, loc, targetPtr, srcPtr, totalSize,
                           /*isVolatile=*/false);
    rewriter.eraseOp(op);

    return success();
  }

  LogicalResult
  lowerToMemCopyFunctionCall(memref::CopyOp op, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcType = cast<BaseMemRefType>(op.getSource().getType());
    auto targetType = cast<BaseMemRefType>(op.getTarget().getType());

    // First make sure we have an unranked memref descriptor representation.
    auto makeUnranked = [&, this](Value ranked, MemRefType type) {
      auto rank = LLVM::ConstantOp::create(rewriter, loc, getIndexType(),
                                           type.getRank());
      auto *typeConverter = getTypeConverter();
      auto ptr =
          typeConverter->promoteOneMemRefDescriptor(loc, ranked, rewriter);

      auto unrankedType =
          UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace());
      return UnrankedMemRefDescriptor::pack(
          rewriter, loc, *typeConverter, unrankedType, ValueRange{rank, ptr});
    };

    // Save stack position before promoting descriptors
    auto stackSaveOp = LLVM::StackSaveOp::create(rewriter, loc, getPtrType());

    auto srcMemRefType = dyn_cast<MemRefType>(srcType);
    Value unrankedSource =
        srcMemRefType ? makeUnranked(adaptor.getSource(), srcMemRefType)
                      : adaptor.getSource();
    auto targetMemRefType = dyn_cast<MemRefType>(targetType);
    Value unrankedTarget =
        targetMemRefType ? makeUnranked(adaptor.getTarget(), targetMemRefType)
                         : adaptor.getTarget();

    // Now promote the unranked descriptors to the stack.
    auto one = LLVM::ConstantOp::create(rewriter, loc, getIndexType(),
                                        rewriter.getIndexAttr(1));
    auto promote = [&](Value desc) {
      auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
      auto allocated =
          LLVM::AllocaOp::create(rewriter, loc, ptrType, desc.getType(), one);
      LLVM::StoreOp::create(rewriter, loc, desc, allocated);
      return allocated;
    };

    auto sourcePtr = promote(unrankedSource);
    auto targetPtr = promote(unrankedTarget);

    // Derive size from llvm.getelementptr which will account for any
    // potential alignment
    auto elemSize = getSizeInBytes(loc, srcType.getElementType(), rewriter);
    auto copyFn = LLVM::lookupOrCreateMemRefCopyFn(
        rewriter, op->getParentOfType<ModuleOp>(), getIndexType(),
        sourcePtr.getType(), symbolTables);
    if (failed(copyFn))
      return failure();
    LLVM::CallOp::create(rewriter, loc, copyFn.value(),
                         ValueRange{elemSize, sourcePtr, targetPtr});

    // Restore stack used for descriptors
    LLVM::StackRestoreOp::create(rewriter, loc, stackSaveOp);

    rewriter.eraseOp(op);

    return success();
  }

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = cast<BaseMemRefType>(op.getSource().getType());
    auto targetType = cast<BaseMemRefType>(op.getTarget().getType());

    auto isContiguousMemrefType = [&](BaseMemRefType type) {
      auto memrefType = dyn_cast<mlir::MemRefType>(type);
      // We can use memcpy for memrefs if they have an identity layout or are
      // contiguous with an arbitrary offset. Ignore empty memrefs, which is a
      // special case handled by memrefCopy.
      return memrefType &&
             (memrefType.getLayout().isIdentity() ||
              (memrefType.hasStaticShape() && memrefType.getNumElements() > 0 &&
               memref::isStaticShapeAndContiguousRowMajor(memrefType)));
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
    if (auto resultTypeR = dyn_cast<MemRefType>(resultType)) {
      auto resultDescType =
          cast<LLVM::LLVMStructType>(typeConverter->convertType(resultTypeR));
      Type newPtrType = resultDescType.getBody()[0];

      SmallVector<Value> descVals;
      MemRefDescriptor::unpack(rewriter, loc, adaptor.getSource(), resultTypeR,
                               descVals);
      descVals[0] =
          LLVM::AddrSpaceCastOp::create(rewriter, loc, newPtrType, descVals[0]);
      descVals[1] =
          LLVM::AddrSpaceCastOp::create(rewriter, loc, newPtrType, descVals[1]);
      Value result = MemRefDescriptor::pack(rewriter, loc, *getTypeConverter(),
                                            resultTypeR, descVals);
      rewriter.replaceOp(op, result);
      return success();
    }
    if (auto resultTypeU = dyn_cast<UnrankedMemRefType>(resultType)) {
      // Since the type converter won't be doing this for us, get the address
      // space.
      auto sourceType = cast<UnrankedMemRefType>(op.getSource().getType());
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
      auto result = UnrankedMemRefDescriptor::poison(
          rewriter, loc, typeConverter->convertType(resultTypeU));
      result.setRank(rewriter, loc, rank);
      Value resultUnderlyingSize = UnrankedMemRefDescriptor::computeSize(
          rewriter, loc, *getTypeConverter(), result, resultAddrSpace);
      Value resultUnderlyingDesc =
          LLVM::AllocaOp::create(rewriter, loc, getPtrType(),
                                 rewriter.getI8Type(), resultUnderlyingSize);
      result.setMemRefDescPtr(rewriter, loc, resultUnderlyingDesc);

      // Copy pointers, performing address space casts.
      auto sourceElemPtrType =
          LLVM::LLVMPointerType::get(rewriter.getContext(), sourceAddrSpace);
      auto resultElemPtrType =
          LLVM::LLVMPointerType::get(rewriter.getContext(), resultAddrSpace);

      Value allocatedPtr = sourceDesc.allocatedPtr(
          rewriter, loc, sourceUnderlyingDesc, sourceElemPtrType);
      Value alignedPtr =
          sourceDesc.alignedPtr(rewriter, loc, *getTypeConverter(),
                                sourceUnderlyingDesc, sourceElemPtrType);
      allocatedPtr = LLVM::AddrSpaceCastOp::create(
          rewriter, loc, resultElemPtrType, allocatedPtr);
      alignedPtr = LLVM::AddrSpaceCastOp::create(rewriter, loc,
                                                 resultElemPtrType, alignedPtr);

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
          2 * llvm::divideCeil(
                  getTypeConverter()->getPointerBitwidth(resultAddrSpace), 8);
      Value bytesToSkipConst = LLVM::ConstantOp::create(
          rewriter, loc, getIndexType(), rewriter.getIndexAttr(bytesToSkip));
      Value copySize =
          LLVM::SubOp::create(rewriter, loc, getIndexType(),
                              resultUnderlyingSize, bytesToSkipConst);
      LLVM::MemcpyOp::create(rewriter, loc, resultIndexVals, sourceIndexVals,
                             copySize, /*isVolatile=*/false);

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
                                     const LLVMTypeConverter &typeConverter,
                                     Value originalOperand,
                                     Value convertedOperand,
                                     Value *allocatedPtr, Value *alignedPtr,
                                     Value *offset = nullptr) {
  Type operandType = originalOperand.getType();
  if (isa<MemRefType>(operandType)) {
    MemRefDescriptor desc(convertedOperand);
    *allocatedPtr = desc.allocatedPtr(rewriter, loc);
    *alignedPtr = desc.alignedPtr(rewriter, loc);
    if (offset != nullptr)
      *offset = desc.offset(rewriter, loc);
    return;
  }

  // These will all cause assert()s on unconvertible types.
  unsigned memorySpace = *typeConverter.getMemRefAddressSpace(
      cast<UnrankedMemRefType>(operandType));
  auto elementPtrType =
      LLVM::LLVMPointerType::get(rewriter.getContext(), memorySpace);

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
        cast<MemRefType>(castOp.getResult().getType());
    auto llvmTargetDescriptorTy = dyn_cast_or_null<LLVM::LLVMStructType>(
        typeConverter->convertType(targetMemRefType));
    if (!llvmTargetDescriptorTy)
      return failure();

    // Create descriptor.
    Location loc = castOp.getLoc();
    auto desc = MemRefDescriptor::poison(rewriter, loc, llvmTargetDescriptorTy);

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
    auto shapeMemRefType = cast<MemRefType>(reshapeOp.getShape().getType());
    if (shapeMemRefType.hasStaticShape()) {
      MemRefType targetMemRefType =
          cast<MemRefType>(reshapeOp.getResult().getType());
      auto llvmTargetDescriptorTy = dyn_cast_or_null<LLVM::LLVMStructType>(
          typeConverter->convertType(targetMemRefType));
      if (!llvmTargetDescriptorTy)
        return failure();

      // Create descriptor.
      Location loc = reshapeOp.getLoc();
      auto desc =
          MemRefDescriptor::poison(rewriter, loc, llvmTargetDescriptorTy);

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
      if (failed(targetMemRefType.getStridesAndOffset(strides, offset)))
        return rewriter.notifyMatchFailure(
            reshapeOp, "failed to get stride and offset exprs");

      if (!isStaticStrideOrOffset(offset))
        return rewriter.notifyMatchFailure(reshapeOp,
                                           "dynamic offset is unsupported");

      desc.setConstantOffset(rewriter, loc, offset);

      assert(targetMemRefType.getLayout().isIdentity() &&
             "Identity layout map is a precondition of a valid reshape op");

      Type indexType = getIndexType();
      Value stride = nullptr;
      int64_t targetRank = targetMemRefType.getRank();
      for (auto i : llvm::reverse(llvm::seq<int64_t>(0, targetRank))) {
        if (ShapedType::isStatic(strides[i])) {
          // If the stride for this dimension is dynamic, then use the product
          // of the sizes of the inner dimensions.
          stride =
              createIndexAttrConstant(rewriter, loc, indexType, strides[i]);
        } else if (!stride) {
          // `stride` is null only in the first iteration of the loop.  However,
          // since the target memref has an identity layout, we can safely set
          // the innermost stride to 1.
          stride = createIndexAttrConstant(rewriter, loc, indexType, 1);
        }

        Value dimSize;
        // If the size of this dimension is dynamic, then load it at runtime
        // from the shape operand.
        if (!targetMemRefType.isDynamicDim(i)) {
          dimSize = createIndexAttrConstant(rewriter, loc, indexType,
                                            targetMemRefType.getDimSize(i));
        } else {
          Value shapeOp = reshapeOp.getShape();
          Value index = createIndexAttrConstant(rewriter, loc, indexType, i);
          dimSize = memref::LoadOp::create(rewriter, loc, shapeOp, index);
          Type indexType = getIndexType();
          if (dimSize.getType() != indexType)
            dimSize = typeConverter->materializeTargetConversion(
                rewriter, loc, indexType, dimSize);
          assert(dimSize && "Invalid memref element type");
        }

        desc.setSize(rewriter, loc, i, dimSize);
        desc.setStride(rewriter, loc, i, stride);

        // Prepare the stride value for the next dimension.
        stride = LLVM::MulOp::create(rewriter, loc, stride, dimSize);
      }

      *descriptor = desc;
      return success();
    }

    // The shape is a rank-1 tensor with unknown length.
    Location loc = reshapeOp.getLoc();
    MemRefDescriptor shapeDesc(adaptor.getShape());
    Value resultRank = shapeDesc.size(rewriter, loc, 0);

    // Extract address space and element type.
    auto targetType = cast<UnrankedMemRefType>(reshapeOp.getResult().getType());
    unsigned addressSpace =
        *getTypeConverter()->getMemRefAddressSpace(targetType);

    // Create the unranked memref descriptor that holds the ranked one. The
    // inner descriptor is allocated on stack.
    auto targetDesc = UnrankedMemRefDescriptor::poison(
        rewriter, loc, typeConverter->convertType(targetType));
    targetDesc.setRank(rewriter, loc, resultRank);
    Value allocationSize = UnrankedMemRefDescriptor::computeSize(
        rewriter, loc, *getTypeConverter(), targetDesc, addressSpace);
    Value underlyingDescPtr = LLVM::AllocaOp::create(
        rewriter, loc, getPtrType(), IntegerType::get(getContext(), 8),
        allocationSize);
    targetDesc.setMemRefDescPtr(rewriter, loc, underlyingDescPtr);

    // Extract pointers and offset from the source memref.
    Value allocatedPtr, alignedPtr, offset;
    extractPointersAndOffset(loc, rewriter, *getTypeConverter(),
                             reshapeOp.getSource(), adaptor.getSource(),
                             &allocatedPtr, &alignedPtr, &offset);

    // Set pointers and offset.
    auto elementPtrType =
        LLVM::LLVMPointerType::get(rewriter.getContext(), addressSpace);

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
    Value oneIndex = createIndexAttrConstant(rewriter, loc, getIndexType(), 1);
    Value resultRankMinusOne =
        LLVM::SubOp::create(rewriter, loc, resultRank, oneIndex);

    Block *initBlock = rewriter.getInsertionBlock();
    Type indexType = getTypeConverter()->getIndexType();
    Block::iterator remainingOpsIt = std::next(rewriter.getInsertionPoint());

    Block *condBlock = rewriter.createBlock(initBlock->getParent(), {},
                                            {indexType, indexType}, {loc, loc});

    // Move the remaining initBlock ops to condBlock.
    Block *remainingBlock = rewriter.splitBlock(initBlock, remainingOpsIt);
    rewriter.mergeBlocks(remainingBlock, condBlock, ValueRange());

    rewriter.setInsertionPointToEnd(initBlock);
    LLVM::BrOp::create(rewriter, loc,
                       ValueRange({resultRankMinusOne, oneIndex}), condBlock);
    rewriter.setInsertionPointToStart(condBlock);
    Value indexArg = condBlock->getArgument(0);
    Value strideArg = condBlock->getArgument(1);

    Value zeroIndex = createIndexAttrConstant(rewriter, loc, indexType, 0);
    Value pred = LLVM::ICmpOp::create(
        rewriter, loc, IntegerType::get(rewriter.getContext(), 1),
        LLVM::ICmpPredicate::sge, indexArg, zeroIndex);

    Block *bodyBlock =
        rewriter.splitBlock(condBlock, rewriter.getInsertionPoint());
    rewriter.setInsertionPointToStart(bodyBlock);

    // Copy size from shape to descriptor.
    auto llvmIndexPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value sizeLoadGep = LLVM::GEPOp::create(
        rewriter, loc, llvmIndexPtrType,
        typeConverter->convertType(shapeMemRefType.getElementType()),
        shapeOperandPtr, indexArg);
    Value size = LLVM::LoadOp::create(rewriter, loc, indexType, sizeLoadGep);
    UnrankedMemRefDescriptor::setSize(rewriter, loc, *getTypeConverter(),
                                      targetSizesBase, indexArg, size);

    // Write stride value and compute next one.
    UnrankedMemRefDescriptor::setStride(rewriter, loc, *getTypeConverter(),
                                        targetStridesBase, indexArg, strideArg);
    Value nextStride = LLVM::MulOp::create(rewriter, loc, strideArg, size);

    // Decrement loop counter and branch back.
    Value decrement = LLVM::SubOp::create(rewriter, loc, indexArg, oneIndex);
    LLVM::BrOp::create(rewriter, loc, ValueRange({decrement, nextStride}),
                       condBlock);

    Block *remainder =
        rewriter.splitBlock(bodyBlock, rewriter.getInsertionPoint());

    // Hook up the cond exit to the remainder.
    rewriter.setInsertionPointToEnd(condBlock);
    LLVM::CondBrOp::create(rewriter, loc, pred, bodyBlock, ValueRange(),
                           remainder, ValueRange());

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

    auto targetMemRef = MemRefDescriptor::poison(
        rewriter, loc,
        typeConverter->convertType(transposeOp.getIn().getType()));

    // Copy the base and aligned pointers from the old descriptor to the new
    // one.
    targetMemRef.setAllocatedPtr(rewriter, loc,
                                 viewMemRef.allocatedPtr(rewriter, loc));
    targetMemRef.setAlignedPtr(rewriter, loc,
                               viewMemRef.alignedPtr(rewriter, loc));

    // Copy the offset pointer from the old descriptor to the new one.
    targetMemRef.setOffset(rewriter, loc, viewMemRef.offset(rewriter, loc));

    // Iterate over the dimensions and apply size/stride permutation:
    // When enumerating the results of the permutation map, the enumeration
    // index is the index into the target dimensions and the DimExpr points to
    // the dimension of the source memref.
    for (const auto &en :
         llvm::enumerate(transposeOp.getPermutation().getResults())) {
      int targetPos = en.index();
      int sourcePos = cast<AffineDimExpr>(en.value()).getPosition();
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
                ArrayRef<int64_t> shape, ValueRange dynamicSizes, unsigned idx,
                Type indexType) const {
    assert(idx < shape.size());
    if (ShapedType::isStatic(shape[idx]))
      return createIndexAttrConstant(rewriter, loc, indexType, shape[idx]);
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
                  Value runningStride, unsigned idx, Type indexType) const {
    assert(idx < strides.size());
    if (ShapedType::isStatic(strides[idx]))
      return createIndexAttrConstant(rewriter, loc, indexType, strides[idx]);
    if (nextSize)
      return runningStride
                 ? LLVM::MulOp::create(rewriter, loc, runningStride, nextSize)
                 : nextSize;
    assert(!runningStride);
    return createIndexAttrConstant(rewriter, loc, indexType, 1);
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
    auto successStrides = viewMemRefType.getStridesAndOffset(strides, offset);
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
    auto targetMemRef = MemRefDescriptor::poison(rewriter, loc, targetDescTy);

    // Field 1: Copy the allocated pointer, used for malloc/free.
    Value allocatedPtr = sourceMemRef.allocatedPtr(rewriter, loc);
    auto srcMemRefType = cast<MemRefType>(viewOp.getSource().getType());
    targetMemRef.setAllocatedPtr(rewriter, loc, allocatedPtr);

    // Field 2: Copy the actual aligned pointer to payload.
    Value alignedPtr = sourceMemRef.alignedPtr(rewriter, loc);
    alignedPtr = LLVM::GEPOp::create(
        rewriter, loc, alignedPtr.getType(),
        typeConverter->convertType(srcMemRefType.getElementType()), alignedPtr,
        adaptor.getByteShift());

    targetMemRef.setAlignedPtr(rewriter, loc, alignedPtr);

    Type indexType = getIndexType();
    // Field 3: The offset in the resulting type must be 0. This is
    // because of the type change: an offset on srcType* may not be
    // expressible as an offset on dstType*.
    targetMemRef.setOffset(
        rewriter, loc,
        createIndexAttrConstant(rewriter, loc, indexType, offset));

    // Early exit for 0-D corner case.
    if (viewMemRefType.getRank() == 0)
      return rewriter.replaceOp(viewOp, {targetMemRef}), success();

    // Fields 4 and 5: Update sizes and strides.
    Value stride = nullptr, nextSize = nullptr;
    for (int i = viewMemRefType.getRank() - 1; i >= 0; --i) {
      // Update size.
      Value size = getSize(rewriter, loc, viewMemRefType.getShape(),
                           adaptor.getSizes(), i, indexType);
      targetMemRef.setSize(rewriter, loc, i, size);
      // Update stride.
      stride =
          getStride(rewriter, loc, strides, nextSize, stride, i, indexType);
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
  case arith::AtomicRMWKind::maximumf:
    // TODO: remove this by end of 2025.
    LDBG() << "the lowering of memref.atomicrmw maximumf changed "
              "from fmax to fmaximum, expect more NaNs";
    return LLVM::AtomicBinOp::fmaximum;
  case arith::AtomicRMWKind::maxnumf:
    return LLVM::AtomicBinOp::fmax;
  case arith::AtomicRMWKind::maxs:
    return LLVM::AtomicBinOp::max;
  case arith::AtomicRMWKind::maxu:
    return LLVM::AtomicBinOp::umax;
  case arith::AtomicRMWKind::minimumf:
    // TODO: remove this by end of 2025.
    LDBG() << "the lowering of memref.atomicrmw minimum changed "
              "from fmin to fminimum, expect more NaNs";
    return LLVM::AtomicBinOp::fminimum;
  case arith::AtomicRMWKind::minnumf:
    return LLVM::AtomicBinOp::fmin;
  case arith::AtomicRMWKind::mins:
    return LLVM::AtomicBinOp::min;
  case arith::AtomicRMWKind::minu:
    return LLVM::AtomicBinOp::umin;
  case arith::AtomicRMWKind::ori:
    return LLVM::AtomicBinOp::_or;
  case arith::AtomicRMWKind::xori:
    return LLVM::AtomicBinOp::_xor;
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
    auto maybeKind = matchSimpleAtomicOp(atomicOp);
    if (!maybeKind)
      return failure();
    auto memRefType = atomicOp.getMemRefType();
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(memRefType.getStridesAndOffset(strides, offset)))
      return failure();
    auto dataPtr =
        getStridedElementPtr(rewriter, atomicOp.getLoc(), memRefType,
                             adaptor.getMemref(), adaptor.getIndices());
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
    BaseMemRefType sourceTy = extractOp.getSource().getType();

    Value alignedPtr;
    if (sourceTy.hasRank()) {
      MemRefDescriptor desc(adaptor.getSource());
      alignedPtr = desc.alignedPtr(rewriter, extractOp->getLoc());
    } else {
      auto elementPtrTy = LLVM::LLVMPointerType::get(
          rewriter.getContext(), sourceTy.getMemorySpaceAsInt());

      UnrankedMemRefDescriptor desc(adaptor.getSource());
      Value descPtr = desc.memRefDescPtr(rewriter, extractOp->getLoc());

      alignedPtr = UnrankedMemRefDescriptor::alignedPtr(
          rewriter, extractOp->getLoc(), *getTypeConverter(), descPtr,
          elementPtrTy);
    }

    rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(
        extractOp, getTypeConverter()->getIndexType(), alignedPtr);
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

    auto sourceMemRefType = cast<MemRefType>(source.getType());
    int64_t rank = sourceMemRefType.getRank();
    SmallVector<Value> results;
    results.reserve(2 + rank * 2);

    // Base buffer.
    Value baseBuffer = sourceMemRef.allocatedPtr(rewriter, loc);
    Value alignedBuffer = sourceMemRef.alignedPtr(rewriter, loc);
    MemRefDescriptor dstMemRef = MemRefDescriptor::fromStaticShape(
        rewriter, loc, *getTypeConverter(),
        cast<MemRefType>(extractStridedMetadataOp.getBaseBuffer().getType()),
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
    const LLVMTypeConverter &converter, RewritePatternSet &patterns,
    SymbolTableCollection *symbolTables) {
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
      GetGlobalMemrefOpLowering,
      LoadOpLowering,
      MemRefCastOpLowering,
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
  patterns.add<GlobalMemrefOpLowering, MemRefCopyOpLowering>(converter,
                                                             symbolTables);
  auto allocLowering = converter.getOptions().allocLowering;
  if (allocLowering == LowerToLLVMOptions::AllocLowering::AlignedAlloc)
    patterns.add<AlignedAllocOpLowering, DeallocOpLowering>(converter,
                                                            symbolTables);
  else if (allocLowering == LowerToLLVMOptions::AllocLowering::Malloc)
    patterns.add<AllocOpLowering, DeallocOpLowering>(converter, symbolTables);
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

    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    LLVMTypeConverter typeConverter(&getContext(), options,
                                    &dataLayoutAnalysis);
    RewritePatternSet patterns(&getContext());
    SymbolTableCollection symbolTables;
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns,
                                                   &symbolTables);
    LLVMConversionTarget target(getContext());
    target.addLegalOp<func::FuncOp>();
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};

/// Implement the interface to convert MemRef to LLVM.
struct MemRefToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  }
};

} // namespace

void mlir::registerConvertMemRefToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
    dialect->addInterfaces<MemRefToLLVMDialectInterface>();
  });
}
