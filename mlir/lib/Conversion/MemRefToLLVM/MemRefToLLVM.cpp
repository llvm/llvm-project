//===- MemRefToLLVM.cpp - MemRef to LLVM dialect conversion ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "../PassDetail.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/AllocLikeConversion.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"

using namespace mlir;

namespace {

struct AllocOpLowering : public AllocLikeOpLLVMLowering {
  AllocOpLowering(LLVMTypeConverter &converter)
      : AllocLikeOpLLVMLowering(memref::AllocOp::getOperationName(),
                                converter) {}

  std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          Operation *op) const override {
    // Heap allocations.
    memref::AllocOp allocOp = cast<memref::AllocOp>(op);
    MemRefType memRefType = allocOp.getType();

    Value alignment;
    if (auto alignmentAttr = allocOp.alignment()) {
      alignment = createIndexConstant(rewriter, loc, *alignmentAttr);
    } else if (!memRefType.getElementType().isSignlessIntOrIndexOrFloat()) {
      // In the case where no alignment is specified, we may want to override
      // `malloc's` behavior. `malloc` typically aligns at the size of the
      // biggest scalar on a target HW. For non-scalars, use the natural
      // alignment of the LLVM type given by the LLVM DataLayout.
      alignment = getSizeInBytes(loc, memRefType.getElementType(), rewriter);
    }

    if (alignment) {
      // Adjust the allocation size to consider alignment.
      sizeBytes = rewriter.create<LLVM::AddOp>(loc, sizeBytes, alignment);
    }

    // Allocate the underlying buffer and store a pointer to it in the MemRef
    // descriptor.
    Type elementPtrType = this->getElementPtrType(memRefType);
    auto allocFuncOp = LLVM::lookupOrCreateMallocFn(
        allocOp->getParentOfType<ModuleOp>(), getIndexType());
    auto results = createLLVMCall(rewriter, loc, allocFuncOp, {sizeBytes},
                                  getVoidPtrType());
    Value allocatedPtr =
        rewriter.create<LLVM::BitcastOp>(loc, elementPtrType, results[0]);

    Value alignedPtr = allocatedPtr;
    if (alignment) {
      // Compute the aligned type pointer.
      Value allocatedInt =
          rewriter.create<LLVM::PtrToIntOp>(loc, getIndexType(), allocatedPtr);
      Value alignmentInt =
          createAligned(rewriter, loc, allocatedInt, alignment);
      alignedPtr =
          rewriter.create<LLVM::IntToPtrOp>(loc, elementPtrType, alignmentInt);
    }

    return std::make_tuple(allocatedPtr, alignedPtr);
  }
};

struct AlignedAllocOpLowering : public AllocLikeOpLLVMLowering {
  AlignedAllocOpLowering(LLVMTypeConverter &converter)
      : AllocLikeOpLLVMLowering(memref::AllocOp::getOperationName(),
                                converter) {}

  /// Returns the memref's element size in bytes using the data layout active at
  /// `op`.
  // TODO: there are other places where this is used. Expose publicly?
  unsigned getMemRefEltSizeInBytes(MemRefType memRefType, Operation *op) const {
    const DataLayout *layout = &defaultLayout;
    if (const DataLayoutAnalysis *analysis =
            getTypeConverter()->getDataLayoutAnalysis()) {
      layout = &analysis->getAbove(op);
    }
    Type elementType = memRefType.getElementType();
    if (auto memRefElementType = elementType.dyn_cast<MemRefType>())
      return getTypeConverter()->getMemRefDescriptorSize(memRefElementType,
                                                         *layout);
    if (auto memRefElementType = elementType.dyn_cast<UnrankedMemRefType>())
      return getTypeConverter()->getUnrankedMemRefDescriptorSize(
          memRefElementType, *layout);
    return layout->getTypeSize(elementType);
  }

  /// Returns true if the memref size in bytes is known to be a multiple of
  /// factor assuming the data layout active at `op`.
  bool isMemRefSizeMultipleOf(MemRefType type, uint64_t factor,
                              Operation *op) const {
    uint64_t sizeDivisor = getMemRefEltSizeInBytes(type, op);
    for (unsigned i = 0, e = type.getRank(); i < e; i++) {
      if (type.isDynamic(type.getDimSize(i)))
        continue;
      sizeDivisor = sizeDivisor * type.getDimSize(i);
    }
    return sizeDivisor % factor == 0;
  }

  /// Returns the alignment to be used for the allocation call itself.
  /// aligned_alloc requires the allocation size to be a power of two, and the
  /// allocation size to be a multiple of alignment,
  int64_t getAllocationAlignment(memref::AllocOp allocOp) const {
    if (Optional<uint64_t> alignment = allocOp.alignment())
      return *alignment;

    // Whenever we don't have alignment set, we will use an alignment
    // consistent with the element type; since the allocation size has to be a
    // power of two, we will bump to the next power of two if it already isn't.
    auto eltSizeBytes = getMemRefEltSizeInBytes(allocOp.getType(), allocOp);
    return std::max(kMinAlignedAllocAlignment,
                    llvm::PowerOf2Ceil(eltSizeBytes));
  }

  std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          Operation *op) const override {
    // Heap allocations.
    memref::AllocOp allocOp = cast<memref::AllocOp>(op);
    MemRefType memRefType = allocOp.getType();
    int64_t alignment = getAllocationAlignment(allocOp);
    Value allocAlignment = createIndexConstant(rewriter, loc, alignment);

    // aligned_alloc requires size to be a multiple of alignment; we will pad
    // the size to the next multiple if necessary.
    if (!isMemRefSizeMultipleOf(memRefType, alignment, op))
      sizeBytes = createAligned(rewriter, loc, sizeBytes, allocAlignment);

    Type elementPtrType = this->getElementPtrType(memRefType);
    auto allocFuncOp = LLVM::lookupOrCreateAlignedAllocFn(
        allocOp->getParentOfType<ModuleOp>(), getIndexType());
    auto results =
        createLLVMCall(rewriter, loc, allocFuncOp, {allocAlignment, sizeBytes},
                       getVoidPtrType());
    Value allocatedPtr =
        rewriter.create<LLVM::BitcastOp>(loc, elementPtrType, results[0]);

    return std::make_tuple(allocatedPtr, allocatedPtr);
  }

  /// The minimum alignment to use with aligned_alloc (has to be a power of 2).
  static constexpr uint64_t kMinAlignedAllocAlignment = 16UL;

  /// Default layout to use in absence of the corresponding analysis.
  DataLayout defaultLayout;
};

// Out of line definition, required till C++17.
constexpr uint64_t AlignedAllocOpLowering::kMinAlignedAllocAlignment;

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
    auto elementPtrType = this->getElementPtrType(allocaOp.getType());

    auto allocatedElementPtr = rewriter.create<LLVM::AllocaOp>(
        loc, elementPtrType, sizeBytes,
        allocaOp.alignment() ? *allocaOp.alignment() : 0);

    return std::make_tuple(allocatedElementPtr, allocatedElementPtr);
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
      continueBlock = rewriter.createBlock(remainingOpsBlock,
                                           allocaScopeOp.getResultTypes());
      rewriter.create<LLVM::BrOp>(loc, ValueRange(), remainingOpsBlock);
    }

    // Inline body region.
    Block *beforeBody = &allocaScopeOp.bodyRegion().front();
    Block *afterBody = &allocaScopeOp.bodyRegion().back();
    rewriter.inlineRegionBefore(allocaScopeOp.bodyRegion(), continueBlock);

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
        returnOp, returnOp.results(), continueBlock);

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
    Value memref = adaptor.memref();
    unsigned alignment = op.alignment();
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
    auto freeFunc = LLVM::lookupOrCreateFreeFn(op->getParentOfType<ModuleOp>());
    MemRefDescriptor memref(adaptor.memref());
    Value casted = rewriter.create<LLVM::BitcastOp>(
        op.getLoc(), getVoidPtrType(),
        memref.allocatedPtr(rewriter, op.getLoc()));
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange(), SymbolRefAttr::get(freeFunc), casted);
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
    Type operandType = dimOp.source().getType();
    if (operandType.isa<UnrankedMemRefType>()) {
      rewriter.replaceOp(
          dimOp, {extractSizeOfUnrankedMemRef(
                     operandType, dimOp, adaptor.getOperands(), rewriter)});

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
  Value extractSizeOfUnrankedMemRef(Type operandType, memref::DimOp dimOp,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    Location loc = dimOp.getLoc();

    auto unrankedMemRefType = operandType.cast<UnrankedMemRefType>();
    auto scalarMemRefType =
        MemRefType::get({}, unrankedMemRefType.getElementType());
    unsigned addressSpace = unrankedMemRefType.getMemorySpaceAsInt();

    // Extract pointer to the underlying ranked descriptor and bitcast it to a
    // memref<element_type> descriptor pointer to minimize the number of GEP
    // operations.
    UnrankedMemRefDescriptor unrankedDesc(adaptor.source());
    Value underlyingRankedDesc = unrankedDesc.memRefDescPtr(rewriter, loc);
    Value scalarMemRefDescPtr = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(typeConverter->convertType(scalarMemRefType),
                                   addressSpace),
        underlyingRankedDesc);

    // Get pointer to offset field of memref<element_type> descriptor.
    Type indexPtrTy = LLVM::LLVMPointerType::get(
        getTypeConverter()->getIndexType(), addressSpace);
    Value two = rewriter.create<LLVM::ConstantOp>(
        loc, typeConverter->convertType(rewriter.getI32Type()),
        rewriter.getI32IntegerAttr(2));
    Value offsetPtr = rewriter.create<LLVM::GEPOp>(
        loc, indexPtrTy, scalarMemRefDescPtr,
        ValueRange({createIndexConstant(rewriter, loc, 0), two}));

    // The size value that we have to extract can be obtained using GEPop with
    // `dimOp.index() + 1` index argument.
    Value idxPlusOne = rewriter.create<LLVM::AddOp>(
        loc, createIndexConstant(rewriter, loc, 1), adaptor.index());
    Value sizePtr = rewriter.create<LLVM::GEPOp>(loc, indexPtrTy, offsetPtr,
                                                 ValueRange({idxPlusOne}));
    return rewriter.create<LLVM::LoadOp>(loc, sizePtr);
  }

  Optional<int64_t> getConstantDimIndex(memref::DimOp dimOp) const {
    if (Optional<int64_t> idx = dimOp.getConstantIndex())
      return idx;

    if (auto constantOp = dimOp.index().getDefiningOp<LLVM::ConstantOp>())
      return constantOp.getValue()
          .cast<IntegerAttr>()
          .getValue()
          .getSExtValue();

    return llvm::None;
  }

  Value extractSizeOfRankedMemRef(Type operandType, memref::DimOp dimOp,
                                  OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    Location loc = dimOp.getLoc();

    // Take advantage if index is constant.
    MemRefType memRefType = operandType.cast<MemRefType>();
    if (Optional<int64_t> index = getConstantDimIndex(dimOp)) {
      int64_t i = index.getValue();
      if (memRefType.isDynamicDim(i)) {
        // extract dynamic size from the memref descriptor.
        MemRefDescriptor descriptor(adaptor.source());
        return descriptor.size(rewriter, loc, i);
      }
      // Use constant for static size.
      int64_t dimSize = memRefType.getDimSize(i);
      return createIndexConstant(rewriter, loc, dimSize);
    }
    Value index = adaptor.index();
    int64_t rank = memRefType.getRank();
    MemRefDescriptor memrefDescriptor(adaptor.source());
    return memrefDescriptor.size(rewriter, loc, index, rank);
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
    MemRefType type = global.type();
    if (!isConvertibleAndHasIdentityMaps(type))
      return failure();

    Type arrayTy = convertGlobalMemrefTypeToLLVM(type, *getTypeConverter());

    LLVM::Linkage linkage =
        global.isPublic() ? LLVM::Linkage::External : LLVM::Linkage::Private;

    Attribute initialValue = nullptr;
    if (!global.isExternal() && !global.isUninitialized()) {
      auto elementsAttr = global.initial_value()->cast<ElementsAttr>();
      initialValue = elementsAttr;

      // For scalar memrefs, the global variable created is of the element type,
      // so unpack the elements attribute to extract the value.
      if (type.getRank() == 0)
        initialValue = elementsAttr.getSplatValue<Attribute>();
    }

    uint64_t alignment = global.alignment().getValueOr(0);

    auto newGlobal = rewriter.replaceOpWithNewOp<LLVM::GlobalOp>(
        global, arrayTy, global.constant(), linkage, global.sym_name(),
        initialValue, alignment, type.getMemorySpaceAsInt());
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
    MemRefType type = getGlobalOp.result().getType().cast<MemRefType>();
    unsigned memSpace = type.getMemorySpaceAsInt();

    Type arrayTy = convertGlobalMemrefTypeToLLVM(type, *getTypeConverter());
    auto addressOf = rewriter.create<LLVM::AddressOfOp>(
        loc, LLVM::LLVMPointerType::get(arrayTy, memSpace), getGlobalOp.name());

    // Get the address of the first element in the array by creating a GEP with
    // the address of the GV as the base, and (rank + 1) number of 0 indices.
    Type elementType = typeConverter->convertType(type.getElementType());
    Type elementPtrType = LLVM::LLVMPointerType::get(elementType, memSpace);

    SmallVector<Value, 4> operands = {addressOf};
    operands.insert(operands.end(), type.getRank() + 1,
                    createIndexConstant(rewriter, loc, 0));
    auto gep = rewriter.create<LLVM::GEPOp>(loc, elementPtrType, operands);

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

// Common base for load and store operations on MemRefs. Restricts the match
// to supported MemRef types. Provides functionality to emit code accessing a
// specific element of the underlying data buffer.
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

// Load operation is lowered to obtaining a pointer to the indexed element
// and loading it.
struct LoadOpLowering : public LoadStoreOpLowering<memref::LoadOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(memref::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = loadOp.getMemRefType();

    Value dataPtr = getStridedElementPtr(
        loadOp.getLoc(), type, adaptor.memref(), adaptor.indices(), rewriter);
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(loadOp, dataPtr);
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

    Value dataPtr = getStridedElementPtr(op.getLoc(), type, adaptor.memref(),
                                         adaptor.indices(), rewriter);
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.value(), dataPtr);
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

    Value dataPtr = getStridedElementPtr(loc, type, adaptor.memref(),
                                         adaptor.indices(), rewriter);

    // Replace with llvm.prefetch.
    auto llvmI32Type = typeConverter->convertType(rewriter.getIntegerType(32));
    auto isWrite = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32Type, rewriter.getI32IntegerAttr(prefetchOp.isWrite()));
    auto localityHint = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32Type,
        rewriter.getI32IntegerAttr(prefetchOp.localityHint()));
    auto isData = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32Type, rewriter.getI32IntegerAttr(prefetchOp.isDataCache()));

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
    Type operandType = op.memref().getType();
    if (auto unrankedMemRefType = operandType.dyn_cast<UnrankedMemRefType>()) {
      UnrankedMemRefDescriptor desc(adaptor.memref());
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
      return rewriter.replaceOp(memRefCastOp, {adaptor.source()});

    if (srcType.isa<MemRefType>() && dstType.isa<UnrankedMemRefType>()) {
      // Casting ranked to unranked memref type
      // Set the rank in the destination from the memref type
      // Allocate space on the stack and copy the src memref descriptor
      // Set the ptr in the destination to the stack space
      auto srcMemRefType = srcType.cast<MemRefType>();
      int64_t rank = srcMemRefType.getRank();
      // ptr = AllocaOp sizeof(MemRefDescriptor)
      auto ptr = getTypeConverter()->promoteOneMemRefDescriptor(
          loc, adaptor.source(), rewriter);
      // voidptr = BitCastOp srcType* to void*
      auto voidPtr =
          rewriter.create<LLVM::BitcastOp>(loc, getVoidPtrType(), ptr)
              .getResult();
      // rank = ConstantOp srcRank
      auto rankVal = rewriter.create<LLVM::ConstantOp>(
          loc, typeConverter->convertType(rewriter.getIntegerType(64)),
          rewriter.getI64IntegerAttr(rank));
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
      UnrankedMemRefDescriptor memRefDesc(adaptor.source());
      // ptr = ExtractValueOp src, 1
      auto ptr = memRefDesc.memRefDescPtr(rewriter, loc);
      // castPtr = BitCastOp i8* to structTy*
      auto castPtr =
          rewriter
              .create<LLVM::BitcastOp>(
                  loc, LLVM::LLVMPointerType::get(targetStructType), ptr)
              .getResult();
      // struct = LoadOp castPtr
      auto loadOp = rewriter.create<LLVM::LoadOp>(loc, castPtr);
      rewriter.replaceOp(memRefCastOp, loadOp.getResult());
    } else {
      llvm_unreachable("Unsupported unranked memref to unranked memref cast");
    }
  }
};

struct MemRefCopyOpLowering : public ConvertOpToLLVMPattern<memref::CopyOp> {
  using ConvertOpToLLVMPattern<memref::CopyOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto srcType = op.source().getType().cast<BaseMemRefType>();
    auto targetType = op.target().getType().cast<BaseMemRefType>();

    // First make sure we have an unranked memref descriptor representation.
    auto makeUnranked = [&, this](Value ranked, BaseMemRefType type) {
      auto rank = rewriter.create<LLVM::ConstantOp>(
          loc, getIndexType(), rewriter.getIndexAttr(type.getRank()));
      auto *typeConverter = getTypeConverter();
      auto ptr =
          typeConverter->promoteOneMemRefDescriptor(loc, ranked, rewriter);
      auto voidPtr =
          rewriter.create<LLVM::BitcastOp>(loc, getVoidPtrType(), ptr)
              .getResult();
      auto unrankedType =
          UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace());
      return UnrankedMemRefDescriptor::pack(rewriter, loc, *typeConverter,
                                            unrankedType,
                                            ValueRange{rank, voidPtr});
    };

    Value unrankedSource = srcType.hasRank()
                               ? makeUnranked(adaptor.source(), srcType)
                               : adaptor.source();
    Value unrankedTarget = targetType.hasRank()
                               ? makeUnranked(adaptor.target(), targetType)
                               : adaptor.target();

    // Now promote the unranked descriptors to the stack.
    auto one = rewriter.create<LLVM::ConstantOp>(loc, getIndexType(),
                                                 rewriter.getIndexAttr(1));
    auto promote = [&](Value desc) {
      auto ptrType = LLVM::LLVMPointerType::get(desc.getType());
      auto allocated =
          rewriter.create<LLVM::AllocaOp>(loc, ptrType, ValueRange{one});
      rewriter.create<LLVM::StoreOp>(loc, desc, allocated);
      return allocated;
    };

    auto sourcePtr = promote(unrankedSource);
    auto targetPtr = promote(unrankedTarget);

    auto elemSize = rewriter.create<LLVM::ConstantOp>(
        loc, getIndexType(),
        rewriter.getIndexAttr(srcType.getElementTypeBitWidth() / 8));
    auto copyFn = LLVM::lookupOrCreateMemRefCopyFn(
        op->getParentOfType<ModuleOp>(), getIndexType(), sourcePtr.getType());
    rewriter.create<LLVM::CallOp>(loc, copyFn,
                                  ValueRange{elemSize, sourcePtr, targetPtr});
    rewriter.eraseOp(op);

    return success();
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

  unsigned memorySpace =
      operandType.cast<UnrankedMemRefType>().getMemorySpaceAsInt();
  Type elementType = operandType.cast<UnrankedMemRefType>().getElementType();
  Type llvmElementType = typeConverter.convertType(elementType);
  Type elementPtrPtrType = LLVM::LLVMPointerType::get(
      LLVM::LLVMPointerType::get(llvmElementType, memorySpace));

  // Extract pointer to the underlying ranked memref descriptor and cast it to
  // ElemType**.
  UnrankedMemRefDescriptor unrankedDesc(convertedOperand);
  Value underlyingDescPtr = unrankedDesc.memRefDescPtr(rewriter, loc);

  *allocatedPtr = UnrankedMemRefDescriptor::allocatedPtr(
      rewriter, loc, underlyingDescPtr, elementPtrPtrType);
  *alignedPtr = UnrankedMemRefDescriptor::alignedPtr(
      rewriter, loc, typeConverter, underlyingDescPtr, elementPtrPtrType);
  if (offset != nullptr) {
    *offset = UnrankedMemRefDescriptor::offset(
        rewriter, loc, typeConverter, underlyingDescPtr, elementPtrPtrType);
  }
}

struct MemRefReinterpretCastOpLowering
    : public ConvertOpToLLVMPattern<memref::ReinterpretCastOp> {
  using ConvertOpToLLVMPattern<
      memref::ReinterpretCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type srcType = castOp.source().getType();

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
                             castOp.source(), adaptor.source(), &allocatedPtr,
                             &alignedPtr);
    desc.setAllocatedPtr(rewriter, loc, allocatedPtr);
    desc.setAlignedPtr(rewriter, loc, alignedPtr);

    // Set offset.
    if (castOp.isDynamicOffset(0))
      desc.setOffset(rewriter, loc, adaptor.offsets()[0]);
    else
      desc.setConstantOffset(rewriter, loc, castOp.getStaticOffset(0));

    // Set sizes and strides.
    unsigned dynSizeId = 0;
    unsigned dynStrideId = 0;
    for (unsigned i = 0, e = targetMemRefType.getRank(); i < e; ++i) {
      if (castOp.isDynamicSize(i))
        desc.setSize(rewriter, loc, i, adaptor.sizes()[dynSizeId++]);
      else
        desc.setConstantSize(rewriter, loc, i, castOp.getStaticSize(i));

      if (castOp.isDynamicStride(i))
        desc.setStride(rewriter, loc, i, adaptor.strides()[dynStrideId++]);
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
    Type srcType = reshapeOp.source().getType();

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
    // Conversion for statically-known shape args is performed via
    // `memref_reinterpret_cast`.
    auto shapeMemRefType = reshapeOp.shape().getType().cast<MemRefType>();
    if (shapeMemRefType.hasStaticShape())
      return failure();

    // The shape is a rank-1 tensor with unknown length.
    Location loc = reshapeOp.getLoc();
    MemRefDescriptor shapeDesc(adaptor.shape());
    Value resultRank = shapeDesc.size(rewriter, loc, 0);

    // Extract address space and element type.
    auto targetType =
        reshapeOp.getResult().getType().cast<UnrankedMemRefType>();
    unsigned addressSpace = targetType.getMemorySpaceAsInt();
    Type elementType = targetType.getElementType();

    // Create the unranked memref descriptor that holds the ranked one. The
    // inner descriptor is allocated on stack.
    auto targetDesc = UnrankedMemRefDescriptor::undef(
        rewriter, loc, typeConverter->convertType(targetType));
    targetDesc.setRank(rewriter, loc, resultRank);
    SmallVector<Value, 4> sizes;
    UnrankedMemRefDescriptor::computeSizes(rewriter, loc, *getTypeConverter(),
                                           targetDesc, sizes);
    Value underlyingDescPtr = rewriter.create<LLVM::AllocaOp>(
        loc, getVoidPtrType(), sizes.front(), llvm::None);
    targetDesc.setMemRefDescPtr(rewriter, loc, underlyingDescPtr);

    // Extract pointers and offset from the source memref.
    Value allocatedPtr, alignedPtr, offset;
    extractPointersAndOffset(loc, rewriter, *getTypeConverter(),
                             reshapeOp.source(), adaptor.source(),
                             &allocatedPtr, &alignedPtr, &offset);

    // Set pointers and offset.
    Type llvmElementType = typeConverter->convertType(elementType);
    auto elementPtrPtrType = LLVM::LLVMPointerType::get(
        LLVM::LLVMPointerType::get(llvmElementType, addressSpace));
    UnrankedMemRefDescriptor::setAllocatedPtr(rewriter, loc, underlyingDescPtr,
                                              elementPtrPtrType, allocatedPtr);
    UnrankedMemRefDescriptor::setAlignedPtr(rewriter, loc, *getTypeConverter(),
                                            underlyingDescPtr,
                                            elementPtrPtrType, alignedPtr);
    UnrankedMemRefDescriptor::setOffset(rewriter, loc, *getTypeConverter(),
                                        underlyingDescPtr, elementPtrPtrType,
                                        offset);

    // Use the offset pointer as base for further addressing. Copy over the new
    // shape and compute strides. For this, we create a loop from rank-1 to 0.
    Value targetSizesBase = UnrankedMemRefDescriptor::sizeBasePtr(
        rewriter, loc, *getTypeConverter(), underlyingDescPtr,
        elementPtrPtrType);
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
                                            {indexType, indexType});

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
    Type llvmIndexPtrType = LLVM::LLVMPointerType::get(indexType);
    Value sizeLoadGep = rewriter.create<LLVM::GEPOp>(
        loc, llvmIndexPtrType, shapeOperandPtr, ValueRange{indexArg});
    Value size = rewriter.create<LLVM::LoadOp>(loc, sizeLoadGep);
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
    rewriter.create<LLVM::CondBrOp>(loc, pred, bodyBlock, llvm::None, remainder,
                                    llvm::None);

    // Reset position to beginning of new remainder block.
    rewriter.setInsertionPointToStart(remainder);

    *descriptor = targetDesc;
    return success();
  }
};

/// Helper function to convert a vector of `OpFoldResult`s into a vector of
/// `Value`s.
static SmallVector<Value> getAsValues(OpBuilder &b, Location loc,
                                      Type &llvmIndexType,
                                      ArrayRef<OpFoldResult> valueOrAttrVec) {
  return llvm::to_vector<4>(
      llvm::map_range(valueOrAttrVec, [&](OpFoldResult value) -> Value {
        if (auto attr = value.dyn_cast<Attribute>())
          return b.create<LLVM::ConstantOp>(loc, llvmIndexType, attr);
        return value.get<Value>();
      }));
}

/// Compute a map that for a given dimension of the expanded type gives the
/// dimension in the collapsed type it maps to. Essentially its the inverse of
/// the `reassocation` maps.
static DenseMap<int64_t, int64_t>
getExpandedDimToCollapsedDimMap(ArrayRef<ReassociationIndices> reassociation) {
  llvm::DenseMap<int64_t, int64_t> expandedDimToCollapsedDim;
  for (auto &en : enumerate(reassociation)) {
    for (auto dim : en.value())
      expandedDimToCollapsedDim[dim] = en.index();
  }
  return expandedDimToCollapsedDim;
}

static OpFoldResult
getExpandedOutputDimSize(OpBuilder &b, Location loc, Type &llvmIndexType,
                         int64_t outDimIndex, ArrayRef<int64_t> outStaticShape,
                         MemRefDescriptor &inDesc,
                         ArrayRef<int64_t> inStaticShape,
                         ArrayRef<ReassociationIndices> reassocation,
                         DenseMap<int64_t, int64_t> &outDimToInDimMap) {
  int64_t outDimSize = outStaticShape[outDimIndex];
  if (!ShapedType::isDynamic(outDimSize))
    return b.getIndexAttr(outDimSize);

  // Calculate the multiplication of all the out dim sizes except the
  // current dim.
  int64_t inDimIndex = outDimToInDimMap[outDimIndex];
  int64_t otherDimSizesMul = 1;
  for (auto otherDimIndex : reassocation[inDimIndex]) {
    if (otherDimIndex == static_cast<unsigned>(outDimIndex))
      continue;
    int64_t otherDimSize = outStaticShape[otherDimIndex];
    assert(!ShapedType::isDynamic(otherDimSize) &&
           "single dimension cannot be expanded into multiple dynamic "
           "dimensions");
    otherDimSizesMul *= otherDimSize;
  }

  // outDimSize = inDimSize / otherOutDimSizesMul
  int64_t inDimSize = inStaticShape[inDimIndex];
  Value inDimSizeDynamic =
      ShapedType::isDynamic(inDimSize)
          ? inDesc.size(b, loc, inDimIndex)
          : b.create<LLVM::ConstantOp>(loc, llvmIndexType,
                                       b.getIndexAttr(inDimSize));
  Value outDimSizeDynamic = b.create<LLVM::SDivOp>(
      loc, inDimSizeDynamic,
      b.create<LLVM::ConstantOp>(loc, llvmIndexType,
                                 b.getIndexAttr(otherDimSizesMul)));
  return outDimSizeDynamic;
}

static OpFoldResult getCollapsedOutputDimSize(
    OpBuilder &b, Location loc, Type &llvmIndexType, int64_t outDimIndex,
    int64_t outDimSize, ArrayRef<int64_t> inStaticShape,
    MemRefDescriptor &inDesc, ArrayRef<ReassociationIndices> reassocation) {
  if (!ShapedType::isDynamic(outDimSize))
    return b.getIndexAttr(outDimSize);

  Value c1 = b.create<LLVM::ConstantOp>(loc, llvmIndexType, b.getIndexAttr(1));
  Value outDimSizeDynamic = c1;
  for (auto inDimIndex : reassocation[outDimIndex]) {
    int64_t inDimSize = inStaticShape[inDimIndex];
    Value inDimSizeDynamic =
        ShapedType::isDynamic(inDimSize)
            ? inDesc.size(b, loc, inDimIndex)
            : b.create<LLVM::ConstantOp>(loc, llvmIndexType,
                                         b.getIndexAttr(inDimSize));
    outDimSizeDynamic =
        b.create<LLVM::MulOp>(loc, outDimSizeDynamic, inDimSizeDynamic);
  }
  return outDimSizeDynamic;
}

static SmallVector<OpFoldResult, 4>
getCollapsedOutputShape(OpBuilder &b, Location loc, Type &llvmIndexType,
                        ArrayRef<ReassociationIndices> reassocation,
                        ArrayRef<int64_t> inStaticShape,
                        MemRefDescriptor &inDesc,
                        ArrayRef<int64_t> outStaticShape) {
  return llvm::to_vector<4>(llvm::map_range(
      llvm::seq<int64_t>(0, outStaticShape.size()), [&](int64_t outDimIndex) {
        return getCollapsedOutputDimSize(b, loc, llvmIndexType, outDimIndex,
                                         outStaticShape[outDimIndex],
                                         inStaticShape, inDesc, reassocation);
      }));
}

static SmallVector<OpFoldResult, 4>
getExpandedOutputShape(OpBuilder &b, Location loc, Type &llvmIndexType,
                       ArrayRef<ReassociationIndices> reassocation,
                       ArrayRef<int64_t> inStaticShape,
                       MemRefDescriptor &inDesc,
                       ArrayRef<int64_t> outStaticShape) {
  DenseMap<int64_t, int64_t> outDimToInDimMap =
      getExpandedDimToCollapsedDimMap(reassocation);
  return llvm::to_vector<4>(llvm::map_range(
      llvm::seq<int64_t>(0, outStaticShape.size()), [&](int64_t outDimIndex) {
        return getExpandedOutputDimSize(b, loc, llvmIndexType, outDimIndex,
                                        outStaticShape, inDesc, inStaticShape,
                                        reassocation, outDimToInDimMap);
      }));
}

static SmallVector<Value>
getDynamicOutputShape(OpBuilder &b, Location loc, Type &llvmIndexType,
                      ArrayRef<ReassociationIndices> reassocation,
                      ArrayRef<int64_t> inStaticShape, MemRefDescriptor &inDesc,
                      ArrayRef<int64_t> outStaticShape) {
  return outStaticShape.size() < inStaticShape.size()
             ? getAsValues(b, loc, llvmIndexType,
                           getCollapsedOutputShape(b, loc, llvmIndexType,
                                                   reassocation, inStaticShape,
                                                   inDesc, outStaticShape))
             : getAsValues(b, loc, llvmIndexType,
                           getExpandedOutputShape(b, loc, llvmIndexType,
                                                  reassocation, inStaticShape,
                                                  inDesc, outStaticShape));
}

// ReshapeOp creates a new view descriptor of the proper rank.
// For now, the only conversion supported is for target MemRef with static sizes
// and strides.
template <typename ReshapeOp>
class ReassociatingReshapeOpConversion
    : public ConvertOpToLLVMPattern<ReshapeOp> {
public:
  using ConvertOpToLLVMPattern<ReshapeOp>::ConvertOpToLLVMPattern;
  using ReshapeOpAdaptor = typename ReshapeOp::Adaptor;

  LogicalResult
  matchAndRewrite(ReshapeOp reshapeOp, typename ReshapeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType dstType = reshapeOp.getResultType();
    MemRefType srcType = reshapeOp.getSrcType();
    if (!srcType.getLayout().isIdentity() ||
        !dstType.getLayout().isIdentity()) {
      return rewriter.notifyMatchFailure(reshapeOp,
                                         "only empty layout map is supported");
    }

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    if (failed(getStridesAndOffset(dstType, strides, offset))) {
      return rewriter.notifyMatchFailure(
          reshapeOp, "failed to get stride and offset exprs");
    }

    MemRefDescriptor srcDesc(adaptor.src());
    Location loc = reshapeOp->getLoc();
    auto dstDesc = MemRefDescriptor::undef(
        rewriter, loc, this->typeConverter->convertType(dstType));
    dstDesc.setAllocatedPtr(rewriter, loc, srcDesc.allocatedPtr(rewriter, loc));
    dstDesc.setAlignedPtr(rewriter, loc, srcDesc.alignedPtr(rewriter, loc));
    dstDesc.setOffset(rewriter, loc, srcDesc.offset(rewriter, loc));

    ArrayRef<int64_t> srcStaticShape = srcType.getShape();
    ArrayRef<int64_t> dstStaticShape = dstType.getShape();
    Type llvmIndexType =
        this->typeConverter->convertType(rewriter.getIndexType());
    SmallVector<Value> dstShape = getDynamicOutputShape(
        rewriter, loc, llvmIndexType, reshapeOp.getReassociationIndices(),
        srcStaticShape, srcDesc, dstStaticShape);
    for (auto &en : llvm::enumerate(dstShape))
      dstDesc.setSize(rewriter, loc, en.index(), en.value());

    auto isStaticStride = [](int64_t stride) {
      return !ShapedType::isDynamicStrideOrOffset(stride);
    };
    if (llvm::all_of(strides, isStaticStride)) {
      for (auto &en : llvm::enumerate(strides))
        dstDesc.setConstantStride(rewriter, loc, en.index(), en.value());
    } else {
      Value c1 = rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType,
                                                   rewriter.getIndexAttr(1));
      Value stride = c1;
      for (auto dimIndex :
           llvm::reverse(llvm::seq<int64_t>(0, dstShape.size()))) {
        dstDesc.setStride(rewriter, loc, dimIndex, stride);
        stride = rewriter.create<LLVM::MulOp>(loc, dstShape[dimIndex], stride);
      }
    }
    rewriter.replaceOp(reshapeOp, {dstDesc});
    return success();
  }
};

/// Conversion pattern that transforms a subview op into:
///   1. An `llvm.mlir.undef` operation to create a memref descriptor
///   2. Updates to the descriptor to introduce the data ptr, offset, size
///      and stride.
/// The subview op is replaced by the descriptor.
struct SubViewOpLowering : public ConvertOpToLLVMPattern<memref::SubViewOp> {
  using ConvertOpToLLVMPattern<memref::SubViewOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::SubViewOp subViewOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = subViewOp.getLoc();

    auto sourceMemRefType = subViewOp.source().getType().cast<MemRefType>();
    auto sourceElementTy =
        typeConverter->convertType(sourceMemRefType.getElementType());

    auto viewMemRefType = subViewOp.getType();
    auto inferredType = memref::SubViewOp::inferResultType(
                            subViewOp.getSourceType(),
                            extractFromI64ArrayAttr(subViewOp.static_offsets()),
                            extractFromI64ArrayAttr(subViewOp.static_sizes()),
                            extractFromI64ArrayAttr(subViewOp.static_strides()))
                            .cast<MemRefType>();
    auto targetElementTy =
        typeConverter->convertType(viewMemRefType.getElementType());
    auto targetDescTy = typeConverter->convertType(viewMemRefType);
    if (!sourceElementTy || !targetDescTy || !targetElementTy ||
        !LLVM::isCompatibleType(sourceElementTy) ||
        !LLVM::isCompatibleType(targetElementTy) ||
        !LLVM::isCompatibleType(targetDescTy))
      return failure();

    // Extract the offset and strides from the type.
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides = getStridesAndOffset(inferredType, strides, offset);
    if (failed(successStrides))
      return failure();

    // Create the descriptor.
    if (!LLVM::isCompatibleType(adaptor.getOperands().front().getType()))
      return failure();
    MemRefDescriptor sourceMemRef(adaptor.getOperands().front());
    auto targetMemRef = MemRefDescriptor::undef(rewriter, loc, targetDescTy);

    // Copy the buffer pointer from the old descriptor to the new one.
    Value extracted = sourceMemRef.allocatedPtr(rewriter, loc);
    Value bitcastPtr = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(targetElementTy,
                                   viewMemRefType.getMemorySpaceAsInt()),
        extracted);
    targetMemRef.setAllocatedPtr(rewriter, loc, bitcastPtr);

    // Copy the aligned pointer from the old descriptor to the new one.
    extracted = sourceMemRef.alignedPtr(rewriter, loc);
    bitcastPtr = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(targetElementTy,
                                   viewMemRefType.getMemorySpaceAsInt()),
        extracted);
    targetMemRef.setAlignedPtr(rewriter, loc, bitcastPtr);

    size_t inferredShapeRank = inferredType.getRank();
    size_t resultShapeRank = viewMemRefType.getRank();

    // Extract strides needed to compute offset.
    SmallVector<Value, 4> strideValues;
    strideValues.reserve(inferredShapeRank);
    for (unsigned i = 0; i < inferredShapeRank; ++i)
      strideValues.push_back(sourceMemRef.stride(rewriter, loc, i));

    // Offset.
    auto llvmIndexType = typeConverter->convertType(rewriter.getIndexType());
    if (!ShapedType::isDynamicStrideOrOffset(offset)) {
      targetMemRef.setConstantOffset(rewriter, loc, offset);
    } else {
      Value baseOffset = sourceMemRef.offset(rewriter, loc);
      // `inferredShapeRank` may be larger than the number of offset operands
      // because of trailing semantics. In this case, the offset is guaranteed
      // to be interpreted as 0 and we can just skip the extra dimensions.
      for (unsigned i = 0, e = std::min(inferredShapeRank,
                                        subViewOp.getMixedOffsets().size());
           i < e; ++i) {
        Value offset =
            // TODO: need OpFoldResult ODS adaptor to clean this up.
            subViewOp.isDynamicOffset(i)
                ? adaptor.getOperands()[subViewOp.getIndexOfDynamicOffset(i)]
                : rewriter.create<LLVM::ConstantOp>(
                      loc, llvmIndexType,
                      rewriter.getI64IntegerAttr(subViewOp.getStaticOffset(i)));
        Value mul = rewriter.create<LLVM::MulOp>(loc, offset, strideValues[i]);
        baseOffset = rewriter.create<LLVM::AddOp>(loc, baseOffset, mul);
      }
      targetMemRef.setOffset(rewriter, loc, baseOffset);
    }

    // Update sizes and strides.
    SmallVector<OpFoldResult> mixedSizes = subViewOp.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = subViewOp.getMixedStrides();
    assert(mixedSizes.size() == mixedStrides.size() &&
           "expected sizes and strides of equal length");
    llvm::SmallDenseSet<unsigned> unusedDims = subViewOp.getDroppedDims();
    for (int i = inferredShapeRank - 1, j = resultShapeRank - 1;
         i >= 0 && j >= 0; --i) {
      if (unusedDims.contains(i))
        continue;

      // `i` may overflow subViewOp.getMixedSizes because of trailing semantics.
      // In this case, the size is guaranteed to be interpreted as Dim and the
      // stride as 1.
      Value size, stride;
      if (static_cast<unsigned>(i) >= mixedSizes.size()) {
        // If the static size is available, use it directly. This is similar to
        // the folding of dim(constant-op) but removes the need for dim to be
        // aware of LLVM constants and for this pass to be aware of std
        // constants.
        int64_t staticSize =
            subViewOp.source().getType().cast<MemRefType>().getShape()[i];
        if (staticSize != ShapedType::kDynamicSize) {
          size = rewriter.create<LLVM::ConstantOp>(
              loc, llvmIndexType, rewriter.getI64IntegerAttr(staticSize));
        } else {
          Value pos = rewriter.create<LLVM::ConstantOp>(
              loc, llvmIndexType, rewriter.getI64IntegerAttr(i));
          Value dim =
              rewriter.create<memref::DimOp>(loc, subViewOp.source(), pos);
          auto cast = rewriter.create<UnrealizedConversionCastOp>(
              loc, llvmIndexType, dim);
          size = cast.getResult(0);
        }
        stride = rewriter.create<LLVM::ConstantOp>(
            loc, llvmIndexType, rewriter.getI64IntegerAttr(1));
      } else {
        // TODO: need OpFoldResult ODS adaptor to clean this up.
        size =
            subViewOp.isDynamicSize(i)
                ? adaptor.getOperands()[subViewOp.getIndexOfDynamicSize(i)]
                : rewriter.create<LLVM::ConstantOp>(
                      loc, llvmIndexType,
                      rewriter.getI64IntegerAttr(subViewOp.getStaticSize(i)));
        if (!ShapedType::isDynamicStrideOrOffset(strides[i])) {
          stride = rewriter.create<LLVM::ConstantOp>(
              loc, llvmIndexType, rewriter.getI64IntegerAttr(strides[i]));
        } else {
          stride =
              subViewOp.isDynamicStride(i)
                  ? adaptor.getOperands()[subViewOp.getIndexOfDynamicStride(i)]
                  : rewriter.create<LLVM::ConstantOp>(
                        loc, llvmIndexType,
                        rewriter.getI64IntegerAttr(
                            subViewOp.getStaticStride(i)));
          stride = rewriter.create<LLVM::MulOp>(loc, stride, strideValues[i]);
        }
      }
      targetMemRef.setSize(rewriter, loc, j, size);
      targetMemRef.setStride(rewriter, loc, j, stride);
      j--;
    }

    rewriter.replaceOp(subViewOp, {targetMemRef});
    return success();
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
    MemRefDescriptor viewMemRef(adaptor.in());

    // No permutation, early exit.
    if (transposeOp.permutation().isIdentity())
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
    for (auto en : llvm::enumerate(transposeOp.permutation().getResults())) {
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
    unsigned nDynamic = llvm::count_if(shape.take_front(idx), [](int64_t v) {
      return ShapedType::isDynamic(v);
    });
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
    if (!MemRefType::isDynamicStrideOrOffset(strides[idx]))
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

    // Create the descriptor.
    MemRefDescriptor sourceMemRef(adaptor.source());
    auto targetMemRef = MemRefDescriptor::undef(rewriter, loc, targetDescTy);

    // Field 1: Copy the allocated pointer, used for malloc/free.
    Value allocatedPtr = sourceMemRef.allocatedPtr(rewriter, loc);
    auto srcMemRefType = viewOp.source().getType().cast<MemRefType>();
    Value bitcastPtr = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(targetElementTy,
                                   srcMemRefType.getMemorySpaceAsInt()),
        allocatedPtr);
    targetMemRef.setAllocatedPtr(rewriter, loc, bitcastPtr);

    // Field 2: Copy the actual aligned pointer to payload.
    Value alignedPtr = sourceMemRef.alignedPtr(rewriter, loc);
    alignedPtr = rewriter.create<LLVM::GEPOp>(loc, alignedPtr.getType(),
                                              alignedPtr, adaptor.byte_shift());
    bitcastPtr = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(targetElementTy,
                                   srcMemRefType.getMemorySpaceAsInt()),
        alignedPtr);
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
    if (strides.back() != 1)
      return viewOp.emitWarning("cannot cast to non-contiguous shape"),
             failure();
    Value stride = nullptr, nextSize = nullptr;
    for (int i = viewMemRefType.getRank() - 1; i >= 0; --i) {
      // Update size.
      Value size =
          getSize(rewriter, loc, viewMemRefType.getShape(), adaptor.sizes(), i);
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

} // namespace

void mlir::populateMemRefToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                  RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
      AllocaOpLowering,
      AllocaScopeOpLowering,
      AssumeAlignmentOpLowering,
      DimOpLowering,
      GlobalMemrefOpLowering,
      GetGlobalMemrefOpLowering,
      LoadOpLowering,
      MemRefCastOpLowering,
      MemRefCopyOpLowering,
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
    patterns.add<AlignedAllocOpLowering, DeallocOpLowering>(converter);
  else if (allocLowering == LowerToLLVMOptions::AllocLowering::Malloc)
    patterns.add<AllocOpLowering, DeallocOpLowering>(converter);
}

namespace {
struct MemRefToLLVMPass : public ConvertMemRefToLLVMBase<MemRefToLLVMPass> {
  MemRefToLLVMPass() = default;

  void runOnOperation() override {
    Operation *op = getOperation();
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
    LowerToLLVMOptions options(&getContext(),
                               dataLayoutAnalysis.getAtOrAbove(op));
    options.allocLowering =
        (useAlignedAlloc ? LowerToLLVMOptions::AllocLowering::AlignedAlloc
                         : LowerToLLVMOptions::AllocLowering::Malloc);
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    LLVMTypeConverter typeConverter(&getContext(), options,
                                    &dataLayoutAnalysis);
    RewritePatternSet patterns(&getContext());
    populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
    LLVMConversionTarget target(getContext());
    target.addLegalOp<FuncOp>();
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createMemRefToLLVMPass() {
  return std::make_unique<MemRefToLLVMPass>();
}
