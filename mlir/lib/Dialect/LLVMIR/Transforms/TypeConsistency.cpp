//===- TypeConsistency.cpp - Rewrites to improve type consistency ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/Transforms/TypeConsistency.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace LLVM {
#define GEN_PASS_DEF_LLVMTYPECONSISTENCY
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace mlir

using namespace mlir;
using namespace LLVM;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

/// Checks that a pointer value has a pointee type hint consistent with the
/// expected type. Returns the type it actually hints to if it differs, or
/// nullptr if the type is consistent or impossible to analyze.
static Type isElementTypeInconsistent(Value addr, Type expectedType) {
  auto defOp = dyn_cast_or_null<GetResultPtrElementType>(addr.getDefiningOp());
  if (!defOp)
    return nullptr;

  Type elemType = defOp.getResultPtrElementType();
  if (!elemType)
    return nullptr;

  if (elemType == expectedType)
    return nullptr;

  return elemType;
}

/// Checks that two types are the same or can be bitcast into one another.
static bool areBitcastCompatible(DataLayout &layout, Type lhs, Type rhs) {
  return lhs == rhs || (!isa<LLVMStructType, LLVMArrayType>(lhs) &&
                        !isa<LLVMStructType, LLVMArrayType>(rhs) &&
                        layout.getTypeSize(lhs) == layout.getTypeSize(rhs));
}

//===----------------------------------------------------------------------===//
// AddFieldGetterToStructDirectUse
//===----------------------------------------------------------------------===//

/// Gets the type of the first subelement of `type` if `type` is destructurable,
/// nullptr otherwise.
static Type getFirstSubelementType(Type type) {
  auto destructurable = dyn_cast<DestructurableTypeInterface>(type);
  if (!destructurable)
    return nullptr;

  Type subelementType = destructurable.getTypeAtIndex(
      IntegerAttr::get(IntegerType::get(type.getContext(), 32), 0));
  if (subelementType)
    return subelementType;

  return nullptr;
}

/// Extracts a pointer to the first field of an `elemType` from the address
/// pointer of the provided MemOp, and rewires the MemOp so it uses that pointer
/// instead.
template <class MemOp>
static void insertFieldIndirection(MemOp op, PatternRewriter &rewriter,
                                   Type elemType) {
  PatternRewriter::InsertionGuard guard(rewriter);

  rewriter.setInsertionPointAfterValue(op.getAddr());
  SmallVector<GEPArg> firstTypeIndices{0, 0};

  Value properPtr = rewriter.create<GEPOp>(
      op->getLoc(), LLVM::LLVMPointerType::get(op.getContext()), elemType,
      op.getAddr(), firstTypeIndices);

  rewriter.updateRootInPlace(op,
                             [&]() { op.getAddrMutable().assign(properPtr); });
}

template <>
LogicalResult AddFieldGetterToStructDirectUse<LoadOp>::matchAndRewrite(
    LoadOp load, PatternRewriter &rewriter) const {
  PatternRewriter::InsertionGuard guard(rewriter);

  // Load from typed pointers are not supported.
  if (!load.getAddr().getType().isOpaque())
    return failure();

  Type inconsistentElementType =
      isElementTypeInconsistent(load.getAddr(), load.getType());
  if (!inconsistentElementType)
    return failure();
  Type firstType = getFirstSubelementType(inconsistentElementType);
  if (!firstType)
    return failure();
  DataLayout layout = DataLayout::closest(load);
  if (!areBitcastCompatible(layout, firstType, load.getResult().getType()))
    return failure();

  insertFieldIndirection<LoadOp>(load, rewriter, inconsistentElementType);

  // If the load does not use the first type but a type that can be casted from
  // it, add a bitcast and change the load type.
  if (firstType != load.getResult().getType()) {
    rewriter.setInsertionPointAfterValue(load.getResult());
    BitcastOp bitcast = rewriter.create<BitcastOp>(
        load->getLoc(), load.getResult().getType(), load.getResult());
    rewriter.updateRootInPlace(load,
                               [&]() { load.getResult().setType(firstType); });
    rewriter.replaceAllUsesExcept(load.getResult(), bitcast.getResult(),
                                  bitcast);
  }

  return success();
}

template <>
LogicalResult AddFieldGetterToStructDirectUse<StoreOp>::matchAndRewrite(
    StoreOp store, PatternRewriter &rewriter) const {
  PatternRewriter::InsertionGuard guard(rewriter);

  // Store to typed pointers are not supported.
  if (!store.getAddr().getType().isOpaque())
    return failure();

  Type inconsistentElementType =
      isElementTypeInconsistent(store.getAddr(), store.getValue().getType());
  if (!inconsistentElementType)
    return failure();
  Type firstType = getFirstSubelementType(inconsistentElementType);
  if (!firstType)
    return failure();

  DataLayout layout = DataLayout::closest(store);
  // Check that the first field has the right type or can at least be bitcast
  // to the right type.
  if (!areBitcastCompatible(layout, firstType, store.getValue().getType()))
    return failure();

  insertFieldIndirection<StoreOp>(store, rewriter, inconsistentElementType);

  rewriter.updateRootInPlace(
      store, [&]() { store.getValueMutable().assign(store.getValue()); });

  return success();
}

//===----------------------------------------------------------------------===//
// CanonicalizeAlignedGep
//===----------------------------------------------------------------------===//

/// Returns the amount of bytes the provided GEP elements will offset the
/// pointer by. Returns nullopt if the offset could not be computed.
static std::optional<uint64_t> gepToByteOffset(DataLayout &layout, GEPOp gep) {

  SmallVector<uint32_t> indices;
  // Ensures all indices are static and fetches them.
  for (auto index : gep.getIndices()) {
    IntegerAttr indexInt = llvm::dyn_cast_if_present<IntegerAttr>(index);
    if (!indexInt)
      return std::nullopt;
    indices.push_back(indexInt.getInt());
  }

  uint64_t offset = indices[0] * layout.getTypeSize(gep.getSourceElementType());

  Type currentType = gep.getSourceElementType();
  for (uint32_t index : llvm::drop_begin(indices)) {
    bool shouldCancel =
        TypeSwitch<Type, bool>(currentType)
            .Case([&](LLVMArrayType arrayType) {
              if (arrayType.getNumElements() <= index)
                return true;
              offset += index * layout.getTypeSize(arrayType.getElementType());
              currentType = arrayType.getElementType();
              return false;
            })
            .Case([&](LLVMStructType structType) {
              ArrayRef<Type> body = structType.getBody();
              if (body.size() <= index)
                return true;
              for (uint32_t i = 0; i < index; i++) {
                if (!structType.isPacked())
                  offset = llvm::alignTo(offset,
                                         layout.getTypeABIAlignment(body[i]));
                offset += layout.getTypeSize(body[i]);
              }
              currentType = body[index];
              return false;
            })
            .Default([](Type) { return true; });

    if (shouldCancel)
      return std::nullopt;
  }

  return offset;
}

/// Fills in `equivalentIndicesOut` with GEP indices that would be equivalent to
/// offsetting a pointer by `offset` bytes, assuming the GEP has `base` as base
/// type.
static LogicalResult
findIndicesForOffset(DataLayout &layout, Type base, uint64_t offset,
                     SmallVectorImpl<GEPArg> &equivalentIndicesOut) {

  uint64_t baseSize = layout.getTypeSize(base);
  uint64_t rootIndex = offset / baseSize;
  if (rootIndex > std::numeric_limits<uint32_t>::max())
    return failure();
  equivalentIndicesOut.push_back(rootIndex);

  uint64_t distanceToStart = rootIndex * baseSize;

#ifndef NDEBUG
  auto isWithinCurrentType = [&](Type currentType) {
    return offset < distanceToStart + layout.getTypeSize(currentType);
  };
#endif

  Type currentType = base;
  while (distanceToStart < offset) {
    // While an index that does not perfectly align with offset has not been
    // reached...

    assert(isWithinCurrentType(currentType));

    bool shouldCancel =
        TypeSwitch<Type, bool>(currentType)
            .Case([&](LLVMArrayType arrayType) {
              // Find which element of the array contains the offset.
              uint64_t elemSize =
                  layout.getTypeSize(arrayType.getElementType());
              uint64_t index = (offset - distanceToStart) / elemSize;
              equivalentIndicesOut.push_back(index);
              distanceToStart += index * elemSize;

              // Then, try to find where in the element the offset is. If the
              // offset is exactly the beginning of the element, the loop is
              // complete.
              currentType = arrayType.getElementType();

              // Only continue if the element in question can be indexed using
              // an i32.
              return index > std::numeric_limits<uint32_t>::max();
            })
            .Case([&](LLVMStructType structType) {
              ArrayRef<Type> body = structType.getBody();
              uint32_t index = 0;

              // Walk over the elements of the struct to find in which of them
              // the offset is.
              for (Type elem : body) {
                uint64_t elemSize = layout.getTypeSize(elem);
                if (!structType.isPacked()) {
                  distanceToStart = llvm::alignTo(
                      distanceToStart, layout.getTypeABIAlignment(elem));
                  // If the offset is in padding, cancel the rewrite.
                  if (offset < distanceToStart)
                    return true;
                }

                if (offset < distanceToStart + elemSize) {
                  // The offset is within this element, stop iterating the
                  // struct and look within the current element.
                  equivalentIndicesOut.push_back(index);
                  currentType = elem;
                  return false;
                }

                // The offset is not within this element, continue walking over
                // the struct.
                distanceToStart += elemSize;
                index++;
              }

              // The offset was supposed to be within this struct but is not.
              // This can happen if the offset points into final padding.
              // Anyway, nothing can be done.
              return true;
            })
            .Default([](Type) {
              // If the offset is within a type that cannot be split, no indices
              // will yield this offset. This can happen if the offset is not
              // perfectly aligned with a leaf type.
              // TODO: support vectors.
              return true;
            });

    if (shouldCancel)
      return failure();
  }

  return success();
}

/// Returns the consistent type for the GEP if the GEP is not type-consistent.
/// Returns failure if the GEP is already consistent.
static FailureOr<Type> getRequiredConsistentGEPType(GEPOp gep) {
  // GEP of typed pointers are not supported.
  if (!gep.getElemType())
    return failure();

  std::optional<Type> maybeBaseType = gep.getElemType();
  if (!maybeBaseType)
    return failure();
  Type baseType = *maybeBaseType;

  Type typeHint = isElementTypeInconsistent(gep.getBase(), baseType);
  if (!typeHint)
    return failure();
  return typeHint;
}

LogicalResult
CanonicalizeAlignedGep::matchAndRewrite(GEPOp gep,
                                        PatternRewriter &rewriter) const {
  FailureOr<Type> typeHint = getRequiredConsistentGEPType(gep);
  if (failed(typeHint)) {
    // GEP is already canonical, nothing to do here.
    return failure();
  }

  DataLayout layout = DataLayout::closest(gep);
  std::optional<uint64_t> desiredOffset = gepToByteOffset(layout, gep);
  if (!desiredOffset)
    return failure();

  SmallVector<GEPArg> newIndices;
  if (failed(
          findIndicesForOffset(layout, *typeHint, *desiredOffset, newIndices)))
    return failure();

  rewriter.replaceOpWithNewOp<GEPOp>(
      gep, LLVM::LLVMPointerType::get(getContext()), *typeHint, gep.getBase(),
      newIndices, gep.getInbounds());

  return success();
}

/// Returns the list of fields of `structType` that are written to by a store
/// operation writing `storeSize` bytes at `storeOffset` within the struct.
/// `storeOffset` is required to cleanly point to an immediate field within
/// the struct.
/// If the write operation were to write to any padding, write beyond the
/// struct, partially write to a field, or contains currently unsupported
/// types, failure is returned.
static FailureOr<ArrayRef<Type>>
getWrittenToFields(const DataLayout &dataLayout, LLVMStructType structType,
                   int storeSize, unsigned storeOffset) {
  ArrayRef<Type> body = structType.getBody();
  unsigned currentOffset = 0;
  body = body.drop_until([&](Type type) {
    if (!structType.isPacked()) {
      unsigned alignment = dataLayout.getTypeABIAlignment(type);
      currentOffset = llvm::alignTo(currentOffset, alignment);
    }

    // currentOffset is guaranteed to be equal to offset since offset is either
    // 0 or stems from a type-consistent GEP indexing into just a single
    // aggregate.
    if (currentOffset == storeOffset)
      return true;

    assert(currentOffset < storeOffset &&
           "storeOffset should cleanly point into an immediate field");

    currentOffset += dataLayout.getTypeSize(type);
    return false;
  });

  size_t exclusiveEnd = 0;
  for (; exclusiveEnd < body.size() && storeSize > 0; exclusiveEnd++) {
    // Not yet recursively handling aggregates, only primitives.
    if (!isa<IntegerType, FloatType>(body[exclusiveEnd]))
      return failure();

    if (!structType.isPacked()) {
      unsigned alignment = dataLayout.getTypeABIAlignment(body[exclusiveEnd]);
      // No padding allowed inbetween fields at this point in time.
      if (!llvm::isAligned(llvm::Align(alignment), currentOffset))
        return failure();
    }

    unsigned fieldSize = dataLayout.getTypeSize(body[exclusiveEnd]);
    currentOffset += fieldSize;
    storeSize -= fieldSize;
  }

  // If the storeSize is not 0 at this point we are either partially writing
  // into a field or writing past the aggregate as a whole. Abort.
  if (storeSize != 0)
    return failure();
  return body.take_front(exclusiveEnd);
}

/// Splits a store of the vector `value` into `address` at `storeOffset` into
/// multiple stores of each element with the goal of each generated store
/// becoming type-consistent through subsequent pattern applications.
static void splitVectorStore(const DataLayout &dataLayout, Location loc,
                             RewriterBase &rewriter, Value address,
                             TypedValue<VectorType> value,
                             unsigned storeOffset) {
  VectorType vectorType = value.getType();
  unsigned elementSize = dataLayout.getTypeSize(vectorType.getElementType());

  // Extract every element in the vector and store it in the given address.
  for (size_t index : llvm::seq<size_t>(0, vectorType.getNumElements())) {
    auto pos =
        rewriter.create<ConstantOp>(loc, rewriter.getI32IntegerAttr(index));
    auto extractOp = rewriter.create<ExtractElementOp>(loc, value, pos);

    // For convenience, we do indexing by calculating the final byte offset.
    // Other patterns will turn this into a type-consistent GEP.
    auto gepOp = rewriter.create<GEPOp>(
        loc, address.getType(), rewriter.getI8Type(), address,
        ArrayRef<GEPArg>{storeOffset + index * elementSize});

    rewriter.create<StoreOp>(loc, extractOp, gepOp);
  }
}

/// Splits a store of the integer `value` into `address` at `storeOffset` into
/// multiple stores to each 'writtenFields', making each store operation
/// type-consistent.
static void splitIntegerStore(const DataLayout &dataLayout, Location loc,
                              RewriterBase &rewriter, Value address,
                              Value value, unsigned storeOffset,
                              ArrayRef<Type> writtenToFields) {
  unsigned currentOffset = storeOffset;
  for (Type type : writtenToFields) {
    unsigned fieldSize = dataLayout.getTypeSize(type);

    // Extract the data out of the integer by first shifting right and then
    // truncating it.
    auto pos = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(value.getType(),
                                     (currentOffset - storeOffset) * 8));

    auto shrOp = rewriter.create<LShrOp>(loc, value, pos);

    IntegerType fieldIntType = rewriter.getIntegerType(fieldSize * 8);
    Value valueToStore = rewriter.create<TruncOp>(loc, fieldIntType, shrOp);

    // We create an `i8` indexed GEP here as that is the easiest (offset is
    // already known). Other patterns turn this into a type-consistent GEP.
    auto gepOp =
        rewriter.create<GEPOp>(loc, address.getType(), rewriter.getI8Type(),
                               address, ArrayRef<GEPArg>{currentOffset});
    rewriter.create<StoreOp>(loc, valueToStore, gepOp);

    // No need to care about padding here since we already checked previously
    // that no padding exists in this range.
    currentOffset += fieldSize;
  }
}

LogicalResult SplitStores::matchAndRewrite(StoreOp store,
                                           PatternRewriter &rewriter) const {
  Type sourceType = store.getValue().getType();
  if (!isa<IntegerType, VectorType>(sourceType)) {
    // We currently only support integer and vector sources.
    return failure();
  }

  Type typeHint = isElementTypeInconsistent(store.getAddr(), sourceType);
  if (!typeHint) {
    // Nothing to do, since it is already consistent.
    return failure();
  }

  auto dataLayout = DataLayout::closest(store);

  unsigned offset = 0;
  Value address = store.getAddr();
  if (auto gepOp = address.getDefiningOp<GEPOp>()) {
    // Currently only handle canonical GEPs with exactly two indices,
    // indexing a single aggregate deep.
    // Recursing into sub-structs is left as a future exercise.
    // If the GEP is not canonical we have to fail, otherwise we would not
    // create type-consistent IR.
    if (gepOp.getIndices().size() != 2 ||
        succeeded(getRequiredConsistentGEPType(gepOp)))
      return failure();

    // A GEP might point somewhere into the middle of an aggregate with the
    // store storing into multiple adjacent elements. Destructure into
    // the base address with an offset.
    std::optional<uint64_t> byteOffset = gepToByteOffset(dataLayout, gepOp);
    if (!byteOffset)
      return failure();

    offset = *byteOffset;
    typeHint = gepOp.getSourceElementType();
    address = gepOp.getBase();
  }

  auto structType = typeHint.dyn_cast<LLVMStructType>();
  if (!structType) {
    // TODO: Handle array types in the future.
    return failure();
  }

  FailureOr<ArrayRef<Type>> writtenToFields =
      getWrittenToFields(dataLayout, structType,
                         /*storeSize=*/dataLayout.getTypeSize(sourceType),
                         /*storeOffset=*/offset);
  if (failed(writtenToFields))
    return failure();

  if (writtenToFields->size() <= 1) {
    // Other patterns should take care of this case, we are only interested in
    // splitting field stores.
    return failure();
  }

  if (isa<IntegerType>(sourceType)) {
    splitIntegerStore(dataLayout, store.getLoc(), rewriter, address,
                      store.getValue(), offset, *writtenToFields);
    rewriter.eraseOp(store);
    return success();
  }

  // Add a reasonable bound to not split very large vectors that would end up
  // generating lots of code.
  if (dataLayout.getTypeSizeInBits(sourceType) > maxVectorSplitSize)
    return failure();

  // Vector types are simply split into its elements and new stores generated
  // with those. Subsequent pattern applications will split these stores further
  // if required.
  splitVectorStore(dataLayout, store.getLoc(), rewriter, address,
                   cast<TypedValue<VectorType>>(store.getValue()), offset);
  rewriter.eraseOp(store);
  return success();
}

LogicalResult BitcastStores::matchAndRewrite(StoreOp store,
                                             PatternRewriter &rewriter) const {
  Type sourceType = store.getValue().getType();
  Type typeHint = isElementTypeInconsistent(store.getAddr(), sourceType);
  if (!typeHint) {
    // Nothing to do, since it is already consistent.
    return failure();
  }

  auto dataLayout = DataLayout::closest(store);
  if (!areBitcastCompatible(dataLayout, typeHint, sourceType))
    return failure();

  auto bitcastOp =
      rewriter.create<BitcastOp>(store.getLoc(), typeHint, store.getValue());
  rewriter.updateRootInPlace(
      store, [&] { store.getValueMutable().assign(bitcastOp); });
  return success();
}

//===----------------------------------------------------------------------===//
// Type consistency pass
//===----------------------------------------------------------------------===//

namespace {
struct LLVMTypeConsistencyPass
    : public LLVM::impl::LLVMTypeConsistencyBase<LLVMTypeConsistencyPass> {
  void runOnOperation() override {
    RewritePatternSet rewritePatterns(&getContext());
    rewritePatterns.add<AddFieldGetterToStructDirectUse<LoadOp>>(&getContext());
    rewritePatterns.add<AddFieldGetterToStructDirectUse<StoreOp>>(
        &getContext());
    rewritePatterns.add<CanonicalizeAlignedGep>(&getContext());
    rewritePatterns.add<SplitStores>(&getContext(), maxVectorSplitSize);
    rewritePatterns.add<BitcastStores>(&getContext());
    FrozenRewritePatternSet frozen(std::move(rewritePatterns));

    if (failed(applyPatternsAndFoldGreedily(getOperation(), frozen)))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> LLVM::createTypeConsistencyPass() {
  return std::make_unique<LLVMTypeConsistencyPass>();
}
