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
static bool areCastCompatible(DataLayout &layout, Type lhs, Type rhs) {
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
  if (!areCastCompatible(layout, firstType, load.getResult().getType()))
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
  if (!areCastCompatible(layout, firstType, store.getValue().getType()))
    return failure();

  insertFieldIndirection<StoreOp>(store, rewriter, inconsistentElementType);

  Value replaceValue = store.getValue();
  if (firstType != store.getValue().getType()) {
    rewriter.setInsertionPointAfterValue(store.getValue());
    replaceValue = rewriter.create<BitcastOp>(store->getLoc(), firstType,
                                              store.getValue());
  }

  rewriter.updateRootInPlace(
      store, [&]() { store.getValueMutable().assign(replaceValue); });

  return success();
}

//===----------------------------------------------------------------------===//
// CanonicalizeAlignedGep
//===----------------------------------------------------------------------===//

/// Returns the amount of bytes the provided GEP elements will offset the
/// pointer by. Returns nullopt if the offset could not be computed.
static std::optional<uint64_t> gepToByteOffset(DataLayout &layout, Type base,
                                               ArrayRef<uint32_t> indices) {
  uint64_t offset = indices[0] * layout.getTypeSize(base);

  Type currentType = base;
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

LogicalResult
CanonicalizeAlignedGep::matchAndRewrite(GEPOp gep,
                                        PatternRewriter &rewriter) const {
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

  SmallVector<uint32_t> indices;
  // Ensures all indices are static and fetches them.
  for (auto index : gep.getIndices()) {
    IntegerAttr indexInt = llvm::dyn_cast_if_present<IntegerAttr>(index);
    if (!indexInt)
      return failure();
    indices.push_back(indexInt.getInt());
  }

  DataLayout layout = DataLayout::closest(gep);
  std::optional<uint64_t> desiredOffset =
      gepToByteOffset(layout, gep.getSourceElementType(), indices);
  if (!desiredOffset)
    return failure();

  SmallVector<GEPArg> newIndices;
  if (failed(
          findIndicesForOffset(layout, typeHint, *desiredOffset, newIndices)))
    return failure();

  rewriter.replaceOpWithNewOp<GEPOp>(
      gep, LLVM::LLVMPointerType::get(getContext()), typeHint, gep.getBase(),
      newIndices, gep.getInbounds());

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
    FrozenRewritePatternSet frozen(std::move(rewritePatterns));

    if (failed(applyPatternsAndFoldGreedily(getOperation(), frozen)))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> LLVM::createTypeConsistencyPass() {
  return std::make_unique<LLVMTypeConsistencyPass>();
}
