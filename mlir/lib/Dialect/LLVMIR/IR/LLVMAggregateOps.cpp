//===- LLVMAggregateOps.cpp - LLVM dialect aggregate ops ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `Aggregate` of the LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMAggregateOps.h"

#include "IR/LLVMOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::LLVM;

//===----------------------------------------------------------------------===//
// ExtractValueOp
//===----------------------------------------------------------------------===//

/// Extract the type at `position` in the LLVM IR aggregate type
/// `containerType`. Each element of `position` is an index into a nested
/// aggregate type. Return the resulting type or emit an error.
static Type getInsertExtractValueElementType(
    function_ref<InFlightDiagnostic(StringRef)> emitError, Type containerType,
    ArrayRef<int64_t> position) {
  Type llvmType = containerType;
  if (!isCompatibleType(containerType)) {
    emitError("expected LLVM IR Dialect type, got ") << containerType;
    return {};
  }

  // Infer the element type from the structure type: iteratively step inside the
  // type by taking the element type, indexed by the position attribute for
  // structures.  Check the position index before accessing, it is supposed to
  // be in bounds.
  for (int64_t idx : position) {
    if (auto arrayType = llvm::dyn_cast<LLVMArrayType>(llvmType)) {
      if (idx < 0 || static_cast<unsigned>(idx) >= arrayType.getNumElements()) {
        emitError("position out of bounds: ") << idx;
        return {};
      }
      llvmType = arrayType.getElementType();
    } else if (auto structType = llvm::dyn_cast<LLVMStructType>(llvmType)) {
      if (idx < 0 ||
          static_cast<unsigned>(idx) >= structType.getBody().size()) {
        emitError("position out of bounds: ") << idx;
        return {};
      }
      llvmType = structType.getBody()[idx];
    } else {
      emitError("expected LLVM IR structure/array type, got: ") << llvmType;
      return {};
    }
  }
  return llvmType;
}

/// Extract the type at `position` in the wrapped LLVM IR aggregate type
/// `containerType`.
static Type getInsertExtractValueElementType(Type llvmType,
                                             ArrayRef<int64_t> position) {
  for (int64_t idx : position) {
    if (auto structType = llvm::dyn_cast<LLVMStructType>(llvmType))
      llvmType = structType.getBody()[idx];
    else
      llvmType = llvm::cast<LLVMArrayType>(llvmType).getElementType();
  }
  return llvmType;
}

/// Extracts the element at the given index from an attribute. For
/// `ElementsAttr`, returns the element at the specified index, or `nullptr` if
/// the shaped type does not have rank 1. For `ArrayAttr`, returns the element
/// at the specified index. For `ZeroAttr`, `UndefAttr`, and `PoisonAttr`,
/// returns the attribute itself unchanged. Returns `nullptr` if the attribute
/// is not one of these types or if the index is out of bounds.
static Attribute extractElementAt(Attribute attr, size_t index) {
  if (auto elementsAttr = dyn_cast<ElementsAttr>(attr)) {
    ShapedType shapedType = elementsAttr.getShapedType();
    if (!shapedType.hasRank() || shapedType.getRank() != 1)
      return nullptr;
    if (index < static_cast<size_t>(elementsAttr.getNumElements()))
      return elementsAttr.getValues<Attribute>()[index];
    return nullptr;
  }
  if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
    if (index < arrayAttr.getValue().size())
      return arrayAttr[index];
    return nullptr;
  }
  if (isa<ZeroAttr, UndefAttr, PoisonAttr>(attr))
    return attr;
  return nullptr;
}

OpFoldResult LLVM::ExtractValueOp::fold(FoldAdaptor adaptor) {
  if (auto extractValueOp = getContainer().getDefiningOp<ExtractValueOp>()) {
    SmallVector<int64_t, 4> newPos(extractValueOp.getPosition());
    newPos.append(getPosition().begin(), getPosition().end());
    setPosition(newPos);
    getContainerMutable().set(extractValueOp.getContainer());
    return getResult();
  }

  Attribute containerAttr;
  if (matchPattern(getContainer(), m_Constant(&containerAttr))) {
    for (int64_t pos : getPosition()) {
      containerAttr = extractElementAt(containerAttr, pos);
      if (!containerAttr)
        return nullptr;
    }
    return containerAttr;
  }

  Value container = getContainer();
  ArrayRef<int64_t> extractPos = getPosition();
  while (auto insertValueOp = container.getDefiningOp<InsertValueOp>()) {
    ArrayRef<int64_t> insertPos = insertValueOp.getPosition();
    auto extractPosSize = extractPos.size();
    auto insertPosSize = insertPos.size();

    // Case 1: Exact match of positions.
    if (extractPos == insertPos)
      return insertValueOp.getValue();

    // Case 2: Insert position is a prefix of extract position. Continue
    // traversal with the inserted value. Example:
    // ```
    // %0 = llvm.insertvalue %arg1, %undef[0] : !llvm.struct<(i32, i32, i32)>
    // %1 = llvm.insertvalue %arg2, %0[1] : !llvm.struct<(i32, i32, i32)>
    // %2 = llvm.insertvalue %arg3, %1[2] : !llvm.struct<(i32, i32, i32)>
    // %3 = llvm.insertvalue %2, %foo[0]
    //     : !llvm.struct<(struct<(i32, i32, i32)>, i64)>
    // %4 = llvm.extractvalue %3[0, 0]
    //     : !llvm.struct<(struct<(i32, i32, i32)>, i64)>
    // ```
    // In the above example, %4 is folded to %arg1.
    if (extractPosSize > insertPosSize &&
        extractPos.take_front(insertPosSize) == insertPos) {
      container = insertValueOp.getValue();
      extractPos = extractPos.drop_front(insertPosSize);
      continue;
    }

    // Case 3: Try to continue the traversal with the container value.

    // If extract position is a prefix of insert position, stop propagating back
    // as it will miss dependencies. For instance, %3 should not fold to %f0 in
    // the following example:
    // ```
    //   %1 = llvm.insertvalue %f0, %0[0, 0] :
    //     !llvm.array<4 x !llvm.array<4 x f32>>
    //   %2 = llvm.insertvalue %arr, %1[0] :
    //     !llvm.array<4 x !llvm.array<4 x f32>>
    //   %3 = llvm.extractvalue %2[0, 0] : !llvm.array<4 x !llvm.array<4 x f32>>
    // ```
    if (insertPosSize > extractPosSize &&
        extractPos == insertPos.take_front(extractPosSize))
      break;
    // If neither a prefix, nor the exact position, we can extract out of the
    // value being inserted into. Moreover, we can try again if that operand
    // is itself an insertvalue expression.
    container = insertValueOp.getContainer();
  }

  // We failed to resolve past this container either because it is not an
  // InsertValueOp, or it is an InsertValueOp that partially overlaps with the
  // value being extracted. Update to read from this container instead.
  if (container == getContainer())
    return {};
  setPosition(extractPos);
  getContainerMutable().assign(container);
  return getResult();
}

LogicalResult ExtractValueOp::verify() {
  auto emitError = [this](StringRef msg) { return emitOpError(msg); };
  Type valueType = getInsertExtractValueElementType(
      emitError, getContainer().getType(), getPosition());
  if (!valueType)
    return failure();

  if (getRes().getType() != valueType)
    return emitOpError() << "Type mismatch: extracting from "
                         << getContainer().getType() << " should produce "
                         << valueType << " but this op returns "
                         << getRes().getType();
  return success();
}

void ExtractValueOp::build(OpBuilder &builder, OperationState &state,
                           Value container, ArrayRef<int64_t> position) {
  build(builder, state,
        getInsertExtractValueElementType(container.getType(), position),
        container, builder.getAttr<DenseI64ArrayAttr>(position));
}

//===----------------------------------------------------------------------===//
// InsertValueOp
//===----------------------------------------------------------------------===//

namespace {
/// Update any ExtractValueOps using a given InsertValueOp to instead read from
/// the closest InsertValueOp in the chain leading up to the current op that
/// writes to the same member. This traversal could be done entirely in
/// ExtractValueOp::fold, but doing it here significantly speeds things up
/// because we can handle several ExtractValueOps with a single traversal.
/// For instance, in this example:
///   %i0 = llvm.insertvalue %v0, %undef[0]
///   %i1 = llvm.insertvalue %v1, %0[1]
///   ...
///   %i999 = llvm.insertvalue %v999, %998[999]
///   %e0 = llvm.extractvalue %i999[0]
///   %e1 = llvm.extractvalue %i999[1]
///   ...
///   %e999 = llvm.extractvalue %i999[999]
/// Individually running the folder on each extractvalue would require
/// traversing the insertvalue chain 1000 times, but running this pattern on the
/// InsertValueOp would allow us to achieve the same result with a single
/// traversal. The resulting IR after this pattern will then be:
///   %i0 = llvm.insertvalue %v0, %undef[0]
///   %i1 = llvm.insertvalue %v1, %0[1]
///   ...
///   %i999 = llvm.insertvalue %v999, %998[999]
///   %e0 = llvm.extractvalue %i0[0]
///   %e1 = llvm.extractvalue %i1[1]
///   ...
///   %e999 = llvm.extractvalue %i999[999]
struct ResolveExtractValueSource : public OpRewritePattern<InsertValueOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(InsertValueOp insertOp,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    // Map each position in the top-level struct to the ExtractOps that read
    // from it. For the example in the doc-comment above this map will be empty
    // when we visit ops %i0 - %i998. For %i999, it will contain:
    //   0 -> { %e0 }, 1 -> { %e1 }, ... 999-> { %e999 }
    DenseMap<int64_t, SmallVector<ExtractValueOp, 4>> posToExtractOps;
    auto insertBaseIdx = insertOp.getPosition()[0];
    for (auto &use : insertOp->getUses()) {
      if (auto extractOp = dyn_cast<ExtractValueOp>(use.getOwner())) {
        auto baseIdx = extractOp.getPosition()[0];
        // We can skip reads of the member that insertOp writes to since they
        // will not be updated.
        if (baseIdx == insertBaseIdx)
          continue;
        posToExtractOps[baseIdx].push_back(extractOp);
      }
    }
    // Walk up the chain of insertions and try to resolve the remaining
    // extractions that access the same member.
    Value nextContainer = insertOp.getContainer();
    while (!posToExtractOps.empty()) {
      auto curInsert =
          dyn_cast_or_null<InsertValueOp>(nextContainer.getDefiningOp());
      if (!curInsert)
        break;
      nextContainer = curInsert.getContainer();

      // Check if any extractions read the member written by this insertion.
      auto curInsertBaseIdx = curInsert.getPosition()[0];
      auto it = posToExtractOps.find(curInsertBaseIdx);
      if (it == posToExtractOps.end())
        continue;

      // Update the ExtractOps to read from the current insertion.
      for (auto &extractOp : it->second) {
        rewriter.modifyOpInPlace(extractOp, [&] {
          extractOp.getContainerMutable().assign(curInsert);
        });
      }
      // The entry should never be empty if it exists, so if we are at this
      // point, set changed to true.
      assert(!it->second.empty());
      changed |= true;
      posToExtractOps.erase(it);
    }
    // There was no insertion along the chain that wrote the member accessed by
    // these extracts. So we can update them to use the top of the chain.
    for (auto &[baseIdx, extracts] : posToExtractOps) {
      for (auto &extractOp : extracts) {
        rewriter.modifyOpInPlace(extractOp, [&] {
          extractOp.getContainerMutable().assign(nextContainer);
        });
      }
      assert(!extracts.empty() && "Empty list in map");
      changed = true;
    }
    return success(changed);
  }
};
} // namespace

void InsertValueOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<ResolveExtractValueSource>(context);
}

/// Infer the value type from the container type and position.
ParseResult mlir::LLVM::parseInsertExtractValueElementType(
    AsmParser &parser, Type &valueType, Type containerType,
    DenseI64ArrayAttr position) {
  valueType = getInsertExtractValueElementType(
      [&](StringRef msg) {
        return parser.emitError(parser.getCurrentLocation(), msg);
      },
      containerType, position.asArrayRef());
  return success(!!valueType);
}

/// Nothing to print for an inferred type.
void mlir::LLVM::printInsertExtractValueElementType(
    AsmPrinter &printer, Operation *op, Type valueType, Type containerType,
    DenseI64ArrayAttr position) {}

LogicalResult InsertValueOp::verify() {
  auto emitError = [this](StringRef msg) { return emitOpError(msg); };
  Type valueType = getInsertExtractValueElementType(
      emitError, getContainer().getType(), getPosition());
  if (!valueType)
    return failure();

  if (getValue().getType() != valueType)
    return emitOpError() << "Type mismatch: cannot insert "
                         << getValue().getType() << " into "
                         << getContainer().getType();

  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMAggregateOps.cpp.inc"
