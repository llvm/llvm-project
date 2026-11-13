//===- LLVMArithOps.cpp - LLVM dialect arithmetic ops ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `Arith` of the LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMArithOps.h"

#include "IR/LLVMOps.h"
#include "mlir/Dialect/LLVMIR/LLVMConstantOps.h"
#include "mlir/Dialect/LLVMIR/LLVMMemoryOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

using namespace mlir;
using namespace mlir::LLVM;

/// Returns a boolean type that has the same shape as `type`. It supports both
/// fixed size vectors as well as scalable vectors.
Type mlir::LLVM::getI1SameShape(Type type) {
  Type i1Type = IntegerType::get(type.getContext(), 1);
  if (LLVM::isCompatibleVectorType(type))
    return LLVM::getVectorType(i1Type, LLVM::getVectorNumElements(type));
  return i1Type;
}

//===----------------------------------------------------------------------===//
// Printing, parsing, folding and builder for LLVM::CmpOp.
//===----------------------------------------------------------------------===//

template <typename PredicateAttr, typename Predicate>
static ParseResult parseCmpPredicateImpl(
    OpAsmParser &parser, PredicateAttr &predicate,
    function_ref<std::optional<Predicate>(StringRef)> symbolize) {
  std::string spelling;
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseString(&spelling))
    return failure();
  std::optional<Predicate> value = symbolize(spelling);
  if (!value)
    return parser.emitError(loc)
           << "'" << spelling
           << "' is an incorrect value of the 'predicate' attribute";
  predicate = PredicateAttr::get(parser.getContext(), *value);
  return success();
}

ParseResult mlir::LLVM::parseCmpPredicate(OpAsmParser &parser,
                                          ICmpPredicateAttr &predicate) {
  return parseCmpPredicateImpl<ICmpPredicateAttr, ICmpPredicate>(
      parser, predicate,
      [](StringRef spelling) { return symbolizeICmpPredicate(spelling); });
}

ParseResult mlir::LLVM::parseCmpPredicate(OpAsmParser &parser,
                                          FCmpPredicateAttr &predicate) {
  return parseCmpPredicateImpl<FCmpPredicateAttr, FCmpPredicate>(
      parser, predicate,
      [](StringRef spelling) { return symbolizeFCmpPredicate(spelling); });
}

void mlir::LLVM::printCmpPredicate(OpAsmPrinter &printer, Operation *,
                                   ICmpPredicateAttr predicate) {
  printer << '"' << stringifyICmpPredicate(predicate.getValue()) << '"';
}

void mlir::LLVM::printCmpPredicate(OpAsmPrinter &printer, Operation *,
                                   FCmpPredicateAttr predicate) {
  printer << '"' << stringifyFCmpPredicate(predicate.getValue()) << '"';
}

/// Returns a scalar or vector boolean attribute of the given type.
static Attribute getBoolAttribute(Type type, MLIRContext *ctx, bool value) {
  auto boolAttr = BoolAttr::get(ctx, value);
  ShapedType shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType)
    return boolAttr;
  return DenseElementsAttr::get(shapedType, boolAttr);
}

OpFoldResult ICmpOp::fold(FoldAdaptor adaptor) {
  if (getPredicate() != ICmpPredicate::eq &&
      getPredicate() != ICmpPredicate::ne)
    return {};

  // cmpi(eq/ne, x, x) -> true/false
  if (getLhs() == getRhs())
    return getBoolAttribute(getType(), getContext(),
                            getPredicate() == ICmpPredicate::eq);

  // cmpi(eq/ne, alloca, null) -> false/true
  if (getLhs().getDefiningOp<AllocaOp>() && getRhs().getDefiningOp<ZeroOp>())
    return getBoolAttribute(getType(), getContext(),
                            getPredicate() == ICmpPredicate::ne);

  // cmpi(eq/ne, null, alloca) -> cmpi(eq/ne, alloca, null)
  if (getLhs().getDefiningOp<ZeroOp>() && getRhs().getDefiningOp<AllocaOp>()) {
    Value lhs = getLhs();
    Value rhs = getRhs();
    getLhsMutable().assign(rhs);
    getRhsMutable().assign(lhs);
    return getResult();
  }

  return {};
}

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

//===----------------------------------------------------------------------===//
// ShuffleVectorOp
//===----------------------------------------------------------------------===//

void ShuffleVectorOp::build(OpBuilder &builder, OperationState &state, Value v1,
                            Value v2, DenseI32ArrayAttr mask,
                            ArrayRef<NamedAttribute> attrs) {
  auto containerType = v1.getType();
  auto vType = LLVM::getVectorType(
      cast<VectorType>(containerType).getElementType(), mask.size(),
      LLVM::isScalableVectorType(containerType));
  build(builder, state, vType, v1, v2, mask);
  state.addAttributes(attrs);
}

void ShuffleVectorOp::build(OpBuilder &builder, OperationState &state, Value v1,
                            Value v2, ArrayRef<int32_t> mask) {
  build(builder, state, v1, v2, builder.getDenseI32ArrayAttr(mask));
}

/// Build the result type of a shuffle vector operation.
ParseResult mlir::LLVM::parseShuffleType(AsmParser &parser, Type v1Type,
                                         Type &resType,
                                         DenseI32ArrayAttr mask) {
  if (!LLVM::isCompatibleVectorType(v1Type))
    return parser.emitError(parser.getCurrentLocation(),
                            "expected an LLVM compatible vector type");
  resType =
      LLVM::getVectorType(cast<VectorType>(v1Type).getElementType(),
                          mask.size(), LLVM::isScalableVectorType(v1Type));
  return success();
}

/// Nothing to do when the result type is inferred.
void mlir::LLVM::printShuffleType(AsmPrinter &printer, Operation *op,
                                  Type v1Type, Type resType,
                                  DenseI32ArrayAttr mask) {}

LogicalResult ShuffleVectorOp::verify() {
  if (LLVM::isScalableVectorType(getV1().getType()) &&
      llvm::any_of(getMask(), [](int32_t v) { return v != 0; }))
    return emitOpError("expected a splat operation for scalable vectors");
  return success();
}

// Folding for shufflevector op when v1 is single element 1D vector
// and the mask is a single zero. OpFoldResult will be v1 in this case.
OpFoldResult ShuffleVectorOp::fold(FoldAdaptor adaptor) {
  // Check if operand 0 is a single element vector.
  auto vecType = llvm::dyn_cast<VectorType>(getV1().getType());
  if (!vecType || vecType.getRank() != 1 || vecType.getNumElements() != 1)
    return {};
  // Check if the mask is a single zero.
  // Note: The mask is guaranteed to be non-empty.
  if (getMask().size() != 1 || getMask()[0] != 0)
    return {};
  return getV1();
}

//===----------------------------------------------------------------------===//
// ShlOp
//===----------------------------------------------------------------------===//

OpFoldResult LLVM::ShlOp::fold(FoldAdaptor adaptor) {
  auto rhs = dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!rhs)
    return {};

  if (rhs.getValue().uge(getLhs().getType().getIntOrFloatBitWidth()))
    return {}; // TODO: Fold into poison.

  auto lhs = dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  if (!lhs)
    return {};

  return IntegerAttr::get(getType(), lhs.getValue().shl(rhs.getValue()));
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

OpFoldResult LLVM::OrOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  if (!lhs)
    return {};

  auto rhs = dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!rhs)
    return {};

  return IntegerAttr::get(getType(), lhs.getValue() | rhs.getValue());
}

//===----------------------------------------------------------------------===//
// UDivOp
//===----------------------------------------------------------------------===//
Speculation::Speculatability UDivOp::getSpeculatability() {
  // X / 0 => UB
  Value divisor = getRhs();
  if (matchPattern(divisor, m_IntRangeWithoutZeroU()))
    return Speculation::Speculatable;

  return Speculation::NotSpeculatable;
}

//===----------------------------------------------------------------------===//
// SDivOp
//===----------------------------------------------------------------------===//
Speculation::Speculatability SDivOp::getSpeculatability() {
  // This function conservatively assumes that all signed division by -1 are
  // not speculatable.
  // X / 0 => UB
  // INT_MIN / -1 => UB
  Value divisor = getRhs();
  if (matchPattern(divisor, m_IntRangeWithoutZeroS()) &&
      matchPattern(divisor, m_IntRangeWithoutNegOneS()))
    return Speculation::Speculatable;

  return Speculation::NotSpeculatable;
}

//===----------------------------------------------------------------------===//
// URemOp
//===----------------------------------------------------------------------===//

Speculation::Speculatability URemOp::getSpeculatability() {
  // urem(x, 0) => UB
  Value divisor = getRhs();
  if (matchPattern(divisor, m_IntRangeWithoutZeroU()))
    return Speculation::Speculatable;

  return Speculation::NotSpeculatable;
}

//===----------------------------------------------------------------------===//
// SRemOp
//===----------------------------------------------------------------------===//

Speculation::Speculatability SRemOp::getSpeculatability() {
  // srem(x, 0) => UB
  // srem(INT_MIN, -1) => UB
  Value divisor = getRhs();
  if (matchPattern(divisor, m_IntRangeWithoutZeroS()) &&
      matchPattern(divisor, m_IntRangeWithoutNegOneS()))
    return Speculation::Speculatable;

  return Speculation::NotSpeculatable;
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMArithOps.cpp.inc"
