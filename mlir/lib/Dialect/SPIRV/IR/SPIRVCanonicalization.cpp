//===- SPIRVCanonicalization.cpp - MLIR SPIR-V canonicalization patterns --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the folders and canonicalization patterns for SPIR-V ops.
//
//===----------------------------------------------------------------------===//

#include <optional>
#include <utility>

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Common utility functions
//===----------------------------------------------------------------------===//

/// Returns the boolean value under the hood if the given `boolAttr` is a scalar
/// or splat vector bool constant.
static std::optional<bool> getScalarOrSplatBoolAttr(Attribute attr) {
  if (!attr)
    return std::nullopt;

  if (auto boolAttr = llvm::dyn_cast<BoolAttr>(attr))
    return boolAttr.getValue();
  if (auto splatAttr = llvm::dyn_cast<SplatElementsAttr>(attr))
    if (splatAttr.getElementType().isInteger(1))
      return splatAttr.getSplatValue<bool>();
  return std::nullopt;
}

// Extracts an element from the given `composite` by following the given
// `indices`. Returns a null Attribute if error happens.
static Attribute extractCompositeElement(Attribute composite,
                                         ArrayRef<unsigned> indices) {
  // Check that given composite is a constant.
  if (!composite)
    return {};
  // Return composite itself if we reach the end of the index chain.
  if (indices.empty())
    return composite;

  if (auto vector = llvm::dyn_cast<ElementsAttr>(composite)) {
    assert(indices.size() == 1 && "must have exactly one index for a vector");
    return vector.getValues<Attribute>()[indices[0]];
  }

  if (auto array = llvm::dyn_cast<ArrayAttr>(composite)) {
    assert(!indices.empty() && "must have at least one index for an array");
    return extractCompositeElement(array.getValue()[indices[0]],
                                   indices.drop_front());
  }

  return {};
}

//===----------------------------------------------------------------------===//
// TableGen'erated canonicalizers
//===----------------------------------------------------------------------===//

namespace {
#include "SPIRVCanonicalization.inc"
} // namespace

//===----------------------------------------------------------------------===//
// spirv.AccessChainOp
//===----------------------------------------------------------------------===//

namespace {

/// Combines chained `spirv::AccessChainOp` operations into one
/// `spirv::AccessChainOp` operation.
struct CombineChainedAccessChain final
    : OpRewritePattern<spirv::AccessChainOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::AccessChainOp accessChainOp,
                                PatternRewriter &rewriter) const override {
    auto parentAccessChainOp =
        accessChainOp.getBasePtr().getDefiningOp<spirv::AccessChainOp>();

    if (!parentAccessChainOp) {
      return failure();
    }

    // Combine indices.
    SmallVector<Value, 4> indices(parentAccessChainOp.getIndices());
    llvm::append_range(indices, accessChainOp.getIndices());

    rewriter.replaceOpWithNewOp<spirv::AccessChainOp>(
        accessChainOp, parentAccessChainOp.getBasePtr(), indices);

    return success();
  }
};
} // namespace

void spirv::AccessChainOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<CombineChainedAccessChain>(context);
}

//===----------------------------------------------------------------------===//
// spirv.UMod
//===----------------------------------------------------------------------===//

// Input:
//    %0 = spirv.UMod %arg0, %const32 : i32
//    %1 = spirv.UMod %0, %const4 : i32
// Output:
//    %0 = spirv.UMod %arg0, %const32 : i32
//    %1 = spirv.UMod %arg0, %const4 : i32

// The transformation is only applied if one divisor is a multiple of the other.

// TODO(https://github.com/llvm/llvm-project/issues/63174): Add support for vector constants
struct UModSimplification final : OpRewritePattern<spirv::UModOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::UModOp umodOp,
                                PatternRewriter &rewriter) const override {
    auto prevUMod = umodOp.getOperand(0).getDefiningOp<spirv::UModOp>();
    if (!prevUMod)
      return failure();

    IntegerAttr prevValue;
    IntegerAttr currValue;
    if (!matchPattern(prevUMod.getOperand(1), m_Constant(&prevValue)) ||
        !matchPattern(umodOp.getOperand(1), m_Constant(&currValue)))
      return failure();

    APInt prevConstValue = prevValue.getValue();
    APInt currConstValue = currValue.getValue();

    // Ensure that one divisor is a multiple of the other. If not, fail the
    // transformation.
    if (prevConstValue.urem(currConstValue) != 0 &&
        currConstValue.urem(prevConstValue) != 0)
      return failure();

    // The transformation is safe. Replace the existing UMod operation with a
    // new UMod operation, using the original dividend and the current divisor.
    rewriter.replaceOpWithNewOp<spirv::UModOp>(
        umodOp, umodOp.getType(), prevUMod.getOperand(0), umodOp.getOperand(1));

    return success();
  }
};

void spirv::UModOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.insert<UModSimplification>(context);
}

//===----------------------------------------------------------------------===//
// spirv.BitcastOp
//===----------------------------------------------------------------------===//

OpFoldResult spirv::BitcastOp::fold(FoldAdaptor /*adaptor*/) {
  Value curInput = getOperand();
  if (getType() == curInput.getType())
    return curInput;

  // Look through nested bitcasts.
  if (auto prevCast = curInput.getDefiningOp<spirv::BitcastOp>()) {
    Value prevInput = prevCast.getOperand();
    if (prevInput.getType() == getType())
      return prevInput;

    getOperandMutable().assign(prevInput);
    return getResult();
  }

  // TODO(kuhar): Consider constant-folding the operand attribute.
  return {};
}

//===----------------------------------------------------------------------===//
// spirv.CompositeExtractOp
//===----------------------------------------------------------------------===//

OpFoldResult spirv::CompositeExtractOp::fold(FoldAdaptor adaptor) {
  Value compositeOp = getComposite();

  while (auto insertOp =
             compositeOp.getDefiningOp<spirv::CompositeInsertOp>()) {
    if (getIndices() == insertOp.getIndices())
      return insertOp.getObject();
    compositeOp = insertOp.getComposite();
  }

  if (auto constructOp =
          compositeOp.getDefiningOp<spirv::CompositeConstructOp>()) {
    auto type = llvm::cast<spirv::CompositeType>(constructOp.getType());
    if (getIndices().size() == 1 &&
        constructOp.getConstituents().size() == type.getNumElements()) {
      auto i = llvm::cast<IntegerAttr>(*getIndices().begin());
      if (i.getValue().getSExtValue() <
          static_cast<int64_t>(constructOp.getConstituents().size()))
        return constructOp.getConstituents()[i.getValue().getSExtValue()];
    }
  }

  auto indexVector = llvm::map_to_vector(getIndices(), [](Attribute attr) {
    return static_cast<unsigned>(llvm::cast<IntegerAttr>(attr).getInt());
  });
  return extractCompositeElement(adaptor.getComposite(), indexVector);
}

//===----------------------------------------------------------------------===//
// spirv.Constant
//===----------------------------------------------------------------------===//

OpFoldResult spirv::ConstantOp::fold(FoldAdaptor /*adaptor*/) {
  return getValue();
}

//===----------------------------------------------------------------------===//
// spirv.IAdd
//===----------------------------------------------------------------------===//

OpFoldResult spirv::IAddOp::fold(FoldAdaptor adaptor) {
  // x + 0 = x
  if (matchPattern(getOperand2(), m_Zero()))
    return getOperand1();

  // According to the SPIR-V spec:
  //
  // The resulting value will equal the low-order N bits of the correct result
  // R, where N is the component width and R is computed with enough precision
  // to avoid overflow and underflow.
  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](APInt a, const APInt &b) { return std::move(a) + b; });
}

//===----------------------------------------------------------------------===//
// spirv.IMul
//===----------------------------------------------------------------------===//

OpFoldResult spirv::IMulOp::fold(FoldAdaptor adaptor) {
  // x * 0 == 0
  if (matchPattern(getOperand2(), m_Zero()))
    return getOperand2();
  // x * 1 = x
  if (matchPattern(getOperand2(), m_One()))
    return getOperand1();

  // According to the SPIR-V spec:
  //
  // The resulting value will equal the low-order N bits of the correct result
  // R, where N is the component width and R is computed with enough precision
  // to avoid overflow and underflow.
  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](const APInt &a, const APInt &b) { return a * b; });
}

//===----------------------------------------------------------------------===//
// spirv.ISub
//===----------------------------------------------------------------------===//

OpFoldResult spirv::ISubOp::fold(FoldAdaptor adaptor) {
  // x - x = 0
  if (getOperand1() == getOperand2())
    return Builder(getContext()).getIntegerAttr(getType(), 0);

  // According to the SPIR-V spec:
  //
  // The resulting value will equal the low-order N bits of the correct result
  // R, where N is the component width and R is computed with enough precision
  // to avoid overflow and underflow.
  return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(),
      [](APInt a, const APInt &b) { return std::move(a) - b; });
}

//===----------------------------------------------------------------------===//
// spirv.LogicalAnd
//===----------------------------------------------------------------------===//

OpFoldResult spirv::LogicalAndOp::fold(FoldAdaptor adaptor) {
  if (std::optional<bool> rhs =
          getScalarOrSplatBoolAttr(adaptor.getOperand2())) {
    // x && true = x
    if (*rhs)
      return getOperand1();

    // x && false = false
    if (!*rhs)
      return adaptor.getOperand2();
  }

  return Attribute();
}

//===----------------------------------------------------------------------===//
// spirv.LogicalNotEqualOp
//===----------------------------------------------------------------------===//

OpFoldResult spirv::LogicalNotEqualOp::fold(FoldAdaptor adaptor) {
  if (std::optional<bool> rhs =
          getScalarOrSplatBoolAttr(adaptor.getOperand2())) {
    // x && false = x
    if (!rhs.value())
      return getOperand1();
  }

  return Attribute();
}

//===----------------------------------------------------------------------===//
// spirv.LogicalNot
//===----------------------------------------------------------------------===//

void spirv::LogicalNotOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results
      .add<ConvertLogicalNotOfIEqual, ConvertLogicalNotOfINotEqual,
           ConvertLogicalNotOfLogicalEqual, ConvertLogicalNotOfLogicalNotEqual>(
          context);
}

//===----------------------------------------------------------------------===//
// spirv.LogicalOr
//===----------------------------------------------------------------------===//

OpFoldResult spirv::LogicalOrOp::fold(FoldAdaptor adaptor) {
  if (auto rhs = getScalarOrSplatBoolAttr(adaptor.getOperand2())) {
    if (*rhs) {
      // x || true = true
      return adaptor.getOperand2();
    }

    if (!*rhs) {
      // x || false = x
      return getOperand1();
    }
  }

  return Attribute();
}

//===----------------------------------------------------------------------===//
// spirv.mlir.selection
//===----------------------------------------------------------------------===//

namespace {
// Blocks from the given `spirv.mlir.selection` operation must satisfy the
// following layout:
//
//       +-----------------------------------------------+
//       | header block                                  |
//       | spirv.BranchConditionalOp %cond, ^case0, ^case1 |
//       +-----------------------------------------------+
//                            /   \
//                             ...
//
//
//   +------------------------+    +------------------------+
//   | case #0                |    | case #1                |
//   | spirv.Store %ptr %value0 |    | spirv.Store %ptr %value1 |
//   | spirv.Branch ^merge      |    | spirv.Branch ^merge      |
//   +------------------------+    +------------------------+
//
//
//                             ...
//                            \   /
//                              v
//                       +-------------+
//                       | merge block |
//                       +-------------+
//
struct ConvertSelectionOpToSelect final : OpRewritePattern<spirv::SelectionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::SelectionOp selectionOp,
                                PatternRewriter &rewriter) const override {
    Operation *op = selectionOp.getOperation();
    Region &body = op->getRegion(0);
    // Verifier allows an empty region for `spirv.mlir.selection`.
    if (body.empty()) {
      return failure();
    }

    // Check that region consists of 4 blocks:
    // header block, `true` block, `false` block and merge block.
    if (llvm::range_size(body) != 4) {
      return failure();
    }

    Block *headerBlock = selectionOp.getHeaderBlock();
    if (!onlyContainsBranchConditionalOp(headerBlock)) {
      return failure();
    }

    auto brConditionalOp =
        cast<spirv::BranchConditionalOp>(headerBlock->front());

    Block *trueBlock = brConditionalOp.getSuccessor(0);
    Block *falseBlock = brConditionalOp.getSuccessor(1);
    Block *mergeBlock = selectionOp.getMergeBlock();

    if (failed(canCanonicalizeSelection(trueBlock, falseBlock, mergeBlock)))
      return failure();

    Value trueValue = getSrcValue(trueBlock);
    Value falseValue = getSrcValue(falseBlock);
    Value ptrValue = getDstPtr(trueBlock);
    auto storeOpAttributes =
        cast<spirv::StoreOp>(trueBlock->front())->getAttrs();

    auto selectOp = rewriter.create<spirv::SelectOp>(
        selectionOp.getLoc(), trueValue.getType(),
        brConditionalOp.getCondition(), trueValue, falseValue);
    rewriter.create<spirv::StoreOp>(selectOp.getLoc(), ptrValue,
                                    selectOp.getResult(), storeOpAttributes);

    // `spirv.mlir.selection` is not needed anymore.
    rewriter.eraseOp(op);
    return success();
  }

private:
  // Checks that given blocks follow the following rules:
  // 1. Each conditional block consists of two operations, the first operation
  //    is a `spirv.Store` and the last operation is a `spirv.Branch`.
  // 2. Each `spirv.Store` uses the same pointer and the same memory attributes.
  // 3. A control flow goes into the given merge block from the given
  //    conditional blocks.
  LogicalResult canCanonicalizeSelection(Block *trueBlock, Block *falseBlock,
                                         Block *mergeBlock) const;

  bool onlyContainsBranchConditionalOp(Block *block) const {
    return llvm::hasSingleElement(*block) &&
           isa<spirv::BranchConditionalOp>(block->front());
  }

  bool isSameAttrList(spirv::StoreOp lhs, spirv::StoreOp rhs) const {
    return lhs->getDiscardableAttrDictionary() ==
               rhs->getDiscardableAttrDictionary() &&
           lhs.getProperties() == rhs.getProperties();
  }

  // Returns a source value for the given block.
  Value getSrcValue(Block *block) const {
    auto storeOp = cast<spirv::StoreOp>(block->front());
    return storeOp.getValue();
  }

  // Returns a destination value for the given block.
  Value getDstPtr(Block *block) const {
    auto storeOp = cast<spirv::StoreOp>(block->front());
    return storeOp.getPtr();
  }
};

LogicalResult ConvertSelectionOpToSelect::canCanonicalizeSelection(
    Block *trueBlock, Block *falseBlock, Block *mergeBlock) const {
  // Each block must consists of 2 operations.
  if (llvm::range_size(*trueBlock) != 2 || llvm::range_size(*falseBlock) != 2) {
    return failure();
  }

  auto trueBrStoreOp = dyn_cast<spirv::StoreOp>(trueBlock->front());
  auto trueBrBranchOp =
      dyn_cast<spirv::BranchOp>(*std::next(trueBlock->begin()));
  auto falseBrStoreOp = dyn_cast<spirv::StoreOp>(falseBlock->front());
  auto falseBrBranchOp =
      dyn_cast<spirv::BranchOp>(*std::next(falseBlock->begin()));

  if (!trueBrStoreOp || !trueBrBranchOp || !falseBrStoreOp ||
      !falseBrBranchOp) {
    return failure();
  }

  // Checks that given type is valid for `spirv.SelectOp`.
  // According to SPIR-V spec:
  // "Before version 1.4, Result Type must be a pointer, scalar, or vector.
  // Starting with version 1.4, Result Type can additionally be a composite type
  // other than a vector."
  bool isScalarOrVector =
      llvm::cast<spirv::SPIRVType>(trueBrStoreOp.getValue().getType())
          .isScalarOrVector();

  // Check that each `spirv.Store` uses the same pointer, memory access
  // attributes and a valid type of the value.
  if ((trueBrStoreOp.getPtr() != falseBrStoreOp.getPtr()) ||
      !isSameAttrList(trueBrStoreOp, falseBrStoreOp) || !isScalarOrVector) {
    return failure();
  }

  if ((trueBrBranchOp->getSuccessor(0) != mergeBlock) ||
      (falseBrBranchOp->getSuccessor(0) != mergeBlock)) {
    return failure();
  }

  return success();
}
} // namespace

void spirv::SelectionOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results.add<ConvertSelectionOpToSelect>(context);
}
