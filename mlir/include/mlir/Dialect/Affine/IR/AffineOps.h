//===- AffineOps.h - MLIR Affine Operations -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines convenience types for working with Affine operations
// in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_IR_AFFINEOPS_H
#define MLIR_DIALECT_AFFINE_IR_AFFINEOPS_H

#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
namespace mlir {
namespace affine {

class AffineApplyOp;
class AffineBound;
class AffineMaxOp;
class AffineMinOp;
class AffineValueMap;

/// A utility function to check if a value is defined at the top level of an
/// op with trait `AffineScope` or is a region argument for such an op. A value
/// of index type defined at the top level is always a valid symbol for all its
/// uses.
bool isTopLevelValue(Value value);

/// A utility function to check if a value is defined at the top level of
/// `region` or is an argument of `region`. A value of index type defined at the
/// top level of a `AffineScope` region is always a valid symbol for all
/// uses in that region.
bool isTopLevelValue(Value value, Region *region);

/// Returns the closest region enclosing `op` that is held by an operation with
/// trait `AffineScope`; `nullptr` if there is no such region.
Region *getAffineScope(Operation *op);

/// Returns the closest region enclosing `op` that is held by a non-affine
/// operation; `nullptr` if there is no such region. This method is meant to
/// be used by affine analysis methods (e.g. dependence analysis) which are
/// only meaningful when performed among/between operations from the same
/// analysis scope.
Region *getAffineAnalysisScope(Operation *op);

/// Return the product of `terms`, creating an `affine.apply` if any of them are
/// non-constant values. If any of `terms` is `nullptr`, return `nullptr`.
OpFoldResult computeProduct(Location loc, OpBuilder &builder,
                            ArrayRef<OpFoldResult> terms);

/// Returns true if the given Value can be used as a dimension id in the region
/// of the closest surrounding op that has the trait `AffineScope`.
bool isValidDim(Value value);

/// Returns true if the given Value can be used as a dimension id in `region`,
/// i.e., for all its uses in `region`.
bool isValidDim(Value value, Region *region);

/// Returns true if the given value can be used as a symbol in the region of the
/// closest surrounding op that has the trait `AffineScope`.
bool isValidSymbol(Value value);

/// Returns true if the given Value can be used as a symbol for `region`, i.e.,
/// for all its uses in `region`.
bool isValidSymbol(Value value, Region *region);

/// Parses dimension and symbol list. `numDims` is set to the number of
/// dimensions in the list parsed.
ParseResult parseDimAndSymbolList(OpAsmParser &parser,
                                  SmallVectorImpl<Value> &operands,
                                  unsigned &numDims);

/// Modifies both `map` and `operands` in-place so as to:
/// 1. drop duplicate operands
/// 2. drop unused dims and symbols from map
/// 3. promote valid symbols to symbolic operands in case they appeared as
///    dimensional operands
/// 4. propagate constant operands and drop them
void canonicalizeMapAndOperands(AffineMap *map,
                                SmallVectorImpl<Value> *operands);

/// Canonicalizes an integer set the same way canonicalizeMapAndOperands does
/// for affine maps.
void canonicalizeSetAndOperands(IntegerSet *set,
                                SmallVectorImpl<Value> *operands);

/// Returns a composed AffineApplyOp by composing `map` and `operands` with
/// other AffineApplyOps supplying those operands. The operands of the resulting
/// AffineApplyOp do not change the length of  AffineApplyOp chains.
AffineApplyOp makeComposedAffineApply(OpBuilder &b, Location loc, AffineMap map,
                                      ArrayRef<OpFoldResult> operands,
                                      bool composeAffineMin = false);
AffineApplyOp makeComposedAffineApply(OpBuilder &b, Location loc, AffineExpr e,
                                      ArrayRef<OpFoldResult> operands,
                                      bool composeAffineMin = false);

/// Constructs an AffineApplyOp that applies `map` to `operands` after composing
/// the map with the maps of any other AffineApplyOp supplying the operands,
/// then immediately attempts to fold it. If folding results in a constant
/// value, no ops are actually created. The `map` must be a single-result affine
/// map.
OpFoldResult makeComposedFoldedAffineApply(OpBuilder &b, Location loc,
                                           AffineMap map,
                                           ArrayRef<OpFoldResult> operands,
                                           bool composeAffineMin = false);
/// Variant of `makeComposedFoldedAffineApply` that applies to an expression.
OpFoldResult makeComposedFoldedAffineApply(OpBuilder &b, Location loc,
                                           AffineExpr expr,
                                           ArrayRef<OpFoldResult> operands,
                                           bool composeAffineMin = false);
/// Variant of `makeComposedFoldedAffineApply` suitable for multi-result maps.
/// Note that this may create as many affine.apply operations as the map has
/// results given that affine.apply must be single-result.
SmallVector<OpFoldResult> makeComposedFoldedMultiResultAffineApply(
    OpBuilder &b, Location loc, AffineMap map, ArrayRef<OpFoldResult> operands,
    bool composeAffineMin = false);

/// Returns an AffineMinOp obtained by composing `map` and `operands` with
/// AffineApplyOps supplying those operands.
AffineMinOp makeComposedAffineMin(OpBuilder &b, Location loc, AffineMap map,
                                  ArrayRef<OpFoldResult> operands);

/// Constructs an AffineMinOp that computes a minimum across the results of
/// applying `map` to `operands`, then immediately attempts to fold it. If
/// folding results in a constant value, no ops are actually created.
OpFoldResult makeComposedFoldedAffineMin(OpBuilder &b, Location loc,
                                         AffineMap map,
                                         ArrayRef<OpFoldResult> operands);

/// Constructs an AffineMinOp that computes a maximum across the results of
/// applying `map` to `operands`, then immediately attempts to fold it. If
/// folding results in a constant value, no ops are actually created.
OpFoldResult makeComposedFoldedAffineMax(OpBuilder &b, Location loc,
                                         AffineMap map,
                                         ArrayRef<OpFoldResult> operands);

/// Given an affine map `map` and its input `operands`, this method composes
/// into `map`, maps of AffineApplyOps whose results are the values in
/// `operands`, iteratively until no more of `operands` are the result of an
/// AffineApplyOp. When this function returns, `map` becomes the composed affine
/// map, and each Value in `operands` is guaranteed to be either a loop IV or a
/// terminal symbol, i.e., a symbol defined at the top level or a block/function
/// argument.
void fullyComposeAffineMapAndOperands(AffineMap *map,
                                      SmallVectorImpl<Value> *operands,
                                      bool composeAffineMin = false);

} // namespace affine
} // namespace mlir

#include "mlir/Dialect/Affine/IR/AffineOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Affine/IR/AffineOps.h.inc"

namespace mlir {
namespace affine {

/// Returns true if the provided value is the induction variable of an
/// AffineForOp.
bool isAffineForInductionVar(Value val);

/// Returns true if `val` is the induction variable of an AffineParallelOp.
bool isAffineParallelInductionVar(Value val);

/// Returns true if the provided value is the induction variable of an
/// AffineForOp or AffineParallelOp.
bool isAffineInductionVar(Value val);

/// Returns the loop parent of an induction variable. If the provided value is
/// not an induction variable, then return nullptr.
AffineForOp getForInductionVarOwner(Value val);

/// Returns true if the provided value is among the induction variables of an
/// AffineParallelOp.
AffineParallelOp getAffineParallelInductionVarOwner(Value val);

/// Extracts the induction variables from a list of AffineForOps and places them
/// in the output argument `ivs`.
void extractForInductionVars(ArrayRef<AffineForOp> forInsts,
                             SmallVectorImpl<Value> *ivs);

/// Extracts the induction variables from a list of either AffineForOp or
/// AffineParallelOp and places them in the output argument `ivs`.
void extractInductionVars(ArrayRef<Operation *> affineOps,
                          SmallVectorImpl<Value> &ivs);

/// Builds a perfect nest of affine.for loops, i.e., each loop except the
/// innermost one contains only another loop and a terminator. The loops iterate
/// from "lbs" to "ubs" with "steps". The body of the innermost loop is
/// populated by calling "bodyBuilderFn" and providing it with an OpBuilder, a
/// Location and a list of loop induction variables.
void buildAffineLoopNest(OpBuilder &builder, Location loc,
                         ArrayRef<int64_t> lbs, ArrayRef<int64_t> ubs,
                         ArrayRef<int64_t> steps,
                         function_ref<void(OpBuilder &, Location, ValueRange)>
                             bodyBuilderFn = nullptr);
void buildAffineLoopNest(OpBuilder &builder, Location loc, ValueRange lbs,
                         ValueRange ubs, ArrayRef<int64_t> steps,
                         function_ref<void(OpBuilder &, Location, ValueRange)>
                             bodyBuilderFn = nullptr);

/// AffineBound represents a lower or upper bound in the for operation.
/// This class does not own the underlying operands. Instead, it refers
/// to the operands stored in the AffineForOp. Its life span should not exceed
/// that of the for operation it refers to.
class AffineBound {
public:
  AffineForOp getAffineForOp() { return op; }
  AffineMap getMap() { return map; }

  unsigned getNumOperands() { return operands.size(); }
  Value getOperand(unsigned idx) {
    return op.getOperand(operands.getBeginOperandIndex() + idx);
  }

  using operand_iterator = AffineForOp::operand_iterator;
  using operand_range = AffineForOp::operand_range;

  operand_iterator operandBegin() { return operands.begin(); }
  operand_iterator operandEnd() { return operands.end(); }
  operand_range getOperands() { return {operandBegin(), operandEnd()}; }

private:
  // 'affine.for' operation that contains this bound.
  AffineForOp op;
  // Operands of the affine map.
  OperandRange operands;
  // Affine map for this bound.
  AffineMap map;

  AffineBound(AffineForOp op, OperandRange operands, AffineMap map)
      : op(op), operands(operands), map(map) {}

  friend class AffineForOp;
};

} // namespace affine
} // namespace mlir

#endif
