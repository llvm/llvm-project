//===- TosaReduceTransposes.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ----------
// Motivation:
// ----------

// Some legalization pathways introduce redundant tosa.TRANSPOSE
// operations that result in avoidable data movement. For example,
// PyTorch -> TOSA contains a lot of unnecessary transposes due
// to conversions between NCHW and NHWC.

// We wish to remove all the ones that we can, since in general
// it is possible to remove the overwhelming majority.

// -------------------
// High-Level Overview:
// -------------------

// The pass works through the transpose operators in the program. It begins at
// some transpose operator with an associated permutations tensor. It traverses
// upwards through the dependencies of this transpose and verifies that we
// encounter only operators with the TosaElementwiseOperator trait and terminate
// in either constants, reshapes, or transposes.

// We then evaluate whether there are any additional restrictions (the
// transposes it terminates in must invert the one we began at, and the reshapes
// must be ones in which we can fold the transpose into), and then we hoist the
// transpose through the intervening operators, folding it at the constants,
// reshapes, and transposes.

// Finally, we ensure that we do not need both the transposed form (the form
// that had the transpose hoisted through it) and the untransposed form (which
// it was prior), by analyzing the usages of those dependent operators of a
// given transpose we are attempting to hoist and replace.

// If they are such that it would require both forms to be necessary, then we do
// not replace the hoisted transpose, causing the new chain to be dead.
// Otherwise, we do and the old chain (untransposed form) becomes dead. Only one
// chain will ever then be live, resulting in no duplication.

// We then perform a simple one-pass DCE, so no canonicalization is necessary.

// -----------
// Future Work:
// -----------

// (1) Evaluate tradeoffs with permitting ConstOp to be duplicated across
// hoisted
//     transposes with different permutation tensors.

// (2) Expand the class of foldable upstream ReshapeOp we permit beyond
//     N -> 1x1x...x1xNx1x...x1x1.

// (3) Enchance the pass to permit folding arbitrary transpose pairs, beyond
//     those that form the identity.

// (4) Add support for more instructions besides TosaElementwiseOperator as
//     the intervening ones (for example, the reduce_* operators).

// (5) Support hoisting transposes up to an input parameter.

//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/TypeSwitch.h"
#include <memory>
#include <set>
#include <stack>

namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSAREDUCETRANSPOSES
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace mlir

using namespace mlir;
using namespace mlir::tosa;

//===----------------------------------------------------------------------===//
// TOSA Reduce Transposes Pass.
//===----------------------------------------------------------------------===//

namespace {

struct TosaReduceTransposes final
    : public tosa::impl::TosaReduceTransposesBase<TosaReduceTransposes> {
  void runOnOperation() override;

private:
  // This will collect all the data dependencies for the given Operation
  // up to and including ConstOp, ReshapeOp, and TransposeOp.
  bool collectFanIn(Operation *op, SetVector<Operation *> &collected);
  bool convertDependentOps(SetVector<Operation *> &dependentOps,
                           DenseMap<Value, Value> &valuesMap,
                           IRRewriter &rewriter,
                           ArrayRef<int32_t> hoistedPerms);

  // Checks if the two permutations, when applied consecutively, result
  // in the identity.
  bool areInvolutionTransposes(ArrayRef<int32_t> perms1,
                               ArrayRef<int32_t> perms2);

  // This is meant to apply to operations with the TosaElementwiseOperator
  // trait.
  std::optional<Value>
  buildMappedToValue(Operation *op, const DenseMap<Value, Value> &valuesMap,
                     IRRewriter &rewriter, ArrayRef<int32_t> hoistedPerms);

  // This updates valuesMap when we encounter another TransposeOp as a
  // dependency of the hoisted one. %0 = tosa.transpose %arg0 <- applies to
  // this %1 = tosa.transpose %0 <- when tracking back from this
  std::optional<Value>
  buildMappedToValue(TransposeOp transposeOp,
                     const DenseMap<Value, Value> &valuesMap,
                     IRRewriter &rewriter, ArrayRef<int32_t> hoistedPerms);

  // Checks if ReshapeOp can have hoisted TransposeOp folded into it. If so,
  // it creates new ReshapeOp with that fold.
  std::optional<Value>
  buildMappedToValue(ReshapeOp reshapeOp,
                     const DenseMap<Value, Value> &valuesMap,
                     IRRewriter &rewriter, ArrayRef<int32_t> hoistedPerms);

  // We may have something like:
  // %0 = tosa.const
  // %1 = tosa.transpose
  // %2 = tosa.add %0, %1
  // %3 = tosa.transpose %2
  // that --tosa-layerwise-const-fold wouldn't handle. This use shows up
  // in MobilenetV3.
  std::optional<Value>
  buildMappedToValue(ConstOp constOp, const DenseMap<Value, Value> &valuesMap,
                     IRRewriter &rewriter, ArrayRef<int32_t> hoistedPerms);

  // Checks which TransposeOp we should "replace", turning their converted
  // chains of ops, through which they were propagated, "live", and the old code
  // "dead." Attempts to avoid doing so when doing so would result in the old
  // code staying "live," resulting in duplication.
  std::set<TransposeOp> getGoodReplacements(
      ArrayRef<int32_t> perms,
      std::vector<std::pair<TransposeOp, SetVector<Operation *>>>
          &transposeInfo);

  // Helper function for dependenciesAreValid.
  bool userNotContainedInValidTransposeDependencies(
      Operation *user, std::set<TransposeOp> &validTransposes,
      std::vector<std::pair<TransposeOp, SetVector<Operation *>>>
          &transposeInfo);

  // Helper function for getGoodReplacements to check if some TransposeOp's
  // dependencies are OK.
  bool dependenciesAreValid(
      ArrayRef<int32_t> perms, const SetVector<Operation *> &dependentOps,
      std::set<TransposeOp> &validTransposes,
      std::vector<std::pair<TransposeOp, SetVector<Operation *>>>
          &transposeInfo);

  // Applies perms to the DenseElementsAttr.
  // If it returns std::nullopt, it also triggers pass failure, since verifier
  // guarantees from TOSA are not in place (and otherwise, if used elsewhere,
  // it should fail).
  // This is a basic API and may benefit from refactor into the core MLIR APIs.
  std::optional<DenseElementsAttr>
  transposeDenseAttribute(DenseElementsAttr input, ArrayRef<int32_t> perms);
};

std::optional<DenseElementsAttr>
TosaReduceTransposes::transposeDenseAttribute(DenseElementsAttr input,
                                              ArrayRef<int32_t> perms) {
  RankedTensorType oldType = llvm::cast<RankedTensorType>(input.getType());
  RankedTensorType newType =
      RankedTensorType::get(applyTOSAPermutation(oldType.getShape(), perms),
                            oldType.getElementType());
  size_t rank = oldType.getRank();

  // Asserted by TransposeOp verifier and TOSA disallowing tensor with dimension
  // 0. If not in place, something is very wrong.
  if (rank <= 0 || oldType.getNumElements() <= 0) {
    signalPassFailure();
    return std::nullopt;
  }

  if (input.isSplat())
    return input.reshape(newType);

  // The algorithm is approximately as follows:
  // input: perms, input flat array, input tensor type
  // (1/2) determine the strides of input/output if
  // they were strided in row-major order. (3) adjust the strides for the
  // input to be in the same order of indices as the output is written.
  // (4) process dimension by dimension. example: perms 2, 0, 1; input
  // 2x3x4; output 4x2x3 for i ... 4, j ... 2, k ... 3: output[i][j][k] =
  // input[j][k][i] output[6i + 3j + k] = input[12j + 4k + i] and we adjust
  // input strides to be as input[i + 12j + 4k] so we may process
  // layer-by-layer.

  // Step 1/2: Strides for input. We ignore output since row-major and can just
  // push_back.

  SmallVector<int64_t> originalInputStrides(rank);
  originalInputStrides[rank - 1] = 1;
  // index with int64_t to avoid overflow
  for (int64_t i = rank - 2; i >= 0; i--)
    originalInputStrides[i] =
        originalInputStrides[i + 1] * oldType.getDimSize(i + 1);

  // Step 3: Transpose strides of input to be same indexing (i, j, k, ...) as
  // output which is done in row-major order.

  SmallVector<int64_t> newInputStrides;
  newInputStrides.reserve(rank);
  for (int32_t v : perms)
    newInputStrides.push_back(originalInputStrides[v]);

  // Step 4: Write out the transposed "flat array" dimension by dimension.

  auto inputArray = input.getValues<Attribute>();
  SmallVector<std::pair<int64_t, int64_t>> boundsAndStrides;
  for (size_t i = 0; i < rank; i++)
    boundsAndStrides.push_back({newType.getDimSize(i), newInputStrides[i]});

  SmallVector<Attribute> resultArray;
  resultArray.reserve(inputArray.size());

  std::function<void(int64_t,
                     SmallVector<std::pair<int64_t, int64_t>>::const_iterator)>
      processTransposeDim = [&](auto accumulatedIndex, auto it) {
        if (it == boundsAndStrides.end()) {
          resultArray.push_back(inputArray[accumulatedIndex]);
          return;
        }

        for (int64_t i = 0; i < it->first; i++) {
          int64_t j = accumulatedIndex + i * it->second;
          processTransposeDim(j, it + 1);
        }
      };

  processTransposeDim(0, boundsAndStrides.begin());

  return DenseElementsAttr::get(newType, resultArray);
}

// The SetVector should only contain ConstOp, ReshapeOp, TransposeOp
// as the sources of the data dependencies, and TosaElementWiseOperator
// after that, if the function returns true.
bool TosaReduceTransposes::collectFanIn(Operation *op,
                                        SetVector<Operation *> &collected) {
  // Can occur if defined through the parameter to a func.func.
  if (!op)
    return false;

  if (!llvm::isa_and_present<tosa::TosaDialect>(op->getDialect()))
    return false;

  // Prevent extra work if already seen.
  if (collected.contains(op))
    return true;

  // Throw it out so later don't have to deal with this.
  if (op->getNumResults() != 1 ||
      !llvm::isa<RankedTensorType>(op->getResult(0).getType()))
    return false;

  // We don't wish to traverse up a ReshapeOp, since generally we can't
  // propagate a TransposeOp through it.  TransposeOp, ReshapeOp, ConstOp
  // will have no in-edges in the data dependency graph we construct for
  // the downstream TransposeOp.
  if (!llvm::isa<tosa::TransposeOp>(op) && !llvm::isa<tosa::ReshapeOp>(op) &&
      !llvm::isa<tosa::ConstOp>(op)) {

    if (!llvm::isa<tosa::MulOp>(op) &&
        !op->hasTrait<OpTrait::tosa::TosaElementwiseOperator>())
      return false;

    for (Value operand : op->getOperands()) {
      // If this is a problem in future, think about alternatives to recursion.
      if (llvm::isa<tosa::MulOp>(op) && op->getNumOperands() == 3 &&
          operand == op->getOperand(2)) {
        // do not recurse into MulOp's shift operand
        continue;
      }
      if (!collectFanIn(operand.getDefiningOp(), collected))
        return false;
    }
  }

  // Insert in topological order.
  collected.insert(op);

  return true;
}

// Assuming that due to the verification of TransposeOp perms arrays are
// permutations of 0 - perms.size() - 1.
bool TosaReduceTransposes::areInvolutionTransposes(ArrayRef<int32_t> perms1,
                                                   ArrayRef<int32_t> perms2) {
  if (perms1.size() != perms2.size())
    return false;
  int32_t n = perms1.size();
  for (int32_t i = 0; i < n; i++)
    if (perms2[perms1[i]] != i)
      return false;
  return true;
}

// Primary overload for those with TosaElementwiseOperator trait.
// The other ones handle the case of the operations that occur at the
// roots of the data dependency graph (ConstOp, ReshapeOp, TransposeOp).
std::optional<Value> TosaReduceTransposes::buildMappedToValue(
    Operation *op, const DenseMap<Value, Value> &valuesMap,
    IRRewriter &rewriter, ArrayRef<int32_t> hoistedPerms) {
  if (op->getNumResults() != 1 ||
      (!llvm::isa<tosa::MulOp>(op) &&
       !op->hasTrait<OpTrait::tosa::TosaElementwiseOperator>()))
    return std::nullopt;

  auto resultType = llvm::cast<RankedTensorType>(op->getResult(0).getType());
  SmallVector<Value, 3> operands;
  for (Value v : op->getOperands()) {
    if (valuesMap.contains(v)) {
      operands.push_back(valuesMap.at(v));
    } else if (llvm::isa<tosa::MulOp>(op) && op->getNumOperands() == 3 &&
               v == op->getOperand(2)) {
      // special case for MulOp's shift operand
      operands.push_back(v);
    } else {
      return std::nullopt;
    }
  }

  // Conceptually, we propagate the hoisted TransposeOp through
  // these interveaning operations. For example,

  // %0 = tosa.clamp %input : (tensor<2x3xi32>) -> tensor<2x3xi32>
  // %1 = tosa.transpose %0 {perms = [1, 0]} : (tensor<2x3xi32>) ->
  // tensor<3x2xi32>

  // becomes:
  // %0 = tosa.transpose %input {perms = [1, 0]} : (tensor<2x3xi32>) ->
  // tensor<3x2xi32>
  // %1 = tosa.clamp %0 : (tensor<3x2xi32>) -> tensor<3x2xi32>)

  // We construct this new tosa.clamp here, but it doesn't
  // turn "live" until the transpose being hoisted through this chain
  // is replaced with the proper value from the new chain.

  return rewriter
      .create(op->getLoc(), op->getName().getIdentifier(), operands,
              RankedTensorType::get(
                  applyTOSAPermutation(resultType.getShape(), hoistedPerms),
                  resultType.getElementType()),
              op->getAttrs())
      ->getResult(0);
}

std::optional<Value> TosaReduceTransposes::buildMappedToValue(
    TransposeOp transposeOp, const DenseMap<Value, Value> &valuesMap,
    IRRewriter &rewriter, ArrayRef<int32_t> hoistedPerms) {
  SmallVector<int32_t> perms;
  if (failed(transposeOp.getConstantPerms(perms)) ||
      !areInvolutionTransposes(hoistedPerms, perms))
    return std::nullopt;
  return transposeOp.getInput1();
}

std::optional<Value> TosaReduceTransposes::buildMappedToValue(
    ReshapeOp reshapeOp, const DenseMap<Value, Value> &valuesMap,
    IRRewriter &rewriter, ArrayRef<int32_t> hoistedPerms) {
  auto reshapeOutput = reshapeOp.getOutput();
  auto reshapeInputType =
      llvm::dyn_cast<RankedTensorType>(reshapeOp.getInput1().getType());
  auto reshapeInputShape = reshapeInputType.getShape();
  // want reshape N -> 1x1x...x1xNx1x...x1x1
  if (!reshapeInputType || reshapeInputShape.size() != 1)
    return std::nullopt;
  auto reshapeOutputType =
      llvm::cast<RankedTensorType>(reshapeOutput.getType());

  // Instead of inserting a TransposeOp here, we check if we can fold it into
  // the ReshapeOp. There is more complex cases where this is possible, and
  // this check can be extended.

  // Checking if reshape is N -> 1x1x...x1xNx1x...x1x1
  auto shape = reshapeOutputType.getShape();
  size_t ones = llvm::count(shape, 1);
  // N == 1 and N != 1
  if (ones != shape.size() - 1 &&
      !(ones == shape.size() && reshapeInputShape[0] == 1))
    return std::nullopt;

  // Do not insert a TransposeOp, instead we fold the reshape and its attribute.
  auto foldedReshape = rewriter.create<ReshapeOp>(
      reshapeOp.getLoc(),
      RankedTensorType::get(applyTOSAPermutation(shape, hoistedPerms),
                            reshapeOutputType.getElementType()),
      reshapeOp.getInput1(),
      rewriter.getDenseI64ArrayAttr(
          applyTOSAPermutation(reshapeOp.getNewShape(), hoistedPerms)));
  return foldedReshape->getResult(0);
}

std::optional<Value> TosaReduceTransposes::buildMappedToValue(
    ConstOp constOp, const DenseMap<Value, Value> &valuesMap,
    IRRewriter &rewriter, ArrayRef<int32_t> hoistedPerms) {
  auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(constOp.getValue());
  if (!denseAttr)
    return std::nullopt;
  auto maybeNewDenseAttr = transposeDenseAttribute(denseAttr, hoistedPerms);
  if (!maybeNewDenseAttr.has_value())
    return std::nullopt;
  auto newDenseAttr = maybeNewDenseAttr.value();
  auto newConstOp = rewriter.create<ConstOp>(
      constOp.getLoc(), newDenseAttr.getType(), newDenseAttr);
  return newConstOp->getResult(0);
}

bool TosaReduceTransposes::convertDependentOps(
    SetVector<Operation *> &dependentOps, DenseMap<Value, Value> &valuesMap,
    IRRewriter &rewriter, ArrayRef<int32_t> hoistedPerms) {

  for (Operation *op : dependentOps) {
    if (!op || op->getNumResults() != 1)
      return false;

    Value priorValue = op->getResult(0);

    // It's possible on a prior transposeOp we had the same dependency and
    // already resolved it.
    if (valuesMap.contains(priorValue))
      continue;

    // Keep converted ops close to the original.
    rewriter.setInsertionPointAfter(op);

    std::optional<Value> maybeValue =
        llvm::TypeSwitch<Operation *, std::optional<Value>>(op)
            .Case<TransposeOp, ReshapeOp, ConstOp>([&](auto transposeOp) {
              return buildMappedToValue(transposeOp, valuesMap, rewriter,
                                        hoistedPerms);
            })
            .Default([&](Operation *op) {
              return buildMappedToValue(op, valuesMap, rewriter, hoistedPerms);
            });

    if (!maybeValue.has_value())
      return false;

    valuesMap[priorValue] = maybeValue.value();
  }

  return true;
}

bool TosaReduceTransposes::userNotContainedInValidTransposeDependencies(
    Operation *user, std::set<TransposeOp> &validTransposes,
    std::vector<std::pair<TransposeOp, SetVector<Operation *>>>
        &transposeInfo) {
  return llvm::none_of(
      transposeInfo,
      [&validTransposes,
       user](const std::pair<TransposeOp, SetVector<Operation *>> &info) {
        const auto &[transposeOp, dependentOps] = info;
        return validTransposes.count(transposeOp) &&
               dependentOps.contains(user);
      });
}

// Dependencies are valid for an operation if none of them occur outside
// of the proper fan-in cones of the hoisted TransposeOp with the same perms
// that we can replace. Described in more detail within.
bool TosaReduceTransposes::dependenciesAreValid(
    ArrayRef<int32_t> perms, const SetVector<Operation *> &dependentOps,
    std::set<TransposeOp> &validTransposes,
    std::vector<std::pair<TransposeOp, SetVector<Operation *>>>
        &transposeInfo) {
  for (Operation *op : dependentOps) {

    // It's OK wherever ConstOp has uses -- in the worst case, we duplicate.
    // This can be changed later if we find the memory impact is too high.
    if (llvm::isa<ConstOp>(op))
      continue;

    for (OpOperand &use : op->getUses()) {
      // Want the uses to be (1) contained in the dependentOps of other
      // validTransposes, or (2) to be directly used in a TransposeOp with the
      // same perms. For (2) it means the fan-in is a subset of our
      // dependentOps, so it is also a validTranspose that will eventually be
      // replaced.
      Operation *user = use.getOwner();
      if (auto otherTranspose = llvm::dyn_cast<TransposeOp>(user)) {
        SmallVector<int32_t> otherPerms;

        // Can later think about cases where transpose -> transpose
        // or reshape -> transpose, where the transposes are not necessarily
        // the same perms as the hoisted, if implementing a more general
        // transform. These could be permitted.
        if (failed(otherTranspose.getConstantPerms(otherPerms)) ||
            !llvm::equal(perms, otherPerms))
          return false;
      } else if (userNotContainedInValidTransposeDependencies(
                     user, validTransposes, transposeInfo)) {
        return false;
      }
    }
  }

  return true;
}

// Getting the set of TransposeOp that we can replace without causing
// the old fan-in cones of any TransposeOp to remain "live", i.e, -- not being
// dead code. This is done by iterating the set until convergence, since
// if you are used outside your own fan-in cone, it's possible to be used
// in another fan-in cone of a TransposeOp that is being replaced -- unless
// we find that that one has a usage outside of it too.
std::set<TransposeOp> TosaReduceTransposes::getGoodReplacements(
    ArrayRef<int32_t> perms,
    std::vector<std::pair<TransposeOp, SetVector<Operation *>>>
        &transposeInfo) {
  // Initially, we assume they are all good to replace,
  // and we whittle them down based on our criteria.
  std::set<TransposeOp> ableToReplace;
  for (const auto &[transposeOp, _] : transposeInfo)
    ableToReplace.insert(transposeOp);

  bool gotRid;
  do {
    gotRid = false;
    for (const auto &[transposeOp, dependentOps] : transposeInfo) {
      // We don't care about it. Already invalidated.
      if (!ableToReplace.count(transposeOp))
        continue;

      // Check for validity.
      if (!dependenciesAreValid(perms, dependentOps, ableToReplace,
                                transposeInfo)) {
        ableToReplace.erase(transposeOp);
        gotRid = true;
        break;
      }
    }

  } while (gotRid);

  return ableToReplace;
}

void TosaReduceTransposes::runOnOperation() {
  // We want to operate only within a single block.
  if (!getOperation().getRegion().hasOneBlock())
    return;

  IRRewriter rewriter(&getContext());
  // For each perms, maintain a mapping for converted ops, avoid duplication.
  DenseMap<ArrayRef<int32_t>, DenseMap<Value, Value>> permsToValues;
  // For each perms, we keep track of which TransposeOp are eligible
  // for replacement alongside their dependentOps.
  DenseMap<ArrayRef<int32_t>,
           std::vector<std::pair<TransposeOp, SetVector<Operation *>>>>
      permsToTransposeInfo;

  // Necessary for lifetime, since DenseMap keeps a copy of the ArrayRef.
  // Use SmallVector for perms (common-case is <= 4) but std::vector otherwise
  // since no guarantee of smallness.
  std::vector<SmallVector<int32_t>> collectedPerms;

  // This keeps track of the order across all eligible-for-replacement
  // TransposeOp and their perms, a necessity for the final replacements.
  std::stack<std::pair<TransposeOp, ArrayRef<int32_t>>> totalTransposeOrder;

  // We want to reserve the space up front, since SmallVector stores some data
  // internally and the ArrayRef can reference that, which we don't want to get
  // invalidated.
  size_t expectedMaxPerms = 0;
  getOperation().walk([&](TransposeOp) { expectedMaxPerms += 1; });
  collectedPerms.reserve(expectedMaxPerms);

  getOperation().walk([&](TransposeOp transposeOp) {
    SetVector<Operation *> dependentOps;
    collectedPerms.emplace_back();
    SmallVector<int32_t> &perms = collectedPerms.back();

    // Dynamic shapes are OK, but the incompatible ones will be rejected later.
    auto input = transposeOp.getInput1();
    auto output = transposeOp.getOutput();

    // However, we don't support unranked tensors.
    if (!llvm::isa<RankedTensorType>(input.getType()) ||
        !llvm::isa<RankedTensorType>(output.getType()))
      return;

    // No transformation when transpose permutation non-constant.
    if (failed(transposeOp.getConstantPerms(perms)))
      return;

    // We let --canonicalize deal with identity transpose.
    if (llvm::equal(llvm::seq<int32_t>(0, perms.size()), perms))
      return;

    // Can fail if some set of basic invariants is not met that we want to
    // perform our conversions.
    if (!collectFanIn(input.getDefiningOp(), dependentOps))
      return;

    // Want to associate valuesMap for already converted of the same perms,
    // since it's possible multiple hoisted transposes w/ different perms
    // converge on an op, which would result in different transformations.
    DenseMap<Value, Value> &valuesMap = permsToValues[perms];

    // Attempt to perform the conversions and placements into IR
    // without turning inserted code "live". Also fills out valuesMap.
    // Fails if there is an intermediary we do not support.
    if (!convertDependentOps(dependentOps, valuesMap, rewriter, perms))
      // Some additional operations may have been inserted, but will be
      // removed by dead code elimination.
      return;

    // This should not happen. If it does -- it's unexpected,
    // so we fail the pass.
    if (!valuesMap.contains(input))
      return signalPassFailure();

    // It's possible the types are not compatible (because of dynamic shapes),
    // and in these cases, want to resolve dynamic shapes before running the
    // pass.
    if (output.getType() != valuesMap.at(input).getType())
      return;

    auto &transposeInfo = permsToTransposeInfo[perms];

    // In general, we might also want to introduce "newDependentOps"
    // if there are new usages that don't fall inside the original fan-ins
    // (like the TransposeOp we insert for ReshapeOp),
    // but in this case, that is specialized enough and overlaps
    // with another direct-use TransposeOp case we need to cover anyway.
    transposeInfo.push_back({transposeOp, dependentOps});

    // This is for the final replacement across all transposes.
    totalTransposeOrder.push({transposeOp, perms});
  });

  // We want to do a full fan-in analysis on a perms-level,
  // since if we do it on a multi-perms level, and they share (due to a shared
  // dependency on a Reshape) then we would also get duplicate ops.
  // Const is special cased.
  std::set<TransposeOp> ableToReplace;
  for (auto &[perms, transposeInfo] : permsToTransposeInfo) {
    // Gives us back replacements that would never result in any duplicate
    // operations being inserted by us in the IR (i.e, our goal is only to
    // remove transposes, and not create a "new chain" to do so, but replace
    // the existing chains).
    // Ideally, --canonicalize is run before this pass, since it helps this
    // analysis by removing dead code to allow more potentially acceptable
    // transformations.
    auto goodReplacementsForPerms = getGoodReplacements(perms, transposeInfo);
    ableToReplace.insert(goodReplacementsForPerms.begin(),
                         goodReplacementsForPerms.end());
  }

  // We want to do replacement across all transposes
  // in reverse order, due to invalidation of valuesMap mappings
  // if we did it otherwise.
  while (!totalTransposeOrder.empty()) {
    auto [transposeOp, perms] = totalTransposeOrder.top();
    totalTransposeOrder.pop();

    if (ableToReplace.count(transposeOp) == 0)
      continue;

    auto &valuesMap = permsToValues[perms];
    auto input = transposeOp.getInput1();

    // The purpose of this reverse iteration
    // is to avoid valuesMap invalidation. If it happens,
    // something is wrong.
    if (!valuesMap.contains(input))
      return signalPassFailure();

    rewriter.replaceOp(transposeOp, valuesMap.at(input));
  }

  // We can remove all dead code by going in reverse.
  // This is because we would remove usages before we
  // see the users.
  getOperation().walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](Operation *op) {
        if (isOpTriviallyDead(op))
          rewriter.eraseOp(op);
      });
}

} // namespace
