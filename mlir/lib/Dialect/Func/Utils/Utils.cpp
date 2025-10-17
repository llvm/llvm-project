//===- Utils.cpp - Utilities to support the Func dialect ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the Func dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "func-utils"

using namespace mlir;

/// This method creates an inverse mapping of the provided map `oldToNew`.
/// Given an array where `oldIdxToNewIdx[i] = j` means old index `i` maps
/// to new index `j`,
/// This method returns a vector where `result[j]` contains all old indices
/// that map to new index `j`.
///
/// Example:
/// ```
/// oldIdxToNewIdx = [0, 1, 2, 2, 3]
/// getInverseMapping(oldIdxToNewIdx) = [[0], [1], [2, 3], [4]]
/// ```
///
static llvm::SmallVector<llvm::SmallVector<int>>
getInverseMapping(ArrayRef<int> oldIdxToNewIdx) {
  int numOfNewIdxs = 0;
  if (!oldIdxToNewIdx.empty())
    numOfNewIdxs = 1 + *llvm::max_element(oldIdxToNewIdx);
  llvm::SmallVector<llvm::SmallVector<int>> newToOldIdxs(numOfNewIdxs);
  for (auto [oldIdx, newIdx] : llvm::enumerate(oldIdxToNewIdx))
    newToOldIdxs[newIdx].push_back(oldIdx);
  return newToOldIdxs;
}

/// This method returns a new vector of elements that are mapped from the
/// `origElements` based on the `newIdxToOldIdxs` mapping. This function assumes
/// that the `newIdxToOldIdxs` mapping is valid, i.e. for each new index, there
/// is at least one old index that maps to it. Also, It assumes that mapping to
/// the same old index has the same element in the `origElements` vector.
template <typename Element>
static SmallVector<Element> getMappedElements(
    ArrayRef<Element> origElements,
    const llvm::SmallVector<llvm::SmallVector<int>> &newIdxToOldIdxs) {
  SmallVector<Element> newElements;
  for (const auto &oldIdxs : newIdxToOldIdxs) {
    assert(llvm::all_of(oldIdxs,
                        [&origElements](int idx) -> bool {
                          return idx >= 0 &&
                                 static_cast<size_t>(idx) < origElements.size();
                        }) &&
           "idx must be less than the number of elements in the original "
           "elements");
    assert(!oldIdxs.empty() && "oldIdx must not be empty");
    Element origTypeToCheck = origElements[oldIdxs.front()];
    assert(llvm::all_of(oldIdxs,
                        [&](int idx) -> bool {
                          return origElements[idx] == origTypeToCheck;
                        }) &&
           "all oldIdxs must be equal");
    newElements.push_back(origTypeToCheck);
  }
  return newElements;
}

FailureOr<func::FuncOp>
func::replaceFuncWithNewMapping(RewriterBase &rewriter, func::FuncOp funcOp,
                                ArrayRef<int> oldArgIdxToNewArgIdx,
                                ArrayRef<int> oldResIdxToNewResIdx) {
  // Generate an empty new function operation with the same name as the
  // original.
  assert(funcOp.getNumArguments() == oldArgIdxToNewArgIdx.size() &&
         "oldArgIdxToNewArgIdx must match the number of arguments in the "
         "function");
  assert(
      funcOp.getNumResults() == oldResIdxToNewResIdx.size() &&
      "oldResIdxToNewResIdx must match the number of results in the function");

  if (!funcOp.getBody().hasOneBlock())
    return rewriter.notifyMatchFailure(
        funcOp, "expected function to have exactly one block");

  // We may have some duplicate arguments in the old function, i.e.
  // in the mapping `newArgIdxToOldArgIdxs` for some new argument index
  // there may be multiple old argument indices.
  llvm::SmallVector<llvm::SmallVector<int>> newArgIdxToOldArgIdxs =
      getInverseMapping(oldArgIdxToNewArgIdx);
  SmallVector<Type> newInputTypes = getMappedElements(
      funcOp.getFunctionType().getInputs(), newArgIdxToOldArgIdxs);

  SmallVector<Location> locs;
  for (const auto &oldArgIdxs : newArgIdxToOldArgIdxs)
    locs.push_back(funcOp.getArgument(oldArgIdxs.front()).getLoc());

  llvm::SmallVector<llvm::SmallVector<int>> newResToOldResIdxs =
      getInverseMapping(oldResIdxToNewResIdx);
  SmallVector<Type> newOutputTypes = getMappedElements(
      funcOp.getFunctionType().getResults(), newResToOldResIdxs);

  rewriter.setInsertionPoint(funcOp);
  auto newFuncOp = func::FuncOp::create(
      rewriter, funcOp.getLoc(), funcOp.getName(),
      rewriter.getFunctionType(newInputTypes, newOutputTypes));

  Region &newRegion = newFuncOp.getBody();
  rewriter.createBlock(&newRegion, newRegion.begin(), newInputTypes, locs);
  newFuncOp.setVisibility(funcOp.getVisibility());

  // Map the arguments of the original function to the new function in
  // the new order and adjust the attributes accordingly.
  IRMapping operandMapper;
  SmallVector<DictionaryAttr> argAttrs, resultAttrs;
  funcOp.getAllArgAttrs(argAttrs);
  for (auto [oldArgIdx, newArgIdx] : llvm::enumerate(oldArgIdxToNewArgIdx))
    operandMapper.map(funcOp.getArgument(oldArgIdx),
                      newFuncOp.getArgument(newArgIdx));
  for (auto [newArgIdx, oldArgIdx] : llvm::enumerate(newArgIdxToOldArgIdxs))
    newFuncOp.setArgAttrs(newArgIdx, argAttrs[oldArgIdx.front()]);

  funcOp.getAllResultAttrs(resultAttrs);
  for (auto [newResIdx, oldResIdx] : llvm::enumerate(newResToOldResIdxs))
    newFuncOp.setResultAttrs(newResIdx, resultAttrs[oldResIdx.front()]);

  // Clone the operations from the original function to the new function.
  rewriter.setInsertionPointToStart(&newFuncOp.getBody().front());
  for (Operation &op : funcOp.getOps())
    rewriter.clone(op, operandMapper);

  // Handle the return operation.
  auto returnOp = cast<func::ReturnOp>(
      newFuncOp.getFunctionBody().begin()->getTerminator());
  SmallVector<Value> newReturnValues;
  for (const auto &oldResIdxs : newResToOldResIdxs)
    newReturnValues.push_back(returnOp.getOperand(oldResIdxs.front()));

  rewriter.setInsertionPoint(returnOp);
  func::ReturnOp::create(rewriter, newFuncOp.getLoc(), newReturnValues);
  rewriter.eraseOp(returnOp);

  rewriter.eraseOp(funcOp);

  return newFuncOp;
}

func::CallOp
func::replaceCallOpWithNewMapping(RewriterBase &rewriter, func::CallOp callOp,
                                  ArrayRef<int> oldArgIdxToNewArgIdx,
                                  ArrayRef<int> oldResIdxToNewResIdx) {
  assert(callOp.getNumOperands() == oldArgIdxToNewArgIdx.size() &&
         "oldArgIdxToNewArgIdx must match the number of operands in the call "
         "operation");
  assert(callOp.getNumResults() == oldResIdxToNewResIdx.size() &&
         "oldResIdxToNewResIdx must match the number of results in the call "
         "operation");

  SmallVector<Value> origOperands = callOp.getOperands();
  SmallVector<llvm::SmallVector<int>> newArgIdxToOldArgIdxs =
      getInverseMapping(oldArgIdxToNewArgIdx);
  SmallVector<Value> newOperandsValues =
      getMappedElements<Value>(origOperands, newArgIdxToOldArgIdxs);
  SmallVector<llvm::SmallVector<int>> newResToOldResIdxs =
      getInverseMapping(oldResIdxToNewResIdx);
  SmallVector<Type> origResultTypes = llvm::to_vector(callOp.getResultTypes());
  SmallVector<Type> newResultTypes =
      getMappedElements<Type>(origResultTypes, newResToOldResIdxs);

  // Replace the kernel call operation with a new one that has the
  // mapped arguments.
  rewriter.setInsertionPoint(callOp);
  auto newCallOp =
      func::CallOp::create(rewriter, callOp.getLoc(), callOp.getCallee(),
                           newResultTypes, newOperandsValues);
  newCallOp.setNoInlineAttr(callOp.getNoInlineAttr());
  for (auto &&[oldResIdx, newResIdx] : llvm::enumerate(oldResIdxToNewResIdx))
    rewriter.replaceAllUsesWith(callOp.getResult(oldResIdx),
                                newCallOp.getResult(newResIdx));
  rewriter.eraseOp(callOp);

  return newCallOp;
}

FailureOr<std::pair<func::FuncOp, func::CallOp>>
func::deduplicateArgsOfFuncOp(RewriterBase &rewriter, func::FuncOp funcOp,
                              ModuleOp moduleOp) {
  SmallVector<func::CallOp> callOps;
  auto traversalResult = moduleOp.walk([&](func::CallOp callOp) {
    if (callOp.getCallee() == funcOp.getSymName()) {
      if (!callOps.empty())
        // Only support one callOp for now
        return WalkResult::interrupt();
      callOps.push_back(callOp);
    }
    return WalkResult::advance();
  });

  if (traversalResult.wasInterrupted()) {
    LDBG() << "function " << funcOp.getName() << " has more than one callOp";
    return failure();
  }

  if (callOps.empty()) {
    LDBG() << "function " << funcOp.getName() << " does not have any callOp";
    return failure();
  }

  func::CallOp callOp = callOps.front();

  // Create mapping for arguments (deduplicate operands)
  SmallVector<int> oldArgIdxToNewArgIdx(callOp.getNumOperands());
  llvm::DenseMap<Value, int> valueToNewArgIdx;
  for (auto [operandIdx, operand] : llvm::enumerate(callOp.getOperands())) {
    auto [iterator, inserted] = valueToNewArgIdx.insert(
        {operand, static_cast<int>(valueToNewArgIdx.size())});
    // Reduce the duplicate operands and maintain the original order.
    oldArgIdxToNewArgIdx[operandIdx] = iterator->second;
  }

  bool hasDuplicateOperands =
      valueToNewArgIdx.size() != callOp.getNumOperands();
  if (!hasDuplicateOperands) {
    LDBG() << "function " << funcOp.getName()
           << " does not have duplicate operands";
    return failure();
  }

  // Create identity mapping for results (no deduplication needed)
  SmallVector<int> oldResIdxToNewResIdx(callOp.getNumResults());
  for (int resultIdx : llvm::seq<int>(0, callOp.getNumResults()))
    oldResIdxToNewResIdx[resultIdx] = resultIdx;

  // Apply the transformation to create new function and call operations
  FailureOr<func::FuncOp> newFuncOpOrFailure = replaceFuncWithNewMapping(
      rewriter, funcOp, oldArgIdxToNewArgIdx, oldResIdxToNewResIdx);
  if (failed(newFuncOpOrFailure)) {
    LDBG() << "failed to replace function signature with name "
           << funcOp.getName() << " with new order";
    return failure();
  }

  func::CallOp newCallOp = replaceCallOpWithNewMapping(
      rewriter, callOp, oldArgIdxToNewArgIdx, oldResIdxToNewResIdx);

  return std::make_pair(*newFuncOpOrFailure, newCallOp);
}
