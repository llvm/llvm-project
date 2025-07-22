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
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

FailureOr<func::FuncOp>
func::replaceFuncWithNewOrder(RewriterBase &rewriter, func::FuncOp funcOp,
                              ArrayRef<unsigned> newArgsOrder,
                              ArrayRef<unsigned> newResultsOrder) {
  // Generate an empty new function operation with the same name as the
  // original.
  assert(funcOp.getNumArguments() == newArgsOrder.size() &&
         "newArgsOrder must match the number of arguments in the function");
  assert(funcOp.getNumResults() == newResultsOrder.size() &&
         "newResultsOrder must match the number of results in the function");

  if (!funcOp.getBody().hasOneBlock())
    return rewriter.notifyMatchFailure(
        funcOp, "expected function to have exactly one block");

  ArrayRef<Type> origInputTypes = funcOp.getFunctionType().getInputs();
  ArrayRef<Type> origOutputTypes = funcOp.getFunctionType().getResults();
  SmallVector<Type> newInputTypes, newOutputTypes;
  SmallVector<Location> locs;
  for (unsigned int idx : newArgsOrder) {
    newInputTypes.push_back(origInputTypes[idx]);
    locs.push_back(funcOp.getArgument(newArgsOrder[idx]).getLoc());
  }
  for (unsigned int idx : newResultsOrder)
    newOutputTypes.push_back(origOutputTypes[idx]);
  rewriter.setInsertionPoint(funcOp);
  auto newFuncOp = func::FuncOp::create(
      rewriter, funcOp.getLoc(), funcOp.getName(),
      rewriter.getFunctionType(newInputTypes, newOutputTypes));

  Region &newRegion = newFuncOp.getBody();
  rewriter.createBlock(&newRegion, newRegion.begin(), newInputTypes, locs);
  newFuncOp.setVisibility(funcOp.getVisibility());
  newFuncOp->setDiscardableAttrs(funcOp->getDiscardableAttrDictionary());

  // Map the arguments of the original function to the new function in
  // the new order and adjust the attributes accordingly.
  IRMapping operandMapper;
  SmallVector<DictionaryAttr> argAttrs, resultAttrs;
  funcOp.getAllArgAttrs(argAttrs);
  for (unsigned int i = 0; i < newArgsOrder.size(); ++i) {
    operandMapper.map(funcOp.getArgument(newArgsOrder[i]),
                      newFuncOp.getArgument(i));
    newFuncOp.setArgAttrs(i, argAttrs[newArgsOrder[i]]);
  }
  funcOp.getAllResultAttrs(resultAttrs);
  for (unsigned int i = 0; i < newResultsOrder.size(); ++i)
    newFuncOp.setResultAttrs(i, resultAttrs[newResultsOrder[i]]);

  // Clone the operations from the original function to the new function.
  rewriter.setInsertionPointToStart(&newFuncOp.getBody().front());
  for (Operation &op : funcOp.getOps())
    rewriter.clone(op, operandMapper);

  // Handle the return operation.
  auto returnOp = cast<func::ReturnOp>(
      newFuncOp.getFunctionBody().begin()->getTerminator());
  SmallVector<Value> newReturnValues;
  for (unsigned int idx : newResultsOrder)
    newReturnValues.push_back(returnOp.getOperand(idx));
  rewriter.setInsertionPoint(returnOp);
  auto newReturnOp =
      func::ReturnOp::create(rewriter, newFuncOp.getLoc(), newReturnValues);
  newReturnOp->setDiscardableAttrs(returnOp->getDiscardableAttrDictionary());
  rewriter.eraseOp(returnOp);

  rewriter.eraseOp(funcOp);

  return newFuncOp;
}

func::CallOp
func::replaceCallOpWithNewOrder(RewriterBase &rewriter, func::CallOp callOp,
                                ArrayRef<unsigned> newArgsOrder,
                                ArrayRef<unsigned> newResultsOrder) {
  assert(
      callOp.getNumOperands() == newArgsOrder.size() &&
      "newArgsOrder must match the number of operands in the call operation");
  assert(
      callOp.getNumResults() == newResultsOrder.size() &&
      "newResultsOrder must match the number of results in the call operation");
  SmallVector<Value> newArgsOrderValues;
  for (unsigned int argIdx : newArgsOrder)
    newArgsOrderValues.push_back(callOp.getOperand(argIdx));
  SmallVector<Type> newResultTypes;
  for (unsigned int resIdx : newResultsOrder)
    newResultTypes.push_back(callOp.getResult(resIdx).getType());

  // Replace the kernel call operation with a new one that has the
  // reordered arguments.
  rewriter.setInsertionPoint(callOp);
  auto newCallOp =
      func::CallOp::create(rewriter, callOp.getLoc(), callOp.getCallee(),
                           newResultTypes, newArgsOrderValues);
  newCallOp.setNoInlineAttr(callOp.getNoInlineAttr());
  for (auto &&[newIndex, origIndex] : llvm::enumerate(newResultsOrder))
    rewriter.replaceAllUsesWith(callOp.getResult(origIndex),
                                newCallOp.getResult(newIndex));
  rewriter.eraseOp(callOp);

  return newCallOp;
}
