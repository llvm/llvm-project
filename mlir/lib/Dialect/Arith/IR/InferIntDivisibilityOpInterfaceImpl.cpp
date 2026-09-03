//===- InferIntDivisibilityOpInterfaceImpl.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Direct implementations of `InferIntDivisibilityOpInterface` for arith ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/InferIntDivisibilityOpInterface.h"

#include <cstdlib>

using namespace mlir;
using namespace mlir::arith;

static ConstantIntDivisibility
getDivisibilityOfOperand(Value v, IntegerDivisibility divisibility) {
  if (!divisibility.isUninitialized())
    return divisibility.getValue();
  APInt intVal;
  if (matchPattern(v, m_ConstantInt(&intVal))) {
    uint64_t udiv = intVal.getZExtValue();
    uint64_t sdiv = std::abs(intVal.getSExtValue());
    return ConstantIntDivisibility(udiv, sdiv);
  }
  return ConstantIntDivisibility(1, 1);
}

// Result divisibility is the GCD (union) of the operand divisibilities.
template <typename OpTy>
static void
inferBinaryGCDResultDivisibility(OpTy op, ArrayRef<IntegerDivisibility> argDivs,
                                 SetIntDivisibilityFn setResultDivs) {
  auto lhsDiv = getDivisibilityOfOperand(op.getLhs(), argDivs[0]);
  auto rhsDiv = getDivisibilityOfOperand(op.getRhs(), argDivs[1]);
  setResultDivs(op.getResult(), lhsDiv.getUnion(rhsDiv));
}

void ConstantOp::inferResultDivisibility(ArrayRef<IntegerDivisibility> argDivs,
                                         SetIntDivisibilityFn setResultDivs) {
  auto constAttr = dyn_cast_if_present<IntegerAttr>(getValue());
  if (!constAttr)
    return;
  const APInt &value = constAttr.getValue();
  uint64_t udiv = value.getZExtValue();
  uint64_t sdiv = std::abs(value.getSExtValue());
  setResultDivs(getResult(), ConstantIntDivisibility(udiv, sdiv));
}

void AddIOp::inferResultDivisibility(ArrayRef<IntegerDivisibility> argDivs,
                                     SetIntDivisibilityFn setResultDivs) {
  inferBinaryGCDResultDivisibility(*this, argDivs, setResultDivs);
}

void SubIOp::inferResultDivisibility(ArrayRef<IntegerDivisibility> argDivs,
                                     SetIntDivisibilityFn setResultDivs) {
  inferBinaryGCDResultDivisibility(*this, argDivs, setResultDivs);
}

void MinUIOp::inferResultDivisibility(ArrayRef<IntegerDivisibility> argDivs,
                                      SetIntDivisibilityFn setResultDivs) {
  inferBinaryGCDResultDivisibility(*this, argDivs, setResultDivs);
}

void MaxUIOp::inferResultDivisibility(ArrayRef<IntegerDivisibility> argDivs,
                                      SetIntDivisibilityFn setResultDivs) {
  inferBinaryGCDResultDivisibility(*this, argDivs, setResultDivs);
}

void MinSIOp::inferResultDivisibility(ArrayRef<IntegerDivisibility> argDivs,
                                      SetIntDivisibilityFn setResultDivs) {
  inferBinaryGCDResultDivisibility(*this, argDivs, setResultDivs);
}

void MaxSIOp::inferResultDivisibility(ArrayRef<IntegerDivisibility> argDivs,
                                      SetIntDivisibilityFn setResultDivs) {
  inferBinaryGCDResultDivisibility(*this, argDivs, setResultDivs);
}

void MulIOp::inferResultDivisibility(ArrayRef<IntegerDivisibility> argDivs,
                                     SetIntDivisibilityFn setResultDivs) {
  auto lhsDivisibility = getDivisibilityOfOperand(getLhs(), argDivs[0]);
  auto rhsDivisibility = getDivisibilityOfOperand(getRhs(), argDivs[1]);

  uint64_t mulUDiv = lhsDivisibility.udiv() * rhsDivisibility.udiv();
  uint64_t mulSDiv = lhsDivisibility.sdiv() * rhsDivisibility.sdiv();

  setResultDivs(getResult(), ConstantIntDivisibility(mulUDiv, mulSDiv));
}

void DivUIOp::inferResultDivisibility(ArrayRef<IntegerDivisibility> argDivs,
                                      SetIntDivisibilityFn setResultDivs) {
  APInt intVal;
  if (!matchPattern(getRhs(), m_ConstantInt(&intVal)))
    return;

  auto lhsDivisibility = getDivisibilityOfOperand(getLhs(), argDivs[0]);

  uint64_t divUDiv = lhsDivisibility.udiv() % intVal.getZExtValue() == 0
                         ? lhsDivisibility.udiv() / intVal.getZExtValue()
                         : 1;
  uint64_t divSDiv =
      lhsDivisibility.sdiv() % std::abs(intVal.getSExtValue()) == 0
          ? lhsDivisibility.sdiv() / std::abs(intVal.getSExtValue())
          : 1;

  setResultDivs(getResult(), ConstantIntDivisibility(divUDiv, divSDiv));
}

void SelectOp::inferResultDivisibility(ArrayRef<IntegerDivisibility> argDivs,
                                       SetIntDivisibilityFn setResultDivs) {
  // argDivs[0] is the condition (i1), argDivs[1] is true, argDivs[2] is false.
  auto trueDiv = getDivisibilityOfOperand(getTrueValue(), argDivs[1]);
  auto falseDiv = getDivisibilityOfOperand(getFalseValue(), argDivs[2]);
  setResultDivs(getResult(), trueDiv.getUnion(falseDiv));
}
