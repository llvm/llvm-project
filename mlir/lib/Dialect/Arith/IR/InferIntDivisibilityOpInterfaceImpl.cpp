//===- InferIntDivisibilityOpInterfaceImpl.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/InferIntDivisibilityOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/InferIntDivisibilityOpInterface.h"

#include <cstdlib>

using namespace mlir;

namespace {

static ConstantIntDivisibility
getDivisibilityOfOperand(Value v, IntegerDivisibility divisibility) {
  if (!divisibility.isUninitialized()) {
    return divisibility.getValue();
  }
  APInt intVal;
  if (matchPattern(v, m_ConstantInt(&intVal))) {
    uint64_t udiv = intVal.getZExtValue();
    uint64_t sdiv = std::abs(intVal.getSExtValue());
    return ConstantIntDivisibility(udiv, sdiv);
  }
  return ConstantIntDivisibility(1, 1);
}

/// Helper for binary arith ops whose result divisibility is the GCD (union) of
/// their operands' divisibilities. This covers add, sub, min, and max.
template <typename OpTy>
struct ArithBinaryGCDInferIntDivisibilityOpInterface
    : InferIntDivisibilityOpInterface::ExternalModel<
          ArithBinaryGCDInferIntDivisibilityOpInterface<OpTy>, OpTy> {

  void inferResultDivisibility(Operation *op,
                               ArrayRef<IntegerDivisibility> argDivs,
                               SetIntDivisibilityFn setResultDivs) const {
    auto binOp = cast<OpTy>(op);
    auto lhsDiv = getDivisibilityOfOperand(binOp.getLhs(), argDivs[0]);
    auto rhsDiv = getDivisibilityOfOperand(binOp.getRhs(), argDivs[1]);
    setResultDivs(binOp.getResult(), lhsDiv.getUnion(rhsDiv));
  }
};

/// For arith.select, the result divisibility is the GCD of the true and false
/// operands' divisibilities. The condition (operand 0) is i1 and irrelevant.
struct ArithSelectInferIntDivisibilityOpInterface
    : InferIntDivisibilityOpInterface::ExternalModel<
          ArithSelectInferIntDivisibilityOpInterface, arith::SelectOp> {

  void inferResultDivisibility(Operation *op,
                               ArrayRef<IntegerDivisibility> argDivs,
                               SetIntDivisibilityFn setResultDivs) const {
    auto selectOp = cast<arith::SelectOp>(op);
    // argDivs[0] is the condition (i1), argDivs[1] is true, argDivs[2] is
    // false.
    auto trueDiv =
        getDivisibilityOfOperand(selectOp.getTrueValue(), argDivs[1]);
    auto falseDiv =
        getDivisibilityOfOperand(selectOp.getFalseValue(), argDivs[2]);
    setResultDivs(selectOp.getResult(), trueDiv.getUnion(falseDiv));
  }
};

struct ArithConstantInferIntDivisibilityOpInterface
    : InferIntDivisibilityOpInterface::ExternalModel<
          ArithConstantInferIntDivisibilityOpInterface, arith::ConstantOp> {

  void inferResultDivisibility(Operation *op,
                               ArrayRef<IntegerDivisibility> argDivs,
                               SetIntDivisibilityFn setResultDivs) const {
    auto constOp = cast<arith::ConstantOp>(op);
    auto constAttr = dyn_cast_if_present<IntegerAttr>(constOp.getValue());
    if (constAttr) {
      const APInt &value = constAttr.getValue();
      uint64_t udiv = value.getZExtValue();
      uint64_t sdiv = std::abs(value.getSExtValue());
      setResultDivs(constOp.getResult(), ConstantIntDivisibility(udiv, sdiv));
    }
  }
};

struct ArithMulIInferIntDivisibilityOpInterface
    : InferIntDivisibilityOpInterface::ExternalModel<
          ArithMulIInferIntDivisibilityOpInterface, arith::MulIOp> {

  void inferResultDivisibility(Operation *op,
                               ArrayRef<IntegerDivisibility> argDivs,
                               SetIntDivisibilityFn setResultDivs) const {
    auto mulOp = cast<arith::MulIOp>(op);

    auto lhsDivisibility = getDivisibilityOfOperand(mulOp.getLhs(), argDivs[0]);
    auto rhsDivisibility = getDivisibilityOfOperand(mulOp.getRhs(), argDivs[1]);

    uint64_t mulUDiv = lhsDivisibility.udiv() * rhsDivisibility.udiv();
    uint64_t mulSDiv = lhsDivisibility.sdiv() * rhsDivisibility.sdiv();

    setResultDivs(mulOp.getResult(), ConstantIntDivisibility(mulUDiv, mulSDiv));
  }
};

struct ArithDivUIInferIntDivisibilityOpInterface
    : InferIntDivisibilityOpInterface::ExternalModel<
          ArithDivUIInferIntDivisibilityOpInterface, arith::DivUIOp> {

  void inferResultDivisibility(Operation *op,
                               ArrayRef<IntegerDivisibility> argDivs,
                               SetIntDivisibilityFn setResultDivs) const {
    auto divOp = cast<arith::DivUIOp>(op);

    APInt intVal;
    if (!matchPattern(divOp.getRhs(), m_ConstantInt(&intVal))) {
      return;
    }

    auto lhsDivisibility = getDivisibilityOfOperand(divOp.getLhs(), argDivs[0]);

    uint64_t divUDiv = lhsDivisibility.udiv() % intVal.getZExtValue() == 0
                           ? lhsDivisibility.udiv() / intVal.getZExtValue()
                           : 1;
    uint64_t divSDiv =
        lhsDivisibility.sdiv() % std::abs(intVal.getSExtValue()) == 0
            ? lhsDivisibility.sdiv() / std::abs(intVal.getSExtValue())
            : 1;

    setResultDivs(divOp, ConstantIntDivisibility(divUDiv, divSDiv));
  }
};

} // namespace

void mlir::arith::registerInferIntDivisibilityOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, ArithDialect *dialect) {
    ConstantOp::attachInterface<ArithConstantInferIntDivisibilityOpInterface>(
        *context);
    MulIOp::attachInterface<ArithMulIInferIntDivisibilityOpInterface>(*context);
    DivUIOp::attachInterface<ArithDivUIInferIntDivisibilityOpInterface>(
        *context);
    AddIOp::attachInterface<
        ArithBinaryGCDInferIntDivisibilityOpInterface<AddIOp>>(*context);
    SubIOp::attachInterface<
        ArithBinaryGCDInferIntDivisibilityOpInterface<SubIOp>>(*context);
    MinUIOp::attachInterface<
        ArithBinaryGCDInferIntDivisibilityOpInterface<MinUIOp>>(*context);
    MaxUIOp::attachInterface<
        ArithBinaryGCDInferIntDivisibilityOpInterface<MaxUIOp>>(*context);
    MinSIOp::attachInterface<
        ArithBinaryGCDInferIntDivisibilityOpInterface<MinSIOp>>(*context);
    MaxSIOp::attachInterface<
        ArithBinaryGCDInferIntDivisibilityOpInterface<MaxSIOp>>(*context);
    SelectOp::attachInterface<ArithSelectInferIntDivisibilityOpInterface>(
        *context);
  });
}
