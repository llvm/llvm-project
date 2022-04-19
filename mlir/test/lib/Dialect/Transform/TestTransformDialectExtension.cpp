//===- TestTransformDialectExtension.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an extension of the MLIR Transform dialect for testing
// purposes.
//
//===----------------------------------------------------------------------===//

#include "TestTransformDialectExtension.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;

namespace {
/// Simple transform op defined outside of the dialect. Just emits a remark when
/// applied.
class TestTransformOp
    : public Op<TestTransformOp, transform::TransformOpInterface::Trait> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTransformOp)

  using Op::Op;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static constexpr llvm::StringLiteral getOperationName() {
    return llvm::StringLiteral("transform.test_transform_op");
  }

  LogicalResult apply(transform::TransformResults &results,
                      transform::TransformState &state) {
    InFlightDiagnostic remark = emitRemark() << "applying transformation";
    if (Attribute message = getMessage())
      remark << " " << message;

    return success();
  }

  Attribute getMessage() { return getOperation()->getAttr("message"); }

  static ParseResult parse(OpAsmParser &parser, OperationState &state) {
    StringAttr message;
    OptionalParseResult result = parser.parseOptionalAttribute(message);
    if (!result.hasValue())
      return success();

    if (result.getValue().succeeded())
      state.addAttribute("message", message);
    return result.getValue();
  }

  void print(OpAsmPrinter &printer) {
    if (getMessage())
      printer << " " << getMessage();
  }
};
} // namespace

LogicalResult mlir::test::TestProduceParamOrForwardOperandOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  if (getOperation()->getNumOperands() != 0) {
    results.set(getResult().cast<OpResult>(),
                getOperation()->getOperand(0).getDefiningOp());
  } else {
    results.set(getResult().cast<OpResult>(),
                reinterpret_cast<Operation *>(*getParameter()));
  }
  return success();
}

LogicalResult mlir::test::TestProduceParamOrForwardOperandOp::verify() {
  if (getParameter().hasValue() ^ (getNumOperands() != 1))
    return emitOpError() << "expects either a parameter or an operand";
  return success();
}

LogicalResult mlir::test::TestConsumeOperandIfMatchesParamOrFail::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  ArrayRef<Operation *> payload = state.getPayloadOps(getOperand());
  assert(payload.size() == 1 && "expected a single target op");
  auto value = reinterpret_cast<intptr_t>(payload[0]);
  if (static_cast<uint64_t>(value) != getParameter()) {
    return emitOpError() << "expected the operand to be associated with "
                         << getParameter() << " got " << value;
  }

  emitRemark() << "succeeded";
  return success();
}

namespace {
/// Test extension of the Transform dialect. Registers additional ops and
/// declares PDL as dependent dialect since the additional ops are using PDL
/// types for operands and results.
class TestTransformDialectExtension
    : public transform::TransformDialectExtension<
          TestTransformDialectExtension> {
public:
  TestTransformDialectExtension() {
    declareDependentDialect<pdl::PDLDialect>();
    registerTransformOps<TestTransformOp,
#define GET_OP_LIST
#include "TestTransformDialectExtension.cpp.inc"
                         >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "TestTransformDialectExtension.cpp.inc"

void ::test::registerTestTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<TestTransformDialectExtension>();
}
