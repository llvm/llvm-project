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
#include "TestTransformStateExtension.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;

namespace {
/// Simple transform op defined outside of the dialect. Just emits a remark when
/// applied. This op is defined in C++ to test that C++ definitions also work
/// for op injection into the Transform dialect.
class TestTransformOp
    : public Op<TestTransformOp, transform::TransformOpInterface::Trait,
                MemoryEffectOpInterface::Trait> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTransformOp)

  using Op::Op;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static constexpr llvm::StringLiteral getOperationName() {
    return llvm::StringLiteral("transform.test_transform_op");
  }

  DiagnosedSilenceableFailure apply(transform::TransformResults &results,
                                    transform::TransformState &state) {
    InFlightDiagnostic remark = emitRemark() << "applying transformation";
    if (Attribute message = getMessage())
      remark << " " << message;

    return DiagnosedSilenceableFailure::success();
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

  // No side effects.
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {}
};

/// A test op to exercise the verifier of the PossibleTopLevelTransformOpTrait
/// in cases where it is attached to ops that do not comply with the trait
/// requirements. This op cannot be defined in ODS because ODS generates strict
/// verifiers that overalp with those in the trait and run earlier.
class TestTransformUnrestrictedOpNoInterface
    : public Op<TestTransformUnrestrictedOpNoInterface,
                transform::PossibleTopLevelTransformOpTrait,
                transform::TransformOpInterface::Trait,
                MemoryEffectOpInterface::Trait> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestTransformUnrestrictedOpNoInterface)

  using Op::Op;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static constexpr llvm::StringLiteral getOperationName() {
    return llvm::StringLiteral(
        "transform.test_transform_unrestricted_op_no_interface");
  }

  DiagnosedSilenceableFailure apply(transform::TransformResults &results,
                                    transform::TransformState &state) {
    return DiagnosedSilenceableFailure::success();
  }

  // No side effects.
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {}
};
} // namespace

DiagnosedSilenceableFailure
mlir::test::TestProduceParamOrForwardOperandOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  if (getOperation()->getNumOperands() != 0) {
    results.set(getResult().cast<OpResult>(),
                getOperation()->getOperand(0).getDefiningOp());
  } else {
    results.set(getResult().cast<OpResult>(),
                reinterpret_cast<Operation *>(*getParameter()));
  }
  return DiagnosedSilenceableFailure::success();
}

LogicalResult mlir::test::TestProduceParamOrForwardOperandOp::verify() {
  if (getParameter().has_value() ^ (getNumOperands() != 1))
    return emitOpError() << "expects either a parameter or an operand";
  return success();
}

DiagnosedSilenceableFailure
mlir::test::TestConsumeOperand::apply(transform::TransformResults &results,
                                      transform::TransformState &state) {
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
mlir::test::TestConsumeOperandIfMatchesParamOrFail::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  ArrayRef<Operation *> payload = state.getPayloadOps(getOperand());
  assert(payload.size() == 1 && "expected a single target op");
  auto value = reinterpret_cast<intptr_t>(payload[0]);
  if (static_cast<uint64_t>(value) != getParameter()) {
    return emitSilenceableError()
           << "op expected the operand to be associated with " << getParameter()
           << " got " << value;
  }

  emitRemark() << "succeeded";
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::test::TestPrintRemarkAtOperandOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  ArrayRef<Operation *> payload = state.getPayloadOps(getOperand());
  for (Operation *op : payload)
    op->emitRemark() << getMessage();

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
mlir::test::TestAddTestExtensionOp::apply(transform::TransformResults &results,
                                          transform::TransformState &state) {
  state.addExtension<TestTransformStateExtension>(getMessageAttr());
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
mlir::test::TestCheckIfTestExtensionPresentOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  auto *extension = state.getExtension<TestTransformStateExtension>();
  if (!extension) {
    emitRemark() << "extension absent";
    return DiagnosedSilenceableFailure::success();
  }

  InFlightDiagnostic diag = emitRemark()
                            << "extension present, " << extension->getMessage();
  for (Operation *payload : state.getPayloadOps(getOperand())) {
    diag.attachNote(payload->getLoc()) << "associated payload op";
    assert(state.getHandleForPayloadOp(payload) == getOperand() &&
           "inconsistent mapping between transform IR handles and payload IR "
           "operations");
  }

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::test::TestRemapOperandPayloadToSelfOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  auto *extension = state.getExtension<TestTransformStateExtension>();
  if (!extension) {
    emitError() << "TestTransformStateExtension missing";
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  if (failed(extension->updateMapping(state.getPayloadOps(getOperand()).front(),
                                      getOperation())))
    return DiagnosedSilenceableFailure::definiteFailure();
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::test::TestRemoveTestExtensionOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  state.removeExtension<TestTransformStateExtension>();
  return DiagnosedSilenceableFailure::success();
}
DiagnosedSilenceableFailure mlir::test::TestTransformOpWithRegions::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestTransformOpWithRegions::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {}

DiagnosedSilenceableFailure
mlir::test::TestBranchingTransformOpTerminator::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestBranchingTransformOpTerminator::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {}

DiagnosedSilenceableFailure mlir::test::TestEmitRemarkAndEraseOperandOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  emitRemark() << getRemark();
  for (Operation *op : state.getPayloadOps(getTarget()))
    op->erase();

  if (getFailAfterErase())
    return emitSilenceableError() << "silencable error";
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::test::TestWrongNumberOfResultsOp::applyToOne(
    Operation *target, SmallVectorImpl<Operation *> &results,
    transform::TransformState &state) {
  OperationState opState(target->getLoc(), "foo");
  results.push_back(OpBuilder(target).create(opState));
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
mlir::test::TestWrongNumberOfMultiResultsOp::applyToOne(
    Operation *target, SmallVectorImpl<Operation *> &results,
    transform::TransformState &state) {
  static int count = 0;
  if (count++ == 0) {
    OperationState opState(target->getLoc(), "foo");
    results.push_back(OpBuilder(target).create(opState));
  }
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
mlir::test::TestCorrectNumberOfMultiResultsOp::applyToOne(
    Operation *target, SmallVectorImpl<Operation *> &results,
    transform::TransformState &state) {
  OperationState opState(target->getLoc(), "foo");
  results.push_back(OpBuilder(target).create(opState));
  results.push_back(OpBuilder(target).create(opState));
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
mlir::test::TestMixedNullAndNonNullResultsOp::applyToOne(
    Operation *target, SmallVectorImpl<Operation *> &results,
    transform::TransformState &state) {
  OperationState opState(target->getLoc(), "foo");
  results.push_back(nullptr);
  results.push_back(OpBuilder(target).create(opState));
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
mlir::test::TestMixedSuccessAndSilenceableOp::applyToOne(
    Operation *target, SmallVectorImpl<Operation *> &results,
    transform::TransformState &state) {
  if (target->hasAttr("target_me"))
    return DiagnosedSilenceableFailure::success();
  return emitDefaultSilenceableFailure(target);
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
                         TestTransformUnrestrictedOpNoInterface,
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
