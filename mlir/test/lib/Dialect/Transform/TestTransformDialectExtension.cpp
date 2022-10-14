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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Compiler.h"

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
    if (!result.has_value())
      return success();

    if (result.value().succeeded())
      state.addAttribute("message", message);
    return result.value();
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
#ifndef NDEBUG
    SmallVector<Value> handles;
    assert(succeeded(state.getHandlesForPayloadOp(payload, handles)));
    assert(llvm::is_contained(handles, getOperand()) &&
           "inconsistent mapping between transform IR handles and payload IR "
           "operations");
#endif // NDEBUG
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

DiagnosedSilenceableFailure
mlir::test::TestReversePayloadOpsOp::apply(transform::TransformResults &results,
                                           transform::TransformState &state) {
  ArrayRef<Operation *> payloadOps = state.getPayloadOps(getTarget());
  auto reversedOps = llvm::to_vector(llvm::reverse(payloadOps));
  results.set(getResult().cast<OpResult>(), reversedOps);
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
    return emitSilenceableError() << "silenceable error";
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

DiagnosedSilenceableFailure
mlir::test::TestPrintNumberOfAssociatedPayloadIROps::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  if (!getHandle())
    emitRemark() << 0;
  emitRemark() << state.getPayloadOps(getHandle()).size();
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestPrintNumberOfAssociatedPayloadIROps::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getHandle(), effects);
}

DiagnosedSilenceableFailure
mlir::test::TestCopyPayloadOp::apply(transform::TransformResults &results,
                                     transform::TransformState &state) {
  results.set(getCopy().cast<OpResult>(), state.getPayloadOps(getHandle()));
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::TestDialectOpType::checkPayload(
    Location loc, ArrayRef<Operation *> payload) const {
  if (payload.empty())
    return DiagnosedSilenceableFailure::success();

  for (Operation *op : payload) {
    if (op->getName().getDialectNamespace() != "test") {
      Diagnostic diag(loc, DiagnosticSeverity::Error);
      diag << "expected the payload operation to belong to the 'test' dialect";
      return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
    }
  }

  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestReportNumberOfTrackedHandlesNestedUnder::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
}

DiagnosedSilenceableFailure
mlir::test::TestReportNumberOfTrackedHandlesNestedUnder::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  int64_t count = 0;
  for (Operation *op : state.getPayloadOps(getTarget())) {
    op->walk([&](Operation *nested) {
      SmallVector<Value> handles;
      (void)state.getHandlesForPayloadOp(nested, handles);
      count += handles.size();
    });
  }
  emitRemark() << count << " handles nested under";
  return DiagnosedSilenceableFailure::success();
}

namespace {
/// Test extension of the Transform dialect. Registers additional ops and
/// declares PDL as dependent dialect since the additional ops are using PDL
/// types for operands and results.
class TestTransformDialectExtension
    : public transform::TransformDialectExtension<
          TestTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareDependentDialect<pdl::PDLDialect>();
    registerTransformOps<TestTransformOp,
                         TestTransformUnrestrictedOpNoInterface,
#define GET_OP_LIST
#include "TestTransformDialectExtension.cpp.inc"
                         >();
    registerTypes<
#define GET_TYPEDEF_LIST
#include "TestTransformDialectExtensionTypes.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "TestTransformDialectExtension.cpp.inc"

// These are automatically generated by ODS but are not used as the Transform
// dialect uses a different dispatch mechanism to support dialect extensions.
LLVM_ATTRIBUTE_UNUSED static OptionalParseResult
generatedTypeParser(AsmParser &parser, StringRef *mnemonic, Type &value);
LLVM_ATTRIBUTE_UNUSED static LogicalResult
generatedTypePrinter(Type def, AsmPrinter &printer);

#define GET_TYPEDEF_CLASSES
#include "TestTransformDialectExtensionTypes.cpp.inc"

void ::test::registerTestTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<TestTransformDialectExtension>();
}
