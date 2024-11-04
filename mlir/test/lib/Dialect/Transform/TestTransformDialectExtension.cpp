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
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/PDLExtension/PDLExtensionOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

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

  DiagnosedSilenceableFailure apply(transform::TransformRewriter &rewriter,
                                    transform::TransformResults &results,
                                    transform::TransformState &state) {
    InFlightDiagnostic remark = emitRemark() << "applying transformation";
    if (Attribute message = getMessage())
      remark << " " << message;

    return DiagnosedSilenceableFailure::success();
  }

  Attribute getMessage() {
    return getOperation()->getDiscardableAttr("message");
  }

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

  DiagnosedSilenceableFailure apply(transform::TransformRewriter &rewriter,
                                    transform::TransformResults &results,
                                    transform::TransformState &state) {
    return DiagnosedSilenceableFailure::success();
  }

  // No side effects.
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {}
};
} // namespace

DiagnosedSilenceableFailure
mlir::test::TestProduceSelfHandleOrForwardOperandOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  if (getOperation()->getNumOperands() != 0) {
    results.set(cast<OpResult>(getResult()),
                {getOperation()->getOperand(0).getDefiningOp()});
  } else {
    results.set(cast<OpResult>(getResult()), {getOperation()});
  }
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestProduceSelfHandleOrForwardOperandOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (getOperand())
    transform::onlyReadsHandle(getOperand(), effects);
  transform::producesHandle(getRes(), effects);
}

DiagnosedSilenceableFailure
mlir::test::TestProduceValueHandleToSelfOperand::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  results.setValues(llvm::cast<OpResult>(getOut()), {getIn()});
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestProduceValueHandleToSelfOperand::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getIn(), effects);
  transform::producesHandle(getOut(), effects);
  transform::onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure
mlir::test::TestProduceValueHandleToResult::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  if (target->getNumResults() <= getNumber())
    return emitSilenceableError() << "payload has no result #" << getNumber();
  results.push_back(target->getResult(getNumber()));
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestProduceValueHandleToResult::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getIn(), effects);
  transform::producesHandle(getOut(), effects);
  transform::onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure
mlir::test::TestProduceValueHandleToArgumentOfParentBlock::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  if (!target->getBlock())
    return emitSilenceableError() << "payload has no parent block";
  if (target->getBlock()->getNumArguments() <= getNumber())
    return emitSilenceableError()
           << "parent of the payload has no argument #" << getNumber();
  results.push_back(target->getBlock()->getArgument(getNumber()));
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestProduceValueHandleToArgumentOfParentBlock::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getIn(), effects);
  transform::producesHandle(getOut(), effects);
  transform::onlyReadsPayload(effects);
}

bool mlir::test::TestConsumeOperand::allowsRepeatedHandleOperands() {
  return getAllowRepeatedHandles();
}

DiagnosedSilenceableFailure
mlir::test::TestConsumeOperand::apply(transform::TransformRewriter &rewriter,
                                      transform::TransformResults &results,
                                      transform::TransformState &state) {
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestConsumeOperand::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getOperand(), effects);
  if (getSecondOperand())
    transform::consumesHandle(getSecondOperand(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure mlir::test::TestConsumeOperandOfOpKindOrFail::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  auto payload = state.getPayloadOps(getOperand());
  assert(llvm::hasSingleElement(payload) && "expected a single target op");
  if ((*payload.begin())->getName().getStringRef() != getOpKind()) {
    return emitSilenceableError()
           << "op expected the operand to be associated a payload op of kind "
           << getOpKind() << " got "
           << (*payload.begin())->getName().getStringRef();
  }

  emitRemark() << "succeeded";
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestConsumeOperandOfOpKindOrFail::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getOperand(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure
mlir::test::TestSucceedIfOperandOfOpKind::matchOperation(
    Operation *op, transform::TransformResults &results,
    transform::TransformState &state) {
  if (op->getName().getStringRef() != getOpKind()) {
    return emitSilenceableError()
           << "op expected the operand to be associated with a payload op of "
              "kind "
           << getOpKind() << " got " << op->getName().getStringRef();
  }
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestSucceedIfOperandOfOpKind::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getOperand(), effects);
  transform::onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure mlir::test::TestAddTestExtensionOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  state.addExtension<TestTransformStateExtension>(getMessageAttr());
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
mlir::test::TestCheckIfTestExtensionPresentOp::apply(
    transform::TransformRewriter &rewriter,
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

void mlir::test::TestCheckIfTestExtensionPresentOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getOperand(), effects);
  transform::onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure mlir::test::TestRemapOperandPayloadToSelfOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  auto *extension = state.getExtension<TestTransformStateExtension>();
  if (!extension)
    return emitDefiniteFailure("TestTransformStateExtension missing");

  if (failed(extension->updateMapping(
          *state.getPayloadOps(getOperand()).begin(), getOperation())))
    return DiagnosedSilenceableFailure::definiteFailure();
  if (getNumResults() > 0)
    results.set(cast<OpResult>(getResult(0)), {getOperation()});
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestRemapOperandPayloadToSelfOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getOperand(), effects);
  transform::producesHandle(getOut(), effects);
  transform::onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure mlir::test::TestRemoveTestExtensionOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  state.removeExtension<TestTransformStateExtension>();
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::test::TestReversePayloadOpsOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  auto payloadOps = state.getPayloadOps(getTarget());
  auto reversedOps = llvm::to_vector(llvm::reverse(payloadOps));
  results.set(llvm::cast<OpResult>(getResult()), reversedOps);
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::test::TestTransformOpWithRegions::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestTransformOpWithRegions::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {}

DiagnosedSilenceableFailure
mlir::test::TestBranchingTransformOpTerminator::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestBranchingTransformOpTerminator::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {}

DiagnosedSilenceableFailure mlir::test::TestEmitRemarkAndEraseOperandOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  emitRemark() << getRemark();
  for (Operation *op : state.getPayloadOps(getTarget()))
    rewriter.eraseOp(op);

  if (getFailAfterErase())
    return emitSilenceableError() << "silenceable error";
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestEmitRemarkAndEraseOperandOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure mlir::test::TestWrongNumberOfResultsOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  OperationState opState(target->getLoc(), "foo");
  results.push_back(OpBuilder(target).create(opState));
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
mlir::test::TestWrongNumberOfMultiResultsOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
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
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  OperationState opState(target->getLoc(), "foo");
  results.push_back(OpBuilder(target).create(opState));
  results.push_back(OpBuilder(target).create(opState));
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
mlir::test::TestMixedNullAndNonNullResultsOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  OperationState opState(target->getLoc(), "foo");
  results.push_back(nullptr);
  results.push_back(OpBuilder(target).create(opState));
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
mlir::test::TestMixedSuccessAndSilenceableOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  if (target->hasAttr("target_me"))
    return DiagnosedSilenceableFailure::success();
  return emitDefaultSilenceableFailure(target);
}

DiagnosedSilenceableFailure
mlir::test::TestCopyPayloadOp::apply(transform::TransformRewriter &rewriter,
                                     transform::TransformResults &results,
                                     transform::TransformState &state) {
  results.set(llvm::cast<OpResult>(getCopy()),
              state.getPayloadOps(getHandle()));
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestCopyPayloadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getHandle(), effects);
  transform::producesHandle(getCopy(), effects);
  transform::onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure mlir::transform::TestDialectOpType::checkPayload(
    Location loc, ArrayRef<Operation *> payload) const {
  if (payload.empty())
    return DiagnosedSilenceableFailure::success();

  for (Operation *op : payload) {
    if (op->getName().getDialectNamespace() != "test") {
      return emitSilenceableError(loc) << "expected the payload operation to "
                                          "belong to the 'test' dialect";
    }
  }

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::TestDialectParamType::checkPayload(
    Location loc, ArrayRef<Attribute> payload) const {
  for (Attribute attr : payload) {
    auto integerAttr = llvm::dyn_cast<IntegerAttr>(attr);
    if (integerAttr && integerAttr.getType().isSignlessInteger(32))
      continue;
    return emitSilenceableError(loc)
           << "expected the parameter to be a i32 integer attribute";
  }

  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestReportNumberOfTrackedHandlesNestedUnder::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
}

DiagnosedSilenceableFailure
mlir::test::TestReportNumberOfTrackedHandlesNestedUnder::apply(
    transform::TransformRewriter &rewriter,
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

DiagnosedSilenceableFailure
mlir::test::TestAddToParamOp::apply(transform::TransformRewriter &rewriter,
                                    transform::TransformResults &results,
                                    transform::TransformState &state) {
  SmallVector<uint32_t> values(/*Size=*/1, /*Value=*/0);
  if (Value param = getParam()) {
    values = llvm::to_vector(
        llvm::map_range(state.getParams(param), [](Attribute attr) -> uint32_t {
          return llvm::cast<IntegerAttr>(attr).getValue().getLimitedValue(
              UINT32_MAX);
        }));
  }

  Builder builder(getContext());
  SmallVector<Attribute> result = llvm::to_vector(
      llvm::map_range(values, [this, &builder](uint32_t value) -> Attribute {
        return builder.getI32IntegerAttr(value + getAddendum());
      }));
  results.setParams(llvm::cast<OpResult>(getResult()), result);
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
mlir::test::TestProduceParamWithNumberOfTestOps::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  Builder builder(getContext());
  SmallVector<Attribute> result = llvm::to_vector(
      llvm::map_range(state.getPayloadOps(getHandle()),
                      [&builder](Operation *payload) -> Attribute {
                        int32_t count = 0;
                        payload->walk([&count](Operation *op) {
                          if (op->getName().getDialectNamespace() == "test")
                            ++count;
                        });
                        return builder.getI32IntegerAttr(count);
                      }));
  results.setParams(llvm::cast<OpResult>(getResult()), result);
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
mlir::test::TestProduceParamOp::apply(transform::TransformRewriter &rewriter,
                                      transform::TransformResults &results,
                                      transform::TransformState &state) {
  results.setParams(llvm::cast<OpResult>(getResult()), getAttr());
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestProduceTransformParamOrForwardOperandOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getIn(), effects);
  transform::producesHandle(getOut(), effects);
  transform::producesHandle(getParam(), effects);
}

DiagnosedSilenceableFailure
mlir::test::TestProduceTransformParamOrForwardOperandOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    ::transform::ApplyToEachResultList &results,
    ::transform::TransformState &state) {
  Builder builder(getContext());
  if (getFirstResultIsParam()) {
    results.push_back(builder.getI64IntegerAttr(0));
  } else if (getFirstResultIsNull()) {
    results.push_back(nullptr);
  } else {
    results.push_back(*state.getPayloadOps(getIn()).begin());
  }

  if (getSecondResultIsHandle()) {
    results.push_back(*state.getPayloadOps(getIn()).begin());
  } else {
    results.push_back(builder.getI64IntegerAttr(42));
  }

  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestProduceNullPayloadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::producesHandle(getOut(), effects);
}

DiagnosedSilenceableFailure mlir::test::TestProduceNullPayloadOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  SmallVector<Operation *, 1> null({nullptr});
  results.set(llvm::cast<OpResult>(getOut()), null);
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::test::TestProduceEmptyPayloadOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  results.set(cast<OpResult>(getOut()), {});
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestProduceNullParamOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::producesHandle(getOut(), effects);
}

DiagnosedSilenceableFailure mlir::test::TestProduceNullParamOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  results.setParams(llvm::cast<OpResult>(getOut()), Attribute());
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestProduceNullValueOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::producesHandle(getOut(), effects);
}

DiagnosedSilenceableFailure mlir::test::TestProduceNullValueOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  results.setValues(llvm::cast<OpResult>(getOut()), {Value()});
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestRequiredMemoryEffectsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (getHasOperandEffect())
    transform::consumesHandle(getIn(), effects);

  if (getHasResultEffect())
    transform::producesHandle(getOut(), effects);
  else
    transform::onlyReadsHandle(getOut(), effects);

  if (getModifiesPayload())
    transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure mlir::test::TestRequiredMemoryEffectsOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  results.set(llvm::cast<OpResult>(getOut()), state.getPayloadOps(getIn()));
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestTrackedRewriteOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getIn(), effects);
  transform::modifiesPayload(effects);
}

void mlir::test::TestDummyPayloadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  for (OpResult result : getResults())
    transform::producesHandle(result, effects);
}

LogicalResult mlir::test::TestDummyPayloadOp::verify() {
  if (getFailToVerify())
    return emitOpError() << "fail_to_verify is set";
  return success();
}

DiagnosedSilenceableFailure
mlir::test::TestTrackedRewriteOp::apply(transform::TransformRewriter &rewriter,
                                        transform::TransformResults &results,
                                        transform::TransformState &state) {
  int64_t numIterations = 0;

  // `getPayloadOps` returns an iterator that skips ops that are erased in the
  // loop body. Replacement ops are not enumerated.
  for (Operation *op : state.getPayloadOps(getIn())) {
    ++numIterations;
    (void)op;

    // Erase all payload ops. The outer loop should have only one iteration.
    for (Operation *op : state.getPayloadOps(getIn())) {
      rewriter.setInsertionPoint(op);
      if (op->hasAttr("erase_me")) {
        rewriter.eraseOp(op);
        continue;
      }
      if (!op->hasAttr("replace_me")) {
        continue;
      }

      SmallVector<NamedAttribute> attributes;
      attributes.emplace_back(rewriter.getStringAttr("new_op"),
                              rewriter.getUnitAttr());
      OperationState opState(op->getLoc(), op->getName().getIdentifier(),
                             /*operands=*/ValueRange(),
                             /*types=*/op->getResultTypes(), attributes);
      Operation *newOp = rewriter.create(opState);
      rewriter.replaceOp(op, newOp->getResults());
    }
  }

  emitRemark() << numIterations << " iterations";
  return DiagnosedSilenceableFailure::success();
}

namespace {
// Test pattern to replace an operation with a new op.
class ReplaceWithNewOp : public RewritePattern {
public:
  ReplaceWithNewOp(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto newName = op->getAttrOfType<StringAttr>("replace_with_new_op");
    if (!newName)
      return failure();
    Operation *newOp = rewriter.create(
        op->getLoc(), OperationName(newName, op->getContext()).getIdentifier(),
        op->getOperands(), op->getResultTypes());
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

// Test pattern to erase an operation.
class EraseOp : public RewritePattern {
public:
  EraseOp(MLIRContext *context)
      : RewritePattern("test.erase_op", /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void mlir::test::ApplyTestPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  patterns.insert<ReplaceWithNewOp, EraseOp>(patterns.getContext());
}

void mlir::test::TestReEnterRegionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getOperands(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure
mlir::test::TestReEnterRegionOp::apply(transform::TransformRewriter &rewriter,
                                       transform::TransformResults &results,
                                       transform::TransformState &state) {

  SmallVector<SmallVector<transform::MappedValue>> mappings;
  for (BlockArgument arg : getBody().front().getArguments()) {
    mappings.emplace_back(llvm::to_vector(llvm::map_range(
        state.getPayloadOps(getOperand(arg.getArgNumber())),
        [](Operation *op) -> transform::MappedValue { return op; })));
  }

  for (int i = 0; i < 4; ++i) {
    auto scope = state.make_region_scope(getBody());
    for (BlockArgument arg : getBody().front().getArguments()) {
      if (failed(state.mapBlockArgument(arg, mappings[arg.getArgNumber()])))
        return DiagnosedSilenceableFailure::definiteFailure();
    }
    for (Operation &op : getBody().front().without_terminator()) {
      DiagnosedSilenceableFailure diag =
          state.applyTransform(cast<transform::TransformOpInterface>(op));
      if (!diag.succeeded())
        return diag;
    }
  }
  return DiagnosedSilenceableFailure::success();
}

LogicalResult mlir::test::TestReEnterRegionOp::verify() {
  if (getNumOperands() != getBody().front().getNumArguments()) {
    return emitOpError() << "expects as many operands as block arguments";
  }
  return success();
}

DiagnosedSilenceableFailure mlir::test::TestNotifyPayloadOpReplacedOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  auto originalOps = state.getPayloadOps(getOriginal());
  auto replacementOps = state.getPayloadOps(getReplacement());
  if (llvm::range_size(originalOps) != llvm::range_size(replacementOps))
    return emitSilenceableError() << "expected same number of original and "
                                     "replacement payload operations";
  for (const auto &[original, replacement] :
       llvm::zip(originalOps, replacementOps)) {
    if (failed(
            rewriter.notifyPayloadOperationReplaced(original, replacement))) {
      auto diag = emitSilenceableError()
                  << "unable to replace payload op in transform mapping";
      diag.attachNote(original->getLoc()) << "original payload op";
      diag.attachNote(replacement->getLoc()) << "replacement payload op";
      return diag;
    }
  }
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestNotifyPayloadOpReplacedOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getOriginal(), effects);
  transform::onlyReadsHandle(getReplacement(), effects);
}

DiagnosedSilenceableFailure mlir::test::TestProduceInvalidIR::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  // Provide some IR that does not verify.
  rewriter.setInsertionPointToStart(&target->getRegion(0).front());
  rewriter.create<TestDummyPayloadOp>(target->getLoc(), TypeRange(),
                                      ValueRange(), /*failToVerify=*/true);
  return DiagnosedSilenceableFailure::success();
}

void mlir::test::TestProduceInvalidIR::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

namespace {
/// Test conversion pattern that replaces ops with the "replace_with_new_op"
/// attribute with "test.new_op".
class ReplaceWithNewOpConversion : public ConversionPattern {
public:
  ReplaceWithNewOpConversion(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, RewritePattern::MatchAnyOpTypeTag(),
                          /*benefit=*/1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op->hasAttr("replace_with_new_op"))
      return failure();
    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                newResultTypes)))
      return failure();
    Operation *newOp = rewriter.create(
        op->getLoc(),
        OperationName("test.new_op", op->getContext()).getIdentifier(),
        operands, newResultTypes);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};
} // namespace

void mlir::test::ApplyTestConversionPatternsOp::populatePatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.insert<ReplaceWithNewOpConversion>(typeConverter,
                                              patterns.getContext());
}

namespace {
/// Test type converter that converts tensor types to memref types.
class TestTypeConverter : public TypeConverter {
public:
  TestTypeConverter() {
    addConversion([](Type t) { return t; });
    addConversion([](RankedTensorType type) -> Type {
      return MemRefType::get(type.getShape(), type.getElementType());
    });
    auto unrealizedCastConverter = [&](OpBuilder &builder, Type resultType,
                                       ValueRange inputs,
                                       Location loc) -> std::optional<Value> {
      if (inputs.size() != 1)
        return std::nullopt;
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    };
    addSourceMaterialization(unrealizedCastConverter);
    addTargetMaterialization(unrealizedCastConverter);
  }
};
} // namespace

std::unique_ptr<::mlir::TypeConverter>
mlir::test::TestTypeConverterOp::getTypeConverter() {
  return std::make_unique<TestTypeConverter>();
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

    auto verboseConstraint = [](PatternRewriter &rewriter,
                                ArrayRef<PDLValue> pdlValues) {
      for (const PDLValue &pdlValue : pdlValues) {
        if (Operation *op = pdlValue.dyn_cast<Operation *>()) {
          op->emitWarning() << "from PDL constraint";
        }
      }
      return success();
    };

    addDialectDataInitializer<transform::PDLMatchHooks>(
        [&](transform::PDLMatchHooks &hooks) {
          llvm::StringMap<PDLConstraintFunction> constraints;
          constraints.try_emplace("verbose_constraint", verboseConstraint);
          hooks.mergeInPDLMatchHooks(std::move(constraints));
        });
  }
};
} // namespace

// These are automatically generated by ODS but are not used as the Transform
// dialect uses a different dispatch mechanism to support dialect extensions.
LLVM_ATTRIBUTE_UNUSED static OptionalParseResult
generatedTypeParser(AsmParser &parser, StringRef *mnemonic, Type &value);
LLVM_ATTRIBUTE_UNUSED static LogicalResult
generatedTypePrinter(Type def, AsmPrinter &printer);

#define GET_TYPEDEF_CLASSES
#include "TestTransformDialectExtensionTypes.cpp.inc"

#define GET_OP_CLASSES
#include "TestTransformDialectExtension.cpp.inc"

void ::test::registerTestTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<TestTransformDialectExtension>();
}
