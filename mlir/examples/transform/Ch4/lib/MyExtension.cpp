//===-- MyExtension.cpp - Transform dialect tutorial ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Transform dialect extension operations used in the
// Chapter 4 of the Transform dialect tutorial.
//
//===----------------------------------------------------------------------===//

#include "MyExtension.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE_MATCHER "transform-matcher"
#define DBGS_MATCHER() (llvm::dbgs() << "[" DEBUG_TYPE_MATCHER "] ")
#define DEBUG_MATCHER(x) DEBUG_WITH_TYPE(DEBUG_TYPE_MATCHER, x)

#define GET_OP_CLASSES
#include "MyExtension.cpp.inc"

//===---------------------------------------------------------------------===//
// MyExtension
//===---------------------------------------------------------------------===//

// Define a new transform dialect extension. This uses the CRTP idiom to
// identify extensions.
class MyExtension
    : public ::mlir::transform::TransformDialectExtension<MyExtension> {
public:
  // The extension must derive the base constructor.
  using Base::Base;

  // This function initializes the extension, similarly to `initialize` in
  // dialect definitions. List individual operations and dependent dialects
  // here.
  void init();
};

void MyExtension::init() {
  // Register the additional match operations with the dialect similarly to
  // other transform operations. List all operations generated from ODS. This
  // call will perform additional checks that the operations implement the
  // transform and memory effect interfaces required by the dialect interpreter
  // and assert if they do not.
  registerTransformOps<
#define GET_OP_LIST
#include "MyExtension.cpp.inc"
      >();
}

//===---------------------------------------------------------------------===//
// HasOperandSatisfyingOp
//===---------------------------------------------------------------------===//

/// Returns `true` if both types implement one of the interfaces provided as
/// template parameters.
template <typename... Tys>
static bool implementSameInterface(mlir::Type t1, mlir::Type t2) {
  return ((llvm::isa<Tys>(t1) && llvm::isa<Tys>(t2)) || ... || false);
}

/// Returns `true` if both types implement one of the transform dialect
/// interfaces.
static bool implementSameTransformInterface(mlir::Type t1, mlir::Type t2) {
  return implementSameInterface<
      mlir::transform::TransformHandleTypeInterface,
      mlir::transform::TransformParamTypeInterface,
      mlir::transform::TransformValueHandleTypeInterface>(t1, t2);
}

// Matcher ops implement `apply` similarly to other transform ops. They are not
// expected to modify payload, but use the tri-state result to signal failure or
// success to match, as well as potential irrecoverable errors.
mlir::DiagnosedSilenceableFailure
mlir::transform::HasOperandSatisfyingOp::apply(
    mlir::transform::TransformRewriter &rewriter,
    mlir::transform::TransformResults &results,
    mlir::transform::TransformState &state) {
  // For simplicity, only handle a single payload op. Actual implementations
  // can use `SingleOpMatcher` trait to simplify implementation and document
  // this expectation.
  auto payloadOps = state.getPayloadOps(getOp());
  if (!llvm::hasSingleElement(payloadOps))
    return emitSilenceableError() << "expected single payload";

  // Iterate over all operands of the payload op to see if they can be matched
  // using the body of this op.
  Operation *payload = *payloadOps.begin();
  for (OpOperand &operand : payload->getOpOperands()) {
    // Create a scope for transform values defined in the body. This corresponds
    // to the syntactic scope of the region attached to this op. Any values
    // associated with payloads from now on will be automatically dissociated
    // when this object is destroyed, i.e. at the end of the iteration.
    // Associate the block argument handle with the operand.
    auto matchScope = state.make_region_scope(getBody());
    if (failed(state.mapBlockArgument(getBody().getArgument(0),
                                      {operand.get()}))) {
      return DiagnosedSilenceableFailure::definiteFailure();
    }

    // Iterate over all nested matchers with the current mapping and see if they
    // succeed.
    bool matchSucceeded = true;
    for (Operation &matcher : getBody().front().without_terminator()) {
      // Matcher ops are applied similarly to any other transform op.
      DiagnosedSilenceableFailure diag =
          state.applyTransform(cast<TransformOpInterface>(matcher));

      // Definite failures are immediately propagated as they are irrecoverable.
      if (diag.isDefiniteFailure())
        return diag;

      // On success, keep checking the remaining conditions.
      if (diag.succeeded())
        continue;

      // Report failure-to-match for debugging purposes and stop matching this
      // operand.
      assert(diag.isSilenceableFailure());
      DEBUG_MATCHER(DBGS_MATCHER()
                    << "failed to match operand #" << operand.getOperandNumber()
                    << ": " << diag.getMessage());
      (void)diag.silence();
      matchSucceeded = false;
      break;
    }
    // If failed to match this operand, try other operands.
    if (!matchSucceeded)
      continue;

    // If we reached this point, the matching succeeded for the current operand.
    // Remap the values associated with terminator operands to be associated
    // with op results, and also map the parameter result to the operand's
    // position. Note that it is safe to do here despite the end of the scope
    // as `results` are integrated into `state` by the interpreter after `apply`
    // returns rather than immediately.
    SmallVector<SmallVector<MappedValue>> yieldedMappings;
    transform::detail::prepareValueMappings(
        yieldedMappings, getBody().front().getTerminator()->getOperands(),
        state);
    results.setParams(getPosition().cast<OpResult>(),
                      {rewriter.getI32IntegerAttr(operand.getOperandNumber())});
    for (auto &&[result, mapping] : llvm::zip(getResults(), yieldedMappings))
      results.setMappedValues(result, mapping);
    return DiagnosedSilenceableFailure::success();
  }

  // If we reached this point, none of the operands succeeded the match.
  return emitSilenceableError()
         << "none of the operands satisfied the conditions";
}

// By convention, operations implementing MatchOpInterface must not modify
// payload IR and must therefore specify that they only read operand handles and
// payload as their effects.
void mlir::transform::HasOperandSatisfyingOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  onlyReadsPayload(effects);
  onlyReadsHandle(getOp(), effects);
  producesHandle(getPosition(), effects);
  producesHandle(getResults(), effects);
}

// Verify well-formedness of the operation and emit diagnostics if it is
// ill-formed.
mlir::LogicalResult mlir::transform::HasOperandSatisfyingOp::verify() {
  mlir::Block &bodyBlock = getBody().front();
  if (bodyBlock.getNumArguments() != 1 ||
      !isa<TransformValueHandleTypeInterface>(
          bodyBlock.getArgument(0).getType())) {
    return emitOpError()
           << "expects the body to have one value handle argument";
  }
  if (bodyBlock.getTerminator()->getNumOperands() != getNumResults() - 1) {
    return emitOpError() << "expects the body to yield "
                         << (getNumResults() - 1) << " values, got "
                         << bodyBlock.getTerminator()->getNumOperands();
  }
  for (auto &&[i, operand, result] :
       llvm::enumerate(bodyBlock.getTerminator()->getOperands().getTypes(),
                       getResults().getTypes())) {
    if (implementSameTransformInterface(operand, result))
      continue;
    return emitOpError() << "expects terminator operand #" << i
                         << " and result #" << (i + 1)
                         << " to implement the same transform interface";
  }

  for (Operation &op : bodyBlock.without_terminator()) {
    if (!isa<TransformOpInterface>(op) || !isa<MatchOpInterface>(op)) {
      InFlightDiagnostic diag = emitOpError()
                                << "expects body to contain match ops";
      diag.attachNote(op.getLoc()) << "non-match operation";
      return diag;
    }
  }

  return success();
}

void registerMyExtension(::mlir::DialectRegistry &registry) {
  registry.addExtensions<MyExtension>();
}
