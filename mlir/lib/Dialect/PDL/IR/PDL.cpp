//===- PDL.cpp - Pattern Descriptor Language Dialect ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::pdl;

#include "mlir/Dialect/PDL/IR/PDLOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// PDLDialect
//===----------------------------------------------------------------------===//

void PDLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/PDL/IR/PDLOps.cpp.inc"
      >();
  registerTypes();
}

//===----------------------------------------------------------------------===//
// PDL Operations
//===----------------------------------------------------------------------===//

/// Returns true if the given operation is used by a "binding" pdl operation
/// within the main matcher body of a `pdl.pattern`.
static bool hasBindingUseInMatcher(Operation *op, Block *matcherBlock) {
  for (OpOperand &use : op->getUses()) {
    Operation *user = use.getOwner();
    if (user->getBlock() != matcherBlock)
      continue;
    if (isa<AttributeOp, OperandOp, OperandsOp, OperationOp>(user))
      return true;
    // Only the first operand of RewriteOp may be bound to, i.e. the root
    // operation of the pattern.
    if (isa<RewriteOp>(user) && use.getOperandNumber() == 0)
      return true;
    // A result by itself is not binding, it must also be bound.
    if (isa<ResultOp, ResultsOp>(user) &&
        hasBindingUseInMatcher(user, matcherBlock))
      return true;
  }
  return false;
}

/// Returns success if the given operation is used by a "binding" pdl operation
/// within the main matcher body of a `pdl.pattern`. On failure, emits an error
/// with the given context message.
static LogicalResult
verifyHasBindingUseInMatcher(Operation *op,
                             StringRef bindableContextStr = "`pdl.operation`") {
  // If the pattern is not a pattern, there is nothing to do.
  if (!isa<PatternOp>(op->getParentOp()))
    return success();
  if (hasBindingUseInMatcher(op, op->getBlock()))
    return success();
  return op->emitOpError()
         << "expected a bindable (i.e. " << bindableContextStr
         << ") user when defined in the matcher body of a `pdl.pattern`";
}

//===----------------------------------------------------------------------===//
// pdl::ApplyNativeConstraintOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ApplyNativeConstraintOp op) {
  if (op.getNumOperands() == 0)
    return op.emitOpError("expected at least one argument");
  return success();
}

//===----------------------------------------------------------------------===//
// pdl::ApplyNativeRewriteOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ApplyNativeRewriteOp op) {
  if (op.getNumOperands() == 0 && op.getNumResults() == 0)
    return op.emitOpError("expected at least one argument or result");
  return success();
}

//===----------------------------------------------------------------------===//
// pdl::AttributeOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(AttributeOp op) {
  Value attrType = op.type();
  Optional<Attribute> attrValue = op.value();

  if (!attrValue && isa<RewriteOp>(op->getParentOp()))
    return op.emitOpError("expected constant value when specified within a "
                          "`pdl.rewrite`");
  if (attrValue && attrType)
    return op.emitOpError("expected only one of [`type`, `value`] to be set");
  return verifyHasBindingUseInMatcher(op);
}

//===----------------------------------------------------------------------===//
// pdl::OperandOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(OperandOp op) {
  return verifyHasBindingUseInMatcher(op);
}

//===----------------------------------------------------------------------===//
// pdl::OperandsOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(OperandsOp op) {
  return verifyHasBindingUseInMatcher(op);
}

//===----------------------------------------------------------------------===//
// pdl::OperationOp
//===----------------------------------------------------------------------===//

static ParseResult parseOperationOpAttributes(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::OperandType> &attrOperands,
    ArrayAttr &attrNamesAttr) {
  Builder &builder = p.getBuilder();
  SmallVector<Attribute, 4> attrNames;
  if (succeeded(p.parseOptionalLBrace())) {
    do {
      StringAttr nameAttr;
      OpAsmParser::OperandType operand;
      if (p.parseAttribute(nameAttr) || p.parseEqual() ||
          p.parseOperand(operand))
        return failure();
      attrNames.push_back(nameAttr);
      attrOperands.push_back(operand);
    } while (succeeded(p.parseOptionalComma()));
    if (p.parseRBrace())
      return failure();
  }
  attrNamesAttr = builder.getArrayAttr(attrNames);
  return success();
}

static void printOperationOpAttributes(OpAsmPrinter &p, OperationOp op,
                                       OperandRange attrArgs,
                                       ArrayAttr attrNames) {
  if (attrNames.empty())
    return;
  p << " {";
  interleaveComma(llvm::seq<int>(0, attrNames.size()), p,
                  [&](int i) { p << attrNames[i] << " = " << attrArgs[i]; });
  p << '}';
}

/// Verifies that the result types of this operation, defined within a
/// `pdl.rewrite`, can be inferred.
static LogicalResult verifyResultTypesAreInferrable(OperationOp op,
                                                    OperandRange resultTypes) {
  // Functor that returns if the given use can be used to infer a type.
  Block *rewriterBlock = op->getBlock();
  auto canInferTypeFromUse = [&](OpOperand &use) {
    // If the use is within a ReplaceOp and isn't the operation being replaced
    // (i.e. is not the first operand of the replacement), we can infer a type.
    ReplaceOp replOpUser = dyn_cast<ReplaceOp>(use.getOwner());
    if (!replOpUser || use.getOperandNumber() == 0)
      return false;
    // Make sure the replaced operation was defined before this one.
    Operation *replacedOp = replOpUser.operation().getDefiningOp();
    return replacedOp->getBlock() != rewriterBlock ||
           replacedOp->isBeforeInBlock(op);
  };

  // Check to see if the uses of the operation itself can be used to infer
  // types.
  if (llvm::any_of(op.op().getUses(), canInferTypeFromUse))
    return success();

  // Otherwise, make sure each of the types can be inferred.
  for (auto it : llvm::enumerate(resultTypes)) {
    Operation *resultTypeOp = it.value().getDefiningOp();
    assert(resultTypeOp && "expected valid result type operation");

    // If the op was defined by a `apply_native_rewrite`, it is guaranteed to be
    // usable.
    if (isa<ApplyNativeRewriteOp>(resultTypeOp))
      continue;

    // If the type operation was defined in the matcher and constrains the
    // result of an input operation, it can be used.
    auto constrainsInputOp = [rewriterBlock](Operation *user) {
      return user->getBlock() != rewriterBlock && isa<OperationOp>(user);
    };
    if (TypeOp typeOp = dyn_cast<TypeOp>(resultTypeOp)) {
      if (typeOp.type() || llvm::any_of(typeOp->getUsers(), constrainsInputOp))
        continue;
    } else if (TypesOp typeOp = dyn_cast<TypesOp>(resultTypeOp)) {
      if (typeOp.types() || llvm::any_of(typeOp->getUsers(), constrainsInputOp))
        continue;
    }

    return op
        .emitOpError("must have inferable or constrained result types when "
                     "nested within `pdl.rewrite`")
        .attachNote()
        .append("result type #", it.index(), " was not constrained");
  }
  return success();
}

static LogicalResult verify(OperationOp op) {
  bool isWithinRewrite = isa<RewriteOp>(op->getParentOp());
  if (isWithinRewrite && !op.name())
    return op.emitOpError("must have an operation name when nested within "
                          "a `pdl.rewrite`");
  ArrayAttr attributeNames = op.attributeNames();
  auto attributeValues = op.attributes();
  if (attributeNames.size() != attributeValues.size()) {
    return op.emitOpError()
           << "expected the same number of attribute values and attribute "
              "names, got "
           << attributeNames.size() << " names and " << attributeValues.size()
           << " values";
  }

  // If the operation is within a rewrite body and doesn't have type inference,
  // ensure that the result types can be resolved.
  if (isWithinRewrite && !op.hasTypeInference()) {
    if (failed(verifyResultTypesAreInferrable(op, op.types())))
      return failure();
  }

  return verifyHasBindingUseInMatcher(op, "`pdl.operation` or `pdl.rewrite`");
}

bool OperationOp::hasTypeInference() {
  Optional<StringRef> opName = name();
  if (!opName)
    return false;

  OperationName name(*opName, getContext());
  if (const AbstractOperation *op = name.getAbstractOperation())
    return op->getInterface<InferTypeOpInterface>();
  return false;
}

//===----------------------------------------------------------------------===//
// pdl::PatternOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(PatternOp pattern) {
  Region &body = pattern.body();
  auto *term = body.front().getTerminator();
  if (!isa<RewriteOp>(term)) {
    return pattern.emitOpError("expected body to terminate with `pdl.rewrite`")
        .attachNote(term->getLoc())
        .append("see terminator defined here");
  }

  // Check that all values defined in the top-level pattern are referenced at
  // least once in the source tree.
  WalkResult result = body.walk([&](Operation *op) -> WalkResult {
    if (!isa_and_nonnull<PDLDialect>(op->getDialect())) {
      pattern
          .emitOpError("expected only `pdl` operations within the pattern body")
          .attachNote(op->getLoc())
          .append("see non-`pdl` operation defined here");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

void PatternOp::build(OpBuilder &builder, OperationState &state,
                      Optional<StringRef> rootKind, Optional<uint16_t> benefit,
                      Optional<StringRef> name) {
  build(builder, state,
        rootKind ? builder.getStringAttr(*rootKind) : StringAttr(),
        builder.getI16IntegerAttr(benefit ? *benefit : 0),
        name ? builder.getStringAttr(*name) : StringAttr());
  state.regions[0]->emplaceBlock();
}

/// Returns the rewrite operation of this pattern.
RewriteOp PatternOp::getRewriter() {
  return cast<RewriteOp>(body().front().getTerminator());
}

/// Return the root operation kind that this pattern matches, or None if
/// there isn't a specific root.
Optional<StringRef> PatternOp::getRootKind() {
  OperationOp rootOp = cast<OperationOp>(getRewriter().root().getDefiningOp());
  return rootOp.name();
}

//===----------------------------------------------------------------------===//
// pdl::ReplaceOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ReplaceOp op) {
  if (op.replOperation() && !op.replValues().empty())
    return op.emitOpError() << "expected no replacement values to be provided"
                               " when the replacement operation is present";
  return success();
}

//===----------------------------------------------------------------------===//
// pdl::ResultsOp
//===----------------------------------------------------------------------===//

static ParseResult parseResultsValueType(OpAsmParser &p, IntegerAttr index,
                                         Type &resultType) {
  if (!index) {
    resultType = RangeType::get(p.getBuilder().getType<ValueType>());
    return success();
  }
  if (p.parseArrow() || p.parseType(resultType))
    return failure();
  return success();
}

static void printResultsValueType(OpAsmPrinter &p, ResultsOp op,
                                  IntegerAttr index, Type resultType) {
  if (index)
    p << " -> " << resultType;
}

static LogicalResult verify(ResultsOp op) {
  if (!op.index() && op.getType().isa<pdl::ValueType>()) {
    return op.emitOpError() << "expected `pdl.range<value>` result type when "
                               "no index is specified, but got: "
                            << op.getType();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// pdl::RewriteOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(RewriteOp op) {
  Region &rewriteRegion = op.body();

  // Handle the case where the rewrite is external.
  if (op.name()) {
    if (!rewriteRegion.empty()) {
      return op.emitOpError()
             << "expected rewrite region to be empty when rewrite is external";
    }
    return success();
  }

  // Otherwise, check that the rewrite region only contains a single block.
  if (rewriteRegion.empty()) {
    return op.emitOpError() << "expected rewrite region to be non-empty if "
                               "external name is not specified";
  }

  // Check that no additional arguments were provided.
  if (!op.externalArgs().empty()) {
    return op.emitOpError() << "expected no external arguments when the "
                               "rewrite is specified inline";
  }
  if (op.externalConstParams()) {
    return op.emitOpError() << "expected no external constant parameters when "
                               "the rewrite is specified inline";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// pdl::TypeOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(TypeOp op) {
  return verifyHasBindingUseInMatcher(
      op, "`pdl.attribute`, `pdl.operand`, or `pdl.operation`");
}

//===----------------------------------------------------------------------===//
// pdl::TypesOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(TypesOp op) {
  return verifyHasBindingUseInMatcher(op, "`pdl.operands`, or `pdl.operation`");
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/PDL/IR/PDLOps.cpp.inc"
