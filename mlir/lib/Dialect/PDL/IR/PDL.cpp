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

//===----------------------------------------------------------------------===//
// PDLDialect
//===----------------------------------------------------------------------===//

void PDLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/PDL/IR/PDLOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/PDL/IR/PDLOpsTypes.cpp.inc"
      >();
}

/// Returns true if the given operation is used by a "binding" pdl operation
/// within the main matcher body of a `pdl.pattern`.
static LogicalResult
verifyHasBindingUseInMatcher(Operation *op,
                             StringRef bindableContextStr = "`pdl.operation`") {
  // If the pattern is not a pattern, there is nothing to do.
  if (!isa<PatternOp>(op->getParentOp()))
    return success();
  Block *matcherBlock = op->getBlock();
  for (Operation *user : op->getUsers()) {
    if (user->getBlock() != matcherBlock)
      continue;
    if (isa<AttributeOp, InputOp, OperationOp, RewriteOp>(user))
      return success();
  }
  return op->emitOpError()
         << "expected a bindable (i.e. " << bindableContextStr
         << ") user when defined in the matcher body of a `pdl.pattern`";
}

//===----------------------------------------------------------------------===//
// pdl::ApplyConstraintOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ApplyConstraintOp op) {
  if (op.getNumOperands() == 0)
    return op.emitOpError("expected at least one argument");
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
// pdl::InputOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(InputOp op) {
  return verifyHasBindingUseInMatcher(op);
}

//===----------------------------------------------------------------------===//
// pdl::OperationOp
//===----------------------------------------------------------------------===//

static ParseResult parseOperationOp(OpAsmParser &p, OperationState &state) {
  Builder &builder = p.getBuilder();

  // Parse the optional operation name.
  bool startsWithOperands = succeeded(p.parseOptionalLParen());
  bool startsWithAttributes =
      !startsWithOperands && succeeded(p.parseOptionalLBrace());
  bool startsWithOpName = false;
  if (!startsWithAttributes && !startsWithOperands) {
    StringAttr opName;
    OptionalParseResult opNameResult =
        p.parseOptionalAttribute(opName, "name", state.attributes);
    startsWithOpName = opNameResult.hasValue();
    if (startsWithOpName && failed(*opNameResult))
      return failure();
  }

  // Parse the operands.
  SmallVector<OpAsmParser::OperandType, 4> operands;
  if (startsWithOperands ||
      (!startsWithAttributes && succeeded(p.parseOptionalLParen()))) {
    if (p.parseOperandList(operands) || p.parseRParen() ||
        p.resolveOperands(operands, builder.getType<ValueType>(),
                          state.operands))
      return failure();
  }

  // Parse the attributes.
  SmallVector<Attribute, 4> attrNames;
  if (startsWithAttributes || succeeded(p.parseOptionalLBrace())) {
    SmallVector<OpAsmParser::OperandType, 4> attrOps;
    do {
      StringAttr nameAttr;
      OpAsmParser::OperandType operand;
      if (p.parseAttribute(nameAttr) || p.parseEqual() ||
          p.parseOperand(operand))
        return failure();
      attrNames.push_back(nameAttr);
      attrOps.push_back(operand);
    } while (succeeded(p.parseOptionalComma()));

    if (p.parseRBrace() ||
        p.resolveOperands(attrOps, builder.getType<AttributeType>(),
                          state.operands))
      return failure();
  }
  state.addAttribute("attributeNames", builder.getArrayAttr(attrNames));
  state.addTypes(builder.getType<OperationType>());

  // Parse the result types.
  SmallVector<OpAsmParser::OperandType, 4> opResultTypes;
  if (succeeded(p.parseOptionalArrow())) {
    if (p.parseOperandList(opResultTypes) ||
        p.resolveOperands(opResultTypes, builder.getType<TypeType>(),
                          state.operands))
      return failure();
    state.types.append(opResultTypes.size(), builder.getType<ValueType>());
  }

  if (p.parseOptionalAttrDict(state.attributes))
    return failure();

  int32_t operandSegmentSizes[] = {static_cast<int32_t>(operands.size()),
                                   static_cast<int32_t>(attrNames.size()),
                                   static_cast<int32_t>(opResultTypes.size())};
  state.addAttribute("operand_segment_sizes",
                     builder.getI32VectorAttr(operandSegmentSizes));
  return success();
}

static void print(OpAsmPrinter &p, OperationOp op) {
  p << "pdl.operation ";
  if (Optional<StringRef> name = op.name())
    p << '"' << *name << '"';

  auto operandValues = op.operands();
  if (!operandValues.empty())
    p << '(' << operandValues << ')';

  // Emit the optional attributes.
  ArrayAttr attrNames = op.attributeNames();
  if (!attrNames.empty()) {
    Operation::operand_range attrArgs = op.attributes();
    p << " {";
    interleaveComma(llvm::seq<int>(0, attrNames.size()), p,
                    [&](int i) { p << attrNames[i] << " = " << attrArgs[i]; });
    p << '}';
  }

  // Print the result type constraints of the operation.
  if (!op.results().empty())
    p << " -> " << op.types();
  p.printOptionalAttrDict(op->getAttrs(),
                          {"attributeNames", "name", "operand_segment_sizes"});
}

/// Verifies that the result types of this operation, defined within a
/// `pdl.rewrite`, can be inferred.
static LogicalResult verifyResultTypesAreInferrable(OperationOp op,
                                                    ResultRange opResults,
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
  for (int i : llvm::seq<int>(0, opResults.size())) {
    Operation *resultTypeOp = resultTypes[i].getDefiningOp();
    assert(resultTypeOp && "expected valid result type operation");

    // If the op was defined by a `create_native`, it is guaranteed to be
    // usable.
    if (isa<CreateNativeOp>(resultTypeOp))
      continue;

    // If the type is already constrained, there is nothing to do.
    TypeOp typeOp = cast<TypeOp>(resultTypeOp);
    if (typeOp.type())
      continue;

    // If the type operation was defined in the matcher and constrains the
    // result of an input operation, it can be used.
    auto constrainsInputOp = [rewriterBlock](Operation *user) {
      return user->getBlock() != rewriterBlock && isa<OperationOp>(user);
    };
    if (llvm::any_of(typeOp.getResult().getUsers(), constrainsInputOp))
      continue;

    // Otherwise, check to see if any uses of the result can infer the type.
    if (llvm::any_of(opResults[i].getUses(), canInferTypeFromUse))
      continue;
    return op
        .emitOpError("must have inferable or constrained result types when "
                     "nested within `pdl.rewrite`")
        .attachNote()
        .append("result type #", i, " was not constrained");
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

  OperandRange resultTypes = op.types();
  auto opResults = op.results();
  if (resultTypes.size() != opResults.size()) {
    return op.emitOpError() << "expected the same number of result values and "
                               "result type constraints, got "
                            << opResults.size() << " results and "
                            << resultTypes.size() << " constraints";
  }

  // If the operation is within a rewrite body and doesn't have type inference,
  // ensure that the result types can be resolved.
  if (isWithinRewrite && !op.hasTypeInference()) {
    if (failed(verifyResultTypesAreInferrable(op, opResults, resultTypes)))
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
  builder.createBlock(state.addRegion());
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
  auto sourceOp = cast<OperationOp>(op.operation().getDefiningOp());
  auto sourceOpResults = sourceOp.results();
  auto replValues = op.replValues();

  if (Value replOpVal = op.replOperation()) {
    auto replOp = cast<OperationOp>(replOpVal.getDefiningOp());
    auto replOpResults = replOp.results();
    if (sourceOpResults.size() != replOpResults.size()) {
      return op.emitOpError()
             << "expected source operation to have the same number of results "
                "as the replacement operation, replacement operation provided "
             << replOpResults.size() << " but expected "
             << sourceOpResults.size();
    }

    if (!replValues.empty()) {
      return op.emitOpError() << "expected no replacement values to be provided"
                                 " when the replacement operation is present";
    }

    return success();
  }

  if (sourceOpResults.size() != replValues.size()) {
    return op.emitOpError()
           << "expected source operation to have the same number of results "
              "as the provided replacement values, found "
           << replValues.size() << " replacement values but expected "
           << sourceOpResults.size();
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
      op, "`pdl.attribute`, `pdl.input`, or `pdl.operation`");
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/PDL/IR/PDLOps.cpp.inc"
