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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

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

/// Returns true if the given operation is used by a "binding" pdl operation.
static bool hasBindingUse(Operation *op) {
  for (Operation *user : op->getUsers())
    // A result by itself is not binding, it must also be bound.
    if (!isa<ResultOp, ResultsOp>(user) || hasBindingUse(user))
      return true;
  return false;
}

/// Returns success if the given operation is not in the main matcher body or
/// is used by a "binding" operation. On failure, emits an error.
static LogicalResult verifyHasBindingUse(Operation *op) {
  // If the parent is not a pattern, there is nothing to do.
  if (!isa<PatternOp>(op->getParentOp()))
    return success();
  if (hasBindingUse(op))
    return success();
  return op->emitOpError(
      "expected a bindable user when defined in the matcher body of a "
      "`pdl.pattern`");
}

/// Visits all the pdl.operand(s), pdl.result(s), and pdl.operation(s)
/// connected to the given operation.
static void visit(Operation *op, DenseSet<Operation *> &visited) {
  // If the parent is not a pattern, there is nothing to do.
  if (!isa<PatternOp>(op->getParentOp()) || isa<RewriteOp>(op))
    return;

  // Ignore if already visited.
  if (visited.contains(op))
    return;

  // Mark as visited.
  visited.insert(op);

  // Traverse the operands / parent.
  TypeSwitch<Operation *>(op)
      .Case<OperationOp>([&visited](auto operation) {
        for (Value operand : operation.operands())
          visit(operand.getDefiningOp(), visited);
      })
      .Case<ResultOp, ResultsOp>([&visited](auto result) {
        visit(result.parent().getDefiningOp(), visited);
      });

  // Traverse the users.
  for (Operation *user : op->getUsers())
    visit(user, visited);
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

  if (!attrValue) {
    if (isa<RewriteOp>(op->getParentOp()))
      return op.emitOpError("expected constant value when specified within a "
                            "`pdl.rewrite`");
    return verifyHasBindingUse(op);
  }
  if (attrType)
    return op.emitOpError("expected only one of [`type`, `value`] to be set");
  return success();
}

//===----------------------------------------------------------------------===//
// pdl::OperandOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(OperandOp op) { return verifyHasBindingUse(op); }

//===----------------------------------------------------------------------===//
// pdl::OperandsOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(OperandsOp op) { return verifyHasBindingUse(op); }

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
  for (const auto &it : llvm::enumerate(resultTypes)) {
    Operation *resultTypeOp = it.value().getDefiningOp();
    assert(resultTypeOp && "expected valid result type operation");

    // If the op was defined by a `apply_native_rewrite`, it is guaranteed to be
    // usable.
    if (isa<ApplyNativeRewriteOp>(resultTypeOp))
      continue;

    // If the type operation was defined in the matcher and constrains an
    // operand or the result of an input operation, it can be used.
    auto constrainsInput = [rewriterBlock](Operation *user) {
      return user->getBlock() != rewriterBlock &&
             isa<OperandOp, OperandsOp, OperationOp>(user);
    };
    if (TypeOp typeOp = dyn_cast<TypeOp>(resultTypeOp)) {
      if (typeOp.type() || llvm::any_of(typeOp->getUsers(), constrainsInput))
        continue;
    } else if (TypesOp typeOp = dyn_cast<TypesOp>(resultTypeOp)) {
      if (typeOp.types() || llvm::any_of(typeOp->getUsers(), constrainsInput))
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

  return verifyHasBindingUse(op);
}

bool OperationOp::hasTypeInference() {
  Optional<StringRef> opName = name();
  if (!opName)
    return false;

  if (auto rInfo = RegisteredOperationName::lookup(*opName, getContext()))
    return rInfo->hasInterface<InferTypeOpInterface>();
  return false;
}

//===----------------------------------------------------------------------===//
// pdl::PatternOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(PatternOp pattern) {
  Region &body = pattern.body();
  Operation *term = body.front().getTerminator();
  auto rewriteOp = dyn_cast<RewriteOp>(term);
  if (!rewriteOp) {
    return pattern.emitOpError("expected body to terminate with `pdl.rewrite`")
        .attachNote(term->getLoc())
        .append("see terminator defined here");
  }

  // Check that all values defined in the top-level pattern belong to the PDL
  // dialect.
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
  if (result.wasInterrupted())
    return failure();

  // Check that there is at least one operation.
  if (body.front().getOps<OperationOp>().empty())
    return pattern.emitOpError(
        "the pattern must contain at least one `pdl.operation`");

  // Determine if the operations within the pdl.pattern form a connected
  // component. This is determined by starting the search from the first
  // operand/result/operation and visiting their users / parents / operands.
  // We limit our attention to operations that have a user in pdl.rewrite,
  // those that do not will be detected via other means (expected bindable
  // user).
  bool first = true;
  DenseSet<Operation *> visited;
  for (Operation &op : body.front()) {
    // The following are the operations forming the connected component.
    if (!isa<OperandOp, OperandsOp, ResultOp, ResultsOp, OperationOp>(op))
      continue;

    // Determine if the operation has a user in `pdl.rewrite`.
    bool hasUserInRewrite = false;
    for (Operation *user : op.getUsers()) {
      Region *region = user->getParentRegion();
      if (isa<RewriteOp>(user) ||
          (region && isa<RewriteOp>(region->getParentOp()))) {
        hasUserInRewrite = true;
        break;
      }
    }

    // If the operation does not have a user in `pdl.rewrite`, ignore it.
    if (!hasUserInRewrite)
      continue;

    if (first) {
      // For the first operation, invoke visit.
      visit(&op, visited);
      first = false;
    } else if (!visited.count(&op)) {
      // For the subsequent operations, check if already visited.
      return pattern
          .emitOpError("the operations must form a connected component")
          .attachNote(op.getLoc())
          .append("see a disconnected value / operation here");
    }
  }

  return success();
}

void PatternOp::build(OpBuilder &builder, OperationState &state,
                      Optional<uint16_t> benefit, Optional<StringRef> name) {
  build(builder, state, builder.getI16IntegerAttr(benefit ? *benefit : 0),
        name ? builder.getStringAttr(*name) : StringAttr());
  state.regions[0]->emplaceBlock();
}

/// Returns the rewrite operation of this pattern.
RewriteOp PatternOp::getRewriter() {
  return cast<RewriteOp>(body().front().getTerminator());
}

/// The default dialect is `pdl`.
StringRef PatternOp::getDefaultDialect() {
  return PDLDialect::getDialectNamespace();
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

/// The default dialect is `pdl`.
StringRef RewriteOp::getDefaultDialect() {
  return PDLDialect::getDialectNamespace();
}

//===----------------------------------------------------------------------===//
// pdl::TypeOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(TypeOp op) {
  if (!op.typeAttr())
    return verifyHasBindingUse(op);
  return success();
}

//===----------------------------------------------------------------------===//
// pdl::TypesOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(TypesOp op) {
  if (!op.typesAttr())
    return verifyHasBindingUse(op);
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/PDL/IR/PDLOps.cpp.inc"
