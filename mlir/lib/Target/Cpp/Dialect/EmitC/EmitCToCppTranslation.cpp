//===---- EmitCToCppTranslation.cpp - Translate EmitC dialect to Cpp ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements translation between the EmitC dialect and Cpp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/Cpp/Dialect/EmitC/EmitCToCppTranslation.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Target/Cpp/CppTranslationInterface.h"
#include "mlir/Target/Cpp/CppTranslationUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::emitc;

static LogicalResult printOperation(CppEmitter &emitter,
                                    emitc::ConstantOp constantOp) {
  Operation *operation = constantOp.getOperation();
  Attribute value = constantOp.getValue();

  return printConstantOp(emitter, operation, value);
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    emitc::VariableOp variableOp) {
  Operation *operation = variableOp.getOperation();
  Attribute value = variableOp.getValue();

  return printConstantOp(emitter, operation, value);
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    emitc::AssignOp assignOp) {
  auto variableOp = cast<emitc::VariableOp>(assignOp.getVar().getDefiningOp());
  OpResult result = variableOp->getResult(0);

  if (failed(emitter.emitVariableAssignment(result)))
    return failure();

  return emitter.emitOperand(assignOp.getValue());
}

static LogicalResult printOperation(CppEmitter &emitter, emitc::AddOp addOp) {
  Operation *operation = addOp.getOperation();

  return printBinaryOperation(emitter, operation, "+");
}

static LogicalResult printOperation(CppEmitter &emitter, emitc::DivOp divOp) {
  Operation *operation = divOp.getOperation();

  return printBinaryOperation(emitter, operation, "/");
}

static LogicalResult printOperation(CppEmitter &emitter, emitc::MulOp mulOp) {
  Operation *operation = mulOp.getOperation();

  return printBinaryOperation(emitter, operation, "*");
}

static LogicalResult printOperation(CppEmitter &emitter, emitc::RemOp remOp) {
  Operation *operation = remOp.getOperation();

  return printBinaryOperation(emitter, operation, "%");
}

static LogicalResult printOperation(CppEmitter &emitter, emitc::SubOp subOp) {
  Operation *operation = subOp.getOperation();

  return printBinaryOperation(emitter, operation, "-");
}

static LogicalResult printOperation(CppEmitter &emitter, emitc::CmpOp cmpOp) {
  Operation *operation = cmpOp.getOperation();

  StringRef binaryOperator;

  switch (cmpOp.getPredicate()) {
  case emitc::CmpPredicate::eq:
    binaryOperator = "==";
    break;
  case emitc::CmpPredicate::ne:
    binaryOperator = "!=";
    break;
  case emitc::CmpPredicate::lt:
    binaryOperator = "<";
    break;
  case emitc::CmpPredicate::le:
    binaryOperator = "<=";
    break;
  case emitc::CmpPredicate::gt:
    binaryOperator = ">";
    break;
  case emitc::CmpPredicate::ge:
    binaryOperator = ">=";
    break;
  case emitc::CmpPredicate::three_way:
    binaryOperator = "<=>";
    break;
  }

  return printBinaryOperation(emitter, operation, binaryOperator);
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    emitc::ConditionalOp conditionalOp) {
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*conditionalOp)))
    return failure();

  if (failed(emitter.emitOperand(conditionalOp.getCondition())))
    return failure();

  os << " ? ";

  if (failed(emitter.emitOperand(conditionalOp.getTrueValue())))
    return failure();

  os << " : ";

  if (failed(emitter.emitOperand(conditionalOp.getFalseValue())))
    return failure();

  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    emitc::VerbatimOp verbatimOp) {
  raw_ostream &os = emitter.ostream();

  os << verbatimOp.getValue();

  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, emitc::CallOp callOp) {
  Operation *operation = callOp.getOperation();
  StringRef callee = callOp.getCallee();

  return printCallOperation(emitter, operation, callee);
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    emitc::CallOpaqueOp callOpaqueOp) {
  raw_ostream &os = emitter.ostream();
  Operation &op = *callOpaqueOp.getOperation();

  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  os << callOpaqueOp.getCallee();

  auto emitArgs = [&](Attribute attr) -> LogicalResult {
    if (auto t = dyn_cast<IntegerAttr>(attr)) {
      // Index attributes are treated specially as operand index.
      if (t.getType().isIndex()) {
        int64_t idx = t.getInt();
        Value operand = op.getOperand(idx);
        auto literalDef =
            dyn_cast_if_present<emitc::LiteralOp>(operand.getDefiningOp());
        if (!literalDef && !emitter.hasValueInScope(operand))
          return op.emitOpError("operand ")
                 << idx << "'s value not defined in scope";
        os << emitter.getOrCreateName(operand);
        return success();
      }
    }
    if (failed(emitter.emitAttribute(op.getLoc(), attr)))
      return failure();

    return success();
  };

  if (callOpaqueOp.getTemplateArgs()) {
    os << "<";
    if (failed(interleaveCommaWithError(*callOpaqueOp.getTemplateArgs(), os,
                                        emitArgs)))
      return failure();
    os << ">";
  }

  os << "(";

  LogicalResult emittedArgs =
      callOpaqueOp.getArgs()
          ? interleaveCommaWithError(*callOpaqueOp.getArgs(), os, emitArgs)
          : emitter.emitOperands(op);
  if (failed(emittedArgs))
    return failure();
  os << ")";
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    emitc::ApplyOp applyOp) {
  raw_ostream &os = emitter.ostream();
  Operation &op = *applyOp.getOperation();

  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  os << applyOp.getApplicableOperator();
  os << emitter.getOrCreateName(applyOp.getOperand());

  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    emitc::BitwiseAndOp bitwiseAndOp) {
  Operation *operation = bitwiseAndOp.getOperation();
  return printBinaryOperation(emitter, operation, "&");
}

static LogicalResult
printOperation(CppEmitter &emitter,
               emitc::BitwiseLeftShiftOp bitwiseLeftShiftOp) {
  Operation *operation = bitwiseLeftShiftOp.getOperation();
  return printBinaryOperation(emitter, operation, "<<");
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    emitc::BitwiseNotOp bitwiseNotOp) {
  Operation *operation = bitwiseNotOp.getOperation();
  return printUnaryOperation(emitter, operation, "~");
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    emitc::BitwiseOrOp bitwiseOrOp) {
  Operation *operation = bitwiseOrOp.getOperation();
  return printBinaryOperation(emitter, operation, "|");
}

static LogicalResult
printOperation(CppEmitter &emitter,
               emitc::BitwiseRightShiftOp bitwiseRightShiftOp) {
  Operation *operation = bitwiseRightShiftOp.getOperation();
  return printBinaryOperation(emitter, operation, ">>");
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    emitc::BitwiseXorOp bitwiseXorOp) {
  Operation *operation = bitwiseXorOp.getOperation();
  return printBinaryOperation(emitter, operation, "^");
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    emitc::UnaryPlusOp unaryPlusOp) {
  Operation *operation = unaryPlusOp.getOperation();
  return printUnaryOperation(emitter, operation, "+");
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    emitc::UnaryMinusOp unaryMinusOp) {
  Operation *operation = unaryMinusOp.getOperation();
  return printUnaryOperation(emitter, operation, "-");
}

static LogicalResult printOperation(CppEmitter &emitter, emitc::CastOp castOp) {
  raw_ostream &os = emitter.ostream();
  Operation &op = *castOp.getOperation();

  if (failed(emitter.emitAssignPrefix(op)))
    return failure();
  os << "(";
  if (failed(emitter.emitType(op.getLoc(), op.getResult(0).getType())))
    return failure();
  os << ") ";
  return emitter.emitOperand(castOp.getOperand());
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    emitc::ExpressionOp expressionOp) {
  if (shouldBeInlined(expressionOp))
    return success();

  Operation &op = *expressionOp.getOperation();

  if (failed(emitter.emitAssignPrefix(op)))
    return failure();

  return emitter.emitExpression(expressionOp);
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    emitc::IncludeOp includeOp) {
  raw_ostream &os = emitter.ostream();

  os << "#include ";
  if (includeOp.getIsStandardInclude())
    os << "<" << includeOp.getInclude() << ">";
  else
    os << "\"" << includeOp.getInclude() << "\"";

  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    emitc::LogicalAndOp logicalAndOp) {
  Operation *operation = logicalAndOp.getOperation();
  return printBinaryOperation(emitter, operation, "&&");
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    emitc::LogicalNotOp logicalNotOp) {
  Operation *operation = logicalNotOp.getOperation();
  return printUnaryOperation(emitter, operation, "!");
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    emitc::LogicalOrOp logicalOrOp) {
  Operation *operation = logicalOrOp.getOperation();
  return printBinaryOperation(emitter, operation, "||");
}

static LogicalResult printOperation(CppEmitter &emitter, emitc::ForOp forOp) {

  raw_indented_ostream &os = emitter.ostream();

  // Utility function to determine whether a value is an expression that will be
  // inlined, and as such should be wrapped in parentheses in order to guarantee
  // its precedence and associativity.
  auto requiresParentheses = [&](Value value) {
    auto expressionOp =
        dyn_cast_if_present<ExpressionOp>(value.getDefiningOp());
    if (!expressionOp)
      return false;
    return shouldBeInlined(expressionOp);
  };

  os << "for (";
  if (failed(
          emitter.emitType(forOp.getLoc(), forOp.getInductionVar().getType())))
    return failure();
  os << " ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " = ";
  if (failed(emitter.emitOperand(forOp.getLowerBound())))
    return failure();
  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " < ";
  Value upperBound = forOp.getUpperBound();
  bool upperBoundRequiresParentheses = requiresParentheses(upperBound);
  if (upperBoundRequiresParentheses)
    os << "(";
  if (failed(emitter.emitOperand(upperBound)))
    return failure();
  if (upperBoundRequiresParentheses)
    os << ")";
  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " += ";
  if (failed(emitter.emitOperand(forOp.getStep())))
    return failure();
  os << ") {\n";
  os.indent();

  Region &forRegion = forOp.getRegion();
  auto regionOps = forRegion.getOps();

  // We skip the trailing yield op.
  for (auto it = regionOps.begin(); std::next(it) != regionOps.end(); ++it) {
    if (failed(emitter.emitOperation(*it, /*trailingSemicolon=*/true)))
      return failure();
  }

  os.unindent() << "}";

  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, emitc::IfOp ifOp) {
  raw_indented_ostream &os = emitter.ostream();

  // Helper function to emit all ops except the last one, expected to be
  // emitc::yield.
  auto emitAllExceptLast = [&emitter](Region &region) {
    Region::OpIterator it = region.op_begin(), end = region.op_end();
    for (; std::next(it) != end; ++it) {
      if (failed(emitter.emitOperation(*it, /*trailingSemicolon=*/true)))
        return failure();
    }
    assert(isa<emitc::YieldOp>(*it) &&
           "Expected last operation in the region to be emitc::yield");
    return success();
  };

  os << "if (";
  if (failed(emitter.emitOperand(ifOp.getCondition())))
    return failure();
  os << ") {\n";
  os.indent();
  if (failed(emitAllExceptLast(ifOp.getThenRegion())))
    return failure();
  os.unindent() << "}";

  Region &elseRegion = ifOp.getElseRegion();
  if (!elseRegion.empty()) {
    os << " else {\n";
    os.indent();
    if (failed(emitAllExceptLast(elseRegion)))
      return failure();
    os.unindent() << "}";
  }

  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    emitc::ReturnOp returnOp) {
  raw_ostream &os = emitter.ostream();
  os << "return";
  if (returnOp.getNumOperands() == 0)
    return success();

  os << " ";
  if (failed(emitter.emitOperand(returnOp.getOperand())))
    return failure();
  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    emitc::FuncOp functionOp) {
  // We need to declare variables at top if the function has multiple blocks.
  if (!emitter.shouldDeclareVariablesAtTop() &&
      functionOp.getBlocks().size() > 1) {
    return functionOp.emitOpError(
        "with multiple blocks needs variables declared at top");
  }

  CppEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  if (functionOp.getSpecifiers()) {
    for (Attribute specifier : functionOp.getSpecifiersAttr()) {
      os << cast<StringAttr>(specifier).str() << " ";
    }
  }

  if (failed(emitter.emitTypes(functionOp.getLoc(),
                               functionOp.getFunctionType().getResults())))
    return failure();
  os << " " << functionOp.getName();

  os << "(";
  Operation *operation = functionOp.getOperation();
  if (functionOp.isExternal()) {
    if (failed(printFunctionArgs(emitter, operation,
                                 functionOp.getArgumentTypes())))
      return failure();
    os << ");";
    return success();
  }
  if (failed(printFunctionArgs(emitter, operation, functionOp.getArguments())))
    return failure();
  os << ") {\n";
  if (failed(printFunctionBody(emitter, operation, functionOp.getBlocks())))
    return failure();
  os << "}\n";

  return success();
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    emitc::DeclareFuncOp declareFuncOp) {
  CppEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();

  auto functionOp = SymbolTable::lookupNearestSymbolFrom<emitc::FuncOp>(
      declareFuncOp, declareFuncOp.getSymNameAttr());

  if (!functionOp)
    return failure();

  if (functionOp.getSpecifiers()) {
    for (Attribute specifier : functionOp.getSpecifiersAttr()) {
      os << cast<StringAttr>(specifier).str() << " ";
    }
  }

  if (failed(emitter.emitTypes(functionOp.getLoc(),
                               functionOp.getFunctionType().getResults())))
    return failure();
  os << " " << functionOp.getName();

  os << "(";
  Operation *operation = functionOp.getOperation();
  if (failed(printFunctionArgs(emitter, operation, functionOp.getArguments())))
    return failure();
  os << ");";

  return success();
}

namespace {
/// Implementation of the dialect interface that converts EmitC ops to Cpp.
class EmitCDialectCppTranslationInterface
    : public CppTranslationDialectInterface {
public:
  using CppTranslationDialectInterface::CppTranslationDialectInterface;

  LogicalResult emitOperation(Operation *op, CppEmitter &cppEmitter,
                              bool trailingSemicolon) const final {
    LogicalResult status =
        llvm::TypeSwitch<Operation *, LogicalResult>(op)
            // EmitC ops.
            .Case<emitc::AddOp, emitc::ApplyOp, emitc::AssignOp,
                  emitc::BitwiseAndOp, emitc::BitwiseLeftShiftOp,
                  emitc::BitwiseNotOp, emitc::BitwiseOrOp,
                  emitc::BitwiseRightShiftOp, emitc::BitwiseXorOp,
                  emitc::CallOp, emitc::CallOpaqueOp, emitc::CastOp,
                  emitc::CmpOp, emitc::ConditionalOp, emitc::ConstantOp,
                  emitc::DeclareFuncOp, emitc::DivOp, emitc::ExpressionOp,
                  emitc::ForOp, emitc::FuncOp, emitc::IfOp, emitc::IncludeOp,
                  emitc::LogicalAndOp, emitc::LogicalNotOp, emitc::LogicalOrOp,
                  emitc::MulOp, emitc::RemOp, emitc::ReturnOp, emitc::SubOp,
                  emitc::UnaryMinusOp, emitc::UnaryPlusOp, emitc::VariableOp,
                  emitc::VerbatimOp>(
                [&](auto op) { return printOperation(cppEmitter, op); })
            .Case<emitc::LiteralOp>([&](auto op) { return success(); })
            .Default([&](Operation *) {
              return op->emitOpError("unable to find printer for op")
                     << op->getName();
            });

    if (failed(status))
      return failure();

    if (isa<emitc::LiteralOp>(op))
      return success();

    if (cppEmitter.getEmittedExpression() ||
        (isa<emitc::ExpressionOp>(op) &&
         shouldBeInlined(cast<emitc::ExpressionOp>(op))))
      return success();

    cppEmitter.ostream() << (trailingSemicolon ? ";\n" : "\n");

    return success();
  }
};
} // namespace

void mlir::registerEmitCDialectCppTranslation(DialectRegistry &registry) {
  registry.insert<emitc::EmitCDialect>();
  registry.addExtension(+[](MLIRContext *ctx, emitc::EmitCDialect *dialect) {
    dialect->addInterfaces<EmitCDialectCppTranslationInterface>();
  });
}

void mlir::registerEmitCDialectCppTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerEmitCDialectCppTranslation(registry);
  context.appendDialectRegistry(registry);
}
