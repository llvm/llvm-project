//===- LLVMTerminatorOps.cpp - LLVM dialect terminator ops ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `Terminator` of the LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMTerminatorOps.h"

#include "IR/LLVMOps.h"
#include "mlir/Dialect/LLVMIR/LLVMCallOps.h"
#include "mlir/Dialect/LLVMIR/LLVMConstantOps.h"
#include "mlir/Dialect/LLVMIR/LLVMModuleOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

using namespace mlir;
using namespace mlir::LLVM;

//===----------------------------------------------------------------------===//
// LLVM::BrOp
//===----------------------------------------------------------------------===//

LogicalResult BrOp::canonicalize(BrOp op, PatternRewriter &rewriter) {
  // Merge a branch to a block that has a single predecessor, the same
  // transform as cf::BranchOp; the successor block is inlined into the
  // branch block and the branch is erased.
  Block *succ = op.getDest();
  Block *opParent = op->getBlock();
  if (succ == opParent || !llvm::hasSingleElement(succ->getPredecessors()))
    return failure();

  // If any branch operand is itself a block argument of the successor,
  // merging would replace uses of that argument with itself and leave
  // dangling uses after the successor block is erased.
  for (Value operand : op.getDestOperands())
    if (auto arg = dyn_cast<BlockArgument>(operand))
      if (arg.getOwner() == succ)
        return failure();

  // The successor must be mergeable. A landing pad must stay the first
  // operation of its block, a block tag means the block address is taken,
  // and merging would silently drop a loop annotation on the branch.
  if (!succ->empty()) {
    if (isa<LandingpadOp>(succ->front()))
      return failure();
    if (llvm::any_of(succ->without_terminator(),
                     [](Operation &inner) { return isa<BlockTagOp>(inner); }))
      return failure();
  }
  if (op.getLoopAnnotationAttr())
    return failure();

  SmallVector<Value> brOperands(op.getDestOperands());
  rewriter.eraseOp(op);
  rewriter.mergeBlocks(succ, opParent, brOperands);
  return success();
}

SuccessorOperands BrOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return SuccessorOperands(getDestOperandsMutable());
}

//===----------------------------------------------------------------------===//
// LLVM::CondBrOp
//===----------------------------------------------------------------------===//

SuccessorOperands CondBrOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return SuccessorOperands(index == 0 ? getTrueDestOperandsMutable()
                                      : getFalseDestOperandsMutable());
}

void CondBrOp::build(OpBuilder &builder, OperationState &result,
                     Value condition, Block *trueDest, ValueRange trueOperands,
                     Block *falseDest, ValueRange falseOperands,
                     std::optional<std::pair<uint32_t, uint32_t>> weights) {
  DenseI32ArrayAttr weightsAttr;
  if (weights)
    weightsAttr =
        builder.getDenseI32ArrayAttr({static_cast<int32_t>(weights->first),
                                      static_cast<int32_t>(weights->second)});

  build(builder, result, condition, trueOperands, falseOperands, weightsAttr,
        /*loop_annotation=*/{}, trueDest, falseDest);
}

//===----------------------------------------------------------------------===//
// LLVM::SwitchOp
//===----------------------------------------------------------------------===//

void SwitchOp::build(OpBuilder &builder, OperationState &result, Value value,
                     Block *defaultDestination, ValueRange defaultOperands,
                     DenseIntElementsAttr caseValues,
                     BlockRange caseDestinations,
                     ArrayRef<ValueRange> caseOperands,
                     ArrayRef<int32_t> branchWeights) {
  DenseI32ArrayAttr weightsAttr;
  if (!branchWeights.empty())
    weightsAttr = builder.getDenseI32ArrayAttr(branchWeights);

  build(builder, result, value, defaultOperands, caseOperands, caseValues,
        weightsAttr, defaultDestination, caseDestinations);
}

void SwitchOp::build(OpBuilder &builder, OperationState &result, Value value,
                     Block *defaultDestination, ValueRange defaultOperands,
                     ArrayRef<APInt> caseValues, BlockRange caseDestinations,
                     ArrayRef<ValueRange> caseOperands,
                     ArrayRef<int32_t> branchWeights) {
  DenseIntElementsAttr caseValuesAttr;
  if (!caseValues.empty()) {
    ShapedType caseValueType = VectorType::get(
        static_cast<int64_t>(caseValues.size()), value.getType());
    caseValuesAttr = DenseIntElementsAttr::get(caseValueType, caseValues);
  }

  build(builder, result, value, defaultDestination, defaultOperands,
        caseValuesAttr, caseDestinations, caseOperands, branchWeights);
}

void SwitchOp::build(OpBuilder &builder, OperationState &result, Value value,
                     Block *defaultDestination, ValueRange defaultOperands,
                     ArrayRef<int32_t> caseValues, BlockRange caseDestinations,
                     ArrayRef<ValueRange> caseOperands,
                     ArrayRef<int32_t> branchWeights) {
  DenseIntElementsAttr caseValuesAttr;
  if (!caseValues.empty()) {
    ShapedType caseValueType = VectorType::get(
        static_cast<int64_t>(caseValues.size()), value.getType());
    caseValuesAttr = DenseIntElementsAttr::get(caseValueType, caseValues);
  }

  build(builder, result, value, defaultDestination, defaultOperands,
        caseValuesAttr, caseDestinations, caseOperands, branchWeights);
}

/// <cases> ::= `[` (case (`,` case )* )? `]`
/// <case>  ::= integer `:` bb-id (`(` ssa-use-and-type-list `)`)?
ParseResult mlir::LLVM::parseSwitchOpCases(
    OpAsmParser &parser, Type flagType, DenseIntElementsAttr &caseValues,
    SmallVectorImpl<Block *> &caseDestinations,
    SmallVectorImpl<SmallVector<OpAsmParser::UnresolvedOperand>> &caseOperands,
    SmallVectorImpl<SmallVector<Type>> &caseOperandTypes) {
  if (failed(parser.parseLSquare()))
    return failure();
  if (succeeded(parser.parseOptionalRSquare()))
    return success();
  SmallVector<APInt> values;
  unsigned bitWidth = flagType.getIntOrFloatBitWidth();
  auto parseCase = [&]() {
    int64_t value = 0;
    if (failed(parser.parseInteger(value)))
      return failure();
    values.push_back(APInt(bitWidth, value, /*isSigned=*/true));

    Block *destination;
    SmallVector<OpAsmParser::UnresolvedOperand> operands;
    SmallVector<Type> operandTypes;
    if (parser.parseColon() || parser.parseSuccessor(destination))
      return failure();
    if (!parser.parseOptionalLParen()) {
      if (parser.parseOperandList(operands, OpAsmParser::Delimiter::None) ||
          parser.parseColonTypeList(operandTypes) || parser.parseRParen())
        return failure();
    }
    caseDestinations.push_back(destination);
    caseOperands.emplace_back(operands);
    caseOperandTypes.emplace_back(operandTypes);
    return success();
  };
  if (failed(parser.parseCommaSeparatedList(parseCase)))
    return failure();

  ShapedType caseValueType =
      VectorType::get(static_cast<int64_t>(values.size()), flagType);
  caseValues = DenseIntElementsAttr::get(caseValueType, values);
  return parser.parseRSquare();
}

void mlir::LLVM::printSwitchOpCases(OpAsmPrinter &p, SwitchOp op, Type flagType,
                                    DenseIntElementsAttr caseValues,
                                    SuccessorRange caseDestinations,
                                    OperandRangeRange caseOperands,
                                    const TypeRangeRange &caseOperandTypes) {
  p << '[';
  p.printNewline();
  if (!caseValues) {
    p << ']';
    return;
  }

  size_t index = 0;
  llvm::interleave(
      llvm::zip(caseValues, caseDestinations),
      [&](auto i) {
        p << "  ";
        p << std::get<0>(i);
        p << ": ";
        p.printSuccessorAndUseList(std::get<1>(i), caseOperands[index++]);
      },
      [&] {
        p << ',';
        p.printNewline();
      });
  p.printNewline();
  p << ']';
}

LogicalResult SwitchOp::verify() {
  if ((!getCaseValues() && !getCaseDestinations().empty()) ||
      (getCaseValues() &&
       getCaseValues()->size() !=
           static_cast<int64_t>(getCaseDestinations().size())))
    return emitOpError("expects number of case values to match number of "
                       "case destinations");
  if (getCaseValues() &&
      getValue().getType() != getCaseValues()->getElementType())
    return emitError("expects case value type to match condition value type");

  // LLVM IR rejects a switch with a repeated case value.
  if (getCaseValues()) {
    llvm::SmallDenseMap<APInt, unsigned> firstCase;
    for (auto [index, value] :
         llvm::enumerate(getCaseValues()->getValues<APInt>())) {
      auto [it, inserted] = firstCase.try_emplace(value, index);
      if (inserted)
        continue;
      // Diagnostics cannot stream APInt directly, so print it into a buffer.
      SmallString<32> valueStr;
      llvm::raw_svector_ostream os(valueStr);
      value.print(os, /*isSigned=*/true);
      if (value.isNegative()) {
        os << " (unsigned ";
        value.print(os, /*isSigned=*/false);
        os << ")";
      }
      return emitOpError() << "case value " << valueStr
                           << " is repeated (cases " << it->second << " and "
                           << index << ")";
    }
  }
  return success();
}

SuccessorOperands SwitchOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return SuccessorOperands(index == 0 ? getDefaultOperandsMutable()
                                      : getCaseOperandsMutable(index - 1));
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto parent = (*this)->getParentOfType<LLVMFuncOp>();
  if (!parent)
    return success();

  Type expectedType = parent.getFunctionType().getReturnType();
  if (llvm::isa<LLVMVoidType>(expectedType)) {
    if (!getArg())
      return success();
    InFlightDiagnostic diag = emitOpError("expected no operands");
    diag.attachNote(parent->getLoc()) << "when returning from function";
    return diag;
  }
  if (!getArg()) {
    if (llvm::isa<LLVMVoidType>(expectedType))
      return success();
    InFlightDiagnostic diag = emitOpError("expected 1 operand");
    diag.attachNote(parent->getLoc()) << "when returning from function";
    return diag;
  }
  if (expectedType != getArg().getType()) {
    InFlightDiagnostic diag = emitOpError("mismatching result types");
    diag.attachNote(parent->getLoc()) << "when returning from function";
    return diag;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// LLVM::IndirectBrOp
//===----------------------------------------------------------------------===//

SuccessorOperands IndirectBrOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return SuccessorOperands(getSuccOperandsMutable()[index]);
}

void IndirectBrOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                         Value addr, ArrayRef<ValueRange> succOperands,
                         BlockRange successors) {
  odsState.addOperands(addr);
  for (ValueRange range : succOperands)
    odsState.addOperands(range);
  SmallVector<int32_t> rangeSegments;
  for (ValueRange range : succOperands)
    rangeSegments.push_back(range.size());
  odsState.getOrAddProperties<Properties>().indbr_operand_segments =
      odsBuilder.getDenseI32ArrayAttr(rangeSegments);
  odsState.addSuccessors(successors);
}

ParseResult mlir::LLVM::parseIndirectBrOpSucessors(
    OpAsmParser &parser, Type &flagType,
    SmallVectorImpl<Block *> &succOperandBlocks,
    SmallVectorImpl<SmallVector<OpAsmParser::UnresolvedOperand>> &succOperands,
    SmallVectorImpl<SmallVector<Type>> &succOperandsTypes) {
  if (failed(parser.parseCommaSeparatedList(
          OpAsmParser::Delimiter::Square,
          [&]() {
            Block *destination = nullptr;
            SmallVector<OpAsmParser::UnresolvedOperand> operands;
            SmallVector<Type> operandTypes;

            if (parser.parseSuccessor(destination).failed())
              return failure();

            if (succeeded(parser.parseOptionalLParen())) {
              if (failed(parser.parseOperandList(
                      operands, OpAsmParser::Delimiter::None)) ||
                  failed(parser.parseColonTypeList(operandTypes)) ||
                  failed(parser.parseRParen()))
                return failure();
            }
            succOperandBlocks.push_back(destination);
            succOperands.emplace_back(operands);
            succOperandsTypes.emplace_back(operandTypes);
            return success();
          },
          "successor blocks")))
    return failure();
  return success();
}

void mlir::LLVM::printIndirectBrOpSucessors(
    OpAsmPrinter &p, IndirectBrOp op, Type flagType, SuccessorRange succs,
    OperandRangeRange succOperands, const TypeRangeRange &succOperandsTypes) {
  p << "[";
  llvm::interleave(
      llvm::zip(succs, succOperands),
      [&](auto i) {
        p.printNewline();
        p.printSuccessorAndUseList(std::get<0>(i), std::get<1>(i));
      },
      [&] { p << ','; });
  if (!succOperands.empty())
    p.printNewline();
  p << "]";
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMTerminatorOps.cpp.inc"
