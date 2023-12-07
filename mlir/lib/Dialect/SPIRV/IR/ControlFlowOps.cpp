//===- ControlFlowOps.cpp - MLIR SPIR-V Control Flow Ops  -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the control flow operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include "SPIRVOpUtils.h"
#include "SPIRVParsingUtils.h"

using namespace mlir::spirv::AttrNames;

namespace mlir::spirv {

/// Parses Function, Selection and Loop control attributes. If no control is
/// specified, "None" is used as a default.
template <typename EnumAttrClass, typename EnumClass>
static ParseResult
parseControlAttribute(OpAsmParser &parser, OperationState &state,
                      StringRef attrName = spirv::attributeName<EnumClass>()) {
  if (succeeded(parser.parseOptionalKeyword(kControl))) {
    EnumClass control;
    if (parser.parseLParen() ||
        spirv::parseEnumKeywordAttr<EnumAttrClass>(control, parser, state) ||
        parser.parseRParen())
      return failure();
    return success();
  }
  // Set control to "None" otherwise.
  Builder builder = parser.getBuilder();
  state.addAttribute(attrName,
                     builder.getAttr<EnumAttrClass>(static_cast<EnumClass>(0)));
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.BranchOp
//===----------------------------------------------------------------------===//

SuccessorOperands BranchOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return SuccessorOperands(0, getTargetOperandsMutable());
}

//===----------------------------------------------------------------------===//
// spirv.BranchConditionalOp
//===----------------------------------------------------------------------===//

SuccessorOperands BranchConditionalOp::getSuccessorOperands(unsigned index) {
  assert(index < 2 && "invalid successor index");
  return SuccessorOperands(index == kTrueIndex
                               ? getTrueTargetOperandsMutable()
                               : getFalseTargetOperandsMutable());
}

ParseResult BranchConditionalOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  auto &builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand condInfo;
  Block *dest;

  // Parse the condition.
  Type boolTy = builder.getI1Type();
  if (parser.parseOperand(condInfo) ||
      parser.resolveOperand(condInfo, boolTy, result.operands))
    return failure();

  // Parse the optional branch weights.
  if (succeeded(parser.parseOptionalLSquare())) {
    IntegerAttr trueWeight, falseWeight;
    NamedAttrList weights;

    auto i32Type = builder.getIntegerType(32);
    if (parser.parseAttribute(trueWeight, i32Type, "weight", weights) ||
        parser.parseComma() ||
        parser.parseAttribute(falseWeight, i32Type, "weight", weights) ||
        parser.parseRSquare())
      return failure();

    result.addAttribute(kBranchWeightAttrName,
                        builder.getArrayAttr({trueWeight, falseWeight}));
  }

  // Parse the true branch.
  SmallVector<Value, 4> trueOperands;
  if (parser.parseComma() ||
      parser.parseSuccessorAndUseList(dest, trueOperands))
    return failure();
  result.addSuccessors(dest);
  result.addOperands(trueOperands);

  // Parse the false branch.
  SmallVector<Value, 4> falseOperands;
  if (parser.parseComma() ||
      parser.parseSuccessorAndUseList(dest, falseOperands))
    return failure();
  result.addSuccessors(dest);
  result.addOperands(falseOperands);
  result.addAttribute(spirv::BranchConditionalOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(
                          {1, static_cast<int32_t>(trueOperands.size()),
                           static_cast<int32_t>(falseOperands.size())}));

  return success();
}

void BranchConditionalOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getCondition();

  if (auto weights = getBranchWeights()) {
    printer << " [";
    llvm::interleaveComma(weights->getValue(), printer, [&](Attribute a) {
      printer << llvm::cast<IntegerAttr>(a).getInt();
    });
    printer << "]";
  }

  printer << ", ";
  printer.printSuccessorAndUseList(getTrueBlock(), getTrueBlockArguments());
  printer << ", ";
  printer.printSuccessorAndUseList(getFalseBlock(), getFalseBlockArguments());
}

LogicalResult BranchConditionalOp::verify() {
  if (auto weights = getBranchWeights()) {
    if (weights->getValue().size() != 2) {
      return emitOpError("must have exactly two branch weights");
    }
    if (llvm::all_of(*weights, [](Attribute attr) {
          return llvm::cast<IntegerAttr>(attr).getValue().isZero();
        }))
      return emitOpError("branch weights cannot both be zero");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.FunctionCall
//===----------------------------------------------------------------------===//

LogicalResult FunctionCallOp::verify() {
  auto fnName = getCalleeAttr();

  auto funcOp = dyn_cast_or_null<spirv::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom((*this)->getParentOp(), fnName));
  if (!funcOp) {
    return emitOpError("callee function '")
           << fnName.getValue() << "' not found in nearest symbol table";
  }

  auto functionType = funcOp.getFunctionType();

  if (getNumResults() > 1) {
    return emitOpError(
               "expected callee function to have 0 or 1 result, but provided ")
           << getNumResults();
  }

  if (functionType.getNumInputs() != getNumOperands()) {
    return emitOpError("has incorrect number of operands for callee: expected ")
           << functionType.getNumInputs() << ", but provided "
           << getNumOperands();
  }

  for (uint32_t i = 0, e = functionType.getNumInputs(); i != e; ++i) {
    if (getOperand(i).getType() != functionType.getInput(i)) {
      return emitOpError("operand type mismatch: expected operand type ")
             << functionType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;
    }
  }

  if (functionType.getNumResults() != getNumResults()) {
    return emitOpError(
               "has incorrect number of results has for callee: expected ")
           << functionType.getNumResults() << ", but provided "
           << getNumResults();
  }

  if (getNumResults() &&
      (getResult(0).getType() != functionType.getResult(0))) {
    return emitOpError("result type mismatch: expected ")
           << functionType.getResult(0) << ", but provided "
           << getResult(0).getType();
  }

  return success();
}

CallInterfaceCallable FunctionCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>(kCallee);
}

void FunctionCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr(kCallee, callee.get<SymbolRefAttr>());
}

Operation::operand_range FunctionCallOp::getArgOperands() {
  return getArguments();
}

MutableOperandRange FunctionCallOp::getArgOperandsMutable() {
  return getArgumentsMutable();
}

//===----------------------------------------------------------------------===//
// spirv.mlir.loop
//===----------------------------------------------------------------------===//

void LoopOp::build(OpBuilder &builder, OperationState &state) {
  state.addAttribute("loop_control", builder.getAttr<spirv::LoopControlAttr>(
                                         spirv::LoopControl::None));
  state.addRegion();
}

ParseResult LoopOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parseControlAttribute<spirv::LoopControlAttr, spirv::LoopControl>(parser,
                                                                        result))
    return failure();
  return parser.parseRegion(*result.addRegion(), /*arguments=*/{});
}

void LoopOp::print(OpAsmPrinter &printer) {
  auto control = getLoopControl();
  if (control != spirv::LoopControl::None)
    printer << " control(" << spirv::stringifyLoopControl(control) << ")";
  printer << ' ';
  printer.printRegion(getRegion(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

/// Returns true if the given `srcBlock` contains only one `spirv.Branch` to the
/// given `dstBlock`.
static bool hasOneBranchOpTo(Block &srcBlock, Block &dstBlock) {
  // Check that there is only one op in the `srcBlock`.
  if (!llvm::hasSingleElement(srcBlock))
    return false;

  auto branchOp = dyn_cast<spirv::BranchOp>(srcBlock.back());
  return branchOp && branchOp.getSuccessor() == &dstBlock;
}

/// Returns true if the given `block` only contains one `spirv.mlir.merge` op.
static bool isMergeBlock(Block &block) {
  return !block.empty() && std::next(block.begin()) == block.end() &&
         isa<spirv::MergeOp>(block.front());
}

LogicalResult LoopOp::verifyRegions() {
  auto *op = getOperation();

  // We need to verify that the blocks follow the following layout:
  //
  //                     +-------------+
  //                     | entry block |
  //                     +-------------+
  //                            |
  //                            v
  //                     +-------------+
  //                     | loop header | <-----+
  //                     +-------------+       |
  //                                           |
  //                           ...             |
  //                          \ | /            |
  //                            v              |
  //                    +---------------+      |
  //                    | loop continue | -----+
  //                    +---------------+
  //
  //                           ...
  //                          \ | /
  //                            v
  //                     +-------------+
  //                     | merge block |
  //                     +-------------+

  auto &region = op->getRegion(0);
  // Allow empty region as a degenerated case, which can come from
  // optimizations.
  if (region.empty())
    return success();

  // The last block is the merge block.
  Block &merge = region.back();
  if (!isMergeBlock(merge))
    return emitOpError("last block must be the merge block with only one "
                       "'spirv.mlir.merge' op");

  if (std::next(region.begin()) == region.end())
    return emitOpError(
        "must have an entry block branching to the loop header block");
  // The first block is the entry block.
  Block &entry = region.front();

  if (std::next(region.begin(), 2) == region.end())
    return emitOpError(
        "must have a loop header block branched from the entry block");
  // The second block is the loop header block.
  Block &header = *std::next(region.begin(), 1);

  if (!hasOneBranchOpTo(entry, header))
    return emitOpError(
        "entry block must only have one 'spirv.Branch' op to the second block");

  if (std::next(region.begin(), 3) == region.end())
    return emitOpError(
        "requires a loop continue block branching to the loop header block");
  // The second to last block is the loop continue block.
  Block &cont = *std::prev(region.end(), 2);

  // Make sure that we have a branch from the loop continue block to the loop
  // header block.
  if (llvm::none_of(
          llvm::seq<unsigned>(0, cont.getNumSuccessors()),
          [&](unsigned index) { return cont.getSuccessor(index) == &header; }))
    return emitOpError("second to last block must be the loop continue "
                       "block that branches to the loop header block");

  // Make sure that no other blocks (except the entry and loop continue block)
  // branches to the loop header block.
  for (auto &block : llvm::make_range(std::next(region.begin(), 2),
                                      std::prev(region.end(), 2))) {
    for (auto i : llvm::seq<unsigned>(0, block.getNumSuccessors())) {
      if (block.getSuccessor(i) == &header) {
        return emitOpError("can only have the entry and loop continue "
                           "block branching to the loop header block");
      }
    }
  }

  return success();
}

Block *LoopOp::getEntryBlock() {
  assert(!getBody().empty() && "op region should not be empty!");
  return &getBody().front();
}

Block *LoopOp::getHeaderBlock() {
  assert(!getBody().empty() && "op region should not be empty!");
  // The second block is the loop header block.
  return &*std::next(getBody().begin());
}

Block *LoopOp::getContinueBlock() {
  assert(!getBody().empty() && "op region should not be empty!");
  // The second to last block is the loop continue block.
  return &*std::prev(getBody().end(), 2);
}

Block *LoopOp::getMergeBlock() {
  assert(!getBody().empty() && "op region should not be empty!");
  // The last block is the loop merge block.
  return &getBody().back();
}

void LoopOp::addEntryAndMergeBlock() {
  assert(getBody().empty() && "entry and merge block already exist");
  getBody().push_back(new Block());
  auto *mergeBlock = new Block();
  getBody().push_back(mergeBlock);
  OpBuilder builder = OpBuilder::atBlockEnd(mergeBlock);

  // Add a spirv.mlir.merge op into the merge block.
  builder.create<spirv::MergeOp>(getLoc());
}

//===----------------------------------------------------------------------===//
// spirv.mlir.merge
//===----------------------------------------------------------------------===//

LogicalResult MergeOp::verify() {
  auto *parentOp = (*this)->getParentOp();
  if (!parentOp || !isa<spirv::SelectionOp, spirv::LoopOp>(parentOp))
    return emitOpError(
        "expected parent op to be 'spirv.mlir.selection' or 'spirv.mlir.loop'");

  // TODO: This check should be done in `verifyRegions` of parent op.
  Block &parentLastBlock = (*this)->getParentRegion()->back();
  if (getOperation() != parentLastBlock.getTerminator())
    return emitOpError("can only be used in the last block of "
                       "'spirv.mlir.selection' or 'spirv.mlir.loop'");
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.Return
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  // Verification is performed in spirv.func op.
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.ReturnValue
//===----------------------------------------------------------------------===//

LogicalResult ReturnValueOp::verify() {
  // Verification is performed in spirv.func op.
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.Select
//===----------------------------------------------------------------------===//

LogicalResult SelectOp::verify() {
  if (auto conditionTy = llvm::dyn_cast<VectorType>(getCondition().getType())) {
    auto resultVectorTy = llvm::dyn_cast<VectorType>(getResult().getType());
    if (!resultVectorTy) {
      return emitOpError("result expected to be of vector type when "
                         "condition is of vector type");
    }
    if (resultVectorTy.getNumElements() != conditionTy.getNumElements()) {
      return emitOpError("result should have the same number of elements as "
                         "the condition when condition is of vector type");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// spirv.mlir.selection
//===----------------------------------------------------------------------===//

ParseResult SelectionOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parseControlAttribute<spirv::SelectionControlAttr,
                            spirv::SelectionControl>(parser, result))
    return failure();
  return parser.parseRegion(*result.addRegion(), /*arguments=*/{});
}

void SelectionOp::print(OpAsmPrinter &printer) {
  auto control = getSelectionControl();
  if (control != spirv::SelectionControl::None)
    printer << " control(" << spirv::stringifySelectionControl(control) << ")";
  printer << ' ';
  printer.printRegion(getRegion(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

LogicalResult SelectionOp::verifyRegions() {
  auto *op = getOperation();

  // We need to verify that the blocks follow the following layout:
  //
  //                     +--------------+
  //                     | header block |
  //                     +--------------+
  //                          / | \
  //                           ...
  //
  //
  //         +---------+   +---------+   +---------+
  //         | case #0 |   | case #1 |   | case #2 |  ...
  //         +---------+   +---------+   +---------+
  //
  //
  //                           ...
  //                          \ | /
  //                            v
  //                     +-------------+
  //                     | merge block |
  //                     +-------------+

  auto &region = op->getRegion(0);
  // Allow empty region as a degenerated case, which can come from
  // optimizations.
  if (region.empty())
    return success();

  // The last block is the merge block.
  if (!isMergeBlock(region.back()))
    return emitOpError("last block must be the merge block with only one "
                       "'spirv.mlir.merge' op");

  if (std::next(region.begin()) == region.end())
    return emitOpError("must have a selection header block");

  return success();
}

Block *SelectionOp::getHeaderBlock() {
  assert(!getBody().empty() && "op region should not be empty!");
  // The first block is the loop header block.
  return &getBody().front();
}

Block *SelectionOp::getMergeBlock() {
  assert(!getBody().empty() && "op region should not be empty!");
  // The last block is the loop merge block.
  return &getBody().back();
}

void SelectionOp::addMergeBlock() {
  assert(getBody().empty() && "entry and merge block already exist");
  auto *mergeBlock = new Block();
  getBody().push_back(mergeBlock);
  OpBuilder builder = OpBuilder::atBlockEnd(mergeBlock);

  // Add a spirv.mlir.merge op into the merge block.
  builder.create<spirv::MergeOp>(getLoc());
}

SelectionOp
SelectionOp::createIfThen(Location loc, Value condition,
                          function_ref<void(OpBuilder &builder)> thenBody,
                          OpBuilder &builder) {
  auto selectionOp =
      builder.create<spirv::SelectionOp>(loc, spirv::SelectionControl::None);

  selectionOp.addMergeBlock();
  Block *mergeBlock = selectionOp.getMergeBlock();
  Block *thenBlock = nullptr;

  // Build the "then" block.
  {
    OpBuilder::InsertionGuard guard(builder);
    thenBlock = builder.createBlock(mergeBlock);
    thenBody(builder);
    builder.create<spirv::BranchOp>(loc, mergeBlock);
  }

  // Build the header block.
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(thenBlock);
    builder.create<spirv::BranchConditionalOp>(
        loc, condition, thenBlock,
        /*trueArguments=*/ArrayRef<Value>(), mergeBlock,
        /*falseArguments=*/ArrayRef<Value>());
  }

  return selectionOp;
}

//===----------------------------------------------------------------------===//
// spirv.Unreachable
//===----------------------------------------------------------------------===//

LogicalResult spirv::UnreachableOp::verify() {
  auto *block = (*this)->getBlock();
  // Fast track: if this is in entry block, its invalid. Otherwise, if no
  // predecessors, it's valid.
  if (block->isEntryBlock())
    return emitOpError("cannot be used in reachable block");
  if (block->hasNoPredecessors())
    return success();

  // TODO: further verification needs to analyze reachability from
  // the entry block.

  return success();
}

} // namespace mlir::spirv
