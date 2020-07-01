//===- SCFToSPIRV.cpp - Convert SCF ops to SPIR-V dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the conversion patterns from SCF ops to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/IR/Module.h"

using namespace mlir;

namespace {

/// Pattern to convert a scf::ForOp within kernel functions into spirv::LoopOp.
class ForOpConversion final : public SPIRVOpLowering<scf::ForOp> {
public:
  using SPIRVOpLowering<scf::ForOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(scf::ForOp forOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to convert a scf::IfOp within kernel functions into
/// spirv::SelectionOp.
class IfOpConversion final : public SPIRVOpLowering<scf::IfOp> {
public:
  using SPIRVOpLowering<scf::IfOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(scf::IfOp ifOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to erase a scf::YieldOp.
class TerminatorOpConversion final : public SPIRVOpLowering<scf::YieldOp> {
public:
  using SPIRVOpLowering<scf::YieldOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(scf::YieldOp terminatorOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(terminatorOp);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// scf::ForOp.
//===----------------------------------------------------------------------===//

LogicalResult
ForOpConversion::matchAndRewrite(scf::ForOp forOp, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const {
  // scf::ForOp can be lowered to the structured control flow represented by
  // spirv::LoopOp by making the continue block of the spirv::LoopOp the loop
  // latch and the merge block the exit block. The resulting spirv::LoopOp has a
  // single back edge from the continue to header block, and a single exit from
  // header to merge.
  scf::ForOpAdaptor forOperands(operands);
  auto loc = forOp.getLoc();
  auto loopControl = rewriter.getI32IntegerAttr(
      static_cast<uint32_t>(spirv::LoopControl::None));
  auto loopOp = rewriter.create<spirv::LoopOp>(loc, loopControl);
  loopOp.addEntryAndMergeBlock();

  OpBuilder::InsertionGuard guard(rewriter);
  // Create the block for the header.
  auto *header = new Block();
  // Insert the header.
  loopOp.body().getBlocks().insert(std::next(loopOp.body().begin(), 1), header);

  // Create the new induction variable to use.
  BlockArgument newIndVar =
      header->addArgument(forOperands.lowerBound().getType());
  Block *body = forOp.getBody();

  // Apply signature conversion to the body of the forOp. It has a single block,
  // with argument which is the induction variable. That has to be replaced with
  // the new induction variable.
  TypeConverter::SignatureConversion signatureConverter(
      body->getNumArguments());
  signatureConverter.remapInput(0, newIndVar);
  FailureOr<Block *> newBody = rewriter.convertRegionTypes(
      &forOp.getLoopBody(), typeConverter, &signatureConverter);
  if (failed(newBody))
    return failure();
  body = *newBody;

  // Delete the loop terminator.
  rewriter.eraseOp(body->getTerminator());

  // Move the blocks from the forOp into the loopOp. This is the body of the
  // loopOp.
  rewriter.inlineRegionBefore(forOp.getOperation()->getRegion(0), loopOp.body(),
                              std::next(loopOp.body().begin(), 2));

  // Branch into it from the entry.
  rewriter.setInsertionPointToEnd(&(loopOp.body().front()));
  rewriter.create<spirv::BranchOp>(loc, header, forOperands.lowerBound());

  // Generate the rest of the loop header.
  rewriter.setInsertionPointToEnd(header);
  auto *mergeBlock = loopOp.getMergeBlock();
  auto cmpOp = rewriter.create<spirv::SLessThanOp>(
      loc, rewriter.getI1Type(), newIndVar, forOperands.upperBound());
  rewriter.create<spirv::BranchConditionalOp>(
      loc, cmpOp, body, ArrayRef<Value>(), mergeBlock, ArrayRef<Value>());

  // Generate instructions to increment the step of the induction variable and
  // branch to the header.
  Block *continueBlock = loopOp.getContinueBlock();
  rewriter.setInsertionPointToEnd(continueBlock);

  // Add the step to the induction variable and branch to the header.
  Value updatedIndVar = rewriter.create<spirv::IAddOp>(
      loc, newIndVar.getType(), newIndVar, forOperands.step());
  rewriter.create<spirv::BranchOp>(loc, header, updatedIndVar);

  rewriter.eraseOp(forOp);
  return success();
}

//===----------------------------------------------------------------------===//
// scf::IfOp.
//===----------------------------------------------------------------------===//

LogicalResult
IfOpConversion::matchAndRewrite(scf::IfOp ifOp, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const {
  // When lowering `scf::IfOp` we explicitly create a selection header block
  // before the control flow diverges and a merge block where control flow
  // subsequently converges.
  scf::IfOpAdaptor ifOperands(operands);
  auto loc = ifOp.getLoc();

  // Create `spv.selection` operation, selection header block and merge block.
  auto selectionControl = rewriter.getI32IntegerAttr(
      static_cast<uint32_t>(spirv::SelectionControl::None));
  auto selectionOp = rewriter.create<spirv::SelectionOp>(loc, selectionControl);
  selectionOp.addMergeBlock();
  auto *mergeBlock = selectionOp.getMergeBlock();

  OpBuilder::InsertionGuard guard(rewriter);
  auto *selectionHeaderBlock = new Block();
  selectionOp.body().getBlocks().push_front(selectionHeaderBlock);

  // Inline `then` region before the merge block and branch to it.
  auto &thenRegion = ifOp.thenRegion();
  auto *thenBlock = &thenRegion.front();
  rewriter.setInsertionPointToEnd(&thenRegion.back());
  rewriter.create<spirv::BranchOp>(loc, mergeBlock);
  rewriter.inlineRegionBefore(thenRegion, mergeBlock);

  auto *elseBlock = mergeBlock;
  // If `else` region is not empty, inline that region before the merge block
  // and branch to it.
  if (!ifOp.elseRegion().empty()) {
    auto &elseRegion = ifOp.elseRegion();
    elseBlock = &elseRegion.front();
    rewriter.setInsertionPointToEnd(&elseRegion.back());
    rewriter.create<spirv::BranchOp>(loc, mergeBlock);
    rewriter.inlineRegionBefore(elseRegion, mergeBlock);
  }

  // Create a `spv.BranchConditional` operation for selection header block.
  rewriter.setInsertionPointToEnd(selectionHeaderBlock);
  rewriter.create<spirv::BranchConditionalOp>(loc, ifOperands.condition(),
                                              thenBlock, ArrayRef<Value>(),
                                              elseBlock, ArrayRef<Value>());

  rewriter.eraseOp(ifOp);
  return success();
}

void mlir::populateSCFToSPIRVPatterns(MLIRContext *context,
                                      SPIRVTypeConverter &typeConverter,
                                      OwningRewritePatternList &patterns) {
  patterns.insert<ForOpConversion, IfOpConversion, TerminatorOpConversion>(
      context, typeConverter);
}
