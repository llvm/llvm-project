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

namespace mlir {
struct ScfToSPIRVContextImpl {
  // Map between the spirv region control flow operation (spv.loop or
  // spv.selection) to the VariableOp created to store the region results. The
  // order of the VariableOp matches the order of the results.
  DenseMap<Operation *, SmallVector<spirv::VariableOp, 8>> outputVars;
};
} // namespace mlir

/// We use ScfToSPIRVContext to store information about the lowering of the scf
/// region that need to be used later on. When we lower scf.for/scf.if we create
/// VariableOp to store the results. We need to keep track of the VariableOp
/// created as we need to insert stores into them when lowering Yield. Those
/// StoreOp cannot be created earlier as they may use a different type than
/// yield operands.
ScfToSPIRVContext::ScfToSPIRVContext() {
  impl = std::make_unique<ScfToSPIRVContextImpl>();
}
ScfToSPIRVContext::~ScfToSPIRVContext() = default;

namespace {
/// Common class for all vector to GPU patterns.
template <typename OpTy>
class SCFToSPIRVPattern : public SPIRVOpLowering<OpTy> {
public:
  SCFToSPIRVPattern<OpTy>(MLIRContext *context, SPIRVTypeConverter &converter,
                          ScfToSPIRVContextImpl *scfToSPIRVContext)
      : SPIRVOpLowering<OpTy>::SPIRVOpLowering(context, converter),
        scfToSPIRVContext(scfToSPIRVContext) {}

protected:
  ScfToSPIRVContextImpl *scfToSPIRVContext;
};

/// Pattern to convert a scf::ForOp within kernel functions into spirv::LoopOp.
class ForOpConversion final : public SCFToSPIRVPattern<scf::ForOp> {
public:
  using SCFToSPIRVPattern<scf::ForOp>::SCFToSPIRVPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp forOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to convert a scf::IfOp within kernel functions into
/// spirv::SelectionOp.
class IfOpConversion final : public SCFToSPIRVPattern<scf::IfOp> {
public:
  using SCFToSPIRVPattern<scf::IfOp>::SCFToSPIRVPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp ifOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

class TerminatorOpConversion final : public SCFToSPIRVPattern<scf::YieldOp> {
public:
  using SCFToSPIRVPattern<scf::YieldOp>::SCFToSPIRVPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp terminatorOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

/// Helper function to replaces SCF op outputs with SPIR-V variable loads.
/// We create VariableOp to handle the results value of the control flow region.
/// spv.loop/spv.selection currently don't yield value. Right after the loop
/// we load the value from the allocation and use it as the SCF op result.
template <typename ScfOp, typename OpTy>
static void replaceSCFOutputValue(ScfOp scfOp, OpTy newOp,
                                  SPIRVTypeConverter &typeConverter,
                                  ConversionPatternRewriter &rewriter,
                                  ScfToSPIRVContextImpl *scfToSPIRVContext) {

  Location loc = scfOp.getLoc();
  auto &allocas = scfToSPIRVContext->outputVars[newOp];
  SmallVector<Value, 8> resultValue;
  for (Value result : scfOp.results()) {
    auto convertedType = typeConverter.convertType(result.getType());
    auto pointerType =
        spirv::PointerType::get(convertedType, spirv::StorageClass::Function);
    rewriter.setInsertionPoint(newOp);
    auto alloc = rewriter.create<spirv::VariableOp>(
        loc, pointerType, spirv::StorageClass::Function,
        /*initializer=*/nullptr);
    allocas.push_back(alloc);
    rewriter.setInsertionPointAfter(newOp);
    Value loadResult = rewriter.create<spirv::LoadOp>(loc, alloc);
    resultValue.push_back(loadResult);
  }
  rewriter.replaceOp(scfOp, resultValue);
}

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
  for (Value arg : forOperands.initArgs())
    header->addArgument(arg.getType());
  Block *body = forOp.getBody();

  // Apply signature conversion to the body of the forOp. It has a single block,
  // with argument which is the induction variable. That has to be replaced with
  // the new induction variable.
  TypeConverter::SignatureConversion signatureConverter(
      body->getNumArguments());
  signatureConverter.remapInput(0, newIndVar);
  for (unsigned i = 1, e = body->getNumArguments(); i < e; i++)
    signatureConverter.remapInput(i, header->getArgument(i));
  body = rewriter.applySignatureConversion(&forOp.getLoopBody(),
                                           signatureConverter);

  // Move the blocks from the forOp into the loopOp. This is the body of the
  // loopOp.
  rewriter.inlineRegionBefore(forOp.getOperation()->getRegion(0), loopOp.body(),
                              std::next(loopOp.body().begin(), 2));

  SmallVector<Value, 8> args(1, forOperands.lowerBound());
  args.append(forOperands.initArgs().begin(), forOperands.initArgs().end());
  // Branch into it from the entry.
  rewriter.setInsertionPointToEnd(&(loopOp.body().front()));
  rewriter.create<spirv::BranchOp>(loc, header, args);

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

  replaceSCFOutputValue(forOp, loopOp, typeConverter, rewriter,
                        scfToSPIRVContext);
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

  replaceSCFOutputValue(ifOp, selectionOp, typeConverter, rewriter,
                        scfToSPIRVContext);
  return success();
}

/// Yield is lowered to stores to the VariableOp created during lowering of the
/// parent region. For loops we also need to update the branch looping back to
/// the header with the loop carried values.
LogicalResult TerminatorOpConversion::matchAndRewrite(
    scf::YieldOp terminatorOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // If the region is return values, store each value into the associated
  // VariableOp created during lowering of the parent region.
  if (!operands.empty()) {
    auto loc = terminatorOp.getLoc();
    auto &allocas = scfToSPIRVContext->outputVars[terminatorOp.getParentOp()];
    assert(allocas.size() == operands.size());
    for (unsigned i = 0, e = operands.size(); i < e; i++)
      rewriter.create<spirv::StoreOp>(loc, allocas[i], operands[i]);
    if (isa<spirv::LoopOp>(terminatorOp.getParentOp())) {
      // For loops we also need to update the branch jumping back to the header.
      auto br =
          cast<spirv::BranchOp>(rewriter.getInsertionBlock()->getTerminator());
      SmallVector<Value, 8> args(br.getBlockArguments());
      args.append(operands.begin(), operands.end());
      rewriter.setInsertionPoint(br);
      rewriter.create<spirv::BranchOp>(terminatorOp.getLoc(), br.getTarget(),
                                       args);
      rewriter.eraseOp(br);
    }
  }
  rewriter.eraseOp(terminatorOp);
  return success();
}

void mlir::populateSCFToSPIRVPatterns(MLIRContext *context,
                                      SPIRVTypeConverter &typeConverter,
                                      ScfToSPIRVContext &scfToSPIRVContext,
                                      OwningRewritePatternList &patterns) {
  patterns.insert<ForOpConversion, IfOpConversion, TerminatorOpConversion>(
      context, typeConverter, scfToSPIRVContext.getImpl());
}
