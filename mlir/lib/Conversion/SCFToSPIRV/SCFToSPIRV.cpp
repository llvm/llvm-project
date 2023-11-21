//===- SCFToSPIRV.cpp - SCF to SPIR-V Patterns ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert SCF dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Context
//===----------------------------------------------------------------------===//

namespace mlir {
struct ScfToSPIRVContextImpl {
  // Map between the spirv region control flow operation (spirv.mlir.loop or
  // spirv.mlir.selection) to the VariableOp created to store the region
  // results. The order of the VariableOp matches the order of the results.
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
  impl = std::make_unique<::ScfToSPIRVContextImpl>();
}

ScfToSPIRVContext::~ScfToSPIRVContext() = default;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Replaces SCF op outputs with SPIR-V variable loads.
/// We create VariableOp to handle the results value of the control flow region.
/// spirv.mlir.loop/spirv.mlir.selection currently don't yield value. Right
/// after the loop we load the value from the allocation and use it as the SCF
/// op result.
template <typename ScfOp, typename OpTy>
void replaceSCFOutputValue(ScfOp scfOp, OpTy newOp,
                           ConversionPatternRewriter &rewriter,
                           ScfToSPIRVContextImpl *scfToSPIRVContext,
                           ArrayRef<Type> returnTypes) {

  Location loc = scfOp.getLoc();
  auto &allocas = scfToSPIRVContext->outputVars[newOp];
  // Clearing the allocas is necessary in case a dialect conversion path failed
  // previously, and this is the second attempt of this conversion.
  allocas.clear();
  SmallVector<Value, 8> resultValue;
  for (Type convertedType : returnTypes) {
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

Region::iterator getBlockIt(Region &region, unsigned index) {
  return std::next(region.begin(), index);
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Common class for all vector to GPU patterns.
template <typename OpTy>
class SCFToSPIRVPattern : public OpConversionPattern<OpTy> {
public:
  SCFToSPIRVPattern<OpTy>(MLIRContext *context, SPIRVTypeConverter &converter,
                          ScfToSPIRVContextImpl *scfToSPIRVContext)
      : OpConversionPattern<OpTy>::OpConversionPattern(converter, context),
        scfToSPIRVContext(scfToSPIRVContext), typeConverter(converter) {}

protected:
  ScfToSPIRVContextImpl *scfToSPIRVContext;
  // FIXME: We explicitly keep a reference of the type converter here instead of
  // passing it to OpConversionPattern during construction. This effectively
  // bypasses the conversion framework's automation on type conversion. This is
  // needed right now because the conversion framework will unconditionally
  // legalize all types used by SCF ops upon discovering them, for example, the
  // types of loop carried values. We use SPIR-V variables for those loop
  // carried values. Depending on the available capabilities, the SPIR-V
  // variable can be different, for example, cooperative matrix or normal
  // variable. We'd like to detach the conversion of the loop carried values
  // from the SCF ops (which is mainly a region). So we need to "mark" types
  // used by SCF ops as legal, if to use the conversion framework for type
  // conversion. There isn't a straightforward way to do that yet, as when
  // converting types, ops aren't taken into consideration. Therefore, we just
  // bypass the framework's type conversion for now.
  SPIRVTypeConverter &typeConverter;
};

//===----------------------------------------------------------------------===//
// scf::ForOp
//===----------------------------------------------------------------------===//

/// Pattern to convert a scf::ForOp within kernel functions into spirv::LoopOp.
struct ForOpConversion final : SCFToSPIRVPattern<scf::ForOp> {
  using SCFToSPIRVPattern::SCFToSPIRVPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp forOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // scf::ForOp can be lowered to the structured control flow represented by
    // spirv::LoopOp by making the continue block of the spirv::LoopOp the loop
    // latch and the merge block the exit block. The resulting spirv::LoopOp has
    // a single back edge from the continue to header block, and a single exit
    // from header to merge.
    auto loc = forOp.getLoc();
    auto loopOp = rewriter.create<spirv::LoopOp>(loc, spirv::LoopControl::None);
    loopOp.addEntryAndMergeBlock();

    OpBuilder::InsertionGuard guard(rewriter);
    // Create the block for the header.
    auto *header = new Block();
    // Insert the header.
    loopOp.getBody().getBlocks().insert(getBlockIt(loopOp.getBody(), 1),
                                        header);

    // Create the new induction variable to use.
    Value adapLowerBound = adaptor.getLowerBound();
    BlockArgument newIndVar =
        header->addArgument(adapLowerBound.getType(), adapLowerBound.getLoc());
    for (Value arg : adaptor.getInitArgs())
      header->addArgument(arg.getType(), arg.getLoc());
    Block *body = forOp.getBody();

    // Apply signature conversion to the body of the forOp. It has a single
    // block, with argument which is the induction variable. That has to be
    // replaced with the new induction variable.
    TypeConverter::SignatureConversion signatureConverter(
        body->getNumArguments());
    signatureConverter.remapInput(0, newIndVar);
    for (unsigned i = 1, e = body->getNumArguments(); i < e; i++)
      signatureConverter.remapInput(i, header->getArgument(i));
    body = rewriter.applySignatureConversion(&forOp.getRegion(),
                                             signatureConverter);

    // Move the blocks from the forOp into the loopOp. This is the body of the
    // loopOp.
    rewriter.inlineRegionBefore(forOp->getRegion(0), loopOp.getBody(),
                                getBlockIt(loopOp.getBody(), 2));

    SmallVector<Value, 8> args(1, adaptor.getLowerBound());
    args.append(adaptor.getInitArgs().begin(), adaptor.getInitArgs().end());
    // Branch into it from the entry.
    rewriter.setInsertionPointToEnd(&(loopOp.getBody().front()));
    rewriter.create<spirv::BranchOp>(loc, header, args);

    // Generate the rest of the loop header.
    rewriter.setInsertionPointToEnd(header);
    auto *mergeBlock = loopOp.getMergeBlock();
    auto cmpOp = rewriter.create<spirv::SLessThanOp>(
        loc, rewriter.getI1Type(), newIndVar, adaptor.getUpperBound());

    rewriter.create<spirv::BranchConditionalOp>(
        loc, cmpOp, body, ArrayRef<Value>(), mergeBlock, ArrayRef<Value>());

    // Generate instructions to increment the step of the induction variable and
    // branch to the header.
    Block *continueBlock = loopOp.getContinueBlock();
    rewriter.setInsertionPointToEnd(continueBlock);

    // Add the step to the induction variable and branch to the header.
    Value updatedIndVar = rewriter.create<spirv::IAddOp>(
        loc, newIndVar.getType(), newIndVar, adaptor.getStep());
    rewriter.create<spirv::BranchOp>(loc, header, updatedIndVar);

    // Infer the return types from the init operands. Vector type may get
    // converted to CooperativeMatrix or to Vector type, to avoid having complex
    // extra logic to figure out the right type we just infer it from the Init
    // operands.
    SmallVector<Type, 8> initTypes;
    for (auto arg : adaptor.getInitArgs())
      initTypes.push_back(arg.getType());
    replaceSCFOutputValue(forOp, loopOp, rewriter, scfToSPIRVContext,
                          initTypes);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// scf::IfOp
//===----------------------------------------------------------------------===//

/// Pattern to convert a scf::IfOp within kernel functions into
/// spirv::SelectionOp.
struct IfOpConversion : SCFToSPIRVPattern<scf::IfOp> {
  using SCFToSPIRVPattern::SCFToSPIRVPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp ifOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // When lowering `scf::IfOp` we explicitly create a selection header block
    // before the control flow diverges and a merge block where control flow
    // subsequently converges.
    auto loc = ifOp.getLoc();

    // Create `spirv.selection` operation, selection header block and merge
    // block.
    auto selectionOp =
        rewriter.create<spirv::SelectionOp>(loc, spirv::SelectionControl::None);
    auto *mergeBlock = rewriter.createBlock(&selectionOp.getBody(),
                                            selectionOp.getBody().end());
    rewriter.create<spirv::MergeOp>(loc);

    OpBuilder::InsertionGuard guard(rewriter);
    auto *selectionHeaderBlock =
        rewriter.createBlock(&selectionOp.getBody().front());

    // Inline `then` region before the merge block and branch to it.
    auto &thenRegion = ifOp.getThenRegion();
    auto *thenBlock = &thenRegion.front();
    rewriter.setInsertionPointToEnd(&thenRegion.back());
    rewriter.create<spirv::BranchOp>(loc, mergeBlock);
    rewriter.inlineRegionBefore(thenRegion, mergeBlock);

    auto *elseBlock = mergeBlock;
    // If `else` region is not empty, inline that region before the merge block
    // and branch to it.
    if (!ifOp.getElseRegion().empty()) {
      auto &elseRegion = ifOp.getElseRegion();
      elseBlock = &elseRegion.front();
      rewriter.setInsertionPointToEnd(&elseRegion.back());
      rewriter.create<spirv::BranchOp>(loc, mergeBlock);
      rewriter.inlineRegionBefore(elseRegion, mergeBlock);
    }

    // Create a `spirv.BranchConditional` operation for selection header block.
    rewriter.setInsertionPointToEnd(selectionHeaderBlock);
    rewriter.create<spirv::BranchConditionalOp>(loc, adaptor.getCondition(),
                                                thenBlock, ArrayRef<Value>(),
                                                elseBlock, ArrayRef<Value>());

    SmallVector<Type, 8> returnTypes;
    for (auto result : ifOp.getResults()) {
      auto convertedType = typeConverter.convertType(result.getType());
      if (!convertedType)
        return rewriter.notifyMatchFailure(
            loc,
            llvm::formatv("failed to convert type '{0}'", result.getType()));

      returnTypes.push_back(convertedType);
    }
    replaceSCFOutputValue(ifOp, selectionOp, rewriter, scfToSPIRVContext,
                          returnTypes);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// scf::YieldOp
//===----------------------------------------------------------------------===//

struct TerminatorOpConversion final : SCFToSPIRVPattern<scf::YieldOp> {
public:
  using SCFToSPIRVPattern::SCFToSPIRVPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp terminatorOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange operands = adaptor.getOperands();

    Operation *parent = terminatorOp->getParentOp();

    // TODO: Implement conversion for the remaining `scf` ops.
    if (parent->getDialect()->getNamespace() ==
            scf::SCFDialect::getDialectNamespace() &&
        !isa<scf::IfOp, scf::ForOp, scf::WhileOp>(parent))
      return rewriter.notifyMatchFailure(
          terminatorOp,
          llvm::formatv("conversion not supported for parent op: '{0}'",
                        parent->getName()));

    // If the region return values, store each value into the associated
    // VariableOp created during lowering of the parent region.
    if (!operands.empty()) {
      auto &allocas = scfToSPIRVContext->outputVars[parent];
      if (allocas.size() != operands.size())
        return failure();

      auto loc = terminatorOp.getLoc();
      for (unsigned i = 0, e = operands.size(); i < e; i++)
        rewriter.create<spirv::StoreOp>(loc, allocas[i], operands[i]);
      if (isa<spirv::LoopOp>(parent)) {
        // For loops we also need to update the branch jumping back to the
        // header.
        auto br = cast<spirv::BranchOp>(
            rewriter.getInsertionBlock()->getTerminator());
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
};

//===----------------------------------------------------------------------===//
// scf::WhileOp
//===----------------------------------------------------------------------===//

struct WhileOpConversion final : SCFToSPIRVPattern<scf::WhileOp> {
  using SCFToSPIRVPattern::SCFToSPIRVPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp whileOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = whileOp.getLoc();
    auto loopOp = rewriter.create<spirv::LoopOp>(loc, spirv::LoopControl::None);
    loopOp.addEntryAndMergeBlock();

    Region &beforeRegion = whileOp.getBefore();
    Region &afterRegion = whileOp.getAfter();

    if (failed(rewriter.convertRegionTypes(&beforeRegion, typeConverter)) ||
        failed(rewriter.convertRegionTypes(&afterRegion, typeConverter)))
      return rewriter.notifyMatchFailure(whileOp,
                                         "Failed to convert region types");

    OpBuilder::InsertionGuard guard(rewriter);

    Block &entryBlock = *loopOp.getEntryBlock();
    Block &beforeBlock = beforeRegion.front();
    Block &afterBlock = afterRegion.front();
    Block &mergeBlock = *loopOp.getMergeBlock();

    auto cond = cast<scf::ConditionOp>(beforeBlock.getTerminator());
    SmallVector<Value> condArgs;
    if (failed(rewriter.getRemappedValues(cond.getArgs(), condArgs)))
      return failure();

    Value conditionVal = rewriter.getRemappedValue(cond.getCondition());
    if (!conditionVal)
      return failure();

    auto yield = cast<scf::YieldOp>(afterBlock.getTerminator());
    SmallVector<Value> yieldArgs;
    if (failed(rewriter.getRemappedValues(yield.getResults(), yieldArgs)))
      return failure();

    // Move the while before block as the initial loop header block.
    rewriter.inlineRegionBefore(beforeRegion, loopOp.getBody(),
                                getBlockIt(loopOp.getBody(), 1));

    // Move the while after block as the initial loop body block.
    rewriter.inlineRegionBefore(afterRegion, loopOp.getBody(),
                                getBlockIt(loopOp.getBody(), 2));

    // Jump from the loop entry block to the loop header block.
    rewriter.setInsertionPointToEnd(&entryBlock);
    rewriter.create<spirv::BranchOp>(loc, &beforeBlock, adaptor.getInits());

    auto condLoc = cond.getLoc();

    SmallVector<Value> resultValues(condArgs.size());

    // For other SCF ops, the scf.yield op yields the value for the whole SCF
    // op. So we use the scf.yield op as the anchor to create/load/store SPIR-V
    // local variables. But for the scf.while op, the scf.yield op yields a
    // value for the before region, which may not matching the whole op's
    // result. Instead, the scf.condition op returns values matching the whole
    // op's results. So we need to create/load/store variables according to
    // that.
    for (const auto &it : llvm::enumerate(condArgs)) {
      auto res = it.value();
      auto i = it.index();
      auto pointerType =
          spirv::PointerType::get(res.getType(), spirv::StorageClass::Function);

      // Create local variables before the scf.while op.
      rewriter.setInsertionPoint(loopOp);
      auto alloc = rewriter.create<spirv::VariableOp>(
          condLoc, pointerType, spirv::StorageClass::Function,
          /*initializer=*/nullptr);

      // Load the final result values after the scf.while op.
      rewriter.setInsertionPointAfter(loopOp);
      auto loadResult = rewriter.create<spirv::LoadOp>(condLoc, alloc);
      resultValues[i] = loadResult;

      // Store the current iteration's result value.
      rewriter.setInsertionPointToEnd(&beforeBlock);
      rewriter.create<spirv::StoreOp>(condLoc, alloc, res);
    }

    rewriter.setInsertionPointToEnd(&beforeBlock);
    rewriter.replaceOpWithNewOp<spirv::BranchConditionalOp>(
        cond, conditionVal, &afterBlock, condArgs, &mergeBlock, std::nullopt);

    // Convert the scf.yield op to a branch back to the header block.
    rewriter.setInsertionPointToEnd(&afterBlock);
    rewriter.replaceOpWithNewOp<spirv::BranchOp>(yield, &beforeBlock,
                                                 yieldArgs);

    rewriter.replaceOp(whileOp, resultValues);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

void mlir::populateSCFToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                      ScfToSPIRVContext &scfToSPIRVContext,
                                      RewritePatternSet &patterns) {
  patterns.add<ForOpConversion, IfOpConversion, TerminatorOpConversion,
               WhileOpConversion>(patterns.getContext(), typeConverter,
                                  scfToSPIRVContext.getImpl());
}
