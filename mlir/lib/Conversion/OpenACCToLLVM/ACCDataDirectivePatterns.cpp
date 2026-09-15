//===- ACCDataDirectivePatterns.cpp - ACC data to LLVM ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowering of the OpenACC data constructs - data, enter data, exit data and
// update - to the data entry points of the OpenACC runtime. Each construct
// becomes one call per point where it establishes or tears down its mappings,
// which carries the argument arrays its operands are packed into, the queue
// its async clause selects and, before them, the wait its wait clause asks
// for. An if clause guards the calls it applies to, and the body of a
// structured construct is left in place of the construct.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenACCToLLVM/ACCDataRuntime.h"
#include "mlir/Conversion/OpenACCToLLVM/ACCToLLVM.h"
#include "mlir/Conversion/OpenACCToLLVM/ACCToLLVMUtils.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"

using namespace mlir;
using namespace mlir::acc;

/// Emits the call to the mapping entry point \p fn for \p mappingOperands, the
/// data-clause operands of a data construct, paired with their type converted
/// values in \p convertedOperands.
static LogicalResult emitDataRuntimeCall(
    Location loc, RuntimeFunction fn, ValueRange mappingOperands,
    ValueRange convertedOperands, Value asyncQueue, ACCDataCallKind callKind,
    ConversionPatternRewriter &rewriter, OpenACCSupport &accSupport,
    Region &globalSymbolRegion, SymbolTable &symbolTable,
    const ACCRuntimeCallConfig &config) {
  if (mappingOperands.empty())
    return success();

  Operation *firstMapOp = mappingOperands.front().getDefiningOp();
  ACCDataRuntimeArgs runtimeArgs;
  if (failed(emitACCDataRuntimeArgs(
          loc, mappingOperands, convertedOperands, rewriter, globalSymbolRegion,
          accSupport, config, runtimeArgs, callKind, &symbolTable)))
    return rewriter.notifyMatchFailure(
        firstMapOp, "unsupported OpenACC data-clause operand");

  SmallVector<Value> arguments = runtimeArgs.getCallArgs();
  arguments.push_back(asyncQueue);
  return createRuntimeCall(loc, rewriter, globalSymbolRegion, symbolTable, fn,
                           config, arguments);
}

/// Splices \p region, the body of a structured data construct, into the block
/// holding \p op, so that the construct itself can be erased. The terminators
/// of the region branch to the code that follows \p op.
static void spliceDataRegion(Operation *op, Region &region,
                             RewriterBase &rewriter) {
  Block *prev = op->getBlock();
  Block *succ = rewriter.splitBlock(prev, Block::iterator(op));
  rewriter.setInsertionPointToEnd(prev);
  LLVM::BrOp::create(rewriter, op->getLoc(), &region.getBlocks().front());
  for (Block &block : region.getBlocks()) {
    Operation *terminator = block.getTerminator();
    if (isa<acc::TerminatorOp>(terminator)) {
      rewriter.setInsertionPoint(terminator);
      LLVM::BrOp::create(rewriter, op->getLoc(), succ);
      rewriter.eraseOp(terminator);
    }
  }
  rewriter.inlineRegionBefore(region, succ);
}

namespace {
template <typename OpTy>
struct ACCDataDirectivePattern : public ConvertOpToLLVMPattern<OpTy> {
  ACCDataDirectivePattern(const LLVMTypeConverter &converter,
                          OpenACCSupport &accSupport,
                          Region &globalSymbolRegion, SymbolTable &symbolTable,
                          const ACCRuntimeCallConfig &config,
                          DeviceType clauseDeviceType,
                          PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<OpTy>(converter, benefit),
        accSupport(accSupport), globalSymbolRegion(globalSymbolRegion),
        symbolTable(symbolTable), config(config),
        clauseDeviceType(clauseDeviceType) {}

  /// Returns the queue that the mapping calls of \p op run on.
  Value getAsyncQueue(OpTy op, ConversionPatternRewriter &rewriter) const {
    return acc::getAsyncQueue(op, clauseDeviceType, rewriter, config);
  }

  /// Emits the wait that a wait clause on \p op asks for before its mapping
  /// calls.
  LogicalResult emitWaitClause(OpTy op, Value asyncQueue,
                               ConversionPatternRewriter &rewriter) const {
    return acc::emitWaitClause(op, clauseDeviceType, asyncQueue, rewriter,
                               accSupport, globalSymbolRegion, symbolTable,
                               config);
  }

  /// Emits one mapping call for the data-clause operands of \p op.
  LogicalResult emitMappingCall(OpTy op, ValueRange convertedOperands,
                                Location loc, RuntimeFunction fn,
                                ACCDataCallKind callKind, Value asyncQueue,
                                ConversionPatternRewriter &rewriter) const {
    return emitDataRuntimeCall(loc, fn, op.getDataClauseOperands(),
                               convertedOperands, asyncQueue, callKind,
                               rewriter, accSupport, globalSymbolRegion,
                               symbolTable, config);
  }

  OpenACCSupport &accSupport;
  Region &globalSymbolRegion;
  SymbolTable &symbolTable;
  ACCRuntimeCallConfig config;
  DeviceType clauseDeviceType;
};

struct DataOpLowering : public ACCDataDirectivePattern<DataOp> {
  using ACCDataDirectivePattern<DataOp>::ACCDataDirectivePattern;

  LogicalResult
  matchAndRewrite(DataOp op, DataOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The mappings are established where the clause operations are, and torn
    // down where the exit operations of the construct were.
    ValueRange mappingOperands = op.getDataClauseOperands();
    Location beginLoc = mappingOperands.empty()
                            ? op.getLoc()
                            : mappingOperands.front().getLoc();
    Location endLoc =
        acc::getMappingExitLoc(mappingOperands).value_or(beginLoc);
    Value asyncQueue = getAsyncQueue(op, rewriter);
    if (failed(emitWaitClause(op, asyncQueue, rewriter)))
      return failure();

    auto emitBegin = [&]() {
      return emitMappingCall(op, adaptor.getDataClauseOperands(), beginLoc,
                             RuntimeFunction::ACCRTL_tgt_acc_data_begin,
                             ACCDataCallKind::DataEnter, asyncQueue, rewriter);
    };
    if (failed(acc::emitGuardedByIfCond(beginLoc, adaptor.getIfCond(), rewriter,
                                        emitBegin)))
      return failure();

    rewriter.setInsertionPointAfter(op);
    auto emitEnd = [&]() {
      return emitMappingCall(op, adaptor.getDataClauseOperands(), endLoc,
                             RuntimeFunction::ACCRTL_tgt_acc_data_end,
                             ACCDataCallKind::DataExit, asyncQueue, rewriter);
    };
    if (failed(acc::emitGuardedByIfCond(endLoc, adaptor.getIfCond(), rewriter,
                                        emitEnd)))
      return failure();

    spliceDataRegion(op, op.getRegion(), rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

/// Lowering shared by the directives that map their operands with a single
/// call: enter data, exit data and update.
template <typename OpTy, RuntimeFunction fn, ACCDataCallKind callKind>
struct ACCDataCallLowering : public ACCDataDirectivePattern<OpTy> {
  using ACCDataDirectivePattern<OpTy>::ACCDataDirectivePattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value asyncQueue = this->getAsyncQueue(op, rewriter);
    if (failed(this->emitWaitClause(op, asyncQueue, rewriter)))
      return failure();

    auto emit = [&]() {
      return this->emitMappingCall(op, adaptor.getDataClauseOperands(), loc, fn,
                                   callKind, asyncQueue, rewriter);
    };
    if (failed(
            acc::emitGuardedByIfCond(loc, adaptor.getIfCond(), rewriter, emit)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

using EnterDataOpLowering =
    ACCDataCallLowering<EnterDataOp, RuntimeFunction::ACCRTL_tgt_acc_data_enter,
                        ACCDataCallKind::DataEnter>;
using ExitDataOpLowering =
    ACCDataCallLowering<ExitDataOp, RuntimeFunction::ACCRTL_tgt_acc_data_exit,
                        ACCDataCallKind::DataExit>;
using UpdateOpLowering =
    ACCDataCallLowering<UpdateOp, RuntimeFunction::ACCRTL_tgt_acc_data_update,
                        ACCDataCallKind::DataEnter>;

/// A data clause operation only describes a mapping; the runtime calls of the
/// construct carry everything it said, so the operation itself goes away once
/// they are in place. Whatever else referred to the mapped object is given the
/// address of that object, which is what the clause result stood for.
template <typename OpTy>
struct DataEntryOpLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getVar());
    return success();
  }
};

/// A data exit operation and the bounds of a mapping have no result to stand
/// in for, so they are simply removed with the construct that used them.
template <typename OpTy>
struct EraseDataClauseOp : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::configureACCDataDirectiveConversionLegality(
    ConversionTarget &target) {
  // The map entries themselves stay legal: a construct removes the ones it
  // consumes, and the ones describing a construct that is lowered elsewhere
  // have to survive this conversion.
  target.addIllegalOp<acc::DataOp, acc::EnterDataOp, acc::ExitDataOp,
                      acc::UpdateOp>();
}

void mlir::populateACCDataClauseOpPatterns(LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns) {
  // Bounds describe a mapping rather than data, and they are removed together
  // with the clause operations holding them, so their type only has to survive
  // this conversion.
  converter.addConversion(
      [](acc::DataBoundsType type) -> Type { return type; });

  patterns.add<
      DataEntryOpLowering<acc::MapInfoOp>, DataEntryOpLowering<acc::CopyinOp>,
      DataEntryOpLowering<acc::CreateOp>, DataEntryOpLowering<acc::PresentOp>,
      DataEntryOpLowering<acc::NoCreateOp>, DataEntryOpLowering<acc::AttachOp>,
      DataEntryOpLowering<acc::DevicePtrOp>,
      DataEntryOpLowering<acc::GetDevicePtrOp>,
      DataEntryOpLowering<acc::UpdateDeviceOp>,
      DataEntryOpLowering<acc::PrivateOp>,
      DataEntryOpLowering<acc::FirstprivateOp>,
      DataEntryOpLowering<acc::FirstprivateMapInitialOp>,
      DataEntryOpLowering<acc::DeclareDeviceResidentOp>,
      EraseDataClauseOp<acc::CopyoutOp>, EraseDataClauseOp<acc::DeleteOp>,
      EraseDataClauseOp<acc::DetachOp>, EraseDataClauseOp<acc::UpdateHostOp>,
      EraseDataClauseOp<acc::DataBoundsOp>>(converter);
}

void mlir::populateACCDataDirectivePatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    acc::OpenACCSupport &accSupport, Region &globalSymbolRegion,
    SymbolTable &symbolTable, const acc::ACCRuntimeCallConfig &config,
    acc::DeviceType clauseDeviceType) {
  patterns.add<DataOpLowering, EnterDataOpLowering, ExitDataOpLowering,
               UpdateOpLowering>(converter, accSupport, globalSymbolRegion,
                                 symbolTable, config, clauseDeviceType);
}
