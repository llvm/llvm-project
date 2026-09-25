//===- ACCDataDirectivePatterns.cpp - ACC data to LLVM ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowering of the OpenACC data constructs - data, enter data, exit data,
// update and declare - to the data entry points of the OpenACC runtime. Each
// construct becomes one call per point where it establishes or tears down its
// mappings, which carries the argument arrays its operands are packed into,
// the queue its async clause selects and, before them, the wait its wait
// clause asks for. An if clause guards the calls it applies to, and the body
// of a structured construct is left in place of the construct.
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
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;
using namespace mlir::acc;

/// Emits the call to the mapping entry point \p fn for \p mappingOperands, the
/// data-clause operands of a data construct, paired with their type converted
/// values in \p convertedOperands.
static LogicalResult emitDataRuntimeCall(
    Location loc, RuntimeFunction fn, ValueRange mappingOperands,
    ValueRange convertedOperands, ACCDataCallKind callKind,
    ConversionPatternRewriter &rewriter, OpenACCSupport &accSupport,
    Region &globalSymbolRegion, SymbolTable &symbolTable,
    const ACCRuntimeCallConfig &config, ArrayRef<Value> trailingArguments) {
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
  arguments.append(trailingArguments.begin(), trailingArguments.end());
  return createRuntimeCall(loc, rewriter, globalSymbolRegion, symbolTable, fn,
                           config, arguments);
}

/// Returns the action that the declare-action recipe holding \p op carries
/// out, which the recipe states by naming itself in the matching slot of its
/// `acc.declare_action` attribute. Returns a null attribute when \p op is not
/// in such a recipe.
static DeclareActionAttr getEnclosingDeclareAction(Operation *op) {
  auto func = dyn_cast<FunctionOpInterface>(op);
  if (!func)
    func = op->getParentOfType<FunctionOpInterface>();
  if (!func)
    return {};
  return func->getAttrOfType<DeclareActionAttr>(getDeclareActionAttrName());
}

static Value getDeclareAsyncQueue(Location loc,
                                  ConversionPatternRewriter &rewriter,
                                  const ACCRuntimeCallConfig &config) {
  return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                  config.getAsyncSyncRuntimeValue());
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
                               convertedOperands, callKind, rewriter,
                               accSupport, globalSymbolRegion, symbolTable,
                               config, {asyncQueue});
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

    acc::spliceConstructRegion(op, op.getRegion(), rewriter);
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

/// Structured `acc.declare_enter` (token used by a matching exit) maps like
/// `acc.data` enter. In a declare-action recipe the object the directive
/// names is allocated by the host, so only the device mirror of it is
/// allocated here. An unstructured enter registers the objects it names with
/// `__tgt_acc_declare`.
struct DeclareEnterOpLowering : public ConvertOpToLLVMPattern<DeclareEnterOp> {
  DeclareEnterOpLowering(const LLVMTypeConverter &converter,
                         OpenACCSupport &accSupport, Region &globalSymbolRegion,
                         SymbolTable &symbolTable,
                         const ACCRuntimeCallConfig &config,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<DeclareEnterOp>(converter, benefit),
        accSupport(accSupport), globalSymbolRegion(globalSymbolRegion),
        symbolTable(symbolTable), config(config) {}

  LogicalResult
  matchAndRewrite(DeclareEnterOp op, DeclareEnterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ValueRange mappingOperands = op.getDataClauseOperands();
    ValueRange convertedOperands = adaptor.getDataClauseOperands();
    Value asyncQueue = getDeclareAsyncQueue(loc, rewriter, config);

    LogicalResult emitted = success();
    if (getEnclosingDeclareAction(op)) {
      emitted = emitDataRuntimeCall(
          loc, RuntimeFunction::ACCRTL_tgt_acc_mirror_alloc, mappingOperands,
          convertedOperands, ACCDataCallKind::DataEnter, rewriter, accSupport,
          globalSymbolRegion, symbolTable, config, {});
    } else if (!op.getToken().use_empty()) {
      emitted = emitDataRuntimeCall(
          loc, RuntimeFunction::ACCRTL_tgt_acc_data_begin, mappingOperands,
          convertedOperands, ACCDataCallKind::DataEnter, rewriter, accSupport,
          globalSymbolRegion, symbolTable, config, {asyncQueue});
    } else {
      Value binaryDescriptor =
          config.createDeclareBinaryDescriptor(loc, rewriter);
      emitted = emitDataRuntimeCall(
          loc, RuntimeFunction::ACCRTL_tgt_acc_declare, mappingOperands,
          convertedOperands, ACCDataCallKind::DataEnter, rewriter, accSupport,
          globalSymbolRegion, symbolTable, config,
          {asyncQueue, binaryDescriptor});
    }
    if (failed(emitted))
      return failure();

    if (op.getToken().use_empty()) {
      rewriter.eraseOp(op);
    } else {
      // The token only ties a structured enter to its exit. Keep a converted
      // placeholder until the exit is erased; it carries no runtime state.
      Type tokenTy = getTypeConverter()->convertType(op.getToken().getType());
      rewriter.replaceOp(op, LLVM::ZeroOp::create(rewriter, loc, tokenTy));
    }
    return success();
  }

  OpenACCSupport &accSupport;
  Region &globalSymbolRegion;
  SymbolTable &symbolTable;
  ACCRuntimeCallConfig config;
};

/// A recipe that runs before the host object is deallocated frees the device
/// mirror of it, as the mapping is torn down once the host storage is gone.
/// Every other `acc.declare_exit` ends the mapping with `__tgt_acc_data_end`.
struct DeclareExitOpLowering : public ConvertOpToLLVMPattern<DeclareExitOp> {
  DeclareExitOpLowering(const LLVMTypeConverter &converter,
                        OpenACCSupport &accSupport, Region &globalSymbolRegion,
                        SymbolTable &symbolTable,
                        const ACCRuntimeCallConfig &config,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<DeclareExitOp>(converter, benefit),
        accSupport(accSupport), globalSymbolRegion(globalSymbolRegion),
        symbolTable(symbolTable), config(config) {}

  LogicalResult
  matchAndRewrite(DeclareExitOp op, DeclareExitOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ValueRange mappingOperands = op.getDataClauseOperands();
    ValueRange convertedOperands = adaptor.getDataClauseOperands();
    Value asyncQueue = getDeclareAsyncQueue(loc, rewriter, config);

    LogicalResult emitted = success();
    DeclareActionAttr action = getEnclosingDeclareAction(op);
    if (action && action.getPreDealloc()) {
      emitted = emitDataRuntimeCall(
          loc, RuntimeFunction::ACCRTL_tgt_acc_mirror_dealloc, mappingOperands,
          convertedOperands, ACCDataCallKind::DataEnter, rewriter, accSupport,
          globalSymbolRegion, symbolTable, config, {});
    } else {
      emitted = emitDataRuntimeCall(
          loc, RuntimeFunction::ACCRTL_tgt_acc_data_end, mappingOperands,
          convertedOperands, ACCDataCallKind::DataExit, rewriter, accSupport,
          globalSymbolRegion, symbolTable, config, {asyncQueue});
    }
    if (failed(emitted))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }

  OpenACCSupport &accSupport;
  Region &globalSymbolRegion;
  SymbolTable &symbolTable;
  ACCRuntimeCallConfig config;
};

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

/// An `acc.getdeviceptr` clause stands for the device address of an object
/// whose mapping was established elsewhere. Where it names the object of a
/// data exit operation, the runtime call of that construct carries the host
/// address and looks the device one up itself, so the clause is replaced by
/// the address of the object as any other data entry clause is. Read as a
/// value, it is the runtime that has to be asked for the address.
struct GetDevicePtrOpLowering
    : public ConvertOpToLLVMPattern<acc::GetDevicePtrOp> {
  GetDevicePtrOpLowering(const LLVMTypeConverter &converter,
                         OpenACCSupport &accSupport, Region &globalSymbolRegion,
                         SymbolTable &symbolTable,
                         const ACCRuntimeCallConfig &config,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<acc::GetDevicePtrOp>(converter, benefit),
        accSupport(accSupport), globalSymbolRegion(globalSymbolRegion),
        symbolTable(symbolTable), config(config) {}

  /// Returns whether \p user only states the mapping the clause describes,
  /// which is what a data exit operation and the construct holding the clause
  /// do.
  static bool isMappingUse(Operation *user) {
    return isa<ACC_DATA_EXIT_OPS, ACC_COMPUTE_AND_DATA_CONSTRUCT_OPS,
               acc::KernelEnvironmentOp>(user);
  }

  LogicalResult
  matchAndRewrite(acc::GetDevicePtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool readAsValue = llvm::any_of(
        op->getUsers(), [](Operation *user) { return !isMappingUse(user); });
    if (!readAsValue) {
      rewriter.replaceOp(op, adaptor.getVar());
      return success();
    }
    // The mapping hands the host address of the object to the runtime call of
    // the construct, so the clause cannot stand for the device address at the
    // same time.
    if (llvm::any_of(op->getUsers(), isMappingUse)) {
      (void)accSupport.emitNYI(
          op.getLoc(), "device address read from a clause that also states a "
                       "mapping of the object");
      return failure();
    }

    FailureOr<Value> devicePtr = acc::emitGetDevicePtrCall(
        op, adaptor.getVar(), /*ifPresent=*/false, rewriter, globalSymbolRegion,
        symbolTable, config);
    if (failed(devicePtr))
      return failure();
    rewriter.replaceOp(op, *devicePtr);
    return success();
  }

  OpenACCSupport &accSupport;
  Region &globalSymbolRegion;
  SymbolTable &symbolTable;
  ACCRuntimeCallConfig config;
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
  // have to survive this conversion. An `acc.getdeviceptr` read as a value is
  // the exception, as the address it stands for is only known to the runtime,
  // so leaving it behind would state a mapping that nothing carries out.
  target.addIllegalOp<acc::DataOp, acc::EnterDataOp, acc::ExitDataOp,
                      acc::UpdateOp, acc::DeclareEnterOp, acc::DeclareExitOp,
                      acc::GetDevicePtrOp>();
}

void mlir::populateACCDataClauseOpPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    acc::OpenACCSupport &accSupport, Region &globalSymbolRegion,
    SymbolTable &symbolTable, const acc::ACCRuntimeCallConfig &config) {
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
      DataEntryOpLowering<acc::UpdateDeviceOp>,
      DataEntryOpLowering<acc::PrivateOp>,
      DataEntryOpLowering<acc::FirstprivateOp>,
      DataEntryOpLowering<acc::FirstprivateMapInitialOp>,
      DataEntryOpLowering<acc::DeclareDeviceResidentOp>,
      DataEntryOpLowering<acc::DeclareLinkOp>,
      EraseDataClauseOp<acc::CopyoutOp>, EraseDataClauseOp<acc::DeleteOp>,
      EraseDataClauseOp<acc::DetachOp>, EraseDataClauseOp<acc::UpdateHostOp>,
      EraseDataClauseOp<acc::DataBoundsOp>>(converter);
  patterns.add<GetDevicePtrOpLowering>(converter, accSupport,
                                       globalSymbolRegion, symbolTable, config);
}

void mlir::populateACCDataDirectivePatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    acc::OpenACCSupport &accSupport, Region &globalSymbolRegion,
    SymbolTable &symbolTable, const acc::ACCRuntimeCallConfig &config,
    acc::DeviceType clauseDeviceType) {
  converter.addConversion([](acc::DeclareTokenType type) -> Type {
    return LLVM::LLVMPointerType::get(type.getContext());
  });

  patterns.add<DataOpLowering, EnterDataOpLowering, ExitDataOpLowering,
               UpdateOpLowering>(converter, accSupport, globalSymbolRegion,
                                 symbolTable, config, clauseDeviceType);
  patterns.add<DeclareEnterOpLowering, DeclareExitOpLowering>(
      converter, accSupport, globalSymbolRegion, symbolTable, config);
}
