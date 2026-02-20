//===- LowerGpuOpsToNVVMOps.cpp - MLIR GPU to NVVM lowering passes --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate NVVMIR operations for higher-level
// GPU operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToNVVM/MathToNVVM.h"
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "../GPUCommon/GPUOpsLowering.h"
#include "../GPUCommon/IndexIntrinsicsOpLowering.h"
#include "../GPUCommon/OpToFuncCallLowering.h"
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CONVERTGPUOPSTONVVMOPS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Convert gpu dialect shfl mode enum to the equivalent nvvm one.
static NVVM::ShflKind convertShflKind(gpu::ShuffleMode mode) {
  switch (mode) {
  case gpu::ShuffleMode::XOR:
    return NVVM::ShflKind::bfly;
  case gpu::ShuffleMode::UP:
    return NVVM::ShflKind::up;
  case gpu::ShuffleMode::DOWN:
    return NVVM::ShflKind::down;
  case gpu::ShuffleMode::IDX:
    return NVVM::ShflKind::idx;
  }
  llvm_unreachable("unknown shuffle mode");
}

static std::optional<NVVM::ReductionKind>
convertToNVVMReductionKind(gpu::AllReduceOperation mode) {
  switch (mode) {
  case gpu::AllReduceOperation::ADD:
    return NVVM::ReductionKind::ADD;
  case gpu::AllReduceOperation::MUL:
    return std::nullopt;
  case gpu::AllReduceOperation::MINSI:
    return NVVM::ReductionKind::MIN;
  case gpu::AllReduceOperation::MINUI:
    return std::nullopt;
  case gpu::AllReduceOperation::MINNUMF:
    return NVVM::ReductionKind::MIN;
  case gpu::AllReduceOperation::MAXSI:
    return NVVM::ReductionKind::MAX;
  case gpu::AllReduceOperation::MAXUI:
    return std::nullopt;
  case gpu::AllReduceOperation::MAXNUMF:
    return NVVM::ReductionKind::MAX;
  case gpu::AllReduceOperation::AND:
    return NVVM::ReductionKind::AND;
  case gpu::AllReduceOperation::OR:
    return NVVM::ReductionKind::OR;
  case gpu::AllReduceOperation::XOR:
    return NVVM::ReductionKind::XOR;
  case gpu::AllReduceOperation::MINIMUMF:
  case gpu::AllReduceOperation::MAXIMUMF:
    return std::nullopt;
  }
  return std::nullopt;
}

/// This pass lowers gpu.subgroup_reduce op into to the nvvm.redux op. The op
/// must be run by the entire subgroup, otherwise it is undefined behaviour.
struct GPUSubgroupReduceOpLowering
    : public ConvertOpToLLVMPattern<gpu::SubgroupReduceOp> {
  using ConvertOpToLLVMPattern<gpu::SubgroupReduceOp>::ConvertOpToLLVMPattern;
  LogicalResult

  matchAndRewrite(gpu::SubgroupReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getClusterSize())
      return rewriter.notifyMatchFailure(
          op, "lowering for clustered reduce not implemented");

    if (!op.getUniform())
      return rewriter.notifyMatchFailure(
          op, "cannot be lowered to redux as the op must be run "
              "uniformly (entire subgroup).");
    if (!op.getValue().getType().isInteger(32))
      return rewriter.notifyMatchFailure(op, "unsupported data type");

    std::optional<NVVM::ReductionKind> mode =
        convertToNVVMReductionKind(op.getOp());
    if (!mode.has_value())
      return rewriter.notifyMatchFailure(
          op, "unsupported reduction mode for redux");

    Location loc = op->getLoc();
    auto int32Type = IntegerType::get(rewriter.getContext(), 32);
    Value offset = LLVM::ConstantOp::create(rewriter, loc, int32Type, -1);

    auto reduxOp = NVVM::ReduxOp::create(rewriter, loc, int32Type,
                                         op.getValue(), mode.value(), offset);

    rewriter.replaceOp(op, reduxOp->getResult(0));
    return success();
  }
};

struct GPUShuffleOpLowering : public ConvertOpToLLVMPattern<gpu::ShuffleOp> {
  using ConvertOpToLLVMPattern<gpu::ShuffleOp>::ConvertOpToLLVMPattern;

  /// Lowers a shuffle to the corresponding NVVM op.
  ///
  /// Convert the `width` argument into an activeMask (a bitmask which specifies
  /// which threads participate in the shuffle) and a maskAndClamp (specifying
  /// the highest lane which participates in the shuffle).
  ///
  ///     %one = llvm.constant(1 : i32) : i32
  ///     %minus_one = llvm.constant(-1 : i32) : i32
  ///     %thirty_two = llvm.constant(32 : i32) : i32
  ///     %num_lanes = llvm.sub %thirty_two, %width : i32
  ///     %active_mask = llvm.lshr %minus_one, %num_lanes : i32
  ///     %mask_and_clamp = llvm.sub %width, %one : i32
  ///     %shfl = nvvm.shfl.sync.bfly %active_mask, %value, %offset,
  ///         %mask_and_clamp : !llvm<"{ float, i1 }">
  ///     %shfl_value = llvm.extractvalue %shfl[0] :
  ///         !llvm<"{ float, i1 }">
  ///     %shfl_pred = llvm.extractvalue %shfl[1] :
  ///         !llvm<"{ float, i1 }">
  LogicalResult
  matchAndRewrite(gpu::ShuffleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    auto valueTy = adaptor.getValue().getType();
    auto int32Type = IntegerType::get(rewriter.getContext(), 32);
    auto predTy = IntegerType::get(rewriter.getContext(), 1);

    Value one = LLVM::ConstantOp::create(rewriter, loc, int32Type, 1);
    Value minusOne = LLVM::ConstantOp::create(rewriter, loc, int32Type, -1);
    Value thirtyTwo = LLVM::ConstantOp::create(rewriter, loc, int32Type, 32);
    Value numLeadInactiveLane = LLVM::SubOp::create(
        rewriter, loc, int32Type, thirtyTwo, adaptor.getWidth());
    // Bit mask of active lanes: `(-1) >> (32 - activeWidth)`.
    Value activeMask = LLVM::LShrOp::create(rewriter, loc, int32Type, minusOne,
                                            numLeadInactiveLane);
    Value maskAndClamp;
    if (op.getMode() == gpu::ShuffleMode::UP) {
      // Clamp lane: `32 - activeWidth`
      maskAndClamp = numLeadInactiveLane;
    } else {
      // Clamp lane: `activeWidth - 1`
      maskAndClamp = LLVM::SubOp::create(rewriter, loc, int32Type,
                                         adaptor.getWidth(), one);
    }

    bool predIsUsed = !op->getResult(1).use_empty();
    UnitAttr returnValueAndIsValidAttr = nullptr;
    Type resultTy = valueTy;
    if (predIsUsed) {
      returnValueAndIsValidAttr = rewriter.getUnitAttr();
      resultTy = LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                                  {valueTy, predTy});
    }
    Value shfl = NVVM::ShflOp::create(
        rewriter, loc, resultTy, activeMask, adaptor.getValue(),
        adaptor.getOffset(), maskAndClamp, convertShflKind(op.getMode()),
        returnValueAndIsValidAttr);
    if (predIsUsed) {
      Value shflValue = LLVM::ExtractValueOp::create(rewriter, loc, shfl, 0);
      Value isActiveSrcLane =
          LLVM::ExtractValueOp::create(rewriter, loc, shfl, 1);
      rewriter.replaceOp(op, {shflValue, isActiveSrcLane});
    } else {
      rewriter.replaceOp(op, {shfl, nullptr});
    }
    return success();
  }
};

struct GPULaneIdOpToNVVM : ConvertOpToLLVMPattern<gpu::LaneIdOp> {
  using ConvertOpToLLVMPattern<gpu::LaneIdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::LaneIdOp op, gpu::LaneIdOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *context = rewriter.getContext();
    LLVM::ConstantRangeAttr bounds = nullptr;
    if (std::optional<APInt> upperBound = op.getUpperBound())
      bounds = rewriter.getAttr<LLVM::ConstantRangeAttr>(
          /*bitWidth=*/32, /*lower=*/0, upperBound->getZExtValue());
    else
      bounds = rewriter.getAttr<LLVM::ConstantRangeAttr>(
          /*bitWidth=*/32, /*lower=*/0, /*upper=*/kWarpSize);
    Value newOp =
        NVVM::LaneIdOp::create(rewriter, loc, rewriter.getI32Type(), bounds);
    // Truncate or extend the result depending on the index bitwidth specified
    // by the LLVMTypeConverter options.
    const unsigned indexBitwidth = getTypeConverter()->getIndexTypeBitwidth();
    if (indexBitwidth > 32) {
      newOp = LLVM::SExtOp::create(
          rewriter, loc, IntegerType::get(context, indexBitwidth), newOp);
    } else if (indexBitwidth < 32) {
      newOp = LLVM::TruncOp::create(
          rewriter, loc, IntegerType::get(context, indexBitwidth), newOp);
    }
    rewriter.replaceOp(op, {newOp});
    return success();
  }
};

/// Lowering of cf.assert into a conditional __assertfail.
struct AssertOpToAssertfailLowering
    : public ConvertOpToLLVMPattern<cf::AssertOp> {
  using ConvertOpToLLVMPattern<cf::AssertOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cf::AssertOp assertOp, cf::AssertOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = rewriter.getContext();
    Location loc = assertOp.getLoc();
    Type i8Type = typeConverter->convertType(rewriter.getIntegerType(8));
    Type i32Type = typeConverter->convertType(rewriter.getIntegerType(32));
    Type i64Type = typeConverter->convertType(rewriter.getIntegerType(64));
    Type ptrType = LLVM::LLVMPointerType::get(ctx);
    Type voidType = LLVM::LLVMVoidType::get(ctx);

    // Find or create __assertfail function declaration.
    auto moduleOp = assertOp->getParentOfType<gpu::GPUModuleOp>();
    auto assertfailType = LLVM::LLVMFunctionType::get(
        voidType, {ptrType, ptrType, i32Type, ptrType, i64Type});
    LLVM::LLVMFuncOp assertfailDecl = getOrDefineFunction(
        moduleOp, loc, rewriter, "__assertfail", assertfailType);
    assertfailDecl.setPassthroughAttr(
        ArrayAttr::get(ctx, StringAttr::get(ctx, "noreturn")));

    // Split blocks and insert conditional branch.
    // ^before:
    //   ...
    //   cf.cond_br %condition, ^after, ^assert
    // ^assert:
    //   cf.assert
    //   cf.br ^after
    // ^after:
    //   ...
    Block *beforeBlock = assertOp->getBlock();
    Block *assertBlock =
        rewriter.splitBlock(beforeBlock, assertOp->getIterator());
    Block *afterBlock =
        rewriter.splitBlock(assertBlock, ++assertOp->getIterator());
    rewriter.setInsertionPointToEnd(beforeBlock);
    cf::CondBranchOp::create(rewriter, loc, adaptor.getArg(), afterBlock,
                             assertBlock);
    rewriter.setInsertionPointToEnd(assertBlock);
    cf::BranchOp::create(rewriter, loc, afterBlock);

    // Continue cf.assert lowering.
    rewriter.setInsertionPoint(assertOp);

    // Populate file name, file number and function name from the location of
    // the AssertOp.
    StringRef fileName = "(unknown)";
    StringRef funcName = "(unknown)";
    int32_t fileLine = 0;
    while (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc))
      loc = callSiteLoc.getCallee();
    if (auto fileLineColLoc = dyn_cast<FileLineColRange>(loc)) {
      fileName = fileLineColLoc.getFilename().strref();
      fileLine = fileLineColLoc.getStartLine();
    } else if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
      funcName = nameLoc.getName().strref();
      if (auto fileLineColLoc =
              dyn_cast<FileLineColRange>(nameLoc.getChildLoc())) {
        fileName = fileLineColLoc.getFilename().strref();
        fileLine = fileLineColLoc.getStartLine();
      }
    }

    // Create constants.
    auto getGlobal = [&](LLVM::GlobalOp global) {
      // Get a pointer to the format string's first element.
      Value globalPtr = LLVM::AddressOfOp::create(
          rewriter, loc, LLVM::LLVMPointerType::get(ctx, global.getAddrSpace()),
          global.getSymNameAttr());
      Value start =
          LLVM::GEPOp::create(rewriter, loc, ptrType, global.getGlobalType(),
                              globalPtr, ArrayRef<LLVM::GEPArg>{0, 0});
      return start;
    };
    Value assertMessage = getGlobal(getOrCreateStringConstant(
        rewriter, loc, moduleOp, i8Type, "assert_message_", assertOp.getMsg()));
    Value assertFile = getGlobal(getOrCreateStringConstant(
        rewriter, loc, moduleOp, i8Type, "assert_file_", fileName));
    Value assertFunc = getGlobal(getOrCreateStringConstant(
        rewriter, loc, moduleOp, i8Type, "assert_func_", funcName));
    Value assertLine =
        LLVM::ConstantOp::create(rewriter, loc, i32Type, fileLine);
    Value c1 = LLVM::ConstantOp::create(rewriter, loc, i64Type, 1);

    // Insert function call to __assertfail.
    SmallVector<Value> arguments{assertMessage, assertFile, assertLine,
                                 assertFunc, c1};
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(assertOp, assertfailDecl,
                                              arguments);
    return success();
  }
};

/// Import the GPU Ops to NVVM Patterns.
#include "GPUToNVVM.cpp.inc"

/// A pass that replaces all occurrences of GPU device operations with their
/// corresponding NVVM equivalent.
///
/// This pass only handles device code and is not meant to be run on GPU host
/// code.
struct LowerGpuOpsToNVVMOpsPass final
    : public impl::ConvertGpuOpsToNVVMOpsBase<LowerGpuOpsToNVVMOpsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    Base::getDependentDialects(registry);
    registerConvertToLLVMDependentDialectLoading(registry);
  }

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();

    // Request C wrapper emission.
    for (auto func : m.getOps<func::FuncOp>()) {
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(&getContext()));
    }

    // Customize the bitwidth used for the device side index computations.
    LowerToLLVMOptions options(
        m.getContext(),
        DataLayout(cast<DataLayoutOpInterface>(m.getOperation())));
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);
    options.useBarePtrCallConv = useBarePtrCallConv;

    // Apply in-dialect lowering. In-dialect lowering will replace
    // ops which need to be lowered further, which is not supported by a
    // single conversion pass.
    {
      RewritePatternSet patterns(m.getContext());
      populateGpuRewritePatterns(patterns);
      // Transform N-D vector.from_elements to 1-D vector.from_elements before
      // conversion.
      vector::populateVectorFromElementsUnrollPatterns(patterns);
      if (failed(applyPatternsGreedily(m, std::move(patterns))))
        return signalPassFailure();
    }

    LLVMTypeConverter converter(m.getContext(), options);
    configureGpuToNVVMTypeConverter(converter);
    RewritePatternSet llvmPatterns(m.getContext());
    LLVMConversionTarget target(getContext());

    // Set higher benefit, so patterns will run before generic LLVM lowering.
    populateGpuToNVVMConversionPatterns(converter, llvmPatterns,
                                        /*benefit=*/10);

    llvm::SmallDenseSet<StringRef> allowedDialectsSet(allowedDialects.begin(),
                                                      allowedDialects.end());
    for (Dialect *dialect : getContext().getLoadedDialects()) {
      // Skip math patterns as nvvm needs custom math lowering.
      if (isa<math::MathDialect>(dialect))
        continue;

      bool allowed = allowedDialectsSet.contains(dialect->getNamespace());
      // Empty `allowedDialectsSet` means all dialects are allowed.
      if (!allowedDialectsSet.empty() && !allowed)
        continue;

      auto *iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect);
      if (!iface) {
        // Error out if dialect was explicily specified but doesn't implement
        // conversion interface.
        if (allowed) {
          m.emitError()
              << "dialect does not implement ConvertToLLVMPatternInterface: "
              << dialect->getNamespace();
          return signalPassFailure();
        }
        continue;
      }

      iface->populateConvertToLLVMConversionPatterns(target, converter,
                                                     llvmPatterns);
    }

    populateGpuWMMAToNVVMConversionPatterns(converter, llvmPatterns);
    if (this->hasRedux)
      populateGpuSubgroupReduceOpLoweringPattern(converter, llvmPatterns);
    configureGpuToNVVMConversionLegality(target);
    ConversionConfig config;
    config.allowPatternRollback = allowPatternRollback;
    if (failed(
            applyPartialConversion(m, target, std::move(llvmPatterns), config)))
      signalPassFailure();
  }
};

} // namespace

void mlir::configureGpuToNVVMConversionLegality(ConversionTarget &target) {
  target.addIllegalOp<func::FuncOp>();
  target.addIllegalOp<cf::AssertOp>();
  target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<::mlir::NVVM::NVVMDialect>();
  target.addIllegalDialect<gpu::GPUDialect>();
  target.addIllegalOp<LLVM::CopySignOp, LLVM::CosOp, LLVM::ExpOp, LLVM::Exp2Op,
                      LLVM::FAbsOp, LLVM::FCeilOp, LLVM::FFloorOp, LLVM::FRemOp,
                      LLVM::LogOp, LLVM::Log10Op, LLVM::Log2Op, LLVM::PowOp,
                      LLVM::RoundEvenOp, LLVM::RoundOp, LLVM::SinOp,
                      LLVM::SincosOp, LLVM::SqrtOp>();

  // TODO: Remove once we support replacing non-root ops.
  target.addLegalOp<gpu::YieldOp, gpu::GPUModuleOp>();
}

void mlir::configureGpuToNVVMTypeConverter(LLVMTypeConverter &converter) {
  nvgpu::populateCommonGPUTypeAndAttributeConversions(converter);

  // Lowering for MMAMatrixType.
  converter.addConversion([&](gpu::MMAMatrixType type) -> Type {
    return convertMMAToLLVMType(type);
  });
}

void mlir::populateGpuSubgroupReduceOpLoweringPattern(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GPUSubgroupReduceOpLowering>(converter, benefit);
}

void mlir::populateGpuToNVVMConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  using gpu::index_lowering::IndexKind;
  using gpu::index_lowering::IntrType;

  // TODO: Pass benefit to generated patterns.
  populateWithGenerated(patterns);

  patterns.add<GPUPrintfOpToVPrintfLowering, AssertOpToAssertfailLowering>(
      converter, benefit);
  patterns.add<
      gpu::index_lowering::OpLowering<gpu::ThreadIdOp, NVVM::ThreadIdXOp,
                                      NVVM::ThreadIdYOp, NVVM::ThreadIdZOp>>(
      converter, IndexKind::Block, IntrType::Id, benefit);
  patterns.add<
      gpu::index_lowering::OpLowering<gpu::BlockDimOp, NVVM::BlockDimXOp,
                                      NVVM::BlockDimYOp, NVVM::BlockDimZOp>>(
      converter, IndexKind::Block, IntrType::Dim, benefit);
  patterns.add<
      gpu::index_lowering::OpLowering<gpu::ClusterIdOp, NVVM::ClusterIdXOp,
                                      NVVM::ClusterIdYOp, NVVM::ClusterIdZOp>>(
      converter, IndexKind::Other, IntrType::Id, benefit);
  patterns.add<gpu::index_lowering::OpLowering<
      gpu::ClusterDimOp, NVVM::ClusterDimXOp, NVVM::ClusterDimYOp,
      NVVM::ClusterDimZOp>>(converter, IndexKind::Other, IntrType::Dim,
                            benefit);
  patterns.add<gpu::index_lowering::OpLowering<
      gpu::ClusterBlockIdOp, NVVM::BlockInClusterIdXOp,
      NVVM::BlockInClusterIdYOp, NVVM::BlockInClusterIdZOp>>(
      converter, IndexKind::Cluster, IntrType::Id, benefit);
  patterns.add<gpu::index_lowering::OpLowering<
      gpu::ClusterDimBlocksOp, NVVM::ClusterDimBlocksXOp,
      NVVM::ClusterDimBlocksYOp, NVVM::ClusterDimBlocksZOp>>(
      converter, IndexKind::Cluster, IntrType::Dim, benefit);
  patterns.add<gpu::index_lowering::OpLowering<
      gpu::BlockIdOp, NVVM::BlockIdXOp, NVVM::BlockIdYOp, NVVM::BlockIdZOp>>(
      converter, IndexKind::Grid, IntrType::Id, benefit);
  patterns.add<gpu::index_lowering::OpLowering<
      gpu::GridDimOp, NVVM::GridDimXOp, NVVM::GridDimYOp, NVVM::GridDimZOp>>(
      converter, IndexKind::Grid, IntrType::Dim, benefit);
  patterns.add<GPULaneIdOpToNVVM, GPUShuffleOpLowering, GPUReturnOpLowering>(
      converter, benefit);

  patterns.add<GPUDynamicSharedMemoryOpLowering>(
      converter, NVVM::kSharedMemoryAlignmentBit, benefit);

  // Explicitly drop memory space when lowering private memory
  // attributions since NVVM models it as `alloca`s in the default
  // memory space and does not support `alloca`s with addrspace(5).
  patterns.add<GPUFuncOpLowering>(
      converter,
      GPUFuncOpLoweringOptions{
          /*allocaAddrSpace=*/0,
          /*workgroupAddrSpace=*/
          static_cast<unsigned>(NVVM::NVVMMemorySpace::Shared),
          StringAttr::get(&converter.getContext(),
                          NVVM::NVVMDialect::getKernelFuncAttrName()),
          StringAttr::get(&converter.getContext(),
                          NVVM::NVVMDialect::getMaxntidAttrName()),
          StringAttr::get(&converter.getContext(),
                          NVVM::NVVMDialect::getClusterDimAttrName())},
      benefit);

  populateLibDeviceConversionPatterns(converter, patterns, benefit);
}

//===----------------------------------------------------------------------===//
// NVVMTargetAttr convert to LLVM attr interface
//===----------------------------------------------------------------------===//

namespace {
struct NVVMTargetConvertToLLVMAttrInterface
    : public ConvertToLLVMAttrInterface::ExternalModel<
          NVVMTargetConvertToLLVMAttrInterface, NVVM::NVVMTargetAttr> {
  /// Configure GPU to NVVM.
  void populateConvertToLLVMConversionPatterns(
      Attribute attr, ConversionTarget &target,
      LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) const;
};
} // namespace

void NVVMTargetConvertToLLVMAttrInterface::
    populateConvertToLLVMConversionPatterns(Attribute attr,
                                            ConversionTarget &target,
                                            LLVMTypeConverter &typeConverter,
                                            RewritePatternSet &patterns) const {
  configureGpuToNVVMConversionLegality(target);
  configureGpuToNVVMTypeConverter(typeConverter);
  populateGpuToNVVMConversionPatterns(typeConverter, patterns);
}

void mlir::NVVM::registerConvertGpuToNVVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, NVVMDialect *dialect) {
    NVVMTargetAttr::attachInterface<NVVMTargetConvertToLLVMAttrInterface>(*ctx);
  });
}
