//===- LowerGpuOpsToAGXIROps.cpp - MLIR GPU to AGXIR lowering passes ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate AGXIR IR operations for higher-level
// GPU operations, for Apple GPUs. It follows the GPU to ROCDL lowering:
//
//   gpu.thread_id         agxir.thread.position.in.threadgroup.*
//   gpu.block_id          agxir.threadgroup.position.in.grid.*
//   gpu.block_dim         agxir.threads.per.threadgroup.*
//   gpu.grid_dim          agxir.threadgroups.per.grid.*
//   gpu.global_id         agxir.thread.position.in.grid.*
//   gpu.lane_id           agxir.thread.index.in.simdgroup
//   gpu.subgroup_id       agxir.simdgroup.index.in.threadgroup
//   gpu.subgroup_size     32
//   gpu.barrier           fences and agxir.threadgroup.barrier
//   gpu.shuffle           agxir.simd.shuffle
//   gpu.ballot            agxir.simd.ballot
//   gpu.subgroup_reduce   agxir.simd.reduce
//   gpu.func              agx_kernel functions (agxir.kernel)
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToAGXIR/GPUToAGXIRPass.h"

#include "mlir/Conversion/AGXToAGXIR/AGXToAGXIR.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/AGX/IR/AGXDialect.h"
#include "mlir/Dialect/AGX/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/AGXIRDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "../GPUCommon/GPUOpsLowering.h"
#include "../GPUCommon/IndexIntrinsicsOpLowering.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTGPUOPSTOAGXIROPS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

const char *const mlir::kAGXDataLayout =
    "e-m:e-p:64:64-p1:64:64-p3:32:32-p4:64:64-p5:32:32-i1:8-i8:8-i16:16-"
    "i32:32-i64:64-f16:16-f32:32-f64:64-v16:16-v32:32-v64:64-v96:32-"
    "v128:128-n16:32:64-S32-A5-G1";

/// Extend or truncate an i32 to the index type.
static Value toIndexBitwidth(ConversionPatternRewriter &rewriter, Location loc,
                             Value value, unsigned indexBitwidth) {
  Type indexType = rewriter.getIntegerType(indexBitwidth);
  if (indexBitwidth > 32)
    return LLVM::SExtOp::create(rewriter, loc, indexType, value);
  if (indexBitwidth < 32)
    return LLVM::TruncOp::create(rewriter, loc, indexType, value);
  return value;
}

namespace {

template <typename SourceOp, typename AGXIROp>
struct GPUScalarIndexOpLowering : ConvertOpToLLVMPattern<SourceOp> {
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value value = AGXIROp::create(rewriter, loc, rewriter.getI32Type());
    rewriter.replaceOp(
        op, toIndexBitwidth(rewriter, loc, value,
                            this->getTypeConverter()->getIndexTypeBitwidth()));
    return success();
  }
};

struct GPUSubgroupSizeOpLowering
    : ConvertOpToLLVMPattern<gpu::SubgroupSizeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupSizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type indexType = rewriter.getIntegerType(
        getTypeConverter()->getIndexTypeBitwidth());
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
        op, indexType, AGXIR::AGXIRDialect::kSIMDGroupSize);
    return success();
  }
};

/// gpu.barrier: fence threadgroup memory around the execution barrier.
struct GPUBarrierOpLowering : ConvertOpToLLVMPattern<gpu::BarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::release,
                          "workgroup");
    AGXIR::ThreadgroupBarrierOp::create(rewriter, loc);
    rewriter.replaceOpWithNewOp<LLVM::FenceOp>(
        op, LLVM::AtomicOrdering::acquire, "workgroup");
    return success();
  }
};

/// gpu.shuffle: compute the source lane like the ROCDL lowering, falling back
/// to the current lane outside of the width, and shuffle each 32-bit part.
struct GPUShuffleOpLowering : ConvertOpToLLVMPattern<gpu::ShuffleOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::ShuffleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();
    Value lane = AGXIR::ThreadIndexInSIMDGroupOp::create(rewriter, loc, i32);
    Value width = adaptor.getWidth();
    Value zero = LLVM::ConstantOp::create(rewriter, loc, i32, 0);
    Value negWidth = LLVM::SubOp::create(rewriter, loc, i32, zero, width);
    Value add = LLVM::AddOp::create(rewriter, loc, i32, lane, width);
    // The end of the lane's segment of `width` lanes.
    Value segmentEnd = LLVM::AndOp::create(rewriter, loc, i32, add, negWidth);
    Value srcLane;
    switch (op.getMode()) {
    case gpu::ShuffleMode::UP:
      srcLane = LLVM::SubOp::create(rewriter, loc, i32, lane,
                                    adaptor.getOffset());
      break;
    case gpu::ShuffleMode::DOWN:
      srcLane = LLVM::AddOp::create(rewriter, loc, i32, lane,
                                    adaptor.getOffset());
      break;
    case gpu::ShuffleMode::XOR:
      srcLane = LLVM::XOrOp::create(rewriter, loc, i32, lane,
                                    adaptor.getOffset());
      break;
    case gpu::ShuffleMode::IDX:
      srcLane = adaptor.getOffset();
      break;
    }
    Value valid = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::slt,
                                       srcLane, segmentEnd);
    Value selected =
        LLVM::SelectOp::create(rewriter, loc, valid, srcLane, lane);

    SmallVector<Value> parts;
    if (failed(LLVM::decomposeValue(rewriter, loc, adaptor.getValue(), i32,
                                    parts)))
      return rewriter.notifyMatchFailure(op, "cannot split value into i32");
    SmallVector<Value> shuffled;
    for (Value part : parts)
      shuffled.push_back(
          AGXIR::SIMDShuffleOp::create(rewriter, loc, i32, part, selected));
    Value result = LLVM::composeValue(rewriter, loc, shuffled,
                                      adaptor.getValue().getType());
    rewriter.replaceOp(op, {result, valid});
    return success();
  }
};

struct GPUBallotOpLowering : ConvertOpToLLVMPattern<gpu::BallotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::BallotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto intType = dyn_cast<IntegerType>(op.getType());
    if (!intType || intType.getWidth() < 32)
      return rewriter.notifyMatchFailure(op, "result must be at least 32 bits");
    Value mask = AGXIR::SIMDBallotOp::create(rewriter, loc,
                                             rewriter.getI32Type(),
                                             adaptor.getPredicate());
    if (intType.getWidth() > 32)
      mask = LLVM::ZExtOp::create(rewriter, loc, intType, mask);
    rewriter.replaceOp(op, mask);
    return success();
  }
};

/// gpu.subgroup_reduce of 32-bit values over the whole SIMD-group.
struct GPUSubgroupReduceOpLowering
    : ConvertOpToLLVMPattern<gpu::SubgroupReduceOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getClusterSize())
      return rewriter.notifyMatchFailure(op, "clustered reductions");
    Type type = adaptor.getValue().getType();
    bool isFloat = type.isF32();
    if (!isFloat && !type.isInteger(32))
      return rewriter.notifyMatchFailure(op, "only i32 and f32 are supported");
    std::optional<AGXIR::ReduceKind> kind;
    switch (op.getOp()) {
    case gpu::AllReduceOperation::ADD:
      kind = isFloat ? AGXIR::ReduceKind::fadd : AGXIR::ReduceKind::iadd;
      break;
    case gpu::AllReduceOperation::MUL:
      if (isFloat)
        kind = AGXIR::ReduceKind::fmul;
      break;
    case gpu::AllReduceOperation::MINSI:
      kind = AGXIR::ReduceKind::smin;
      break;
    case gpu::AllReduceOperation::MAXSI:
      kind = AGXIR::ReduceKind::smax;
      break;
    case gpu::AllReduceOperation::MINUI:
      kind = AGXIR::ReduceKind::umin;
      break;
    case gpu::AllReduceOperation::MAXUI:
      kind = AGXIR::ReduceKind::umax;
      break;
    case gpu::AllReduceOperation::MINNUMF:
      kind = AGXIR::ReduceKind::fmin;
      break;
    case gpu::AllReduceOperation::MAXNUMF:
      kind = AGXIR::ReduceKind::fmax;
      break;
    case gpu::AllReduceOperation::AND:
      kind = AGXIR::ReduceKind::and_;
      break;
    case gpu::AllReduceOperation::OR:
      kind = AGXIR::ReduceKind::or_;
      break;
    case gpu::AllReduceOperation::XOR:
      kind = AGXIR::ReduceKind::xor_;
      break;
    default:
      break;
    }
    if (!kind)
      return rewriter.notifyMatchFailure(op, "unsupported reduction");
    rewriter.replaceOpWithNewOp<AGXIR::SIMDReduceOp>(op, type,
                                                     adaptor.getValue(), *kind);
    return success();
  }
};

struct LowerGpuOpsToAGXIROpsPass final
    : public impl::ConvertGpuOpsToAGXIROpsBase<LowerGpuOpsToAGXIROpsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    Base::getDependentDialects(registry);
    registerConvertToLLVMDependentDialectLoading(registry);
  }

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();
    MLIRContext *ctx = m.getContext();

    auto llvmDataLayout = m->getAttrOfType<StringAttr>(
        LLVM::LLVMDialect::getDataLayoutAttrName());
    if (!llvmDataLayout) {
      llvmDataLayout = StringAttr::get(ctx, kAGXDataLayout);
      m->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(), llvmDataLayout);
    }

    LowerToLLVMOptions options(
        ctx, DataLayout(cast<DataLayoutOpInterface>(m.getOperation())));
    options.dataLayout = llvm::DataLayout(llvmDataLayout.getValue());
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);
    options.useBarePtrCallConv = useBarePtrCallConv;

    // In-dialect lowering of gpu.all_reduce and friends first, and the
    // portable forms of the agx operations the module's chip does not run
    // natively.
    {
      // The backend's default processor when the module names no chip.
      FailureOr<agx::ChipInfo> chip = agx::ChipInfo::get("agx-g14g");
      if (ArrayAttr targets = m.getTargetsAttr())
        for (Attribute t : targets)
          if (auto target = dyn_cast<AGXIR::AGXIRTargetAttr>(t))
            chip = agx::ChipInfo::get(target.getChip());
      if (failed(chip)) {
        m.emitError() << "unknown AGX chip in the module's #agxir.target";
        return signalPassFailure();
      }
      RewritePatternSet patterns(ctx);
      populateGpuRewritePatterns(patterns);
      agx::populateAGXExpandOpsPatterns(patterns, *chip);
      // The vector to LLVM lowering has no strided slices; the portable
      // forms of the matrix ops make them.
      vector::populateVectorInsertExtractStridedSliceTransforms(patterns);
      (void)applyPatternsGreedily(m, std::move(patterns));
    }

    LLVMTypeConverter converter(ctx, options);
    populateGpuMemorySpaceAttributeConversions(
        converter, [](gpu::AddressSpace space) -> unsigned {
          switch (space) {
          case gpu::AddressSpace::Global:
            return AGXIR::AGXIRDialect::kDeviceMemoryAddressSpace;
          case gpu::AddressSpace::Workgroup:
            return AGXIR::AGXIRDialect::kThreadgroupMemoryAddressSpace;
          case gpu::AddressSpace::Private:
            return AGXIR::AGXIRDialect::kPrivateMemoryAddressSpace;
          case gpu::AddressSpace::Constant:
            return AGXIR::AGXIRDialect::kConstantMemoryAddressSpace;
          }
          llvm_unreachable("unknown address space enum value");
        });

    RewritePatternSet llvmPatterns(ctx);
    LLVMConversionTarget target(getContext());
    for (Dialect *dialect : ctx->getLoadedDialects())
      if (auto *iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect))
        iface->populateConvertToLLVMConversionPatterns(target, converter,
                                                       llvmPatterns);
    // Structured control flow (loops in kernels) becomes branches.
    populateSCFToControlFlowConversionPatterns(llvmPatterns);
    target.addIllegalDialect<scf::SCFDialect>();
    populateAGXToAGXIRConversionPatterns(converter, llvmPatterns);
    populateGpuToAGXIRConversionPatterns(converter, llvmPatterns);
    configureGpuToAGXIRConversionLegality(target);
    if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::configureGpuToAGXIRConversionLegality(ConversionTarget &target) {
  target.addIllegalOp<func::FuncOp>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalDialect<AGXIR::AGXIRDialect>();
  target.addIllegalDialect<gpu::GPUDialect>();
  target.addIllegalDialect<agx::AGXDialect>();
  // TODO: Remove once we support replacing non-root ops.
  target.addLegalOp<gpu::YieldOp, gpu::GPUModuleOp>();
}

void mlir::populateGpuToAGXIRConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  using gpu::index_lowering::IndexKind;
  using gpu::index_lowering::IntrType;
  using gpu::index_lowering::OpLowering;
  auto *agxirDialect =
      converter.getContext().getLoadedDialect<AGXIR::AGXIRDialect>();

  patterns.add<OpLowering<gpu::ThreadIdOp, AGXIR::ThreadPositionInThreadgroupXOp,
                          AGXIR::ThreadPositionInThreadgroupYOp,
                          AGXIR::ThreadPositionInThreadgroupZOp>>(
      converter, IndexKind::Block, IntrType::Id);
  patterns.add<OpLowering<gpu::BlockIdOp, AGXIR::ThreadgroupPositionInGridXOp,
                          AGXIR::ThreadgroupPositionInGridYOp,
                          AGXIR::ThreadgroupPositionInGridZOp>>(
      converter, IndexKind::Grid, IntrType::Id);
  patterns.add<OpLowering<gpu::BlockDimOp, AGXIR::ThreadsPerThreadgroupXOp,
                          AGXIR::ThreadsPerThreadgroupYOp,
                          AGXIR::ThreadsPerThreadgroupZOp>>(
      converter, IndexKind::Block, IntrType::Dim);
  patterns.add<OpLowering<gpu::GridDimOp, AGXIR::ThreadgroupsPerGridXOp,
                          AGXIR::ThreadgroupsPerGridYOp,
                          AGXIR::ThreadgroupsPerGridZOp>>(
      converter, IndexKind::Grid, IntrType::Dim);
  patterns.add<OpLowering<gpu::GlobalIdOp, AGXIR::ThreadPositionInGridXOp,
                          AGXIR::ThreadPositionInGridYOp,
                          AGXIR::ThreadPositionInGridZOp>>(converter);
  patterns.add<
      GPUScalarIndexOpLowering<gpu::LaneIdOp, AGXIR::ThreadIndexInSIMDGroupOp>,
      GPUScalarIndexOpLowering<gpu::SubgroupIdOp,
                               AGXIR::SIMDGroupIndexInThreadgroupOp>,
      GPUSubgroupSizeOpLowering, GPUBarrierOpLowering, GPUShuffleOpLowering,
      GPUBallotOpLowering, GPUSubgroupReduceOpLowering>(converter);
  patterns.add<GPUReturnOpLowering>(converter);
  patterns.add<GPUFuncOpLowering>(
      converter,
      GPUFuncOpLoweringOptions{
          /*allocaAddrSpace=*/
          AGXIR::AGXIRDialect::kPrivateMemoryAddressSpace,
          /*workgroupAddrSpace=*/
          AGXIR::AGXIRDialect::kThreadgroupMemoryAddressSpace,
          agxirDialect->getKernelAttrHelper().getName(),
          /*kernelBlockSizeAttributeName=*/{},
          /*kernelClusterSizeAttributeName=*/{},
          /*kernelCallingConvention=*/LLVM::CConv::AGX_KERNEL});
}
