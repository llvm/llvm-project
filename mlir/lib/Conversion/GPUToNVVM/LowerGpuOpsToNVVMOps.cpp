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

#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "../GPUCommon/GPUOpsLowering.h"
#include "../GPUCommon/IndexIntrinsicsOpLowering.h"
#include "../GPUCommon/OpToFuncCallLowering.h"

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
    auto resultTy = LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                                     {valueTy, predTy});

    Value one = rewriter.create<LLVM::ConstantOp>(loc, int32Type, 1);
    Value minusOne = rewriter.create<LLVM::ConstantOp>(loc, int32Type, -1);
    Value thirtyTwo = rewriter.create<LLVM::ConstantOp>(loc, int32Type, 32);
    Value numLeadInactiveLane = rewriter.create<LLVM::SubOp>(
        loc, int32Type, thirtyTwo, adaptor.getWidth());
    // Bit mask of active lanes: `(-1) >> (32 - activeWidth)`.
    Value activeMask = rewriter.create<LLVM::LShrOp>(loc, int32Type, minusOne,
                                                     numLeadInactiveLane);
    Value maskAndClamp;
    if (op.getMode() == gpu::ShuffleMode::UP) {
      // Clamp lane: `32 - activeWidth`
      maskAndClamp = numLeadInactiveLane;
    } else {
      // Clamp lane: `activeWidth - 1`
      maskAndClamp =
          rewriter.create<LLVM::SubOp>(loc, int32Type, adaptor.getWidth(), one);
    }

    auto returnValueAndIsValidAttr = rewriter.getUnitAttr();
    Value shfl = rewriter.create<NVVM::ShflOp>(
        loc, resultTy, activeMask, adaptor.getValue(), adaptor.getOffset(),
        maskAndClamp, convertShflKind(op.getMode()), returnValueAndIsValidAttr);
    Value shflValue = rewriter.create<LLVM::ExtractValueOp>(loc, shfl, 0);
    Value isActiveSrcLane = rewriter.create<LLVM::ExtractValueOp>(loc, shfl, 1);

    rewriter.replaceOp(op, {shflValue, isActiveSrcLane});
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
    Value newOp = rewriter.create<NVVM::LaneIdOp>(loc, rewriter.getI32Type());
    // Truncate or extend the result depending on the index bitwidth specified
    // by the LLVMTypeConverter options.
    const unsigned indexBitwidth = getTypeConverter()->getIndexTypeBitwidth();
    if (indexBitwidth > 32) {
      newOp = rewriter.create<LLVM::SExtOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp);
    } else if (indexBitwidth < 32) {
      newOp = rewriter.create<LLVM::TruncOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp);
    }
    rewriter.replaceOp(op, {newOp});
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
struct LowerGpuOpsToNVVMOpsPass
    : public impl::ConvertGpuOpsToNVVMOpsBase<LowerGpuOpsToNVVMOpsPass> {
  LowerGpuOpsToNVVMOpsPass() = default;
  LowerGpuOpsToNVVMOpsPass(unsigned indexBitwidth) {
    this->indexBitwidth = indexBitwidth;
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

    // MemRef conversion for GPU to NVVM lowering. The GPU dialect uses memory
    // space 5 for private memory attributions, but NVVM represents private
    // memory allocations as local `alloca`s in the default address space. This
    // converter drops the private memory space to support the use case above.
    LLVMTypeConverter converter(m.getContext(), options);
    converter.addConversion([&](MemRefType type) -> std::optional<Type> {
      if (type.getMemorySpaceAsInt() !=
          gpu::GPUDialect::getPrivateAddressSpace())
        return std::nullopt;
      return converter.convertType(MemRefType::Builder(type).setMemorySpace(
          IntegerAttr::get(IntegerType::get(m.getContext(), 64), 0)));
    });
    // Lowering for MMAMatrixType.
    converter.addConversion([&](gpu::MMAMatrixType type) -> Type {
      return convertMMAToLLVMType(type);
    });
    RewritePatternSet patterns(m.getContext());
    RewritePatternSet llvmPatterns(m.getContext());

    // Apply in-dialect lowering first. In-dialect lowering will replace ops
    // which need to be lowered further, which is not supported by a single
    // conversion pass.
    populateGpuRewritePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(m, std::move(patterns));

    arith::populateArithToLLVMConversionPatterns(converter, llvmPatterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, llvmPatterns);
    populateFuncToLLVMConversionPatterns(converter, llvmPatterns);
    populateMemRefToLLVMConversionPatterns(converter, llvmPatterns);
    populateGpuToNVVMConversionPatterns(converter, llvmPatterns);
    populateGpuWMMAToNVVMConversionPatterns(converter, llvmPatterns);
    LLVMConversionTarget target(getContext());
    configureGpuToNVVMConversionLegality(target);
    if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::configureGpuToNVVMConversionLegality(ConversionTarget &target) {
  target.addIllegalOp<func::FuncOp>();
  target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<::mlir::NVVM::NVVMDialect>();
  target.addIllegalDialect<gpu::GPUDialect>();
  target.addIllegalOp<LLVM::CosOp, LLVM::ExpOp, LLVM::Exp2Op, LLVM::FAbsOp,
                      LLVM::FCeilOp, LLVM::FFloorOp, LLVM::LogOp, LLVM::Log10Op,
                      LLVM::Log2Op, LLVM::PowOp, LLVM::SinOp, LLVM::SqrtOp>();

  // TODO: Remove once we support replacing non-root ops.
  target.addLegalOp<gpu::YieldOp, gpu::GPUModuleOp, gpu::ModuleEndOp>();
}

template <typename OpTy>
static void populateOpPatterns(LLVMTypeConverter &converter,
                               RewritePatternSet &patterns, StringRef f32Func,
                               StringRef f64Func) {
  patterns.add<ScalarizeVectorOpLowering<OpTy>>(converter);
  patterns.add<OpToFuncCallLowering<OpTy>>(converter, f32Func, f64Func);
}

void mlir::populateGpuToNVVMConversionPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns) {
  populateWithGenerated(patterns);
  patterns.add<GPUPrintfOpToVPrintfLowering>(converter);
  patterns
      .add<GPUIndexIntrinsicOpLowering<gpu::ThreadIdOp, NVVM::ThreadIdXOp,
                                       NVVM::ThreadIdYOp, NVVM::ThreadIdZOp>,
           GPUIndexIntrinsicOpLowering<gpu::BlockDimOp, NVVM::BlockDimXOp,
                                       NVVM::BlockDimYOp, NVVM::BlockDimZOp>,
           GPUIndexIntrinsicOpLowering<gpu::BlockIdOp, NVVM::BlockIdXOp,
                                       NVVM::BlockIdYOp, NVVM::BlockIdZOp>,
           GPUIndexIntrinsicOpLowering<gpu::GridDimOp, NVVM::GridDimXOp,
                                       NVVM::GridDimYOp, NVVM::GridDimZOp>,
           GPULaneIdOpToNVVM, GPUShuffleOpLowering, GPUReturnOpLowering>(
          converter);

  // Explicitly drop memory space when lowering private memory
  // attributions since NVVM models it as `alloca`s in the default
  // memory space and does not support `alloca`s with addrspace(5).
  patterns.add<GPUFuncOpLowering>(
      converter, /*allocaAddrSpace=*/0,
      StringAttr::get(&converter.getContext(),
                      NVVM::NVVMDialect::getKernelFuncAttrName()));

  populateOpPatterns<math::AbsFOp>(converter, patterns, "__nv_fabsf",
                                   "__nv_fabs");
  populateOpPatterns<math::AtanOp>(converter, patterns, "__nv_atanf",
                                   "__nv_atan");
  populateOpPatterns<math::Atan2Op>(converter, patterns, "__nv_atan2f",
                                    "__nv_atan2");
  populateOpPatterns<math::CbrtOp>(converter, patterns, "__nv_cbrtf",
                                   "__nv_cbrt");
  populateOpPatterns<math::CeilOp>(converter, patterns, "__nv_ceilf",
                                   "__nv_ceil");
  populateOpPatterns<math::CosOp>(converter, patterns, "__nv_cosf", "__nv_cos");
  populateOpPatterns<math::ExpOp>(converter, patterns, "__nv_expf", "__nv_exp");
  populateOpPatterns<math::Exp2Op>(converter, patterns, "__nv_exp2f",
                                   "__nv_exp2");
  populateOpPatterns<math::ExpM1Op>(converter, patterns, "__nv_expm1f",
                                    "__nv_expm1");
  populateOpPatterns<math::FloorOp>(converter, patterns, "__nv_floorf",
                                    "__nv_floor");
  populateOpPatterns<math::LogOp>(converter, patterns, "__nv_logf", "__nv_log");
  populateOpPatterns<math::Log1pOp>(converter, patterns, "__nv_log1pf",
                                    "__nv_log1p");
  populateOpPatterns<math::Log10Op>(converter, patterns, "__nv_log10f",
                                    "__nv_log10");
  populateOpPatterns<math::Log2Op>(converter, patterns, "__nv_log2f",
                                   "__nv_log2");
  populateOpPatterns<math::PowFOp>(converter, patterns, "__nv_powf",
                                   "__nv_pow");
  populateOpPatterns<math::RsqrtOp>(converter, patterns, "__nv_rsqrtf",
                                    "__nv_rsqrt");
  populateOpPatterns<math::SinOp>(converter, patterns, "__nv_sinf", "__nv_sin");
  populateOpPatterns<math::SqrtOp>(converter, patterns, "__nv_sqrtf",
                                   "__nv_sqrt");
  populateOpPatterns<math::TanhOp>(converter, patterns, "__nv_tanhf",
                                   "__nv_tanh");
  populateOpPatterns<math::TanOp>(converter, patterns, "__nv_tanf", "__nv_tan");
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createLowerGpuOpsToNVVMOpsPass(unsigned indexBitwidth) {
  return std::make_unique<LowerGpuOpsToNVVMOpsPass>(indexBitwidth);
}
