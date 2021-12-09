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

#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/FormatVariadic.h"

#include "../GPUCommon/GPUOpsLowering.h"
#include "../GPUCommon/IndexIntrinsicsOpLowering.h"
#include "../GPUCommon/OpToFuncCallLowering.h"
#include "../PassDetail.h"

using namespace mlir;

namespace {

/// Convert gpu dialect shfl mode enum to the equivalent nvvm one.
static NVVM::ShflKind convertShflKind(gpu::ShuffleModeAttr mode) {
  switch (mode) {
  case gpu::ShuffleModeAttr::XOR:
    return NVVM::ShflKind::bfly;
  case gpu::ShuffleModeAttr::UP:
    return NVVM::ShflKind::up;
  case gpu::ShuffleModeAttr::DOWN:
    return NVVM::ShflKind::down;
  case gpu::ShuffleModeAttr::IDX:
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
  ///     %shl = llvm.shl %one, %width : i32
  ///     %active_mask = llvm.sub %shl, %one : i32
  ///     %mask_and_clamp = llvm.sub %width, %one : i32
  ///     %shfl = nvvm.shfl.sync.bfly %active_mask, %value, %offset,
  ///         %mask_and_clamp : !llvm<"{ float, i1 }">
  ///     %shfl_value = llvm.extractvalue %shfl[0 : index] :
  ///         !llvm<"{ float, i1 }">
  ///     %shfl_pred = llvm.extractvalue %shfl[1 : index] :
  ///         !llvm<"{ float, i1 }">
  LogicalResult
  matchAndRewrite(gpu::ShuffleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    auto valueTy = adaptor.value().getType();
    auto int32Type = IntegerType::get(rewriter.getContext(), 32);
    auto predTy = IntegerType::get(rewriter.getContext(), 1);
    auto resultTy = LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                                     {valueTy, predTy});

    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, int32Type, rewriter.getI32IntegerAttr(1));
    // Bit mask of active lanes: `(1 << activeWidth) - 1`.
    Value activeMask = rewriter.create<LLVM::SubOp>(
        loc, int32Type,
        rewriter.create<LLVM::ShlOp>(loc, int32Type, one, adaptor.width()),
        one);
    // Clamp lane: `activeWidth - 1`
    Value maskAndClamp =
        rewriter.create<LLVM::SubOp>(loc, int32Type, adaptor.width(), one);

    auto returnValueAndIsValidAttr = rewriter.getUnitAttr();
    Value shfl = rewriter.create<NVVM::ShflOp>(
        loc, resultTy, activeMask, adaptor.value(), adaptor.offset(),
        maskAndClamp, convertShflKind(op.mode()), returnValueAndIsValidAttr);
    Value shflValue = rewriter.create<LLVM::ExtractValueOp>(
        loc, valueTy, shfl, rewriter.getIndexArrayAttr(0));
    Value isActiveSrcLane = rewriter.create<LLVM::ExtractValueOp>(
        loc, predTy, shfl, rewriter.getIndexArrayAttr(1));

    rewriter.replaceOp(op, {shflValue, isActiveSrcLane});
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
    : public ConvertGpuOpsToNVVMOpsBase<LowerGpuOpsToNVVMOpsPass> {
  LowerGpuOpsToNVVMOpsPass() = default;
  LowerGpuOpsToNVVMOpsPass(unsigned indexBitwidth) {
    this->indexBitwidth = indexBitwidth;
  }

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();

    /// Customize the bitwidth used for the device side index computations.
    LowerToLLVMOptions options(
        m.getContext(),
        DataLayout(cast<DataLayoutOpInterface>(m.getOperation())));
    options.emitCWrappers = true;
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    /// MemRef conversion for GPU to NVVM lowering. The GPU dialect uses memory
    /// space 5 for private memory attributions, but NVVM represents private
    /// memory allocations as local `alloca`s in the default address space. This
    /// converter drops the private memory space to support the use case above.
    LLVMTypeConverter converter(m.getContext(), options);
    converter.addConversion([&](MemRefType type) -> Optional<Type> {
      if (type.getMemorySpaceAsInt() !=
          gpu::GPUDialect::getPrivateAddressSpace())
        return llvm::None;
      return converter.convertType(MemRefType::Builder(type).setMemorySpace(0));
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

    mlir::arith::populateArithmeticToLLVMConversionPatterns(converter,
                                                            llvmPatterns);
    populateStdToLLVMConversionPatterns(converter, llvmPatterns);
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
  target.addIllegalOp<FuncOp>();
  target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<::mlir::NVVM::NVVMDialect>();
  target.addIllegalDialect<gpu::GPUDialect>();
  target.addIllegalOp<LLVM::CosOp, LLVM::ExpOp, LLVM::Exp2Op, LLVM::FAbsOp,
                      LLVM::FCeilOp, LLVM::FFloorOp, LLVM::LogOp, LLVM::Log10Op,
                      LLVM::Log2Op, LLVM::PowOp, LLVM::SinOp, LLVM::SqrtOp>();

  // TODO: Remove once we support replacing non-root ops.
  target.addLegalOp<gpu::YieldOp, gpu::GPUModuleOp, gpu::ModuleEndOp>();
}

void mlir::populateGpuToNVVMConversionPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns) {
  populateWithGenerated(patterns);
  patterns
      .add<GPUIndexIntrinsicOpLowering<gpu::ThreadIdOp, NVVM::ThreadIdXOp,
                                       NVVM::ThreadIdYOp, NVVM::ThreadIdZOp>,
           GPUIndexIntrinsicOpLowering<gpu::BlockDimOp, NVVM::BlockDimXOp,
                                       NVVM::BlockDimYOp, NVVM::BlockDimZOp>,
           GPUIndexIntrinsicOpLowering<gpu::BlockIdOp, NVVM::BlockIdXOp,
                                       NVVM::BlockIdYOp, NVVM::BlockIdZOp>,
           GPUIndexIntrinsicOpLowering<gpu::GridDimOp, NVVM::GridDimXOp,
                                       NVVM::GridDimYOp, NVVM::GridDimZOp>,
           GPUShuffleOpLowering, GPUReturnOpLowering>(converter);

  // Explicitly drop memory space when lowering private memory
  // attributions since NVVM models it as `alloca`s in the default
  // memory space and does not support `alloca`s with addrspace(5).
  patterns.add<GPUFuncOpLowering>(
      converter, /*allocaAddrSpace=*/0,
      StringAttr::get(&converter.getContext(),
                      NVVM::NVVMDialect::getKernelFuncAttrName()));

  patterns.add<OpToFuncCallLowering<math::AbsOp>>(converter, "__nv_fabsf",
                                                  "__nv_fabs");
  patterns.add<OpToFuncCallLowering<math::AtanOp>>(converter, "__nv_atanf",
                                                   "__nv_atan");
  patterns.add<OpToFuncCallLowering<math::Atan2Op>>(converter, "__nv_atan2f",
                                                    "__nv_atan2");
  patterns.add<OpToFuncCallLowering<math::CeilOp>>(converter, "__nv_ceilf",
                                                   "__nv_ceil");
  patterns.add<OpToFuncCallLowering<math::CosOp>>(converter, "__nv_cosf",
                                                  "__nv_cos");
  patterns.add<OpToFuncCallLowering<math::ExpOp>>(converter, "__nv_expf",
                                                  "__nv_exp");
  patterns.add<OpToFuncCallLowering<math::Exp2Op>>(converter, "__nv_exp2f",
                                                   "__nv_exp2");
  patterns.add<OpToFuncCallLowering<math::ExpM1Op>>(converter, "__nv_expm1f",
                                                    "__nv_expm1");
  patterns.add<OpToFuncCallLowering<math::FloorOp>>(converter, "__nv_floorf",
                                                    "__nv_floor");
  patterns.add<OpToFuncCallLowering<math::LogOp>>(converter, "__nv_logf",
                                                  "__nv_log");
  patterns.add<OpToFuncCallLowering<math::Log1pOp>>(converter, "__nv_log1pf",
                                                    "__nv_log1p");
  patterns.add<OpToFuncCallLowering<math::Log10Op>>(converter, "__nv_log10f",
                                                    "__nv_log10");
  patterns.add<OpToFuncCallLowering<math::Log2Op>>(converter, "__nv_log2f",
                                                   "__nv_log2");
  patterns.add<OpToFuncCallLowering<math::PowFOp>>(converter, "__nv_powf",
                                                   "__nv_pow");
  patterns.add<OpToFuncCallLowering<math::RsqrtOp>>(converter, "__nv_rsqrtf",
                                                    "__nv_rsqrt");
  patterns.add<OpToFuncCallLowering<math::SinOp>>(converter, "__nv_sinf",
                                                  "__nv_sin");
  patterns.add<OpToFuncCallLowering<math::SqrtOp>>(converter, "__nv_sqrtf",
                                                   "__nv_sqrt");
  patterns.add<OpToFuncCallLowering<math::TanhOp>>(converter, "__nv_tanhf",
                                                   "__nv_tanh");
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createLowerGpuOpsToNVVMOpsPass(unsigned indexBitwidth) {
  return std::make_unique<LowerGpuOpsToNVVMOpsPass>(indexBitwidth);
}
