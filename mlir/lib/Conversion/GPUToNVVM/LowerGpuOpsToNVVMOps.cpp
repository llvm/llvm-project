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

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
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
  matchAndRewrite(gpu::ShuffleOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    gpu::ShuffleOpAdaptor adaptor(operands);

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
    Value shfl = rewriter.create<NVVM::ShflBflyOp>(
        loc, resultTy, activeMask, adaptor.value(), adaptor.offset(),
        maskAndClamp, returnValueAndIsValidAttr);
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
    LowerToLLVMOptions options = {/*useBarePtrCallConv =*/false,
                                  /*emitCWrappers =*/true,
                                  /*indexBitwidth =*/indexBitwidth,
                                  /*useAlignedAlloc =*/false};

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

    OwningRewritePatternList patterns, llvmPatterns;

    // Apply in-dialect lowering first. In-dialect lowering will replace ops
    // which need to be lowered further, which is not supported by a single
    // conversion pass.
    populateGpuRewritePatterns(m.getContext(), patterns);
    (void)applyPatternsAndFoldGreedily(m, std::move(patterns));

    populateStdToLLVMConversionPatterns(converter, llvmPatterns);
    populateGpuToNVVMConversionPatterns(converter, llvmPatterns);
    LLVMConversionTarget target(getContext());
    configureGpuToNVVMConversionLegality(target);
    if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
      signalPassFailure();
  }
};

} // anonymous namespace

void mlir::configureGpuToNVVMConversionLegality(ConversionTarget &target) {
  target.addIllegalOp<FuncOp>();
  target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<::mlir::NVVM::NVVMDialect>();
  target.addIllegalDialect<gpu::GPUDialect>();
  target.addIllegalOp<LLVM::CosOp, LLVM::ExpOp, LLVM::FAbsOp, LLVM::FCeilOp,
                      LLVM::FFloorOp, LLVM::LogOp, LLVM::Log10Op, LLVM::Log2Op,
                      LLVM::PowOp, LLVM::SinOp, LLVM::SqrtOp>();

  // TODO: Remove once we support replacing non-root ops.
  target.addLegalOp<gpu::YieldOp, gpu::GPUModuleOp, gpu::ModuleEndOp>();
}

void mlir::populateGpuToNVVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  populateWithGenerated(converter.getDialect()->getContext(), patterns);
  patterns
      .insert<GPUIndexIntrinsicOpLowering<gpu::ThreadIdOp, NVVM::ThreadIdXOp,
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
  patterns.insert<GPUFuncOpLowering>(
      converter, /*allocaAddrSpace=*/0,
      Identifier::get(NVVM::NVVMDialect::getKernelFuncAttrName(),
                      &converter.getContext()));

  patterns.insert<OpToFuncCallLowering<AbsFOp>>(converter, "__nv_fabsf",
                                                "__nv_fabs");
  patterns.insert<OpToFuncCallLowering<math::AtanOp>>(converter, "__nv_atanf",
                                                      "__nv_atan");
  patterns.insert<OpToFuncCallLowering<math::Atan2Op>>(converter, "__nv_atan2f",
                                                       "__nv_atan2");
  patterns.insert<OpToFuncCallLowering<CeilFOp>>(converter, "__nv_ceilf",
                                                 "__nv_ceil");
  patterns.insert<OpToFuncCallLowering<math::CosOp>>(converter, "__nv_cosf",
                                                     "__nv_cos");
  patterns.insert<OpToFuncCallLowering<math::ExpOp>>(converter, "__nv_expf",
                                                     "__nv_exp");
  patterns.insert<OpToFuncCallLowering<math::ExpM1Op>>(converter, "__nv_expm1f",
                                                       "__nv_expm1");
  patterns.insert<OpToFuncCallLowering<FloorFOp>>(converter, "__nv_floorf",
                                                  "__nv_floor");
  patterns.insert<OpToFuncCallLowering<math::LogOp>>(converter, "__nv_logf",
                                                     "__nv_log");
  patterns.insert<OpToFuncCallLowering<math::Log1pOp>>(converter, "__nv_log1pf",
                                                       "__nv_log1p");
  patterns.insert<OpToFuncCallLowering<math::Log10Op>>(converter, "__nv_log10f",
                                                       "__nv_log10");
  patterns.insert<OpToFuncCallLowering<math::Log2Op>>(converter, "__nv_log2f",
                                                      "__nv_log2");
  patterns.insert<OpToFuncCallLowering<math::PowFOp>>(converter, "__nv_powf",
                                                      "__nv_pow");
  patterns.insert<OpToFuncCallLowering<math::RsqrtOp>>(converter, "__nv_rsqrtf",
                                                       "__nv_rsqrt");
  patterns.insert<OpToFuncCallLowering<math::SinOp>>(converter, "__nv_sinf",
                                                     "__nv_sin");
  patterns.insert<OpToFuncCallLowering<math::SqrtOp>>(converter, "__nv_sqrtf",
                                                      "__nv_sqrt");
  patterns.insert<OpToFuncCallLowering<math::TanhOp>>(converter, "__nv_tanhf",
                                                      "__nv_tanh");
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createLowerGpuOpsToNVVMOpsPass(unsigned indexBitwidth) {
  return std::make_unique<LowerGpuOpsToNVVMOpsPass>(indexBitwidth);
}
