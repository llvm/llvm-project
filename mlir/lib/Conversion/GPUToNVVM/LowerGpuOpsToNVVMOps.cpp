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
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FormatVariadic.h"

#include "../GPUCommon/GPUOpsLowering.h"
#include "../GPUCommon/IndexIntrinsicsOpLowering.h"
#include "../GPUCommon/OpToFuncCallLowering.h"
#include "../PassDetail.h"

using namespace mlir;

namespace {

struct GPUShuffleOpLowering : public ConvertToLLVMPattern {
  explicit GPUShuffleOpLowering(LLVMTypeConverter &lowering_)
      : ConvertToLLVMPattern(gpu::ShuffleOp::getOperationName(),
                             lowering_.getDialect()->getContext(), lowering_) {}

  /// Lowers a shuffle to the corresponding NVVM op.
  ///
  /// Convert the `width` argument into an activeMask (a bitmask which specifies
  /// which threads participate in the shuffle) and a maskAndClamp (specifying
  /// the highest lane which participates in the shuffle).
  ///
  ///     %one = llvm.constant(1 : i32) : !llvm.i32
  ///     %shl = llvm.shl %one, %width : !llvm.i32
  ///     %active_mask = llvm.sub %shl, %one : !llvm.i32
  ///     %mask_and_clamp = llvm.sub %width, %one : !llvm.i32
  ///     %shfl = nvvm.shfl.sync.bfly %active_mask, %value, %offset,
  ///         %mask_and_clamp : !llvm<"{ float, i1 }">
  ///     %shfl_value = llvm.extractvalue %shfl[0 : index] :
  ///         !llvm<"{ float, i1 }">
  ///     %shfl_pred = llvm.extractvalue %shfl[1 : index] :
  ///         !llvm<"{ float, i1 }">
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    gpu::ShuffleOpAdaptor adaptor(operands);

    auto valueTy = adaptor.value().getType().cast<LLVM::LLVMType>();
    auto int32Type = LLVM::LLVMType::getInt32Ty(rewriter.getContext());
    auto predTy = LLVM::LLVMType::getInt1Ty(rewriter.getContext());
    auto resultTy =
        LLVM::LLVMType::getStructTy(rewriter.getContext(), {valueTy, predTy});

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
      if (type.getMemorySpace() != gpu::GPUDialect::getPrivateAddressSpace())
        return llvm::None;
      return converter.convertType(MemRefType::Builder(type).setMemorySpace(0));
    });

    OwningRewritePatternList patterns;

    // Apply in-dialect lowering first. In-dialect lowering will replace ops
    // which need to be lowered further, which is not supported by a single
    // conversion pass.
    populateGpuRewritePatterns(m.getContext(), patterns);
    applyPatternsAndFoldGreedily(m, patterns);
    patterns.clear();

    populateStdToLLVMConversionPatterns(converter, patterns);
    populateGpuToNVVMConversionPatterns(converter, patterns);
    LLVMConversionTarget target(getContext());
    target.addIllegalDialect<gpu::GPUDialect>();
    target.addIllegalOp<LLVM::CosOp, LLVM::ExpOp, LLVM::FAbsOp, LLVM::FCeilOp,
                        LLVM::LogOp, LLVM::Log10Op, LLVM::Log2Op>();
    target.addIllegalOp<FuncOp>();
    target.addLegalDialect<NVVM::NVVMDialect>();
    // TODO: Remove once we support replacing non-root ops.
    target.addLegalOp<gpu::YieldOp, gpu::GPUModuleOp, gpu::ModuleEndOp>();
    if (failed(applyPartialConversion(m, target, patterns)))
      signalPassFailure();
  }
};

} // anonymous namespace

void mlir::populateGpuToNVVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  populateWithGenerated(converter.getDialect()->getContext(), &patterns);
  patterns
      .insert<GPUIndexIntrinsicOpLowering<gpu::ThreadIdOp, NVVM::ThreadIdXOp,
                                          NVVM::ThreadIdYOp, NVVM::ThreadIdZOp>,
              GPUIndexIntrinsicOpLowering<gpu::BlockDimOp, NVVM::BlockDimXOp,
                                          NVVM::BlockDimYOp, NVVM::BlockDimZOp>,
              GPUIndexIntrinsicOpLowering<gpu::BlockIdOp, NVVM::BlockIdXOp,
                                          NVVM::BlockIdYOp, NVVM::BlockIdZOp>,
              GPUIndexIntrinsicOpLowering<gpu::GridDimOp, NVVM::GridDimXOp,
                                          NVVM::GridDimYOp, NVVM::GridDimZOp>,
              GPUShuffleOpLowering, GPUReturnOpLowering,
              // Explicitly drop memory space when lowering private memory
              // attributions since NVVM models it as `alloca`s in the default
              // memory space and does not support `alloca`s with addrspace(5).
              GPUFuncOpLowering<0>>(converter);
  patterns.insert<OpToFuncCallLowering<AbsFOp>>(converter, "__nv_fabsf",
                                                "__nv_fabs");
  patterns.insert<OpToFuncCallLowering<CeilFOp>>(converter, "__nv_ceilf",
                                                 "__nv_ceil");
  patterns.insert<OpToFuncCallLowering<CosOp>>(converter, "__nv_cosf",
                                               "__nv_cos");
  patterns.insert<OpToFuncCallLowering<ExpOp>>(converter, "__nv_expf",
                                               "__nv_exp");
  patterns.insert<OpToFuncCallLowering<LogOp>>(converter, "__nv_logf",
                                               "__nv_log");
  patterns.insert<OpToFuncCallLowering<Log10Op>>(converter, "__nv_log10f",
                                                 "__nv_log10");
  patterns.insert<OpToFuncCallLowering<Log2Op>>(converter, "__nv_log2f",
                                                "__nv_log2");
  patterns.insert<OpToFuncCallLowering<TanhOp>>(converter, "__nv_tanhf",
                                                "__nv_tanh");
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createLowerGpuOpsToNVVMOpsPass(unsigned indexBitwidth) {
  return std::make_unique<LowerGpuOpsToNVVMOpsPass>(indexBitwidth);
}
