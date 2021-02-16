//===- LowerGpuOpsToROCDLOps.cpp - MLIR GPU to ROCDL lowering passes ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate ROCDLIR operations for higher-level
// GPU operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToROCDL/VectorToROCDL.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/FormatVariadic.h"

#include "../GPUCommon/GPUOpsLowering.h"
#include "../GPUCommon/IndexIntrinsicsOpLowering.h"
#include "../GPUCommon/OpToFuncCallLowering.h"
#include "../PassDetail.h"

using namespace mlir;

namespace {

/// Import the GPU Ops to ROCDL Patterns.
#include "GPUToROCDL.cpp.inc"

// A pass that replaces all occurrences of GPU device operations with their
// corresponding ROCDL equivalent.
//
// This pass only handles device code and is not meant to be run on GPU host
// code.
struct LowerGpuOpsToROCDLOpsPass
    : public ConvertGpuOpsToROCDLOpsBase<LowerGpuOpsToROCDLOpsPass> {
  LowerGpuOpsToROCDLOpsPass() = default;
  LowerGpuOpsToROCDLOpsPass(unsigned indexBitwidth) {
    this->indexBitwidth = indexBitwidth;
  }

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();

    /// Customize the bitwidth used for the device side index computations.
    LowerToLLVMOptions options = {/*useBarePtrCallConv =*/false,
                                  /*emitCWrappers =*/true,
                                  /*indexBitwidth =*/indexBitwidth,
                                  /*useAlignedAlloc =*/false};
    LLVMTypeConverter converter(m.getContext(), options);

    OwningRewritePatternList patterns, llvmPatterns;

    populateGpuRewritePatterns(m.getContext(), patterns);
    (void)applyPatternsAndFoldGreedily(m, std::move(patterns));

    populateVectorToLLVMConversionPatterns(converter, llvmPatterns);
    populateVectorToROCDLConversionPatterns(converter, llvmPatterns);
    populateStdToLLVMConversionPatterns(converter, llvmPatterns);
    populateGpuToROCDLConversionPatterns(converter, llvmPatterns);
    LLVMConversionTarget target(getContext());
    configureGpuToROCDLConversionLegality(target);
    if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
      signalPassFailure();
  }
};

} // anonymous namespace

void mlir::configureGpuToROCDLConversionLegality(ConversionTarget &target) {
  target.addIllegalOp<FuncOp>();
  target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<ROCDL::ROCDLDialect>();
  target.addIllegalDialect<gpu::GPUDialect>();
  target.addIllegalOp<LLVM::CosOp, LLVM::ExpOp, LLVM::FAbsOp, LLVM::FCeilOp,
                      LLVM::FFloorOp, LLVM::LogOp, LLVM::Log10Op, LLVM::Log2Op,
                      LLVM::PowOp, LLVM::SinOp, LLVM::SqrtOp>();

  // TODO: Remove once we support replacing non-root ops.
  target.addLegalOp<gpu::YieldOp, gpu::GPUModuleOp, gpu::ModuleEndOp>();
}

void mlir::populateGpuToROCDLConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  populateWithGenerated(converter.getDialect()->getContext(), patterns);
  patterns.insert<
      GPUIndexIntrinsicOpLowering<gpu::ThreadIdOp, ROCDL::ThreadIdXOp,
                                  ROCDL::ThreadIdYOp, ROCDL::ThreadIdZOp>,
      GPUIndexIntrinsicOpLowering<gpu::BlockDimOp, ROCDL::BlockDimXOp,
                                  ROCDL::BlockDimYOp, ROCDL::BlockDimZOp>,
      GPUIndexIntrinsicOpLowering<gpu::BlockIdOp, ROCDL::BlockIdXOp,
                                  ROCDL::BlockIdYOp, ROCDL::BlockIdZOp>,
      GPUIndexIntrinsicOpLowering<gpu::GridDimOp, ROCDL::GridDimXOp,
                                  ROCDL::GridDimYOp, ROCDL::GridDimZOp>,
      GPUReturnOpLowering>(converter);
  patterns.insert<GPUFuncOpLowering>(
      converter, /*allocaAddrSpace=*/5,
      Identifier::get(ROCDL::ROCDLDialect::getKernelFuncAttrName(),
                      &converter.getContext()));
  patterns.insert<OpToFuncCallLowering<AbsFOp>>(converter, "__ocml_fabs_f32",
                                                "__ocml_fabs_f64");
  patterns.insert<OpToFuncCallLowering<math::AtanOp>>(
      converter, "__ocml_atan_f32", "__ocml_atan_f64");
  patterns.insert<OpToFuncCallLowering<math::Atan2Op>>(
      converter, "__ocml_atan2_f32", "__ocml_atan2_f64");
  patterns.insert<OpToFuncCallLowering<CeilFOp>>(converter, "__ocml_ceil_f32",
                                                 "__ocml_ceil_f64");
  patterns.insert<OpToFuncCallLowering<math::CosOp>>(
      converter, "__ocml_cos_f32", "__ocml_cos_f64");
  patterns.insert<OpToFuncCallLowering<math::ExpOp>>(
      converter, "__ocml_exp_f32", "__ocml_exp_f64");
  patterns.insert<OpToFuncCallLowering<math::ExpM1Op>>(
      converter, "__ocml_expm1_f32", "__ocml_expm1_f64");
  patterns.insert<OpToFuncCallLowering<FloorFOp>>(converter, "__ocml_floor_f32",
                                                  "__ocml_floor_f64");
  patterns.insert<OpToFuncCallLowering<math::LogOp>>(
      converter, "__ocml_log_f32", "__ocml_log_f64");
  patterns.insert<OpToFuncCallLowering<math::Log10Op>>(
      converter, "__ocml_log10_f32", "__ocml_log10_f64");
  patterns.insert<OpToFuncCallLowering<math::Log1pOp>>(
      converter, "__ocml_log1p_f32", "__ocml_log1p_f64");
  patterns.insert<OpToFuncCallLowering<math::Log2Op>>(
      converter, "__ocml_log2_f32", "__ocml_log2_f64");
  patterns.insert<OpToFuncCallLowering<math::PowFOp>>(
      converter, "__ocml_pow_f32", "__ocml_pow_f64");
  patterns.insert<OpToFuncCallLowering<math::RsqrtOp>>(
      converter, "__ocml_rsqrt_f32", "__ocml_rsqrt_f64");
  patterns.insert<OpToFuncCallLowering<math::SinOp>>(
      converter, "__ocml_sin_f32", "__ocml_sin_f64");
  patterns.insert<OpToFuncCallLowering<math::SqrtOp>>(
      converter, "__ocml_sqrt_f32", "__ocml_sqrt_f64");
  patterns.insert<OpToFuncCallLowering<math::TanhOp>>(
      converter, "__ocml_tanh_f32", "__ocml_tanh_f64");
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createLowerGpuOpsToROCDLOpsPass(unsigned indexBitwidth) {
  return std::make_unique<LowerGpuOpsToROCDLOpsPass>(indexBitwidth);
}
