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
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
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
class LowerGpuOpsToROCDLOpsPass
    : public ConvertGpuOpsToROCDLOpsBase<LowerGpuOpsToROCDLOpsPass> {
public:
  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();

    LLVMTypeConverter converter(m.getContext());

    OwningRewritePatternList patterns;

    populateGpuRewritePatterns(m.getContext(), patterns);
    applyPatternsAndFoldGreedily(m, patterns);
    patterns.clear();

    populateVectorToLLVMConversionPatterns(converter, patterns);
    populateVectorToROCDLConversionPatterns(converter, patterns);
    populateStdToLLVMConversionPatterns(converter, patterns);
    populateGpuToROCDLConversionPatterns(converter, patterns);
    LLVMConversionTarget target(getContext());
    target.addIllegalDialect<gpu::GPUDialect>();
    target.addIllegalOp<LLVM::CosOp, LLVM::ExpOp, LLVM::FAbsOp, LLVM::FCeilOp,
                        LLVM::LogOp, LLVM::Log10Op, LLVM::Log2Op>();
    target.addIllegalOp<FuncOp>();
    target.addLegalDialect<ROCDL::ROCDLDialect>();
    // TODO(whchung): Remove once we support replacing non-root ops.
    target.addLegalOp<gpu::YieldOp, gpu::GPUModuleOp, gpu::ModuleEndOp>();
    if (failed(applyPartialConversion(m, target, patterns)))
      signalPassFailure();
  }
};

} // anonymous namespace

void mlir::populateGpuToROCDLConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  populateWithGenerated(converter.getDialect()->getContext(), &patterns);
  patterns.insert<
      GPUIndexIntrinsicOpLowering<gpu::ThreadIdOp, ROCDL::ThreadIdXOp,
                                  ROCDL::ThreadIdYOp, ROCDL::ThreadIdZOp>,
      GPUIndexIntrinsicOpLowering<gpu::BlockDimOp, ROCDL::BlockDimXOp,
                                  ROCDL::BlockDimYOp, ROCDL::BlockDimZOp>,
      GPUIndexIntrinsicOpLowering<gpu::BlockIdOp, ROCDL::BlockIdXOp,
                                  ROCDL::BlockIdYOp, ROCDL::BlockIdZOp>,
      GPUIndexIntrinsicOpLowering<gpu::GridDimOp, ROCDL::GridDimXOp,
                                  ROCDL::GridDimYOp, ROCDL::GridDimZOp>,
      GPUFuncOpLowering<5>, GPUReturnOpLowering>(converter);
  patterns.insert<OpToFuncCallLowering<AbsFOp>>(converter, "__ocml_fabs_f32",
                                                "__ocml_fabs_f64");
  patterns.insert<OpToFuncCallLowering<CeilFOp>>(converter, "__ocml_ceil_f32",
                                                 "__ocml_ceil_f64");
  patterns.insert<OpToFuncCallLowering<CosOp>>(converter, "__ocml_cos_f32",
                                               "__ocml_cos_f64");
  patterns.insert<OpToFuncCallLowering<ExpOp>>(converter, "__ocml_exp_f32",
                                               "__ocml_exp_f64");
  patterns.insert<OpToFuncCallLowering<LogOp>>(converter, "__ocml_log_f32",
                                               "__ocml_log_f64");
  patterns.insert<OpToFuncCallLowering<Log10Op>>(converter, "__ocml_log10_f32",
                                                 "__ocml_log10_f64");
  patterns.insert<OpToFuncCallLowering<Log2Op>>(converter, "__ocml_log2_f32",
                                                "__ocml_log2_f64");
  patterns.insert<OpToFuncCallLowering<TanhOp>>(converter, "__ocml_tanh_f32",
                                                "__ocml_tanh_f64");
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createLowerGpuOpsToROCDLOpsPass() {
  return std::make_unique<LowerGpuOpsToROCDLOpsPass>();
}
