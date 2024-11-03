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

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"

#include "mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/FormatVariadic.h"

#include "../GPUCommon/GPUOpsLowering.h"
#include "../GPUCommon/IndexIntrinsicsOpLowering.h"
#include "../GPUCommon/OpToFuncCallLowering.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTGPUOPSTOROCDLOPS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

/// Returns true if the given `gpu.func` can be safely called using the bare
/// pointer calling convention.
static bool canBeCalledWithBarePointers(gpu::GPUFuncOp func) {
  bool canBeBare = true;
  for (Type type : func.getArgumentTypes())
    if (auto memrefTy = type.dyn_cast<BaseMemRefType>())
      canBeBare &= LLVMTypeConverter::canConvertToBarePtr(memrefTy);
  return canBeBare;
}

namespace {

/// Import the GPU Ops to ROCDL Patterns.
#include "GPUToROCDL.cpp.inc"

// A pass that replaces all occurrences of GPU device operations with their
// corresponding ROCDL equivalent.
//
// This pass only handles device code and is not meant to be run on GPU host
// code.
struct LowerGpuOpsToROCDLOpsPass
    : public impl::ConvertGpuOpsToROCDLOpsBase<LowerGpuOpsToROCDLOpsPass> {
  LowerGpuOpsToROCDLOpsPass() = default;
  LowerGpuOpsToROCDLOpsPass(const std::string &chipset, unsigned indexBitwidth,
                            bool useBarePtrCallConv,
                            gpu::amd::Runtime runtime) {
    if (this->chipset.getNumOccurrences() == 0)
      this->chipset = chipset;
    if (this->indexBitwidth.getNumOccurrences() == 0)
      this->indexBitwidth = indexBitwidth;
    if (this->useBarePtrCallConv.getNumOccurrences() == 0)
      this->useBarePtrCallConv = useBarePtrCallConv;
    if (this->runtime.getNumOccurrences() == 0)
      this->runtime = runtime;
  }

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();
    MLIRContext *ctx = m.getContext();

    // Request C wrapper emission.
    for (auto func : m.getOps<func::FuncOp>()) {
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(ctx));
    }

    FailureOr<amdgpu::Chipset> maybeChipset = amdgpu::Chipset::parse(chipset);
    if (failed(maybeChipset)) {
      emitError(UnknownLoc::get(ctx), "Invalid chipset name: " + chipset);
      return signalPassFailure();
    }

    /// Customize the bitwidth used for the device side index computations.
    LowerToLLVMOptions options(
        ctx, DataLayout(cast<DataLayoutOpInterface>(m.getOperation())));
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    if (useBarePtrCallConv) {
      options.useBarePtrCallConv = true;
      WalkResult canUseBarePointers =
          m.walk([](gpu::GPUFuncOp func) -> WalkResult {
            if (canBeCalledWithBarePointers(func))
              return WalkResult::advance();
            return WalkResult::interrupt();
          });
      if (canUseBarePointers.wasInterrupted()) {
        emitError(UnknownLoc::get(ctx),
                  "bare pointer calling convention requires all memrefs to "
                  "have static shape and use the identity map");
        return signalPassFailure();
      }
    }

    // Apply in-dialect lowering. In-dialect lowering will replace
    // ops which need to be lowered further, which is not supported by a
    // single conversion pass.
    {
      RewritePatternSet patterns(ctx);
      populateGpuRewritePatterns(patterns);
      (void)applyPatternsAndFoldGreedily(m, std::move(patterns));
    }

    LLVMTypeConverter converter(ctx, options);
    populateGpuMemorySpaceAttributeConversions(
        converter, [](gpu::AddressSpace space) {
          switch (space) {
          case gpu::AddressSpace::Global:
            return 1;
          case gpu::AddressSpace::Workgroup:
            return 3;
          case gpu::AddressSpace::Private:
            return 5;
          }
          llvm_unreachable("unknown address space enum value");
          return 0;
        });

    RewritePatternSet llvmPatterns(ctx);

    mlir::arith::populateArithToLLVMConversionPatterns(converter, llvmPatterns);
    populateAMDGPUToROCDLConversionPatterns(converter, llvmPatterns,
                                            *maybeChipset);
    populateVectorToLLVMConversionPatterns(converter, llvmPatterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, llvmPatterns);
    populateFuncToLLVMConversionPatterns(converter, llvmPatterns);
    populateFinalizeMemRefToLLVMConversionPatterns(converter, llvmPatterns);
    populateGpuToROCDLConversionPatterns(converter, llvmPatterns, runtime);
    LLVMConversionTarget target(getContext());
    configureGpuToROCDLConversionLegality(target);
    if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
      signalPassFailure();

    // Manually rewrite known block size attributes so the LLVMIR translation
    // infrastructure can pick them up.
    m.walk([ctx](LLVM::LLVMFuncOp op) {
      if (auto blockSizes =
              op->removeAttr(gpu::GPUFuncOp::getKnownBlockSizeAttrName())
                  .dyn_cast_or_null<DenseI32ArrayAttr>()) {
        op->setAttr(ROCDL::ROCDLDialect::getReqdWorkGroupSizeAttrName(),
                    blockSizes);
        // Also set up the rocdl.flat_work_group_size attribute to prevent
        // conflicting metadata.
        uint32_t flatSize = 1;
        for (uint32_t size : blockSizes.asArrayRef()) {
          flatSize *= size;
        }
        StringAttr flatSizeAttr =
            StringAttr::get(ctx, Twine(flatSize) + "," + Twine(flatSize));
        op->setAttr(ROCDL::ROCDLDialect::getFlatWorkGroupSizeAttrName(),
                    flatSizeAttr);
      }
    });
  }
};

} // namespace

void mlir::configureGpuToROCDLConversionLegality(ConversionTarget &target) {
  target.addIllegalOp<func::FuncOp>();
  target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<ROCDL::ROCDLDialect>();
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

void mlir::populateGpuToROCDLConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    mlir::gpu::amd::Runtime runtime) {
  using mlir::gpu::amd::Runtime;

  populateWithGenerated(patterns);
  patterns
      .add<GPUIndexIntrinsicOpLowering<gpu::ThreadIdOp, ROCDL::ThreadIdXOp,
                                       ROCDL::ThreadIdYOp, ROCDL::ThreadIdZOp>>(
          converter, gpu::GPUFuncOp::getKnownBlockSizeAttrName());
  patterns.add<GPUIndexIntrinsicOpLowering<
      gpu::BlockIdOp, ROCDL::BlockIdXOp, ROCDL::BlockIdYOp, ROCDL::BlockIdZOp>>(
      converter, gpu::GPUFuncOp::getKnownGridSizeAttrName());
  patterns
      .add<GPUIndexIntrinsicOpLowering<gpu::BlockDimOp, ROCDL::BlockDimXOp,
                                       ROCDL::BlockDimYOp, ROCDL::BlockDimZOp>,
           GPUIndexIntrinsicOpLowering<gpu::GridDimOp, ROCDL::GridDimXOp,
                                       ROCDL::GridDimYOp, ROCDL::GridDimZOp>,
           GPUReturnOpLowering>(converter);
  patterns.add<GPUFuncOpLowering>(
      converter,
      /*allocaAddrSpace=*/ROCDL::ROCDLDialect::kPrivateMemoryAddressSpace,
      /*workgroupAddrSpace=*/ROCDL::ROCDLDialect::kSharedMemoryAddressSpace,
      StringAttr::get(&converter.getContext(),
                      ROCDL::ROCDLDialect::getKernelFuncAttrName()));
  if (Runtime::HIP == runtime) {
    patterns.add<GPUPrintfOpToHIPLowering>(converter);
  } else if (Runtime::OpenCL == runtime) {
    // Use address space = 4 to match the OpenCL definition of printf()
    patterns.add<GPUPrintfOpToLLVMCallLowering>(converter, /*addressSpace=*/4);
  }

  populateOpPatterns<math::AbsFOp>(converter, patterns, "__ocml_fabs_f32",
                                   "__ocml_fabs_f64");
  populateOpPatterns<math::AtanOp>(converter, patterns, "__ocml_atan_f32",
                                   "__ocml_atan_f64");
  populateOpPatterns<math::Atan2Op>(converter, patterns, "__ocml_atan2_f32",
                                    "__ocml_atan2_f64");
  populateOpPatterns<math::CbrtOp>(converter, patterns, "__ocml_cbrt_f32",
                                   "__ocml_cbrt_f64");
  populateOpPatterns<math::CeilOp>(converter, patterns, "__ocml_ceil_f32",
                                   "__ocml_ceil_f64");
  populateOpPatterns<math::CosOp>(converter, patterns, "__ocml_cos_f32",
                                  "__ocml_cos_f64");
  populateOpPatterns<math::ExpOp>(converter, patterns, "__ocml_exp_f32",
                                  "__ocml_exp_f64");
  populateOpPatterns<math::Exp2Op>(converter, patterns, "__ocml_exp2_f32",
                                   "__ocml_exp2_f64");
  populateOpPatterns<math::ExpM1Op>(converter, patterns, "__ocml_expm1_f32",
                                    "__ocml_expm1_f64");
  populateOpPatterns<math::FloorOp>(converter, patterns, "__ocml_floor_f32",
                                    "__ocml_floor_f64");
  populateOpPatterns<math::LogOp>(converter, patterns, "__ocml_log_f32",
                                  "__ocml_log_f64");
  populateOpPatterns<math::Log10Op>(converter, patterns, "__ocml_log10_f32",
                                    "__ocml_log10_f64");
  populateOpPatterns<math::Log1pOp>(converter, patterns, "__ocml_log1p_f32",
                                    "__ocml_log1p_f64");
  populateOpPatterns<math::Log2Op>(converter, patterns, "__ocml_log2_f32",
                                   "__ocml_log2_f64");
  populateOpPatterns<math::PowFOp>(converter, patterns, "__ocml_pow_f32",
                                   "__ocml_pow_f64");
  populateOpPatterns<math::RsqrtOp>(converter, patterns, "__ocml_rsqrt_f32",
                                    "__ocml_rsqrt_f64");
  populateOpPatterns<math::SinOp>(converter, patterns, "__ocml_sin_f32",
                                  "__ocml_sin_f64");
  populateOpPatterns<math::SqrtOp>(converter, patterns, "__ocml_sqrt_f32",
                                   "__ocml_sqrt_f64");
  populateOpPatterns<math::TanhOp>(converter, patterns, "__ocml_tanh_f32",
                                   "__ocml_tanh_f64");
  populateOpPatterns<math::TanOp>(converter, patterns, "__ocml_tan_f32",
                                  "__ocml_tan_f64");
}

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createLowerGpuOpsToROCDLOpsPass(const std::string &chipset,
                                      unsigned indexBitwidth,
                                      bool useBarePtrCallConv,
                                      gpu::amd::Runtime runtime) {
  return std::make_unique<LowerGpuOpsToROCDLOpsPass>(
      chipset, indexBitwidth, useBarePtrCallConv, runtime);
}
