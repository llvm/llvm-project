//===- TileReducerLLVMLowering.cpp - Milestone 29 ---------------*- C++ -*-===//
//
// Device:  gpu.func  -> NVVM / LLVM dialect -> LLVM IR (.ll)
// Host:    gpu.launch_func -> GPU runtime ABI / LLVM dialect
//
// Uses specific upstream conversion headers. Does not include
// mlir/Conversion/Passes.h (that registers every conversion).
//
//===----------------------------------------------------------------------===//

#include "TileReducer/TileReducerPasses.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::tr {
#define GEN_PASS_DEF_LOWERTRDEVICETONVVM
#define GEN_PASS_DEF_LOWERTRHOSTTOLLVM
#define GEN_PASS_DEF_EMITTRDEVICELLVMIR
#include "TileReducer/TileReducerPasses.h.inc"

namespace {

struct SubgroupSizeToConst : OpRewritePattern<gpu::SubgroupSizeOp> {
  unsigned warpSize;
  SubgroupSizeToConst(MLIRContext *ctx, unsigned warpSize)
      : OpRewritePattern<gpu::SubgroupSizeOp>(ctx), warpSize(warpSize) {}
  LogicalResult matchAndRewrite(gpu::SubgroupSizeOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, warpSize);
    return success();
  }
};

static LogicalResult
prepareDeviceGPUOps(gpu::GPUModuleOp gpuMod, unsigned warpSize) {
  RewritePatternSet patterns(gpuMod.getContext());
  populateGpuSubgroupIdPatterns(patterns);
  patterns.add<SubgroupSizeToConst>(gpuMod.getContext(), warpSize);
  populateGpuLowerSubgroupReduceToShufflePatterns(patterns, warpSize);
  return applyPatternsGreedily(gpuMod, std::move(patterns));
}

struct LowerTRDeviceToNVVM
    : impl::LowerTRDeviceToNVVMBase<LowerTRDeviceToNVVM> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool any = false;
    bool prepFailed = false;
    module.walk([&](gpu::GPUModuleOp gpuMod) {
      any = true;
      if (failed(prepareDeviceGPUOps(gpuMod, /*warpSize=*/32))) {
        gpuMod.emitError() << "failed to lower subgroup_reduce / subgroup_id";
        prepFailed = true;
      }
    });
    if (!any) {
      module.emitError() << "tr-lower-device-to-nvvm expects a gpu.module";
      return signalPassFailure();
    }
    if (prepFailed)
      return signalPassFailure();

    OpPassManager pm(module.getOperationName());
    pm.addNestedPass<gpu::GPUModuleOp>(createSCFToControlFlowPass());
    pm.addNestedPass<gpu::GPUModuleOp>(
        memref::createExpandStridedMetadataPass());
    ConvertGpuOpsToNVVMOpsOptions nvvmOpt;
    nvvmOpt.indexBitwidth = 64;
    nvvmOpt.hasRedux = false;
    pm.addNestedPass<gpu::GPUModuleOp>(createConvertGpuOpsToNVVMOps(nvvmOpt));
    pm.addNestedPass<gpu::GPUModuleOp>(createReconcileUnrealizedCastsPass());
    if (failed(runPipeline(pm, module)))
      signalPassFailure();
  }
};

struct LowerTRHostToLLVM : impl::LowerTRHostToLLVMBase<LowerTRHostToLLVM> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module.walk([](gpu::LaunchFuncOp) { return WalkResult::interrupt(); })
             .wasInterrupted()) {
      module.emitError() << "tr-lower-host-to-llvm expects gpu.launch_func";
      return signalPassFailure();
    }

    OpPassManager pm(module.getOperationName());
    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertIndexToLLVMPass());
    pm.addPass(createGpuToLLVMConversionPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    if (failed(runPipeline(pm, module)))
      signalPassFailure();
  }
};

static LogicalResult writeDeviceLLVMIR(gpu::GPUModuleOp gpuMod,
                                       raw_ostream &os) {
  MLIRContext *ctx = gpuMod.getContext();

  OpBuilder b(ctx);
  OwningOpRef<ModuleOp> tmp = ModuleOp::create(gpuMod.getLoc());
  (*tmp)->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
                  b.getStringAttr("nvptx64-nvidia-cuda"));
  b.setInsertionPointToStart(tmp->getBody());
  for (Operation &op : gpuMod.getOps()) {
    if (isa<LLVM::LLVMFuncOp, LLVM::GlobalOp>(op))
      b.clone(op);
  }

  llvm::LLVMContext llvmCtx;
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(tmp.get(), llvmCtx, gpuMod.getName());
  if (!llvmModule)
    return gpuMod.emitError("failed to translate device LLVM dialect to LLVM IR");
  llvmModule->print(os, nullptr);
  return success();
}

struct EmitTRDeviceLLVMIR : impl::EmitTRDeviceLLVMIRBase<EmitTRDeviceLLVMIR> {
  using impl::EmitTRDeviceLLVMIRBase<EmitTRDeviceLLVMIR>::EmitTRDeviceLLVMIRBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    gpu::GPUModuleOp gpuMod;
    module.walk([&](gpu::GPUModuleOp m) {
      if (!gpuMod)
        gpuMod = m;
    });
    if (!gpuMod) {
      module.emitError() << "tr-emit-device-llvmir expects a gpu.module";
      return signalPassFailure();
    }
    if (gpuMod.getOps<LLVM::LLVMFuncOp>().empty()) {
      gpuMod.emitError()
          << "device module has no llvm.func (run --tr-lower-device-to-nvvm)";
      return signalPassFailure();
    }

    if (outputPath.empty()) {
      if (failed(writeDeviceLLVMIR(gpuMod, llvm::errs())))
        return signalPassFailure();
      return;
    }

    std::error_code ec;
    llvm::raw_fd_ostream file(outputPath, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      module.emitError() << "cannot write " << outputPath << ": " << ec.message();
      return signalPassFailure();
    }
    if (failed(writeDeviceLLVMIR(gpuMod, file)))
      return signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tr
