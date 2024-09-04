#include "mlir/Registration/Pipelines.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRVPass.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Process.h"
#include <optional>

using namespace mlir;

void mlir::createTosaFuserPipeline(OpPassManager &pm, const TosaFuserPipelineOptions &options,
                                   unsigned optLevel) {
    pm.addNestedPass<func::FuncOp>(tosa::createTosaToLinalg());
    pm.addPass(bufferization::createEmptyTensorEliminationPass());
    pm.addNestedPass<func::FuncOp>(bufferization::createEmptyTensorToAllocTensorPass());
    pm.addPass(bufferization::createOneShotBufferizePass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createConvertLinalgToAffineLoopsPass());
}

static void tosaFuser3(OpPassManager &pm, const TosaFuserPipelineOptions &options) {
    createTosaFuserPipeline(pm, options, 3);
}

void mlir::registerTosaFuserPipeline () {
    static bool init_once = []() {
        PassPipelineRegistration<TosaFuserPipelineOptions>(
            "O3", "Tosa-Fuser Pipeline O3", tosaFuser3);
        return true;
    }();
}