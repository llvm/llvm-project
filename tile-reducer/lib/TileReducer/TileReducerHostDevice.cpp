//===- TileReducerHostDevice.cpp - Milestone 28 -----------------*- C++ -*-===//
//
// Make the host/device split explicit. Host stays in func.func with
// gpu.launch_func. Device lives in gpu.module / gpu.func and is annotated
// with #nvvm.target so later GPU->NVVM lowering has a concrete backend.
//
//===----------------------------------------------------------------------===//

#include "TileReducer/TileReducerPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;

namespace mlir::tr {
#define GEN_PASS_DEF_SPLITTRHOSTDEVICE
#include "TileReducer/TileReducerPasses.h.inc"

namespace {

constexpr StringRef kRoleAttr = "tr.role";
constexpr StringRef kSplitAttr = "tr.host_device_split";
constexpr StringRef kHostRole = "host";
constexpr StringRef kDeviceRole = "device";

static NVVM::NVVMTargetAttr a100Target(MLIRContext *ctx) {
  // A100-class chip. verifyTarget is off so attaching the attribute does
  // not reject existing GPU ops before they are lowered.
  return NVVM::NVVMTargetAttr::get(ctx, /*optLevel=*/2,
                                   "nvptx64-nvidia-cuda", "sm_80",
                                   /*features=*/"", /*targetFlags=*/nullptr,
                                   /*linkFiles=*/nullptr,
                                   /*verifyTarget=*/false);
}

struct SplitTRHostDevice : impl::SplitTRHostDeviceBase<SplitTRHostDevice> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    OpBuilder b(ctx);

    SmallVector<gpu::GPUModuleOp> gpuMods;
    module.walk([&](gpu::GPUModuleOp gpuMod) { gpuMods.push_back(gpuMod); });
    if (gpuMods.empty()) {
      module.emitError()
          << "tr-split-host-device expects a gpu.module (run "
             "--tr-emit-gpu-kernels first)";
      return signalPassFailure();
    }

    NVVM::NVVMTargetAttr target = a100Target(ctx);
    for (gpu::GPUModuleOp gpuMod : gpuMods) {
      SmallVector<Attribute> targets;
      if (std::optional<ArrayAttr> existing = gpuMod.getTargets())
        targets.append(existing->begin(), existing->end());
      if (!llvm::is_contained(targets, target))
        targets.push_back(target);
      gpuMod.setTargetsAttr(b.getArrayAttr(targets));

      gpuMod.walk([&](gpu::GPUFuncOp kernel) {
        kernel->setAttr(kRoleAttr, b.getStringAttr(kDeviceRole));
      });
    }

    unsigned launchCount = 0;
    module.walk([&](func::FuncOp func) {
      if (func->getParentOfType<gpu::GPUModuleOp>())
        return;
      bool isHost = false;
      func.walk([&](gpu::LaunchFuncOp launch) {
        isHost = true;
        ++launchCount;
        auto ref = launch.getKernel();
        if (!SymbolTable::lookupNearestSymbolFrom(launch, ref)) {
          launch.emitError()
              << "host launch does not resolve kernel symbol " << ref;
          signalPassFailure();
        }
      });
      if (isHost)
        func->setAttr(kRoleAttr, b.getStringAttr(kHostRole));
    });

    if (launchCount == 0) {
      module.emitError()
          << "tr-split-host-device expects at least one gpu.launch_func";
      return signalPassFailure();
    }

    module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                    b.getUnitAttr());
    module->setAttr(kSplitAttr, b.getUnitAttr());
  }
};

} // namespace
} // namespace mlir::tr
