//===- ACCDeclareGPUModuleInsertion.cpp
//------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass copies globals marked with the `acc.declare` attribute into the
// GPU module so that device code (e.g. acc routine, compute regions) can
// reference them.
//
// Overview:
// ---------
// Globals that have the `acc.declare` attribute (from the OpenACC declare
// directive or from the `ACCImplicitDeclare` pass) must be present in the
// GPU module for device code to use them. This pass inserts copies of those
// globals into the GPU module, creating the module if it does not yet exist.
// The host copy of each global remains in the parent module.
//
// Example:
// --------
//
// Before:
//   module {
//     memref.global @arr : memref<7xf32> = dense<0.0>
//         {acc.declare = #acc.declare<dataClause = acc_create>}
//   }
//
// After:
//   module attributes {gpu.container_module} {
//     memref.global @arr : memref<7xf32> = dense<0.0>
//         {acc.declare = #acc.declare<dataClause = acc_create>}
//     gpu.module @acc_gpu_module {
//       memref.global @arr : memref<7xf32> = dense<0.0>
//           {acc.declare = #acc.declare<dataClause = acc_create>}
//     }
//   }
//
// Requirements:
// -------------
// The pass uses the `acc::OpenACCSupport` for:
// - getOrCreateGPUModule: to obtain or create the GPU module.
// - emitNYI: to report failure when GPU module creation is not supported.
// If no custom implementation is registered, the default implementation is
// used (see OpenACCSupport).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_ACCDECLAREGPUMODULEINSERTION
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

#define DEBUG_TYPE "acc-declare-gpu-module-insertion"

using namespace mlir;

namespace {

static bool hasAccDeclareGlobals(ModuleOp mod) {
  for (Operation &op : mod.getBody()->getOperations())
    if (op.getAttr(acc::getDeclareAttrName()))
      return true;
  return false;
}

static void makeDeviceGlobalDeclaration(Operation &globalOp) {
  globalOp.removeAttr("initVal");
  globalOp.removeAttr("linkName");
  for (Region &region : globalOp.getRegions()) {
    region.dropAllReferences();
    region.getBlocks().clear();
  }
}

class ACCDeclareGPUModuleInsertion
    : public acc::impl::ACCDeclareGPUModuleInsertionBase<
          ACCDeclareGPUModuleInsertion> {
public:
  using acc::impl::ACCDeclareGPUModuleInsertionBase<
      ACCDeclareGPUModuleInsertion>::ACCDeclareGPUModuleInsertionBase;

  LogicalResult copyGlobalsToGPUModule(gpu::GPUModuleOp gpuMod, ModuleOp mod,
                                       acc::OpenACCSupport &accSupport) const {
    SymbolTable gpuSymTable(gpuMod);

    for (Operation &globalOp : mod.getBody()->getOperations()) {
      if (!globalOp.getAttr(acc::getDeclareAttrName()))
        continue;

      auto symOp = dyn_cast<SymbolOpInterface>(&globalOp);
      if (!symOp)
        continue;

      StringAttr name = symOp.getNameAttr();
      Operation *deviceGlobal = globalOp.clone();
      auto declareAttr =
          globalOp.getAttrOfType<acc::DeclareAttr>(acc::getDeclareAttrName());
      if (cudaUnified && declareAttr.getDataClause().getValue() !=
                             acc::DataClause::acc_declare_device_resident)
        makeDeviceGlobalDeclaration(*deviceGlobal);

      if (Operation *existing = gpuSymTable.lookup(name.getValue())) {
        // Reuse when structurally equivalent ignoring locations and discardable
        // attrs such as `acc.declare` attributes. Only a different op type or a
        // true definition mismatch is a conflict.
        if (existing->getName() != globalOp.getName() ||
            !OperationEquivalence::isEquivalentTo(
                existing, deviceGlobal,
                OperationEquivalence::ignoreValueEquivalence,
                /*markEquivalent=*/nullptr,
                OperationEquivalence::IgnoreLocations |
                    OperationEquivalence::IgnoreDiscardableAttrs)) {
          deviceGlobal->destroy();
          accSupport.emitNYI(globalOp.getLoc(),
                             llvm::Twine("duplicate global symbol '") +
                                 name.getValue() + "' in gpu module");
          return failure();
        }
        // Propagate acc.declare onto the GPU copy if it was cloned before the
        // host global was marked.
        if (!existing->getAttr(acc::getDeclareAttrName()))
          if (Attribute declareAttr =
                  globalOp.getAttr(acc::getDeclareAttrName()))
            existing->setAttr(acc::getDeclareAttrName(), declareAttr);
        deviceGlobal->destroy();
        continue;
      }

      gpuSymTable.insert(deviceGlobal);
    }
    return success();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Check for any candidates first - do this to avoid creating the GPU module
    // if there are no candidates.
    if (!hasAccDeclareGlobals(mod))
      return;

    acc::OpenACCSupport &accSupport = getAnalysis<acc::OpenACCSupport>();
    std::optional<gpu::GPUModuleOp> gpuMod =
        accSupport.getOrCreateGPUModule(mod);
    if (!gpuMod) {
      accSupport.emitNYI(mod.getLoc(), "Failed to create GPU module");
      return;
    }

    if (failed(copyGlobalsToGPUModule(*gpuMod, mod, accSupport)))
      return;
  }
};

} // namespace
