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

      if (Operation *existing = gpuSymTable.lookup(name.getValue())) {
        // Reuse only when the existing GPU symbol is structurally equivalent to
        // the global we would insert. Otherwise treat as a conflict (different
        // op type or different definition).
        if (existing->getName() != globalOp.getName() ||
            !OperationEquivalence::isEquivalentTo(
                existing, &globalOp,
                OperationEquivalence::ignoreValueEquivalence,
                /*markEquivalent=*/nullptr,
                OperationEquivalence::IgnoreLocations)) {
          accSupport.emitNYI(globalOp.getLoc(),
                             llvm::Twine("duplicate global symbol '") +
                                 name.getValue() + "' in gpu module");
          return failure();
        }
        continue;
      }

      gpuSymTable.insert(globalOp.clone());
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
