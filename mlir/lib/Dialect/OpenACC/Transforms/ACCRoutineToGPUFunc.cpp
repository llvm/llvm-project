//===- ACCRoutineToGPUFunc.cpp - Move ACC routines to GPU module ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The OpenACC `routine` directive defines functions that may be invoked from
// device code. Those functions need to be available in the device compilation
// unit. This pass moves materialized acc routines into the GPU module as
// gpu.func operations so they can be compiled for the device.
//
// Overview:
// ---------
// For each acc.routine that is not bound by name, the corresponding
// specialized function (created by ACCRoutineLowering) or the original
// host function (in case of seq) is cloned into theGPU module as a gpu.func.
// Callees referenced from those routines are processed: device-valid callees
// (runtime, intrinsics, other acc routines) are added to the GPU module as
// declarations or full clones as needed. Bind-name routines are not moved;
// their acc.routine ops are erased. After cloning, the host copies of
// specialized device functions and nohost routines are removed.
//
// Approach:
// ----------------
// 1. Collect materialized routines (acc.routine without bind(name)); record
//    bind-name routines for erasure. Emit remarks for materialized routines.
//
// 2. Process calls: walk each materialized function; for each call, if the
//    callee is already in the GPU module or is an acc routine (or specialized
//    acc routine), skip; otherwise require OpenACCSupport::isValidSymbolUse.
//    Valid callees are added to the clone set (as declaration or full clone).
//
// 3. Clone into GPU module: each function in the clone set is turned into a
//    gpu.func (body cloned or declaration only). acc.specialized_routine is
//    preserved and symbol uses are updated so the routine name is unchanged.
//
// 4. Cleanup: erase from the host module the specialized device function
//    bodies and any nohost routine (host copy removed after move to device).
//
// Example:
// --------
// Before (after ACCRoutineLowering):
//   acc.routine @r_seq func(@foo) seq
//   func.func @foo() attributes {acc.specialized_routine = ...} { ... }
//
// After:
//   acc.routine @r_seq func(@foo) seq
//   gpu.module @acc_gpu_module {
//     gpu.func @foo() attributes {acc.specialized_routine = ...} { ... }
//   }
//   (host @foo erased)
//
// Requirements:
// -------------
// - Must run after `ACCRoutineLowering` pass  which ensures variants for all
//   levels of parallelism are created.
// - Uses OpenACCSupport: getOrCreateGPUModule, isValidSymbolUse, emitRemark,
//   emitNYI. If no custom implementation is registered, the default is used.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include <string>

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_ACCROUTINETOGPUFUNC
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

#define DEBUG_TYPE "acc-routine-to-gpu-func"

using namespace mlir;
using namespace mlir::acc;

namespace {

/// Create a gpu.func from a func.func by cloning the body.
static gpu::GPUFuncOp createGPUFuncFromFunc(OpBuilder &builder,
                                            func::FuncOp sourceFunc) {
  Location loc = sourceFunc.getLoc();
  StringRef name = sourceFunc.getName();
  FunctionType type = sourceFunc.getFunctionType();
  // Do not copy any attributes from the source; specialized_routine is set
  // later when applicable.
  gpu::GPUFuncOp gpuFunc =
      gpu::GPUFuncOp::create(builder, loc, name, type,
                             /*workgroupAttributions=*/TypeRange(),
                             /*privateAttributions=*/TypeRange(), /*attrs=*/{});

  Region &sourceBody = sourceFunc.getBody();
  Region &deviceBody = gpuFunc.getBody();
  Block &deviceEntryBlock = deviceBody.front();

  // Map source block arguments to the GPU func's entry block arguments (which
  // GPUFuncOp::create already created).
  IRMapping mapping;
  Block &sourceEntryBlock = sourceBody.front();
  for (auto [srcArg, destArg] : llvm::zip(sourceEntryBlock.getArguments(),
                                          deviceEntryBlock.getArguments()))
    mapping.map(srcArg, destArg);

  sourceBody.cloneInto(&deviceBody, mapping);

  // Replace func.return with gpu.return in the cloned blocks.
  gpuFunc.walk([](func::ReturnOp op) {
    OpBuilder replacer(op);
    gpu::ReturnOp gpuReturn = gpu::ReturnOp::create(replacer, op.getLoc());
    gpuReturn->setOperands(op.getOperands());
    op.erase();
  });

  // Splice the cloned entry block's operations into the GPU func's entry block
  // (cloneInto created a separate block for the cloned content), then remove
  // the now-empty cloned block.
  Block *clonedSourceEntry = mapping.lookup(&sourceEntryBlock);
  deviceEntryBlock.getOperations().splice(
      deviceEntryBlock.getOperations().end(),
      clonedSourceEntry->getOperations());
  clonedSourceEntry->erase();

  return gpuFunc;
}

using CloneCandidate = std::pair<func::FuncOp, RoutineOp>;

/// Collect materialized and bind routines; fill candidate func names and
/// materialized routine set. Emit remarks for materialized routines.
static void collectRoutineCandidates(
    ModuleOp mod, SymbolTable &symTab, acc::DeviceType deviceType,
    OpenACCSupport &accSupport,
    llvm::SmallSetVector<llvm::StringRef, 4> &funcsToCloneCandidates,
    llvm::SmallSetVector<RoutineOp, 4> &materializedAccRoutines,
    llvm::SmallSetVector<RoutineOp, 4> &bindAccRoutines) {
  auto isParallelRoutine = [deviceType](RoutineOp routineOp) {
    return routineOp.hasGang(deviceType) || routineOp.hasGang() ||
           routineOp.hasWorker(deviceType) || routineOp.hasWorker() ||
           routineOp.hasVector(deviceType) || routineOp.hasVector() ||
           routineOp.getGangDimValue(deviceType) || routineOp.getGangDimValue();
  };

  mod.walk([&](RoutineOp op) {
    if (op.getBindNameValue() || op.getBindNameValue(deviceType)) {
      bindAccRoutines.insert(op);
      return;
    }
    func::FuncOp callee =
        symTab.lookup<func::FuncOp>(op.getFuncName().getLeafReference());
    accSupport.emitRemark(
        callee ? callee.getOperation() : op.getOperation(),
        [&op, &isParallelRoutine]() {
          std::string msg = "Generating";
          if (op.getImplicitAttr())
            msg += " implicit";
          msg += " acc routine";
          if (!isParallelRoutine(op))
            msg += " seq";
          return msg;
        },
        DEBUG_TYPE);
    funcsToCloneCandidates.insert(op.getFuncName().getLeafReference());
    materializedAccRoutines.insert(op);
  });
}

/// Process calls in ACC routines: add valid callees to funcsToClone (for
/// declaration or clone). Returns failure() if any call is unsupported.
static LogicalResult processCallsInRoutines(
    SymbolTable &symTab, SymbolTable &gpuSymTab, OpenACCSupport &accSupport,
    const llvm::SmallSetVector<llvm::StringRef, 4> &funcsToCloneCandidates,
    const llvm::SmallSetVector<RoutineOp, 4> &materializedAccRoutines,
    llvm::SmallSetVector<CloneCandidate, 4> &funcsToClone) {
  LogicalResult callCheckResult = success();
  auto processCalls = [&](CallOpInterface callOp) {
    if (!callOp.getCallableForCallee())
      return;
    auto calleeSymbolRef =
        dyn_cast<SymbolRefAttr>(callOp.getCallableForCallee());
    if (!calleeSymbolRef)
      return;

    auto callee =
        symTab.lookup<func::FuncOp>(calleeSymbolRef.getLeafReference());
    if (!callee)
      return;

    if (gpuSymTab.lookup(callee.getName()))
      return;
    if (isAccRoutine(callee) || isSpecializedAccRoutine(callee))
      return;

    if (!accSupport.isValidSymbolUse(callOp.getOperation(), calleeSymbolRef)) {
      accSupport.emitNYI(callOp->getLoc(), "Unsupported call in acc routine");
      callCheckResult = failure();
      return;
    }
    funcsToClone.insert({callee, RoutineOp{}});
  };

  for (auto [funcName, accRoutine] :
       llvm::zip(funcsToCloneCandidates, materializedAccRoutines)) {
    func::FuncOp func = symTab.lookup<func::FuncOp>(funcName);
    if (!func)
      continue;
    if (!gpuSymTab.lookup(funcName))
      funcsToClone.insert({func, accRoutine});
    func.walk([&](CallOpInterface callOp) { processCalls(callOp); });
    if (failed(callCheckResult))
      return failure();
  }
  return success();
}

/// Clone each function in funcsToClone into the GPU module (declaration or
/// full body). Fix up symbol names and specialized_routine attr for ACC
/// routines.
static LogicalResult cloneFuncsToGPUModule(
    ModuleOp mod, OpenACCSupport &accSupport, SymbolTable &gpuSymTab,
    const llvm::SmallSetVector<CloneCandidate, 4> &funcsToClone) {
  MLIRContext *ctx = mod.getContext();
  OpBuilder builder(ctx);

  for (CloneCandidate candidate : funcsToClone) {
    func::FuncOp srcFunc = candidate.first;

    if (srcFunc.isDeclaration()) {
      Operation *cloned = srcFunc->clone();
      gpuSymTab.insert(cloned);
      continue;
    }

    gpu::GPUFuncOp deviceFuncOp = createGPUFuncFromFunc(builder, srcFunc);

    if (auto specRoutineAttr = srcFunc->getAttrOfType<SpecializedRoutineAttr>(
            getSpecializedRoutineAttrName())) {
      StringAttr funcName = specRoutineAttr.getFuncName();
      if (failed(SymbolTable::replaceAllSymbolUses(
              StringAttr::get(ctx, deviceFuncOp.getName()), funcName, mod))) {
        accSupport.emitNYI(deviceFuncOp.getLoc(),
                           "cannot replace symbol for acc routine");
        return failure();
      }
      deviceFuncOp->setAttr(SymbolTable::getSymbolAttrName(), funcName);
    }
    if (auto specAttr = srcFunc->getAttrOfType<SpecializedRoutineAttr>(
            getSpecializedRoutineAttrName()))
      deviceFuncOp->setAttr(getSpecializedRoutineAttrName(), specAttr);

    gpuSymTab.insert(deviceFuncOp);
  }
  return success();
}

/// Remove specialized device copies and nohost routines from the host module.
static void
cleanupHostModule(const llvm::SmallSetVector<CloneCandidate, 4> &funcsToClone) {
  for (CloneCandidate candidate : funcsToClone) {
    func::FuncOp funcCandidate = candidate.first;
    RoutineOp routineCandidate = candidate.second;
    if ((routineCandidate && routineCandidate.getNohost()) ||
        acc::isSpecializedAccRoutine(funcCandidate))
      funcCandidate.erase();
  }
}

class ACCRoutineToGPUFunc
    : public acc::impl::ACCRoutineToGPUFuncBase<ACCRoutineToGPUFunc> {
public:
  using acc::impl::ACCRoutineToGPUFuncBase<
      ACCRoutineToGPUFunc>::ACCRoutineToGPUFuncBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (mod.getOps<RoutineOp>().empty()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skipping ACCRoutineToGPUFunc - no acc.routine ops\n");
      return;
    }

    OpenACCSupport &accSupport = getAnalysis<OpenACCSupport>();
    std::optional<gpu::GPUModuleOp> gpuModOpt =
        accSupport.getOrCreateGPUModule(mod);
    if (!gpuModOpt) {
      accSupport.emitNYI(mod.getLoc(), "Failed to create GPU module");
      return signalPassFailure();
    }
    gpu::GPUModuleOp gpuMod = *gpuModOpt;

    SymbolTable symTab(mod);
    SymbolTable gpuSymTab(gpuMod);

    llvm::SmallSetVector<llvm::StringRef, 4> funcsToCloneCandidates;
    llvm::SmallSetVector<RoutineOp, 4> materializedAccRoutines;
    llvm::SmallSetVector<RoutineOp, 4> bindAccRoutines;

    collectRoutineCandidates(mod, symTab, this->deviceType, accSupport,
                             funcsToCloneCandidates, materializedAccRoutines,
                             bindAccRoutines);

    llvm::SmallSetVector<CloneCandidate, 4> funcsToClone;
    if (failed(processCallsInRoutines(symTab, gpuSymTab, accSupport,
                                      funcsToCloneCandidates,
                                      materializedAccRoutines, funcsToClone)))
      return signalPassFailure();

    if (failed(cloneFuncsToGPUModule(mod, accSupport, gpuSymTab, funcsToClone)))
      return signalPassFailure();

    cleanupHostModule(funcsToClone);
    for (RoutineOp bindOp : bindAccRoutines)
      bindOp.erase();
  }
};

} // namespace
