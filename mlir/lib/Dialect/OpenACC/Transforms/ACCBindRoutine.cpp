//===- ACCBindRoutine.cpp - OpenACC bind routine transform ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The OpenACC `routine` directive may specify a `bind(name)` clause to
// associate the routine with a different symbol for device code. This pass
// finds calls inside offload regions that target such routines and rewrites the
// callee to the bound symbol.
//
// Overview:
// ---------
// For the current function, walk call operations inside offload regions, or
// gpu.func). If the callee is a function with an acc routine that has
// bind(name), replace the call to use the bound symbol.
//
// Requirements:
// -------------
// - OffloadRegionOpInterface: the pass walks operations implementing this
//   interface to discover offload regions (e.g. acc.compute_region) and
//   rewrites calls inside their getOffloadRegion().
// - CallOpInterface with working setCalleeFromCallable: call operations
//   must implement CallOpInterface and setCalleeFromCallable so the pass
//   can rewrite the callee to the symbol without invalidating the call.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_ACCBINDROUTINE
#define GEN_PASS_DEF_ACCMATERIALIZEROUTINEBINDTARGETS
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

#define DEBUG_TYPE "acc-bind-routine"

using namespace mlir;
using namespace mlir::acc;

namespace {

static RoutineOp getFirstAccRoutineOp(FunctionOpInterface funcOp,
                                      const SymbolTable &fallbackSymTab) {
  SymbolRefAttr routineRef;
  if (isSpecializedAccRoutine(funcOp)) {
    auto attr = funcOp->getDiscardableAttrOfType<SpecializedRoutineAttr>(
        getSpecializedRoutineAttrName());
    routineRef = attr.getRoutine();
  } else {
    auto routineInfo = funcOp->getDiscardableAttrOfType<RoutineInfoAttr>(
        getRoutineInfoAttrName());
    assert(routineInfo && "expected acc.routine_info for acc routine function");
    auto accRoutines = routineInfo.getAccRoutines();
    assert(!accRoutines.empty() && "expected at least one acc routine");
    routineRef = accRoutines[0];
  }

  RoutineOp routine =
      SymbolTable::lookupNearestSymbolFrom<RoutineOp>(funcOp, routineRef);
  if (routine)
    return routine;
  return fallbackSymTab.lookup<RoutineOp>(routineRef.getLeafReference());
}

static bool isACCRoutineBindDefaultOrDeviceType(RoutineOp op,
                                                DeviceType deviceType) {
  if (!op.getBindIdName() && !op.getBindStrName())
    return false;
  return op.getBindNameValue().has_value() ||
         op.getBindNameValue(deviceType).has_value();
}

class ACCMaterializeRoutineBindTargets
    : public acc::impl::ACCMaterializeRoutineBindTargetsBase<
          ACCMaterializeRoutineBindTargets> {
  using Base = acc::impl::ACCMaterializeRoutineBindTargetsBase<
      ACCMaterializeRoutineBindTargets>;

public:
  using Base::Base;

  void runOnOperation() override {
    Operation *symbolTableOp = getOperation();
    if (!symbolTableOp->hasTrait<OpTrait::SymbolTable>()) {
      symbolTableOp->emitError(
          "root operation must have the SymbolTable trait");
      signalPassFailure();
      return;
    }

    ModuleOp mod = dyn_cast<ModuleOp>(symbolTableOp);
    if (!mod)
      mod = symbolTableOp->getParentOfType<ModuleOp>();
    if (!mod) {
      symbolTableOp->emitError(
          "root operation must be a builtin.module or nested in one");
      signalPassFailure();
      return;
    }

    SymbolTable outerSymTab(mod);
    SymbolTable insertSymTab(symbolTableOp);
    Block *insertBlock = &symbolTableOp->getRegion(0).front();

    symbolTableOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op != symbolTableOp && op->hasTrait<OpTrait::SymbolTable>())
        return WalkResult::skip();

      auto callOp = dyn_cast<CallOpInterface>(op);
      if (!callOp || !callOp.getCallableForCallee())
        return WalkResult::advance();
      if (!callOp->getParentOfType<OffloadRegionOpInterface>() &&
          !callOp->getParentOfType<gpu::GPUFuncOp>())
        return WalkResult::advance();

      auto calleeSymbolRef =
          dyn_cast<SymbolRefAttr>(callOp.getCallableForCallee());
      if (!calleeSymbolRef)
        return WalkResult::advance();
      FunctionOpInterface callee =
          SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(
              callOp, calleeSymbolRef);
      if (!callee)
        callee = outerSymTab.lookup<FunctionOpInterface>(
            calleeSymbolRef.getLeafReference());
      if (!callee || !(isAccRoutine(callee) || isSpecializedAccRoutine(callee)))
        return WalkResult::advance();

      if (auto routineInfo = callee->getDiscardableAttrOfType<RoutineInfoAttr>(
              getRoutineInfoAttrName()))
        if (routineInfo.getAccRoutines().size() > 1)
          return WalkResult::advance();

      RoutineOp routine = getFirstAccRoutineOp(callee, outerSymTab);
      if (!routine ||
          !isACCRoutineBindDefaultOrDeviceType(routine, this->deviceType))
        return WalkResult::advance();

      auto bindNameOpt = routine.getBindNameValue(this->deviceType);
      if (!bindNameOpt)
        bindNameOpt = routine.getBindNameValue();
      if (!bindNameOpt || !std::holds_alternative<StringAttr>(*bindNameOpt))
        return WalkResult::advance();

      StringRef bindName = std::get<StringAttr>(*bindNameOpt).getValue();
      if (insertSymTab.lookup(bindName))
        return WalkResult::advance();

      OpBuilder builder(mod.getContext());
      builder.setInsertionPointToEnd(insertBlock);
      auto funcType = cast<FunctionType>(callee.getFunctionType());
      func::FuncOp bindFunc =
          func::FuncOp::create(builder, callee.getLoc(), bindName, funcType);
      bindFunc.setPrivate();
      insertSymTab.insert(bindFunc);
      return WalkResult::advance();
    });
  }
};

class ACCBindRoutine : public acc::impl::ACCBindRoutineBase<ACCBindRoutine> {
public:
  using acc::impl::ACCBindRoutineBase<ACCBindRoutine>::ACCBindRoutineBase;

  void runOnOperation() override {
    FunctionOpInterface func = getOperation();
    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return;

    auto cachedAnalysis =
        getCachedParentAnalysis<OpenACCSupport>(func->getParentOp());
    OpenACCSupport &accSupport =
        cachedAnalysis ? cachedAnalysis->get() : getAnalysis<OpenACCSupport>();
    SymbolTable outerSymTab(module);
    bool failed = false;

    func.walk([&](CallOpInterface callOp) {
      if (!callOp.getCallableForCallee())
        return;
      if (!callOp->getParentOfType<OffloadRegionOpInterface>() &&
          !callOp->getParentOfType<gpu::GPUFuncOp>())
        return;
      SymbolRefAttr calleeSymbolRef =
          dyn_cast<SymbolRefAttr>(callOp.getCallableForCallee());
      if (!calleeSymbolRef)
        return;
      FunctionOpInterface callee =
          SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(
              callOp, calleeSymbolRef);
      if (!callee)
        callee = outerSymTab.lookup<FunctionOpInterface>(
            calleeSymbolRef.getLeafReference());
      if (!callee)
        return;

      if (!(isAccRoutine(callee) || isSpecializedAccRoutine(callee)))
        return;

      if (auto routineInfo = callee->getDiscardableAttrOfType<RoutineInfoAttr>(
              getRoutineInfoAttrName())) {
        if (routineInfo.getAccRoutines().size() > 1) {
          (void)accSupport.emitNYI(callOp.getLoc(), "multiple `acc routine`s");
          failed = true;
          return;
        }
      }

      RoutineOp routine = getFirstAccRoutineOp(callee, outerSymTab);
      if (!routine ||
          !isACCRoutineBindDefaultOrDeviceType(routine, this->deviceType))
        return;

      auto bindNameOpt = routine.getBindNameValue(this->deviceType);
      if (!bindNameOpt)
        bindNameOpt = routine.getBindNameValue();
      if (!bindNameOpt)
        return;
      SymbolRefAttr calleeRef;
      if (auto *symRef = std::get_if<SymbolRefAttr>(&*bindNameOpt)) {
        calleeRef = *symRef;
      } else {
        StringRef bindName = std::get<StringAttr>(*bindNameOpt).getValue();
        auto gpuMod = func->getParentOfType<gpu::GPUModuleOp>();
        Operation *symbolTableOp =
            gpuMod ? gpuMod.getOperation() : module.getOperation();
        SymbolTable symbolTable(symbolTableOp);
        if (!symbolTable.lookup(bindName)) {
          callOp.emitError(
              "string-bound ACC routine target was not materialized");
          failed = true;
          return;
        }
        calleeRef = FlatSymbolRefAttr::get(callOp.getContext(), bindName);
      }
      callOp.setCalleeFromCallable(calleeRef);
    });

    if (failed)
      signalPassFailure();
  }
};

} // namespace
