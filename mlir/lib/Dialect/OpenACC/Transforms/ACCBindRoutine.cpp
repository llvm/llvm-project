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
// For each function, walk operations that implement OffloadRegionOpInterface.
// For each call inside the offload region, if the callee is a function with
// an acc routine that has bind(name), replace the call to use the bound
// symbol.
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
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

#define DEBUG_TYPE "acc-bind-routine"

using namespace mlir;
using namespace mlir::acc;

namespace {

static RoutineOp getFirstAccRoutineOp(FunctionOpInterface funcOp,
                                      const SymbolTable &symTab) {
  if (isSpecializedAccRoutine(funcOp)) {
    auto attr = funcOp->getAttrOfType<SpecializedRoutineAttr>(
        getSpecializedRoutineAttrName());
    return symTab.lookup<RoutineOp>(attr.getRoutine().getLeafReference());
  }
  auto routineInfo =
      funcOp->getAttrOfType<RoutineInfoAttr>(getRoutineInfoAttrName());
  assert(routineInfo && "expected acc.routine_info for acc routine function");
  auto accRoutines = routineInfo.getAccRoutines();
  assert(!accRoutines.empty() && "expected at least one acc routine");
  return symTab.lookup<RoutineOp>(accRoutines[0].getLeafReference());
}

static bool isACCRoutineBindDefaultOrDeviceType(RoutineOp op,
                                                DeviceType deviceType) {
  if (!op.getBindIdName() && !op.getBindStrName())
    return false;
  return op.getBindNameValue().has_value() ||
         op.getBindNameValue(deviceType).has_value();
}

class ACCBindRoutine : public acc::impl::ACCBindRoutineBase<ACCBindRoutine> {
public:
  using acc::impl::ACCBindRoutineBase<ACCBindRoutine>::ACCBindRoutineBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return;

    SymbolTable symTab(module);
    auto cachedAnalysis =
        getCachedParentAnalysis<OpenACCSupport>(func->getParentOp());
    OpenACCSupport &accSupport =
        cachedAnalysis ? cachedAnalysis->get() : getAnalysis<OpenACCSupport>();

    bool failed = false;

    func.walk([&](acc::OffloadRegionOpInterface offload) {
      Region &region = offload.getOffloadRegion();
      region.walk([&](CallOpInterface callOp) {
        if (!callOp.getCallableForCallee())
          return;
        SymbolRefAttr calleeSymbolRef =
            dyn_cast<SymbolRefAttr>(callOp.getCallableForCallee());
        if (!calleeSymbolRef)
          return;

        FunctionOpInterface callee = symTab.lookup<FunctionOpInterface>(
            calleeSymbolRef.getLeafReference());
        if (!callee)
          return;

        if (!(isAccRoutine(callee) || isSpecializedAccRoutine(callee)))
          return;

        if (auto routineInfo = callee->getAttrOfType<RoutineInfoAttr>(
                getRoutineInfoAttrName())) {
          if (routineInfo.getAccRoutines().size() > 1) {
            (void)accSupport.emitNYI(callOp.getLoc(),
                                     "multiple `acc routine`s");
            failed = true;
            return;
          }
        }

        RoutineOp routine = getFirstAccRoutineOp(callee, symTab);
        if (!isACCRoutineBindDefaultOrDeviceType(routine, this->deviceType))
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
          calleeRef = FlatSymbolRefAttr::get(
              callOp.getContext(),
              std::get<StringAttr>(*bindNameOpt).getValue());
        }
        callOp.setCalleeFromCallable(calleeRef);
      });
    });

    if (failed)
      signalPassFailure();
  }
};

} // namespace
