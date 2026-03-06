//===- ACCImplicitRoutine.cpp - OpenACC Implicit Routine Transform -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements the implicit rules described in OpenACC specification
// for `Routine Directive` (OpenACC 3.4 spec, section 2.15.1).
//
// "If no explicit routine directive applies to a procedure whose definition
// appears in the program unit being compiled, then the implementation applies
// an implicit routine directive to that procedure if any of the following
// conditions holds:
// - The procedure is called or its address is accessed in a compute region."
//
// The specification further states:
// "When the implementation applies an implicit routine directive to a
// procedure, it must recursively apply implicit routine directives to other
// procedures for which the above rules specify relevant dependencies. Such
// dependencies can form a cycle, so the implementation must take care to avoid
// infinite recursion."
//
// This pass implements these requirements by:
// 1. Walking through all OpenACC compute constructs and functions already
//    marked with `acc routine` in the module and identifying function calls
//    within these regions.
// 2. Creating implicit `acc.routine` operations for functions that don't
//    already have routine declarations.
// 3. Recursively walking through all existing `acc routine` and creating
//    implicit routine operations for function calls within these routines,
//    while avoiding infinite recursion through proper tracking.
//
// Requirements:
// -------------
// To use this pass in a pipeline, the following requirements must be met:
//
// 1. Operation Interface Implementation: Operations that define functions
//    or call functions should implement `mlir::FunctionOpInterface` and
//    `mlir::CallOpInterface` respectively.
//
// 2. Analysis Registration (Optional): If custom behavior is needed for
//    determining if a symbol use is valid within GPU regions, the dialect
//    should pre-register the `acc::OpenACCSupport` analysis.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Transforms/Passes.h"

#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include <queue>

#define DEBUG_TYPE "acc-implicit-routine"

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_ACCIMPLICITROUTINE
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

namespace {

using namespace mlir;

class ACCImplicitRoutine
    : public acc::impl::ACCImplicitRoutineBase<ACCImplicitRoutine> {
private:
  unsigned routineCounter = 0;
  static constexpr llvm::StringRef accRoutinePrefix = "acc_routine_";

  // Count existing routine operations and update counter
  void initRoutineCounter(ModuleOp module) {
    module.walk([&](acc::RoutineOp routineOp) { routineCounter++; });
  }

  // Check if routine has a default bind clause or a device-type specific bind
  // clause. Returns true if `acc routine` has a default bind clause or
  // a device-type specific bind clause.
  bool isACCRoutineBindDefaultOrDeviceType(acc::RoutineOp op,
                                           acc::DeviceType deviceType) {
    // Fast check to avoid device-type specific lookups.
    if (!op.getBindIdName() && !op.getBindStrName())
      return false;
    return op.getBindNameValue().has_value() ||
           op.getBindNameValue(deviceType).has_value();
  }

  // Generate a unique name for the routine and create the routine operation
  acc::RoutineOp createRoutineOp(OpBuilder &builder, Location loc,
                                 FunctionOpInterface &callee) {
    std::string routineName =
        (accRoutinePrefix + std::to_string(routineCounter++)).str();
    auto routineOp = acc::RoutineOp::create(
        builder, loc,
        /* sym_name=*/builder.getStringAttr(routineName),
        /* func_name=*/
        mlir::SymbolRefAttr::get(builder.getContext(),
                                 builder.getStringAttr(callee.getName())),
        /* bindIdName=*/nullptr,
        /* bindStrName=*/nullptr,
        /* bindIdNameDeviceType=*/nullptr,
        /* bindStrNameDeviceType=*/nullptr,
        /* worker=*/nullptr,
        /* vector=*/nullptr,
        /* seq=*/nullptr,
        /* nohost=*/nullptr,
        /* implicit=*/builder.getUnitAttr(),
        /* gang=*/nullptr,
        /* gangDim=*/nullptr,
        /* gangDimDeviceType=*/nullptr);

    // Assert that the callee does not already have routine info attribute
    assert(!callee->hasAttr(acc::getRoutineInfoAttrName()) &&
           "function is already associated with a routine");

    callee->setAttr(
        acc::getRoutineInfoAttrName(),
        mlir::acc::RoutineInfoAttr::get(
            builder.getContext(),
            {mlir::SymbolRefAttr::get(builder.getContext(),
                                      builder.getStringAttr(routineName))}));
    return routineOp;
  }

  // Used to walk through a compute region looking for function calls.
  void
  implicitRoutineForCallsInComputeRegions(Operation *op, SymbolTable &symTab,
                                          mlir::OpBuilder &builder,
                                          acc::OpenACCSupport &accSupport) {
    op->walk([&](CallOpInterface callOp) {
      if (!callOp.getCallableForCallee())
        return;

      auto calleeSymbolRef =
          dyn_cast<SymbolRefAttr>(callOp.getCallableForCallee());
      // When call is done through ssa value, the callee is not a symbol.
      // Skip it because we don't know the call target.
      if (!calleeSymbolRef)
        return;

      auto callee = symTab.lookup<FunctionOpInterface>(
          calleeSymbolRef.getLeafReference().str());
      // If the callee does not exist or is already a valid symbol for GPU
      // regions, skip it

      assert(callee && "callee function must be found in symbol table");
      if (accSupport.isValidSymbolUse(callOp.getOperation(), calleeSymbolRef))
        return;
      builder.setInsertionPoint(callee);
      createRoutineOp(builder, callee.getLoc(), callee);
    });
  }

  // Recursively handle calls within a routine operation
  void implicitRoutineForCallsInRoutine(acc::RoutineOp routineOp,
                                        mlir::OpBuilder &builder,
                                        acc::OpenACCSupport &accSupport,
                                        acc::DeviceType targetDeviceType) {
    // When bind clause is used, it means that the target is different than the
    // function to which the `acc routine` is used with. Skip this case to
    // avoid implicitly recursively marking calls that would not end up on
    // device.
    if (isACCRoutineBindDefaultOrDeviceType(routineOp, targetDeviceType))
      return;

    SymbolTable symTab(routineOp->getParentOfType<ModuleOp>());
    std::queue<acc::RoutineOp> routineQueue;
    routineQueue.push(routineOp);
    while (!routineQueue.empty()) {
      auto currentRoutine = routineQueue.front();
      routineQueue.pop();
      auto func = symTab.lookup<FunctionOpInterface>(
          currentRoutine.getFuncName().getLeafReference());
      func.walk([&](CallOpInterface callOp) {
        if (!callOp.getCallableForCallee())
          return;

        auto calleeSymbolRef =
            dyn_cast<SymbolRefAttr>(callOp.getCallableForCallee());
        // When call is done through ssa value, the callee is not a symbol.
        // Skip it because we don't know the call target.
        if (!calleeSymbolRef)
          return;

        auto callee = symTab.lookup<FunctionOpInterface>(
            calleeSymbolRef.getLeafReference().str());
        // If the callee does not exist or is already a valid symbol for GPU
        // regions, skip it
        assert(callee && "callee function must be found in symbol table");
        if (accSupport.isValidSymbolUse(callOp.getOperation(), calleeSymbolRef))
          return;
        builder.setInsertionPoint(callee);
        auto newRoutineOp = createRoutineOp(builder, callee.getLoc(), callee);
        routineQueue.push(newRoutineOp);
      });
    }
  }

public:
  using ACCImplicitRoutineBase<ACCImplicitRoutine>::ACCImplicitRoutineBase;

  void runOnOperation() override {
    auto module = getOperation();
    mlir::OpBuilder builder(module.getContext());
    SymbolTable symTab(module);
    initRoutineCounter(module);

    acc::OpenACCSupport &accSupport = getAnalysis<acc::OpenACCSupport>();

    // Handle compute regions
    module.walk([&](Operation *op) {
      if (isa<ACC_COMPUTE_CONSTRUCT_OPS>(op))
        implicitRoutineForCallsInComputeRegions(op, symTab, builder,
                                                accSupport);
    });

    // Use the device type option from the pass options.
    acc::DeviceType targetDeviceType = deviceType;

    // Handle existing routines
    module.walk([&](acc::RoutineOp routineOp) {
      implicitRoutineForCallsInRoutine(routineOp, builder, accSupport,
                                       targetDeviceType);
    });
  }
};

} // namespace
