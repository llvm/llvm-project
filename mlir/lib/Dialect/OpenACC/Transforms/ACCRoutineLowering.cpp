//===- ACCRoutineLowering.cpp - Wrap ACC routines in compute_region -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass handles `acc routine` directive by creating specialized
// functions with appropriate parallelism information that can be used for
// eventual creation of device function.
//
// Overview:
// ---------
// For each acc.routine that is not bound by name, the pass creates a new
// function (the "device" copy) whose body is a single acc.compute_region
// containing a clone of the original (host) function body. Parallelism is
// expressed by one acc.par_width derived from the routine's clauses (seq,
// vector, worker, gang). The device copy created is simply a staging
// place for eventual move to device module level function.
//
// Transformations:
// ----------------
// 1. Device function: Same signature as the host; attributes copied except
//    acc.routine_info. The acc.specialized_routine attribute is set with the
//    routine symbol, par level, and original function name.
//
// 2. Body: One acc.par_width, one acc.compute_region that clones the host
//    body. Multi-block host bodies are wrapped in scf.execute_region inside
//    the compute_region.
//
// 3. Finalization: acc.routine's func_name is updated to the device function.
//    For nohost routines, all uses of the host symbol are replaced with the
//    device symbol and the host function is erased.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCParMapping.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_ACCROUTINELOWERING
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

#define DEBUG_TYPE "acc-routine-lowering"

using namespace mlir;
using namespace mlir::acc;

namespace {

/// Compute the ParLevel from an acc.routine op for specialization.
static ParLevel computeParLevel(RoutineOp routineOp, DeviceType deviceType) {
  auto gangDim = routineOp.getGangDimValue(deviceType);
  if (!gangDim)
    gangDim = routineOp.getGangDimValue();
  if (gangDim) {
    switch (*gangDim) {
    case 1:
      return ParLevel::gang_dim1;
    case 2:
      return ParLevel::gang_dim2;
    case 3:
      return ParLevel::gang_dim3;
    default:
      break;
    }
  }
  if (routineOp.hasGang(deviceType) || routineOp.hasGang())
    return ParLevel::gang_dim1;
  if (routineOp.hasWorker(deviceType) || routineOp.hasWorker())
    return ParLevel::worker;
  if (routineOp.hasVector(deviceType) || routineOp.hasVector())
    return ParLevel::vector;
  return ParLevel::seq;
}

/// Collect return operands from the function (first block with func.return).
static void getReturnValues(func::FuncOp func, SmallVectorImpl<Value> &result) {
  result.clear();
  for (Block &block : func.getBody().getBlocks()) {
    if (auto returnOp = dyn_cast<func::ReturnOp>(block.getTerminator())) {
      result.assign(returnOp.operand_begin(), returnOp.operand_end());
      break;
    }
  }
}

/// Create the device function with the same signature as the host, set
/// specialized_routine, and add a single block with the same block arguments.
static func::FuncOp createFunctionForDeviceStaging(func::FuncOp hostFunc,
                                                   RoutineOp routineOp,
                                                   ParLevel parLevel,
                                                   MLIRContext *ctx,
                                                   IRRewriter &rewriter) {
  Location loc = hostFunc.getLoc();
  FunctionType funcType = hostFunc.getFunctionType();
  func::FuncOp deviceFunc =
      func::FuncOp::create(rewriter, loc, hostFunc.getName(), funcType);
  deviceFunc->setAttrs(hostFunc->getAttrs());
  deviceFunc->removeAttr(getRoutineInfoAttrName());
  deviceFunc->setAttr(getSpecializedRoutineAttrName(),
                      SpecializedRoutineAttr::get(
                          ctx, SymbolRefAttr::get(ctx, routineOp.getSymName()),
                          ParLevelAttr::get(ctx, parLevel),
                          StringAttr::get(ctx, hostFunc.getName())));

  Block *sourceBlock = &hostFunc.getBody().front();
  Block *newBlock = rewriter.createBlock(&deviceFunc.getRegion());
  for (BlockArgument arg : sourceBlock->getArguments())
    newBlock->addArgument(arg.getType(), hostFunc.getLoc());

  return deviceFunc;
}

/// Fill the device function body: one acc.par_width, one acc.compute_region
/// (cloning the host body with inputArgsToMap), then func.return.
static LogicalResult
buildRoutineBody(func::FuncOp deviceFunc, func::FuncOp hostFunc,
                 ArrayRef<Value> funcReturnVals, ParLevel parLevel,
                 DefaultACCToGPUMappingPolicy &policy, IRRewriter &rewriter) {
  Block *newBlock = &deviceFunc.getBody().front();
  Block *sourceBlock = &hostFunc.getBody().front();
  Location loc = hostFunc.getLoc();
  MLIRContext *ctx = rewriter.getContext();

  rewriter.setInsertionPointToStart(newBlock);
  GPUParallelDimAttr parDim = policy.map(ctx, parLevel);
  Value parWidthVal = ParWidthOp::create(rewriter, loc, Value(), parDim);
  SmallVector<Value, 4> inputArgs(newBlock->getArguments().begin(),
                                  newBlock->getArguments().end());

  // Normally the region passed to buildComputeRegion is something in the
  // current function. Here we pass the body of the original (host) function as
  // an optimization to avoid cloning twice (once for a staged device copy and
  // again when creating the compute region). Since we clone only once, we must
  // also provide the original function's arguments so the mapping is correct
  // when cloning the body.
  ValueRange sourceArgsToMap = sourceBlock->getArguments();

  IRMapping mapping;
  rewriter.setInsertionPointAfter(parWidthVal.getDefiningOp());
  ComputeRegionOp computeRegion = buildComputeRegion(
      loc, {parWidthVal}, inputArgs, RoutineOp::getOperationName(),
      hostFunc.getBody(), rewriter, mapping,
      /*output=*/funcReturnVals, /*kernelFuncName=*/{},
      /*kernelModuleName=*/{}, /*stream=*/{}, sourceArgsToMap);
  if (!computeRegion)
    return failure();

  rewriter.setInsertionPointAfter(computeRegion);
  if (funcReturnVals.empty())
    func::ReturnOp::create(rewriter, loc);
  else
    func::ReturnOp::create(rewriter, loc, computeRegion.getResults());

  return success();
}

/// Update acc.routine refs and optionally erase host for nohost routines.
static LogicalResult finalizeRoutines(
    SmallVectorImpl<std::tuple<func::FuncOp, func::FuncOp, RoutineOp>>
        &accRoutineInfo,
    ModuleOp mod, MLIRContext *ctx) {
  for (auto &[hostFunc, deviceFunc, routineOp] : accRoutineInfo) {
    routineOp.setFuncNameAttr(SymbolRefAttr::get(ctx, deviceFunc.getName()));
    routineOp->moveBefore(deviceFunc);

    if (routineOp.getNohost()) {
      if (failed(SymbolTable::replaceAllSymbolUses(
              StringAttr::get(ctx, hostFunc.getName()),
              StringAttr::get(ctx, deviceFunc.getName()), mod))) {
        routineOp.emitError("cannot replace symbol uses for acc routine");
        return failure();
      }
      hostFunc->erase();
    }
  }
  return success();
}

class ACCRoutineLowering
    : public acc::impl::ACCRoutineLoweringBase<ACCRoutineLowering> {
public:
  using ACCRoutineLoweringBase::ACCRoutineLoweringBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (mod.getOps<RoutineOp>().empty()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skipping ACCRoutineLowering - no acc.routine ops\n");
      return;
    }

    SymbolTable symTab(mod);
    MLIRContext *ctx = mod.getContext();
    IRRewriter rewriter(ctx);
    DefaultACCToGPUMappingPolicy policy;

    // Tuple: host function, device function, routine operation
    SmallVector<std::tuple<func::FuncOp, func::FuncOp, RoutineOp>, 4>
        accRoutineInfo;

    for (RoutineOp routineOp : mod.getOps<RoutineOp>()) {
      if (routineOp.getBindNameValue() ||
          routineOp.getBindNameValue(deviceType))
        continue;

      func::FuncOp hostFunc = symTab.lookup<func::FuncOp>(
          routineOp.getFuncName().getLeafReference());
      if (!hostFunc) {
        routineOp.emitError("acc routine function not found in symbol table");
        return signalPassFailure();
      }
      if (hostFunc.isExternal())
        continue;

      SmallVector<Value, 4> funcReturnVals;
      getReturnValues(hostFunc, funcReturnVals);

      OpBuilder::InsertionGuard guard(rewriter);
      ParLevel parLevel = computeParLevel(routineOp, deviceType);
      func::FuncOp deviceFunc = createFunctionForDeviceStaging(
          hostFunc, routineOp, parLevel, ctx, rewriter);
      if (failed(buildRoutineBody(deviceFunc, hostFunc, funcReturnVals,
                                  parLevel, policy, rewriter)))
        return signalPassFailure();

      accRoutineInfo.push_back({hostFunc, deviceFunc, routineOp});
      symTab.insert(deviceFunc);
    }

    if (failed(finalizeRoutines(accRoutineInfo, mod, ctx)))
      return signalPassFailure();
  }
};

} // namespace
