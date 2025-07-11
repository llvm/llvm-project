//===- KernelOutlining.cpp - Implementation of GPU kernel outlining -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the GPU dialect kernel outlining pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Utils/GPUUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include <limits>

namespace mlir {
#define GEN_PASS_DEF_GPULAUNCHSINKINDEXCOMPUTATIONSPASS
#define GEN_PASS_DEF_GPUKERNELOUTLININGPASS
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

template <typename OpTy>
static void createForAllDimensions(OpBuilder &builder, Location loc,
                                   SmallVectorImpl<Value> &values) {
  for (auto dim : {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z})
    values.push_back(OpTy::create(builder, loc, builder.getIndexType(), dim));
}

/// Adds operations generating block/thread ids and grid/block dimensions at the
/// beginning of the `launchFuncOpBody` region. Add mapping from argument in
/// entry block of `launchOpBody`, to the corresponding result value of the
/// added operations.
static void injectGpuIndexOperations(Location loc, Region &launchFuncOpBody,
                                     Region &launchOpBody, IRMapping &map,
                                     bool hasCluster = false) {
  OpBuilder builder(loc->getContext());
  Block &firstBlock = launchOpBody.front();
  builder.setInsertionPointToStart(&launchFuncOpBody.front());
  SmallVector<Value> indexOps;
  // The order is important here, as it must match the order of the arguments
  createForAllDimensions<gpu::BlockIdOp>(builder, loc, indexOps);
  createForAllDimensions<gpu::ThreadIdOp>(builder, loc, indexOps);
  createForAllDimensions<gpu::GridDimOp>(builder, loc, indexOps);
  createForAllDimensions<gpu::BlockDimOp>(builder, loc, indexOps);
  if (hasCluster) {
    createForAllDimensions<gpu::ClusterIdOp>(builder, loc, indexOps);
    createForAllDimensions<gpu::ClusterDimOp>(builder, loc, indexOps);
  }
  // Replace the leading 12 function args with the respective thread/block index
  // operations. Iterate backwards since args are erased and indices change.
  for (const auto &indexOp : enumerate(indexOps))
    map.map(firstBlock.getArgument(indexOp.index()), indexOp.value());
}

/// Identifies operations that are beneficial to sink into kernels. These
/// operations may not have side-effects, as otherwise sinking (and hence
/// duplicating them) is not legal.
static bool isLikelyAnIndexComputation(Operation *op) {
  return matchPattern(op, m_Constant()) ||
         isa<memref::DimOp, arith::SelectOp, arith::CmpIOp>(op);
}

/// For a given operation `op`, computes whether it is beneficial to sink the
/// operation into the kernel. An operation can be sunk if doing so does not
/// introduce new kernel arguments. Whether a value is already available in the
/// kernel (and hence does not introduce new arguments) is checked by
/// querying `existingDependencies` and `availableValues`.
/// If an operand is not yet available, we recursively check whether it can be
/// made available by siking its defining op.
/// Operations that are indentified for sinking are added to `beneficiaryOps` in
/// the order they should appear in the kernel. Furthermore, `availableValues`
/// is updated with results that will be available after sinking the identified
/// ops.
static bool extractBeneficiaryOps(
    Operation *op, const SetVector<Value> &existingDependencies,
    SetVector<Operation *> &beneficiaryOps,
    llvm::SmallPtrSetImpl<Value> &availableValues,
    llvm::function_ref<bool(Operation *)> isSinkingBeneficiary) {
  if (beneficiaryOps.count(op))
    return true;

  if (!isSinkingBeneficiary(op))
    return false;

  for (Value operand : op->getOperands()) {
    // It is already visible in the kernel, keep going.
    if (availableValues.count(operand))
      continue;
    // Else check whether it can be made available via sinking or already is a
    // dependency.
    Operation *definingOp = operand.getDefiningOp();
    if ((!definingOp || !extractBeneficiaryOps(definingOp, existingDependencies,
                                               beneficiaryOps, availableValues,
                                               isSinkingBeneficiary)) &&
        !existingDependencies.count(operand))
      return false;
  }
  // We will sink the operation, mark its results as now available.
  beneficiaryOps.insert(op);
  for (Value result : op->getResults())
    availableValues.insert(result);
  return true;
}

LogicalResult mlir::sinkOperationsIntoLaunchOp(
    gpu::LaunchOp launchOp,
    llvm::function_ref<bool(Operation *)> isSinkingBeneficiary) {
  assert(isSinkingBeneficiary);
  Region &launchOpBody = launchOp.getBody();

  // Identify uses from values defined outside of the scope of the launch
  // operation.
  SetVector<Value> sinkCandidates;
  getUsedValuesDefinedAbove(launchOpBody, sinkCandidates);

  SetVector<Operation *> toBeSunk;
  llvm::SmallPtrSet<Value, 4> availableValues;
  for (Value operand : sinkCandidates) {
    Operation *operandOp = operand.getDefiningOp();
    if (!operandOp)
      continue;
    extractBeneficiaryOps(operandOp, sinkCandidates, toBeSunk, availableValues,
                          isSinkingBeneficiary);
  }

  // Insert operations so that the defs get cloned before uses.
  IRMapping map;
  OpBuilder builder(launchOpBody);
  for (Operation *op : toBeSunk) {
    Operation *clonedOp = builder.clone(*op, map);
    // Only replace uses within the launch op.
    for (auto pair : llvm::zip(op->getResults(), clonedOp->getResults()))
      replaceAllUsesInRegionWith(std::get<0>(pair), std::get<1>(pair),
                                 launchOp.getBody());
  }
  return success();
}

/// Return the provided KernelDim3 as an array of i32 constants if possible.
static DenseI32ArrayAttr maybeConstantDimsAttr(gpu::KernelDim3 dims) {
  SmallVector<int32_t, 3> constants;
  MLIRContext *ctx = dims.x.getContext();
  for (Value v : {dims.x, dims.y, dims.z}) {
    APInt constValue;
    if (!matchPattern(v, m_ConstantInt(&constValue)))
      return nullptr;
    // In the event someone called for a too-large block or grid dimension,
    // don't set bounds as it is likely to cause more confusing behavior.
    if (constValue.ugt(std::numeric_limits<uint32_t>::max()))
      return nullptr;
    constants.push_back(
        constValue.getLimitedValue(std::numeric_limits<uint32_t>::max()));
  }
  return DenseI32ArrayAttr::get(ctx, constants);
}

/// Outline the `gpu.launch` operation body into a kernel function. Replace
/// `gpu.terminator` operations by `gpu.return` in the generated function.
/// Set block and grid size bounds if known.
static gpu::GPUFuncOp outlineKernelFuncImpl(gpu::LaunchOp launchOp,
                                            StringRef kernelFnName,
                                            SetVector<Value> &operands) {
  Location loc = launchOp.getLoc();
  // Create a builder with no insertion point, insertion will happen separately
  // due to symbol table manipulation.
  OpBuilder builder(launchOp.getContext());
  Region &launchOpBody = launchOp.getBody();

  // Identify uses from values defined outside of the scope of the launch
  // operation.
  getUsedValuesDefinedAbove(launchOpBody, operands);

  // Create the gpu.func operation.
  SmallVector<Type, 4> kernelOperandTypes;
  kernelOperandTypes.reserve(operands.size());
  for (Value operand : operands) {
    kernelOperandTypes.push_back(operand.getType());
  }
  FunctionType type =
      FunctionType::get(launchOp.getContext(), kernelOperandTypes, {});
  auto outlinedFunc = gpu::GPUFuncOp::create(
      builder, loc, kernelFnName, type,
      TypeRange(ValueRange(launchOp.getWorkgroupAttributions())),
      TypeRange(ValueRange(launchOp.getPrivateAttributions())));
  outlinedFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                        builder.getUnitAttr());

  // If we can infer bounds on the grid and/or block sizes from the arguments
  // to the launch op, propagate them to the generated kernel. This is safe
  // because multiple launches with the same body are not deduplicated.
  if (auto blockBounds =
          maybeConstantDimsAttr(launchOp.getBlockSizeOperandValues()))
    outlinedFunc.setKnownBlockSizeAttr(blockBounds);
  if (auto gridBounds =
          maybeConstantDimsAttr(launchOp.getGridSizeOperandValues()))
    outlinedFunc.setKnownGridSizeAttr(gridBounds);

  IRMapping map;

  // Map the arguments corresponding to the launch parameters like blockIdx,
  // threadIdx, etc. If cluster is present, then we also generate clusterIdx and
  // clusterDim.
  Region &outlinedFuncBody = outlinedFunc.getBody();
  injectGpuIndexOperations(loc, outlinedFuncBody, launchOpBody, map,
                           launchOp.hasClusterSize());

  // Map memory attributions from the LaunOp op to the GPUFuncOp attributions.
  for (const auto &[launchArg, funcArg] :
       llvm::zip(launchOp.getWorkgroupAttributions(),
                 outlinedFunc.getWorkgroupAttributions()))
    map.map(launchArg, funcArg);
  for (const auto &[launchArg, funcArg] :
       llvm::zip(launchOp.getPrivateAttributions(),
                 outlinedFunc.getPrivateAttributions()))
    map.map(launchArg, funcArg);

  // Map arguments from gpu.launch region to the arguments of the gpu.func
  // operation.
  Block &entryBlock = outlinedFuncBody.front();
  for (const auto &operand : enumerate(operands))
    map.map(operand.value(), entryBlock.getArgument(operand.index()));

  // Clone the region of the gpu.launch operation into the gpu.func operation.
  launchOpBody.cloneInto(&outlinedFuncBody, map);

  // Replace the terminator op with returns.
  for (Block &block : launchOpBody) {
    Block *clonedBlock = map.lookup(&block);
    auto terminator = dyn_cast<gpu::TerminatorOp>(clonedBlock->getTerminator());
    if (!terminator)
      continue;
    OpBuilder replacer(terminator);
    gpu::ReturnOp::create(replacer, terminator->getLoc());
    terminator->erase();
  }

  // Splice now the entry block of the gpu.launch operation at the end of the
  // gpu.func entry block and erase the redundant block.
  Block *clonedLaunchOpEntry = map.lookup(&launchOpBody.front());
  entryBlock.getOperations().splice(entryBlock.getOperations().end(),
                                    clonedLaunchOpEntry->getOperations());
  clonedLaunchOpEntry->erase();

  return outlinedFunc;
}

gpu::GPUFuncOp mlir::outlineKernelFunc(gpu::LaunchOp launchOp,
                                       StringRef kernelFnName,
                                       llvm::SmallVectorImpl<Value> &operands) {
  DenseSet<Value> inputOperandSet;
  inputOperandSet.insert_range(operands);
  SetVector<Value> operandSet(llvm::from_range, operands);
  auto funcOp = outlineKernelFuncImpl(launchOp, kernelFnName, operandSet);
  for (auto operand : operandSet) {
    if (!inputOperandSet.count(operand))
      operands.push_back(operand);
  }
  return funcOp;
}

/// Replace `gpu.launch` operations with an `gpu.launch_func` operation
/// launching `kernelFunc`. The kernel func contains the body of the
/// `gpu.launch` with constant region arguments inlined.
static void convertToLaunchFuncOp(gpu::LaunchOp launchOp,
                                  gpu::GPUFuncOp kernelFunc,
                                  ValueRange operands) {
  OpBuilder builder(launchOp);
  // The launch op has an optional dynamic shared memory size. If it doesn't
  // exist, we use zero.
  Value asyncToken = launchOp.getAsyncToken();
  std::optional<gpu::KernelDim3> clusterSize =
      launchOp.getClusterSizeOperandValues();
  auto launchFunc = gpu::LaunchFuncOp::create(
      builder, launchOp.getLoc(), kernelFunc,
      launchOp.getGridSizeOperandValues(), launchOp.getBlockSizeOperandValues(),
      launchOp.getDynamicSharedMemorySize(), operands,
      asyncToken ? asyncToken.getType() : nullptr,
      launchOp.getAsyncDependencies(), clusterSize);
  launchOp.replaceAllUsesWith(launchFunc);
  launchOp.erase();
}

namespace {
/// Pass that moves ops which are likely an index computation into gpu.launch
/// body.
class GpuLaunchSinkIndexComputationsPass
    : public impl::GpuLaunchSinkIndexComputationsPassBase<
          GpuLaunchSinkIndexComputationsPass> {
public:
  void runOnOperation() override {
    Operation *op = getOperation();
    if (op->walk([](gpu::LaunchOp launch) {
            // Pull in instructions that can be sunk
            if (failed(sinkOperationsIntoLaunchOp(launch,
                                                  isLikelyAnIndexComputation)))
              return WalkResult::interrupt();

            return WalkResult::advance();
          }).wasInterrupted())
      signalPassFailure();
  }
};

/// Pass that moves the kernel of each LaunchOp into its separate nested module.
///
/// This pass moves the kernel code of each LaunchOp into a function created
/// inside a nested module. It also creates an external function of the same
/// name in the parent module.
///
/// The gpu.modules are intended to be compiled to a cubin blob independently in
/// a separate pass. The external functions can then be annotated with the
/// symbol of the cubin accessor function.
class GpuKernelOutliningPass
    : public impl::GpuKernelOutliningPassBase<GpuKernelOutliningPass> {
public:
  using Base::Base;

  LogicalResult initialize(MLIRContext *context) override {
    // Initialize the data layout specification from the data layout string.
    if (!dataLayoutStr.empty()) {
      Attribute resultAttr = mlir::parseAttribute(dataLayoutStr, context);
      if (!resultAttr)
        return failure();

      dataLayoutSpec = dyn_cast<DataLayoutSpecInterface>(resultAttr);
      if (!dataLayoutSpec)
        return failure();
    }

    return success();
  }

  void runOnOperation() override {
    SymbolTable symbolTable(getOperation());
    bool modified = false;
    for (auto func : getOperation().getOps<SymbolOpInterface>()) {
      // Insert just after the function.
      Block::iterator insertPt(func->getNextNode());
      auto funcWalkResult = func.walk([&](gpu::LaunchOp op) {
        SetVector<Value> operands;
        std::string kernelFnName;
        if (op.getKernelFunc()) {
          kernelFnName = op.getKernelFunc()->getRootReference().str();
        } else {
          kernelFnName =
              Twine(op->getParentOfType<SymbolOpInterface>().getName(),
                    "_kernel")
                  .str();
        }

        gpu::GPUFuncOp outlinedFunc =
            outlineKernelFuncImpl(op, kernelFnName, operands);

        // Create nested module and insert outlinedFunc. The module will
        // originally get the same name as the function, but may be renamed on
        // insertion into the parent module.
        auto kernelModule = createKernelModule(op, outlinedFunc, symbolTable);
        symbolTable.insert(kernelModule, insertPt);

        // Potentially changes signature, pulling in constants.
        convertToLaunchFuncOp(op, outlinedFunc, operands.getArrayRef());
        modified = true;
        return WalkResult::advance();
      });
      if (funcWalkResult.wasInterrupted())
        return signalPassFailure();
    }

    // If any new module was inserted in this module, annotate this module as
    // a container module.
    if (modified)
      getOperation()->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                              UnitAttr::get(&getContext()));
  }

private:
  /// Returns a gpu.module containing kernelFunc and all callees (recursive).
  gpu::GPUModuleOp createKernelModule(gpu::LaunchOp gpuLaunchOp,
                                      gpu::GPUFuncOp kernelFunc,
                                      const SymbolTable &parentSymbolTable) {
    // TODO: This code cannot use an OpBuilder because it must be inserted into
    // a SymbolTable by the caller. SymbolTable needs to be refactored to
    // prevent manual building of Ops with symbols in code using SymbolTables
    // and then this needs to use the OpBuilder.
    auto *context = getOperation().getContext();
    OpBuilder builder(context);
    std::string kernelModuleName;
    gpu::GPUModuleOp kernelModule;
    if (gpuLaunchOp.getKernelModule()) {
      kernelModuleName =
          gpuLaunchOp.getKernelModule()->getRootReference().str();
      kernelModule =
          parentSymbolTable.lookup<gpu::GPUModuleOp>(kernelModuleName);
    } else {
      kernelModuleName = kernelFunc.getName();
    }

    // Check if the module already exists in the symbol table
    if (!kernelModule) {
      // If not found, create a new GPU module
      kernelModule = gpu::GPUModuleOp::create(builder, kernelFunc.getLoc(),
                                              kernelModuleName);
    }

    // If a valid data layout spec was provided, attach it to the kernel module.
    // Otherwise, the default data layout will be used.
    if (dataLayoutSpec)
      kernelModule->setAttr(DLTIDialect::kDataLayoutAttrName, dataLayoutSpec);

    SymbolTable symbolTable(kernelModule);
    symbolTable.insert(kernelFunc);

    SmallVector<Operation *, 8> symbolDefWorklist = {kernelFunc};
    while (!symbolDefWorklist.empty()) {
      if (std::optional<SymbolTable::UseRange> symbolUses =
              SymbolTable::getSymbolUses(symbolDefWorklist.pop_back_val())) {
        for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
          StringRef symbolName =
              cast<FlatSymbolRefAttr>(symbolUse.getSymbolRef()).getValue();
          if (symbolTable.lookup(symbolName))
            continue;

          Operation *symbolDefClone =
              parentSymbolTable.lookup(symbolName)->clone();
          symbolDefWorklist.push_back(symbolDefClone);
          symbolTable.insert(symbolDefClone);
        }
      }
    }

    return kernelModule;
  }

  DataLayoutSpecInterface dataLayoutSpec;
};

} // namespace
