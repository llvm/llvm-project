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

#include "PassDetail.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/GPU/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;

template <typename OpTy>
static void createForAllDimensions(OpBuilder &builder, Location loc,
                                   SmallVectorImpl<Value> &values) {
  for (StringRef dim : {"x", "y", "z"}) {
    Value v = builder.create<OpTy>(loc, builder.getIndexType(),
                                   builder.getStringAttr(dim));
    values.push_back(v);
  }
}

// Add operations generating block/thread ids and grid/block dimensions at the
// beginning of the `launchFuncOpBody` region. Add mapping from argument in
// entry block of `launchOpBody`, to the corresponding result value of the added
// operations.
static void injectGpuIndexOperations(Location loc, Region &launchFuncOpBody,
                                     Region &launchOpBody,
                                     BlockAndValueMapping &map) {
  OpBuilder builder(loc->getContext());
  Block &firstBlock = launchOpBody.front();
  builder.setInsertionPointToStart(&launchFuncOpBody.front());
  SmallVector<Value, 12> indexOps;
  createForAllDimensions<gpu::BlockIdOp>(builder, loc, indexOps);
  createForAllDimensions<gpu::ThreadIdOp>(builder, loc, indexOps);
  createForAllDimensions<gpu::GridDimOp>(builder, loc, indexOps);
  createForAllDimensions<gpu::BlockDimOp>(builder, loc, indexOps);
  // Replace the leading 12 function args with the respective thread/block index
  // operations. Iterate backwards since args are erased and indices change.
  for (auto indexOp : enumerate(indexOps))
    map.map(firstBlock.getArgument(indexOp.index()), indexOp.value());
}

static bool isSinkingBeneficiary(Operation *op) {
  return isa<ConstantOp, DimOp>(op);
}

LogicalResult mlir::sinkOperationsIntoLaunchOp(gpu::LaunchOp launchOp) {
  Region &launchOpBody = launchOp.body();

  // Identify uses from values defined outside of the scope of the launch
  // operation.
  llvm::SetVector<Value> sinkCandidates;
  getUsedValuesDefinedAbove(launchOpBody, sinkCandidates);

  llvm::SetVector<Value> sunkValues;
  llvm::SetVector<Operation *> sunkOperations;
  for (Value operand : sinkCandidates) {
    Operation *operandOp = operand.getDefiningOp();
    if (!operandOp || !isSinkingBeneficiary(operandOp))
      continue;
    // Only sink operations that do not create new sinkCandidates.
    if (!llvm::all_of(operandOp->getOperands(), [&sinkCandidates](Value value) {
          return sinkCandidates.count(value);
        }))
      continue;
    sunkValues.insert(operand);
    sunkOperations.insert(operandOp);
  }

  // Insert operations so that the defs get cloned before uses.
  BlockAndValueMapping map;
  OpBuilder builder(launchOpBody);
  DenseSet<Operation *> processed;
  SmallVector<Operation *, 2> clonedOps;
  while (processed.size() != sunkOperations.size()) {
    auto startSize = processed.size();
    for (Operation *sunkOperation : sunkOperations) {
      if (processed.count(sunkOperation))
        continue;

      // Operation cant be cloned yet if any of its operands is also being sunk,
      // but isnt cloned yet.
      if (llvm::any_of(
              sunkOperation->getOperands(), [&sunkValues, &map](Value value) {
                return sunkValues.count(value) && !map.lookupOrNull(value);
              }))
        continue;

      Operation *clonedOp = builder.clone(*sunkOperation, map);
      // Only replace uses within the launch op.
      for (auto result : llvm::enumerate(sunkOperation->getResults())) {
        auto replacement = clonedOp->getResult(result.index());
        for (auto &use : llvm::make_early_inc_range(result.value().getUses()))
          if (use.getOwner()->getParentOfType<gpu::LaunchOp>() == launchOp)
            use.set(replacement);
      }
      processed.insert(sunkOperation);
    }
    if (startSize == processed.size())
      return launchOp.emitError(
          "found illegal cyclic dependency between operations while sinking");
  }
  return success();
}

// Outline the `gpu.launch` operation body into a kernel function. Replace
// `gpu.terminator` operations by `gpu.return` in the generated function.
static gpu::GPUFuncOp outlineKernelFuncImpl(gpu::LaunchOp launchOp,
                                            StringRef kernelFnName,
                                            llvm::SetVector<Value> &operands) {
  Location loc = launchOp.getLoc();
  // Create a builder with no insertion point, insertion will happen separately
  // due to symbol table manipulation.
  OpBuilder builder(launchOp.getContext());
  Region &launchOpBody = launchOp.body();

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
      FunctionType::get(kernelOperandTypes, {}, launchOp.getContext());
  auto outlinedFunc = builder.create<gpu::GPUFuncOp>(loc, kernelFnName, type);
  outlinedFunc.setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       builder.getUnitAttr());
  BlockAndValueMapping map;

  // Map the arguments corresponding to the launch parameters like blockIdx,
  // threadIdx, etc.
  Region &outlinedFuncBody = outlinedFunc.body();
  injectGpuIndexOperations(loc, outlinedFuncBody, launchOpBody, map);

  // Map arguments from gpu.launch region to the arguments of the gpu.func
  // operation.
  Block &entryBlock = outlinedFuncBody.front();
  for (auto operand : enumerate(operands))
    map.map(operand.value(), entryBlock.getArgument(operand.index()));

  // Clone the region of the gpu.launch operation into the gpu.func operation.
  // TODO(ravishankarm): If cloneInto can be modified such that if a mapping for
  // a block exists, that block will be used to clone operations into (at the
  // end of the block), instead of creating a new block, this would be much
  // cleaner.
  launchOpBody.cloneInto(&outlinedFuncBody, map);

  // Branch from entry of the gpu.func operation to the block that is cloned
  // from the entry block of the gpu.launch operation.
  Block &launchOpEntry = launchOpBody.front();
  Block *clonedLaunchOpEntry = map.lookup(&launchOpEntry);
  builder.setInsertionPointToEnd(&entryBlock);
  builder.create<BranchOp>(loc, clonedLaunchOpEntry);

  outlinedFunc.walk([](gpu::TerminatorOp op) {
    OpBuilder replacer(op);
    replacer.create<gpu::ReturnOp>(op.getLoc());
    op.erase();
  });
  return outlinedFunc;
}

gpu::GPUFuncOp mlir::outlineKernelFunc(gpu::LaunchOp launchOp,
                                       StringRef kernelFnName,
                                       llvm::SmallVectorImpl<Value> &operands) {
  DenseSet<Value> inputOperandSet;
  inputOperandSet.insert(operands.begin(), operands.end());
  llvm::SetVector<Value> operandSet(operands.begin(), operands.end());
  auto funcOp = outlineKernelFuncImpl(launchOp, kernelFnName, operandSet);
  for (auto operand : operandSet) {
    if (!inputOperandSet.count(operand))
      operands.push_back(operand);
  }
  return funcOp;
}

// Replace `gpu.launch` operations with an `gpu.launch_func` operation launching
// `kernelFunc`. The kernel func contains the body of the `gpu.launch` with
// constant region arguments inlined.
static void convertToLaunchFuncOp(gpu::LaunchOp launchOp,
                                  gpu::GPUFuncOp kernelFunc,
                                  ValueRange operands) {
  OpBuilder builder(launchOp);
  builder.create<gpu::LaunchFuncOp>(
      launchOp.getLoc(), kernelFunc, launchOp.getGridSizeOperandValues(),
      launchOp.getBlockSizeOperandValues(), operands);
  launchOp.erase();
}

namespace {
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
    : public GpuKernelOutliningBase<GpuKernelOutliningPass> {
public:
  void runOnOperation() override {
    SymbolTable symbolTable(getOperation());
    bool modified = false;
    for (auto func : getOperation().getOps<FuncOp>()) {
      // Insert just after the function.
      Block::iterator insertPt(func.getOperation()->getNextNode());
      auto funcWalkResult = func.walk([&](gpu::LaunchOp op) {
        llvm::SetVector<Value> operands;
        std::string kernelFnName =
            Twine(op.getParentOfType<FuncOp>().getName(), "_kernel").str();

        // Pull in instructions that can be sunk
        if (failed(sinkOperationsIntoLaunchOp(op)))
          return WalkResult::interrupt();
        gpu::GPUFuncOp outlinedFunc =
            outlineKernelFuncImpl(op, kernelFnName, operands);

        // Create nested module and insert outlinedFunc. The module will
        // originally get the same name as the function, but may be renamed on
        // insertion into the parent module.
        auto kernelModule = createKernelModule(outlinedFunc, symbolTable);
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
      getOperation().setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                             UnitAttr::get(&getContext()));
  }

private:
  // Returns a gpu.module containing kernelFunc and all callees (recursive).
  gpu::GPUModuleOp createKernelModule(gpu::GPUFuncOp kernelFunc,
                                      const SymbolTable &parentSymbolTable) {
    // TODO: This code cannot use an OpBuilder because it must be inserted into
    // a SymbolTable by the caller. SymbolTable needs to be refactored to
    // prevent manual building of Ops with symbols in code using SymbolTables
    // and then this needs to use the OpBuilder.
    auto context = getOperation().getContext();
    OpBuilder builder(context);
    OperationState state(kernelFunc.getLoc(),
                         gpu::GPUModuleOp::getOperationName());
    gpu::GPUModuleOp::build(builder, state, kernelFunc.getName());
    auto kernelModule = cast<gpu::GPUModuleOp>(Operation::create(state));
    SymbolTable symbolTable(kernelModule);
    symbolTable.insert(kernelFunc);

    SmallVector<Operation *, 8> symbolDefWorklist = {kernelFunc};
    while (!symbolDefWorklist.empty()) {
      if (Optional<SymbolTable::UseRange> symbolUses =
              SymbolTable::getSymbolUses(symbolDefWorklist.pop_back_val())) {
        for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
          StringRef symbolName =
              symbolUse.getSymbolRef().cast<FlatSymbolRefAttr>().getValue();
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
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createGpuKernelOutliningPass() {
  return std::make_unique<GpuKernelOutliningPass>();
}
