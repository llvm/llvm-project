//===- TargetOutlining.cpp - Implementation of Target kernel outlining ----===//
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

#include "mlir/Dialect/OpenMP/Transforms/Passes.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/Support/FileSystem.h"

namespace mlir {
namespace omp {
#define GEN_PASS_DEF_OMPTARGETGPUOUTLINING
#include "mlir/Dialect/OpenMP/Transforms/Passes.h.inc"
} // namespace omp
} // namespace mlir

using namespace mlir;
using namespace mlir::omp;

namespace {
/// Pass that moves the kernel of each LaunchOp into its separate nested module.
///
/// This pass moves the kernel code of each LaunchOp into a function created
/// inside a nested module. It also creates an external function of the same
/// name in the parent module.
class OMPTargetGPUOutlining
    : public omp::impl::OMPTargetGPUOutliningBase<OMPTargetGPUOutlining> {
public:
  using Base::Base;
  void runOnOperation() override;

private:
  struct TargetOpInfo {
    mutable TargetOp op;
    StringRef parentName;
    uint32_t functionId = 0;
    uint32_t opId = 0;
    uint32_t uniqueId = 0;
  };
  // Create the `TargetRegionEntryInfoAttr` for the TargetOp.
  TargetRegionEntryInfoAttr getTargetEntryInfo(OpBuilder builder,
                                               const TargetOpInfo &opInfo,
                                               llvm::StringRef moduleName);
  // Outline the TargetOp to the GPU module.
  LogicalResult outlineTargetOp(gpu::GPUModuleOp module,
                                SymbolTable &devSymbolTable,
                                SymbolTable &hostSymbolTable,
                                const TargetOpInfo &opInfo);
  // Add the referenced declare target symbols to the module.
  LogicalResult cloneDeclareTarget(OpBuilder builder, TargetOp op,
                                   SymbolTable &devSymbolTable,
                                   SymbolTable &hostSymbolTable,
                                   StringRef moduleName);
};
} // namespace

void OMPTargetGPUOutlining::runOnOperation() {
  SymbolTable hostSymbolTable(getOperation());
  // Collect all `omp.target` ops
  SmallVector<TargetOpInfo> targetOps;
  uint32_t uniqueId = 0;
  uint32_t functionId = 0;
  for (auto func : getOperation().getOps<FunctionOpInterface>()) {
    uint32_t opId = 0;
    func.walk([&](omp::TargetOp op) {
      targetOps.push_back({op, func.getName(), functionId, opId++, uniqueId++});
      return WalkResult::advance();
    });
    functionId++;
  }
  // Return early if there's no work to do
  if (targetOps.empty())
    return;
  // Create the GPU module
  OpBuilder builder(getOperation().getContext());
  auto devModule = builder.create<gpu::GPUModuleOp>(
      getOperation().getLoc(), moduleName, nullptr,
      builder.getAttr<gpu::OffloadEmbeddingAttr>(gpu::OffloadKind::OpenMP));
  hostSymbolTable.insert(devModule, getOperation().getBody()->begin());
  if (auto moduleIface =
          dyn_cast<OffloadModuleInterface>(devModule.getOperation())) {
    moduleIface.setIsGPU(true);
    moduleIface.setIsTargetDevice(true);
  }
  getOperation()->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                          UnitAttr::get(&getContext()));
  SymbolTable devSymbolTable(devModule);
  // Outline all the target Ops
  for (TargetOpInfo &opInfo : targetOps)
    if (failed(outlineTargetOp(devModule, devSymbolTable, hostSymbolTable,
                               opInfo)))
      return signalPassFailure();
}

TargetRegionEntryInfoAttr OMPTargetGPUOutlining::getTargetEntryInfo(
    OpBuilder builder, const TargetOpInfo &opInfo, llvm::StringRef moduleName) {
  auto fileLoc = opInfo.op.getLoc()->findInstanceOf<FileLineColLoc>();
  // Try to create the entry info from `FileLineColLoc`
  if (fileLoc) {
    StringRef fileName = fileLoc.getFilename().getValue();

    llvm::sys::fs::UniqueID id;
    if (auto ec = llvm::sys::fs::getUniqueID(fileName, id)) {
      opInfo.op.emitError("unable to get unique ID for file");
      return nullptr;
    }
    uint64_t line = fileLoc.getLine();
    return builder.getAttr<TargetRegionEntryInfoAttr>(
        id.getDevice(), id.getFile(), line,
        builder.getAttr<FlatSymbolRefAttr>(moduleName));
  }
  return builder.getAttr<TargetRegionEntryInfoAttr>(
      opInfo.uniqueId, opInfo.functionId, opInfo.opId,
      builder.getAttr<FlatSymbolRefAttr>(moduleName));
}

LogicalResult OMPTargetGPUOutlining::outlineTargetOp(
    gpu::GPUModuleOp module, SymbolTable &devSymbolTable,
    SymbolTable &hostSymbolTable, const TargetOpInfo &opInfo) {
  TargetOp targetOp = opInfo.op;
  OpBuilder builder(targetOp.getContext());
  Location loc = targetOp.getLoc();
  // Set the entry info.
  targetOp.setTargetRegionEntryInfoAttr(
      getTargetEntryInfo(builder, opInfo, module.getName()));
  // Get the values that have to be mapped.
  SmallVector<Value> outlinedValues;
  SmallVector<Type> outlinedFnArgTypes;
  if (auto ifExpr = targetOp.getIfExpr()) {
    outlinedValues.push_back(ifExpr);
    outlinedFnArgTypes.push_back(ifExpr.getType());
  }
  if (auto dev = targetOp.getDevice()) {
    outlinedValues.push_back(dev);
    outlinedFnArgTypes.push_back(dev.getType());
  }
  if (auto thrLimit = targetOp.getThreadLimit()) {
    outlinedValues.push_back(thrLimit);
    outlinedFnArgTypes.push_back(thrLimit.getType());
  }
  for (const auto &operand : targetOp.getMapOperands()) {
    auto mapInfo = dyn_cast_or_null<MapInfoOp>(operand.getDefiningOp());
    if (!mapInfo)
      return targetOp.emitError("missing map info");
    for (Value operand : mapInfo->getOperands()) {
      outlinedValues.push_back(operand);
      outlinedFnArgTypes.push_back(operand.getType());
    }
  }
  // Create the outlined function.
  FunctionType type =
      FunctionType::get(targetOp.getContext(), outlinedFnArgTypes, {});
  auto outlinedFunc =
      builder.create<func::FuncOp>(loc, opInfo.parentName, type);
  devSymbolTable.insert(outlinedFunc);
  // Map the operands of the outlined function.
  Block &entryBlock = outlinedFunc.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(&entryBlock);
  IRMapping map;
  for (Value arg : outlinedValues)
    map.map(arg, entryBlock.addArgument(arg.getType(), arg.getLoc()));
  for (const auto &operand : targetOp.getMapOperands()) {
    auto mapInfo = dyn_cast_or_null<MapInfoOp>(operand.getDefiningOp());
    auto outlinedInfo =
        dyn_cast<MapInfoOp>(builder.clone(*(mapInfo.getOperation()), map));
    map.map(operand, outlinedInfo);
  }
  // Clone the Op.
  auto devTargetOp =
      dyn_cast<TargetOp>(builder.clone(*(targetOp.getOperation()), map));
  // Add an empty return.
  builder.create<func::ReturnOp>(loc);
  // Set the early outlining information.
  auto outliningIface =
      dyn_cast<EarlyOutliningInterface>(outlinedFunc.getOperation());
  assert(outliningIface && "missing outlining interface");
  outliningIface.setParentName(opInfo.parentName.str());
  if (failed(cloneDeclareTarget(builder, devTargetOp, devSymbolTable,
                                hostSymbolTable, module.getName())))
    return failure();
  return success();
}

LogicalResult OMPTargetGPUOutlining::cloneDeclareTarget(
    OpBuilder builder, TargetOp op, SymbolTable &devSymbolTable,
    SymbolTable &hostSymbolTable, StringRef moduleName) {
  SmallVector<Operation *, 8> symbolDefWorklist = {op};
  // Go through every symbol reference inside the TargetOp.
  while (!symbolDefWorklist.empty()) {
    if (std::optional<SymbolTable::UseRange> symbolUses =
            SymbolTable::getSymbolUses(symbolDefWorklist.pop_back_val())) {
      for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
        StringRef symbolName =
            cast<FlatSymbolRefAttr>(symbolUse.getSymbolRef()).getValue();
        // Check if the symbol is already in the device module.
        if (symbolName == moduleName || devSymbolTable.lookup(symbolName))
          continue;
        // Find the symbol in the host module and determine whether it's valid.
        Operation *symbolDef = hostSymbolTable.lookup(symbolName);
        if (auto iface = dyn_cast<DeclareTargetInterface>(symbolDef);
            !iface || !iface.isDeclareTarget()) {
          return symbolDef->emitError("symbol must be a declare target");
        }
        // Clone the symbol.
        Operation *symbolDefClone = symbolDef->clone();
        symbolDefWorklist.push_back(symbolDefClone);
        devSymbolTable.insert(symbolDefClone);
      }
    }
  }
  return success();
}
