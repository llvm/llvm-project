//===- OpenACCUtilsCG.cpp - OpenACC Code Generation Utilities -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utility functions for OpenACC code generation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsLoop.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace acc {

std::optional<DataLayout> getDataLayout(Operation *op, bool allowDefault) {
  if (!op)
    return std::nullopt;

  // Walk up the parent chain to find the nearest operation with an explicit
  // data layout spec. Check ModuleOp explicitly since it does not actually
  // implement DataLayoutOpInterface as a trait (it just has the same methods).
  Operation *current = op;
  while (current) {
    // Check for ModuleOp with explicit data layout spec
    if (auto mod = llvm::dyn_cast<ModuleOp>(current)) {
      if (mod.getDataLayoutSpec())
        return DataLayout(mod);
    } else if (auto dataLayoutOp =
                   llvm::dyn_cast<DataLayoutOpInterface>(current)) {
      // Check other DataLayoutOpInterface implementations
      if (dataLayoutOp.getDataLayoutSpec())
        return DataLayout(dataLayoutOp);
    }
    current = current->getParentOp();
  }

  // No explicit data layout found; return default if allowed
  if (allowDefault) {
    // Check if op itself is a ModuleOp
    if (auto mod = llvm::dyn_cast<ModuleOp>(op))
      return DataLayout(mod);
    // Otherwise check parents
    if (auto mod = op->getParentOfType<ModuleOp>())
      return DataLayout(mod);
  }

  return std::nullopt;
}

ComputeRegionOp buildComputeRegion(Location loc, ValueRange launchArgs,
                                   ValueRange inputArgs, llvm::StringRef origin,
                                   Region &regionToClone,
                                   RewriterBase &rewriter, IRMapping &mapping,
                                   ValueRange output,
                                   FlatSymbolRefAttr kernelFuncName,
                                   FlatSymbolRefAttr kernelModuleName,
                                   Value stream, ValueRange inputArgsToMap) {
  SmallVector<Type> resultTypes;
  for (auto val : output)
    resultTypes.push_back(val.getType());
  auto computeRegion =
      ComputeRegionOp::create(rewriter, loc, resultTypes, launchArgs, inputArgs,
                              stream, origin, kernelFuncName, kernelModuleName);

  assert(!regionToClone.getBlocks().empty() &&
         "empty region for acc.compute_region");
  OpBuilder::InsertionGuard guard(rewriter);

  ValueRange mapKeys = inputArgsToMap.empty() ? inputArgs : inputArgsToMap;
  assert(mapKeys.size() == inputArgs.size() &&
         "inputArgsToMap must have same size as inputArgs when provided");

  Type indexType = rewriter.getIndexType();
  Block *entryBlock = rewriter.createBlock(&computeRegion.getRegion());
  for (size_t i = 0; i < launchArgs.size(); ++i)
    entryBlock->addArgument(indexType, loc);
  for (Value input : inputArgs)
    entryBlock->addArgument(input.getType(), loc);
  for (size_t i = 0; i < inputArgs.size(); ++i)
    mapping.map(mapKeys[i], entryBlock->getArgument(launchArgs.size() + i));
  rewriter.setInsertionPointToStart(entryBlock);
  if (regionToClone.getBlocks().size() == 1) {
    for (auto &op : regionToClone.front().getOperations()) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        break;
      rewriter.clone(op, mapping);
    }
    SmallVector<Value> yieldOperands;
    for (auto val : output)
      yieldOperands.push_back(mapping.lookup(val));
    rewriter.setInsertionPointToEnd(entryBlock);
    YieldOp::create(rewriter, loc, yieldOperands);
  } else {
    auto exeRegion = mlir::acc::wrapMultiBlockRegionWithSCFExecuteRegion(
        regionToClone, mapping, loc, rewriter);
    if (!exeRegion) {
      rewriter.eraseOp(computeRegion);
      return nullptr;
    }
    SmallVector<scf::YieldOp> yieldOps(
        llvm::to_vector(exeRegion.getOps<scf::YieldOp>()));
    assert(!yieldOps.empty() &&
           "multi-block region must contain at least one scf.yield");
    assert(llvm::all_of(yieldOps,
                        [&output](scf::YieldOp yieldOp) {
                          return yieldOp.getNumOperands() ==
                                     static_cast<int64_t>(output.size()) &&
                                 llvm::all_of(
                                     llvm::zip(yieldOp.getOperands(), output),
                                     [](auto pair) {
                                       return std::get<0>(pair).getType() ==
                                              std::get<1>(pair).getType();
                                     });
                        }) &&
           "each scf.yield operand count and types must match output");
    rewriter.setInsertionPointToEnd(entryBlock);
    YieldOp::create(rewriter, loc, exeRegion.getResults());
  }

  return computeRegion;
}

static SmallVector<GPUParallelDimAttr>::iterator
findParDim(SmallVector<GPUParallelDimAttr> &parDims,
           GPUParallelDimAttr parDim) {
  return llvm::lower_bound(
      parDims, parDim,
      [](const GPUParallelDimAttr &lhs, const GPUParallelDimAttr &rhs) {
        return lhs.getOrder() > rhs.getOrder();
      });
}

void insertParDim(SmallVector<GPUParallelDimAttr> &parDims,
                  GPUParallelDimAttr parDim) {
  SmallVector<GPUParallelDimAttr>::iterator lb = findParDim(parDims, parDim);
  if (lb == parDims.end() || *lb != parDim)
    parDims.insert(lb, parDim);
}

void removeParDim(SmallVector<GPUParallelDimAttr> &parDims,
                  GPUParallelDimAttr parDim) {
  SmallVector<GPUParallelDimAttr>::iterator lb = findParDim(parDims, parDim);
  if (lb != parDims.end() && *lb == parDim)
    parDims.erase(lb);
}

#define ACC_OP_WITH_PAR_DIMS_LIST                                              \
  PrivatizeOp, ReductionAccumulateOp, ReductionAccumulateArrayOp

GPUParallelDimsAttr getParDimsAttr(Operation *op) {
  return llvm::TypeSwitch<Operation *, GPUParallelDimsAttr>(op)
      .Case<ACC_OP_WITH_PAR_DIMS_LIST>(
          [](auto parOp) { return parOp.getParDimsAttr(); })
      .Default([](Operation *op) -> GPUParallelDimsAttr {
        if (Attribute attr = op->getAttr(GPUParallelDimsAttr::name)) {
          GPUParallelDimsAttr parDimsAttr = dyn_cast<GPUParallelDimsAttr>(attr);
          assert(parDimsAttr && "acc.par_dims must be a GPUParallelDimsAttr");
          return parDimsAttr;
        }
        return nullptr;
      });
}

bool hasParDimsAttr(Operation *op) { return getParDimsAttr(op) != nullptr; }

bool hasSeqParDims(Operation *op) {
  if (GPUParallelDimsAttr parDimsAttr = getParDimsAttr(op))
    return parDimsAttr.isSeq();
  return false;
}

void setParDimsAttr(Operation *op, GPUParallelDimsAttr attr) {
  assert(!hasParDimsAttr(op) && "parallel dimensions attribute is already set");
  llvm::TypeSwitch<Operation *>(op)
      .Case<ACC_OP_WITH_PAR_DIMS_LIST>(
          [&](auto parOp) { parOp.setParDimsAttr(attr); })
      .Default(
          [&](Operation *op) { op->setAttr(GPUParallelDimsAttr::name, attr); });
}

void updateParDimsAttr(Operation *op, GPUParallelDimsAttr attr) {
  assert(hasParDimsAttr(op) &&
         "expected parallel dimensions attribute to already be set");
  llvm::TypeSwitch<Operation *>(op)
      .Case<ACC_OP_WITH_PAR_DIMS_LIST>(
          [&](auto parOp) { parOp.setParDimsAttr(attr); })
      .Default(
          [&](Operation *op) { op->setAttr(GPUParallelDimsAttr::name, attr); });
}

#undef ACC_OP_WITH_PAR_DIMS_LIST

void copyParDimsAttr(Operation *from, Operation *to) {
  assert(hasParDimsAttr(from) &&
         "expected parallel dimensions attribute to already be set");
  setParDimsAttr(to, getParDimsAttr(from));
}

} // namespace acc
} // namespace mlir
