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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsLoop.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsReduction.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsType.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

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

int64_t SharedMemoryBudget::alignOffset(int64_t offset, int64_t alignment) {
  assert(alignment > 0 && llvm::isPowerOf2_64(alignment) &&
         "alignment must be a power of two");
  return (offset + alignment - 1) & ~(alignment - 1);
}

bool SharedMemoryBudget::tryAllocate(int64_t bytes, int64_t alignment) {
  int64_t aligned = alignOffset(bytesUsed_, alignment);
  if (aligned + bytes > maxTotalBytes_) {
    return false;
  }
  bytesUsed_ = aligned + bytes;
  return true;
}

int64_t sumExistingSharedMemoryBytes(Region &region) {
  int64_t total = 0;
  region.walk([&](GPUSharedMemoryOp op) {
    int64_t upperBound = op.getStaticUpperBoundBytes();
    total = SharedMemoryBudget::alignOffset(total) + upperBound;
  });
  return total;
}

PrivatizeOp getPrivatizeOp(PrivateLocalOp privateLocal,
                           ComputeRegionOp computeRegion) {
  Value value = privateLocal.getPrivatized();
  if (BlockArgument blockArg = dyn_cast<BlockArgument>(value)) {
    auto owner = dyn_cast<ComputeRegionOp>(blockArg.getOwner()->getParentOp());
    value = (owner ? owner : computeRegion).getOperand(blockArg);
  }
  PrivatizeOp privatizeOp = value.getDefiningOp<PrivatizeOp>();
  assert(privatizeOp && "expected privatize op to be the defining op");
  return privatizeOp;
}

static bool isThreadXPrivatize(PrivatizeOp privatize) {
  if (GPUParallelDimsAttr parDimsAttr = privatize.getParDimsAttr())
    return llvm::any_of(parDimsAttr.getArray(),
                        [](GPUParallelDimAttr d) { return d.isThreadX(); });
  return false;
}

MemRefType getPrivateBaseMemRefType(Type baseTy, ModuleOp module) {
  auto memrefTy = cast<PointerLikeType>(baseTy).getAsMemRefType(module);
  assert(memrefTy && "private base type must be convertible to memref");
  return memrefTy;
}

SmallVector<GPUParallelDimAttr>
collectPrivateLocalParDims(PrivateLocalOp privateLocal,
                           ComputeRegionOp computeRegion) {
  SmallVector<GPUParallelDimAttr> parDims;
  // Walk the enclosing scf.parallel loops, but stop at the compute region
  // boundary: loops outside the compute region do not contribute parallel
  // dimensions to this privatization.
  auto parentLoop = privateLocal->getParentOfType<scf::ParallelOp>();
  while (parentLoop && computeRegion->isProperAncestor(parentLoop)) {
    if (GPUParallelDimsAttr parDimsAttr = getParDimsAttr(parentLoop))
      for (GPUParallelDimAttr parDim : parDimsAttr.getArray())
        insertParDim(parDims, parDim);
    parentLoop = parentLoop->getParentOfType<scf::ParallelOp>();
  }
  if (GPUParallelDimsAttr parDimsAttr = getParDimsAttr(computeRegion))
    for (GPUParallelDimAttr parDim : parDimsAttr.getArray())
      insertParDim(parDims, parDim);
  if (parDims.empty()) {
    for (GPUParallelDimAttr parDim : computeRegion.getLaunchParDims()) {
      if (parDim.isAnyBlock())
        insertParDim(parDims, parDim);
    }
  }

  for (Operation *user : privateLocal.getResult().getUsers()) {
    if (auto accumulateOp = dyn_cast<ReductionAccumulateOp>(user)) {
      if (accumulateOp.getMemref() == privateLocal.getResult())
        for (GPUParallelDimAttr parDim : accumulateOp.getParDims().getArray())
          insertParDim(parDims, parDim);
    }
    if (auto combineOp = dyn_cast<ReductionCombineOp>(user)) {
      if (combineOp.getSrcMemref() == privateLocal.getResult())
        for (GPUParallelDimAttr parDim : getReductionCombineParDims(combineOp))
          insertParDim(parDims, parDim);
    }
    if (auto combineRegionOp = dyn_cast<ReductionCombineRegionOp>(user)) {
      if (combineRegionOp.getSrcVar() == privateLocal.getResult())
        for (GPUParallelDimAttr parDim :
             getReductionCombineParDims(combineRegionOp))
          insertParDim(parDims, parDim);
    }
  }
  return parDims;
}

static FailureOr<std::optional<int64_t>> getWorkerPrivateSharedMemoryNumCopies(
    PrivateLocalOp privateLocal, ComputeRegionOp computeRegion,
    bool isWorkerPrivate, bool isGangPrivate, OpenACCSupport *support) {
  if (!isWorkerPrivate || isGangPrivate)
    return std::optional<int64_t>(1);

  GPUParallelDimAttr threadY =
      GPUParallelDimAttr::threadYDim(privateLocal.getContext());
  std::optional<Value> workerArg = computeRegion.getKnownLaunchArg(threadY);
  if (!workerArg)
    return std::optional<int64_t>();

  auto workerArgConst = workerArg->getDefiningOp<arith::ConstantIndexOp>();
  if (!workerArgConst) {
    if (support) {
      (void)support->emitNYI(privateLocal.getLoc(),
                             "worker-private variables in shared memory "
                             "require compile-time constant num_workers");
      return failure();
    }
    return std::optional<int64_t>();
  }
  return std::optional<int64_t>(workerArgConst.value());
}

static bool isInsideACCSpecializedRoutine(Operation *op) {
  auto funcOp = op->getParentOfType<FunctionOpInterface>();
  return funcOp && isSpecializedAccRoutine(funcOp);
}

FailureOr<bool> isPrivateLocalSharedMemoryCandidate(
    PrivateLocalOp privateLocal, ComputeRegionOp computeRegion, ModuleOp module,
    const ACCToGPUMappingPolicy &policy, OpenACCSupport *support) {
  if (isInsideACCSpecializedRoutine(computeRegion))
    return false;

  if (isThreadXPrivatize(getPrivatizeOp(privateLocal, computeRegion)))
    return false;

  bool isReductionAccumulator =
      llvm::any_of(privateLocal.getResult().getUsers(), [](Operation *user) {
        return isa<ReductionAccumulateOp>(user);
      });

  SmallVector<GPUParallelDimAttr> parDims =
      collectPrivateLocalParDims(privateLocal, computeRegion);
  bool isGangPrivate =
      llvm::any_of(parDims, [&](auto parDim) { return policy.isGang(parDim); });
  bool isWorkerPrivate = llvm::any_of(
      parDims, [&](auto parDim) { return policy.isWorker(parDim); });
  bool isVectorPrivate = llvm::any_of(
      parDims, [&](auto parDim) { return policy.isVector(parDim); });

  auto baseTy = getPrivateBaseMemRefType(
      cast<PrivateType>(privateLocal.getPrivatized().getType()).getBaseTy(),
      module);

  bool isBlockLevelPrivate =
      !isVectorPrivate &&
      (isGangPrivate ||
       (isWorkerPrivate && baseTy.getRank() > 0 && !isReductionAccumulator));
  if (!isBlockLevelPrivate)
    return false;

  for (int64_t dim : baseTy.getShape())
    if (dim == ShapedType::kDynamic)
      return false;

  auto resultMemRefTy = dyn_cast<MemRefType>(privateLocal.getType());
  if (!resultMemRefTy || !resultMemRefTy.getLayout().isIdentity() ||
      resultMemRefTy.getMemorySpace())
    return false;

  if (isGangPrivate && isWorkerPrivate && !isReductionAccumulator)
    return false;

  FailureOr<std::optional<int64_t>> numCopies =
      getWorkerPrivateSharedMemoryNumCopies(
          privateLocal, computeRegion, isWorkerPrivate, isGangPrivate, support);
  if (failed(numCopies))
    return failure();
  return numCopies->has_value();
}

std::optional<int64_t> getPrivateLocalSharedMemoryUpperBoundBytes(
    PrivateLocalOp privateLocal, ComputeRegionOp computeRegion, ModuleOp module,
    const ACCToGPUMappingPolicy &policy, OpenACCSupport *support) {
  FailureOr<bool> isCandidate = isPrivateLocalSharedMemoryCandidate(
      privateLocal, computeRegion, module, policy);
  if (failed(isCandidate) || !*isCandidate)
    return std::nullopt;

  SmallVector<GPUParallelDimAttr> parDims =
      collectPrivateLocalParDims(privateLocal, computeRegion);
  bool isGangPrivate =
      llvm::any_of(parDims, [&](auto parDim) { return policy.isGang(parDim); });
  bool isWorkerPrivate = llvm::any_of(
      parDims, [&](auto parDim) { return policy.isWorker(parDim); });

  FailureOr<std::optional<int64_t>> numCopies =
      getWorkerPrivateSharedMemoryNumCopies(privateLocal, computeRegion,
                                            isWorkerPrivate, isGangPrivate,
                                            /*support=*/nullptr);
  if (failed(numCopies) || !numCopies->has_value())
    return std::nullopt;

  auto baseTy = getPrivateBaseMemRefType(
      cast<PrivateType>(privateLocal.getPrivatized().getType()).getBaseTy(),
      module);
  std::optional<TypeSizeAndAlignment> elementSizeAndAlignment =
      getTypeSizeAndAlignment(baseTy.getElementType(), module, support);
  if (!elementSizeAndAlignment)
    return std::nullopt;

  int64_t numElements = 1;
  for (int64_t dim : baseTy.getShape())
    numElements *= dim;
  return elementSizeAndAlignment->first.getFixedValue() * numElements *
         numCopies->value();
}

} // namespace acc
} // namespace mlir
