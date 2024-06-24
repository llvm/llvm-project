//===- MeshShardingInterfaceImpl.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/MeshShardingInterfaceImpl.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Mesh/Interfaces/ShardingInterfaceImpl.h"
#include "mlir/Dialect/Mesh/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iterator>
#include <numeric>
#include <optional>
#include <utility>

namespace mlir::linalg {

using MeshAxis = mesh::MeshAxis;
using ReductionKind = mesh::ReductionKind;
using MeshShardingAttr = mesh::MeshShardingAttr;
using ShardingArray = mesh::ShardingArray;
using MeshOp = mesh::MeshOp;

// Returns the corresponding mesh reduction kind for the given arith op.
static ReductionKind getReductionKind(Operation *op) {
  return llvm::TypeSwitch<Operation *, ReductionKind>(op)
      // Floating-point operations.
      .Case([](arith::AddFOp op) { return ReductionKind::Sum; })
      .Case([](arith::MulFOp op) { return ReductionKind::Product; })
      // TODO: handle maxnumf and minnumf.
      .Case([](arith::MaximumFOp op) { return ReductionKind::Max; })
      .Case([](arith::MinimumFOp op) { return ReductionKind::Min; })
      // Integer operations.
      .Case([](arith::AddIOp op) { return ReductionKind::Sum; })
      .Case([](arith::OrIOp op) { return ReductionKind::BitwiseOr; })
      .Case([](arith::XOrIOp op) { return ReductionKind::BitwiseXor; })
      .Case([](arith::AndIOp op) { return ReductionKind::Sum; })
      // TODO: handle signless, signed and unsigned types properly.
      // It is assumed that the element type of the collective operands and
      // result drive the meaning of the reduction kind, whether it is signed
      // or unsigned.
      // The reduction op inside the linalg op may have different result type
      // from the element type of the linalg op's result.
      // Also signed and unsigned Arith dialect ops may accept signed, unsigned
      // or signless operands.
      // Maybe expand the reduction kinds.
      .Case([](arith::MaxUIOp op) { return ReductionKind::Max; })
      .Case([](arith::MinUIOp op) { return ReductionKind::Min; })
      .Case([](arith::MaxSIOp op) { return ReductionKind::Max; })
      .Case([](arith::MinSIOp op) { return ReductionKind::Min; })
      .Case([](arith::MulIOp op) { return ReductionKind::Product; })
      .Default([](Operation *op) { return ReductionKind::Generic; });
}

static std::optional<Operation *> getCombinerOp(LinalgOp op) {
  SmallVector<Operation *> combinerOps;
  Value reducedValue = matchReduction(op.getRegionOutputArgs(), 0, combinerOps);
  if (!reducedValue || combinerOps.size() != 1) {
    return std::nullopt;
  }

  return combinerOps[0];
}

static ReductionKind getReductionKindOfLinalgOp(LinalgOp op) {
  std::optional<Operation *> reductionOp = getCombinerOp(op);
  if (!reductionOp) {
    return ReductionKind::Generic;
  }
  [[maybe_unused]] Type resultElementType =
      llvm::cast<RankedTensorType>(op->getResult(0).getType()).getElementType();
  // TODO: handle case when result type of the reduction op does not match the
  // element type of the result tensor.
  // Would it makes sense at all?
  assert(resultElementType == reductionOp.value()->getResult(0).getType());
  return getReductionKind(reductionOp.value());
}

static MeshOp getMesh(Operation *op,
                      ArrayRef<MeshShardingAttr> operandShardings,
                      ArrayRef<MeshShardingAttr> resultShardings,
                      SymbolTableCollection &symbolTable) {
  for (MeshShardingAttr sharding : operandShardings) {
    if (sharding) {
      return mesh::getMesh(op, sharding.getMesh(), symbolTable);
    }
  }

  for (MeshShardingAttr sharding : resultShardings) {
    if (sharding) {
      return mesh::getMesh(op, sharding.getMesh(), symbolTable);
    }
  }

  assert(false);
  return nullptr;
}

// Choose the operand based on the current process index along the reduction
// mesh axes.
// We need to use the initial value only once to avoid including it in the
// reduction multiple times.
// In each process group only the leading process with linear index 0 would use
// the original operand.
// The other processes would use the reduction operation neutral tensor.
static Value createDestinationPassingStyleInitOperand(
    LinalgOp op, Value spmdizedOperand, ArrayRef<MeshAxis> reductionMeshAxes,
    MeshOp meshOp, ImplicitLocOpBuilder &builder) {
  Value processLinearIndexInReductionGroup = mesh::createProcessLinearIndex(
      meshOp.getSymName(), reductionMeshAxes, builder);
  Value zero = builder.create<arith::ConstantIndexOp>(0);
  Value isLeadProcess = builder.create<arith::CmpIOp>(
      builder.getI1Type(), arith::CmpIPredicate::eq,
      processLinearIndexInReductionGroup, zero);
  scf::IfOp ifOp = builder.create<scf::IfOp>(spmdizedOperand.getType(),
                                             isLeadProcess, true, true);
  // Then block.
  {
    OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPointToEnd(&ifOp.getThenRegion().front());
    builder.create<scf::YieldOp>(spmdizedOperand);
  }

  // Else block.
  {
    OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPointToEnd(&ifOp.getElseRegion().front());
    SmallVector<OpFoldResult> shape =
        tensor::getMixedSizes(builder, builder.getLoc(), spmdizedOperand);
    PartialReductionOpInterface partialReductionIface =
        llvm::cast<PartialReductionOpInterface>(op.getOperation());
    assert(op->getNumResults() == 1 && "Multiple results not supported.");
    FailureOr<SmallVector<Value>> reductionNeutralTensor =
        partialReductionIface.generateInitialTensorForPartialReduction(
            builder, builder.getLoc(), shape, {});
    assert(succeeded(reductionNeutralTensor));
    builder.create<scf::YieldOp>(reductionNeutralTensor.value());
  }
  return ifOp.getResult(0);
}

// Create the DPS init operands for the spmdized Linalg op.
// Return all the new spmdized operands.
static SmallVector<Value> createDestinationPassingStyleInitOperands(
    LinalgOp op, MeshOp meshOp, ArrayRef<Value> spmdizedOperands,
    ArrayRef<MeshAxis> reductionMeshAxes, IRMapping &spmdizationMap,
    ImplicitLocOpBuilder &builder) {
  // TODO: add support for multiple destination passing style initial value
  // operands.
  assert(op.getNumDpsInits() == 1 && "Multiple initial values not supported.");
  SmallVector<Value> newOperands = llvm::to_vector(spmdizedOperands);
  auto operandIdx = op.getDpsInitOperand(0)->getOperandNumber();
  Value spmdizedInitOperand =
      spmdizationMap.lookup(op->getOperands()[operandIdx]);
  newOperands[operandIdx] = createDestinationPassingStyleInitOperand(
      op, spmdizedInitOperand, reductionMeshAxes, meshOp, builder);
  return newOperands;
}

static void createAllReduceForResultWithoutPartialSharding(
    Value unshardedLinalgOpResult, ArrayRef<MeshAxis> opReductionMeshAxes,
    MeshShardingAttr resultSharding, ReductionKind reductionKind,
    IRMapping &spmdizationMap, ImplicitLocOpBuilder &builder) {
  SmallVector<MeshAxis> allReduceMeshAxes;
  llvm::copy_if(opReductionMeshAxes, std::back_inserter(allReduceMeshAxes),
                [&resultSharding](MeshAxis axis) {
                  return !llvm::is_contained(resultSharding.getPartialAxes(),
                                             axis);
                });
  if (allReduceMeshAxes.empty()) {
    return;
  }

  Value spmdizedLinalgOpResult = spmdizationMap.lookup(unshardedLinalgOpResult);
  Value reducedValue = builder.create<mesh::AllReduceOp>(
      spmdizedLinalgOpResult, resultSharding.getMesh().getValue(),
      allReduceMeshAxes, reductionKind);
  spmdizationMap.map(unshardedLinalgOpResult, reducedValue);
}

static void createAllReduceForResultsWithoutPartialShardings(
    LinalgOp unshardedOp, ArrayRef<MeshAxis> opReductionMeshAxes,
    ArrayRef<MeshShardingAttr> resultShardings, IRMapping &spmdizationMap,
    ImplicitLocOpBuilder &builder) {
  ReductionKind reductionKind = getReductionKindOfLinalgOp(unshardedOp);
  for (auto [unshardedLinalgOpResult, resultSharding] :
       llvm::zip_equal(unshardedOp->getResults(), resultShardings)) {
    createAllReduceForResultWithoutPartialSharding(
        unshardedLinalgOpResult, opReductionMeshAxes, resultSharding,
        reductionKind, spmdizationMap, builder);
  }
}

static void spmdizeLinalgOpWithShardedReduction(
    LinalgOp op, ArrayRef<Value> spmdizedOperands,
    ArrayRef<MeshShardingAttr> operandShardings,
    ArrayRef<MeshShardingAttr> resultShardings,
    ArrayRef<utils::IteratorType> loopIteratorTypes,
    ArrayRef<SmallVector<MeshAxis>> meshAxisAssignmentForLoopIterators,
    IRMapping &spmdizationMap, SymbolTableCollection &symbolTable,
    ImplicitLocOpBuilder &builder) {
  MeshOp mesh = getMesh(op, operandShardings, resultShardings, symbolTable);
  SmallVector<MeshAxis> reductionMeshAxes = mesh::getReductionMeshAxes(
      loopIteratorTypes, meshAxisAssignmentForLoopIterators);
  SmallVector<Value> spmdizedLinalgOpOperands =
      createDestinationPassingStyleInitOperands(op, mesh, spmdizedOperands,
                                                reductionMeshAxes,
                                                spmdizationMap, builder);
  // We must not change the operand mappings of the original spmdizationMap as
  // they are the mappings for the whole spmdization blob and may be used by
  // others.
  IRMapping internalSpmdizationMap;
  for (auto [unshardedOperand, spmdizedOperand] :
       llvm::zip_equal(op->getOperands(), spmdizedLinalgOpOperands)) {
    internalSpmdizationMap.map(unshardedOperand, spmdizedOperand);
  }
  spmdizeTriviallyShardableOperation(
      *op, spmdizedLinalgOpOperands, operandShardings, resultShardings,
      internalSpmdizationMap, symbolTable, builder);
  for (Value result : op->getResults()) {
    spmdizationMap.map(result, internalSpmdizationMap.lookup(result));
  }

  // Handle partial shardings.
  createAllReduceForResultsWithoutPartialShardings(
      op, reductionMeshAxes, resultShardings, spmdizationMap, builder);
}

namespace {

// ShardingInterface for ops that implement LinalgStructuredInterface.
// The supported ops are only those where the indexing maps are projected
// permutations.
template <typename Op>
struct StructuredOpShardingInterface
    : public mesh::ShardingInterface::ExternalModel<
          StructuredOpShardingInterface<Op>, Op> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    return llvm::cast<LinalgOp>(op).getIteratorTypesArray();
  }

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    LinalgOp linalgOp = llvm::cast<LinalgOp>(op);
    SmallVector<AffineMap> res = linalgOp.getIndexingMapsArray();

    // Results must have the same indexing as destination passing style initial
    // operands.
    for (int64_t i = 0; i < linalgOp.getNumDpsInits(); ++i) {
      res.push_back(res[linalgOp.getDpsInitOperand(i)->getOperandNumber()]);
    }

    return res;
  }

  SmallVector<ReductionKind>
  getReductionLoopIteratorKinds(Operation *op) const {
    LinalgOp linalgOp = llvm::cast<LinalgOp>(op);
    SmallVector<utils::IteratorType> iteratorTypes =
        linalgOp.getIteratorTypesArray();
    unsigned reductionItersCount = std::accumulate(
        iteratorTypes.begin(), iteratorTypes.end(), 0,
        [](unsigned count, utils::IteratorType iter) {
          return count + (iter == utils::IteratorType::reduction);
        });
    mesh::ReductionKind reductionKind = getReductionKindOfLinalgOp(linalgOp);
    return SmallVector<ReductionKind>(reductionItersCount, reductionKind);
  }

  LogicalResult spmdize(Operation *op, ArrayRef<Value> spmdizedOperands,
                        ArrayRef<MeshShardingAttr> operandShardings,
                        ArrayRef<MeshShardingAttr> resultShardings,
                        IRMapping &spmdizationMap,
                        SymbolTableCollection &symbolTable,
                        OpBuilder &builder) const {
    LinalgOp linalgOp = llvm::cast<LinalgOp>(op);

    SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
    bool allIndexingMapsAreProjectedPermutation =
        llvm::all_of(indexingMaps, [](AffineMap map) {
          return map.isProjectedPermutation();
        });
    if (!allIndexingMapsAreProjectedPermutation) {
      // TODO: handle non-projected permutations.
      return op->emitOpError()
             << "supports indexing maps that are only projected permutation.";
    }

    SmallVector<utils::IteratorType> loopIteratorTypes =
        linalgOp.getIteratorTypesArray();
    ShardingArray meshAxisAssignmentForLoopIterators =
        getMeshAxisAssignmentForLoopIterators(operandShardings, resultShardings,
                                              loopIteratorTypes, indexingMaps);
    if (mesh::isAtLeastOneReductionIteratorSharded(
            loopIteratorTypes, meshAxisAssignmentForLoopIterators)) {
      ImplicitLocOpBuilder implicitLocBuilder(op->getLoc(), builder);
      spmdizeLinalgOpWithShardedReduction(
          linalgOp, spmdizedOperands, operandShardings, resultShardings,
          loopIteratorTypes, meshAxisAssignmentForLoopIterators, spmdizationMap,
          symbolTable, implicitLocBuilder);
    } else {
      spmdizeTriviallyShardableOperation(*op, spmdizedOperands,
                                         operandShardings, resultShardings,
                                         spmdizationMap, symbolTable, builder);
    }

    return success();
  }
};

} // namespace

template <typename OpType>
static void registerOne(MLIRContext *ctx) {
  OpType::template attachInterface<StructuredOpShardingInterface<OpType>>(*ctx);
}

/// Variadic helper function.
template <typename... OpTypes>
static void registerAll(MLIRContext *ctx) {
  (registerOne<OpTypes>(ctx), ...);
}

void registerMeshShardingInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LinalgDialect *dialect) {
    DialectRegistry registry;
    registry.insert<affine::AffineDialect, arith::ArithDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
    ctx->appendDialectRegistry(registry);
    for (StringRef name : registry.getDialectNames())
      ctx->getOrLoadDialect(name);

    registerOne<linalg::GenericOp>(ctx);
    registerAll<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
        >(ctx);
  });
}

} // namespace mlir::linalg
