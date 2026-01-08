//===- ShardingInterfaceImpl.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/ShardingInterfaceImpl.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/Dialect/Shard/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Shard/Interfaces/ShardingInterfaceImpl.h"
#include "mlir/Dialect/Shard/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <numeric>
#include <optional>

namespace mlir::linalg {

using GridAxis = shard::GridAxis;
using ReductionKind = shard::ReductionKind;
using Sharding = shard::Sharding;
using ShardingArray = shard::ShardingArray;
using GridOp = shard::GridOp;

// Returns the corresponding grid reduction kind for the given arith op.
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

static GridOp getGrid(Operation *op, ArrayRef<Sharding> operandShardings,
                      ArrayRef<Sharding> resultShardings,
                      SymbolTableCollection &symbolTable) {
  for (const Sharding &sharding : operandShardings) {
    if (sharding) {
      return shard::getGrid(op, sharding.getGridAttr(), symbolTable);
    }
  }

  for (const Sharding &sharding : resultShardings) {
    if (sharding) {
      return shard::getGrid(op, sharding.getGridAttr(), symbolTable);
    }
  }

  assert(false);
  return nullptr;
}

// Choose the operand based on the current process index along the reduction
// grid axes.
// We need to use the initial value only once to avoid including it in the
// reduction multiple times.
// In each process group only the leading process with linear index 0 would use
// the original operand.
// The other processes would use the reduction operation neutral tensor.
static Value createDestinationPassingStyleInitOperand(
    LinalgOp op, int operandNumber, Value partitionedOperand,
    ArrayRef<GridAxis> reductionGridAxes, GridOp gridOp,
    ImplicitLocOpBuilder &builder) {
  Value processLinearIndexInReductionGroup = shard::createProcessLinearIndex(
      gridOp.getSymName(), reductionGridAxes, builder);
  Value zero = arith::ConstantIndexOp::create(builder, 0);
  Value isLeadProcess = arith::CmpIOp::create(
      builder, builder.getI1Type(), arith::CmpIPredicate::eq,
      processLinearIndexInReductionGroup, zero);
  scf::IfOp ifOp = scf::IfOp::create(builder, partitionedOperand.getType(),
                                     isLeadProcess, true, true);
  // Then block.
  {
    OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPointToEnd(&ifOp.getThenRegion().front());
    scf::YieldOp::create(builder, partitionedOperand);
  }

  // Else block.
  {
    OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPointToEnd(&ifOp.getElseRegion().front());
    SmallVector<OpFoldResult> shape =
        tensor::getMixedSizes(builder, builder.getLoc(), partitionedOperand);

    SmallVector<Operation *> combinerOps;
    matchReduction(op.getRegionOutputArgs(), operandNumber, combinerOps);
    assert(combinerOps.size() == 1);
    std::optional<TypedAttr> neutralEl =
        arith::getNeutralElement(combinerOps[0]);

    Value init = tensor::EmptyOp::create(builder, op.getLoc(), shape,
                                         neutralEl.value().getType());
    Value constant =
        arith::ConstantOp::create(builder, op.getLoc(), neutralEl.value());
    Value fill = linalg::FillOp::create(builder, op.getLoc(), constant, init)
                     .getResult(0);

    scf::YieldOp::create(builder, fill);
  }
  return ifOp.getResult(0);
}

// Create the DPS init operands for the partitioned Linalg op.
// Return all the new partitioned operands.
static SmallVector<Value> createDestinationPassingStyleInitOperands(
    LinalgOp op, GridOp gridOp, ArrayRef<Value> partitionedOperands,
    ArrayRef<GridAxis> reductionGridAxes, IRMapping &partitionMap,
    ImplicitLocOpBuilder &builder) {
  // TODO: add support for multiple destination passing style initial value
  // operands.
  assert(op.getNumDpsInits() == 1 && "Multiple initial values not supported.");
  SmallVector<Value> newOperands = llvm::to_vector(partitionedOperands);
  auto operandIdx = op.getDpsInitOperand(0)->getOperandNumber();
  Value partitionedInitOperand =
      partitionMap.lookup(op->getOperands()[operandIdx]);
  newOperands[operandIdx] = createDestinationPassingStyleInitOperand(
      op, 0, partitionedInitOperand, reductionGridAxes, gridOp, builder);
  return newOperands;
}

static void createAllReduceForResultsWithoutPartialShardings(
    LinalgOp unshardedOp, ArrayRef<GridAxis> opReductionGridAxes,
    ArrayRef<Sharding> resultShardings, IRMapping &partitionMap,
    ImplicitLocOpBuilder &builder) {
  ReductionKind reductionKind = getReductionKindOfLinalgOp(unshardedOp);
  for (auto [unshardedLinalgOpResult, resultSharding] :
       llvm::zip_equal(unshardedOp->getResults(), resultShardings)) {
    Value partitionedLinalgOpResult =
        partitionMap.lookup(unshardedLinalgOpResult);
    Value reducedValue = shard::AllReduceOp::create(
        builder, partitionedLinalgOpResult, resultSharding.getGrid(),
        opReductionGridAxes, reductionKind);
    partitionMap.map(unshardedLinalgOpResult, reducedValue);
  }
}

static void partitionLinalgOpWithShardedReduction(
    LinalgOp op, ArrayRef<Value> partitionedOperands,
    ArrayRef<Sharding> operandShardings, ArrayRef<Sharding> resultShardings,
    ArrayRef<utils::IteratorType> loopIteratorTypes,
    ArrayRef<SmallVector<GridAxis>> gridAxisAssignmentForLoopIterators,
    IRMapping &partitionMap, SymbolTableCollection &symbolTable,
    ImplicitLocOpBuilder &builder) {
  GridOp grid = getGrid(op, operandShardings, resultShardings, symbolTable);
  SmallVector<GridAxis> reductionGridAxes = shard::getReductionGridAxes(
      loopIteratorTypes, gridAxisAssignmentForLoopIterators);
  SmallVector<Value> partitionedLinalgOpOperands =
      createDestinationPassingStyleInitOperands(op, grid, partitionedOperands,
                                                reductionGridAxes, partitionMap,
                                                builder);
  // We must not change the operand mappings of the original partitionMap as
  // they are the mappings for the whole partition blob and may be used by
  // others.
  IRMapping internalPartitionMap;
  for (auto [unshardedOperand, partitionedOperand] :
       llvm::zip_equal(op->getOperands(), partitionedLinalgOpOperands)) {
    internalPartitionMap.map(unshardedOperand, partitionedOperand);
  }
  partitionTriviallyShardableOperation(
      *op, partitionedLinalgOpOperands, operandShardings, resultShardings,
      internalPartitionMap, symbolTable, builder);
  for (Value result : op->getResults()) {
    partitionMap.map(result, internalPartitionMap.lookup(result));
  }

  // Handle partial shardings.
  createAllReduceForResultsWithoutPartialShardings(
      op, reductionGridAxes, resultShardings, partitionMap, builder);
}

namespace {

// ShardingInterface for ops that implement LinalgStructuredInterface.
// The supported ops are only those where the indexing maps are projected
// permutations.
template <typename Op>
struct StructuredOpShardingInterface
    : public shard::ShardingInterface::ExternalModel<
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
    unsigned reductionItersCount = llvm::accumulate(
        iteratorTypes, 0u, [](unsigned count, utils::IteratorType iter) {
          return count + (iter == utils::IteratorType::reduction);
        });
    shard::ReductionKind reductionKind = getReductionKindOfLinalgOp(linalgOp);
    return SmallVector<ReductionKind>(reductionItersCount, reductionKind);
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<Sharding> operandShardings,
                          ArrayRef<Sharding> resultShardings,
                          IRMapping &partitionMap,
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
    ShardingArray gridAxisAssignmentForLoopIterators =
        getGridAxisAssignmentForLoopIterators(operandShardings, resultShardings,
                                              loopIteratorTypes, indexingMaps);
    if (shard::isAtLeastOneReductionIteratorSharded(
            loopIteratorTypes, gridAxisAssignmentForLoopIterators)) {
      ImplicitLocOpBuilder implicitLocBuilder(op->getLoc(), builder);
      partitionLinalgOpWithShardedReduction(
          linalgOp, partitionedOperands, operandShardings, resultShardings,
          loopIteratorTypes, gridAxisAssignmentForLoopIterators, partitionMap,
          symbolTable, implicitLocBuilder);
    } else {
      partitionTriviallyShardableOperation(*op, partitionedOperands,
                                           operandShardings, resultShardings,
                                           partitionMap, symbolTable, builder);
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

void registerShardingInterfaceExternalModels(DialectRegistry &registry) {
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
