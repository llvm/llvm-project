//===- ShardingInterfaceImpl.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/ShardingInterfaceImpl.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/Dialect/Shard/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Shard/Interfaces/ShardingInterfaceImpl.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/DialectRegistry.h"

#define DEBUG_TYPE "tosa-sharding-impl"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::tosa;
using namespace mlir::shard;

namespace {

// For an output of rank R, use R parallel loops followed by one reduction
// loop. Right align the input batch dimensions with the output batch loops and
// map A[..., H, C] to [..., d(R-2), d(R)]. MATMUL maps B[..., C, W] to
// [..., d(R), d(R-1)], while MATMUL_T maps B[..., W, C] to
// [..., d(R-1), d(R)].
template <typename OpType, bool TransposeB>
struct MatMulSharding : public ShardingInterface::ExternalModel<
                            MatMulSharding<OpType, TransposeB>, OpType> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto tensorType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!tensorType)
      return {};

    SmallVector<utils::IteratorType> types(tensorType.getRank() + 1,
                                           utils::IteratorType::parallel);
    types[tensorType.getRank()] = utils::IteratorType::reduction;
    return types;
  }

  SmallVector<ReductionKind>
  getReductionLoopIteratorKinds(Operation *op) const {
    return SmallVector<ReductionKind>(1, ReductionKind::Sum);
  }

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    auto aType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto bType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    auto outputType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!aType || !bType || !outputType || aType.getRank() < 2 ||
        bType.getRank() < 2 || outputType.getRank() < aType.getRank() ||
        outputType.getRank() < bType.getRank())
      return {};

    MLIRContext *ctx = op->getContext();
    const int64_t outputRank = outputType.getRank();
    const int64_t loopRank = outputRank + 1;
    const int64_t reductionLoop = outputRank;

    auto getAlignedBatchDim = [&](RankedTensorType operandType,
                                  int64_t outputDim) {
      const int64_t operandDim =
          outputDim - (outputRank - operandType.getRank());
      return operandDim < 0 ? 1 : operandType.getDimSize(operandDim);
    };
    for (int64_t i = 0; i < outputRank - 2; ++i) {
      const int64_t aDim = getAlignedBatchDim(aType, i);
      const int64_t bDim = getAlignedBatchDim(bType, i);
      // A dynamic dimension may be either one (broadcast) or the corresponding
      // output extent. These cases require different indexing maps unless the
      // other operand is known to be a singleton.
      if ((ShapedType::isDynamic(aDim) && bDim != 1) ||
          (ShapedType::isDynamic(bDim) && aDim != 1))
        return {};
    }

    auto getOperandMap = [&](RankedTensorType operandType, bool isB) {
      const int64_t operandRank = operandType.getRank();
      const int64_t operandBatchRank = operandRank - 2;
      const int64_t batchRankDiff = outputRank - operandRank;
      SmallVector<AffineExpr> results;
      results.reserve(operandRank);
      for (int64_t i = 0; i < operandBatchRank; ++i) {
        // A statically sized batch dimension of one is broadcast and must
        // remain replicated rather than inherit the corresponding sharding.
        if (operandType.getDimSize(i) == 1)
          results.push_back(getAffineConstantExpr(0, ctx));
        else
          results.push_back(getAffineDimExpr(batchRankDiff + i, ctx));
      }
      if (isB) {
        if constexpr (TransposeB) {
          results.push_back(getAffineDimExpr(outputRank - 1, ctx));
          results.push_back(getAffineDimExpr(reductionLoop, ctx));
        } else {
          results.push_back(getAffineDimExpr(reductionLoop, ctx));
          results.push_back(getAffineDimExpr(outputRank - 1, ctx));
        }
      } else {
        results.push_back(getAffineDimExpr(outputRank - 2, ctx));
        results.push_back(getAffineDimExpr(reductionLoop, ctx));
      }
      return AffineMap::get(loopRank, 0, results, ctx);
    };

    SmallVector<unsigned> outputTargets;
    outputTargets.reserve(outputRank);
    for (int64_t i = 0; i < outputRank; ++i)
      outputTargets.push_back(i);

    SmallVector<AffineMap> maps;
    maps.push_back(getOperandMap(aType, /*isB=*/false));
    maps.push_back(getOperandMap(bType, /*isB=*/true));
    maps.push_back(AffineMap::get(loopRank, 0, {}, ctx));
    maps.push_back(AffineMap::get(loopRank, 0, {}, ctx));
    maps.push_back(
        AffineMap::getMultiDimMapWithTargets(loopRank, outputTargets, ctx));
    return maps;
  }

  FailureOr<ShardingOption>
  getShardingOption(Operation *op, ArrayRef<Sharding> operandShardings,
                    ArrayRef<Sharding> resultShardings) const {
    // Decline propagation when a dynamic batch dimension makes the operand
    // indexing ambiguous between replication and the corresponding loop.
    if (getIndexingMaps(op).empty())
      return ShardingOption::makeEmpty();
    return shard::detail::defaultGetShardingOption(op, operandShardings,
                                                   resultShardings);
  }
};

using MatMulOpSharding = MatMulSharding<MatMulOp, /*TransposeB=*/false>;
using MatMulTOpSharding = MatMulSharding<MatMulTOp, /*TransposeB=*/true>;

struct NegateOpSharding
    : public ShardingInterface::ExternalModel<NegateOpSharding, NegateOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    Value val = op->getOperand(0);
    auto type = dyn_cast<RankedTensorType>(val.getType());
    if (!type)
      return {};
    SmallVector<utils::IteratorType> types(type.getRank(),
                                           utils::IteratorType::parallel);
    return types;
  }

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    MLIRContext *ctx = op->getContext();
    Value val = op->getOperand(0);
    auto type = dyn_cast<RankedTensorType>(val.getType());
    if (!type)
      return {};
    int64_t rank = type.getRank();
    SmallVector<AffineMap> maps = {
        AffineMap::getMultiDimIdentityMap(rank, ctx),
        AffineMap::get(0, 0, {}, ctx), AffineMap::get(0, 0, {}, ctx),
        AffineMap::getMultiDimIdentityMap(rank, ctx)};
    return maps;
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitiondOperands,
                          ArrayRef<Sharding> operandShardings,
                          ArrayRef<Sharding> resultShardings,
                          IRMapping &partitionMap,
                          SymbolTableCollection &symbolTable,
                          OpBuilder &builder) const {
    partitionTriviallyShardableOperation(*op, partitiondOperands,
                                         operandShardings, resultShardings,
                                         partitionMap, symbolTable, builder);
    return success();
  }
};

template <typename OpType>
static void registerElemwiseOne(MLIRContext *ctx) {
  OpType::template attachInterface<ElementwiseShardingInterface<OpType>>(*ctx);
}

/// Variadic helper function.
template <typename... OpTypes>
static void registerElemwiseAll(MLIRContext *ctx) {
  (registerElemwiseOne<OpTypes>(ctx), ...);
}

} // namespace

void mlir::tosa::registerShardingInterfaceExternalModels(
    DialectRegistry &registry) {

  registry.addExtension(+[](MLIRContext *ctx, TosaDialect *dialect) {
    registerElemwiseAll<
        ClampOp, SigmoidOp, TanhOp, AddOp, ArithmeticRightShiftOp, BitwiseAndOp,
        BitwiseOrOp, BitwiseXorOp, IntDivOp, LogicalAndOp, LogicalLeftShiftOp,
        LogicalRightShiftOp, LogicalOrOp, LogicalXorOp, MaximumOp, MinimumOp,
        MulOp, PowOp, SubOp, AbsOp, BitwiseNotOp, CeilOp, ClzOp, ExpOp, FloorOp,
        LogOp, LogicalNotOp, ReciprocalOp, RsqrtOp, SelectOp, EqualOp,
        GreaterOp, GreaterEqualOp>(ctx);

    MatMulOp::attachInterface<MatMulOpSharding>(*ctx);
    MatMulTOp::attachInterface<MatMulTOpSharding>(*ctx);
    NegateOp::attachInterface<NegateOpSharding>(*ctx);
  });
}
