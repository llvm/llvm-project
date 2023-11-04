//===- ShardingInterfaceImpl.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/ShardingInterfaceImpl.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tosa-sharding-impl"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::tosa;
using namespace mlir::mesh;

namespace {

template <typename ElemwiseOp>
struct ElemwiseSharding
    : public ShardingInterface::ExternalModel<ElemwiseSharding<ElemwiseOp>,
                                              ElemwiseOp> {
  SmallVector<IteratorType> getLoopIteratorTypes(Operation *op) const {
    Value val = op->getOperand(0);
    auto type = val.getType().dyn_cast<RankedTensorType>();
    if (!type)
      return {};
    SmallVector<IteratorType> types(type.getRank(), IteratorType::Parallel);
    return types;
  }

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    MLIRContext *ctx = op->getContext();
    Value val = op->getOperand(0);
    auto type = val.getType().dyn_cast<RankedTensorType>();
    if (!type)
      return {};
    int64_t rank = type.getRank();
    int64_t num = op->getNumOperands() + op->getNumResults();
    SmallVector<AffineMap> maps(num,
                                AffineMap::getMultiDimIdentityMap(rank, ctx));
    return maps;
  }
};

// loop types: [parallel, parallel, parallel, reduction_sum]
// indexing maps:
// (d0, d1, d2, d3) -> (d0, d1, d3)
// (d0, d1, d2, d3) -> (d0, d3, d2)
// (d0, d1, d2, d3) -> (d0, d1, d2)
struct MatMulOpSharding
    : public ShardingInterface::ExternalModel<MatMulOpSharding, MatMulOp> {
  SmallVector<IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto tensorType = op->getResult(0).getType().dyn_cast<RankedTensorType>();
    if (!tensorType)
      return {};

    SmallVector<IteratorType> types(tensorType.getRank() + 1,
                                    IteratorType::Parallel);
    types[tensorType.getRank()] = IteratorType::ReductionSum;
    return types;
  }

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    auto tensorType = op->getResult(0).getType().dyn_cast<RankedTensorType>();
    if (!tensorType)
      return {};
    MLIRContext *ctx = op->getContext();
    SmallVector<AffineMap> maps;
    maps.push_back(AffineMap::getMultiDimMapWithTargets(4, {0, 1, 3}, ctx));
    maps.push_back(AffineMap::getMultiDimMapWithTargets(4, {0, 3, 2}, ctx));
    maps.push_back(AffineMap::getMultiDimMapWithTargets(4, {0, 1, 2}, ctx));
    return maps;
  }
};

template <typename OpType>
static void registerElemwiseOne(MLIRContext *ctx) {
  OpType::template attachInterface<ElemwiseSharding<OpType>>(*ctx);
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
        BitwiseOrOp, BitwiseXorOp, DivOp, LogicalAndOp, LogicalLeftShiftOp,
        LogicalRightShiftOp, LogicalOrOp, LogicalXorOp, MaximumOp, MinimumOp,
        MulOp, PowOp, SubOp, AbsOp, BitwiseNotOp, CeilOp, ClzOp, ExpOp, FloorOp,
        LogOp, LogicalNotOp, NegateOp, ReciprocalOp, RsqrtOp, SelectOp, EqualOp,
        GreaterOp, GreaterEqualOp>(ctx);

    MatMulOp::attachInterface<MatMulOpSharding>(*ctx);
  });
}
