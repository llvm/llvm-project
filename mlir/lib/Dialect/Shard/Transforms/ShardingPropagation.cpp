//===- ShardingPropagation.cpp ------------------------------------- C++ --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Shard/Transforms/Passes.h"

#include "mlir/Dialect/Shard/IR/ShardDialect.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/Dialect/Shard/Interfaces/ShardingInterface.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <vector>

namespace mlir {
namespace shard {
#define GEN_PASS_DEF_SHARDINGPROPAGATION
#include "mlir/Dialect/Shard/Transforms/Passes.h.inc"
} // namespace shard
} // namespace mlir

#define DEBUG_TYPE "sharding-propagation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::shard;

enum class ReshardingRquirementKind {
  NO_RESHARDING = 0,
  NO_RESHARDING_FOR_EXPLICIT_ANNOTATIONS,
  RESHARDING_FOR_EXPLICIT_ANNOTATIONS
};

#ifdef LLVM_DEBUG

template <typename T>
static llvm::raw_ostream &operator<<(llvm::raw_ostream &stream,
                                     const SmallVector<T> &vec);
template <typename... Ts>
static llvm::raw_ostream &operator<<(llvm::raw_ostream &stream,
                                     const std::tuple<Ts...> &t);
static llvm::raw_ostream &operator<<(llvm::raw_ostream &stream,
                                     ReshardingRquirementKind v);

template <typename Stream, typename Range>
static Stream &printRange(Stream &stream, Range &&range) {
  stream << "[";
  for (auto &v : range) {
    stream << v;
    stream << ", ";
  }
  return stream << "]";
}

template <typename T>
static llvm::raw_ostream &operator<<(llvm::raw_ostream &stream,
                                     const SmallVector<T> &vec) {
  return printRange(stream, vec);
}

[[maybe_unused]] static llvm::raw_ostream &operator<<(llvm::raw_ostream &stream,
                                                      const ShardingOption &v) {
  return stream << "{empty = " << v.empty << ", grid" << v.grid
                << ", shardingArray = " << v.shardingArray << "}";
}

template <typename Stream, typename... Ts, size_t... Is>
static Stream &printTuple(Stream &stream, std::tuple<Ts...> tuple,
                          std::index_sequence<Is...>) {
  static_assert(sizeof...(Is) == sizeof...(Ts),
                "Indices must have same number of elements as tuple types!");
  static_assert(sizeof...(Ts) > 0, "Cannot insert empty tuple into stream.");

  stream << "{";
  ((stream << std::get<Is>(tuple) << ", "), ...);
  return stream << "}";
}

template <typename... Ts>
static llvm::raw_ostream &operator<<(llvm::raw_ostream &stream,
                                     const std::tuple<Ts...> &t) {
  return printTuple(stream, t, std::index_sequence_for<Ts...>{});
}

[[maybe_unused]] static llvm::raw_ostream &
operator<<(llvm::raw_ostream &stream, ReshardingRquirementKind v) {
  return stream << static_cast<int>(v);
}

#endif // LLVM_DEBUG

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// This method retrieves all potential sharding attributes, prioritizing
// specific shardings. For example, mustShardings = [shard0, None] and
// optionalShardings = [None, shard1], the result will be [[shard0, shard1],
// [shard0, None]]
static SmallVector<std::vector<Sharding>>
getOrderedPossibleShardingAttrs(ArrayRef<Sharding> mustShardings,
                                ArrayRef<Sharding> optionalShardings) {
  SmallVector<std::vector<Sharding>> allShardingAttrs;
  std::vector<Sharding> curShardingAttrs;

  std::function<void(size_t)> dfsCreateShardingAttrs = [&](size_t i) {
    if (i == mustShardings.size()) {
      allShardingAttrs.push_back(std::vector<Sharding>(curShardingAttrs));
      return;
    }

    if (mustShardings[i]) {
      curShardingAttrs.push_back(mustShardings[i]);
      dfsCreateShardingAttrs(i + 1);
      curShardingAttrs.pop_back();
      return;
    }

    if (optionalShardings[i]) {
      curShardingAttrs.push_back(optionalShardings[i]);
      dfsCreateShardingAttrs(i + 1);
      curShardingAttrs.pop_back();
      curShardingAttrs.emplace_back();
      dfsCreateShardingAttrs(i + 1);
      curShardingAttrs.pop_back();
      return;
    }

    curShardingAttrs.emplace_back();
    dfsCreateShardingAttrs(i + 1);
    curShardingAttrs.pop_back();
  };

  dfsCreateShardingAttrs(0);
  return allShardingAttrs;
}

// The order of preference is form highest to lowest:
// 1. No resharding is required (all existing annotations are compatible).
// 2. No resharding for operands/results that have annotation specifically
//   targeting this operation. This means
//   * operands that are the result of `shard.shard` ops marked with
//     `annotate_for_users`.
//   * results that are annotated with `shard.shard` ops without
//     `annotate_for_users`.
// 3. All other cases. Resharding is required for operands/results with
//   annotation targeting explicitly this operation.
ReshardingRquirementKind getReshardingRquirementKind(
    Operation *op, const std::vector<Sharding> &operandAndResultShardings) {
  ReshardingRquirementKind res = ReshardingRquirementKind::NO_RESHARDING;

  size_t operandsCount = op->getOperands().size();
  auto operandShardings =
      llvm::make_range(operandAndResultShardings.begin(),
                       operandAndResultShardings.begin() + operandsCount);
  auto resultShardings =
      llvm::make_range(operandAndResultShardings.begin() + operandsCount,
                       operandAndResultShardings.end());

  for (auto [operand, sharding] :
       llvm::zip_equal(op->getOperands(), operandShardings)) {
    ShardOp shardOp = llvm::dyn_cast_or_null<ShardOp>(operand.getDefiningOp());
    if (!shardOp) {
      continue;
    }
    bool needsResharding = sharding != shardOp.getSharding();
    bool isExplicitAnnotationForThisOp = shardOp.getAnnotateForUsers();
    if (needsResharding) {
      if (isExplicitAnnotationForThisOp) {
        // This is the worst case. No need to continue.
        return ReshardingRquirementKind::RESHARDING_FOR_EXPLICIT_ANNOTATIONS;
      }
      res = ReshardingRquirementKind::NO_RESHARDING_FOR_EXPLICIT_ANNOTATIONS;
    }
  }

  for (auto [result, sharding] :
       llvm::zip_equal(op->getResults(), resultShardings)) {
    for (auto *user : result.getUsers()) {
      ShardOp shardOp = llvm::dyn_cast<ShardOp>(user);
      if (!shardOp) {
        continue;
      }
      bool needsResharding = sharding != shardOp.getSharding();
      bool isExplicitAnnotationForThisOp = !shardOp.getAnnotateForUsers();
      if (needsResharding) {
        if (isExplicitAnnotationForThisOp) {
          // This is the worst case. No need to continue.
          return ReshardingRquirementKind::RESHARDING_FOR_EXPLICIT_ANNOTATIONS;
        }
        res = ReshardingRquirementKind::NO_RESHARDING_FOR_EXPLICIT_ANNOTATIONS;
      }
    }
  }

  return res;
}

// From all the operand and result sharding combinations,
// return the one that is most desirable.
// The order of preference is:
// 1. No resharding with respect to existing sharding annotations.
// 2. Resharding for values that have already annotations that do not target
//    this op.
// 3. Resharding of existing explicit sharding annotations for this op.
static FailureOr<ShardingOption> selectShardingOption(
    ShardingInterface shardingOp,
    ArrayRef<std::vector<Sharding>> possibleOperandShardingAttrs,
    ArrayRef<std::vector<Sharding>> possibleResultShardingAttrs) {
  SmallVector<std::tuple<ShardingOption, ReshardingRquirementKind>>
      shardingOptionsAndReshardingRequirements;

  for (ArrayRef<Sharding> resultShardings : possibleResultShardingAttrs) {
    for (ArrayRef<Sharding> operandShardings : possibleOperandShardingAttrs) {
      FailureOr<ShardingOption> shardingOption =
          shardingOp.getShardingOption(operandShardings, resultShardings);
      if (failed(shardingOption) || shardingOption->empty) {
        continue;
      }
      // These shardings may not be the same as those in operandShardings and
      // resultShardings.
      // They may be missing some annotations.
      // Whatever is returned by getShardingAnnotations is exactly what the op
      // needs.
      FailureOr<std::vector<Sharding>> operandAndResultShardings =
          shardingOp.getShardingAnnotations(*shardingOption);
      if (failed(operandAndResultShardings)) {
        return failure();
      }

      // LLVM_DEBUG(DBGS() << "operandAndResultShardings = "
      //                   << *operandAndResultShardings << "\n";);

      ReshardingRquirementKind reshardingRquirement =
          getReshardingRquirementKind(shardingOp, *operandAndResultShardings);
      if (reshardingRquirement == ReshardingRquirementKind::NO_RESHARDING) {
        // This is the best case. No need to go on.
        return *shardingOption;
      }

      shardingOptionsAndReshardingRequirements.emplace_back(
          std::move(*shardingOption), reshardingRquirement);
    }
  }

  if (shardingOptionsAndReshardingRequirements.empty()) {
    return ShardingOption::makeEmpty();
  }

  std::partial_sort(
      shardingOptionsAndReshardingRequirements.begin(),
      shardingOptionsAndReshardingRequirements.begin() + 1,
      shardingOptionsAndReshardingRequirements.end(),
      [](const std::tuple<ShardingOption, ReshardingRquirementKind> &a,
         const std::tuple<ShardingOption, ReshardingRquirementKind> &b) {
        return std::get<ReshardingRquirementKind>(a) <
               std::get<ReshardingRquirementKind>(b);
      });

  LLVM_DEBUG(DBGS() << "shardingOptionsAndReshardingRequirements = "
                    << shardingOptionsAndReshardingRequirements << "\n";);

  return std::get<ShardingOption>(
      shardingOptionsAndReshardingRequirements.front());
}

// For each operation that implements the ShardingInterface, infer the sharding
// option of the operation from its operands and/or results using the
// `getShardingOption` method. If the inferred sharding option is not empty, add
// a `shard.shard` operation for all remaining operands and results that do not
// have sharding annotations.
static LogicalResult visitOp(Operation *op, OpBuilder &builder) {
  ShardingInterface shardingOp = llvm::dyn_cast<ShardingInterface>(op);
  if (op->hasTrait<OpTrait::IsTerminator>() ||
      (op->hasTrait<OpTrait::ConstantLike>() && !shardingOp) ||
      llvm::isa<shard::ShardOp, shard::ShardingOp, shard::GetShardingOp>(op))
    return success();

  if (!shardingOp) {
    op->emitOpError() << "sharding interface is not implemented.";
    return failure();
  }

  // collect Sharding from results
  std::vector<Sharding> allowConflictsResultShardings;
  allowConflictsResultShardings.resize(op->getNumResults());
  std::vector<Sharding> resultMustShardings;
  resultMustShardings.resize(op->getNumResults());
  for (OpResult result : op->getResults()) {
    FailureOr<std::pair<bool, Sharding>> maybeShardAttr = getSharding(result);
    if (failed(maybeShardAttr))
      continue;
    if (!maybeShardAttr->first)
      resultMustShardings[result.getResultNumber()] = maybeShardAttr->second;
    else
      allowConflictsResultShardings[result.getResultNumber()] =
          maybeShardAttr->second;
  }

  // collect Sharding from operands
  std::vector<Sharding> allowConflictsOperandShardings;
  allowConflictsOperandShardings.resize(op->getNumOperands());
  std::vector<Sharding> operandMustShardings;
  operandMustShardings.resize(op->getNumOperands());
  for (OpOperand &opOperand : op->getOpOperands()) {
    FailureOr<std::pair<bool, Sharding>> maybeShardAttr =
        getSharding(opOperand);
    if (failed(maybeShardAttr))
      continue;

    if (maybeShardAttr->first)
      operandMustShardings[opOperand.getOperandNumber()] =
          maybeShardAttr->second;
    else
      allowConflictsOperandShardings[opOperand.getOperandNumber()] =
          maybeShardAttr->second;
  }

  // try to get the sharding option
  SmallVector<std::vector<Sharding>> possibleOperandShardingAttrs =
      getOrderedPossibleShardingAttrs(operandMustShardings,
                                      allowConflictsOperandShardings);
  SmallVector<std::vector<Sharding>> possibleResultShardingAttrs =
      getOrderedPossibleShardingAttrs(resultMustShardings,
                                      allowConflictsResultShardings);
  FailureOr<ShardingOption> shardingOption = selectShardingOption(
      shardingOp, possibleOperandShardingAttrs, possibleResultShardingAttrs);

  if (failed(shardingOption)) {
    op->emitOpError() << "fail to get sharding option.";
    return failure();
  }

  LLVM_DEBUG(DBGS() << "Selected sharding option: " << *shardingOption << "\n");

  // sharding info is empty, return immediately
  if (shardingOption->empty)
    return success();

  if (failed(shardingOp.addShardingAnnotations(builder, *shardingOption))) {
    op->emitOpError() << "fail to set sharding annotations.";
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ShardingPropagation
//===----------------------------------------------------------------------===//
struct ShardingPropagation
    : public shard::impl::ShardingPropagationBase<ShardingPropagation> {

  using ShardingPropagationBase<ShardingPropagation>::ShardingPropagationBase;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    Region &region = funcOp.getFunctionBody();
    OpBuilder builder(ctx);
    if (!region.hasOneBlock()) {
      funcOp.emitOpError() << "only one block is supported!";
      return signalPassFailure();
    }
    Block &block = region.front();

    LLVM_DEBUG(
        DBGS() << "print all the ops' iterator types and indexing maps in the "
                  "block.\n";
        for (Operation &op
             : block.getOperations()) {
          if (auto shardingOp = llvm::dyn_cast<ShardingInterface>(&op))
            shardingOp.printLoopTypesAndIndexingMaps(llvm::dbgs());
        });

    auto traverse = [&](auto &&range, OpBuilder &builder,
                        const char *order) -> bool {
      for (Operation &op : range) {
        if (failed(visitOp(&op, builder))) {
          signalPassFailure();
          return true;
        }
      }
      LLVM_DEBUG(DBGS() << "After " << order << " order propagation:\n"
                        << funcOp << "\n");
      LLVM_DEBUG(assert(succeeded(mlir::verify(funcOp))));
      return false;
    };

    // 1. Propagate in reversed order.
    if (traversal == TraversalOrder::Backward ||
        traversal == TraversalOrder::BackwardForward)
      traverse(llvm::reverse(block), builder, "backward");

    // 2. Propagate in original order.
    if (traversal != TraversalOrder::Backward)
      traverse(block, builder, "forward");

    // 3. Propagate in backward order if needed.
    if (traversal == TraversalOrder::ForwardBackward)
      traverse(llvm::reverse(block), builder, "backward");
  }
};
