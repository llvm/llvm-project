//===- IntegerDivisibility.cpp - Divisibility analysis -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Analysis/IntegerDivisibility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/STLExtras.h"

#include <numeric>

using namespace mlir;
using namespace mlir::arith;

LogicalResult PowerOfTwoDivisibilityAnalysis::visitOperation(
    Operation *op,
    ArrayRef<const dataflow::IntegerDivisibilityLattice *> operands,
    ArrayRef<dataflow::IntegerDivisibilityLattice *> results) {
  if (auto constant = dyn_cast<ConstantOp>(op)) {
    auto attr = dyn_cast<IntegerAttr>(constant.getValue());
    if (!attr) {
      setAllToEntryStates(results);
      return success();
    }
    const APInt &value = attr.getValue();
    uint64_t divisor =
        value.isZero() ? 0 : uint64_t{1} << std::min(value.countr_zero(), 63u);
    propagateIfChanged(
        results.front(),
        results.front()->join(ConstantIntDivisibility(divisor, divisor)));
    return success();
  }
  // Division does not preserve the modular-arithmetic proof, even if a
  // general (non-power-of-two) divisor of its numerator is known.
  if (!isa<AddIOp, SubIOp, MulIOp, SelectOp>(op)) {
    setAllToEntryStates(results);
    return success();
  }
  if (llvm::any_of(operands, [](const auto *lattice) {
        return lattice->getValue().isUninitialized();
      }))
    return success();

  if (isa<MulIOp>(op)) {
    uint64_t lhs = operands[0]->getValue().getValue().udiv();
    uint64_t rhs = operands[1]->getValue().getValue().udiv();
    if (lhs && rhs && lhs > UINT64_MAX / rhs) {
      // Each nonzero operand divisor is a power of two. A product outside
      // uint64_t is still divisible by 2^63, but need not be exactly zero.
      uint64_t divisor = uint64_t{1} << 63;
      propagateIfChanged(
          results.front(),
          results.front()->join(ConstantIntDivisibility(divisor, divisor)));
      return success();
    }
  }
  return IntegerDivisibilityAnalysis::visitOperation(op, operands, results);
}

void PowerOfTwoDivisibilityAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor, ValueRange successorInputs,
    ArrayRef<dataflow::IntegerDivisibilityLattice *> argLattices) {
  auto loop = dyn_cast<LoopLikeOpInterface>(op);
  auto ivs =
      loop ? loop.getLoopInductionVars() : std::optional<SmallVector<Value>>();
  // The existing loop-bound query represents constants in int64_t.
  if (ivs && llvm::any_of(*ivs, [](Value iv) {
        auto type = dyn_cast<IntegerType>(iv.getType());
        return type && type.getWidth() > 64;
      })) {
    return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
        op, successor, successorInputs, argLattices);
  }
  IntegerDivisibilityAnalysis::visitNonControlFlowArguments(
      op, successor, successorInputs, argLattices);
  if (!ivs)
    return;
  // Reuse the loop-bound/step inference, retaining only its modular facts.
  for (Value iv : *ivs) {
    auto *lattice = getLatticeElement(iv);
    if (lattice->getValue().isUninitialized())
      continue;
    const auto &div = lattice->getValue().getValue();
    uint64_t udiv = div.udiv() & (uint64_t{0} - div.udiv());
    uint64_t sdiv = div.sdiv() & (uint64_t{0} - div.sdiv());
    // Both interpretations have the same power-of-two factors. Joining also
    // discards odd factors before any arithmetic transfer consumes the IV.
    uint64_t divisor = std::gcd(udiv, sdiv);
    propagateIfChanged(
        lattice, lattice->join(ConstantIntDivisibility(divisor, divisor)));
  }
}
