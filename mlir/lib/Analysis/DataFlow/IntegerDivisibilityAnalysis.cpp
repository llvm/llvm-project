//===- IntegerDivisibilityAnalysis.cpp - Integer divisibility ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the dataflow analysis class for integer divisibility
// inference. Operations participate in the analysis by implementing
// `InferIntDivisibilityOpInterface`.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/IntegerDivisibilityAnalysis.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "int-divisibility-analysis"

using llvm::dbgs;

namespace mlir::dataflow {

void IntegerDivisibilityAnalysis::setToEntryState(
    IntegerDivisibilityLattice *lattice) {
  propagateIfChanged(lattice,
                     lattice->join(IntegerDivisibility::getMinDivisibility()));
}

LogicalResult IntegerDivisibilityAnalysis::visitOperation(
    Operation *op, ArrayRef<const IntegerDivisibilityLattice *> operands,
    ArrayRef<IntegerDivisibilityLattice *> results) {
  auto inferrable = dyn_cast<InferIntDivisibilityOpInterface>(op);
  if (!inferrable) {
    setAllToEntryStates(results);
    return success();
  }

  LLVM_DEBUG(dbgs() << "Inferring divisibility for " << *op << "\n");
  auto argDivs = llvm::map_to_vector(
      operands, [](const IntegerDivisibilityLattice *lattice) {
        return lattice->getValue();
      });
  auto joinCallback = [&](Value v, const IntegerDivisibility &newDiv) {
    auto result = dyn_cast<OpResult>(v);
    if (!result) {
      return;
    }
    assert(llvm::is_contained(op->getResults(), result));

    LLVM_DEBUG(dbgs() << "Inferred divisibility " << newDiv << "\n");
    IntegerDivisibilityLattice *lattice = results[result.getResultNumber()];
    IntegerDivisibility oldDiv = lattice->getValue();

    ChangeResult changed = lattice->join(newDiv);

    // Catch loop results with loop-variant divisibility and conservatively
    // set them to divisibility 1 (no information) so we don't ratchet
    // indefinitely (the dataflow analysis in MLIR doesn't attempt to work
    // out trip counts and often can't).
    bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
      return op->hasTrait<OpTrait::IsTerminator>();
    });
    if (isYieldedResult && !oldDiv.isUninitialized() &&
        !(lattice->getValue() == oldDiv)) {
      LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
      changed |= lattice->join(IntegerDivisibility::getMinDivisibility());
    }
    propagateIfChanged(lattice, changed);
  };

  inferrable.inferResultDivisibility(argDivs, joinCallback);
  return success();
}

void IntegerDivisibilityAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor, ValueRange successorInputs,
    ArrayRef<IntegerDivisibilityLattice *> argLattices) {
  enum class LoopKind { Signed, Unsigned, Unknown };
  auto getLoopKind = [](Operation *loopOp) {
    StringRef name = loopOp->getName().getStringRef();
    if (name == "scf.for" || name == "scf.parallel")
      return loopOp->hasAttr("unsignedCmp") ? LoopKind::Unsigned
                                            : LoopKind::Signed;
    if (name == "scf.forall" || name == "affine.for" ||
        name == "affine.parallel")
      return LoopKind::Signed;
    return LoopKind::Unknown;
  };
  // Only factors of 2^width survive a change between signed and unsigned
  // representations, or truncation to a narrower index type.
  auto commonBitwidthDivisor = [](uint64_t div, unsigned width) {
    if (div == 0)
      return div;
    return width < 64 ? std::gcd(div, uint64_t(1) << width) : div & -div;
  };
  auto getDivFromOfr = [&](OpFoldResult ofr, Block *block, unsigned width,
                           bool isUnsigned) -> uint64_t {
    if (auto constant = getConstantIntValue(ofr)) {
      APInt bits = APInt(64, static_cast<uint64_t>(*constant)).trunc(width);
      if (isUnsigned)
        return bits.getZExtValue();
      int64_t signedValue = bits.getSExtValue();
      // Unsigned subtraction also handles INT64_MIN.
      return signedValue < 0 ? uint64_t(0) - uint64_t(signedValue)
                             : uint64_t(signedValue);
    }
    if (auto value = dyn_cast<Value>(ofr)) {
      const IntegerDivisibilityLattice *lattice =
          getLatticeElementFor(getProgramPointBefore(block), value);
      if (lattice != nullptr && !lattice->getValue().isUninitialized()) {
        uint64_t div = isUnsigned ? lattice->getValue().getValue().udiv()
                                  : lattice->getValue().getValue().sdiv();
        if (!value.getType().isIndex())
          return div;
        // Only a known signed loop IV keeps its signed divisibility across
        // index widths. Other dynamic index expressions may wrap at 32 bits.
        if (!isUnsigned) {
          if (auto definingLoop = dyn_cast<LoopLikeOpInterface>(
                  value.getParentBlock()->getParentOp())) {
            if (getLoopKind(definingLoop.getOperation()) == LoopKind::Signed) {
              if (auto ivs = definingLoop.getLoopInductionVars();
                  ivs && llvm::is_contained(*ivs, value))
                return div;
            }
          }
        }
        return commonBitwidthDivisor(div, 32);
      }
    }
    return isUnsigned
               ? IntegerDivisibility::getMinDivisibility().getValue().udiv()
               : IntegerDivisibility::getMinDivisibility().getValue().sdiv();
  };

  // Infer divisibility for loop induction variables from their bounds and
  // steps.
  if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
    LoopKind loopKind = getLoopKind(op);
    std::optional<SmallVector<Value>> ivs = loop.getLoopInductionVars();
    std::optional<SmallVector<OpFoldResult>> lbs = loop.getLoopLowerBounds();
    std::optional<SmallVector<OpFoldResult>> ubs = loop.getLoopUpperBounds();
    std::optional<SmallVector<OpFoldResult>> steps = loop.getLoopSteps();
    if (!ivs || !lbs || !steps) {
      return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
          op, successor, successorInputs, argLattices);
    }
    unsigned ivIndex = 0;
    for (auto [iv, lb, step] : llvm::zip_equal(*ivs, *lbs, *steps)) {
      unsigned currentIv = ivIndex++;
      IntegerDivisibilityLattice *ivEntry = getLatticeElement(iv);
      // The interface alone does not specify the IV sequence or comparison.
      if (loopKind == LoopKind::Unknown) {
        setToEntryState(ivEntry);
        continue;
      }
      Block *block = iv.getParentBlock();
      unsigned width = iv.getType().isIndex()
                           ? IndexType::kInternalStorageBitWidth
                           : cast<IntegerType>(iv.getType()).getWidth();
      if (width > 64) {
        setToEntryState(ivEntry);
        continue;
      }
      OpFoldResult lowerBound = lb;
      OpFoldResult loopStep = step;
      auto inferAtWidth = [&](unsigned bits) {
        uint64_t lbUDiv = getDivFromOfr(lowerBound, block, bits, true);
        uint64_t lbSDiv = getDivFromOfr(lowerBound, block, bits, false);
        uint64_t stepUDiv = getDivFromOfr(loopStep, block, bits, true);
        uint64_t stepSDiv = getDivFromOfr(loopStep, block, bits, false);
        uint64_t udiv = std::gcd(lbUDiv, stepUDiv);
        uint64_t sdiv = std::gcd(lbSDiv, stepSDiv);

        // The loop comparison keeps the IV in its native signed or unsigned
        // range. In the other interpretation, a sign-boundary crossing can
        // change a value by 2^bits, leaving only common power-of-two factors.
        auto signedConstant =
            [bits](OpFoldResult ofr) -> std::optional<int64_t> {
          if (auto constant = getConstantIntValue(ofr))
            return APInt(64, static_cast<uint64_t>(*constant))
                .trunc(bits)
                .getSExtValue();
          return std::nullopt;
        };
        std::optional<int64_t> constantLb = signedConstant(lowerBound);
        std::optional<int64_t> constantUb;
        if (ubs && currentIv < ubs->size())
          constantUb = signedConstant((*ubs)[currentIv]);
        if (loopKind == LoopKind::Unsigned) {
          if (!constantLb || !constantUb ||
              (*constantLb < 0 && *constantUb > 0) ||
              (*constantLb >= 0 && *constantUb < 0))
            sdiv = commonBitwidthDivisor(udiv, bits);
        } else if (!constantLb ||
                   (*constantLb < 0 && (!constantUb || *constantUb > 0))) {
          udiv = commonBitwidthDivisor(sdiv, bits);
        }
        return ConstantIntDivisibility(udiv, sdiv);
      };
      ConstantIntDivisibility ivDiv = inferAtWidth(width);
      if (iv.getType().isIndex())
        ivDiv = ivDiv.getUnion(inferAtWidth(32));
      propagateIfChanged(ivEntry, ivEntry->join(ivDiv));
    }
    return;
  }

  return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
      op, successor, successorInputs, argLattices);
}

} // namespace mlir::dataflow
