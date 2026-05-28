//===- XeGPUCoalesceGatherScatter.cpp - Coalesce scatter accesses --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass coalesces neighbouring lanes of `xegpu.load` / `xegpu.store` ops
// so that each lane handles `N` contiguous elements along the innermost
// dimension. The decision is driven by a small XeGPU-local axis-info
// dataflow analysis modeled on Triton's `AxisInfo` (`contiguity`,
// `constancy`, `divisibility`) and is applied by attaching a
// `lane_data` layout to the original op. The actual memory-message rewrite
// is left to the downstream WG-to-SG / SG-to-Lane distribution passes,
// which interpret `lane_data`.
//
// The analysis tracks per-axis information for vectors of integer / index
// type at any rank. The coalescing decision is computed against the
// innermost dimension. 2-D offsets vectors with a leading unit dimension
// (e.g. `vector<1x32xindex>`) are handled by treating the inner dim as the
// lane dim.
//
// The `Broadcast` case (constancy along the innermost dim equals the inner
// length) is special: there is no layout-only encoding for "all lanes load
// the same scalar", so for loads we still rewrite to a length-1
// `xegpu.load` followed by `vector.broadcast`. Stores in this shape are
// skipped (last-writer-wins is ambiguous).
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Dialect/XeGPU/uArch/IntelGpuXe2.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>
#include <optional>

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUCOALESCEGATHERSCATTER
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-coalesce-gather-scatter"

using namespace mlir;

// AxisInfo and AxisInfoAnalysis are intentionally placed in a named namespace
// (not anonymous) so the `dataflow::Lattice<AxisInfo>` template instantiation
// gets a stable, externally-visible name. The TypeID machinery requires that
// for `MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID` to apply.
namespace mlir::xegpu::detail::axis_dataflow {

//===----------------------------------------------------------------------===//
// AxisInfo: per-axis contiguity / constancy / divisibility lattice.
//===----------------------------------------------------------------------===//

/// Sentinel "very large" value for unconstrained dimensions. Any real shape
/// is far smaller, so component-wise `min` will collapse this to the truth.
static constexpr int64_t kAxisInfoTop = 1LL << 30;

/// Per-dimension axis information for an SSA value of integer / index type.
///   - `contiguity[d]`: the largest N such that consecutive lanes along
///     dimension `d` differ by exactly 1 in runs of length `N` (lane-stride
///     1 contiguity).
///   - `constancy[d]`: the largest N such that consecutive lanes along
///     dimension `d` are all equal in runs of length `N`.
///   - `divisibility[d]`: a power-of-two divisor of every element along
///     dimension `d`.
///   - `knownConstant`: scalar value if the entire vector is uniformly
///     known to be a single constant.
///   - `innerStride`: when set, every "row" along the innermost dimension
///     is an arithmetic progression with this stride. Per-row base may
///     differ across outer indices; per-row alignment is captured by
///     `divisibility[innerDim]`. `innerStride = 1` implies stride-1
///     contiguity; `innerStride = 0` implies inner-dim constancy.
///
/// Pessimistic / entry value: contiguity=1, constancy=1, divisibility=1,
/// innerStride absent.
struct AxisInfo {
  SmallVector<int64_t> contiguity;
  SmallVector<int64_t> constancy;
  SmallVector<int64_t> divisibility;
  std::optional<int64_t> knownConstant;
  std::optional<int64_t> innerStride;

  AxisInfo() = default;

  static AxisInfo getPessimistic(unsigned rank) {
    AxisInfo v;
    v.contiguity.assign(rank, 1);
    v.constancy.assign(rank, 1);
    v.divisibility.assign(rank, 1);
    return v;
  }

  unsigned getRank() const { return contiguity.size(); }
  bool isInitialized() const { return getRank() > 0; }

  bool operator==(const AxisInfo &rhs) const {
    return contiguity == rhs.contiguity && constancy == rhs.constancy &&
           divisibility == rhs.divisibility &&
           knownConstant == rhs.knownConstant && innerStride == rhs.innerStride;
  }

  /// Conservative join. Two values reaching the same SSA value via different
  /// control-flow paths must agree on what holds.
  static AxisInfo join(const AxisInfo &lhs, const AxisInfo &rhs) {
    if (!lhs.isInitialized())
      return rhs;
    if (!rhs.isInitialized())
      return lhs;
    assert(lhs.getRank() == rhs.getRank());
    AxisInfo out;
    unsigned r = lhs.getRank();
    out.contiguity.resize(r);
    out.constancy.resize(r);
    out.divisibility.resize(r);
    for (unsigned d = 0; d < r; ++d) {
      out.contiguity[d] = std::min(lhs.contiguity[d], rhs.contiguity[d]);
      out.constancy[d] = std::min(lhs.constancy[d], rhs.constancy[d]);
      out.divisibility[d] = std::gcd(lhs.divisibility[d], rhs.divisibility[d]);
    }
    if (lhs.knownConstant && rhs.knownConstant &&
        *lhs.knownConstant == *rhs.knownConstant)
      out.knownConstant = lhs.knownConstant;
    if (lhs.innerStride && rhs.innerStride &&
        *lhs.innerStride == *rhs.innerStride)
      out.innerStride = lhs.innerStride;
    return out;
  }

  void print(raw_ostream &os) const {
    os << "contiguity=[";
    llvm::interleaveComma(contiguity, os);
    os << "] constancy=[";
    llvm::interleaveComma(constancy, os);
    os << "] divisibility=[";
    llvm::interleaveComma(divisibility, os);
    os << "]";
    if (knownConstant)
      os << " const=" << *knownConstant;
    if (innerStride)
      os << " innerStride=" << *innerStride;
  }
};

using AxisInfoLattice = dataflow::Lattice<AxisInfo>;

/// Power-of-two divisor of `v`. Returns `kAxisInfoTop` when `v == 0`.
static int64_t highestPow2Divisor(int64_t v) {
  if (v == 0)
    return kAxisInfoTop;
  uint64_t u = static_cast<uint64_t>(std::abs(v));
  return static_cast<int64_t>(u & (~u + 1));
}

/// Initial lattice value for an SSA value when no transfer function applies.
static AxisInfo entryStateFor(Value v) {
  if (auto vt = dyn_cast<VectorType>(v.getType()))
    return AxisInfo::getPessimistic(vt.getRank());
  return AxisInfo::getPessimistic(1);
}

/// AxisInfo for a tensor that is constant `c` everywhere.
static AxisInfo splatAxisInfo(ArrayRef<int64_t> shape, int64_t c) {
  AxisInfo v;
  unsigned r = shape.size();
  v.contiguity.assign(r, 1);
  v.constancy.assign(shape.begin(), shape.end());
  v.divisibility.assign(r, highestPow2Divisor(c));
  v.knownConstant = c;
  v.innerStride = 0;
  return v;
}

/// Sparse forward dataflow analysis that computes `AxisInfo` for vector
/// values reachable from the entry of the analyzed op.
class AxisInfoAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<AxisInfoLattice> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AxisInfoAnalysis)
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const AxisInfoLattice *> operands,
                               ArrayRef<AxisInfoLattice *> results) override {
    if (auto step = dyn_cast<vector::StepOp>(op))
      return visitStep(step, results);
    if (auto cst = dyn_cast<arith::ConstantOp>(op))
      return visitConstant(cst, results);
    if (auto bcast = dyn_cast<vector::BroadcastOp>(op))
      return visitBroadcast(bcast, operands, results);
    if (auto sc = dyn_cast<vector::ShapeCastOp>(op))
      return visitShapeCast(sc, operands, results);
    if (auto tp = dyn_cast<vector::TransposeOp>(op))
      return visitTranspose(tp, operands, results);
    if (auto add = dyn_cast<arith::AddIOp>(op))
      return visitAddSub</*IsSub=*/false>(add, operands, results);
    if (auto sub = dyn_cast<arith::SubIOp>(op))
      return visitAddSub</*IsSub=*/true>(sub, operands, results);
    if (auto mul = dyn_cast<arith::MulIOp>(op))
      return visitMul(mul, operands, results);
    if (auto div = dyn_cast<arith::DivUIOp>(op))
      return visitDivRem</*IsSigned=*/false, /*IsRem=*/false>(div, operands,
                                                              results);
    if (auto div = dyn_cast<arith::DivSIOp>(op))
      return visitDivRem</*IsSigned=*/true, /*IsRem=*/false>(div, operands,
                                                             results);
    if (auto rem = dyn_cast<arith::RemUIOp>(op))
      return visitDivRem</*IsSigned=*/false, /*IsRem=*/true>(rem, operands,
                                                             results);
    if (auto rem = dyn_cast<arith::RemSIOp>(op))
      return visitDivRem</*IsSigned=*/true, /*IsRem=*/true>(rem, operands,
                                                            results);
    if (auto andi = dyn_cast<arith::AndIOp>(op))
      return visitAndI(andi, operands, results);
    if (auto shl = dyn_cast<arith::ShLIOp>(op))
      return visitShift</*IsLeft=*/true>(shl, operands, results);
    if (auto shr = dyn_cast<arith::ShRUIOp>(op))
      return visitShift</*IsLeft=*/false>(shr, operands, results);
    if (auto sel = dyn_cast<arith::SelectOp>(op))
      return visitSelect(sel, operands, results);
    if (auto cast = dyn_cast<arith::IndexCastOp>(op))
      return visitPassThrough(cast, operands, results);
    if (auto cast = dyn_cast<arith::IndexCastUIOp>(op))
      return visitPassThrough(cast, operands, results);
    setAllPessimistic(op, results);
    return success();
  }

  void setToEntryState(AxisInfoLattice *lattice) override {
    propagateIfChanged(lattice,
                       lattice->join(entryStateFor(lattice->getAnchor())));
  }

private:
  void setAllPessimistic(Operation *op, ArrayRef<AxisInfoLattice *> results) {
    for (auto [r, lat] : llvm::zip(op->getResults(), results)) {
      AxisInfo state = entryStateFor(r);
      propagateIfChanged(lat, lat->join(state));
    }
  }

  // vector.step is always 1-D and produces [0, 1, ..., n-1].
  LogicalResult visitStep(vector::StepOp op,
                          ArrayRef<AxisInfoLattice *> results) {
    auto vt = cast<VectorType>(op.getType());
    int64_t n = vt.getNumElements();
    AxisInfo v;
    v.contiguity = {n};
    v.constancy = {1};
    v.divisibility = {kAxisInfoTop};
    v.innerStride = 1;
    propagateIfChanged(results[0], results[0]->join(v));
    return success();
  }

  LogicalResult visitConstant(arith::ConstantOp op,
                              ArrayRef<AxisInfoLattice *> results) {
    auto vt = dyn_cast<VectorType>(op.getType());
    if (!vt) {
      if (auto intAttr = dyn_cast<IntegerAttr>(op.getValue())) {
        int64_t c = intAttr.getValue().getSExtValue();
        AxisInfo v;
        v.contiguity = {1};
        v.constancy = {1};
        v.divisibility = {highestPow2Divisor(c)};
        v.knownConstant = c;
        propagateIfChanged(results[0], results[0]->join(v));
        return success();
      }
      setAllPessimistic(op, results);
      return success();
    }
    auto dense = dyn_cast<DenseIntElementsAttr>(op.getValue());
    if (!dense) {
      setAllPessimistic(op, results);
      return success();
    }
    auto shape = vt.getShape();
    if (dense.isSplat()) {
      int64_t c = dense.getSplatValue<APInt>().getSExtValue();
      AxisInfo v = splatAxisInfo(shape, c);
      propagateIfChanged(results[0], results[0]->join(v));
      return success();
    }

    // Compute innermost-dim contiguity / constancy / base-divisibility by
    // iterating the dense values along the inner stride. Outer dims report
    // pessimistic (1) unless they collapse trivially below.
    unsigned r = shape.size();
    int64_t inner = shape.back();
    int64_t outer = vt.getNumElements() / inner;
    if (inner < 2 || outer < 1) {
      // Can't meaningfully analyze a 0/1-element inner dim; fall back to
      // splat handling already covered, otherwise pessimistic.
      AxisInfo v = AxisInfo::getPessimistic(r);
      // For a 1-element inner dim the inner-dim contiguity/constancy is
      // trivially 1 (already pessimistic).
      propagateIfChanged(results[0], results[0]->join(v));
      return success();
    }
    auto values = llvm::to_vector(dense.getValues<APInt>());
    int64_t innerCont = inner;
    int64_t innerConst = inner;
    int64_t innerStride =
        values[1].getSExtValue() - values[0].getSExtValue();
    int64_t base = values[0].getSExtValue();
    int64_t baseDiv = highestPow2Divisor(base);
    for (int64_t o = 0; o < outer; ++o) {
      int64_t origin = values[o * inner].getSExtValue();
      baseDiv = std::gcd(baseDiv, highestPow2Divisor(origin));
      for (int64_t i = 1; i < inner; ++i) {
        int64_t cur = values[o * inner + i].getSExtValue();
        int64_t prev = values[o * inner + i - 1].getSExtValue();
        int64_t diff = cur - prev;
        if (diff != innerStride)
          innerStride = std::numeric_limits<int64_t>::min(); // not AP
        if (diff != 1)
          innerCont = std::min<int64_t>(innerCont, i);
        if (diff != 0)
          innerConst = std::min<int64_t>(innerConst, i);
      }
    }
    AxisInfo v = AxisInfo::getPessimistic(r);
    if (innerStride == 1)
      v.contiguity[r - 1] = innerCont;
    else if (innerStride == 0)
      v.constancy[r - 1] = innerConst;
    // For a non-AP inner dim, leave at pessimistic.
    v.divisibility[r - 1] = baseDiv;
    if (innerStride != std::numeric_limits<int64_t>::min())
      v.innerStride = innerStride;
    propagateIfChanged(results[0], results[0]->join(v));
    return success();
  }

  // vector.broadcast: source lattice extends to the broadcast dims with
  // constancy = full extent on those dims. The trailing dims of the source
  // (if any) align with the trailing dims of the result.
  LogicalResult visitBroadcast(vector::BroadcastOp op,
                               ArrayRef<const AxisInfoLattice *> operands,
                               ArrayRef<AxisInfoLattice *> results) {
    auto resTy = dyn_cast<VectorType>(op.getType());
    if (!resTy) {
      setAllPessimistic(op, results);
      return success();
    }
    unsigned rRank = resTy.getRank();
    AxisInfo src = operands[0]->getValue();
    AxisInfo v = AxisInfo::getPessimistic(rRank);
    auto resShape = resTy.getShape();
    auto srcVt = dyn_cast<VectorType>(op.getSource().getType());
    unsigned sRank = srcVt ? srcVt.getRank() : 0;
    // Broadcast aligns trailing dims of the source with trailing dims of
    // the result. Leading dims that are 1 in source (or absent) are filled
    // with constancy = result extent.
    for (unsigned d = 0; d < rRank; ++d) {
      int64_t resExt = resShape[d];
      // Index in source aligned with result dim d, or -1 if d is a
      // broadcast (front-padded) dim.
      int sIdx = static_cast<int>(d) - static_cast<int>(rRank - sRank);
      if (sIdx < 0) {
        v.constancy[d] = resExt;
        v.contiguity[d] = 1;
        v.divisibility[d] =
            src.isInitialized() ? src.divisibility.front() : 1;
        continue;
      }
      int64_t srcExt = srcVt.getShape()[sIdx];
      if (srcExt == 1 && resExt > 1) {
        v.constancy[d] = resExt;
        v.contiguity[d] = 1;
        v.divisibility[d] =
            src.isInitialized() ? src.divisibility[sIdx] : 1;
      } else if (src.isInitialized()) {
        v.contiguity[d] = src.contiguity[sIdx];
        v.constancy[d] = src.constancy[sIdx];
        v.divisibility[d] = src.divisibility[sIdx];
      }
    }
    if (src.knownConstant)
      v.knownConstant = src.knownConstant;
    // A broadcast that fans out a scalar / leading-1 source has the broadcast
    // dim repeating its value -> inner stride 0. Otherwise, the trailing
    // source dim's stride is preserved when its extent matches the result.
    auto resShapeArr = resTy.getShape();
    int64_t innerExt = resShapeArr.back();
    int sIdxInner = static_cast<int>(rRank - 1) - static_cast<int>(rRank - sRank);
    if (sIdxInner < 0) {
      v.innerStride = 0;
    } else if (srcVt) {
      int64_t srcInner = srcVt.getShape()[sIdxInner];
      if (srcInner == 1 && innerExt > 1)
        v.innerStride = 0;
      else if (srcInner == innerExt && src.innerStride)
        v.innerStride = src.innerStride;
    }
    propagateIfChanged(results[0], results[0]->join(v));
    return success();
  }

  // vector.shape_cast: handle two cases.
  //   (a) Identity-like: shapes match after stripping leading-1 dims —
  //       rebind per-dim info to the new dim positions.
  //   (b) General reshape with the same total element count and row-major
  //       linearization — propagate the source's innermost-dim info (inner
  //       contiguity / constancy) to the destination's innermost dim,
  //       capped by the inner extent. Outer dims stay pessimistic.
  LogicalResult visitShapeCast(vector::ShapeCastOp op,
                               ArrayRef<const AxisInfoLattice *> operands,
                               ArrayRef<AxisInfoLattice *> results) {
    auto srcTy = dyn_cast<VectorType>(op.getSource().getType());
    auto dstTy = dyn_cast<VectorType>(op.getType());
    if (!srcTy || !dstTy) {
      setAllPessimistic(op, results);
      return success();
    }
    AxisInfo src = operands[0]->getValue();
    if (!src.isInitialized()) {
      setAllPessimistic(op, results);
      return success();
    }
    // Strip leading 1-dims on both sides; if remaining shapes match, this
    // is an identity-like reshape.
    auto stripLeading = [](ArrayRef<int64_t> s) {
      unsigned i = 0;
      while (i < s.size() && s[i] == 1)
        ++i;
      return s.drop_front(i);
    };
    auto sCore = stripLeading(srcTy.getShape());
    auto dCore = stripLeading(dstTy.getShape());
    unsigned dRank = dstTy.getRank();
    AxisInfo v = AxisInfo::getPessimistic(dRank);
    if (sCore == dCore) {
      unsigned sLead = srcTy.getRank() - sCore.size();
      unsigned dLead = dRank - dCore.size();
      for (unsigned d = dLead; d < dRank; ++d) {
        unsigned sIdx = sLead + (d - dLead);
        v.contiguity[d] = src.contiguity[sIdx];
        v.constancy[d] = src.constancy[sIdx];
        v.divisibility[d] = src.divisibility[sIdx];
      }
    } else {
      // General linear reshape. Propagate source's inner-dim contiguity /
      // constancy to dst's inner dim, capped by inner extent. Treat inner
      // info conservatively as the min across all source dims (so a 1-D
      // source with full contig => inner-dim contig on dst; an N-D source
      // collapsed to 1-D inherits the inner-dim info).
      int64_t innerExt = dstTy.getShape().back();
      int64_t srcContig = std::numeric_limits<int64_t>::max();
      int64_t srcConst = std::numeric_limits<int64_t>::max();
      int64_t srcDiv = src.divisibility[src.getRank() - 1];
      for (unsigned d = 0; d < src.getRank(); ++d) {
        srcContig = std::min(srcContig, src.contiguity[d]);
        srcConst = std::min(srcConst, src.constancy[d]);
      }
      v.contiguity[dRank - 1] = std::min<int64_t>(srcContig, innerExt);
      v.constancy[dRank - 1] = std::min<int64_t>(srcConst, innerExt);
      v.divisibility[dRank - 1] = srcDiv;
    }
    if (src.knownConstant)
      v.knownConstant = src.knownConstant;
    // Identity-like and general row-major reshape both preserve the source
    // inner-stride property when the source has a single AP characterization.
    if (src.innerStride)
      v.innerStride = src.innerStride;
    propagateIfChanged(results[0], results[0]->join(v));
    return success();
  }

  // vector.transpose: permute per-dim contiguity / constancy / divisibility
  // according to the transpose permutation. permutation[i] is the source
  // dim that ends up at result dim i.
  LogicalResult visitTranspose(vector::TransposeOp op,
                               ArrayRef<const AxisInfoLattice *> operands,
                               ArrayRef<AxisInfoLattice *> results) {
    auto resTy = dyn_cast<VectorType>(op.getType());
    if (!resTy) {
      setAllPessimistic(op, results);
      return success();
    }
    AxisInfo src = operands[0]->getValue();
    if (!src.isInitialized()) {
      setAllPessimistic(op, results);
      return success();
    }
    ArrayRef<int64_t> perm = op.getPermutation();
    unsigned r = resTy.getRank();
    AxisInfo v = AxisInfo::getPessimistic(r);
    for (unsigned d = 0; d < r; ++d) {
      unsigned s = static_cast<unsigned>(perm[d]);
      v.contiguity[d] = src.contiguity[s];
      v.constancy[d] = src.constancy[s];
      v.divisibility[d] = src.divisibility[s];
    }
    if (src.knownConstant)
      v.knownConstant = src.knownConstant;
    // innerStride only survives when the new inner dim came from the old
    // inner dim (otherwise a different axis is now the contiguous one).
    if (src.innerStride && perm.back() == src.getRank() - 1)
      v.innerStride = src.innerStride;
    propagateIfChanged(results[0], results[0]->join(v));
    return success();
  }

  template <bool IsSub, typename OpTy>
  LogicalResult visitAddSub(OpTy op,
                            ArrayRef<const AxisInfoLattice *> operands,
                            ArrayRef<AxisInfoLattice *> results) {
    auto vt = dyn_cast<VectorType>(op.getType());
    if (!vt) {
      setAllPessimistic(op, results);
      return success();
    }
    AxisInfo lhs = operands[0]->getValue();
    AxisInfo rhs = operands[1]->getValue();
    if (!lhs.isInitialized() || !rhs.isInitialized()) {
      setAllPessimistic(op, results);
      return success();
    }
    unsigned r = vt.getRank();
    AxisInfo v = AxisInfo::getPessimistic(r);
    for (unsigned d = 0; d < r; ++d) {
      int64_t lhsCont = lhs.contiguity[d];
      int64_t rhsCont = rhs.contiguity[d];
      int64_t lhsConst = lhs.constancy[d];
      int64_t rhsConst = rhs.constancy[d];
      // contiguity propagates through add when one side is constant on the
      // run, and through sub only when the rhs is constant on the run.
      int64_t cont = IsSub ? std::min(lhsCont, rhsConst)
                            : std::max(std::min(lhsCont, rhsConst),
                                       std::min(rhsCont, lhsConst));
      v.contiguity[d] = std::max<int64_t>(1, cont);
      v.constancy[d] = std::min(lhsConst, rhsConst);
      v.divisibility[d] = std::gcd(lhs.divisibility[d], rhs.divisibility[d]);
    }
    // x + uniform-c: stride preserved. x - uniform-c: same. uniform-c - x:
    // stride flips sign (only useful for the "stride 0" case, which it
    // preserves trivially).
    auto isUniform = [&](const AxisInfo &a) {
      unsigned inner = vt.getRank() - 1;
      return a.constancy[inner] >= vt.getShape()[inner];
    };
    if (lhs.innerStride && isUniform(rhs)) {
      v.innerStride = *lhs.innerStride;
    } else if (rhs.innerStride && isUniform(lhs)) {
      v.innerStride = IsSub ? -*rhs.innerStride : *rhs.innerStride;
    }
    propagateIfChanged(results[0], results[0]->join(v));
    return success();
  }

  LogicalResult visitMul(arith::MulIOp op,
                         ArrayRef<const AxisInfoLattice *> operands,
                         ArrayRef<AxisInfoLattice *> results) {
    auto vt = dyn_cast<VectorType>(op.getType());
    if (!vt) {
      setAllPessimistic(op, results);
      return success();
    }
    AxisInfo lhs = operands[0]->getValue();
    AxisInfo rhs = operands[1]->getValue();
    if (!lhs.isInitialized() || !rhs.isInitialized()) {
      setAllPessimistic(op, results);
      return success();
    }
    unsigned r = vt.getRank();
    auto shape = vt.getShape();
    AxisInfo v = AxisInfo::getPessimistic(r);
    auto unitConstant = [](const AxisInfo &a, unsigned d, int64_t lanes) {
      return a.knownConstant && *a.knownConstant == 1 &&
             a.constancy[d] >= lanes;
    };
    for (unsigned d = 0; d < r; ++d) {
      v.constancy[d] = std::min({shape[d], lhs.constancy[d], rhs.constancy[d]});
      v.divisibility[d] = std::min<int64_t>(
          kAxisInfoTop, lhs.divisibility[d] * rhs.divisibility[d]);
      // Multiplying by uniform `s` only keeps contiguity when `s == 1`.
      if (unitConstant(lhs, d, shape[d]))
        v.contiguity[d] = std::min(rhs.contiguity[d], shape[d]);
      else if (unitConstant(rhs, d, shape[d]))
        v.contiguity[d] = std::min(lhs.contiguity[d], shape[d]);
      else
        v.contiguity[d] = 1;
    }
    // x * uniform-c: stride scales by c. (Both operands uniform => 0.)
    unsigned inner = vt.getRank() - 1;
    auto isUniformInner = [&](const AxisInfo &a) {
      return a.constancy[inner] >= shape[inner];
    };
    if (lhs.innerStride && isUniformInner(rhs) && rhs.knownConstant) {
      v.innerStride = *lhs.innerStride * *rhs.knownConstant;
    } else if (rhs.innerStride && isUniformInner(lhs) && lhs.knownConstant) {
      v.innerStride = *rhs.innerStride * *lhs.knownConstant;
    }
    propagateIfChanged(results[0], results[0]->join(v));
    return success();
  }

  // arith.divui / arith.divsi / arith.remui / arith.remsi by a uniform
  // positive constant `c`.
  //
  // Division (IsRem=false): when the lhs is an inner-dim AP `(base, s, n)`
  // with `c | s` and `c | divisibility[inner]` (so the per-row base / c is
  // exact), the result is an AP with stride `s / c` and inner divisibility
  // `divisibility[inner] / c`. Special cases: `s/c == 1` flags inner-dim
  // contiguity; `s/c == 0` flags inner-dim constancy.
  //
  // Remainder (IsRem=true): when `c | s`, every element of a row sits at
  // the same residue class -> inner-dim constant -> `innerStride = 0`,
  // `constancy[inner] = inner` (matches the analysis's notion of
  // chunk-uniform values along the inner dim).
  //
  // Signed vs unsigned only differs in the constant interpretation; we
  // require positive constants so the signed/unsigned distinction is moot
  // here.
  template <bool IsSigned, bool IsRem, typename OpTy>
  LogicalResult visitDivRem(OpTy op,
                            ArrayRef<const AxisInfoLattice *> operands,
                            ArrayRef<AxisInfoLattice *> results) {
    auto vt = dyn_cast<VectorType>(op.getType());
    if (!vt) {
      setAllPessimistic(op, results);
      return success();
    }
    AxisInfo lhs = operands[0]->getValue();
    AxisInfo rhs = operands[1]->getValue();
    if (!lhs.isInitialized() || !rhs.isInitialized()) {
      setAllPessimistic(op, results);
      return success();
    }
    unsigned r = vt.getRank();
    unsigned inner = r - 1;
    auto shape = vt.getShape();
    AxisInfo v = AxisInfo::getPessimistic(r);

    bool rhsUniform = rhs.constancy[inner] >= shape[inner] && rhs.knownConstant;
    if (!rhsUniform || *rhs.knownConstant <= 0) {
      propagateIfChanged(results[0], results[0]->join(v));
      return success();
    }
    int64_t c = *rhs.knownConstant;

    if (!lhs.innerStride) {
      propagateIfChanged(results[0], results[0]->join(v));
      return success();
    }
    int64_t s = *lhs.innerStride;
    int64_t baseDivLhs = lhs.divisibility[inner];
    if (s % c != 0) {
      propagateIfChanged(results[0], results[0]->join(v));
      return success();
    }

    if (IsRem) {
      // (base + i*s) mod c, with c | s, is the constant base mod c.
      v.innerStride = 0;
      v.constancy[inner] = shape[inner];
      // The remainder is in [0, c-1], so any power-of-two divisor of c is a
      // lower bound on alignment. Use lhs's existing divisibility too.
      v.divisibility[inner] =
          std::gcd(baseDivLhs, highestPow2Divisor(c));
    } else {
      if (baseDivLhs % c != 0) {
        propagateIfChanged(results[0], results[0]->join(v));
        return success();
      }
      int64_t newStride = s / c;
      v.innerStride = newStride;
      if (newStride == 1)
        v.contiguity[inner] = shape[inner];
      else if (newStride == 0)
        v.constancy[inner] = shape[inner];
      v.divisibility[inner] = baseDivLhs / c;
    }
    propagateIfChanged(results[0], results[0]->join(v));
    return success();
  }

  // arith.andi: `x & m` with a uniform positive constant mask `m`.
  // The most useful case is `x & (P - 1)` for `P` a power of 2: this is
  // equivalent to `x mod P`, so when the lhs is an inner-dim AP with stride
  // divisible by `P` the result is inner-dim constant. We also handle the
  // trivial `m == 0` (always zero) and `m == -1`/all-ones (identity)
  // shapes.
  LogicalResult visitAndI(arith::AndIOp op,
                          ArrayRef<const AxisInfoLattice *> operands,
                          ArrayRef<AxisInfoLattice *> results) {
    auto vt = dyn_cast<VectorType>(op.getType());
    if (!vt) {
      setAllPessimistic(op, results);
      return success();
    }
    AxisInfo lhs = operands[0]->getValue();
    AxisInfo rhs = operands[1]->getValue();
    if (!lhs.isInitialized() || !rhs.isInitialized()) {
      setAllPessimistic(op, results);
      return success();
    }
    unsigned r = vt.getRank();
    unsigned inner = r - 1;
    auto shape = vt.getShape();
    AxisInfo v = AxisInfo::getPessimistic(r);

    // Look for a uniform constant mask on either side.
    auto getUniformMask = [&](const AxisInfo &a) -> std::optional<int64_t> {
      if (a.constancy[inner] >= shape[inner] && a.knownConstant)
        return a.knownConstant;
      return std::nullopt;
    };
    std::optional<int64_t> mLhs = getUniformMask(lhs);
    std::optional<int64_t> mRhs = getUniformMask(rhs);
    if (!mLhs && !mRhs) {
      propagateIfChanged(results[0], results[0]->join(v));
      return success();
    }
    const AxisInfo &x = mLhs ? rhs : lhs;
    int64_t m = mLhs ? *mLhs : *mRhs;

    if (m == 0) {
      v.knownConstant = 0;
      v.innerStride = 0;
      v.constancy[inner] = shape[inner];
      v.divisibility[inner] = kAxisInfoTop;
      propagateIfChanged(results[0], results[0]->join(v));
      return success();
    }

    // `m == P - 1` with P a power of 2 -> equivalent to `x mod P`.
    if (m > 0 && llvm::isPowerOf2_64(static_cast<uint64_t>(m + 1))) {
      int64_t P = m + 1;
      if (x.innerStride && *x.innerStride % P == 0) {
        v.innerStride = 0;
        v.constancy[inner] = shape[inner];
        v.divisibility[inner] =
            std::gcd(x.divisibility[inner], highestPow2Divisor(P));
        propagateIfChanged(results[0], results[0]->join(v));
        return success();
      }
    }
    // Conservative fallback for unrecognized masks.
    propagateIfChanged(results[0], results[0]->join(v));
    return success();
  }

  // arith.shli (left shift) / arith.shrui (logical right shift) by a
  // uniform constant `k`. These are `* (1 << k)` and `/ (1 << k)`
  // (truncating, but for non-negative values the trunc is exact when
  // `(1 << k)` divides the value). We model them by reducing to mul/divui.
  template <bool IsLeft, typename OpTy>
  LogicalResult visitShift(OpTy op,
                           ArrayRef<const AxisInfoLattice *> operands,
                           ArrayRef<AxisInfoLattice *> results) {
    auto vt = dyn_cast<VectorType>(op.getType());
    if (!vt) {
      setAllPessimistic(op, results);
      return success();
    }
    AxisInfo lhs = operands[0]->getValue();
    AxisInfo rhs = operands[1]->getValue();
    if (!lhs.isInitialized() || !rhs.isInitialized()) {
      setAllPessimistic(op, results);
      return success();
    }
    unsigned r = vt.getRank();
    unsigned inner = r - 1;
    auto shape = vt.getShape();
    AxisInfo v = AxisInfo::getPessimistic(r);

    if (rhs.constancy[inner] < shape[inner] || !rhs.knownConstant) {
      propagateIfChanged(results[0], results[0]->join(v));
      return success();
    }
    int64_t k = *rhs.knownConstant;
    if (k < 0 || k >= 63) {
      propagateIfChanged(results[0], results[0]->join(v));
      return success();
    }
    int64_t factor = 1LL << k;

    if (IsLeft) {
      // x << k == x * factor.
      if (lhs.innerStride) {
        v.innerStride = *lhs.innerStride * factor;
        if (*v.innerStride == 1)
          v.contiguity[inner] = shape[inner];
        else if (*v.innerStride == 0)
          v.constancy[inner] = shape[inner];
      }
      v.divisibility[inner] = std::min<int64_t>(
          kAxisInfoTop, lhs.divisibility[inner] * factor);
    } else {
      // x >> k == x / factor (for non-negative x); same conditions as divui.
      if (lhs.innerStride && *lhs.innerStride % factor == 0 &&
          lhs.divisibility[inner] % factor == 0) {
        int64_t newStride = *lhs.innerStride / factor;
        v.innerStride = newStride;
        if (newStride == 1)
          v.contiguity[inner] = shape[inner];
        else if (newStride == 0)
          v.constancy[inner] = shape[inner];
        v.divisibility[inner] = lhs.divisibility[inner] / factor;
      }
    }
    propagateIfChanged(results[0], results[0]->join(v));
    return success();
  }

  // arith.select: result is at least as constrained as the meet of the two
  // arms. We propagate fields where both arms agree.
  LogicalResult visitSelect(arith::SelectOp op,
                            ArrayRef<const AxisInfoLattice *> operands,
                            ArrayRef<AxisInfoLattice *> results) {
    auto vt = dyn_cast<VectorType>(op.getType());
    if (!vt) {
      setAllPessimistic(op, results);
      return success();
    }
    // operands: [cond, true, false]
    AxisInfo t = operands[1]->getValue();
    AxisInfo f = operands[2]->getValue();
    if (!t.isInitialized() || !f.isInitialized()) {
      setAllPessimistic(op, results);
      return success();
    }
    AxisInfo v = AxisInfo::join(t, f);
    propagateIfChanged(results[0], results[0]->join(v));
    return success();
  }

  template <typename OpTy>
  LogicalResult visitPassThrough(OpTy op,
                                 ArrayRef<const AxisInfoLattice *> operands,
                                 ArrayRef<AxisInfoLattice *> results) {
    if (!isa<VectorType>(op.getType())) {
      setAllPessimistic(op, results);
      return success();
    }
    propagateIfChanged(results[0], results[0]->join(operands[0]->getValue()));
    return success();
  }
};

} // namespace mlir::xegpu::detail::axis_dataflow

namespace {

using ::mlir::xegpu::detail::axis_dataflow::AxisInfo;
using ::mlir::xegpu::detail::axis_dataflow::AxisInfoLattice;

//===----------------------------------------------------------------------===//
// Coalescing decision.
//===----------------------------------------------------------------------===//

struct CoalesceDecision {
  enum class Kind { None, Broadcast, Chunked };
  Kind kind = Kind::None;
  int64_t laneLayout = 1; // lane_layout along the innermost dim
  int64_t factor = 1;     // lane_data factor along the innermost dim
};

/// Largest power-of-two `<= bound` that divides `numLanes`.
static int64_t largestPow2Divisor(int64_t numLanes, int64_t bound) {
  if (bound < 2 || numLanes < 2)
    return 1;
  int64_t f = std::min<int64_t>(bound, numLanes);
  // Round down to power of 2.
  if (!llvm::isPowerOf2_64(f))
    f = static_cast<int64_t>(llvm::bit_floor(static_cast<uint64_t>(f)));
  while (f >= 2) {
    if (numLanes % f == 0)
      return f;
    f /= 2;
  }
  return 1;
}

/// Decide how to coalesce given the offsets axis info.
///
/// We pick `lane_layout[inner]` first using the same default rule as
/// `XeGPUPropagateLayout`: `lane_layout[inner] = min(subgroupSize, inner)`,
/// rounded down to a power-of-2 divisor of `inner`. The remaining lane
/// budget then becomes `lane_data[inner] = inner / lane_layout`, capped by
/// the per-lane chunk-size budget and the offsets contiguity.
static CoalesceDecision decide(const AxisInfo &info,
                               ArrayRef<int64_t> offsetsShape,
                               int64_t origChunk, unsigned maxChunkSize,
                               unsigned subgroupSize) {
  CoalesceDecision d;
  if (!info.isInitialized() || offsetsShape.empty())
    return d;
  unsigned innerDim = offsetsShape.size() - 1;
  int64_t inner = offsetsShape[innerDim];
  if (inner < 2)
    return d;

  // Broadcast if the innermost dim is uniform across all lanes.
  if (info.constancy[innerDim] >= inner) {
    d.kind = CoalesceDecision::Kind::Broadcast;
    return d;
  }

  // Pick lane_layout first: PropagateLayout's default for a 1-D / inner dim
  // is `min(subgroupSize, inner)`, rounded down to a divisor of inner.
  int64_t laneLayout = largestPow2Divisor(
      inner, std::min<int64_t>(subgroupSize, inner));
  if (laneLayout < 1)
    laneLayout = 1;

  // Each lane sees `inner / laneLayout` elements of the offsets vector.
  int64_t perLane = inner / laneLayout;
  if (perLane < 2)
    return d; // already one element per lane, nothing to coalesce.

  if (origChunk < 1)
    origChunk = 1;
  int64_t budget = static_cast<int64_t>(maxChunkSize) / origChunk;
  if (budget < 2)
    return d;

  int64_t bound = std::min<int64_t>(info.contiguity[innerDim], budget);
  bound = std::min<int64_t>(bound, perLane);
  if (bound < 2)
    return d;

  int64_t factor = largestPow2Divisor(perLane, bound);
  if (factor < 2)
    return d;
  d.kind = CoalesceDecision::Kind::Chunked;
  d.laneLayout = laneLayout;
  d.factor = factor;
  return d;
}

/// Returns true if `mask` is a constant `dense<true>` vector.
static bool isAllTrueMask(Value mask) {
  auto vecTy = dyn_cast<VectorType>(mask.getType());
  if (!vecTy)
    return false;
  auto cst = mask.getDefiningOp<arith::ConstantOp>();
  if (!cst)
    return false;
  auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat())
    return false;
  return dense.getSplatValue<APInt>().getBoolValue();
}

/// Build a `lane_layout`/`lane_data`/`inst_data` layout of rank `rank`,
/// with the given lane_layout / lane_data on the innermost dim (1
/// elsewhere). `inst_data` is `lane_layout * lane_data` per dim, so the
/// invariant `inst_data[d] == lane_layout[d] * lane_data[d]` holds.
static xegpu::LayoutAttr buildLaneDataLayout(MLIRContext *ctx, unsigned rank,
                                             int64_t innerLaneLayout,
                                             int64_t innerLaneData) {
  SmallVector<int32_t> laneLayout(rank, 1);
  SmallVector<int32_t> laneData(rank, 1);
  SmallVector<int32_t> instData(rank, 1);
  laneLayout.back() = static_cast<int32_t>(innerLaneLayout);
  laneData.back() = static_cast<int32_t>(innerLaneData);
  instData.back() = static_cast<int32_t>(innerLaneLayout * innerLaneData);
  return xegpu::LayoutAttr::get(ctx, instData, laneLayout, laneData);
}


//===----------------------------------------------------------------------===//
// Rewrites.
//===----------------------------------------------------------------------===//

/// Replace an `xegpu.load` whose offsets are uniform along the innermost
/// dim with a length-1 load + `vector.broadcast` back to the original
/// value type. Works for any rank; the length-1 load uses an inner-dim
/// length-1 offsets/mask vector.
static LogicalResult rewriteBroadcastLoad(xegpu::LoadGatherOp op,
                                          PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  auto valueTy = op.getValueType();
  auto offsetsTy = dyn_cast<VectorType>(op.getOffsets().getType());
  if (!valueTy || !offsetsTy)
    return failure();

  // Extract a scalar offset from index 0...0.
  SmallVector<int64_t> zeros(offsetsTy.getRank(), 0);
  Value scalarOffset =
      vector::ExtractOp::create(rewriter, loc, op.getOffsets(), zeros);
  auto idxVecTy = VectorType::get({1}, rewriter.getIndexType());
  auto maskVecTy = VectorType::get({1}, rewriter.getI1Type());
  Value newOffsets =
      vector::BroadcastOp::create(rewriter, loc, idxVecTy, scalarOffset);
  Value newMask = arith::ConstantOp::create(
      rewriter, loc, DenseIntElementsAttr::get(maskVecTy, true));
  auto newValueTy = VectorType::get({1}, valueTy.getElementType());
  auto newLoad = xegpu::LoadGatherOp::create(
      rewriter, loc, newValueTy, op.getSource(), newOffsets, newMask,
      /*chunk_size=*/IntegerAttr(), op.getL1HintAttr(), op.getL2HintAttr(),
      op.getL3HintAttr(), /*layout=*/xegpu::DistributeLayoutAttr());
  Value scalar = vector::ExtractOp::create(rewriter, loc, newLoad.getResult(),
                                           ArrayRef<int64_t>{0});
  Value bcast = vector::BroadcastOp::create(rewriter, loc, valueTy, scalar);
  rewriter.replaceOp(op, bcast);
  return success();
}

namespace {

/// Look up the subgroup size from the enclosing gpu.module's xevm.target.
/// Falls back to 16 when no target chip is found or the chip is unknown,
/// matching the typical Intel Xe2 default. This keeps the pass usable on
/// plain `module { ... }` IR (e.g. unit lit tests) where there's no
/// gpu.module / xevm.target wrapper.
static unsigned lookupSubgroupSize(Operation *op) {
  const auto *uArch =
      xegpu::uArch::getUArch(xegpu::getChipStr(op).value_or(""));
  return uArch ? static_cast<unsigned>(uArch->getSubgroupSize()) : 16u;
}

struct CoalesceLoadPattern final : OpRewritePattern<xegpu::LoadGatherOp> {
  CoalesceLoadPattern(MLIRContext *ctx, unsigned maxChunkSize,
                      DataFlowSolver &solver)
      : OpRewritePattern(ctx), maxChunkSize(maxChunkSize), solver(solver) {}

  LogicalResult matchAndRewrite(xegpu::LoadGatherOp op,
                                PatternRewriter &rewriter) const override {
    auto offsetsTy = dyn_cast<VectorType>(op.getOffsets().getType());
    if (!offsetsTy)
      return rewriter.notifyMatchFailure(op, "expected vector offsets");
    if (offsetsTy.getNumElements() <= 1)
      return rewriter.notifyMatchFailure(op, "nothing to coalesce");
    auto valueTy = op.getValueType();
    if (!valueTy)
      return rewriter.notifyMatchFailure(op, "expected vector value");
    if (!isAllTrueMask(op.getMask()))
      return rewriter.notifyMatchFailure(op, "non-uniform mask");

    // Already coalesced (a previous run, or another pass tagged it).
    if (auto layout = op.getLayoutAttr())
      if (!layout.getEffectiveLaneDataAsInt().empty())
        return rewriter.notifyMatchFailure(op, "lane_data already set");

    int64_t origChunk = static_cast<int64_t>(op.getChunkSize().value_or(1));
    if (op.getChunkSizeAttr() && origChunk > 1)
      return rewriter.notifyMatchFailure(
          op, "explicit chunk_size > 1, leaving op alone");

    unsigned subgroupSize = lookupSubgroupSize(op);

    const auto *lat = solver.lookupState<AxisInfoLattice>(op.getOffsets());
    if (!lat || !lat->getValue().isInitialized())
      return rewriter.notifyMatchFailure(op, "no axis-info available");

    auto d = decide(lat->getValue(), offsetsTy.getShape(), origChunk,
                    maxChunkSize, subgroupSize);

    if (d.kind == CoalesceDecision::Kind::Broadcast)
      return rewriteBroadcastLoad(op, rewriter);

    if (d.kind != CoalesceDecision::Kind::Chunked)
      return rewriter.notifyMatchFailure(op, "offsets not coalescible");

    auto layout = buildLaneDataLayout(op.getContext(), valueTy.getRank(),
                                      d.laneLayout, d.factor);
    // If the op carried an explicit chunk_size = 1 (the trivial / default
    // value), drop it: the new lane_data FCD > 1 supersedes it.
    bool dropChunk = op.getChunkSizeAttr() && origChunk == 1 && d.factor > 1;
    rewriter.modifyOpInPlace(op, [&] {
      op.setLayoutAttr(layout);
      if (dropChunk)
        op.removeChunkSizeAttr();
    });
    return success();
  }

  unsigned maxChunkSize;
  DataFlowSolver &solver;
};

struct CoalesceStorePattern final : OpRewritePattern<xegpu::StoreScatterOp> {
  CoalesceStorePattern(MLIRContext *ctx, unsigned maxChunkSize,
                       DataFlowSolver &solver)
      : OpRewritePattern(ctx), maxChunkSize(maxChunkSize), solver(solver) {}

  LogicalResult matchAndRewrite(xegpu::StoreScatterOp op,
                                PatternRewriter &rewriter) const override {
    auto offsetsTy = dyn_cast<VectorType>(op.getOffsets().getType());
    if (!offsetsTy)
      return rewriter.notifyMatchFailure(op, "expected vector offsets");
    if (offsetsTy.getNumElements() <= 1)
      return rewriter.notifyMatchFailure(op, "nothing to coalesce");
    auto valueTy = op.getValueType();
    if (!valueTy)
      return rewriter.notifyMatchFailure(op, "expected vector value");
    if (!isAllTrueMask(op.getMask()))
      return rewriter.notifyMatchFailure(op, "non-uniform mask");

    if (auto layout = op.getLayoutAttr())
      if (!layout.getEffectiveLaneDataAsInt().empty())
        return rewriter.notifyMatchFailure(op, "lane_data already set");

    int64_t origChunk = static_cast<int64_t>(op.getChunkSize().value_or(1));
    if (op.getChunkSizeAttr() && origChunk > 1)
      return rewriter.notifyMatchFailure(
          op, "explicit chunk_size > 1, leaving op alone");

    unsigned subgroupSize = lookupSubgroupSize(op);

    const auto *lat = solver.lookupState<AxisInfoLattice>(op.getOffsets());
    if (!lat || !lat->getValue().isInitialized())
      return rewriter.notifyMatchFailure(op, "no axis-info available");

    auto d = decide(lat->getValue(), offsetsTy.getShape(), origChunk,
                    maxChunkSize, subgroupSize);

    if (d.kind == CoalesceDecision::Kind::Broadcast)
      return rewriter.notifyMatchFailure(
          op, "all-equal offsets on store would be ambiguous");

    if (d.kind != CoalesceDecision::Kind::Chunked)
      return rewriter.notifyMatchFailure(op, "offsets not coalescible");

    auto layout = buildLaneDataLayout(op.getContext(), valueTy.getRank(),
                                      d.laneLayout, d.factor);
    bool dropChunk = op.getChunkSizeAttr() && origChunk == 1 && d.factor > 1;
    rewriter.modifyOpInPlace(op, [&] {
      op.setLayoutAttr(layout);
      if (dropChunk)
        op.removeChunkSizeAttr();
    });
    return success();
  }

  unsigned maxChunkSize;
  DataFlowSolver &solver;
};

struct XeGPUCoalesceGatherScatterPass final
    : public xegpu::impl::XeGPUCoalesceGatherScatterBase<
          XeGPUCoalesceGatherScatterPass> {
  using XeGPUCoalesceGatherScatterBase::XeGPUCoalesceGatherScatterBase;

  void runOnOperation() override {
    Operation *root = getOperation();

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<mlir::xegpu::detail::axis_dataflow::AxisInfoAnalysis>();
    if (failed(solver.initializeAndRun(root)))
      return signalPassFailure();

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<CoalesceLoadPattern, CoalesceStorePattern>(ctx, maxChunkSize,
                                                            solver);
    if (failed(applyPatternsGreedily(root, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

} // namespace
