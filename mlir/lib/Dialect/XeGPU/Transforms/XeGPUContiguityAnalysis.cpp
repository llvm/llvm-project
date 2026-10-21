//===- XeGPUContiguityAnalysis.cpp - Offset contiguity analysis ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the contiguity analysis. It computes, for a memory
// operation (i.e., `xegpu.load` / `xegpu.store`), how many elements are
// contiguous along the innermost offsets dimension, and stamps that count as an
// attribute on the op. The analysis is a small XeGPU-local
// axis-info dataflow tracking per-axis `contiguity`, `constancy`, and
// `divisibility`; the stamped value is the inner-dim `contiguity`.
//
// Contiguity is a target-independent property of the offsets.
//
// The analysis tracks per-axis information for vectors of integer / index
// type at any rank, against the innermost dimension.
//
// The analysis gets it's inspiration from the Triton Axis info analysis.
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>
#include <optional>

#define DEBUG_TYPE "xegpu-contiguity-analysis"

using namespace mlir;

// AxisInfo and AxisInfoAnalysis are intentionally placed in a named namespace
// (not anonymous) so the `dataflow::Lattice<AxisInfo>` template instantiation
// gets a stable, externally-visible name. The TypeID machinery requires that
// for `MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID` to apply.
namespace mlir::xegpu::detail::axis_dataflow {

//===----------------------------------------------------------------------===//
// AxisInfo: per-axis contiguity / constancy / divisibility lattice.
//===----------------------------------------------------------------------===//

/// Sentinel "very large" value for unconstrained dimensions. Any real shape is
/// far smaller, so component-wise `min` collapses this to the truth. It is not
/// `numeric_limits<int64_t>::max()` because divisibility values are multiplied
/// (e.g. in `visitMul`); `1 << 30` keeps those products well within `int64_t`.
static constexpr int64_t kAxisInfoTop = 1LL << 30;

/// Per-dimension axis information for an SSA vector value of integer / index
/// type. The fields describe, for each dimension `d`, the pattern of the values
/// along `d` (examples use a 1-D vector, so `d` is the only / innermost dim):
///   - `contiguity[d]`: longest run that increases by exactly 1.
///       `[0, 1, 2, 3]` -> 4;  `[0, 1, 0, 1]` -> 2.
///   - `constancy[d]`: longest run of equal values.
///       `[5, 5, 5, 5]` -> 4;  `[5, 5, 6, 6]` -> 2.
///   - `divisibility[d]`: a power-of-two divisor of every element.
///       `[8, 16, 24]` -> 8;  `[3, 6, 9]` -> 1.
///   - `knownConstant`: the value, if the whole vector is one constant.
///       `dense<7>` -> 7;  `[0, 1, 2]` -> nullopt.
///   - `innerStride`: if set, consecutive values along the innermost dim differ
///     by this constant step. `1` is the contiguous case (`[0,1,2,3]`), `0` the
///     all-equal case (`[5,5,5,5]`); any other value is a strided progression
///     (`[0,4,8,12]` -> 4) that `contiguity`/`constancy` can't represent (both
///     read 1). For a multi-dim vector each inner-dim slice is its own
///     progression; their bases may differ, with the shared inner alignment in
///     `divisibility[innerDim]`.
///
/// Only `contiguity[innerDim]` is consumed when stamping, but all dimensions
/// are tracked because `vector.transpose` / `vector.shape_cast` permute or move
/// per-dim info between axes, so an intermediate value's outer dims can become
/// the inner dim of a later value.
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

  /// Conservative join (lattice meet). When a value can arrive from several
  /// paths (e.g. a block argument, or `arith.select`), only what holds on
  /// *every* path is safe to assume. So we keep the weaker fact per field:
  /// `min` of each run length (a run is only guaranteed as long as the shortest
  /// incoming one), `gcd` of divisibility, and a value/stride only when both
  /// sides agree. This is what makes the contiguity we later stamp sound rather
  /// than "undecidable" — it is the largest run guaranteed on all paths.
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

  // arith.constant. The four cases below, by example:
  //   - scalar int      `arith.constant 8 : index`
  //   - non-int scalar  `arith.constant 1.0 : f32` (pessimistic)
  //   - splat vector    `arith.constant dense<5> : vector<16xindex>`
  //   - dense vector    `arith.constant dense<[0,1,2,3]> : vector<4xindex>`
  LogicalResult visitConstant(arith::ConstantOp op,
                              ArrayRef<AxisInfoLattice *> results) {
    auto vt = dyn_cast<VectorType>(op.getType());
    if (!vt) {
      // Scalar integer, e.g. `arith.constant 8 : index`: a single known value,
      // contiguity/constancy 1, divisibility from the value (8 -> 8).
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
      // Non-integer scalar, e.g. `arith.constant 1.0 : f32`: nothing to track.
      setAllPessimistic(op, results);
      return success();
    }
    auto dense = dyn_cast<DenseIntElementsAttr>(op.getValue());
    if (!dense) {
      setAllPessimistic(op, results);
      return success();
    }
    auto shape = vt.getShape();
    // Splat, e.g. `arith.constant dense<5> : vector<16xindex>`: every element
    // equal, so constancy = full extent, innerStride 0.
    if (dense.isSplat()) {
      int64_t c = dense.getSplatValue<APInt>().getSExtValue();
      AxisInfo v = splatAxisInfo(shape, c);
      propagateIfChanged(results[0], results[0]->join(v));
      return success();
    }

    // General dense vector, e.g. `arith.constant dense<[0,1,2,3]> :
    // vector<4xindex>`. Compute innermost-dim contiguity / constancy /
    // base-divisibility by iterating the dense values along the inner stride
    // (here stride 1 -> contiguity 4). Outer dims report pessimistic (1)
    // unless they collapse trivially below.
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
    int64_t innerStride = values[1].getSExtValue() - values[0].getSExtValue();
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
        v.divisibility[d] = src.isInitialized() ? src.divisibility.front() : 1;
        continue;
      }
      int64_t srcExt = srcVt.getShape()[sIdx];
      if (srcExt == 1 && resExt > 1) {
        v.constancy[d] = resExt;
        v.contiguity[d] = 1;
        v.divisibility[d] = src.isInitialized() ? src.divisibility[sIdx] : 1;
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
    int sIdxInner =
        static_cast<int>(rRank - 1) - static_cast<int>(rRank - sRank);
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
  LogicalResult visitAddSub(OpTy op, ArrayRef<const AxisInfoLattice *> operands,
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
    auto unitConstant = [](const AxisInfo &a, unsigned d, int64_t extent) {
      return a.knownConstant && *a.knownConstant == 1 &&
             a.constancy[d] >= extent;
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
  // positive constant `c`. The lhs must be an arithmetic progression (AP)
  // along the inner dim, i.e. its values step by a constant stride `s`, and
  // `c` must divide `s`.
  //
  // Take the inner row `[0, 2, 4, 6, 8, 10, 12, 14]` (stride s = 2) and c = 2:
  //   - Division `/ 2` gives `[0, 1, 2, 3, 4, 5, 6, 7]`: a new AP with stride
  //     `s / c = 1`. A resulting stride of 1 is contiguous, 0 is constant.
  //   - Remainder `% 2` gives `[0, 0, 0, 0, 0, 0, 0, 0]`: every element folds
  //     to the same residue, so the row is constant (stride 0).
  //
  // We require positive `c`, so signed and unsigned behave the same.
  template <bool IsSigned, bool IsRem, typename OpTy>
  LogicalResult visitDivRem(OpTy op, ArrayRef<const AxisInfoLattice *> operands,
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
      v.divisibility[inner] = std::gcd(baseDivLhs, highestPow2Divisor(c));
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

  // arith.andi: `x & m` with a uniform positive constant mask `m`. The
  // interesting case is `m = P - 1` for a power of 2 `P`, which is the same
  // as `x % P` (see visitDivRem): masking an inner row whose stride is a
  // multiple of `P` folds it to a constant.
  //
  // Take the row `[0, 2, 4, 6, 8, 10, 12, 14]` (stride 2) and m = 1 (P = 2):
  //   `x & 1` gives `[0, 0, 0, 0, 0, 0, 0, 0]`: constant along the inner dim.
  //
  // Also handles the trivial masks `m == 0` (always zero) and all-ones
  // (identity).
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
  LogicalResult visitShift(OpTy op, ArrayRef<const AxisInfoLattice *> operands,
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
      v.divisibility[inner] =
          std::min<int64_t>(kAxisInfoTop, lhs.divisibility[inner] * factor);
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
// Analysis driver.
//===----------------------------------------------------------------------===//

/// Stamp a `contiguity` attribute on `op` recording the inner-dim contiguity
/// computed by the analysis. The contiguity is a target-independent property
/// of the offsets.
template <typename OpTy>
static void analyzeAndStampContiguity(OpTy op, DataFlowSolver &solver) {
  auto offsetsTy = dyn_cast<VectorType>(op.getOffsets().getType());
  if (!offsetsTy || offsetsTy.getNumElements() <= 1)
    return;
  // A pre-existing `contiguity` (user-authored, or stamped by an earlier run)
  // takes precedence; leave it untouched so the analysis is idempotent.
  if (op.getContiguity())
    return;
  const auto *lat = solver.lookupState<AxisInfoLattice>(op.getOffsets());
  if (!lat || !lat->getValue().isInitialized())
    return;
  const AxisInfo &info = lat->getValue();
  unsigned innerDim = offsetsTy.getRank() - 1;
  int64_t inner = offsetsTy.getShape()[innerDim];
  // The attribute records a contiguity that tiles the inner dim, so it must
  // divide it (verified on the op). Round the measured run length down to the
  // largest divisor of `inner` that does not exceed it.
  int64_t contiguity = std::min<int64_t>(info.contiguity[innerDim], inner);
  while (contiguity >= 2 && inner % contiguity != 0)
    --contiguity;
  if (contiguity < 2)
    return;
  op.setContiguity(contiguity);
}

} // namespace

//===----------------------------------------------------------------------===//
// Public API.
//===----------------------------------------------------------------------===//

void mlir::xegpu::runContiguityAnalysis(Operation *root) {
  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<mlir::xegpu::detail::axis_dataflow::AxisInfoAnalysis>();
  if (failed(solver.initializeAndRun(root)))
    return;

  // The solver computed AxisInfo for the whole region in the single
  // `initializeAndRun` above; offsets shared by several gather/scatter ops are
  // analyzed only once. This walk is just per-op point lookups into that
  // result (no re-analysis), turning each cached fact into an attribute.
  root->walk([&](Operation *op) {
    if (auto load = dyn_cast<xegpu::LoadGatherOp>(op))
      analyzeAndStampContiguity(load, solver);
    else if (auto store = dyn_cast<xegpu::StoreScatterOp>(op))
      analyzeAndStampContiguity(store, solver);
  });
}
