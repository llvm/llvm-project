//===- SCFVectorize.cpp - SCF vectorization utilities ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/SCFVectorize.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h" // getCombinerOpKind
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

static bool isSupportedVecElem(Type type) { return type.isIntOrIndexOrFloat(); }

/// Return type bitwidth for vectorization purposes or empty if type cannot be
/// vectorized.
static std::optional<unsigned> getTypeBitWidth(Type type,
                                               const DataLayout *DL) {
  if (!isSupportedVecElem(type))
    return std::nullopt;

  if (DL)
    return DL->getTypeSizeInBits(type);

  if (type.isIntOrFloat())
    return type.getIntOrFloatBitWidth();

  return std::nullopt;
}

static std::optional<unsigned> getArgsTypeWidth(Operation &op,
                                                const DataLayout *DL) {
  unsigned ret = 0;
  for (auto r : {ValueRange(op.getOperands()), ValueRange(op.getResults())}) {
    for (Value arg : op.getOperands()) {
      std::optional<unsigned> w = getTypeBitWidth(arg.getType(), DL);
      if (!w)
        return std::nullopt;

      ret = std::max(ret, *w);
    }
  }

  return ret;
}

static bool isSupportedVectorOp(Operation &op) {
  return op.hasTrait<OpTrait::Vectorizable>();
}

/// Check if one `ValueRange` is permutation of another, i.e. contains same
/// values, potentially in different order.
static bool isRangePermutation(ValueRange val1, ValueRange val2) {
  if (val1.size() != val2.size())
    return false;

  for (Value v1 : val1) {
    auto it = llvm::find(val2, v1);
    if (it == val2.end())
      return false;
  }
  return true;
}

template <typename Op>
static std::optional<unsigned>
canTriviallyVectorizeMemOpImpl(scf::ParallelOp loop, unsigned dim, Op memOp,
                               const DataLayout *DL) {
  ValueRange loopIndexVars = loop.getInductionVars();
  assert(dim < loopIndexVars.size() && "Invalid loop dimension");
  Value memref = memOp.getMemRef();
  auto type = cast<MemRefType>(memref.getType());
  std::optional<unsigned> width = getTypeBitWidth(type.getElementType(), DL);
  if (!width)
    return std::nullopt;

  if (!type.getLayout().isIdentity())
    return std::nullopt;

  if (!isRangePermutation(memOp.getIndices(), loopIndexVars))
    return std::nullopt;

  if (memOp.getIndices().back() != loopIndexVars[dim])
    return std::nullopt;

  DominanceInfo dom;
  if (!dom.properlyDominates(memref, loop))
    return std::nullopt;

  return width;
}

/// Check if memref load/store can be converted into vectorized load/store
///
/// Returns memref element bitwidth or `std::nullopt` if access cannot be
/// vectorized.
static std::optional<unsigned>
canTriviallyVectorizeMemOp(scf::ParallelOp loop, unsigned dim, Operation &op,
                           const DataLayout *DL) {
  assert(dim < loop.getInductionVars().size() && "Invalid loop dimension");
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return canTriviallyVectorizeMemOpImpl(loop, dim, storeOp, DL);

  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return canTriviallyVectorizeMemOpImpl(loop, dim, loadOp, DL);

  return std::nullopt;
}

template <typename Op>
static std::optional<unsigned> canGatherScatterImpl(scf::ParallelOp loop, Op op,
                                                    const DataLayout *DL) {
  Value memref = op.getMemRef();
  auto memrefType = cast<MemRefType>(memref.getType());
  std::optional<unsigned> width =
      getTypeBitWidth(memrefType.getElementType(), DL);
  if (!width)
    return std::nullopt;

  DominanceInfo dom;
  if (!dom.properlyDominates(memref, loop) || op.getIndices().size() != 1 ||
      !memrefType.getLayout().isIdentity())
    return std::nullopt;

  return width;
}

// Check if memref access can be converted into gather/scatter.
///
/// Returns memref element bitwidth or `std::nullopt` if access cannot be
/// vectorized.
static std::optional<unsigned>
canGatherScatter(scf::ParallelOp loop, Operation &op, const DataLayout *DL) {
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return canGatherScatterImpl(loop, storeOp, DL);

  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return canGatherScatterImpl(loop, loadOp, DL);

  return std::nullopt;
}

static std::optional<unsigned> cenVectorizeMemrefOp(scf::ParallelOp loop,
                                                    unsigned dim, Operation &op,
                                                    const DataLayout *DL) {
  if (std::optional<unsigned> w = canTriviallyVectorizeMemOp(loop, dim, op, DL))
    return w;

  return canGatherScatter(loop, op, DL);
}

/// Returns `vector.reduce` kind for specified `scf.parallel` reduce op ot
/// `std::nullopt` if reduction cannot be handled by `vector.reduce`.
static std::optional<vector::CombiningKind> getReductionKind(Block &body) {
  if (!llvm::hasSingleElement(body.without_terminator()))
    return std::nullopt;

  // TODO: Move getCombinerOpKind to vector dialect.
  return linalg::getCombinerOpKind(&body.front());
}

std::optional<scf::SCFVectorizeInfo>
mlir::scf::getLoopVectorizeInfo(scf::ParallelOp loop, unsigned dim,
                                unsigned vectorBitwidth, const DataLayout *DL) {
  assert(dim < loop.getStep().size() && "Invalid loop dimension");
  assert(vectorBitwidth > 0 && "Invalid vector bitwidth");
  unsigned factor = vectorBitwidth / 8;
  if (factor <= 1)
    return std::nullopt;

  /// Only step==1 is supported for now.
  if (!isConstantIntValue(loop.getStep()[dim], 1))
    return std::nullopt;

  unsigned count = 0;
  bool masked = true;

  /// Check if `scf.reduce` can be handled by `vector.reduce`.
  /// If not we still can vectorize the loop but we cannot use masked
  /// vectorize.
  auto reduce = cast<scf::ReduceOp>(loop.getBody()->getTerminator());
  for (Region &reg : reduce.getReductions()) {
    if (!getReductionKind(reg.front()))
      masked = false;

    continue;
  }

  for (Operation &op : loop.getBody()->without_terminator()) {
    /// Ops with nested regions are not supported yet.
    if (op.getNumRegions() > 0)
      return std::nullopt;

    /// Check mem ops.
    if (std::optional<unsigned> w = cenVectorizeMemrefOp(loop, dim, op, DL)) {
      unsigned newFactor = vectorBitwidth / *w;
      if (newFactor > 1) {
        factor = std::min(factor, newFactor);
        ++count;
      }
      continue;
    }

    /// If met the op which cannot be vectorized, we can replicate it and still
    /// potentially vectorize other ops, but we cannot use masked vectorize.
    if (!isSupportedVectorOp(op)) {
      masked = false;
      continue;
    }

    std::optional<unsigned> width = getArgsTypeWidth(op, DL);
    if (!width)
      return std::nullopt;

    unsigned newFactor = vectorBitwidth / *width;
    if (newFactor <= 1)
      continue;

    factor = std::min(factor, newFactor);

    ++count;
  }

  /// No ops to vectorize.
  if (count == 0)
    return std::nullopt;

  return SCFVectorizeInfo{dim, factor, count, masked};
}

/// Get fastmath flags if ops support them or default (none).
static arith::FastMathFlags getFMF(Operation &op) {
  if (auto fmf = dyn_cast<arith::ArithFastMathInterface>(op))
    return fmf.getFastMathFlagsAttr().getValue();

  return arith::FastMathFlags::none;
}

LogicalResult mlir::scf::vectorizeLoop(scf::ParallelOp loop,
                                       const scf::SCFVectorizeParams &params,
                                       const DataLayout *DL) {
  unsigned dim = params.dim;
  unsigned factor = params.factor;
  bool masked = params.masked;
  assert(dim < loop.getStep().size() && "Invalid loop dimension");
  assert(factor > 1 && "Invalid vectorize factor");
  assert(isConstantIntValue(loop.getStep()[dim], 1) && "Loop stepust be 1");

  OpBuilder builder(loop);
  SmallVector<Value> lower = llvm::to_vector(loop.getLowerBound());
  SmallVector<Value> upper = llvm::to_vector(loop.getUpperBound());
  SmallVector<Value> step = llvm::to_vector(loop.getStep());

  Location loc = loop.getLoc();

  Value origIndexVar = loop.getInductionVars()[dim];

  Value factorVal = builder.create<arith::ConstantIndexOp>(loc, factor);

  Value origLower = lower[dim];
  Value origUpper = upper[dim];
  Value count = builder.createOrFold<arith::SubIOp>(loc, origUpper, origLower);
  Value newCount;

  // Compute new loop count, ceildiv if masked, floordiv otherwise.
  if (masked) {
    newCount = builder.createOrFold<arith::CeilDivSIOp>(loc, count, factorVal);
  } else {
    newCount = builder.createOrFold<arith::DivSIOp>(loc, count, factorVal);
  }

  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  lower[dim] = zero;
  upper[dim] = newCount;

  // Vectorized loop.
  auto newLoop = builder.create<scf::ParallelOp>(loc, lower, upper, step,
                                                 loop.getInitVals());
  Value newIndexVar = newLoop.getInductionVars()[dim];

  auto toVectorType = [&](Type elemType) -> VectorType {
    return VectorType::get(factor, elemType);
  };

  IRMapping mapping;
  IRMapping scalarMapping;

  auto createPosionVec = [&](VectorType vecType) -> Value {
    return builder.create<ub::PoisonOp>(loc, vecType, nullptr);
  };

  Value indexVarMult;
  auto getrIndexVarMult = [&]() -> Value {
    if (indexVarMult)
      return indexVarMult;

    indexVarMult =
        builder.createOrFold<arith::MulIOp>(loc, newIndexVar, factorVal);
    return indexVarMult;
  };

  // Get vector value in new loop for provided `orig` value in source loop.
  auto getVecVal = [&](Value orig) -> Value {
    // Use cached value if present.
    if (Value mapped = mapping.lookupOrNull(orig))
      return mapped;

    // Vectorized loop index, loop index is divided by factor, so for factorN
    // vectorized index will looks like `splat(idx) + (0, 1, ..., N - 1)`
    if (orig == origIndexVar) {
      VectorType vecType = toVectorType(builder.getIndexType());
      SmallVector<Attribute> elems(factor);
      for (auto i : llvm::seq(0u, factor))
        elems[i] = builder.getIndexAttr(i);
      auto attr = DenseElementsAttr::get(vecType, elems);
      Value vec = builder.create<arith::ConstantOp>(loc, vecType, attr);

      Value idx = getrIndexVarMult();
      idx = builder.createOrFold<arith::AddIOp>(loc, idx, origLower);
      idx = builder.create<vector::SplatOp>(loc, idx, vecType);
      vec = builder.createOrFold<arith::AddIOp>(loc, idx, vec);
      mapping.map(orig, vec);
      return vec;
    }
    Type type = orig.getType();
    assert(isSupportedVecElem(type) && "Unsupported vector element type");

    Value val = orig;
    ValueRange origIndexVars = loop.getInductionVars();
    auto it = llvm::find(origIndexVars, orig);

    // If loop index, but not on vectorized dimension, just take new loop index
    // and splat it.
    if (it != origIndexVars.end())
      val = newLoop.getInductionVars()[it - origIndexVars.begin()];

    // Values which are defined inside loop body are preemptively added to the
    // mapper and not handled here. Values defined outside body are just
    // splatted.
    Value vec = builder.create<vector::SplatOp>(loc, val, toVectorType(type));
    mapping.map(orig, vec);
    return vec;
  };

  llvm::DenseMap<Value, SmallVector<Value>> unpackedVals;

  // Get unpacked values for provided `orig` value in source loop.
  // Values are returned as `ValueRange` and not as vector value.
  auto getUnpackedVals = [&](Value val) -> ValueRange {
    // Use cached values if present.
    auto it = unpackedVals.find(val);
    if (it != unpackedVals.end())
      return it->second;

    // Values which are defined inside loop body are preemptively added to the
    // cache and not handled here.

    auto &ret = unpackedVals[val];
    assert(ret.empty() && "Invalid unpackedVals state");
    if (!isSupportedVecElem(val.getType())) {
      // Non vectorizable value, it must be a value defined outside the loop,
      // just replicate it.
      ret.resize(factor, val);
      return ret;
    }

    // Get vector value and extract elements from it.
    Value vecVal = getVecVal(val);
    ret.resize(factor);
    for (auto i : llvm::seq(0u, factor)) {
      Value idx = builder.create<arith::ConstantIndexOp>(loc, i);
      ret[i] = builder.create<vector::ExtractElementOp>(loc, vecVal, idx);
    }
    return ret;
  };

  // Add unpacked values to the cache.
  auto setUnpackedVals = [&](Value origVal, ValueRange newVals) {
    assert(newVals.size() == factor && "Invalid values count");
    assert(unpackedVals.count(origVal) == 0 && "Invalid unpackedVals state");
    unpackedVals[origVal].append(newVals.begin(), newVals.end());

    Type type = origVal.getType();
    if (!isSupportedVecElem(type))
      return;

    // If type is vectorizabale construct a vector add it to vector cache as
    // well.
    VectorType vecType = toVectorType(type);

    Value vec = createPosionVec(vecType);
    for (auto i : llvm::seq(0u, factor)) {
      Value idx = builder.create<arith::ConstantIndexOp>(loc, i);
      vec = builder.create<vector::InsertElementOp>(loc, newVals[i], vec, idx);
    }
    mapping.map(origVal, vec);
  };

  Value mask;

  // Contruct mask value and cache it. If not a masked mode mask is always all
  // 1s.
  auto getMask = [&]() -> Value {
    if (mask)
      return mask;

    OpFoldResult maskSize;
    if (masked) {
      Value size = getrIndexVarMult();
      maskSize = builder.createOrFold<arith::SubIOp>(loc, count, size);
    } else {
      maskSize = builder.getIndexAttr(factor);
    }
    VectorType vecType = toVectorType(builder.getI1Type());
    mask = builder.create<vector::CreateMaskOp>(loc, vecType, maskSize);

    return mask;
  };

  auto canTriviallyVectorizeMemOp = [&](auto op) -> bool {
    return ::canTriviallyVectorizeMemOpImpl(loop, dim, op, DL).has_value();
  };

  auto canGatherScatter = [&](auto op) {
    return ::canGatherScatterImpl(loop, op, DL).has_value();
  };

  // Get idices for vectorized memref load/store.
  auto getMemrefVecIndices = [&](ValueRange indices) -> SmallVector<Value> {
    scalarMapping.clear();
    scalarMapping.map(loop.getInductionVars(), newLoop.getInductionVars());

    SmallVector<Value> ret(indices.size());
    for (auto &&[i, val] : llvm::enumerate(indices)) {
      if (val == origIndexVar) {
        Value idx = getrIndexVarMult();
        idx = builder.createOrFold<arith::AddIOp>(loc, idx, origLower);
        ret[i] = idx;
        continue;
      }
      ret[i] = scalarMapping.lookup(val);
    }

    return ret;
  };

  // Create vectorized memref load for specified non-vectorized load.
  auto genLoad = [&](auto loadOp) {
    SmallVector<Value> indices = getMemrefVecIndices(loadOp.getIndices());
    VectorType resType = toVectorType(loadOp.getResult().getType());
    Value memref = loadOp.getMemRef();
    Value vecLoad;
    if (masked) {
      Value mask = getMask();
      Value init = createPosionVec(resType);
      vecLoad = builder.create<vector::MaskedLoadOp>(loc, resType, memref,
                                                     indices, mask, init);
    } else {
      vecLoad = builder.create<vector::LoadOp>(loc, resType, memref, indices);
    }
    mapping.map(loadOp.getResult(), vecLoad);
  };

  // Create vectorized memref store for specified non-vectorized store.
  auto genStore = [&](auto storeOp) {
    SmallVector<Value> indices = getMemrefVecIndices(storeOp.getIndices());
    Value value = getVecVal(storeOp.getValueToStore());
    Value memref = storeOp.getMemRef();
    if (masked) {
      Value mask = getMask();
      builder.create<vector::MaskedStoreOp>(loc, memref, indices, mask, value);
    } else {
      builder.create<vector::StoreOp>(loc, value, memref, indices);
    }
  };

  SmallVector<Value> duplicatedArgs;
  SmallVector<Value> duplicatedResults;

  builder.setInsertionPointToStart(newLoop.getBody());
  for (Operation &op : loop.getBody()->without_terminator()) {
    loc = op.getLoc();
    if (isSupportedVectorOp(op)) {
      // If op can be vectorized, clone it with vectorized inputs and  update
      // resuls to vectorized types.
      for (Value arg : op.getOperands())
        getVecVal(arg); // init mapper for op args

      Operation *newOp = builder.clone(op, mapping);
      for (Value res : newOp->getResults())
        res.setType(toVectorType(res.getType()));

      continue;
    }

    // Vectorize memref load/store ops, vector load/store are preffered over
    // gather/scatter.
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      if (canTriviallyVectorizeMemOp(loadOp)) {
        genLoad(loadOp);
        continue;
      }
      if (canGatherScatter(loadOp)) {
        VectorType resType = toVectorType(loadOp.getResult().getType());
        Value memref = loadOp.getMemRef();
        Value mask = getMask();
        Value indexVec = getVecVal(loadOp.getIndices()[0]);
        Value init = createPosionVec(resType);

        auto gather = builder.create<vector::GatherOp>(
            loc, resType, memref, zero, indexVec, mask, init);
        mapping.map(loadOp.getResult(), gather.getResult());
        continue;
      }
    }

    if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      if (canTriviallyVectorizeMemOp(storeOp)) {
        genStore(storeOp);
        continue;
      }
      if (canGatherScatter(storeOp)) {
        Value memref = storeOp.getMemRef();
        Value value = getVecVal(storeOp.getValueToStore());
        Value mask = getMask();
        Value indexVec = getVecVal(storeOp.getIndices()[0]);

        builder.create<vector::ScatterOp>(loc, memref, zero, indexVec, mask,
                                          value);
        continue;
      }
    }

    // Fallback: Failed to vectorize op, just duplicate it `factor` times
    if (masked)
      return op.emitError("Cannot vectorize op in masked mode");

    scalarMapping.clear();

    auto numArgs = op.getNumOperands();
    auto numResults = op.getNumResults();
    duplicatedArgs.resize(numArgs * factor);
    duplicatedResults.resize(numResults * factor);

    for (auto &&[i, arg] : llvm::enumerate(op.getOperands())) {
      ValueRange unpacked = getUnpackedVals(arg);
      assert(unpacked.size() == factor && "Invalid unpacked size");
      for (auto j : llvm::seq(0u, factor))
        duplicatedArgs[j * numArgs + i] = unpacked[j];
    }

    for (auto i : llvm::seq(0u, factor)) {
      auto args = ValueRange(duplicatedArgs)
                      .drop_front(numArgs * i)
                      .take_front(numArgs);
      scalarMapping.map(op.getOperands(), args);
      ValueRange results = builder.clone(op, scalarMapping)->getResults();

      for (auto j : llvm::seq(0u, numResults))
        duplicatedResults[j * factor + i] = results[j];
    }

    for (auto i : llvm::seq(0u, numResults)) {
      auto results = ValueRange(duplicatedResults)
                         .drop_front(factor * i)
                         .take_front(factor);
      setUnpackedVals(op.getResult(i), results);
    }
  }

  // Vectorize `scf.reduce` op.
  auto reduceOp = cast<scf::ReduceOp>(loop.getBody()->getTerminator());
  SmallVector<Value> reduceVals;
  reduceVals.reserve(reduceOp.getNumOperands());

  for (auto &&[body, arg] :
       llvm::zip(reduceOp.getReductions(), reduceOp.getOperands())) {
    scalarMapping.clear();
    Block &reduceBody = body.front();
    assert(reduceBody.getNumArguments() == 2 && "Malformed scf.reduce");

    Value reduceVal;
    if (auto redKind = getReductionKind(reduceBody)) {
      // Generate `vector.reduce` if possible.
      Value redArg = getVecVal(arg);
      if (redArg) {
        std::optional<TypedAttr> neutral =
            arith::getNeutralElement(&reduceBody.front());
        assert(neutral && "getNeutralElement has unepectedly failed");
        Value neutralVal = builder.create<arith::ConstantOp>(loc, *neutral);
        Value neutralVec =
            builder.create<vector::SplatOp>(loc, neutralVal, redArg.getType());
        Value mask = getMask();
        redArg = builder.create<arith::SelectOp>(loc, mask, redArg, neutralVec);
      }

      arith::FastMathFlags fmf = getFMF(reduceBody.front());
      reduceVal =
          builder.create<vector::ReductionOp>(loc, *redKind, redArg, fmf);
    } else {
      if (masked)
        return reduceOp.emitError("Cannot vectorize reduce op in masked mode");

      // If `vector.reduce` cannot be used, unpack values and reduce them
      // individually.

      auto reduceTerm = cast<scf::ReduceReturnOp>(reduceBody.getTerminator());
      Value lhs = reduceBody.getArgument(0);
      Value rhs = reduceBody.getArgument(1);
      ValueRange unpacked = getUnpackedVals(arg);
      assert(unpacked.size() == factor && "Invalid unpacked size");
      reduceVal = unpacked.front();
      for (auto i : llvm::seq(1u, factor)) {
        Value val = unpacked[i];
        scalarMapping.map(lhs, reduceVal);
        scalarMapping.map(rhs, val);
        for (Operation &redOp : reduceBody.without_terminator())
          builder.clone(redOp, scalarMapping);

        reduceVal = scalarMapping.lookupOrDefault(reduceTerm.getResult());
      }
    }
    reduceVals.emplace_back(reduceVal);
  }

  // Clone `scf.reduce` op to reduce across loop iterations.
  if (!reduceVals.empty())
    builder.clone(*reduceOp)->setOperands(reduceVals);

  // If in masked mode remove old loop, otherwise update loop bounds to
  // repurpose it for handling remaining values.
  if (masked) {
    loop->replaceAllUsesWith(newLoop.getResults());
    loop->erase();
  } else {
    builder.setInsertionPoint(loop);
    Value newLower =
        builder.createOrFold<arith::MulIOp>(loc, newCount, factorVal);
    newLower = builder.createOrFold<arith::AddIOp>(loc, origLower, newLower);

    SmallVector<Value> lowerCopy = llvm::to_vector(loop.getLowerBound());
    lowerCopy[dim] = newLower;
    loop.getLowerBoundMutable().assign(lowerCopy);
    loop.getInitValsMutable().assign(newLoop.getResults());
  }

  return success();
}
