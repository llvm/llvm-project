//===- ControlFlowSink.cpp - Code to perform control-flow sinking ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/SCFVectorize.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>

/// Return type bitwidth for vectorization purposes or 0 if type cannot be
/// vectorized.
static unsigned getTypeBitWidth(mlir::Type type) {
  if (mlir::isa<mlir::IndexType>(type))
    return 64; // TODO: unhardcode

  if (type.isIntOrFloat())
    return type.getIntOrFloatBitWidth();

  return 0;
}

static unsigned getArgsTypeWidth(mlir::Operation &op) {
  unsigned ret = 0;
  for (auto arg : op.getOperands())
    ret = std::max(ret, getTypeBitWidth(arg.getType()));

  for (auto res : op.getResults())
    ret = std::max(ret, getTypeBitWidth(res.getType()));

  return ret;
}

static bool isSupportedVectorOp(mlir::Operation &op) {
  return op.hasTrait<mlir::OpTrait::Vectorizable>();
}

static bool isSupportedVecElem(mlir::Type type) {
  return type.isIntOrIndexOrFloat();
}

/// Check if one `ValueRange` is permutation of another, i.e. contains same
/// values, potentially in different order.
static bool isRangePermutation(mlir::ValueRange val1, mlir::ValueRange val2) {
  if (val1.size() != val2.size())
    return false;

  for (auto v1 : val1) {
    auto it = llvm::find(val2, v1);
    if (it == val2.end())
      return false;
  }
  return true;
}

template <typename Op>
static std::optional<unsigned>
cavTriviallyVectorizeMemOpImpl(mlir::scf::ParallelOp loop, unsigned dim,
                               Op memOp) {
  auto loopIndexVars = loop.getInductionVars();
  assert(dim < loopIndexVars.size());
  auto memref = memOp.getMemRef();
  auto type = mlir::cast<mlir::MemRefType>(memref.getType());
  auto width = getTypeBitWidth(type.getElementType());
  if (width == 0)
    return std::nullopt;

  if (!type.getLayout().isIdentity())
    return std::nullopt;

  if (!isRangePermutation(memOp.getIndices(), loopIndexVars))
    return std::nullopt;

  if (memOp.getIndices().back() != loopIndexVars[dim])
    return std::nullopt;

  mlir::DominanceInfo dom;
  if (!dom.properlyDominates(memref, loop))
    return std::nullopt;

  return width;
}

/// Check if memref load/store can be converted into vectorized load/store
///
/// Returns memref element bitwidth or `std::nullopt` if access cannot be
/// vectorized.
static std::optional<unsigned>
cavTriviallyVectorizeMemOp(mlir::scf::ParallelOp loop, unsigned dim,
                           mlir::Operation &op) {
  assert(dim < loop.getInductionVars().size());
  if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(op))
    return cavTriviallyVectorizeMemOpImpl(loop, dim, storeOp);

  if (auto loadOp = mlir::dyn_cast<mlir::memref::LoadOp>(op))
    return cavTriviallyVectorizeMemOpImpl(loop, dim, loadOp);

  return std::nullopt;
}

template <typename T>
static bool isOp(mlir::Operation &op) {
  return mlir::isa<T>(op);
}

/// Returns `vector.reduce` kind for specified `scf.parallel` reduce op ot
/// `std::nullopt` if reduction cannot be handled by `vector.reduce`.
static std::optional<mlir::vector::CombiningKind>
getReductionKind(mlir::Block &body) {
  if (!llvm::hasSingleElement(body.without_terminator()))
    return std::nullopt;

  mlir::Operation &redOp = body.front();

  using fptr_t = bool (*)(mlir::Operation &);
  using CC = mlir::vector::CombiningKind;
  const std::pair<fptr_t, CC> handlers[] = {
      // clang-format off
      {&isOp<mlir::arith::AddIOp>, CC::ADD},
      {&isOp<mlir::arith::AddFOp>, CC::ADD},
      {&isOp<mlir::arith::MulIOp>, CC::MUL},
      {&isOp<mlir::arith::MulFOp>, CC::MUL},
      // clang-format on
  };

  for (auto &&[handler, cc] : handlers) {
    if (handler(redOp))
      return cc;
  }

  return std::nullopt;
}

std::optional<mlir::SCFVectorizeInfo>
mlir::getLoopVectorizeInfo(mlir::scf::ParallelOp loop, unsigned dim,
                           unsigned vectorBitwidth) {
  assert(dim < loop.getStep().size());
  assert(vectorBitwidth > 0);
  unsigned factor = vectorBitwidth / 8;
  if (factor <= 1)
    return std::nullopt;

  /// Only step==1 is supported for now.
  if (!mlir::isConstantIntValue(loop.getStep()[dim], 1))
    return std::nullopt;

  unsigned count = 0;
  bool masked = true;

  /// Check if `scf.reduce` can be handled by `vector.reduce`.
  /// If not we still can vectorize the loop but we cannot use masked
  /// vectorize.
  auto reduce =
      mlir::cast<mlir::scf::ReduceOp>(loop.getBody()->getTerminator());
  for (mlir::Region &reg : reduce.getReductions()) {
    if (!getReductionKind(reg.front()))
      masked = false;

    continue;
  }

  for (mlir::Operation &op : loop.getBody()->without_terminator()) {
    /// Ops with nested regions are not supported yet.
    if (op.getNumRegions() > 0)
      return std::nullopt;

    /// Check mem ops.
    if (auto w = cavTriviallyVectorizeMemOp(loop, dim, op)) {
      auto newFactor = vectorBitwidth / *w;
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

    auto width = getArgsTypeWidth(op);
    if (width == 0)
      return std::nullopt;

    auto newFactor = vectorBitwidth / width;
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
static mlir::arith::FastMathFlags getFMF(mlir::Operation &op) {
  if (auto fmf = mlir::dyn_cast<mlir::arith::ArithFastMathInterface>(op))
    return fmf.getFastMathFlagsAttr().getValue();

  return mlir::arith::FastMathFlags::none;
}

mlir::LogicalResult
mlir::vectorizeLoop(mlir::OpBuilder &builder, mlir::scf::ParallelOp loop,
                    const mlir::SCFVectorizeParams &params) {
  auto dim = params.dim;
  auto factor = params.factor;
  auto masked = params.masked;
  assert(dim < loop.getStep().size());
  assert(factor > 1);
  assert(mlir::isConstantIntValue(loop.getStep()[dim], 1));

  mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(loop);

  auto lower = llvm::to_vector(loop.getLowerBound());
  auto upper = llvm::to_vector(loop.getUpperBound());
  auto step = llvm::to_vector(loop.getStep());

  auto loc = loop.getLoc();

  auto origIndexVar = loop.getInductionVars()[dim];

  mlir::Value factorVal =
      builder.create<mlir::arith::ConstantIndexOp>(loc, factor);

  auto origLower = lower[dim];
  auto origUpper = upper[dim];
  mlir::Value count =
      builder.create<mlir::arith::SubIOp>(loc, origUpper, origLower);
  mlir::Value newCount;

  // Compute new loop count, ceildiv if masked, floordiv otherwise.
  if (masked) {
    mlir::Value incCount =
        builder.create<mlir::arith::AddIOp>(loc, count, factorVal);
    mlir::Value one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    incCount = builder.create<mlir::arith::SubIOp>(loc, incCount, one);
    newCount = builder.create<mlir::arith::DivSIOp>(loc, incCount, factorVal);
  } else {
    newCount = builder.create<mlir::arith::DivSIOp>(loc, count, factorVal);
  }

  mlir::Value zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  lower[dim] = zero;
  upper[dim] = newCount;

  // Vectorized loop.
  auto newLoop = builder.create<mlir::scf::ParallelOp>(loc, lower, upper, step,
                                                       loop.getInitVals());
  auto newIndexVar = newLoop.getInductionVars()[dim];

  auto toVectorType = [&](mlir::Type elemType) -> mlir::VectorType {
    int64_t f = factor;
    return mlir::VectorType::get(f, elemType);
  };

  mlir::IRMapping mapping;
  mlir::IRMapping scalarMapping;

  auto createPosionVec = [&](mlir::VectorType vecType) -> mlir::Value {
    return builder.create<mlir::ub::PoisonOp>(loc, vecType, nullptr);
  };

  // Get vector value in new loop for provided `orig` value in source loop.
  auto getVecVal = [&](mlir::Value orig) -> mlir::Value {
    // Use cached value if present.
    if (auto mapped = mapping.lookupOrNull(orig))
      return mapped;

    // Vectorized loop index, loop index is divided by factor, so for factorN
    // vectorized index will looks like `splat(idx) + (0, 1, ..., N - 1)`
    if (orig == origIndexVar) {
      auto vecType = toVectorType(builder.getIndexType());
      llvm::SmallVector<mlir::Attribute> elems(factor);
      for (auto i : llvm::seq(0u, factor))
        elems[i] = builder.getIndexAttr(i);
      auto attr = mlir::DenseElementsAttr::get(vecType, elems);
      mlir::Value vec =
          builder.create<mlir::arith::ConstantOp>(loc, vecType, attr);

      mlir::Value idx =
          builder.create<mlir::arith::MulIOp>(loc, newIndexVar, factorVal);
      idx = builder.create<mlir::arith::AddIOp>(loc, idx, origLower);
      idx = builder.create<mlir::vector::SplatOp>(loc, idx, vecType);
      vec = builder.create<mlir::arith::AddIOp>(loc, idx, vec);
      mapping.map(orig, vec);
      return vec;
    }
    auto type = orig.getType();
    assert(isSupportedVecElem(type));

    mlir::Value val = orig;
    auto origIndexVars = loop.getInductionVars();
    auto it = llvm::find(origIndexVars, orig);

    // If loop index, but not on vectorized dimension, just take new loop index
    // and splat it.
    if (it != origIndexVars.end())
      val = newLoop.getInductionVars()[it - origIndexVars.begin()];

    // Values which are defined inside loop body are preemptively added to the
    // mapper and not handled here. Values defined outside body are just
    // splatted.

    auto vecType = toVectorType(type);
    mlir::Value vec = builder.create<mlir::vector::SplatOp>(loc, val, vecType);
    mapping.map(orig, vec);
    return vec;
  };

  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>> unpackedVals;

  // Get unpacked values for provided `orig` value in source loop.
  // Values are returned as `ValueRange` and not as vector value.
  auto getUnpackedVals = [&](mlir::Value val) -> mlir::ValueRange {
    // Use cached values if present.
    auto it = unpackedVals.find(val);
    if (it != unpackedVals.end())
      return it->second;

    // Values which are defined inside loop body are preemptively added to the
    // cache and not handled here.

    auto &ret = unpackedVals[val];
    assert(ret.empty());
    if (!isSupportedVecElem(val.getType())) {
      // Non vectorizable value, it must be a value defined outside the loop,
      // just replicate it.
      ret.resize(factor, val);
      return ret;
    }

    // Get vector value and extract elements from it.
    auto vecVal = getVecVal(val);
    ret.resize(factor);
    for (auto i : llvm::seq(0u, factor)) {
      mlir::Value idx = builder.create<mlir::arith::ConstantIndexOp>(loc, i);
      ret[i] = builder.create<mlir::vector::ExtractElementOp>(loc, vecVal, idx);
    }
    return ret;
  };

  // Add unpacked values to the cache.
  auto setUnpackedVals = [&](mlir::Value origVal, mlir::ValueRange newVals) {
    assert(newVals.size() == factor);
    assert(unpackedVals.count(origVal) == 0);
    unpackedVals[origVal].append(newVals.begin(), newVals.end());

    auto type = origVal.getType();
    if (!isSupportedVecElem(type))
      return;

    // If type is vectorizabale construct a vector add it to vector cache as
    // well.
    auto vecType = toVectorType(type);

    mlir::Value vec = createPosionVec(vecType);
    for (auto i : llvm::seq(0u, factor)) {
      mlir::Value idx = builder.create<mlir::arith::ConstantIndexOp>(loc, i);
      vec = builder.create<mlir::vector::InsertElementOp>(loc, newVals[i], vec,
                                                          idx);
    }
    mapping.map(origVal, vec);
  };

  mlir::Value mask;

  // Contruct mask value and cache it. If not a masked mode mask is always all
  // 1s.
  auto getMask = [&]() -> mlir::Value {
    if (mask)
      return mask;

    mlir::OpFoldResult maskSize;
    if (masked) {
      mlir::Value size =
          builder.create<mlir::arith::MulIOp>(loc, factorVal, newIndexVar);
      maskSize =
          builder.create<mlir::arith::SubIOp>(loc, count, size).getResult();
    } else {
      maskSize = builder.getIndexAttr(factor);
    }
    auto vecType = toVectorType(builder.getI1Type());
    mask = builder.create<mlir::vector::CreateMaskOp>(loc, vecType, maskSize);

    return mask;
  };

  mlir::DominanceInfo dom;

  auto canTriviallyVectorizeMemOp = [&](auto op) -> bool {
    return !!::cavTriviallyVectorizeMemOpImpl(loop, dim, op);
  };

  // Get idices for vectorized memref load/store.
  auto getMemrefVecIndices = [&](mlir::ValueRange indices) {
    scalarMapping.clear();
    scalarMapping.map(loop.getInductionVars(), newLoop.getInductionVars());

    llvm::SmallVector<mlir::Value> ret(indices.size());
    for (auto &&[i, val] : llvm::enumerate(indices)) {
      if (val == origIndexVar) {
        mlir::Value idx =
            builder.create<mlir::arith::MulIOp>(loc, newIndexVar, factorVal);
        idx = builder.create<mlir::arith::AddIOp>(loc, idx, origLower);
        ret[i] = idx;
        continue;
      }
      ret[i] = scalarMapping.lookup(val);
    }

    return ret;
  };

  // Check if memref access can be converted into gather/scatter.
  auto canGatherScatter = [&](auto op) {
    auto memref = op.getMemRef();
    auto memrefType = mlir::cast<mlir::MemRefType>(memref.getType());
    if (!isSupportedVecElem(memrefType.getElementType()))
      return false;

    return dom.properlyDominates(memref, loop) && op.getIndices().size() == 1 &&
           memrefType.getLayout().isIdentity();
  };

  // Create vectorized memref load for specified non-vectorized load.
  auto genLoad = [&](auto loadOp) {
    auto indices = getMemrefVecIndices(loadOp.getIndices());
    auto resType = toVectorType(loadOp.getResult().getType());
    auto memref = loadOp.getMemRef();
    mlir::Value vecLoad;
    if (masked) {
      auto mask = getMask();
      auto init = createPosionVec(resType);
      vecLoad = builder.create<mlir::vector::MaskedLoadOp>(loc, resType, memref,
                                                           indices, mask, init);
    } else {
      vecLoad =
          builder.create<mlir::vector::LoadOp>(loc, resType, memref, indices);
    }
    mapping.map(loadOp.getResult(), vecLoad);
  };

  // Create vectorized memref store for specified non-vectorized store.
  auto genStore = [&](auto storeOp) {
    auto indices = getMemrefVecIndices(storeOp.getIndices());
    auto value = getVecVal(storeOp.getValueToStore());
    auto memref = storeOp.getMemRef();
    if (masked) {
      auto mask = getMask();
      builder.create<mlir::vector::MaskedStoreOp>(loc, memref, indices, mask,
                                                  value);
    } else {
      builder.create<mlir::vector::StoreOp>(loc, value, memref, indices);
    }
  };

  llvm::SmallVector<mlir::Value> duplicatedArgs;
  llvm::SmallVector<mlir::Value> duplicatedResults;

  builder.setInsertionPointToStart(newLoop.getBody());
  for (mlir::Operation &op : loop.getBody()->without_terminator()) {
    loc = op.getLoc();
    if (isSupportedVectorOp(op)) {
      // If op can be vectorized, clone it with vectorized inputs and  update
      // resuls to vectorized types.
      for (auto arg : op.getOperands())
        getVecVal(arg); // init mapper for op args

      auto newOp = builder.clone(op, mapping);
      for (auto res : newOp->getResults())
        res.setType(toVectorType(res.getType()));

      continue;
    }

    // Vectorize memref load/store ops, vector load/store are preffered over
    // gather/scatter.
    if (auto loadOp = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
      if (canTriviallyVectorizeMemOp(loadOp)) {
        genLoad(loadOp);
        continue;
      }
      if (canGatherScatter(loadOp)) {
        auto resType = toVectorType(loadOp.getResult().getType());
        auto memref = loadOp.getMemRef();
        auto mask = getMask();
        auto indexVec = getVecVal(loadOp.getIndices()[0]);
        auto init = createPosionVec(resType);

        auto gather = builder.create<mlir::vector::GatherOp>(
            loc, resType, memref, zero, indexVec, mask, init);
        mapping.map(loadOp.getResult(), gather.getResult());
        continue;
      }
    }

    if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
      if (canTriviallyVectorizeMemOp(storeOp)) {
        genStore(storeOp);
        continue;
      }
      if (canGatherScatter(storeOp)) {
        auto memref = storeOp.getMemRef();
        auto value = getVecVal(storeOp.getValueToStore());
        auto mask = getMask();
        auto indexVec = getVecVal(storeOp.getIndices()[0]);

        builder.create<mlir::vector::ScatterOp>(loc, memref, zero, indexVec,
                                                mask, value);
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
      auto unpacked = getUnpackedVals(arg);
      assert(unpacked.size() == factor);
      for (auto j : llvm::seq(0u, factor))
        duplicatedArgs[j * numArgs + i] = unpacked[j];
    }

    for (auto i : llvm::seq(0u, factor)) {
      auto args = mlir::ValueRange(duplicatedArgs)
                      .drop_front(numArgs * i)
                      .take_front(numArgs);
      scalarMapping.map(op.getOperands(), args);
      auto results = builder.clone(op, scalarMapping)->getResults();

      for (auto j : llvm::seq(0u, numResults))
        duplicatedResults[j * factor + i] = results[j];
    }

    for (auto i : llvm::seq(0u, numResults)) {
      auto results = mlir::ValueRange(duplicatedResults)
                         .drop_front(factor * i)
                         .take_front(factor);
      setUnpackedVals(op.getResult(i), results);
    }
  }

  // Vectorize `scf.reduce` op.
  auto reduceOp =
      mlir::cast<mlir::scf::ReduceOp>(loop.getBody()->getTerminator());
  llvm::SmallVector<mlir::Value> reduceVals;
  reduceVals.reserve(reduceOp.getNumOperands());

  for (auto &&[body, arg] :
       llvm::zip(reduceOp.getReductions(), reduceOp.getOperands())) {
    scalarMapping.clear();
    mlir::Block &reduceBody = body.front();
    assert(reduceBody.getNumArguments() == 2);

    mlir::Value reduceVal;
    if (auto redKind = getReductionKind(reduceBody)) {
      // Generate `vector.reduce` if possible.
      mlir::Value redArg = getVecVal(arg);
      if (redArg) {
        auto neutral = mlir::arith::getNeutralElement(&reduceBody.front());
        assert(neutral);
        mlir::Value neutralVal =
            builder.create<mlir::arith::ConstantOp>(loc, *neutral);
        mlir::Value neutralVec = builder.create<mlir::vector::SplatOp>(
            loc, neutralVal, redArg.getType());
        auto mask = getMask();
        redArg = builder.create<mlir::arith::SelectOp>(loc, mask, redArg,
                                                       neutralVec);
      }

      auto fmf = getFMF(reduceBody.front());
      reduceVal =
          builder.create<mlir::vector::ReductionOp>(loc, *redKind, redArg, fmf);
    } else {
      if (masked)
        return reduceOp.emitError("Cannot vectorize reduce op in masked mode");

      // If `vector.reduce` cannot be used, unpack values and reduce them
      // individually.

      auto reduceTerm =
          mlir::cast<mlir::scf::ReduceReturnOp>(reduceBody.getTerminator());
      auto lhs = reduceBody.getArgument(0);
      auto rhs = reduceBody.getArgument(1);
      auto unpacked = getUnpackedVals(arg);
      assert(unpacked.size() == factor);
      reduceVal = unpacked.front();
      for (auto i : llvm::seq(1u, factor)) {
        mlir::Value val = unpacked[i];
        scalarMapping.map(lhs, reduceVal);
        scalarMapping.map(rhs, val);
        for (auto &redOp : reduceBody.without_terminator())
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
    mlir::Value newLower =
        builder.create<mlir::arith::MulIOp>(loc, newCount, factorVal);
    newLower = builder.create<mlir::arith::AddIOp>(loc, origLower, newLower);

    auto lowerCopy = llvm::to_vector(loop.getLowerBound());
    lowerCopy[dim] = newLower;
    loop.getLowerBoundMutable().assign(lowerCopy);
    loop.getInitValsMutable().assign(newLoop.getResults());
  }

  return mlir::success();
}

static std::optional<unsigned> getVectorLength(mlir::Operation *op) {
  auto func = op->getParentOfType<mlir::FunctionOpInterface>();
  if (!func)
    return std::nullopt;

  auto attr = func->getAttrOfType<mlir::IntegerAttr>("mlir.vector_length");
  if (!attr)
    return std::nullopt;

  auto val = attr.getInt();
  if (val <= 0 || val > std::numeric_limits<unsigned>::max())
    return std::nullopt;

  return static_cast<unsigned>(val);
}

namespace {
struct SCFVectorizePass
    : public mlir::PassWrapper<SCFVectorizePass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SCFVectorizePass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::ub::UBDialect>();
    registry.insert<mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    llvm::SmallVector<
        std::pair<mlir::scf::ParallelOp, mlir::SCFVectorizeParams>>
        toVectorize;

    // Simple heuristic: total number of elements processed by vector ops, but
    // prefer masked mode over non-masked.
    auto getBenefit = [](const mlir::SCFVectorizeInfo &info) {
      return info.factor * info.count * (int(info.masked) + 1);
    };

    getOperation()->walk([&](mlir::scf::ParallelOp loop) {
      auto len = getVectorLength(loop);
      if (!len)
        return;

      std::optional<mlir::SCFVectorizeInfo> best;
      for (auto dim : llvm::seq(0u, loop.getNumLoops())) {
        auto info = mlir::getLoopVectorizeInfo(loop, dim, *len);
        if (!info)
          continue;

        if (!best) {
          best = *info;
          continue;
        }

        if (getBenefit(*info) > getBenefit(*best))
          best = *info;
      }

      if (!best)
        return;

      toVectorize.emplace_back(
          loop,
          mlir::SCFVectorizeParams{best->dim, best->factor, best->masked});
    });

    if (toVectorize.empty())
      return markAllAnalysesPreserved();

    mlir::OpBuilder builder(&getContext());
    for (auto &&[loop, params] : toVectorize) {
      builder.setInsertionPoint(loop);
      if (mlir::failed(mlir::vectorizeLoop(builder, loop, params)))
        return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::createSCFVectorizePass() {
  return std::make_unique<SCFVectorizePass>();
}