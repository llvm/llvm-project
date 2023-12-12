//===- CodegenEnv.cpp -  Code generation environment class ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodegenEnv.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <optional>

using namespace mlir;
using namespace mlir::sparse_tensor;

//===----------------------------------------------------------------------===//
// Code generation environment helper functions
//===----------------------------------------------------------------------===//

/// Returns true if tensor materializes uninitialized into the computation.
static bool isMaterializing(Value val) {
  return val.getDefiningOp<tensor::EmptyOp>() ||
         val.getDefiningOp<bufferization::AllocTensorOp>();
}

/// Sorts the dependent loops such that it is ordered in the same sequence in
/// which loops will be generated.
static void sortDependentLoops(std::vector<LoopCoeffPair> &target) {
  std::sort(target.begin(), target.end(),
            [](const LoopCoeffPair &l, const LoopCoeffPair &r) {
              assert(std::addressof(l) == std::addressof(r) || l != r);
              return l.first < r.first;
            });
}
//===----------------------------------------------------------------------===//
// Code generation environment constructor and general methods
//===----------------------------------------------------------------------===//

CodegenEnv::CodegenEnv(linalg::GenericOp linop, SparsificationOptions opts,
                       unsigned numTensors, unsigned numLoops, unsigned maxRank)
    : linalgOp(linop), sparseOptions(opts),
      latticeMerger(numTensors, numLoops, maxRank), loopEmitter(),
      sparseOut(nullptr), outerParNest(-1u), insChain(), expValues(),
      expFilled(), expAdded(), expCount(), redVal(), redExp(detail::kInvalidId),
      redCustom(detail::kInvalidId), redValidLexInsert() {}

LogicalResult CodegenEnv::initTensorExp() {
  // Builds the tensor expression for the Linalg operation in SSA form.
  std::optional<ExprId> optExp = latticeMerger.buildTensorExpFromLinalg(op());
  if (!optExp || !isAdmissibleTensorExp(*optExp))
    return failure();

  tensorExp = *optExp;
  return success();
}

void CodegenEnv::startEmit() {
  assert(insChain == nullptr && "must only start emitting once");
  if (sparseOut) {
    insChain = sparseOut->get();
    latticeMerger.setHasSparseOut(true);
  }

  // Sort the related loop array such that they are in the same order as they
  // appears on the topoOrder.
  // TODO: since we only handle affine addition for slice based codegen, and
  // addition is assoicative, the order how we evaluate the expression does
  // not matter. However, to support multiplication, the order of the loop
  // index should match the evaluation order to the affine expression AST.

  // Initialize loop emitter.
  SmallVector<Value> tensors; // input tensors passed to loop emitter
  for (OpOperand &t : linalgOp->getOpOperands()) {
    tensors.push_back(t.get());
    const TensorId tid = makeTensorId(t.getOperandNumber());
    const Level lvlRank = linalgOp.getMatchingIndexingMap(&t).getNumResults();
    const auto enc = getSparseTensorEncoding(t.get().getType());
    (void)enc;
    assert(!enc || lvlRank == enc.getLvlRank());
    for (Level lvl = 0; lvl < lvlRank; lvl++)
      sortDependentLoops(latticeMerger.getDependentLoops(tid, lvl));
  }

  loopEmitter.initialize(
      tensors,
      StringAttr::get(linalgOp.getContext(),
                      linalg::GenericOp::getOperationName()),
      /*hasOutput=*/true,
      /*isSparseOut=*/sparseOut != nullptr, /*numLoops=*/getLoopNum(),
      // TODO: compute the map and pass it to loop emitter directly instead of
      // passing in a callback.
      /*dependentLvlGetter=*/
      [this](TensorId t,
             Level lvl) -> std::vector<std::pair<TensorLevel, unsigned>> {
        // Translates from a list of loop indices to a list of [tid, lvl] pair.
        std::vector<LoopCoeffPair> &rLoops = merger().getDependentLoops(t, lvl);
        std::vector<std::pair<TensorLevel, unsigned>> ret;
        ret.reserve(rLoops.size());
        for (auto [loop, coeff] : rLoops) {
          TensorLevel tl = makeTensorLevel(merger().getLoopDefiningLvl(loop));
          ret.emplace_back(tl, coeff);
        };
        return ret;
      });
}

std::optional<Operation *> CodegenEnv::genLoopBoundary(
    function_ref<std::optional<Operation *>(MutableArrayRef<Value> parameters)>
        callback) {
  SmallVector<Value> params;
  if (isReduc()) {
    params.push_back(redVal);
    if (isValidLexInsert())
      params.push_back(redValidLexInsert);
  } else {
    assert(!isValidLexInsert());
  }
  if (isExpand())
    params.push_back(expCount);
  if (insChain != nullptr)
    params.push_back(insChain);
  auto r = callback(params); // may update parameters
  unsigned i = 0;
  if (isReduc()) {
    updateReduc(params[i++]);
    if (isValidLexInsert())
      updateValidLexInsert(params[i++]);
  }
  if (isExpand())
    updateExpandCount(params[i++]);
  if (insChain != nullptr)
    updateInsertionChain(params[i]);
  return r;
}

//===----------------------------------------------------------------------===//
// Code generation environment verify functions.
//===----------------------------------------------------------------------===//

bool CodegenEnv::isAdmissibleTensorExp(ExprId exp) {
  // We reject any expression that makes a reduction from `-outTensor`, as those
  // expressions create a dependency between the current iteration (i) and the
  // previous iteration (i-1). It would require iterating over the whole
  // coordinate space, which prevent exploiting sparsity for faster code.
  for (utils::IteratorType it : linalgOp.getIteratorTypesArray()) {
    if (it == utils::IteratorType::reduction) {
      if (latticeMerger.hasNegateOnOut(exp))
        return false;
      break;
    }
  }

  OpOperand *lhs = linalgOp.getDpsInitOperand(0);
  const TensorId tensor = makeTensorId(lhs->getOperandNumber());
  // An non-annotated output tensor is assumed dense, and becomes a random
  // access n-dim memref. Admissible since insertions cannot occur.
  if (getSparseTensorType(lhs->get()).isAllDense())
    return true;

  // A tensor expression with a sparse output tensor that changes its values
  // but not its nonzero structure, an operation called "simply dynamic" in
  // [Bik96,Ch9], is also admissible without special env.
  if (latticeMerger.isSingleCondition(tensor, exp))
    return true;

  // Accept "truly dynamic" if the output tensor materializes uninitialized
  // into the computation and insertions occur in lexicographic index order.
  sparseOut = lhs;

  // Find the outermost parallel nest to determine whether compress/expand is
  // needed.
  outerParNest = 0;
  const auto iteratorTypes = linalgOp.getIteratorTypesArray();
  for (unsigned i = 0, e = getLoopNum(); i < e; i++) {
    if (linalg::isReductionIterator(iteratorTypes[i]))
      break; // terminate at first reduction
    outerParNest++;
  }

  // Inadmissible kernel should have already been rejected by the previous
  // path during loop scheduling.
  assert(static_cast<int64_t>(outerParNest) >=
         linalgOp.getRank(linalgOp.getDpsInitOperand(0)) - 1);
  return isMaterializing(lhs->get());
}

//===----------------------------------------------------------------------===//
// Code generation environment topological sort methods
//===----------------------------------------------------------------------===//

Value CodegenEnv::getLoopVar(LoopId i) const {
  return loopEmitter.getLoopIV(i);
}

//===----------------------------------------------------------------------===//
// Code generation environment sparse tensor output and expansion methods
//===----------------------------------------------------------------------===//

void CodegenEnv::updateInsertionChain(Value chain) {
  assert(sparseOut != nullptr && insChain != nullptr);
  insChain = chain;
}

bool CodegenEnv::atExpandLevel(OpOperand *o, unsigned rank, LoopId n) const {
  return sparseOut == o && outerParNest == static_cast<LoopId>(rank - 1) &&
         outerParNest == n;
}

void CodegenEnv::startExpand(Value values, Value filled, Value added,
                             Value count) {
  assert(sparseOut != nullptr && expValues == nullptr);
  expValues = values;
  expFilled = filled;
  expAdded = added;
  expCount = count;
}

void CodegenEnv::updateExpandCount(Value count) {
  assert(sparseOut != nullptr && expValues != nullptr);
  expCount = count;
}

void CodegenEnv::endExpand() {
  assert(sparseOut != nullptr && expValues != nullptr);
  expValues = expFilled = expAdded = expCount = Value();
}

//===----------------------------------------------------------------------===//
// Code generation environment reduction methods
//===----------------------------------------------------------------------===//

void CodegenEnv::startReduc(ExprId exp, Value val) {
  assert(!isReduc() && exp != detail::kInvalidId && val);
  redExp = exp;
  redVal = val;
  latticeMerger.setExprValue(exp, val);
}

void CodegenEnv::updateReduc(Value val) {
  assert(isReduc() && val);
  redVal = val;
  latticeMerger.clearExprValue(redExp);
  latticeMerger.setExprValue(redExp, val);
}

Value CodegenEnv::endReduc() {
  assert(isReduc());
  Value val = redVal;
  redVal = val;
  latticeMerger.clearExprValue(redExp);
  redExp = detail::kInvalidId;
  return val;
}

void CodegenEnv::startValidLexInsert(Value val) {
  assert(!isValidLexInsert() && isReduc() && val);
  redValidLexInsert = val;
}

void CodegenEnv::updateValidLexInsert(Value val) {
  assert(redValidLexInsert && isReduc() && val);
  redValidLexInsert = val;
}

void CodegenEnv::endValidLexInsert() {
  assert(isValidLexInsert() && !isReduc());
  redValidLexInsert = Value();
}

void CodegenEnv::startCustomReduc(ExprId exp) {
  assert(!isCustomReduc() && exp != detail::kInvalidId);
  redCustom = exp;
}

Value CodegenEnv::getCustomRedId() const {
  assert(isCustomReduc());
  return dyn_cast<sparse_tensor::ReduceOp>(exp(redCustom).op).getIdentity();
}

void CodegenEnv::endCustomReduc() {
  assert(isCustomReduc());
  redCustom = detail::kInvalidId;
}
