#include "./Ops.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"
#include <deque>

#define DEBUG_TYPE "affine-cfg"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::affine;

namespace mlir {

#define GEN_PASS_DEF_AFFINECFG
#include "mlir/Conversion/Passes.h.inc"

} // namespace mlir

bool isValidIndex(Value val);

bool isReadOnly(Operation *op);

bool isValidSymbolInt(Value value, bool recur = true);
bool isValidSymbolInt(Operation *defOp, bool recur) {
  Attribute operandCst;
  if (matchPattern(defOp, m_Constant(&operandCst)))
    return true;

  if (recur) {
    if (isa<SelectOp, IndexCastOp, AddIOp, MulIOp, DivSIOp, DivUIOp, RemSIOp,
            RemUIOp, SubIOp, CmpIOp, TruncIOp, ExtUIOp, ExtSIOp>(defOp))
      if (llvm::all_of(defOp->getOperands(),
                       [&](Value v) { return isValidSymbolInt(v, recur); }))
        return true;
    if (auto ifOp = mlir::dyn_cast<scf::IfOp>(defOp)) {
      if (isValidSymbolInt(ifOp.getCondition(), recur)) {
        if (llvm::all_of(
                ifOp.thenBlock()->without_terminator(),
                [&](Operation &o) { return isValidSymbolInt(&o, recur); }) &&
            llvm::all_of(
                ifOp.elseBlock()->without_terminator(),
                [&](Operation &o) { return isValidSymbolInt(&o, recur); }))
          return true;
      }
    }
    if (auto ifOp = dyn_cast<affine::AffineIfOp>(defOp)) {
      if (llvm::all_of(ifOp.getOperands(),
                       [&](Value o) { return isValidSymbolInt(o, recur); }))
        if (llvm::all_of(
                ifOp.getThenBlock()->without_terminator(),
                [&](Operation &o) { return isValidSymbolInt(&o, recur); }) &&
            llvm::all_of(
                ifOp.getElseBlock()->without_terminator(),
                [&](Operation &o) { return isValidSymbolInt(&o, recur); }))
          return true;
    }
  }
  return false;
}

// isValidSymbol, even if not index
bool isValidSymbolInt(Value value, bool recur) {
  // Check that the value is a top level value.
  if (affine::isTopLevelValue(value))
    return true;

  if (auto *defOp = value.getDefiningOp()) {
    if (isValidSymbolInt(defOp, recur))
      return true;
    return affine::isValidSymbol(value, affine::getAffineScope(defOp));
  }

  return false;
}

struct AffineApplyNormalizer {
  AffineApplyNormalizer(AffineMap map, ArrayRef<Value> operands,
                        PatternRewriter &rewriter, DominanceInfo &DI);

  /// Returns the AffineMap resulting from normalization.
  AffineMap getAffineMap() { return affineMap; }

  SmallVector<Value, 8> getOperands() {
    SmallVector<Value, 8> res(reorderedDims);
    res.append(concatenatedSymbols.begin(), concatenatedSymbols.end());
    return res;
  }

private:
  /// Helper function to insert `v` into the coordinate system of the current
  /// AffineApplyNormalizer. Returns the AffineDimExpr with the corresponding
  /// renumbered position.
  AffineDimExpr renumberOneDim(Value v);

  /// Maps of Value to position in `affineMap`.
  DenseMap<Value, unsigned> dimValueToPosition;

  /// Ordered dims and symbols matching positional dims and symbols in
  /// `affineMap`.
  SmallVector<Value, 8> reorderedDims;
  SmallVector<Value, 8> concatenatedSymbols;

  AffineMap affineMap;
};

static bool isAffineForArg(Value val) {
  if (!mlir::isa<BlockArgument>(val))
    return false;
  Operation *parentOp =
      mlir::cast<BlockArgument>(val).getOwner()->getParentOp();
  return (
      isa_and_nonnull<affine::AffineForOp, affine::AffineParallelOp>(parentOp));
}

static bool legalCondition(Value en, bool dim = false) {
  if (en.getDefiningOp<affine::AffineApplyOp>())
    return true;

  if (!dim && !isValidSymbolInt(en, /*recur*/ false)) {
    if (isValidIndex(en) || isValidSymbolInt(en, /*recur*/ true)) {
      return true;
    }
  }

  while (auto ic = en.getDefiningOp<IndexCastOp>())
    en = ic.getIn();

  if ((en.getDefiningOp<AddIOp>() || en.getDefiningOp<SubIOp>() ||
       en.getDefiningOp<MulIOp>() || en.getDefiningOp<RemUIOp>() ||
       en.getDefiningOp<RemSIOp>()) &&
      (en.getDefiningOp()->getOperand(1).getDefiningOp<ConstantIntOp>() ||
       en.getDefiningOp()->getOperand(1).getDefiningOp<ConstantIndexOp>()))
    return true;
  // if (auto IC = dyn_cast_or_null<IndexCastOp>(en.getDefiningOp())) {
  //	if (!outer || legalCondition(IC.getOperand(), false)) return true;
  //}
  if (!dim)
    if (auto BA = dyn_cast<BlockArgument>(en)) {
      if (isa<affine::AffineForOp, affine::AffineParallelOp>(
              BA.getOwner()->getParentOp()))
        return true;
    }
  return false;
}

/// The AffineNormalizer composes AffineApplyOp recursively. Its purpose is to
/// keep a correspondence between the mathematical `map` and the `operands` of
/// a given affine::AffineApplyOp. This correspondence is maintained by
/// iterating over the operands and forming an `auxiliaryMap` that can be
/// composed mathematically with `map`. To keep this correspondence in cases
/// where symbols are produced by affine.apply operations, we perform a local
/// rewrite of symbols as dims.
///
/// Rationale for locally rewriting symbols as dims:
/// ================================================
/// The mathematical composition of AffineMap must always concatenate symbols
/// because it does not have enough information to do otherwise. For example,
/// composing `(d0)[s0] -> (d0 + s0)` with itself must produce
/// `(d0)[s0, s1] -> (d0 + s0 + s1)`.
///
/// The result is only equivalent to `(d0)[s0] -> (d0 + 2 * s0)` when
/// applied to the same mlir::Value for both s0 and s1.
/// As a consequence mathematical composition of AffineMap always concatenates
/// symbols.
///
/// When AffineMaps are used in affine::AffineApplyOp however, they may specify
/// composition via symbols, which is ambiguous mathematically. This corner case
/// is handled by locally rewriting such symbols that come from
/// affine::AffineApplyOp into dims and composing through dims.
/// TODO: Composition via symbols comes at a significant code
/// complexity. Alternatively we should investigate whether we want to
/// explicitly disallow symbols coming from affine.apply and instead force the
/// user to compose symbols beforehand. The annoyances may be small (i.e. 1 or 2
/// extra API calls for such uses, which haven't popped up until now) and the
/// benefit potentially big: simpler and more maintainable code for a
/// non-trivial, recursive, procedure.
AffineApplyNormalizer::AffineApplyNormalizer(AffineMap map,
                                             ArrayRef<Value> operands,
                                             PatternRewriter &rewriter,
                                             DominanceInfo &DI) {
  assert(map.getNumInputs() == operands.size() &&
         "number of operands does not match the number of map inputs");

  LLVM_DEBUG(map.print(llvm::dbgs() << "\nInput map: "));

  SmallVector<Value, 8> addedValues;

  llvm::SmallSet<unsigned, 1> symbolsToPromote;

  unsigned numDims = map.getNumDims();

  SmallVector<AffineExpr, 8> dimReplacements;
  SmallVector<AffineExpr, 8> symReplacements;

  SmallVector<SmallVectorImpl<Value> *> opsTodos;
  auto replaceOp = [&](Operation *oldOp, Operation *newOp) {
    for (auto [oldV, newV] :
         llvm::zip(oldOp->getResults(), newOp->getResults()))
      for (auto *ops : opsTodos)
        for (auto &op : *ops)
          if (op == oldV)
            op = newV;
  };

  std::function<Value(Value, bool)> fix = [&](Value v,
                                              bool index) -> Value /*legal*/ {
    if (isValidSymbolInt(v, /*recur*/ false))
      return v;
    if (index && isAffineForArg(v))
      return v;
    auto *op = v.getDefiningOp();
    if (!op)
      return nullptr;
    if (!op)
      llvm::errs() << v << "\n";
    assert(op);
    if (isa<ConstantOp>(op) || isa<ConstantIndexOp>(op))
      return v;
    if (!isReadOnly(op)) {
      return nullptr;
    }
    Operation *front = nullptr;
    SmallVector<Value> ops;
    opsTodos.push_back(&ops);
    std::function<void(Operation *)> getAllOps = [&](Operation *todo) {
      for (auto v : todo->getOperands()) {
        if (llvm::all_of(op->getRegions(), [&](Region &r) {
              return !r.isAncestor(v.getParentRegion());
            }))
          ops.push_back(v);
      }
      for (auto &r : todo->getRegions()) {
        for (auto &b : r.getBlocks())
          for (auto &o2 : b.without_terminator())
            getAllOps(&o2);
      }
    };
    getAllOps(op);
    for (auto o : ops) {
      Operation *next;
      if (auto *op = o.getDefiningOp()) {
        if (Value nv = fix(o, index)) {
          op = nv.getDefiningOp();
        } else {
          return nullptr;
        }
        next = op->getNextNode();
      } else {
        auto ba = mlir::cast<BlockArgument>(o);
        if (index && isAffineForArg(ba)) {
        } else if (!isValidSymbolInt(o, /*recur*/ false)) {
          return nullptr;
        }
        next = &ba.getOwner()->front();
      }
      if (front == nullptr)
        front = next;
      else if (DI.dominates(front, next))
        front = next;
    }
    opsTodos.pop_back();
    if (!front)
      op->dump();
    assert(front);
    PatternRewriter::InsertionGuard B(rewriter);
    rewriter.setInsertionPoint(front);
    auto *cloned = rewriter.clone(*op);
    replaceOp(op, cloned);
    rewriter.replaceOp(op, cloned->getResults());
    return cloned->getResult(0);
  };
  auto renumberOneSymbol = [&](Value v) {
    for (auto i : llvm::enumerate(addedValues)) {
      if (i.value() == v)
        return getAffineSymbolExpr(i.index(), map.getContext());
    }
    auto expr = getAffineSymbolExpr(addedValues.size(), map.getContext());
    addedValues.push_back(v);
    return expr;
  };

  // 2. Compose affine::AffineApplyOps and dispatch dims or symbols.
  for (unsigned i = 0, e = operands.size(); i < e; ++i) {
    auto t = operands[i];
    auto decast = t;
    while (true) {
      if (auto idx = decast.getDefiningOp<IndexCastOp>()) {
        decast = idx.getIn();
        continue;
      }
      if (auto idx = decast.getDefiningOp<ExtUIOp>()) {
        decast = idx.getIn();
        continue;
      }
      if (auto idx = decast.getDefiningOp<ExtSIOp>()) {
        decast = idx.getIn();
        continue;
      }
      break;
    }

    if (!isValidSymbolInt(t, /*recur*/ false)) {
      t = decast;
    }

    // Only promote one at a time, lest we end up with two dimensions
    // multiplying each other.

    if (((!isValidSymbolInt(t, /*recur*/ false) &&
          (t.getDefiningOp<AddIOp>() || t.getDefiningOp<SubIOp>() ||
           (t.getDefiningOp<MulIOp>() &&
            ((isValidIndex(t.getDefiningOp()->getOperand(0)) &&
              isValidSymbolInt(t.getDefiningOp()->getOperand(1))) ||
             (isValidIndex(t.getDefiningOp()->getOperand(1)) &&
              isValidSymbolInt(t.getDefiningOp()->getOperand(0)))) &&
            !(fix(t.getDefiningOp()->getOperand(0), false) &&
              fix(t.getDefiningOp()->getOperand(1), false))

                ) ||
           ((t.getDefiningOp<DivUIOp>() || t.getDefiningOp<DivSIOp>()) &&
            (isValidIndex(t.getDefiningOp()->getOperand(0)) &&
             isValidSymbolInt(t.getDefiningOp()->getOperand(1))) &&
            (!(fix(t.getDefiningOp()->getOperand(0), false) &&
               fix(t.getDefiningOp()->getOperand(1), false)))) ||
           (t.getDefiningOp<DivSIOp>() &&
            (isValidIndex(t.getDefiningOp()->getOperand(0)) &&
             isValidSymbolInt(t.getDefiningOp()->getOperand(1)))) ||
           (t.getDefiningOp<RemUIOp>() &&
            (isValidIndex(t.getDefiningOp()->getOperand(0)) &&
             isValidSymbolInt(t.getDefiningOp()->getOperand(1)))) ||
           (t.getDefiningOp<RemSIOp>() &&
            (isValidIndex(t.getDefiningOp()->getOperand(0)) &&
             isValidSymbolInt(t.getDefiningOp()->getOperand(1)))) ||
           t.getDefiningOp<ConstantIntOp>() ||
           t.getDefiningOp<ConstantIndexOp>())) ||
         ((decast.getDefiningOp<AddIOp>() || decast.getDefiningOp<SubIOp>() ||
           decast.getDefiningOp<MulIOp>() || decast.getDefiningOp<RemUIOp>() ||
           decast.getDefiningOp<RemSIOp>()) &&
          (decast.getDefiningOp()
               ->getOperand(1)
               .getDefiningOp<ConstantIntOp>() ||
           decast.getDefiningOp()
               ->getOperand(1)
               .getDefiningOp<ConstantIndexOp>())))) {
      t = decast;
      LLVM_DEBUG(llvm::dbgs() << " Replacing: " << t << "\n");

      AffineMap affineApplyMap;
      SmallVector<Value, 8> affineApplyOperands;

      // llvm::dbgs() << "\nop to start: " << t << "\n";

      if (auto op = t.getDefiningOp<AddIOp>()) {
        affineApplyMap =
            AffineMap::get(0, 2,
                           getAffineSymbolExpr(0, op.getContext()) +
                               getAffineSymbolExpr(1, op.getContext()));
        affineApplyOperands.push_back(op.getLhs());
        affineApplyOperands.push_back(op.getRhs());
      } else if (auto op = t.getDefiningOp<SubIOp>()) {
        affineApplyMap =
            AffineMap::get(0, 2,
                           getAffineSymbolExpr(0, op.getContext()) -
                               getAffineSymbolExpr(1, op.getContext()));
        affineApplyOperands.push_back(op.getLhs());
        affineApplyOperands.push_back(op.getRhs());
      } else if (auto op = t.getDefiningOp<MulIOp>()) {
        if (auto ci = op.getRhs().getDefiningOp<ConstantIntOp>()) {
          affineApplyMap = AffineMap::get(
              0, 1, getAffineSymbolExpr(0, op.getContext()) * ci.value());
          affineApplyOperands.push_back(op.getLhs());
        } else if (auto ci = op.getRhs().getDefiningOp<ConstantIndexOp>()) {
          affineApplyMap = AffineMap::get(
              0, 1, getAffineSymbolExpr(0, op.getContext()) * ci.value());
          affineApplyOperands.push_back(op.getLhs());
        } else {
          affineApplyMap =
              AffineMap::get(0, 2,
                             getAffineSymbolExpr(0, op.getContext()) *
                                 getAffineSymbolExpr(1, op.getContext()));
          affineApplyOperands.push_back(op.getLhs());
          affineApplyOperands.push_back(op.getRhs());
        }
      } else if (auto op = t.getDefiningOp<DivSIOp>()) {
        if (auto ci = op.getRhs().getDefiningOp<ConstantIntOp>()) {
          affineApplyMap = AffineMap::get(
              0, 1,
              getAffineSymbolExpr(0, op.getContext()).floorDiv(ci.value()));
          affineApplyOperands.push_back(op.getLhs());
        } else if (auto ci = op.getRhs().getDefiningOp<ConstantIndexOp>()) {
          affineApplyMap = AffineMap::get(
              0, 1,
              getAffineSymbolExpr(0, op.getContext()).floorDiv(ci.value()));
          affineApplyOperands.push_back(op.getLhs());
        } else {
          affineApplyMap = AffineMap::get(
              0, 2,
              getAffineSymbolExpr(0, op.getContext())
                  .floorDiv(getAffineSymbolExpr(1, op.getContext())));
          affineApplyOperands.push_back(op.getLhs());
          affineApplyOperands.push_back(op.getRhs());
        }
      } else if (auto op = t.getDefiningOp<DivUIOp>()) {
        if (auto ci = op.getRhs().getDefiningOp<ConstantIntOp>()) {
          affineApplyMap = AffineMap::get(
              0, 1,
              getAffineSymbolExpr(0, op.getContext()).floorDiv(ci.value()));
          affineApplyOperands.push_back(op.getLhs());
        } else if (auto ci = op.getRhs().getDefiningOp<ConstantIndexOp>()) {
          affineApplyMap = AffineMap::get(
              0, 1,
              getAffineSymbolExpr(0, op.getContext()).floorDiv(ci.value()));
          affineApplyOperands.push_back(op.getLhs());
        } else {
          affineApplyMap = AffineMap::get(
              0, 2,
              getAffineSymbolExpr(0, op.getContext())
                  .floorDiv(getAffineSymbolExpr(1, op.getContext())));
          affineApplyOperands.push_back(op.getLhs());
          affineApplyOperands.push_back(op.getRhs());
        }
      } else if (auto op = t.getDefiningOp<RemSIOp>()) {
        if (auto ci = op.getRhs().getDefiningOp<ConstantIntOp>()) {
          affineApplyMap = AffineMap::get(
              0, 1, getAffineSymbolExpr(0, op.getContext()) % ci.value());
          affineApplyOperands.push_back(op.getLhs());
        } else if (auto ci = op.getRhs().getDefiningOp<ConstantIndexOp>()) {
          affineApplyMap = AffineMap::get(
              0, 1, getAffineSymbolExpr(0, op.getContext()) % ci.value());
          affineApplyOperands.push_back(op.getLhs());
        } else {
          affineApplyMap =
              AffineMap::get(0, 2,
                             getAffineSymbolExpr(0, op.getContext()) %
                                 getAffineSymbolExpr(1, op.getContext()));
          affineApplyOperands.push_back(op.getLhs());
          affineApplyOperands.push_back(op.getRhs());
        }
      } else if (auto op = t.getDefiningOp<RemUIOp>()) {
        if (auto ci = op.getRhs().getDefiningOp<ConstantIntOp>()) {
          affineApplyMap = AffineMap::get(
              0, 1, getAffineSymbolExpr(0, op.getContext()) % ci.value());
          affineApplyOperands.push_back(op.getLhs());
        } else if (auto ci = op.getRhs().getDefiningOp<ConstantIndexOp>()) {
          affineApplyMap = AffineMap::get(
              0, 1, getAffineSymbolExpr(0, op.getContext()) % ci.value());
          affineApplyOperands.push_back(op.getLhs());
        } else {
          affineApplyMap =
              AffineMap::get(0, 2,
                             getAffineSymbolExpr(0, op.getContext()) %
                                 getAffineSymbolExpr(1, op.getContext()));
          affineApplyOperands.push_back(op.getLhs());
          affineApplyOperands.push_back(op.getRhs());
        }
      } else if (auto op = t.getDefiningOp<ConstantIntOp>()) {
        affineApplyMap = AffineMap::get(
            0, 0, getAffineConstantExpr(op.value(), op.getContext()));
      } else if (auto op = t.getDefiningOp<ConstantIndexOp>()) {
        affineApplyMap = AffineMap::get(
            0, 0, getAffineConstantExpr(op.value(), op.getContext()));
      } else {
        llvm_unreachable("");
      }

      SmallVector<AffineExpr, 0> dimRemapping;
      unsigned numOtherSymbols = affineApplyOperands.size();
      SmallVector<AffineExpr, 2> symRemapping(numOtherSymbols);
      for (unsigned idx = 0; idx < numOtherSymbols; ++idx) {
        symRemapping[idx] = renumberOneSymbol(affineApplyOperands[idx]);
      }
      affineApplyMap = affineApplyMap.replaceDimsAndSymbols(
          dimRemapping, symRemapping, reorderedDims.size(), addedValues.size());

      LLVM_DEBUG(affineApplyMap.print(
          llvm::dbgs() << "\nRenumber into current normalizer: "));

      if (i >= numDims)
        symReplacements.push_back(affineApplyMap.getResult(0));
      else
        dimReplacements.push_back(affineApplyMap.getResult(0));

    } else if (isAffineForArg(t)) {
      if (i >= numDims)
        symReplacements.push_back(renumberOneDim(t));
      else
        dimReplacements.push_back(renumberOneDim(t));
    } else if (t.getDefiningOp<affine::AffineApplyOp>()) {
      auto affineApply = t.getDefiningOp<affine::AffineApplyOp>();
      // a. Compose affine.apply operations.
      LLVM_DEBUG(affineApply->print(
          llvm::dbgs() << "\nCompose affine::AffineApplyOp recursively: "));
      AffineMap affineApplyMap = affineApply.getAffineMap();
      SmallVector<Value, 8> affineApplyOperands(
          affineApply.getOperands().begin(), affineApply.getOperands().end());

      SmallVector<AffineExpr, 0> dimRemapping(affineApplyMap.getNumDims());

      for (size_t i = 0; i < affineApplyMap.getNumDims(); ++i) {
        assert(i < affineApplyOperands.size());
        dimRemapping[i] = renumberOneDim(affineApplyOperands[i]);
      }
      unsigned numOtherSymbols = affineApplyOperands.size();
      SmallVector<AffineExpr, 2> symRemapping(numOtherSymbols -
                                              affineApplyMap.getNumDims());
      for (unsigned idx = 0; idx < symRemapping.size(); ++idx) {
        symRemapping[idx] = renumberOneSymbol(
            affineApplyOperands[idx + affineApplyMap.getNumDims()]);
      }
      affineApplyMap = affineApplyMap.replaceDimsAndSymbols(
          dimRemapping, symRemapping, reorderedDims.size(), addedValues.size());

      LLVM_DEBUG(
          affineApplyMap.print(llvm::dbgs() << "\nAffine apply fixup map: "));

      if (i >= numDims)
        symReplacements.push_back(affineApplyMap.getResult(0));
      else
        dimReplacements.push_back(affineApplyMap.getResult(0));
    } else {
      if (!isValidSymbolInt(t, /*recur*/ false)) {
        if (t.getDefiningOp()) {
          if ((t = fix(t, false))) {
            assert(isValidSymbolInt(t, /*recur*/ false));
          } else
            llvm_unreachable("cannot move");
        } else
          llvm_unreachable("cannot move2");
      }
      if (i < numDims) {
        // b. The mathematical composition of AffineMap composes dims.
        dimReplacements.push_back(renumberOneDim(t));
      } else {
        // c. The mathematical composition of AffineMap concatenates symbols.
        //    Note that the map composition will put symbols already present
        //    in the map before any symbols coming from the auxiliary map, so
        //    we insert them before any symbols that are due to renumbering,
        //    and after the proper symbols we have seen already.
        symReplacements.push_back(renumberOneSymbol(t));
      }
    }
  }
  for (auto v : addedValues)
    concatenatedSymbols.push_back(v);

  // Create the new map by replacing each symbol at pos by the next new dim.
  unsigned numNewDims = reorderedDims.size();
  unsigned numNewSymbols = addedValues.size();
  assert(dimReplacements.size() == map.getNumDims());
  assert(symReplacements.size() == map.getNumSymbols());
  auto auxillaryMap = map.replaceDimsAndSymbols(
      dimReplacements, symReplacements, numNewDims, numNewSymbols);
  LLVM_DEBUG(auxillaryMap.print(llvm::dbgs() << "\nRewritten map: "));

  affineMap = auxillaryMap; // simplifyAffineMap(auxillaryMap);

  LLVM_DEBUG(affineMap.print(llvm::dbgs() << "\nSimplified result: "));
  LLVM_DEBUG(llvm::dbgs() << "\n");
}

AffineDimExpr AffineApplyNormalizer::renumberOneDim(Value v) {
  DenseMap<Value, unsigned>::iterator iterPos;
  bool inserted = false;
  std::tie(iterPos, inserted) =
      dimValueToPosition.insert(std::make_pair(v, dimValueToPosition.size()));
  if (inserted) {
    reorderedDims.push_back(v);
  }
  return mlir::cast<AffineDimExpr>(
      getAffineDimExpr(iterPos->second, v.getContext()));
}

static void composeAffineMapAndOperands(AffineMap *map,
                                        SmallVectorImpl<Value> *operands,
                                        PatternRewriter &rewriter,
                                        DominanceInfo &di) {
  AffineApplyNormalizer normalizer(*map, *operands, rewriter, di);
  auto normalizedMap = normalizer.getAffineMap();
  auto normalizedOperands = normalizer.getOperands();
  affine::canonicalizeMapAndOperands(&normalizedMap, &normalizedOperands);
  *map = normalizedMap;
  *operands = normalizedOperands;
  assert(*map);
}

bool need(AffineMap *map, SmallVectorImpl<Value> *operands) {
  assert(map->getNumInputs() == operands->size());
  for (size_t i = 0; i < map->getNumInputs(); ++i) {
    auto v = (*operands)[i];
    if (legalCondition(v, i < map->getNumDims()))
      return true;
  }
  return false;
}
bool need(IntegerSet *map, SmallVectorImpl<Value> *operands) {
  for (size_t i = 0; i < map->getNumInputs(); ++i) {
    auto v = (*operands)[i];
    if (legalCondition(v, i < map->getNumDims()))
      return true;
  }
  return false;
}

void fully2ComposeAffineMapAndOperands(PatternRewriter &builder, AffineMap *map,
                                       SmallVectorImpl<Value> *operands,
                                       DominanceInfo &di) {
  IRMapping indexMap;
  for (auto op : *operands) {
    SmallVector<IndexCastOp> attempt;
    auto idx0 = op.getDefiningOp<IndexCastOp>();
    attempt.push_back(idx0);
    if (!idx0)
      continue;

    for (auto &u : idx0.getIn().getUses()) {
      if (auto idx = dyn_cast<IndexCastOp>(u.getOwner()))
        if (di.dominates((Operation *)idx, &*builder.getInsertionPoint()))
          attempt.push_back(idx);
    }

    for (auto idx : attempt) {
      if (affine::isValidSymbol(idx)) {
        indexMap.map(idx.getIn(), idx);
        break;
      }
    }
  }
  assert(map->getNumInputs() == operands->size());
  while (need(map, operands)) {
    composeAffineMapAndOperands(map, operands, builder, di);
    assert(map->getNumInputs() == operands->size());
  }
  *map = simplifyAffineMap(*map);
  for (auto &op : *operands) {
    if (!op.getType().isIndex()) {
      Operation *toInsert;
      if (auto *o = op.getDefiningOp())
        toInsert = o->getNextNode();
      else {
        auto ba = mlir::cast<BlockArgument>(op);
        toInsert = &ba.getOwner()->front();
      }

      if (auto v = indexMap.lookupOrNull(op))
        op = v;
      else {
        PatternRewriter::InsertionGuard b(builder);
        builder.setInsertionPoint(toInsert);
        op = builder.create<IndexCastOp>(op.getLoc(), builder.getIndexType(),
                                         op);
      }
    }
  }
}

void fully2ComposeIntegerSetAndOperands(PatternRewriter &builder,
                                        IntegerSet *set,
                                        SmallVectorImpl<Value> *operands,
                                        DominanceInfo &DI) {
  IRMapping indexMap;
  for (auto op : *operands) {
    SmallVector<IndexCastOp> attempt;
    auto idx0 = op.getDefiningOp<IndexCastOp>();
    attempt.push_back(idx0);
    if (!idx0)
      continue;

    for (auto &u : idx0.getIn().getUses()) {
      if (auto idx = dyn_cast<IndexCastOp>(u.getOwner()))
        if (DI.dominates((Operation *)idx, &*builder.getInsertionPoint()))
          attempt.push_back(idx);
    }

    for (auto idx : attempt) {
      if (affine::isValidSymbol(idx)) {
        indexMap.map(idx.getIn(), idx);
        break;
      }
    }
  }
  auto map = AffineMap::get(set->getNumDims(), set->getNumSymbols(),
                            set->getConstraints(), set->getContext());
  while (need(&map, operands)) {
    composeAffineMapAndOperands(&map, operands, builder, DI);
  }
  map = simplifyAffineMap(map);
  *set = IntegerSet::get(map.getNumDims(), map.getNumSymbols(),
                         map.getResults(), set->getEqFlags());
  for (auto &op : *operands) {
    if (!op.getType().isIndex()) {
      Operation *toInsert;
      if (auto *o = op.getDefiningOp())
        toInsert = o->getNextNode();
      else {
        auto ba = mlir::cast<BlockArgument>(op);
        toInsert = &ba.getOwner()->front();
      }

      if (auto v = indexMap.lookupOrNull(op))
        op = v;
      else {
        PatternRewriter::InsertionGuard b(builder);
        builder.setInsertionPoint(toInsert);
        op = builder.create<IndexCastOp>(op.getLoc(), builder.getIndexType(),
                                         op);
      }
    }
  }
}

namespace {
struct AffineCFG : public impl::AffineCFGBase<AffineCFG> {
  void runOnOperation() override;
};
} // namespace

struct IndexCastMovement : public OpRewritePattern<IndexCastOp> {
  using OpRewritePattern<IndexCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IndexCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    mlir::Value val = op.getOperand();
    if (auto bop = dyn_cast<mlir::BlockArgument>(val)) {
      if (op.getOperation()->getBlock() != bop.getOwner()) {
        op.getOperation()->moveBefore(bop.getOwner(), bop.getOwner()->begin());
        return success();
      }
      return failure();
    }

    if (val.getDefiningOp()) {
      if (op.getOperation()->getBlock() != val.getDefiningOp()->getBlock()) {
        auto it = val.getDefiningOp()->getIterator();
        op.getOperation()->moveAfter(val.getDefiningOp()->getBlock(), it);
      }
      return failure();
    }
    return failure();
  }
};

struct CanonicalizeAffineApply
    : public OpRewritePattern<affine::AffineApplyOp> {
  using OpRewritePattern<affine::AffineApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineApplyOp affineOp,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value, 4> mapOperands(affineOp.getMapOperands());
    auto map = affineOp.getMap();
    auto prevMap = map;

    auto *scope = affine::getAffineScope(affineOp)->getParentOp();
    DominanceInfo di(scope);

    fully2ComposeAffineMapAndOperands(rewriter, &map, &mapOperands, di);
    affine::canonicalizeMapAndOperands(&map, &mapOperands);
    map = removeDuplicateExprs(map);

    if (map == prevMap)
      return failure();

    rewriter.replaceOpWithNewOp<affine::AffineApplyOp>(affineOp, map,
                                                       mapOperands);
    return success();
  }
};

struct CanonicalizeIndexCast : public OpRewritePattern<IndexCastOp> {
  using OpRewritePattern<IndexCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IndexCastOp indexcastOp,
                                PatternRewriter &rewriter) const override {

    // Fold IndexCast(IndexCast(x)) -> x
    auto cast = indexcastOp.getOperand().getDefiningOp<IndexCastOp>();
    if (cast && cast.getOperand().getType() == indexcastOp.getType()) {
      mlir::Value vals[] = {cast.getOperand()};
      rewriter.replaceOp(indexcastOp, vals);
      return success();
    }

    // Fold IndexCast(constant) -> constant
    // A little hack because we go through int.  Otherwise, the size
    // of the constant might need to change.
    if (auto cst = indexcastOp.getOperand().getDefiningOp<ConstantIntOp>()) {
      rewriter.replaceOpWithNewOp<ConstantIndexOp>(indexcastOp, cst.value());
      return success();
    }
    return failure();
  }
};

bool isValidIndex(Value val) {
  if (isValidSymbolInt(val))
    return true;

  if (auto cast = val.getDefiningOp<IndexCastOp>())
    return isValidIndex(cast.getOperand());

  if (auto cast = val.getDefiningOp<ExtSIOp>())
    return isValidIndex(cast.getOperand());

  if (auto cast = val.getDefiningOp<ExtUIOp>())
    return isValidIndex(cast.getOperand());

  if (auto bop = val.getDefiningOp<AddIOp>())
    return isValidIndex(bop.getOperand(0)) && isValidIndex(bop.getOperand(1));

  if (auto bop = val.getDefiningOp<MulIOp>())
    return (isValidIndex(bop.getOperand(0)) &&
            isValidSymbolInt(bop.getOperand(1))) ||
           (isValidIndex(bop.getOperand(1)) &&
            isValidSymbolInt(bop.getOperand(0)));

  if (auto bop = val.getDefiningOp<DivSIOp>())
    return (isValidIndex(bop.getOperand(0)) &&
            isValidSymbolInt(bop.getOperand(1)));

  if (auto bop = val.getDefiningOp<DivUIOp>())
    return (isValidIndex(bop.getOperand(0)) &&
            isValidSymbolInt(bop.getOperand(1)));

  if (auto bop = val.getDefiningOp<RemSIOp>()) {
    return (isValidIndex(bop.getOperand(0)) &&
            bop.getOperand(1).getDefiningOp<arith::ConstantOp>());
  }

  if (auto bop = val.getDefiningOp<RemUIOp>())
    return (isValidIndex(bop.getOperand(0)) &&
            bop.getOperand(1).getDefiningOp<arith::ConstantOp>());

  if (auto bop = val.getDefiningOp<SubIOp>())
    return isValidIndex(bop.getOperand(0)) && isValidIndex(bop.getOperand(1));

  if (val.getDefiningOp<ConstantIndexOp>())
    return true;

  if (val.getDefiningOp<ConstantIntOp>())
    return true;

  if (auto ba = dyn_cast<BlockArgument>(val)) {
    auto *owner = ba.getOwner();
    assert(owner);

    auto *parentOp = owner->getParentOp();
    if (!parentOp) {
      owner->dump();
      llvm::errs() << " ba: " << ba << "\n";
    }
    assert(parentOp);
    if (isa<FunctionOpInterface>(parentOp))
      return true;
    if (auto af = dyn_cast<affine::AffineForOp>(parentOp))
      return af.getInductionVar() == ba;

    // TODO ensure not a reduced var
    if (isa<affine::AffineParallelOp>(parentOp))
      return true;

    if (isa<FunctionOpInterface>(parentOp))
      return true;
  }

  LLVM_DEBUG(llvm::dbgs() << "illegal isValidIndex: " << val << "\n");
  return false;
}

// returns legality
bool handleMinMax(Value start, SmallVectorImpl<Value> &out, bool &min,
                  bool &max) {

  SmallVector<Value> todo = {start};
  while (todo.size()) {
    auto cur = todo.back();
    todo.pop_back();
    if (isValidIndex(cur)) {
      out.push_back(cur);
      continue;
    }
    if (auto selOp = cur.getDefiningOp<SelectOp>()) {
      // UB only has min of operands
      if (auto cmp = selOp.getCondition().getDefiningOp<CmpIOp>()) {
        if (cmp.getLhs() == selOp.getTrueValue() &&
            cmp.getRhs() == selOp.getFalseValue()) {
          todo.push_back(cmp.getLhs());
          todo.push_back(cmp.getRhs());
          if (cmp.getPredicate() == CmpIPredicate::sle ||
              cmp.getPredicate() == CmpIPredicate::slt) {
            min = true;
            continue;
          }
          if (cmp.getPredicate() == CmpIPredicate::sge ||
              cmp.getPredicate() == CmpIPredicate::sgt) {
            max = true;
            continue;
          }
        }
      }
    }
    return false;
  }
  return !(min && max);
}

bool handle(PatternRewriter &b, CmpIOp cmpi, SmallVectorImpl<AffineExpr> &exprs,
            SmallVectorImpl<bool> &eqflags, SmallVectorImpl<Value> &applies) {
  SmallVector<Value> lhs;
  bool lhsMin = false;
  bool lhsMax = false;
  if (!handleMinMax(cmpi.getLhs(), lhs, lhsMin, lhsMax)) {
    LLVM_DEBUG(llvm::dbgs()
               << "illegal lhs: " << cmpi.getLhs() << " - " << cmpi << "\n");
    return false;
  }
  assert(lhs.size());
  SmallVector<Value> rhs;
  bool rhsMin = false;
  bool rhsMax = false;
  if (!handleMinMax(cmpi.getRhs(), rhs, rhsMin, rhsMax)) {
    LLVM_DEBUG(llvm::dbgs()
               << "illegal rhs: " << cmpi.getRhs() << " - " << cmpi << "\n");
    return false;
  }
  assert(rhs.size());
  for (auto &lhspack : lhs)
    if (!mlir::isa<IndexType>(lhspack.getType())) {
      lhspack = b.create<arith::IndexCastOp>(
          cmpi.getLoc(), IndexType::get(cmpi.getContext()), lhspack);
    }

  for (auto &rhspack : rhs)
    if (!mlir::isa<IndexType>(rhspack.getType())) {
      rhspack = b.create<arith::IndexCastOp>(
          cmpi.getLoc(), IndexType::get(cmpi.getContext()), rhspack);
    }

  switch (cmpi.getPredicate()) {
  case CmpIPredicate::eq: {
    if (lhsMin || lhsMax || rhsMin || rhsMax)
      return false;
    eqflags.push_back(true);

    applies.push_back(lhs[0]);
    applies.push_back(rhs[0]);
    AffineExpr dims[2] = {b.getAffineSymbolExpr(2 * exprs.size() + 0),
                          b.getAffineSymbolExpr(2 * exprs.size() + 1)};
    exprs.push_back(dims[0] - dims[1]);
  } break;

  case CmpIPredicate::ugt:
  case CmpIPredicate::uge:
    for (auto lhspack : lhs)
      if (!valueCmp(Cmp::GE, lhspack, 0)) {
        LLVM_DEBUG(llvm::dbgs() << "illegal greater lhs icmp: " << cmpi << " - "
                                << lhspack << "\n");
        return false;
      }
    for (auto rhspack : rhs)
      if (!valueCmp(Cmp::GE, rhspack, 0)) {
        LLVM_DEBUG(llvm::dbgs() << "illegal greater rhs icmp: " << cmpi << " - "
                                << rhspack << "\n");
        return false;
      }
    LLVM_FALLTHROUGH;
  case CmpIPredicate::sge:
  case CmpIPredicate::sgt: {
    // if lhs >=? rhs
    // if lhs is a min(a, b) both must be true and this is fine
    // if lhs is a max(a, b) either may be true, and sets require and
    // similarly if rhs is a max(), both must be true;
    if (lhsMax || rhsMin)
      return false;
    for (auto lhspack : lhs)
      for (auto rhspack : rhs) {
        eqflags.push_back(false);
        applies.push_back(lhspack);
        applies.push_back(rhspack);
        AffineExpr dims[2] = {b.getAffineSymbolExpr(2 * exprs.size() + 0),
                              b.getAffineSymbolExpr(2 * exprs.size() + 1)};
        auto expr = dims[0] - dims[1];
        if (cmpi.getPredicate() == CmpIPredicate::sgt ||
            cmpi.getPredicate() == CmpIPredicate::ugt)
          expr = expr - 1;
        exprs.push_back(expr);
      }
  } break;

  case CmpIPredicate::ult:
  case CmpIPredicate::ule:
    for (auto lhspack : lhs)
      if (!valueCmp(Cmp::GE, lhspack, 0)) {
        LLVM_DEBUG(llvm::dbgs() << "illegal less lhs icmp: " << cmpi << " - "
                                << lhspack << "\n");
        return false;
      }
    for (auto rhspack : rhs)
      if (!valueCmp(Cmp::GE, rhspack, 0)) {
        LLVM_DEBUG(llvm::dbgs() << "illegal less rhs icmp: " << cmpi << " - "
                                << rhspack << "\n");
        return false;
      }
    LLVM_FALLTHROUGH;
  case CmpIPredicate::slt:
  case CmpIPredicate::sle: {
    if (lhsMin || rhsMax)
      return false;
    for (auto lhspack : lhs)
      for (auto rhspack : rhs) {
        eqflags.push_back(false);
        applies.push_back(lhspack);
        applies.push_back(rhspack);
        AffineExpr dims[2] = {b.getAffineSymbolExpr(2 * exprs.size() + 0),
                              b.getAffineSymbolExpr(2 * exprs.size() + 1)};
        auto expr = dims[1] - dims[0];
        if (cmpi.getPredicate() == CmpIPredicate::slt ||
            cmpi.getPredicate() == CmpIPredicate::ult)
          expr = expr - 1;
        exprs.push_back(expr);
      }
  } break;

  case CmpIPredicate::ne:
    LLVM_DEBUG(llvm::dbgs() << "illegal icmp: " << cmpi << "\n");
    return false;
  }
  return true;
}

struct MoveLoadToAffine : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp load,
                                PatternRewriter &rewriter) const override {
    if (!llvm::all_of(load.getIndices(), isValidIndex))
      return failure();

    auto memrefType = mlir::cast<MemRefType>(load.getMemRef().getType());
    int64_t rank = memrefType.getRank();

    // Create identity map for memrefs with at least one dimension or () -> ()
    // for zero-dimensional memrefs.
    SmallVector<AffineExpr, 4> dimExprs;
    dimExprs.reserve(rank);
    for (unsigned i = 0; i < rank; ++i)
      dimExprs.push_back(rewriter.getAffineSymbolExpr(i));
    auto map = AffineMap::get(/*dimCount=*/0, /*symbolCount=*/rank, dimExprs,
                              rewriter.getContext());

    SmallVector<Value, 4> operands = load.getIndices();

    if (map.getNumInputs() != operands.size()) {
      // load->getParentOfType<FuncOp>().dump();
      llvm::errs() << " load: " << load << "\n";
    }
    auto *scope = affine::getAffineScope(load)->getParentOp();
    DominanceInfo di(scope);
    assert(map.getNumInputs() == operands.size());
    fully2ComposeAffineMapAndOperands(rewriter, &map, &operands, di);
    assert(map.getNumInputs() == operands.size());
    affine::canonicalizeMapAndOperands(&map, &operands);
    assert(map.getNumInputs() == operands.size());

    affine::AffineLoadOp affineLoad = affine::AffineLoadOp::create(
        rewriter, load.getLoc(), load.getMemRef(), map, operands);
    load.getResult().replaceAllUsesWith(affineLoad.getResult());
    rewriter.eraseOp(load);
    return success();
  }
};

struct MoveStoreToAffine : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern<memref::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp store,
                                PatternRewriter &rewriter) const override {
    if (!llvm::all_of(store.getIndices(), isValidIndex))
      return failure();

    auto memrefType = mlir::cast<MemRefType>(store.getMemRef().getType());
    int64_t rank = memrefType.getRank();

    // Create identity map for memrefs with at least one dimension or () -> ()
    // for zero-dimensional memrefs.
    SmallVector<AffineExpr, 4> dimExprs;
    dimExprs.reserve(rank);
    for (unsigned i = 0; i < rank; ++i)
      dimExprs.push_back(rewriter.getAffineSymbolExpr(i));
    auto map = AffineMap::get(/*dimCount=*/0, /*symbolCount=*/rank, dimExprs,
                              rewriter.getContext());
    SmallVector<Value, 4> operands = store.getIndices();

    auto *scope = affine::getAffineScope(store)->getParentOp();
    DominanceInfo di(scope);

    fully2ComposeAffineMapAndOperands(rewriter, &map, &operands, di);
    affine::canonicalizeMapAndOperands(&map, &operands);

    affine::AffineStoreOp::create(rewriter, store.getLoc(),
                                  store.getValueToStore(), store.getMemRef(),
                                  map, operands);
    rewriter.eraseOp(store);
    return success();
  }
};

static bool areChanged(SmallVectorImpl<Value> &afterOperands,
                       SmallVectorImpl<Value> &beforeOperands) {
  if (afterOperands.size() != beforeOperands.size())
    return true;
  if (!std::equal(afterOperands.begin(), afterOperands.end(),
                  beforeOperands.begin()))
    return true;
  return false;
}

template <typename T>
struct AffineFixup : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  /// Replace the affine op with another instance of it with the supplied
  /// map and mapOperands.
  void replaceAffineOp(PatternRewriter &rewriter, T affineOp, AffineMap map,
                       ArrayRef<Value> mapOperands) const;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    auto map = op.getAffineMap();
    SmallVector<Value, 4> operands = op.getMapOperands();

    auto prevMap = map;
    auto prevOperands = operands;

    auto *scope = affine::getAffineScope(op)->getParentOp();
    DominanceInfo di(scope);

    assert(map.getNumInputs() == operands.size());
    fully2ComposeAffineMapAndOperands(rewriter, &map, &operands, di);
    assert(map.getNumInputs() == operands.size());
    affine::canonicalizeMapAndOperands(&map, &operands);
    assert(map.getNumInputs() == operands.size());

    if (map == prevMap && !areChanged(operands, prevOperands))
      return failure();

    replaceAffineOp(rewriter, op, map, operands);
    return success();
  }
};

// Specialize the template to account for the different build signatures for
// affine load, store, and apply ops.
template <>
void AffineFixup<affine::AffineLoadOp>::replaceAffineOp(
    PatternRewriter &rewriter, affine::AffineLoadOp load, AffineMap map,
    ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(load, load.getMemRef(), map,
                                                    mapOperands);
}
template <>
void AffineFixup<affine::AffinePrefetchOp>::replaceAffineOp(
    PatternRewriter &rewriter, affine::AffinePrefetchOp prefetch, AffineMap map,
    ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<affine::AffinePrefetchOp>(
      prefetch, prefetch.getMemref(), map, mapOperands,
      prefetch.getLocalityHint(), prefetch.getIsWrite(),
      prefetch.getIsDataCache());
}
template <>
void AffineFixup<affine::AffineStoreOp>::replaceAffineOp(
    PatternRewriter &rewriter, affine::AffineStoreOp store, AffineMap map,
    ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
      store, store.getValueToStore(), store.getMemRef(), map, mapOperands);
}
template <>
void AffineFixup<affine::AffineVectorLoadOp>::replaceAffineOp(
    PatternRewriter &rewriter, affine::AffineVectorLoadOp vectorload,
    AffineMap map, ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<affine::AffineVectorLoadOp>(
      vectorload, vectorload.getVectorType(), vectorload.getMemRef(), map,
      mapOperands);
}
template <>
void AffineFixup<affine::AffineVectorStoreOp>::replaceAffineOp(
    PatternRewriter &rewriter, affine::AffineVectorStoreOp vectorstore,
    AffineMap map, ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<affine::AffineVectorStoreOp>(
      vectorstore, vectorstore.getValueToStore(), vectorstore.getMemRef(), map,
      mapOperands);
}

// Generic version for ops that don't have extra operands.
template <typename AffineOpTy>
void AffineFixup<AffineOpTy>::replaceAffineOp(
    PatternRewriter &rewriter, AffineOpTy op, AffineMap map,
    ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<AffineOpTy>(op, map, mapOperands);
}

struct CanonicalieForBounds : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp forOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value, 4> lbOperands(forOp.getLowerBoundOperands());
    SmallVector<Value, 4> ubOperands(forOp.getUpperBoundOperands());
    SmallVector<Value, 4> origLbOperands(forOp.getLowerBoundOperands());
    SmallVector<Value, 4> origUbOperands(forOp.getUpperBoundOperands());

    auto lbMap = forOp.getLowerBoundMap();
    auto ubMap = forOp.getUpperBoundMap();
    auto prevLbMap = lbMap;
    auto prevUbMap = ubMap;

    // llvm::errs() << "*********\n";
    // ubMap.dump();

    auto *scope = affine::getAffineScope(forOp)->getParentOp();
    DominanceInfo di(scope);

    fully2ComposeAffineMapAndOperands(rewriter, &lbMap, &lbOperands, di);
    affine::canonicalizeMapAndOperands(&lbMap, &lbOperands);
    lbMap = removeDuplicateExprs(lbMap);

    fully2ComposeAffineMapAndOperands(rewriter, &ubMap, &ubOperands, di);
    affine::canonicalizeMapAndOperands(&ubMap, &ubOperands);
    ubMap = removeDuplicateExprs(ubMap);

    // ubMap.dump();
    // forOp.dump();

    // Any canonicalization change in map or operands always leads to updated
    // map(s).
    if ((lbMap == prevLbMap && ubMap == prevUbMap) &&
        (!areChanged(lbOperands, origLbOperands)) &&
        (!areChanged(ubOperands, origUbOperands)))
      return failure();

    // llvm::errs() << "oldParent:" << *forOp.getParentOp() << "\n";
    // llvm::errs() << "oldfor:" << forOp << "\n";

    if ((lbMap != prevLbMap) || areChanged(lbOperands, origLbOperands))
      forOp.setLowerBound(lbOperands, lbMap);
    if ((ubMap != prevUbMap) || areChanged(ubOperands, origUbOperands))
      forOp.setUpperBound(ubOperands, ubMap);

    // llvm::errs() << "newfor:" << forOp << "\n";
    return success();
  }
};

struct CanonicalizIfBounds : public OpRewritePattern<affine::AffineIfOp> {
  using OpRewritePattern<affine::AffineIfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineIfOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value, 4> operands(op.getOperands());
    SmallVector<Value, 4> origOperands(operands);

    auto map = op.getIntegerSet();
    auto prevMap = map;

    // llvm::errs() << "*********\n";
    // ubMap.dump();

    auto *scope = affine::getAffineScope(op)->getParentOp();
    DominanceInfo DI(scope);

    fully2ComposeIntegerSetAndOperands(rewriter, &map, &operands, DI);
    affine::canonicalizeSetAndOperands(&map, &operands);

    // map(s).
    if (map == prevMap && !areChanged(operands, origOperands))
      return failure();

    op.setConditional(map, operands);

    return success();
  }
};

struct MoveIfToAffine : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (!ifOp->getParentOfType<affine::AffineForOp>() &&
        !ifOp->getParentOfType<affine::AffineParallelOp>())
      return failure();

    std::vector<mlir::Type> types;
    for (auto v : ifOp.getResults()) {
      types.push_back(v.getType());
    }

    SmallVector<AffineExpr, 2> exprs;
    SmallVector<bool, 2> eqflags;
    SmallVector<Value, 4> applies;

    std::deque<Value> todo = {ifOp.getCondition()};
    while (todo.size()) {
      auto cur = todo.front();
      todo.pop_front();
      if (auto cmpi = cur.getDefiningOp<CmpIOp>()) {
        if (!handle(rewriter, cmpi, exprs, eqflags, applies)) {
          return failure();
        }
        continue;
      }
      if (auto andi = cur.getDefiningOp<AndIOp>()) {
        todo.push_back(andi.getOperand(0));
        todo.push_back(andi.getOperand(1));
        continue;
      }
      return failure();
    }

    auto *scope = affine::getAffineScope(ifOp)->getParentOp();
    DominanceInfo di(scope);

    auto iset =
        IntegerSet::get(/*dim*/ 0, /*symbol*/ 2 * exprs.size(), exprs, eqflags);
    fully2ComposeIntegerSetAndOperands(rewriter, &iset, &applies, di);
    affine::canonicalizeSetAndOperands(&iset, &applies);
    affine::AffineIfOp affineIfOp = affine::AffineIfOp::create(
        rewriter, ifOp.getLoc(), types, iset, applies,
        /*elseBlock=*/true);

    rewriter.setInsertionPoint(ifOp.thenYield());
    rewriter.replaceOpWithNewOp<affine::AffineYieldOp>(
        ifOp.thenYield(), ifOp.thenYield().getOperands());

    rewriter.eraseBlock(affineIfOp.getThenBlock());
    rewriter.eraseBlock(affineIfOp.getElseBlock());
    if (ifOp.getElseRegion().getBlocks().size()) {
      rewriter.setInsertionPoint(ifOp.elseYield());
      rewriter.replaceOpWithNewOp<affine::AffineYieldOp>(
          ifOp.elseYield(), ifOp.elseYield().getOperands());
    }

    rewriter.inlineRegionBefore(ifOp.getThenRegion(),
                                affineIfOp.getThenRegion(),
                                affineIfOp.getThenRegion().begin());
    rewriter.inlineRegionBefore(ifOp.getElseRegion(),
                                affineIfOp.getElseRegion(),
                                affineIfOp.getElseRegion().begin());

    rewriter.replaceOp(ifOp, affineIfOp.getResults());
    return success();
  }
};

void AffineCFG::runOnOperation() {
  mlir::RewritePatternSet rpl(getOperation()->getContext());
  rpl.add</*SimplfyIntegerCastMath, */ CanonicalizeAffineApply,
          CanonicalizeIndexCast,
          /* IndexCastMovement,*/ AffineFixup<affine::AffineLoadOp>,
          AffineFixup<affine::AffineStoreOp>, CanonicalizIfBounds,
          MoveStoreToAffine, MoveIfToAffine, MoveLoadToAffine,
          CanonicalieForBounds>(getOperation()->getContext());
  GreedyRewriteConfig config;
  (void)applyPatternsGreedily(getOperation(), std::move(rpl), config);
}

std::unique_ptr<Pass> mlir::replaceAffineCFGPass() {
  return std::make_unique<AffineCFG>();
}
