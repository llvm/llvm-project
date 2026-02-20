
#include "./Ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"

using namespace mlir;
using namespace mlir::arith;

bool valueCmp(Cmp cmp, Value bval, ValueOrInt val) {
  if (auto icast = bval.getDefiningOp<IndexCastOp>()) {
    return valueCmp(cmp, icast.getIn(), val);
  }

  IntegerAttr iattr;
  if (matchPattern(bval, m_Constant(&iattr))) {
    switch (cmp) {
    case Cmp::EQ:
      return val == iattr.getValue();
    case Cmp::LT:
      return val > iattr.getValue();
    case Cmp::LE:
      return val >= iattr.getValue();
    case Cmp::GT:
      return val < iattr.getValue();
    case Cmp::GE:
      return val <= iattr.getValue();
    }
  }

  if (auto baval = dyn_cast<BlockArgument>(bval)) {
    if (affine::AffineForOp afFor =
            dyn_cast<affine::AffineForOp>(baval.getOwner()->getParentOp())) {
      auto forLb = afFor.getLowerBoundMap().getResults()[baval.getArgNumber()];
      auto forUb = afFor.getUpperBoundMap().getResults()[baval.getArgNumber()];
      switch (cmp) {
      // \forall i \in [LB, UB) == k   => LB == k and UB == k+1
      case Cmp::EQ: {
        if (!valueCmp(Cmp::EQ, forLb, afFor.getLowerBoundMap().getNumDims(),
                      afFor.getLowerBoundOperands(), val))
          return false;
        if (!val.isValue) {
          return valueCmp(Cmp::EQ, forUb, afFor.getUpperBoundMap().getNumDims(),
                          afFor.getUpperBoundOperands(), val.iVal + 1);
        }
        return false;
      }
      // \forall i \in [LB, UB) < k   => UB <= k
      case Cmp::LT: {
        return valueCmp(Cmp::LE, forUb, afFor.getUpperBoundMap().getNumDims(),
                        afFor.getUpperBoundOperands(), val);
      }
      // \forall i \in [LB, UB) <= k   => UB-1 <= k  => UB <= k+1
      case Cmp::LE: {
        if (!val.isValue) {
          return valueCmp(Cmp::LE, forUb, afFor.getUpperBoundMap().getNumDims(),
                          afFor.getUpperBoundOperands(), val.iVal + 1);
        }
        return valueCmp(Cmp::LE, forUb, afFor.getUpperBoundMap().getNumDims(),
                        afFor.getUpperBoundOperands(), val);
      }
      // \forall i \in [LB, UB) > k   => LB > k
      case Cmp::GT: {
        return valueCmp(Cmp::GT, forLb, afFor.getLowerBoundMap().getNumDims(),
                        afFor.getLowerBoundOperands(), val);
      }
      // \forall i \in [LB, UB) >= k   => LB >= k
      case Cmp::GE: {
        return valueCmp(Cmp::GE, forLb, afFor.getLowerBoundMap().getNumDims(),
                        afFor.getLowerBoundOperands(), val);
      }
      }
    }
    if (affine::AffineParallelOp afFor = dyn_cast<affine::AffineParallelOp>(
            baval.getOwner()->getParentOp())) {
      switch (cmp) {
      // \forall i \in [max(LB...), min(UB...)) == k   => all(LB == k) and
      // all(UB == k+1)
      case Cmp::EQ: {
        for (auto forLb :
             afFor.getLowerBoundMap(baval.getArgNumber()).getResults())
          if (!valueCmp(Cmp::EQ, forLb, afFor.getLowerBoundsMap().getNumDims(),
                        afFor.getLowerBoundsOperands(), val))
            return false;
        if (!val.isValue) {
          for (auto forUb :
               afFor.getUpperBoundMap(baval.getArgNumber()).getResults())
            if (!valueCmp(Cmp::EQ, forUb,
                          afFor.getUpperBoundsMap().getNumDims(),
                          afFor.getUpperBoundsOperands(), val.iVal + 1))
              return false;
          return true;
        }
        return false;
      }
      // \forall i \in [max(LB...), min(UB...)) < k   => any(UB <= k)
      case Cmp::LT: {
        for (auto forUb :
             afFor.getUpperBoundMap(baval.getArgNumber()).getResults())
          if (valueCmp(Cmp::LE, forUb, afFor.getUpperBoundsMap().getNumDims(),
                       afFor.getUpperBoundsOperands(), val))
            return true;
        return false;
      }
      // \forall i \in [max(LB...), min(UB...)) <= k   => any(UB-1 <= k)  =>
      // any(UB <= k+1)
      case Cmp::LE: {
        if (!val.isValue) {
          for (auto forUb :
               afFor.getUpperBoundMap(baval.getArgNumber()).getResults())
            if (valueCmp(Cmp::LE, forUb, afFor.getUpperBoundsMap().getNumDims(),
                         afFor.getUpperBoundsOperands(), val.iVal + 1))
              return true;
          return false;
        }

        for (auto forUb :
             afFor.getUpperBoundMap(baval.getArgNumber()).getResults())
          if (valueCmp(Cmp::LE, forUb, afFor.getUpperBoundsMap().getNumDims(),
                       afFor.getUpperBoundsOperands(), val))
            return true;
        return false;
      }
      // \forall i \in [max(LB...), min(UB...)) > k   => any(LB > k)
      case Cmp::GT: {
        for (auto forLb :
             afFor.getLowerBoundMap(baval.getArgNumber()).getResults())
          if (valueCmp(Cmp::GT, forLb, afFor.getLowerBoundsMap().getNumDims(),
                       afFor.getLowerBoundsOperands(), val))
            return true;
        return false;
      }
      // \forall i \in [max(LB...), min(UB...)) >= k   => any(LB >= k)
      case Cmp::GE: {
        for (auto forLb :
             afFor.getLowerBoundMap(baval.getArgNumber()).getResults())
          if (valueCmp(Cmp::GE, forLb, afFor.getLowerBoundsMap().getNumDims(),
                       afFor.getLowerBoundsOperands(), val))
            return true;
        return false;
      }
      }
    }

    if (scf::ForOp afFor =
            dyn_cast<scf::ForOp>(baval.getOwner()->getParentOp())) {
      if (baval.getArgNumber() == 0) {
        auto forLb = afFor.getLowerBound();
        auto forUb = afFor.getUpperBound();
        switch (cmp) {
        // \forall i \in [LB, UB) == k   => LB == k and UB == k+1
        case Cmp::EQ: {
          if (!valueCmp(Cmp::EQ, forLb, val))
            return false;
          if (!val.isValue) {
            return valueCmp(Cmp::EQ, forUb, val.iVal + 1);
          }
          return false;
        }
        // \forall i \in [LB, UB) < k   => UB <= k
        case Cmp::LT: {
          return valueCmp(Cmp::LE, forUb, val);
        }
        // \forall i \in [LB, UB) <= k   => UB-1 <= k  => UB <= k+1
        case Cmp::LE: {
          if (!val.isValue) {
            return valueCmp(Cmp::LE, forUb, val.iVal + 1);
          }
          return valueCmp(Cmp::LE, forUb, val);
        }
        // \forall i \in [LB, UB) > k   => LB > k
        case Cmp::GT: {
          return valueCmp(Cmp::GT, forLb, val);
        }
        // \forall i \in [LB, UB) >= k   => LB >= k
        case Cmp::GE: {
          return valueCmp(Cmp::GE, forLb, val);
        }
        }
      }
    }

    if (scf::ParallelOp afFor =
            dyn_cast<scf::ParallelOp>(baval.getOwner()->getParentOp())) {
      auto forLb = afFor.getLowerBound()[baval.getArgNumber()];
      auto forUb = afFor.getUpperBound()[baval.getArgNumber()];
      switch (cmp) {
      // \forall i \in [LB, UB) == k   => LB == k and UB == k+1
      case Cmp::EQ: {
        if (!valueCmp(Cmp::EQ, forLb, val))
          return false;
        if (!val.isValue) {
          return valueCmp(Cmp::EQ, forUb, val.iVal + 1);
        }
        return false;
      }
      // \forall i \in [LB, UB) < k   => UB <= k
      case Cmp::LT: {
        return valueCmp(Cmp::LE, forUb, val);
      }
      // \forall i \in [LB, UB) <= k   => UB-1 <= k  => UB <= k+1
      case Cmp::LE: {
        if (!val.isValue) {
          return valueCmp(Cmp::LE, forUb, val.iVal + 1);
        }
        return valueCmp(Cmp::LE, forUb, val);
      }
      // \forall i \in [LB, UB) > k   => LB > k
      case Cmp::GT: {
        return valueCmp(Cmp::GT, forLb, val);
      }
      // \forall i \in [LB, UB) >= k   => LB >= k
      case Cmp::GE: {
        return valueCmp(Cmp::GE, forLb, val);
      }
      }
    }
  }
  if (val.isValue && val.vVal == bval) {
    switch (cmp) {
    case Cmp::EQ:
      return true;
    case Cmp::LT:
      return false;
    case Cmp::LE:
      return true;
    case Cmp::GT:
      return false;
    case Cmp::GE:
      return true;
    }
  }
  return false;
}

bool valueCmp(Cmp cmp, AffineExpr expr, size_t numDim, ValueRange operands,
              ValueOrInt val) {

  if (auto opd = mlir::dyn_cast<AffineConstantExpr>(expr)) {
    switch (cmp) {
    case Cmp::EQ:
      return val == opd.getValue();
    case Cmp::LT:
      return val > opd.getValue();
    case Cmp::LE:
      return val >= opd.getValue();
    case Cmp::GT:
      return val < opd.getValue();
    case Cmp::GE:
      return val <= opd.getValue();
    }
  }
  if (auto opd = mlir::dyn_cast<AffineDimExpr>(expr)) {
    return valueCmp(cmp, operands[opd.getPosition()], val);
  }
  if (auto opd = mlir::dyn_cast<AffineSymbolExpr>(expr)) {
    return valueCmp(cmp, operands[opd.getPosition() + numDim], val);
  }

  if (auto bop = mlir::dyn_cast<AffineBinaryOpExpr>(expr)) {
    if (bop.getKind() == AffineExprKind::Add) {
      switch (cmp) {
      case Cmp::EQ:
        return (valueCmp(cmp, bop.getLHS(), numDim, operands, val) &&
                valueCmp(cmp, bop.getRHS(), numDim, operands, 0)) ||
               (valueCmp(cmp, bop.getLHS(), numDim, operands, 0) &&
                valueCmp(cmp, bop.getRHS(), numDim, operands, val));
      case Cmp::LT:
        return (valueCmp(cmp, bop.getLHS(), numDim, operands, val) &&
                valueCmp(Cmp::LE, bop.getRHS(), numDim, operands, 0)) ||
               (valueCmp(Cmp::LE, bop.getLHS(), numDim, operands, 0) &&
                valueCmp(cmp, bop.getRHS(), numDim, operands, val)) ||
               (valueCmp(Cmp::LE, bop.getLHS(), numDim, operands, val) &&
                valueCmp(cmp, bop.getRHS(), numDim, operands, 0)) ||
               (valueCmp(cmp, bop.getLHS(), numDim, operands, 0) &&
                valueCmp(Cmp::LE, bop.getRHS(), numDim, operands, val));
      case Cmp::LE:
        return (valueCmp(cmp, bop.getLHS(), numDim, operands, val) &&
                valueCmp(cmp, bop.getRHS(), numDim, operands, 0)) ||
               (valueCmp(cmp, bop.getLHS(), numDim, operands, 0) &&
                valueCmp(cmp, bop.getRHS(), numDim, operands, val));
      case Cmp::GT:
        return (valueCmp(cmp, bop.getLHS(), numDim, operands, val) &&
                valueCmp(Cmp::GE, bop.getRHS(), numDim, operands, 0)) ||
               (valueCmp(Cmp::GE, bop.getLHS(), numDim, operands, 0) &&
                valueCmp(cmp, bop.getRHS(), numDim, operands, val)) ||
               (valueCmp(Cmp::GE, bop.getLHS(), numDim, operands, val) &&
                valueCmp(cmp, bop.getRHS(), numDim, operands, 0)) ||
               (valueCmp(cmp, bop.getLHS(), numDim, operands, 0) &&
                valueCmp(Cmp::GE, bop.getRHS(), numDim, operands, val));
      case Cmp::GE:
        return (valueCmp(cmp, bop.getLHS(), numDim, operands, val) &&
                valueCmp(cmp, bop.getRHS(), numDim, operands, 0)) ||
               (valueCmp(cmp, bop.getLHS(), numDim, operands, 0) &&
                valueCmp(cmp, bop.getRHS(), numDim, operands, val));
      }
    }
    if (bop.getKind() == AffineExprKind::Mul && val == 0) {
      switch (cmp) {
      case Cmp::EQ:
        return (valueCmp(cmp, bop.getLHS(), numDim, operands, val) ||
                valueCmp(cmp, bop.getRHS(), numDim, operands, val));
      case Cmp::LT:
        return (valueCmp(Cmp::LT, bop.getLHS(), numDim, operands, val) &&
                valueCmp(Cmp::GT, bop.getRHS(), numDim, operands, 0)) ||
               (valueCmp(Cmp::GT, bop.getLHS(), numDim, operands, 0) &&
                valueCmp(Cmp::LT, bop.getRHS(), numDim, operands, val));
      case Cmp::LE:
        return valueCmp(Cmp::EQ, bop.getLHS(), numDim, operands, val) ||
               valueCmp(Cmp::EQ, bop.getRHS(), numDim, operands, val) ||
               ((valueCmp(Cmp::GE, bop.getLHS(), numDim, operands, 0) &&
                 valueCmp(Cmp::LE, bop.getRHS(), numDim, operands, val)) ||
                (valueCmp(Cmp::LE, bop.getLHS(), numDim, operands, 0) &&
                 valueCmp(Cmp::GE, bop.getRHS(), numDim, operands, val)));
      case Cmp::GT:
        return (valueCmp(Cmp::LT, bop.getLHS(), numDim, operands, val) &&
                valueCmp(Cmp::LT, bop.getRHS(), numDim, operands, 0)) ||
               (valueCmp(Cmp::GT, bop.getLHS(), numDim, operands, 0) &&
                valueCmp(Cmp::GT, bop.getRHS(), numDim, operands, val));
      case Cmp::GE:
        return valueCmp(Cmp::EQ, bop.getLHS(), numDim, operands, val) ||
               valueCmp(Cmp::EQ, bop.getRHS(), numDim, operands, val) ||
               ((valueCmp(Cmp::GE, bop.getLHS(), numDim, operands, 0) &&
                 valueCmp(Cmp::GE, bop.getRHS(), numDim, operands, val)) ||
                (valueCmp(Cmp::LE, bop.getLHS(), numDim, operands, 0) &&
                 valueCmp(Cmp::LE, bop.getRHS(), numDim, operands, val)));
      }
    }
  }
  return false;
}

bool isReadOnly(Operation *op) {
  bool hasRecursiveEffects = op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
  if (hasRecursiveEffects) {
    for (Region &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &nestedOp : block)
          if (!isReadOnly(&nestedOp))
            return false;
      }
    }
    return true;
  }

  // If the op has memory effects, try to characterize them to see if the op
  // is trivially dead here.
  if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    // Check to see if this op either has no effects, or only allocates/reads
    // memory.
    SmallVector<MemoryEffects::EffectInstance, 1> effects;
    effectInterface.getEffects(effects);
    return llvm::all_of(effects, [](const MemoryEffects::EffectInstance &it) {
      return isa<MemoryEffects::Read>(it.getEffect());
    });
  }
  return false;
}