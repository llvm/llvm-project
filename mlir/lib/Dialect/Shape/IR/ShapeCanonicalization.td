include "mlir/Dialect/Shape/IR/ShapeOps.td"

def EqualBinaryOperands : Constraint<CPred<"$0 == $1">>;

def AllInputShapesEq : Constraint<CPred< [{
  llvm::all_of($0, [&](mlir::Value val) {
    return $0[0] == val;
  })
}]>>;

// Canonicalization patterns.
def CstrBroadcastableEqOps : Pat<(Shape_CstrBroadcastableOp:$op $lhs, $rhs),
  (Shape_ConstWitnessOp ConstBoolAttrTrue),
  [(EqualBinaryOperands $lhs, $rhs)]>;

def CstrEqEqOps : Pat<(Shape_CstrEqOp:$op $shapes),
  (Shape_ConstWitnessOp ConstBoolAttrTrue),
  [(AllInputShapesEq $shapes)]>;
