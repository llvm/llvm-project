include "mlir/Dialect/Shape/IR/ShapeOps.td"

def EqualBinaryOperands : Constraint<CPred<"$0 == $1">>;

// Canonicalization patterns.
def CstrBroadcastableEqOps : Pat<(Shape_CstrBroadcastableOp:$op $lhs, $rhs),
  (Shape_ConstWitnessOp ConstBoolAttrTrue),
  [(EqualBinaryOperands $lhs, $rhs)]>;
