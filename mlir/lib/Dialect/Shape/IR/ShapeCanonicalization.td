include "mlir/Dialect/Shape/IR/ShapeOps.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"

def AllInputShapesEq : Constraint<CPred< [{
  llvm::all_of($0, [&](mlir::Value val) {
    return $0[0] == val;
  })
}]>>;

def HasSingleElement : Constraint<CPred< [{
  $0.size() == 1
}]>>;

// Canonicalization patterns.

def AssumingAllOneOp : Pat<(Shape_AssumingAllOp $args),
                           (replaceWithValue $args),
                           [(HasSingleElement $args)]>;

def CstrBroadcastableEqOps : Pat<(Shape_CstrBroadcastableOp:$op $x, $x),
  (Shape_ConstWitnessOp ConstBoolAttrTrue)>;

def CstrEqEqOps : Pat<(Shape_CstrEqOp:$op $shapes),
  (Shape_ConstWitnessOp ConstBoolAttrTrue),
  [(AllInputShapesEq $shapes)]>;

def IndexToSizeToIndexCanonicalization : Pat<
  (Shape_SizeToIndexOp (Shape_IndexToSizeOp $arg)),
  (replaceWithValue $arg)>;

def SizeToIndexToSizeCanonicalization : Pat<
  (Shape_IndexToSizeOp (Shape_SizeToIndexOp $arg)),
  (replaceWithValue $arg)>;

def TensorCastConstShape : Pat <
  (TensorCastOp (Shape_ConstShapeOp:$c $ty)), (replaceWithValue $c)>;
