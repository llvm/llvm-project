include "mlir/Dialect/Shape/IR/ShapeOps.td"

def AllInputShapesEq : Constraint<CPred< [{
  llvm::all_of($0, [&](mlir::Value val) {
    return $0[0] == val;
  })
}]>>;

// Canonicalization patterns.

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

