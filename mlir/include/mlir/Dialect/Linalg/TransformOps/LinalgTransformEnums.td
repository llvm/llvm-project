include "mlir/IR/EnumAttr.td"

def MatchInterfaceEnum : I32EnumAttr<"MatchInterfaceEnum", "An interface to match",
    [
      I32EnumAttrCase<"LinalgOp", 0>,
      I32EnumAttrCase<"TilingInterface", 1>,
      I32EnumAttrCase<"LoopLikeInterface", 2>,
    ]>{
  let cppNamespace = "mlir::transform";
}

def TransposeMatmulInput : I32EnumAttr<"TransposeMatmulInput",
    "Input to transpose when converting matmul ops to transposed variants",
    [
      I32EnumAttrCase<"lhs", 0>,
      I32EnumAttrCase<"rhs", 1>,
    ]>{
  let cppNamespace = "mlir::transform";
}
