include "mlir/IR/EnumAttr.td"

def MatchInterfaceEnum : I32EnumAttr<"MatchInterfaceEnum", "An interface to match",
    [
      I32EnumAttrCase<"LinalgOp", 0>,
      I32EnumAttrCase<"TilingInterface", 1>
    ]>{
  let cppNamespace = "mlir::transform";
}
