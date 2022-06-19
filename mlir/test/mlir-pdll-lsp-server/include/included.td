include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
}

def OpMulti : Op<Test_Dialect, "multi"> {
  let arguments = (outs I64:$operand, I64:$operand2);
  let results = (outs I64:$result, I64:$result2);
}
