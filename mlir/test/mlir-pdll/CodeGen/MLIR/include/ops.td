include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
}

def OpWithResults : Op<Test_Dialect, "with_results"> {
  let results = (outs I64:$result, Variadic<I64>:$var_result);
}
