include "include/interfaces.td"

def Test_Dialect : Dialect {
  let name = "test";
}

def OpAllEmpty : Op<Test_Dialect, "all_empty">;

def OpAllSingle : Op<Test_Dialect, "all_single"> {
  let arguments = (ins
    I64:$operand,
    I64Attr:$attr,
    Confined<I64Attr, [IntNonNegative]>:$nonNegativeAttr
  );
  let results = (outs I64:$result);
}

def OpAllOptional : Op<Test_Dialect, "all_optional"> {
  let arguments = (ins Optional<I64>:$operand, OptionalAttr<I64Attr>:$attr);
  let results = (outs Optional<I64>:$result);
}

def OpAllVariadic : Op<Test_Dialect, "all_variadic"> {
  let arguments = (ins Variadic<I64>:$operands);
  let results = (outs Variadic<I64>:$results);
}

def OpMultipleSingleResult : Op<Test_Dialect, "multiple_single_result"> {
  let results = (outs I64:$result, I64:$result2);
}
