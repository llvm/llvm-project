// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR2 %s 2>&1 | FileCheck --check-prefix=ERROR2 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR3 %s 2>&1 | FileCheck --check-prefix=ERROR3 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR4 %s 2>&1 | FileCheck --check-prefix=ERROR4 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR5 %s 2>&1 | FileCheck --check-prefix=ERROR5 %s

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test_dialect";
}

#ifdef ERROR1
// ERROR1: error: expected 'ins'
def OpInsMissing : Op<Test_Dialect, "ins_missing"> {
  let builders = [
    OpBuilder<(outs)>
  ];
}
#endif

#ifdef ERROR2
// ERROR2: error: expected an argument with default value after other arguments with default values
def OpDefaultValueNotTrailing : Op<Test_Dialect, "default_value"> {
  let builders = [
    OpBuilder<(ins CArg<"int", "42">, "int")>
  ];
}
#endif

#ifdef ERROR3
// ERROR3: error: expected an argument with default value after other arguments with default values
def OpDefaultValueNotTrailing : Op<Test_Dialect, "default_value"> {
  let builders = [
    OpBuilder<(ins CArg<"int", "42">, CArg<"int">)>
  ];
}
#endif

#ifdef ERROR4
// ERROR4: error: op has two operands with the same name: 'tensor'
def OpWithDuplicatedArgNames : Op<Test_Dialect, "default_value"> {
  let arguments = (ins AnyTensor:$tensor, AnyTensor:$tensor);
}
#endif

#ifdef ERROR5
// ERROR5: error: op has two results with the same name: 'tensor'
def OpWithDuplicatedResultNames : Op<Test_Dialect, "default_value"> {
  let results = (outs AnyTensor:$tensor, AnyTensor:$tensor);
}
#endif
