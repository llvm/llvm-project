// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR2 %s 2>&1 | FileCheck --check-prefix=ERROR2 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR3 %s 2>&1 | FileCheck --check-prefix=ERROR3 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR4 %s 2>&1 | FileCheck --check-prefix=ERROR4 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR5 %s 2>&1 | FileCheck --check-prefix=ERROR5 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR6 %s 2>&1 | FileCheck --check-prefix=ERROR6 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR7 %s 2>&1 | FileCheck --check-prefix=ERROR7 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR8 %s 2>&1 | FileCheck --check-prefix=ERROR8 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR9 %s 2>&1 | FileCheck --check-prefix=ERROR9 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR10 %s 2>&1 | FileCheck --check-prefix=ERROR10 %s
// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include -DERROR11 %s 2>&1 | FileCheck --check-prefix=ERROR11 %s

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
// ERROR4: error: op has a conflict with two operands having the same name 'tensor'
def OpWithDuplicatedArgNames : Op<Test_Dialect, "default_value"> {
  let arguments = (ins AnyTensor:$tensor, AnyTensor:$tensor);
}
#endif

#ifdef ERROR5
// ERROR5: error: op has a conflict with two results having the same name 'tensor'
def OpWithDuplicatedResultNames : Op<Test_Dialect, "default_value"> {
  let results = (outs AnyTensor:$tensor, AnyTensor:$tensor);
}
#endif

#ifdef ERROR6
// ERROR6: error: op has a conflict with operands and results both having an entry with the name 'tensor'
def OpWithDuplicatedArgResultNames : Op<Test_Dialect, "default_value"> {
  let arguments = (ins AnyTensor:$tensor);
  let results = (outs AnyTensor:$tensor);
}
#endif

#ifdef ERROR7
// ERROR7: error: op has a conflict with operands and regions both having an entry with the name 'tensor'
def OpWithDuplicatedArgResultNames : Op<Test_Dialect, "default_value"> {
  let arguments = (ins AnyTensor:$tensor);
  let regions = (region AnyRegion:$tensor);
}
#endif

#ifdef ERROR8
// ERROR8: error: op has a conflict with results and regions both having an entry with the name 'tensor'
def OpWithDuplicatedArgResultNames : Op<Test_Dialect, "default_value"> {
  let results = (outs AnyTensor:$tensor);
  let regions = (region AnyRegion:$tensor);
}
#endif

#ifdef ERROR9
// ERROR9: error: op has a conflict with operands and successors both having an entry with the name 'target'
def OpWithDuplicatedArgResultNames : Op<Test_Dialect, "default_value"> {
  let successors = (successor AnySuccessor:$target);
  let arguments = (ins AnyTensor:$target);
}
#endif

#ifdef ERROR10
// ERROR10: error: op has a conflict with results and successors both having an entry with the name 'target'
def OpWithDuplicatedArgResultNames : Op<Test_Dialect, "default_value"> {
  let successors = (successor AnySuccessor:$target);
  let results = (outs AnyTensor:$target);
}
#endif

#ifdef ERROR11
// ERROR11: error: op has a conflict with regions and successors both having an entry with the name 'target'
def OpWithDuplicatedArgResultNames : Op<Test_Dialect, "default_value"> {
  let successors = (successor AnySuccessor:$target);
  let regions = (region AnyRegion:$target);
}
#endif
