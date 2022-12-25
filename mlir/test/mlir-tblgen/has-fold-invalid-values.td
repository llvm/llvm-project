// RUN: not mlir-tblgen -gen-op-decls -I %S/../../include %s 2>&1 | FileCheck %s

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "NS";
  let useFoldAPI = 3;
}

def InvalidValue_Op : Op<Test_Dialect, "invalid_op"> {
  let hasFolder = 1;
}

// CHECK: Invalid value for dialect field `useFoldAPI`
