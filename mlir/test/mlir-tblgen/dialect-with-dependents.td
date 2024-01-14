// RUN: mlir-tblgen -gen-dialect-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def FooDialect : Dialect {
  let name = "foo";
  let dependentDialects = ["func::FuncDialect", "shape::ShapeDialect"];
}

// CHECK-LABEL: FooDialect::FooDialect
// CHECK: {
// CHECK-NEXT: getContext()->loadDialect<func::FuncDialect>();
// CHECK-NEXT: getContext()->loadDialect<shape::ShapeDialect>();
// CHECK-NEXT: initialize();
// CHECK-NEXT: }
