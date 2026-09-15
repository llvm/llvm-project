// RUN: mlir-tblgen -gen-op-doc -I %S/../../include -dialect=foo %s | FileCheck %s --check-prefix=CHECK-FOO
// RUN: mlir-tblgen -gen-op-doc -I %S/../../include -dialect=bar %s | FileCheck %s --check-prefix=CHECK-BAR

include "mlir/IR/OpBase.td"

def Foo_Dialect : Dialect {
  let name = "foo";
}

def Bar_Dialect : Dialect {
  let name = "bar";
}

// CHECK-FOO: foo.a
// CHECK-BAR-NOT: foo.a
def AOp : Op<Foo_Dialect, "a">;

// CHECK-BAR: bar.b
// CHECK-FOO-NOT: bar.b
def BOp : Op<Bar_Dialect, "b">;
