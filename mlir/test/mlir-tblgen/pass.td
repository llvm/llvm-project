// RUN: mlir-tblgen -gen-pass-decls -I %S/../../include %s | FileCheck %s

include "mlir/Pass/PassBase.td"

def FooPass : Pass<"foo", "ModuleOp"> {
  let summary = "A pass for testing pass code generation.";
  let dependentDialects = ["func::FuncDialect", "shape::ShapeDialect"];
}

// CHECK-LABEL: GEN_PASS_DEF_FOO
// CHECK: FooPassBase
// CHECK: void getDependentDialects
// CHECK-NEXT: registry.insert<func::FuncDialect>();
// CHECK-NEXT: registry.insert<shape::ShapeDialect>();
// CHECK-NEXT: }

// CHECK-LABEL: GEN_PASS_CLASSES
// CHECK: FooPassBase
// CHECK: void getDependentDialects
// CHECK-NEXT: registry.insert<func::FuncDialect>();
// CHECK-NEXT: registry.insert<shape::ShapeDialect>();
// CHECK-NEXT: }
