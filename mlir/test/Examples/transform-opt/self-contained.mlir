// RUN: mlir-transform-opt %s | FileCheck %s
// RUN: mlir-transform-opt %s --transform=%s | FileCheck %s
// RUN: mlir-transform-opt %s --transform=%p/external-decl.mlir --verify-diagnostics
// RUN: mlir-transform-opt %s --transform=%p/external-def.mlir --transform-entry-point=external_def | FileCheck %s --check-prefix=EXTERNAL
// RUN: mlir-transform-opt %s --transform=%p/external-decl.mlir --transform-library=%p/external-def.mlir | FileCheck %s --check-prefix=EXTERNAL
// RUN: mlir-transform-opt %s --transform=%p/syntax-error.mlir --verify-diagnostics

// CHECK: IR printer: in self-contained
// EXTERNAL: IR printer: external_def

// The first occurrence comes from the print operation and the second is the
// roundtrip output. However, we shouldn't have the symbol duplicated because
// of library merging.
// CHECK-COUNT-2: @__transform_main
// CHECK-NOT: @__transform_main
module attributes {transform.with_named_sequence} {
  transform.named_sequence private @__transform_main(%root: !transform.any_op) {
    transform.print %root { name = "in self-contained" } : !transform.any_op
    transform.yield
  }
}
