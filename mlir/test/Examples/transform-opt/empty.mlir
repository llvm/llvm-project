// RUN: mlir-transform-opt %s --transform=%p/self-contained.mlir | FileCheck %s
// RUN: mlir-transform-opt %s --transform=%p/external-decl.mlir --verify-diagnostics
// RUN: mlir-transform-opt %s --transform=%p/external-def.mlir --transform-entry-point=external_def | FileCheck %s --check-prefix=EXTERNAL
// RUN: mlir-transform-opt %s --transform=%p/external-decl.mlir --transform-library=%p/external-def.mlir | FileCheck %s --check-prefix=EXTERNAL
// RUN: mlir-transform-opt %s --transform=%p/syntax-error.mlir --verify-diagnostics
// RUN: mlir-transform-opt %s --transform=%p/self-contained.mlir --transform-library=%p/syntax-error.mlir --verify-diagnostics
// RUN: mlir-transform-opt %s --transform=%p/self-contained.mlir --transform-library=%p/external-def.mlir --transform-library=%p/syntax-error.mlir --verify-diagnostics

// CHECK: IR printer: in self-contained
// EXTERNAL: IR printer: external_def
// CHECK-NOT: @__transform_main
module {}
