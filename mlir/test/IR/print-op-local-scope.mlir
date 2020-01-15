// RUN: mlir-opt %s -mlir-print-local-scope | FileCheck %s --dump-input-on-failure

// CHECK: "foo.op"() : () -> memref<?xf32, affine_map<(d0) -> (d0 * 2)>>
"foo.op"() : () -> (memref<?xf32, affine_map<(d0) -> (2*d0)>>)

