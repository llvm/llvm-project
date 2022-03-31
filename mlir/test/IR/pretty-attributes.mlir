// RUN: mlir-opt %s -mlir-elide-elementsattrs-if-larger=2 | FileCheck %s
// Ensure that the elided version is still parseable, although depending on
// what has been elided, it may not be semantically meaningful.
// In the typical case where what is being elided is a very large constant
// tensor which passes don't look at directly, this isn't an issue.
// RUN: mlir-opt %s -mlir-elide-elementsattrs-if-larger=2 | mlir-opt

// CHECK: opaque<"elided_large_const", "0xDEADBEEF"> : tensor<3xi32>
"test.dense_attr"() {foo.dense_attr = dense<[1, 2, 3]> : tensor<3xi32>} : () -> ()

// CHECK: dense<[1, 2]> : tensor<2xi32>
"test.non_elided_dense_attr"() {foo.dense_attr = dense<[1, 2]> : tensor<2xi32>} : () -> ()

// CHECK: opaque<"elided_large_const", "0xDEADBEEF"> : vector<1x1x10xf16>
"test.sparse_attr"() {foo.sparse_attr = sparse<[[0, 0, 5]],  -2.0> : vector<1x1x10xf16>} : () -> ()

// CHECK: opaque<"elided_large_const", "0xDEADBEEF"> : tensor<100xf32>
"test.opaque_attr"() {foo.opaque_attr = opaque<"elided_large_const", "0xEBFE"> : tensor<100xf32> } : () -> ()

// CHECK: dense<1> : tensor<3xi32>
"test.dense_splat"() {foo.dense_attr = dense<1> : tensor<3xi32>} : () -> ()
