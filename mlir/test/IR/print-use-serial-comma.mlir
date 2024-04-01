// RUN: mlir-opt %s --mlir-print-serial-comma | FileCheck %s

// CHECK: foo.dense_attr = dense<[1, 2, and 3]> : tensor<3xi32>
"test.dense_attr"() {foo.dense_attr = dense<[1, 2, 3]> : tensor<3xi32>} : () -> ()

// Nested attributes not supported, we should use 1-d vectors from the LLVM dialect anyway.
// CHECK{LITERAL}: foo.dense_attr = dense<[[1, 2, 3], [4, 5, 6], [7, 8, and 9]]> : tensor<3x3xi32>
"test.nested_dense_attr"() {foo.dense_attr = dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi32>} : () -> ()

// Two elements, serial comma not necessary.
// CHECK: dense<[1, 2]> : tensor<2xi32>
"test.non_elided_dense_attr"() {foo.dense_attr = dense<[1, 2]> : tensor<2xi32>} : () -> ()

// CHECK{LITERAL}: sparse<[[0, 0, and 5]], -2.000000e+00> : vector<1x1x10xf16>
"test.sparse_attr"() {foo.sparse_attr = sparse<[[0, 0, 5]],  -2.0> : vector<1x1x10xf16>} : () -> ()

// One unique element, do not use serial comma.
// CHECK: dense<1> : tensor<3xi32>
"test.dense_splat"() {foo.dense_attr = dense<1> : tensor<3xi32>} : () -> ()
