// RUN: mlir-opt --show-dialects | FileCheck %s
// CHECK: Available Dialects:
// CHECK: affine
// CHECK: gpu
// CHECK: linalg
// CHECK: llvm
// CHECK: nvvm
// CHECK: omp
// CHECK: quant
// CHECK: rocdl
// CHECK: scf
// CHECK: sdbm
// CHECK: spv
// CHECK: std
// CHECK: test
// CHECK: vector
