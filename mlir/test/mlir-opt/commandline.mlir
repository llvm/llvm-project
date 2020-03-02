// RUN: mlir-opt --show-dialects | FileCheck %s
// CHECK: Registered Dialects:
// CHECK: affine
// CHECK: fxpmath
// CHECK: gpu
// CHECK: linalg
// CHECK: llvm
// CHECK: loop
// CHECK: nvvm
// CHECK: omp
// CHECK: quant
// CHECK: rocdl
// CHECK: sdbm
// CHECK: spv
// CHECK: std
// CHECK: test
// CHECK: vector
