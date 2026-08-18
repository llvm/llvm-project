// RUN: mlir-opt %s -convert-vector-to-scf | FileCheck %s

// The lowering of a transfer op whose rank exceeds the target rank allocates a
// temporary buffer, so it needs an enclosing automatic allocation scope.

// CHECK: vector.transfer_write
%c0 = arith.constant 0 : index
%cst = arith.constant dense<0.0> : vector<2x3xf32>
%m = memref.alloc() : memref<2x3xf32>
vector.transfer_write %cst, %m[%c0, %c0] : vector<2x3xf32>, memref<2x3xf32>
