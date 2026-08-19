// RUN: mlir-opt %s -convert-vector-to-scf | FileCheck %s

// Negative test. Lowering a transfer op whose rank exceeds the target rank
// allocates a temporary buffer, so it needs an enclosing automatic allocation
// scope. These ops sit directly in the implicit `builtin.module`, which is not
// one, so the pattern has to decline and leave the transfer op alone.

// CHECK-NOT: memref.alloca
// CHECK: vector.transfer_write
%c0 = arith.constant 0 : index
%cst = arith.constant dense<0.0> : vector<2x3xf32>
%m = memref.alloc() : memref<2x3xf32>
vector.transfer_write %cst, %m[%c0, %c0] : vector<2x3xf32>, memref<2x3xf32>
