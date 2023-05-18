// RUN: mlir-opt %s --irdl-file=%S/cmath.irdl.mlir | mlir-opt --irdl-file=%S/cmath.irdl.mlir | FileCheck %s

module {
  // CHECK: func.func @conorm(%[[p:[^:]*]]: !cmath.complex<f32>, %[[q:[^:]*]]: !cmath.complex<f32>) -> f32 {
  // CHECK:   %[[norm_p:[^ ]*]] = "cmath.norm"(%[[p]]) : (!cmath.complex<f32>) -> f32
  // CHECK:   %[[norm_q:[^ ]*]] = "cmath.norm"(%[[q]]) : (!cmath.complex<f32>) -> f32
  // CHECK:   %[[pq:[^ ]*]] = arith.mulf %[[norm_p]], %[[norm_q]] : f32
  // CHECK:   return %[[pq]] : f32
  // CHECK: }
  func.func @conorm(%p: !cmath.complex<f32>, %q: !cmath.complex<f32>) -> f32 {
    %norm_p = "cmath.norm"(%p) : (!cmath.complex<f32>) -> f32
    %norm_q = "cmath.norm"(%q) : (!cmath.complex<f32>) -> f32
    %pq = arith.mulf %norm_p, %norm_q : f32
    return %pq : f32
  }

  // CHECK: func.func @conorm2(%[[p:[^:]*]]: !cmath.complex<f32>, %[[q:[^:]*]]: !cmath.complex<f32>) -> f32 {
  // CHECK:   %[[pq:[^ ]*]] = "cmath.mul"(%[[p]], %[[q]]) : (!cmath.complex<f32>, !cmath.complex<f32>) -> !cmath.complex<f32>
  // CHECK:   %[[conorm:[^ ]*]] = "cmath.norm"(%[[pq]]) : (!cmath.complex<f32>) -> f32
  // CHECK:   return %[[conorm]] : f32
  // CHECK: }
  func.func @conorm2(%p: !cmath.complex<f32>, %q: !cmath.complex<f32>) -> f32 {
    %pq = "cmath.mul"(%p, %q) : (!cmath.complex<f32>, !cmath.complex<f32>) -> !cmath.complex<f32>
    %conorm = "cmath.norm"(%pq) : (!cmath.complex<f32>) -> f32
    return %conorm : f32
  }
}
