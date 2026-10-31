// RUN: split-file %s %t
// RUN: mlir-opt %t/missing-factors.mlir -test-parallel-loop-unrolling -verify-diagnostics
// RUN: mlir-opt %t/zero-factor.mlir -test-parallel-loop-unrolling='unroll-factors=1,0' -verify-diagnostics

//--- missing-factors.mlir
// expected-error@unknown {{missing `unroll-factors` pass option}}
module {}

//--- zero-factor.mlir
// expected-error@unknown {{unroll factors must be non-zero}}
module {}
