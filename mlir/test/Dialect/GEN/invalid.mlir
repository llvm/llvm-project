// RUN: mlir-opt -split-input-file %s -verify-diagnostics

func.func @test_3d_nd_range_bounds_low() {
  %c-1 = arith.constant -1 : i32
  // expected-error @below {{'gen.local_id' op input dimension must be in the range [0, 3). Got -1}}
  %0 = gen.local_id %c-1
  func.return
}

// -----

func.func @test_3d_nd_range_bounds_high() {
  %c3 = arith.constant 3 : i32
  // expected-error @below {{'gen.work_group_id' op input dimension must be in the range [0, 3). Got 3}}
  %0 = gen.work_group_id %c3
  func.return
}
