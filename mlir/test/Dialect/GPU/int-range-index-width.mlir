// RUN: mlir-opt %s -int-range-optimizations | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<index = 16 : i32>} {
  // Hardware grid dimensions are truncated to the runtime index width. A grid
  // dimension of 65536 becomes zero, so this comparison can be true.
  // CHECK-LABEL: func.func @grid_dimension_wraps
  // CHECK: gpu.grid_dim x
  // CHECK: arith.cmpi ult
  // CHECK: return
  func.func @grid_dimension_wraps() -> i1 {
    %d = gpu.grid_dim x
    %one = arith.constant 1 : index
    %cmp = arith.cmpi ult, %d, %one : index
    return %cmp : i1
  }

  // CHECK-LABEL: func.func @known_grid_dimension_wraps
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK-NEXT: return %[[ZERO]]
  func.func @known_grid_dimension_wraps() -> index
      attributes {gpu.known_grid_size = array<i32: 65536, 1, 1>} {
    %d = gpu.grid_dim x
    return %d : index
  }

}
