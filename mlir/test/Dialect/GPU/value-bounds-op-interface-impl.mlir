// RUN: mlir-opt %s -pass-pipeline='builtin.module( \
// RUN:       func.func(test-affine-reify-value-bounds), \
// RUN:       gpu.module(llvm.func(test-affine-reify-value-bounds)), \
// RUN:       gpu.module(gpu.func(test-affine-reify-value-bounds)))' \
// RUN:     -verify-diagnostics \
// RUN:     -split-input-file | FileCheck %s

// CHECK-LABEL: func @launch_func
func.func @launch_func(%arg0 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  gpu.launch blocks(%block_id_x, %block_id_y, %block_id_z) in (%grid_dim_x = %arg0, %grid_dim_y = %c4, %grid_dim_z = %c2)
      threads(%thread_id_x, %thread_id_y, %thread_id_z) in (%block_dim_x = %c64, %block_dim_y = %c4, %block_dim_z = %c2) {

    // Sanity checks:
    // expected-error @below{{unknown}}
    "test.compare" (%thread_id_x, %c1) {cmp = "EQ"} : (index, index) -> ()
    // expected-remark @below{{false}}
    "test.compare" (%thread_id_x, %c64) {cmp = "GE"} : (index, index) -> ()

    // expected-remark @below{{true}}
    "test.compare" (%grid_dim_x, %c1) {cmp = "GE"}  : (index, index) -> ()
    // expected-remark @below{{true}}
    "test.compare" (%grid_dim_x, %arg0) {cmp = "EQ"} : (index, index) -> ()
    // expected-remark @below{{true}}
    "test.compare" (%grid_dim_y, %c4) {cmp = "EQ"} : (index, index) -> ()
    // expected-remark @below{{true}}
    "test.compare" (%grid_dim_z, %c2) {cmp = "EQ"} : (index, index) -> ()

    // expected-remark @below{{true}}
    "test.compare"(%block_id_x, %c0) {cmp = "GE"} : (index, index) -> ()
    // expected-remark @below{{true}}
    "test.compare"(%block_id_x, %arg0) {cmp = "LT"} : (index, index) -> ()
    // expected-remark @below{{true}}
    "test.compare"(%block_id_y, %c0) {cmp = "GE"} : (index, index) -> ()
    // expected-remark @below{{true}}
    "test.compare"(%block_id_y, %c4) {cmp = "LT"} : (index, index) -> ()
    // expected-remark @below{{true}}
    "test.compare"(%block_id_z, %c0) {cmp = "GE"} : (index, index) -> ()
    // expected-remark @below{{true}}
    "test.compare"(%block_id_z, %c2) {cmp = "LT"} : (index, index) -> ()

    // expected-remark @below{{true}}
    "test.compare" (%block_dim_x, %c64) {cmp = "EQ"} : (index, index) -> ()
    // expected-remark @below{{true}}
    "test.compare" (%block_dim_y, %c4) {cmp = "EQ"} : (index, index) -> ()
    // expected-remark @below{{true}}
    "test.compare" (%block_dim_z, %c2) {cmp = "EQ"} : (index, index) -> ()

    // expected-remark @below{{true}}
    "test.compare"(%thread_id_x, %c0) {cmp = "GE"} : (index, index) -> ()
    // expected-remark @below{{true}}
    "test.compare"(%thread_id_x, %c64) {cmp = "LT"} : (index, index) -> ()
    // expected-remark @below{{true}}
    "test.compare"(%thread_id_y, %c0) {cmp = "GE"} : (index, index) -> ()
    // expected-remark @below{{true}}
    "test.compare"(%thread_id_y, %c4) {cmp = "LT"} : (index, index) -> ()
    // expected-remark @below{{true}}
    "test.compare"(%thread_id_z, %c0) {cmp = "GE"} : (index, index) -> ()
    // expected-remark @below{{true}}
    "test.compare"(%thread_id_z, %c2) {cmp = "LT"} : (index, index) -> ()

    // expected-remark @below{{true}}
    "test.compare"(%thread_id_x, %block_dim_x) {cmp = "LT"} : (index, index) -> ()
    gpu.terminator
  }

  func.return
}

// -----

// The tests for what the ranges are are located in int-range-interface.mlir,
// so here we just make sure that the results of that interface propagate into
// constraints.

// CHECK-LABEL: func @kernel
module attributes {gpu.container_module} {
  gpu.module @gpu_module {
    llvm.func @kernel() attributes {gpu.kernel} {

      %c0 = arith.constant 0 : index
      %ctid_max = arith.constant 4294967295 : index
      %thread_id_x = gpu.thread_id x

      // expected-remark @below{{true}}
      "test.compare" (%thread_id_x, %c0) {cmp = "GE"} : (index, index) -> ()
      // expected-remark @below{{true}}
      "test.compare" (%thread_id_x, %ctid_max) {cmp = "LT"} : (index, index) -> ()
      llvm.return
    }
  }
}

// -----

// CHECK-LABEL: func @annotated_kernel
module attributes {gpu.container_module} {
  gpu.module @gpu_module {
    gpu.func @annotated_kernel() kernel
      attributes {known_block_size = array<i32: 8, 12, 16>,
          known_grid_size = array<i32: 20, 24, 28>} {

      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %thread_id_x = gpu.thread_id x

      // expected-remark @below{{true}}
      "test.compare"(%thread_id_x, %c0) {cmp = "GE"} : (index, index) -> ()
      // expected-remark @below{{true}}
      "test.compare"(%thread_id_x, %c8) {cmp = "LT"} : (index, index) -> ()

      %block_dim_x = gpu.block_dim x
      // expected-remark @below{{true}}
      "test.compare"(%block_dim_x, %c8) {cmp = "EQ"} : (index, index) -> ()

      // expected-remark @below{{true}}
      "test.compare"(%thread_id_x, %block_dim_x) {cmp = "LT"} : (index, index) -> ()
      gpu.return
    }
  }
}

// -----

// CHECK-LABEL: func @local_bounds_kernel
module attributes {gpu.container_module} {
  gpu.module @gpu_module {
    gpu.func @local_bounds_kernel() kernel {

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index

      %block_dim_x = gpu.block_dim x upper_bound 8
      // expected-remark @below{{true}}
      "test.compare"(%block_dim_x, %c1) {cmp = "GE"} : (index, index) -> ()
      // expected-remark @below{{true}}
      "test.compare"(%block_dim_x, %c8) {cmp = "LE"} : (index, index) -> ()
      // expected-error @below{{unknown}}
      "test.compare"(%block_dim_x, %c8) {cmp = "EQ"} : (index, index) -> ()

      %thread_id_x = gpu.thread_id x upper_bound 8
      // expected-remark @below{{true}}
      "test.compare"(%thread_id_x, %c0) {cmp = "GE"} : (index, index) -> ()
      // expected-remark @below{{true}}
      "test.compare"(%thread_id_x, %c8) {cmp = "LT"} : (index, index) -> ()

      // Note: there isn't a way to express the ID <= size constraint
      // in this form
      // expected-error @below{{unknown}}
      "test.compare"(%thread_id_x, %block_dim_x) {cmp = "LT"} : (index, index) -> ()
      gpu.return
    }
  }
}
