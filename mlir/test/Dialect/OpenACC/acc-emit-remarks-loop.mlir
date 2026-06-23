// RUN: mlir-opt %s -split-input-file -acc-emit-remarks-loop --remarks-filter="(open)?acc.*" 2>&1 | FileCheck %s
// RUN: mlir-opt %s -split-input-file -acc-emit-remarks-loop='gpu-dim-separator=%' --remarks-filter="(open)?acc.*" 2>&1 | FileCheck %s --check-prefix=PERCENT

// CHECK: remark: [Passed] openacc | Category:acc-emit-remarks-loop | Function=vector_loop | Remark="!$acc loop vector(128) ! threadidx.x"
func.func @vector_loop() {
  %c128 = arith.constant 128 : index
  acc.kernel_environment {
    %w0 = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
    acc.compute_region launch(%arg0 = %w0) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c128_inner = arith.constant 128 : index
      scf.parallel (%iv) = (%c0) to (%c128_inner) step (%c1) {
        scf.reduce
      } {acc.par_dims = #acc<par_dims[thread_x]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

// -----

// CHECK: remark: [Passed] openacc | Category:acc-emit-remarks-loop | Function=gang_loop | Remark="!$acc loop gang(8) ! blockidx.x"
func.func @gang_loop() {
  %c8 = arith.constant 8 : index
  acc.kernel_environment {
    %w0 = acc.par_width %c8 {par_dim = #acc.par_dim<block_x>}
    acc.compute_region launch(%arg0 = %w0) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8_inner = arith.constant 8 : index
      scf.parallel (%iv) = (%c0) to (%c8_inner) step (%c1) {
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

// -----

// CHECK: remark: [Passed] openacc | Category:acc-emit-remarks-loop | Function=worker_loop | Remark="!$acc loop worker(4) ! threadidx.y"
func.func @worker_loop() {
  %c4 = arith.constant 4 : index
  acc.kernel_environment {
    %w0 = acc.par_width %c4 {par_dim = #acc.par_dim<thread_y>}
    acc.compute_region launch(%arg0 = %w0) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%iv) = (%c0) to (%c8) step (%c1) {
        scf.reduce
      } {acc.par_dims = #acc<par_dims[thread_y]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

// -----

// CHECK: remark: [Passed] openacc | Category:acc-emit-remarks-loop | Function=sequential_loop | Remark="!$acc loop sequential"
func.func @sequential_loop() {
  acc.kernel_environment {
    acc.compute_region {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%iv) = (%c0) to (%c4) step (%c1) {
        scf.reduce
      } {acc.par_dims = #acc<par_dims[sequential]>}
      acc.yield
    } {origin = "acc.kernels"}
  }
  return
}

// -----

// CHECK: remark: [Passed] openacc | Category:acc-emit-remarks-loop | Function=block_and_vector | Remark="!$acc loop gang(8), vector(128) ! blockidx.x threadidx.x"
func.func @block_and_vector() {
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  acc.kernel_environment {
    %w0 = acc.par_width %c8 {par_dim = #acc.par_dim<block_x>}
    %w1 = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
    acc.compute_region launch(%arg0 = %w0, %arg1 = %w1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8_inner = arith.constant 8 : index
      %c128_inner = arith.constant 128 : index
      scf.parallel (%i, %j) = (%c0, %c0) to (%c8_inner, %c128_inner) step (%c1, %c1) {
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x, thread_x]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

// -----

// CHECK: remark: [Passed] openacc | Category:acc-emit-remarks-loop | Function=scf_for_sequential | Remark="!$acc loop sequential"
func.func @scf_for_sequential() {
  acc.kernel_environment {
    acc.compute_region {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %iv = %c0 to %c4 step %c1 {
      }
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

// -----

// CHECK: remark: [Passed] openacc | Category:acc-emit-remarks-loop | Function=collapse_loop | Remark="!$acc loop sequential collapse(2)"
func.func @collapse_loop() {
  acc.kernel_environment {
    acc.compute_region {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %iv = %c0 to %c4 step %c1 {
      } {acc.par_dims = #acc<par_dims[sequential]>, acc.collapse_count = 2 : i64}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

// -----

// PERCENT: remark: [Passed] openacc | Category:acc-emit-remarks-loop | Function=percent_separator | Remark="!$acc loop vector(128) ! threadidx%x"
func.func @percent_separator() {
  %c128 = arith.constant 128 : index
  acc.kernel_environment {
    %w0 = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
    acc.compute_region launch(%arg0 = %w0) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c128_inner = arith.constant 128 : index
      scf.parallel (%iv) = (%c0) to (%c128_inner) step (%c1) {
        scf.reduce
      } {acc.par_dims = #acc<par_dims[thread_x]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}
