// RUN: mlir-opt %s \
// RUN:  -transform-interpreter \
// RUN:  -test-transform-dialect-erase-schedule \
// RUN:  -gpu-lower-to-nvvm="cubin-chip=sm_80 cubin-features=+ptx76 cubin-format=%gpu_compilation_format" \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

!lhs_memref_type = memref<16x16xf16>
!rhs_memref_type = memref<16x8xf16>
!res_memref_type = memref<16x8xf16>

func.func @compute_linspace_val(%ridx: index, %cidx: index, %strideCidx: index) -> f16 {
  %r = arith.index_cast %ridx : index to i32
  %c = arith.index_cast %cidx : index to i32
  %strideC = arith.index_cast %strideCidx : index to i32
  %2 = arith.muli %r, %strideC : i32
  %3 = arith.addi %c, %2 : i32
  %4 = arith.sitofp %3 : i32 to f16
  %factor = arith.constant 64.0 : f16
  %5 = arith.divf %4, %factor : f16
  return %5: f16
}

func.func @print_lhs_as_memref_32(%lhs: !lhs_memref_type) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %lhs, %c0 : !lhs_memref_type
  %N = memref.dim %lhs, %c1 : !lhs_memref_type
  %tmp_alloc = memref.alloc(%M, %N) : memref<?x?xf32>
  scf.for %m = %c0 to %M step %c1 {
    scf.for %n = %c0 to %N step %c1 {
      %f16 = memref.load %lhs[%m, %n] : !lhs_memref_type
      %f32 = arith.extf %f16 : f16 to f32
      memref.store %f32, %tmp_alloc[%m, %n] : memref<?x?xf32>
    }
  }
  %casted = memref.cast %tmp_alloc : memref<?x?xf32> to memref<*xf32>
  call @printMemrefF32(%casted) : (memref<*xf32>) -> ()
  memref.dealloc %tmp_alloc : memref<?x?xf32>
  return
}

func.func @print_rhs_as_memref_32(%rhs: !rhs_memref_type) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %rhs, %c0 : !rhs_memref_type
  %N = memref.dim %rhs, %c1 : !rhs_memref_type
  %tmp_alloc = memref.alloc(%M, %N) : memref<?x?xf32>
  scf.for %m = %c0 to %M step %c1 {
    scf.for %n = %c0 to %N step %c1 {
      %f16 = memref.load %rhs[%m, %n] : !rhs_memref_type
      %f32 = arith.extf %f16 : f16 to f32
      memref.store %f32, %tmp_alloc[%m, %n] : memref<?x?xf32>
    }
  }
  %casted = memref.cast %tmp_alloc : memref<?x?xf32> to memref<*xf32>
  call @printMemrefF32(%casted) : (memref<*xf32>) -> ()
  memref.dealloc %tmp_alloc : memref<?x?xf32>
  return
}

func.func @print_res_as_memref_32(%res: !res_memref_type) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %res, %c0 : !res_memref_type
  %N = memref.dim %res, %c1 : !res_memref_type
  %tmp_alloc = memref.alloc(%M, %N) : memref<?x?xf32>
  scf.for %m = %c0 to %M step %c1 {
    scf.for %n = %c0 to %N step %c1 {
      %f16 = memref.load %res[%m, %n] : !res_memref_type
      %f32 = arith.extf %f16 : f16 to f32
      memref.store %f32, %tmp_alloc[%m, %n] : memref<?x?xf32>
    }
  }
  %casted = memref.cast %tmp_alloc : memref<?x?xf32> to memref<*xf32>
  call @printMemrefF32(%casted) : (memref<*xf32>) -> ()
  memref.dealloc %tmp_alloc : memref<?x?xf32>
  return
}

func.func @main() {
  %lhs = memref.alloc() : !lhs_memref_type
  %rhs = memref.alloc() : !rhs_memref_type
  %res = memref.alloc() : !res_memref_type

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %res, %c0 : !res_memref_type
  %N = memref.dim %res, %c1 : !res_memref_type
  %K = memref.dim %lhs, %c1 : !lhs_memref_type

  %f1 = arith.constant 1.0e+00 : f16
  %f0 = arith.constant 0.0e+00 : f16
  %c32 = arith.constant 32 : index

  // Intialize the lhs matrix with a linspace function.
  scf.for %r = %c0 to %M step %c1 {
    scf.for %c = %c0 to %K step %c1 {
      %idx = func.call @compute_linspace_val(%r, %c, %K) : (index, index, index) -> f16
      memref.store %idx, %lhs[%r, %c] : !lhs_memref_type
    }
  }
  // Intialize the rhs matrix with a linspace function.
  scf.for %r = %c0 to %K step %c1 {
    scf.for %c = %c0 to %N step %c1 {
      %idx = func.call @compute_linspace_val(%r, %c, %N) : (index, index, index) -> f16
      memref.store %idx, %rhs[%r, %c] : !rhs_memref_type
    }
  }
  // Intialize the rhs matrix with a linspace function.
  scf.for %r = %c0 to %M step %c1 {
    scf.for %c = %c0 to %N step %c1 {
      %idx = func.call @compute_linspace_val(%r, %c, %N) : (index, index, index) -> f16
      memref.store %idx, %res[%r, %c] : !res_memref_type
    }
  }

  %ulhs = memref.cast %lhs : !lhs_memref_type to memref<*xf16>
  %urhs = memref.cast %rhs : !rhs_memref_type to memref<*xf16>
  %ures = memref.cast %res : !res_memref_type to memref<*xf16>
  gpu.host_register %ulhs : memref<*xf16>
  gpu.host_register %urhs : memref<*xf16>
  gpu.host_register %ures : memref<*xf16>

  // Print the memrefs before computation.
  call @print_lhs_as_memref_32(%lhs) : (!lhs_memref_type) -> ()
  // CHECK: [0,   0.015625,   0.03125,   0.046875,   0.0625,   0.078125,   0.09375,   0.109375,   0.125,   0.140625,   0.15625,   0.171875,   0.1875,   0.203125,   0.21875,   0.234375],
  // CHECK: [0.25,   0.265625,   0.28125,   0.296875,   0.3125,   0.328125,   0.34375,   0.359375,   0.375,   0.390625,   0.40625,   0.421875,   0.4375,   0.453125,   0.46875,   0.484375],
  // CHECK: [0.5,   0.515625,   0.53125,   0.546875,   0.5625,   0.578125,   0.59375,   0.609375,   0.625,   0.640625,   0.65625,   0.671875,   0.6875,   0.703125,   0.71875,   0.734375],
  // CHECK: [0.75,   0.765625,   0.78125,   0.796875,   0.8125,   0.828125,   0.84375,   0.859375,   0.875,   0.890625,   0.90625,   0.921875,   0.9375,   0.953125,   0.96875,   0.984375],
  // CHECK: [1,   1.01562,   1.03125,   1.04688,   1.0625,   1.07812,   1.09375,   1.10938,   1.125,   1.14062,   1.15625,   1.17188,   1.1875,   1.20312,   1.21875,   1.23438],
  // CHECK: [1.25,   1.26562,   1.28125,   1.29688,   1.3125,   1.32812,   1.34375,   1.35938,   1.375,   1.39062,   1.40625,   1.42188,   1.4375,   1.45312,   1.46875,   1.48438],
  // CHECK: [1.5,   1.51562,   1.53125,   1.54688,   1.5625,   1.57812,   1.59375,   1.60938,   1.625,   1.64062,   1.65625,   1.67188,   1.6875,   1.70312,   1.71875,   1.73438],
  // CHECK: [1.75,   1.76562,   1.78125,   1.79688,   1.8125,   1.82812,   1.84375,   1.85938,   1.875,   1.89062,   1.90625,   1.92188,   1.9375,   1.95312,   1.96875,   1.98438],
  // CHECK: [2,   2.01562,   2.03125,   2.04688,   2.0625,   2.07812,   2.09375,   2.10938,   2.125,   2.14062,   2.15625,   2.17188,   2.1875,   2.20312,   2.21875,   2.23438],
  // CHECK: [2.25,   2.26562,   2.28125,   2.29688,   2.3125,   2.32812,   2.34375,   2.35938,   2.375,   2.39062,   2.40625,   2.42188,   2.4375,   2.45312,   2.46875,   2.48438],
  // CHECK: [2.5,   2.51562,   2.53125,   2.54688,   2.5625,   2.57812,   2.59375,   2.60938,   2.625,   2.64062,   2.65625,   2.67188,   2.6875,   2.70312,   2.71875,   2.73438],
  // CHECK: [2.75,   2.76562,   2.78125,   2.79688,   2.8125,   2.82812,   2.84375,   2.85938,   2.875,   2.89062,   2.90625,   2.92188,   2.9375,   2.95312,   2.96875,   2.98438],
  // CHECK: [3,   3.01562,   3.03125,   3.04688,   3.0625,   3.07812,   3.09375,   3.10938,   3.125,   3.14062,   3.15625,   3.17188,   3.1875,   3.20312,   3.21875,   3.23438],
  // CHECK: [3.25,   3.26562,   3.28125,   3.29688,   3.3125,   3.32812,   3.34375,   3.35938,   3.375,   3.39062,   3.40625,   3.42188,   3.4375,   3.45312,   3.46875,   3.48438],
  // CHECK: [3.5,   3.51562,   3.53125,   3.54688,   3.5625,   3.57812,   3.59375,   3.60938,   3.625,   3.64062,   3.65625,   3.67188,   3.6875,   3.70312,   3.71875,   3.73438],
  // CHECK: [3.75,   3.76562,   3.78125,   3.79688,   3.8125,   3.82812,   3.84375,   3.85938,   3.875,   3.89062,   3.90625,   3.92188,   3.9375,   3.95312,   3.96875,   3.98438]

  call @print_rhs_as_memref_32(%rhs) : (!rhs_memref_type) -> ()
  // CHECK: [0,   0.015625,   0.03125,   0.046875,   0.0625,   0.078125,   0.09375,   0.109375],
  // CHECK: [0.125,   0.140625,   0.15625,   0.171875,   0.1875,   0.203125,   0.21875,   0.234375],
  // CHECK: [0.25,   0.265625,   0.28125,   0.296875,   0.3125,   0.328125,   0.34375,   0.359375],
  // CHECK: [0.375,   0.390625,   0.40625,   0.421875,   0.4375,   0.453125,   0.46875,   0.484375],
  // CHECK: [0.5,   0.515625,   0.53125,   0.546875,   0.5625,   0.578125,   0.59375,   0.609375],
  // CHECK: [0.625,   0.640625,   0.65625,   0.671875,   0.6875,   0.703125,   0.71875,   0.734375],
  // CHECK: [0.75,   0.765625,   0.78125,   0.796875,   0.8125,   0.828125,   0.84375,   0.859375],
  // CHECK: [0.875,   0.890625,   0.90625,   0.921875,   0.9375,   0.953125,   0.96875,   0.984375],
  // CHECK: [1,   1.01562,   1.03125,   1.04688,   1.0625,   1.07812,   1.09375,   1.10938],
  // CHECK: [1.125,   1.14062,   1.15625,   1.17188,   1.1875,   1.20312,   1.21875,   1.23438],
  // CHECK: [1.25,   1.26562,   1.28125,   1.29688,   1.3125,   1.32812,   1.34375,   1.35938],
  // CHECK: [1.375,   1.39062,   1.40625,   1.42188,   1.4375,   1.45312,   1.46875,   1.48438],
  // CHECK: [1.5,   1.51562,   1.53125,   1.54688,   1.5625,   1.57812,   1.59375,   1.60938],
  // CHECK: [1.625,   1.64062,   1.65625,   1.67188,   1.6875,   1.70312,   1.71875,   1.73438],
  // CHECK: [1.75,   1.76562,   1.78125,   1.79688,   1.8125,   1.82812,   1.84375,   1.85938],
  // CHECK: [1.875,   1.89062,   1.90625,   1.92188,   1.9375,   1.95312,   1.96875,   1.98438]

  call @print_res_as_memref_32(%res) : (!res_memref_type) -> ()
  // CHECK: [0,   0.015625,   0.03125,   0.046875,   0.0625,   0.078125,   0.09375,   0.109375],
  // CHECK: [0.125,   0.140625,   0.15625,   0.171875,   0.1875,   0.203125,   0.21875,   0.234375],
  // CHECK: [0.25,   0.265625,   0.28125,   0.296875,   0.3125,   0.328125,   0.34375,   0.359375],
  // CHECK: [0.375,   0.390625,   0.40625,   0.421875,   0.4375,   0.453125,   0.46875,   0.484375],
  // CHECK: [0.5,   0.515625,   0.53125,   0.546875,   0.5625,   0.578125,   0.59375,   0.609375],
  // CHECK: [0.625,   0.640625,   0.65625,   0.671875,   0.6875,   0.703125,   0.71875,   0.734375],
  // CHECK: [0.75,   0.765625,   0.78125,   0.796875,   0.8125,   0.828125,   0.84375,   0.859375],
  // CHECK: [0.875,   0.890625,   0.90625,   0.921875,   0.9375,   0.953125,   0.96875,   0.984375],
  // CHECK: [1,   1.01562,   1.03125,   1.04688,   1.0625,   1.07812,   1.09375,   1.10938],
  // CHECK: [1.125,   1.14062,   1.15625,   1.17188,   1.1875,   1.20312,   1.21875,   1.23438],
  // CHECK: [1.25,   1.26562,   1.28125,   1.29688,   1.3125,   1.32812,   1.34375,   1.35938],
  // CHECK: [1.375,   1.39062,   1.40625,   1.42188,   1.4375,   1.45312,   1.46875,   1.48438],
  // CHECK: [1.5,   1.51562,   1.53125,   1.54688,   1.5625,   1.57812,   1.59375,   1.60938],
  // CHECK: [1.625,   1.64062,   1.65625,   1.67188,   1.6875,   1.70312,   1.71875,   1.73438],
  // CHECK: [1.75,   1.76562,   1.78125,   1.79688,   1.8125,   1.82812,   1.84375,   1.85938],
  // CHECK: [1.875,   1.89062,   1.90625,   1.92188,   1.9375,   1.95312,   1.96875,   1.98438]

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c32, %block_y = %c1, %block_z = %c1) {

    linalg.matmul ins(%lhs, %rhs: !lhs_memref_type, !rhs_memref_type)
                 outs(%res: !res_memref_type)

    gpu.terminator
  }


  // Print the result memref after computation.
  // This has been verified against other f16 CUDA implementations.
  call @print_res_as_memref_32(%res) : (!res_memref_type) -> ()
  // CHECK: [2.42188,   2.4668,   2.51172,   2.55664,   2.60156,   2.64648,   2.69141,   2.73633],
  // CHECK: [6.29688,   6.40625,   6.51172,   6.61719,   6.72656,   6.83594,   6.94141,   7.04688],
  // CHECK: [10.1719,   10.3438,   10.5156,   10.6797,   10.8516,   11.0234,   11.1875,   11.3594],
  // CHECK: [14.0469,   14.2812,   14.5156,   14.7422,   14.9766,   15.2109,   15.4375,   15.6719],
  // CHECK: [17.9219,   18.2188,   18.5156,   18.8125,   19.0938,   19.3906,   19.6875,   19.9844],
  // CHECK: [21.7969,   22.1562,   22.5156,   22.875,   23.2188,   23.5781,   23.9375,   24.2969],
  // CHECK: [25.6719,   26.0938,   26.5156,   26.9375,   27.3438,   27.7656,   28.1875,   28.6094],
  // CHECK: [29.5469,   30.0312,   30.5156,   31,   31.4688,   31.9531,   32.4375,   32.9375],
  // CHECK: [33.4375,   33.9688,   34.5,   35.0625,   35.5938,   36.1562,   36.6875,   37.25],
  // CHECK: [37.3125,   37.9062,   38.5,   39.125,   39.7188,   40.3438,   40.9375,   41.5625],
  // CHECK: [41.1875,   41.8438,   42.5,   43.1875,   43.8438,   44.5312,   45.1875,   45.875],
  // CHECK: [45.0625,   45.7812,   46.5,   47.25,   47.9688,   48.7188,   49.4375,   50.1875],
  // CHECK: [48.9375,   49.7188,   50.5,   51.3125,   52.0938,   52.9062,   53.6875,   54.5],
  // CHECK: [52.8125,   53.6562,   54.5,   55.375,   56.2188,   57.0938,   57.9375,   58.8125],
  // CHECK: [56.6875,   57.5938,   58.5,   59.4375,   60.3438,   61.2812,   62.1875,   63.125],
  // CHECK: [60.5625,   61.5312,   62.5,   63.5,   64.5,   65.4375,   66.4375,   67.4375]

  return
}

func.func private @printMemrefF32(memref<*xf32>)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.nvgpu.rewrite_matmul_as_mma_sync %matmul
      : (!transform.any_op) -> ()
      transform.yield
  }
}
