// RUN: mlir-opt %s \
// RUN:   -transform-interpreter \
// RUN: | FileCheck %s --check-prefix=CHECK-MMA-SYNC

// CHECK-MMA-SYNC-LABEL: func @main() {
//       CHECK-MMA-SYNC:   nvgpu.mma.sync(%{{.*}}) {mmaShape = [16, 8, 4], tf32Enabled}
//  CHECK-MMA-SYNC-SAME:     : (vector<2x1xf32>, vector<1x1xf32>, vector<2x2xf32>) -> vector<2x2xf32>

// Tested to run locally in 1.7s.

// RUN: mlir-opt %s \
// RUN:   -transform-interpreter \
// RUN:   -test-transform-dialect-erase-schedule \
// RUN:   -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_80 cubin-features=+ptx76 cubin-format=%gpu_compilation_format" \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

!lhs_memref_type = memref<16x4xf32>
!rhs_memref_type = memref<4x8xf32>
!res_memref_type = memref<16x8xf32>

func.func @compute_linspace_val(%ridx: index, %cidx: index, %strideCidx: index) -> f32 {
  %r = arith.index_cast %ridx : index to i32
  %c = arith.index_cast %cidx : index to i32
  %strideC = arith.index_cast %strideCidx : index to i32
  %2 = arith.muli %r, %strideC : i32
  %3 = arith.addi %c, %2 : i32
  %4 = arith.sitofp %3 : i32 to f32
  return %4: f32
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

  %f1 = arith.constant 1.0e+00 : f32
  %f0 = arith.constant 0.0e+00 : f32
  %c32 = arith.constant 32 : index

  // Intialize the lhs matrix with a linspace function.
  scf.for %r = %c0 to %M step %c1 {
    scf.for %c = %c0 to %K step %c1 {
      %idx = func.call @compute_linspace_val(%r, %c, %K) : (index, index, index) -> f32
      memref.store %idx, %lhs[%r, %c] : !lhs_memref_type
    }
  }
  // Intialize the rhs matrix with a linspace function.
  scf.for %r = %c0 to %K step %c1 {
    scf.for %c = %c0 to %N step %c1 {
      %idx = func.call @compute_linspace_val(%r, %c, %N) : (index, index, index) -> f32
      memref.store %idx, %rhs[%r, %c] : !rhs_memref_type
    }
  }
  // Intialize the rhs matrix with a linspace function.
  scf.for %r = %c0 to %M step %c1 {
    scf.for %c = %c0 to %N step %c1 {
      %idx = func.call @compute_linspace_val(%r, %c, %N) : (index, index, index) -> f32
      memref.store %idx, %res[%r, %c] : !res_memref_type
    }
  }

  %ulhs = memref.cast %lhs : !lhs_memref_type to memref<*xf32>
  %urhs = memref.cast %rhs : !rhs_memref_type to memref<*xf32>
  %ures = memref.cast %res : !res_memref_type to memref<*xf32>
  gpu.host_register %ulhs : memref<*xf32>
  gpu.host_register %urhs : memref<*xf32>
  gpu.host_register %ures : memref<*xf32>

  // Print the memrefs before computation.
  call @printMemrefF32(%ulhs) : (memref<*xf32>) -> ()
  // CHECK: [0,  1,  2,  3],
  // CHECK: [4,  5,  6,  7],
  // CHECK: [8,  9, 10, 11],
  // CHECK: [12, 13, 14, 15],
  // CHECK: [16, 17, 18, 19],
  // CHECK: [20, 21, 22, 23],
  // CHECK: [24, 25, 26, 27],
  // CHECK: [28, 29, 30, 31],
  // CHECK: [32, 33, 34, 35],
  // CHECK: [36, 37, 38, 39],
  // CHECK: [40, 41, 42, 43],
  // CHECK: [44, 45, 46, 47],
  // CHECK: [48, 49, 50, 51],
  // CHECK: [52, 53, 54, 55],
  // CHECK: [56, 57, 58, 59],
  // CHECK: [60, 61, 62, 63]

  call @printMemrefF32(%urhs) : (memref<*xf32>) -> ()
  // CHECK: [0,  1,  2,  3,  4,  5,  6,  7],
  // CHECK: [8,  9, 10, 11, 12, 13, 14, 15],
  // CHECK: [16, 17, 18, 19, 20, 21, 22, 23],
  // CHECK: [24, 25, 26, 27, 28, 29, 30, 31]

  call @printMemrefF32(%ures) : (memref<*xf32>) -> ()
  // CHECK: [0,   1,   2,   3,   4,   5,   6,   7],
  // CHECK: [8,   9,  10,  11,  12,  13,  14,  15],
  // CHECK: [16,  17,  18,  19,  20,  21,  22,  23],
  // CHECK: [24,  25,  26,  27,  28,  29,  30,  31],
  // CHECK: [32,  33,  34,  35,  36,  37,  38,  39],
  // CHECK: [40,  41,  42,  43,  44,  45,  46,  47],
  // CHECK: [48,  49,  50,  51,  52,  53,  54,  55],
  // CHECK: [56,  57,  58,  59,  60,  61,  62,  63],
  // CHECK: [64,  65,  66,  67,  68,  69,  70,  71],
  // CHECK: [72,  73,  74,  75,  76,  77,  78,  79],
  // CHECK: [80,  81,  82,  83,  84,  85,  86,  87],
  // CHECK: [88,  89,  90,  91,  92,  93,  94,  95],
  // CHECK: [96,  97,  98,  99, 100, 101, 102, 103],
  // CHECK: [104, 105, 106, 107, 108, 109, 110, 111],
  // CHECK: [112, 113, 114, 115, 116, 117, 118, 119],
  // CHECK: [120, 121, 122, 123, 124, 125, 126, 127]

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c32, %block_y = %c1, %block_z = %c1) {

    linalg.matmul ins(%lhs, %rhs: !lhs_memref_type, !rhs_memref_type)
                 outs(%res: !res_memref_type)

    gpu.terminator
  }


  // Print the result memref after computation.
  call @printMemrefF32(%ures) : (memref<*xf32>) -> ()

  // CHECK: [112, 119, 126, 133, 140, 147, 154, 161],
  // CHECK: [312, 335, 358, 381, 404, 427, 450, 473],
  // CHECK: [512, 551, 590, 629, 668, 707, 746, 785],
  // CHECK: [712, 767, 822, 877, 932, 987, 1042, 1097],
  // CHECK: [912, 983, 1054, 1125, 1196, 1267, 1338, 1409],
  // CHECK: [1112, 1199, 1286, 1373, 1460, 1547, 1634, 1721],
  // CHECK: [1312, 1415, 1518, 1621, 1724, 1827, 1930, 2033],
  // CHECK: [1512, 1631, 1750, 1869, 1988, 2107, 2226, 2345],
  // CHECK: [1712, 1847, 1982, 2117, 2252, 2387, 2522, 2657],
  // CHECK: [1912, 2063, 2214, 2365, 2516, 2667, 2818, 2969],
  // CHECK: [2112, 2279, 2446, 2613, 2780, 2947, 3114, 3281],
  // CHECK: [2312, 2495, 2678, 2861, 3044, 3227, 3410, 3593],
  // CHECK: [2512, 2711, 2910, 3109, 3308, 3507, 3706, 3905],
  // CHECK: [2712, 2927, 3142, 3357, 3572, 3787, 4002, 4217],
  // CHECK: [2912, 3143, 3374, 3605, 3836, 4067, 4298, 4529],
  // CHECK: [3112, 3359, 3606, 3853, 4100, 4347, 4594, 4841]

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
