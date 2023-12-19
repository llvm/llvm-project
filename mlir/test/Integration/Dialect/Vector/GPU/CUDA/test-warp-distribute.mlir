// Run the test cases without distributing ops to test default lowering. Run
// everything on the same thread.
// RUN: mlir-opt %s -test-vector-warp-distribute=rewrite-warp-ops-to-scf-if -canonicalize | \
// RUN: mlir-opt -convert-vector-to-scf -convert-scf-to-cf -convert-cf-to-llvm -convert-vector-to-llvm -convert-arith-to-llvm \
// RUN:  -gpu-lower-to-nvvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_cuda_runtime \
// RUN:   -shared-libs=%mlir_c_runner_utils \
// RUN:   -shared-libs=%mlir_runner_utils | \
// RUN: FileCheck %s

// Run the same test cases with distribution and propagation.
// RUN: mlir-opt %s  -test-vector-warp-distribute="hoist-uniform distribute-transfer-write" \
// RUN:   -test-vector-warp-distribute=rewrite-warp-ops-to-scf-if -canonicalize | \
// RUN: mlir-opt -convert-vector-to-scf -convert-scf-to-cf -convert-cf-to-llvm -convert-vector-to-llvm -convert-arith-to-llvm \
// RUN:  -gpu-lower-to-nvvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_cuda_runtime \
// RUN:   -shared-libs=%mlir_c_runner_utils \
// RUN:   -shared-libs=%mlir_runner_utils | \
// RUN: FileCheck %s

// RUN: mlir-opt %s  -test-vector-warp-distribute="hoist-uniform distribute-transfer-write propagate-distribution" \
// RUN:   -test-vector-warp-distribute=rewrite-warp-ops-to-scf-if -canonicalize | \
// RUN: mlir-opt -convert-vector-to-scf -convert-scf-to-cf -convert-cf-to-llvm -convert-vector-to-llvm -convert-arith-to-llvm \
// RUN:  -gpu-lower-to-nvvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_cuda_runtime \
// RUN:   -shared-libs=%mlir_c_runner_utils \
// RUN:   -shared-libs=%mlir_runner_utils | \
// RUN: FileCheck %s

func.func @gpu_func(%arg1: memref<32xf32>, %arg2: memref<32xf32>) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.000000e+00 : f32
  gpu.launch blocks(%arg3, %arg4, %arg5)
  in (%arg9 = %c1, %arg10 = %c1, %arg11 = %c1)
  threads(%arg6, %arg7, %arg8) in (%arg12 = %c32, %arg13 = %c1, %arg14 = %c1) {
    vector.warp_execute_on_lane_0(%arg6)[32] {
      %0 = vector.transfer_read %arg1[%c0], %cst {in_bounds = [true]} : memref<32xf32>, vector<32xf32>
      %1 = vector.transfer_read %arg2[%c0], %cst {in_bound = [true]} : memref<32xf32>, vector<32xf32>
      %2 = arith.addf %0, %1 : vector<32xf32>
      vector.transfer_write %2, %arg1[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32>
    }
    gpu.terminator
  }
  return
}
func.func @main() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = memref.alloc() : memref<32xf32>
  %1 = memref.alloc() : memref<32xf32>
  %cst_1 = arith.constant dense<[
    0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
    8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
    24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0]> : vector<32xf32>
  %cst_2 = arith.constant dense<2.000000e+00> : vector<32xf32>
  // init the buffers.
  vector.transfer_write %cst_1, %0[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32>
  vector.transfer_write %cst_2, %1[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32>
  %3 = memref.cast %0 : memref<32xf32> to memref<*xf32>
  gpu.host_register %3 : memref<*xf32>
  %5 = memref.cast %1 : memref<32xf32> to memref<*xf32>
  gpu.host_register %5 : memref<*xf32>
  call @gpu_func(%0, %1) : (memref<32xf32>, memref<32xf32>) -> ()
  %6 = vector.transfer_read %0[%c0], %cst : memref<32xf32>, vector<32xf32>
  vector.print %6 : vector<32xf32>
  return
}

// CHECK: ( 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33 )
