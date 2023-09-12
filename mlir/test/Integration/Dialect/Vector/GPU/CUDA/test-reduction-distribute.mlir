// REQUIRES: host-supports-nvptx

// RUN: mlir-opt %s -test-vector-warp-distribute="hoist-uniform distribute-transfer-write propagate-distribution" -canonicalize |\
// RUN: mlir-opt -test-vector-warp-distribute=rewrite-warp-ops-to-scf-if |\
// RUN: mlir-opt -lower-affine -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm \
// RUN:  -convert-arith-to-llvm -test-lower-to-nvvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_cuda_runtime \
// RUN:   -shared-libs=%mlir_c_runner_utils \
// RUN:   -shared-libs=%mlir_runner_utils | \
// RUN: FileCheck %s

// Run a tiled reduction fused with an elementwise op.

func.func @gpu_func(%in: memref<1024xf32>, %out: memref<1xf32>) {
  %c1 = arith.constant 1 : index
  %cst = arith.constant dense<100.0000> : vector<1xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c32 = arith.constant 32 : index
  gpu.launch blocks(%arg3, %arg4, %arg5)
  in (%arg9 = %c1, %arg10 = %c1, %arg11 = %c1)
  threads(%arg6, %arg7, %arg8) in (%arg12 = %c32, %arg13 = %c1, %arg14 = %c1) {
    vector.warp_execute_on_lane_0(%arg6)[32] {
      %init = vector.transfer_read %out[%c0], %cst_0 {in_bounds = [true]} : memref<1xf32>, vector<1xf32>
      %13 = scf.for %arg0 = %c0 to %c1024 step %c32 iter_args(%arg1 = %init) -> (vector<1xf32>) {
        %20 = vector.transfer_read %in[%arg0], %cst_0 {in_bounds = [true]} : memref<1024xf32>, vector<32xf32>
        %21 = vector.reduction <add>, %20 : vector<32xf32> into f32
        %22 = vector.broadcast %21 : f32 to vector<1xf32>
        %23 = arith.addf %22, %arg1 : vector<1xf32>
        scf.yield %23 : vector<1xf32>
      }
      %14 = arith.divf %13, %cst : vector<1xf32>
      vector.transfer_write %14, %out[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32>
    }
    gpu.terminator
  }
  return
}
func.func @main() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1024 = arith.constant 1024 : index
  %0 = memref.alloc() : memref<1024xf32>
  %1 = memref.alloc() : memref<1xf32>
  %cst_1 = arith.constant dense<[
    0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
    8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
    24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0]> : vector<32xf32>
  %cst_2 = arith.constant dense<2.000000e+00> : vector<1xf32>
  // init the buffers.
  scf.for %i = %c0 to %c1024 step %c32 {
    vector.transfer_write %cst_1, %0[%i] {in_bounds = [true]} : vector<32xf32>, memref<1024xf32>
  }
  vector.transfer_write %cst_2, %1[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32>
  %3 = memref.cast %0 : memref<1024xf32> to memref<*xf32>
  gpu.host_register %3 : memref<*xf32>
  %5 = memref.cast %1 : memref<1xf32> to memref<*xf32>
  gpu.host_register %5 : memref<*xf32>
  call @gpu_func(%0, %1) : (memref<1024xf32>, memref<1xf32>) -> ()
  %6 = vector.transfer_read %1[%c0], %cst : memref<1xf32>, vector<1xf32>
  vector.print %6 : vector<1xf32>
  return
}

// CHECK: ( 158.74 )
