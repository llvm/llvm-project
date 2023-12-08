// RUN: mlir-opt %s \
// RUN: | mlir-opt -gpu-lower-to-nvvm="cubin-format=%gpu_compilation_format" \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

func.func @other_func(%arg0 : f32, %arg1 : memref<?xf32>) {
  %cst = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cst2 = memref.dim %arg1, %c0 : memref<?xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst, %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst2, %block_y = %cst, %block_z = %cst) {
    memref.store %arg0, %arg1[%tx] : memref<?xf32>
    gpu.terminator
  }
  return
}

// CHECK: [1, 1, 1, 1, 1]
// CHECK: ( 1, 1 )
func.func @main() {
  %v0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0: index
  %arg0 = memref.alloc() : memref<5xf32>
  %21 = arith.constant 5 : i32
  %22 = memref.cast %arg0 : memref<5xf32> to memref<?xf32>
  %23 = memref.cast %22 : memref<?xf32> to memref<*xf32>
  gpu.host_register %23 : memref<*xf32>
  call @printMemrefF32(%23) : (memref<*xf32>) -> ()
  %24 = arith.constant 1.0 : f32
  call @other_func(%24, %22) : (f32, memref<?xf32>) -> ()
  call @printMemrefF32(%23) : (memref<*xf32>) -> ()
  %val1 = vector.transfer_read %arg0[%c0], %v0: memref<5xf32>, vector<2xf32>
  vector.print %val1: vector<2xf32>
  return
}

func.func private @printMemrefF32(%ptr : memref<*xf32>)
