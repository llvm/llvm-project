// RUN: mlir-opt %s \
// RUN:  | mlir-opt -gpu-lower-to-nvvm-pipeline -debug-only=dump-sass \
// RUN:  2>&1 | FileCheck %s

// CHECK: MOV
// CHECK: STG.E 

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
