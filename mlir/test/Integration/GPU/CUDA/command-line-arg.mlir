// RUN: mlir-opt %s \
// RUN:  | mlir-opt -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_80 ptxas-cmd-options='-v --register-usage-level=8'" -debug-only=serialize-to-binary \
// RUN:  2>&1 | FileCheck %s

func.func @host_function(%arg0 : f32, %arg1 : memref<?xf32>) {
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

// CHECK: ptxas -arch sm_80
// CHECK-SAME: -v 
// CHECK-SAME: --register-usage-level=8
