// RUN: mlir-opt %s -sparse-tensor-codegen -cse | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>
// CHECK-LABEL:  func @for(
// CHECK-SAME:             %[[DIM_SIZE:.*0]]: memref<1xindex>,
// CHECK-SAME:             %[[DIM_CURSOR:.*1]]: memref<1xindex>,
// CHECK-SAME:             %[[MEM_SIZE:.*2]]: memref<3xindex>,
// CHECK-SAME:             %[[POINTER:.*3]]: memref<?xindex>,
// CHECK-SAME:             %[[INDICES:.*4]]: memref<?xindex>,
// CHECK-SAME:             %[[VALUE:.*5]]: memref<?xf32>,
// CHECK-SAME:             %[[LB:.*6]]: index,
// CHECK-SAME:             %[[UB:.*7]]: index,
// CHECK-SAME:             %[[STEP:.*8]]: index)
// CHECK:          %[[OUT:.*]]:6 = scf.for %[[I:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(
// CHECK-SAME:       %[[SIZE:.*]] = %[[DIM_SIZE]],
// CHECK-SAME:       %[[CUR:.*]] = %[[DIM_CURSOR]],
// CHECK-SAME:       %[[MEM:.*]] = %[[MEM_SIZE]],
// CHECK-SAME:       %[[PTR:.*]] = %[[POINTER]],
// CHECK-SAME:       %[[IDX:.*]] = %[[INDICES]],
// CHECK-SAME:       %[[VAL:.*]] = %[[VALUE]])
// CHECK:            scf.yield %[[SIZE]], %[[CUR]], %[[MEM]], %[[PTR]], %[[IDX]], %[[VAL]] : memref<1xindex>, memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>
// CHECK:          }
// CHECK:          return %[[OUT]]#0, %[[OUT]]#1, %[[OUT]]#2, %[[OUT]]#3, %[[OUT]]#4, %[[OUT]]#5 : memref<1xindex>, memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>
func.func @for(%in: tensor<1024xf32, #SparseVector>,
               %lb: index, %ub: index, %step: index) -> tensor<1024xf32, #SparseVector> {
  %1 = scf.for %i = %lb to %ub step %step iter_args(%vin = %in)
     -> tensor<1024xf32, #SparseVector> {
    scf.yield %vin : tensor<1024xf32, #SparseVector>
  }
  return %1 : tensor<1024xf32, #SparseVector>
}

