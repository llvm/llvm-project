// RUN: mlir-opt %s -sparse-tensor-codegen -cse | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>
// CHECK-LABEL:  func @for(
// CHECK-SAME:             %[[DIM_SIZE:.*0]]: memref<1xindex>,
// CHECK-SAME:             %[[MEM_SIZE:.*1]]: memref<3xindex>,
// CHECK-SAME:             %[[POINTER:.*2]]: memref<?xindex>,
// CHECK-SAME:             %[[INDICES:.*3]]: memref<?xindex>,
// CHECK-SAME:             %[[VALUE:.*4]]: memref<?xf32>,
// CHECK-SAME:             %[[TMP_arg5:.*5]]: index,
// CHECK-SAME:             %[[TMP_arg6:.*6]]: index,
// CHECK-SAME:             %[[TMP_arg7:.*7]]: index
// CHECK:          %[[TMP_0:.*]]:5 = scf.for %[[TMP_arg8:.*]] = %[[TMP_arg5]] to %[[TMP_arg6]] step %[[TMP_arg7]] iter_args(
// CHECK-SAME:       %[[TMP_arg9:.*]] = %[[DIM_SIZE]],
// CHECK-SAME:       %[[TMP_arg10:.*]] = %[[MEM_SIZE]],
// CHECK-SAME:       %[[TMP_arg11:.*]] = %[[POINTER]],
// CHECK-SAME:       %[[TMP_arg12:.*]] = %[[INDICES]],
// CHECK-SAME:       %[[TMP_arg13:.*]] = %[[VALUE]]) 
// CHECK:              scf.yield %[[TMP_arg9]], %[[TMP_arg10]], %[[TMP_arg11]], %[[TMP_arg12]], %[[TMP_arg13]] : memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>
// CHECK:            }
// CHECK:          return %[[TMP_0]]#0, %[[TMP_0]]#1, %[[TMP_0]]#2, %[[TMP_0]]#3, %[[TMP_0]]#4 : memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>
func.func @for(%in: tensor<1024xf32, #SparseVector>,
               %lb: index, %ub: index, %step: index) -> tensor<1024xf32, #SparseVector> {
  %1 = scf.for %i = %lb to %ub step %step iter_args(%vin = %in)
     -> tensor<1024xf32, #SparseVector> {
    scf.yield %vin : tensor<1024xf32, #SparseVector>
  }
  return %1 : tensor<1024xf32, #SparseVector>
}
