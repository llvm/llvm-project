// RUN: mlir-opt %s --sparse-tensor-codegen=enable-buffer-initialization=true  --canonicalize --cse | FileCheck %s

#SV = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>

// CHECK-LABEL: func @sparse_alloc_sparse_vector(
//  CHECK-SAME: %[[A:.*]]: index) ->
//  CHECK-SAME: memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf64>
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//       CHECK: %[[T0:.*]] = memref.alloc() : memref<1xindex>
//       CHECK: %[[T1:.*]] = memref.alloc() : memref<3xindex>
//       CHECK: %[[T2:.*]] = memref.alloc() : memref<16xindex>
//       CHECK: %[[T3:.*]] = memref.cast %[[T2]] : memref<16xindex> to memref<?xindex>
//       CHECK: linalg.fill ins(%[[C0]] : index) outs(%[[T2]] : memref<16xindex>)
//       CHECK: %[[T4:.*]] = memref.alloc() : memref<16xindex>
//       CHECK: %[[T5:.*]] = memref.cast %[[T4]] : memref<16xindex> to memref<?xindex>
//       CHECK: linalg.fill ins(%[[C0]] : index) outs(%[[T4]] : memref<16xindex>)
//       CHECK: %[[T6:.*]] = memref.alloc() : memref<16xf64>
//       CHECK: %[[T7:.*]] = memref.cast %[[T6]] : memref<16xf64> to memref<?xf64>
//       CHECK: linalg.fill ins(%{{.*}} : f64) outs(%[[T6]] : memref<16xf64>)
//       CHECK: linalg.fill ins(%[[C0]] : index) outs(%[[T1]] : memref<3xindex>)
//       CHECK: memref.store %[[A]], %[[T0]][%[[C0]]] : memref<1xindex>
//       CHECK: %[[P0:.*]] = sparse_tensor.push_back %[[T1]], %[[T3]]
//       CHECK: %[[P1:.*]] = sparse_tensor.push_back %[[T1]], %[[P0]]
//       CHECK: return %[[T0]], %[[T1]], %[[P1]], %[[T5]], %[[T7]] :
func.func @sparse_alloc_sparse_vector(%arg0: index) -> tensor<?xf64, #SV> {
  %0 = bufferization.alloc_tensor(%arg0) : tensor<?xf64, #SV>
  %1 = sparse_tensor.load %0 : tensor<?xf64, #SV>
  return %1 : tensor<?xf64, #SV>
}
