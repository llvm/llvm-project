// RUN: mlir-opt %s --sparsification --canonicalize --cse | FileCheck %s

#DCSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : compressed) }>
#SparseTensor = #sparse_tensor.encoding<{ lvlTypes = [ "compressed", "compressed", "compressed" ] }>

#trait = {
  indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d2)>,
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  ],
  iterator_types = ["parallel", "parallel", "parallel"]
}

// CHECK-LABEL:  @main(
//  CHECK-SAME:  %[[TMP_arg0:.*]]: tensor<4x5xi32,
//   CHECK-DAG:  %[[TMP_c3:.*]] = arith.constant 3 : index
//   CHECK-DAG:  %[[TMP_c0:.*]] = arith.constant 0 : index
//   CHECK-DAG:  %[[TMP_c1:.*]] = arith.constant 1 : index
//       CHECK:  %[[TMP_0:.*]] = bufferization.alloc_tensor()
//       CHECK:  %[[TMP_1:.*]] = sparse_tensor.positions %[[TMP_arg0]] {level = 0 : index}
//       CHECK:  %[[TMP_2:.*]] = sparse_tensor.coordinates %[[TMP_arg0]] {level = 0 : index}
//       CHECK:  %[[TMP_3:.*]] = sparse_tensor.positions %[[TMP_arg0]] {level = 1 : index}
//       CHECK:  %[[TMP_4:.*]] = sparse_tensor.coordinates %[[TMP_arg0]] {level = 1 : index}
//       CHECK:  %[[TMP_5:.*]] = sparse_tensor.values %[[TMP_arg0]]
//       CHECK:  %[[TMP_6:.*]] = memref.load %[[TMP_1]][%[[TMP_c0]]] : memref<?xindex>
//       CHECK:  %[[TMP_7:.*]] = memref.load %[[TMP_1]][%[[TMP_c1]]] : memref<?xindex>
//       CHECK:  %[[T:.*]] = scf.for %[[TMP_arg1:.*]] = %[[TMP_6]] to %[[TMP_7]] step %[[TMP_c1]] {{.*}} {
//       CHECK:    %[[TMP_9:.*]] = memref.load %[[TMP_2]][%[[TMP_arg1]]] : memref<?xindex>
//       CHECK:    %[[L1:.*]] = scf.for %[[TMP_arg2:.*]] = %[[TMP_c0]] to %[[TMP_c3]] step %[[TMP_c1]] {{.*}} {
//       CHECK:      %[[TMP_10:.*]] = memref.load %[[TMP_3]][%[[TMP_arg1]]] : memref<?xindex>
//       CHECK:      %[[TMP_11:.*]] = arith.addi %[[TMP_arg1]], %[[TMP_c1]] : index
//       CHECK:      %[[TMP_12:.*]] = memref.load %[[TMP_3]][%[[TMP_11]]] : memref<?xindex>
//       CHECK:      %[[L2:.*]] = scf.for %[[TMP_arg3:.*]] = %[[TMP_10]] to %[[TMP_12]] step %[[TMP_c1]] {{.*}} {
//       CHECK:        %[[TMP_13:.*]] = memref.load %[[TMP_4]][%[[TMP_arg3]]] : memref<?xindex>
//       CHECK:        %[[TMP_14:.*]] = memref.load %[[TMP_5]][%[[TMP_arg3]]] : memref<?xi32>
//       CHECK:        %[[Y:.*]] = sparse_tensor.insert %[[TMP_14]] into %{{.*}}[%[[TMP_9]], %[[TMP_arg2]], %[[TMP_13]]]
//       CHECK:        scf.yield %[[Y]]
//       CHECK:      }
//       CHECK:      scf.yield %[[L2]]
//       CHECK:    }
//       CHECK:    scf.yield %[[L1]]
//       CHECK:  }
//       CHECK:  %[[TMP_8:.*]] = sparse_tensor.load %[[T]] hasInserts
//       CHECK:  return %[[TMP_8]]
module @func_sparse {
  func.func public @main(%arg0: tensor<4x5xi32, #DCSR>) -> tensor<4x3x5xi32, #SparseTensor> {
    %0 = bufferization.alloc_tensor() : tensor<4x3x5xi32, #SparseTensor>
    %1 = linalg.generic #trait
    ins(%arg0 : tensor<4x5xi32, #DCSR>) outs(%0 : tensor<4x3x5xi32, #SparseTensor>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<4x3x5xi32, #SparseTensor>
    return %1 : tensor<4x3x5xi32, #SparseTensor>
  }
}
