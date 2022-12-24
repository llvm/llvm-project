// RUN: mlir-opt %s --sparse-tensor-codegen=enable-buffer-initialization=true  --canonicalize --cse | FileCheck %s

#SV = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>

// CHECK-LABEL:   func.func @sparse_alloc_sparse_vector(
//  CHECK-SAME:     %[[VAL_0:.*]]: index) -> (memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier
//       CHECK:     %[[VAL_1:.*]] = arith.constant 1 : index
//       CHECK:     %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f64
//       CHECK:     %[[VAL_3:.*]] = arith.constant 0 : index
//       CHECK:     %[[VAL_4:.*]] = memref.alloc() : memref<16xindex>
//       CHECK:     %[[VAL_5:.*]] = memref.cast %[[VAL_4]] : memref<16xindex> to memref<?xindex>
//       CHECK:     linalg.fill ins(%[[VAL_3]] : index) outs(%[[VAL_4]] : memref<16xindex>)
//       CHECK:     %[[VAL_6:.*]] = memref.alloc() : memref<16xindex>
//       CHECK:     %[[VAL_7:.*]] = memref.cast %[[VAL_6]] : memref<16xindex> to memref<?xindex>
//       CHECK:     linalg.fill ins(%[[VAL_3]] : index) outs(%[[VAL_6]] : memref<16xindex>)
//       CHECK:     %[[VAL_8:.*]] = memref.alloc() : memref<16xf64>
//       CHECK:     %[[VAL_9:.*]] = memref.cast %[[VAL_8]] : memref<16xf64> to memref<?xf64>
//       CHECK:     linalg.fill ins(%[[VAL_2]] : f64) outs(%[[VAL_8]] : memref<16xf64>)
//       CHECK:     %[[VAL_10:.*]] = sparse_tensor.storage_specifier.init : !sparse_tensor.storage_specifier
//       CHECK:     %[[VAL_11:.*]] = arith.index_cast %[[VAL_0]] : index to i64
//       CHECK:     %[[VAL_12:.*]] = sparse_tensor.storage_specifier.set %[[VAL_10]]  dim_sz at 0 with %[[VAL_11]] : i64, !sparse_tensor.storage_specifier
//       CHECK:     %[[VAL_13:.*]] = sparse_tensor.storage_specifier.get %[[VAL_12]]  ptr_mem_sz at 0 : !sparse_tensor.storage_specifier
//       CHECK:     %[[VAL_14:.*]] = arith.index_cast %[[VAL_13]] : i64 to index
//       CHECK:     %[[VAL_15:.*]], %[[VAL_16:.*]] = sparse_tensor.push_back %[[VAL_14]], %[[VAL_5]], %[[VAL_3]] : index, memref<?xindex>, index
//       CHECK:     %[[VAL_17:.*]] = arith.index_cast %[[VAL_16]] : index to i64
//       CHECK:     %[[VAL_18:.*]] = sparse_tensor.storage_specifier.set %[[VAL_12]]  ptr_mem_sz at 0 with %[[VAL_17]] : i64, !sparse_tensor.storage_specifier
//       CHECK:     %[[VAL_19:.*]], %[[VAL_20:.*]] = sparse_tensor.push_back %[[VAL_16]], %[[VAL_15]], %[[VAL_3]], %[[VAL_1]] : index, memref<?xindex>, index, index
//       CHECK:     %[[VAL_21:.*]] = arith.index_cast %[[VAL_20]] : index to i64
//       CHECK:     %[[VAL_22:.*]] = sparse_tensor.storage_specifier.set %[[VAL_18]]  ptr_mem_sz at 0 with %[[VAL_21]] : i64, !sparse_tensor.storage_specifier
//       CHECK:     return %[[VAL_19]], %[[VAL_7]], %[[VAL_9]], %[[VAL_22]] : memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier
func.func @sparse_alloc_sparse_vector(%arg0: index) -> tensor<?xf64, #SV> {
  %0 = bufferization.alloc_tensor(%arg0) : tensor<?xf64, #SV>
  %1 = sparse_tensor.load %0 : tensor<?xf64, #SV>
  return %1 : tensor<?xf64, #SV>
}
