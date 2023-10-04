// RUN: mlir-opt %s | mlir-opt | FileCheck %s --check-prefix=CHECK-ROUND
// RUN: mlir-opt %s --post-sparsification-rewrite="enable-runtime-library=true enable-convert=false" \
// RUN: --cse --canonicalize  | FileCheck %s
// RUN: mlir-opt %s --post-sparsification-rewrite="enable-runtime-library=false enable-convert=false" \
// RUN: --cse --canonicalize  | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
#SparseMatrix = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : compressed) }>

//
// roundtrip:
//
// CHECK-ROUND-LABEL: func.func @sparse_expand(
// CHECK-ROUND-SAME:  %[[A:.*]]: tensor<100xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>>
//      CHECK-ROUND:  %[[E:.*]] = tensor.expand_shape %[[A]] {{\[\[}}0, 1]] : tensor<100xf64, #sparse_tensor.encoding<{{{.*}}}>> into tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>>
//      CHECK-ROUND:  return %[[E]] : tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>>
//
// CHECK-LABEL:   func.func @sparse_expand(
// CHECK-SAME:    %[[S:.*0]]:
// CHECK-DAG:     %[[C10:.*]] = arith.constant 10 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[B:.*]] = bufferization.alloc_tensor()
// CHECK:         %[[P0:.*]] = sparse_tensor.positions %[[S]] {level = 0 : index}
// CHECK:         %[[I0:.*]] = sparse_tensor.coordinates %[[S]] {level = 0 : index}
// CHECK:         %[[V:.*]] = sparse_tensor.values %[[S]]
// CHECK:         %[[S0:.*]] = memref.load %[[P0]]{{\[}}%[[C0]]] : memref<?xindex>
// CHECK:         %[[E0:.*]] = memref.load %[[P0]]{{\[}}%[[C1]]] : memref<?xindex>
// CHECK:         %[[RET:.*]] = scf.for %[[I:.*]] = %[[S0]] to %[[E0]] step %[[C1]] iter_args(%[[R:.*]] = %[[B]])
// CHECK:           %[[SI:.*]] = memref.load %[[I0]]{{\[}}%[[I]]] : memref<?xindex>
// CHECK:           %[[SV:.*]] = memref.load %[[V]]{{\[}}%[[I]]] : memref<?xf64>
// CHECK:           %[[DI0:.*]] = arith.divui %[[SI]], %[[C10]] : index
// CHECK:           %[[DI1:.*]] = arith.remui %[[SI]], %[[C10]] : index
// CHECK:           %[[NT:.*]] = sparse_tensor.insert %[[SV]] into %[[R]]{{\[}}%[[DI0]], %[[DI1]]]
// CHECK:           scf.yield %[[NT:.*]]
// CHECK:         }
// CHECK:         %[[NT1:.*]] = sparse_tensor.load %[[RET]] hasInserts
// CHECK-NOT:     sparse_tensor.convert
// CHECK:         return %[[NT1]] : tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>>
//
func.func @sparse_expand(%arg0: tensor<100xf64, #SparseVector>) -> tensor<10x10xf64, #SparseMatrix> {
  %0 = tensor.expand_shape %arg0 [[0, 1]] :
    tensor<100xf64, #SparseVector> into tensor<10x10xf64, #SparseMatrix>
  return %0 : tensor<10x10xf64, #SparseMatrix>
}

//
// roundtrip:
//
// CHECK-ROUND-LABEL: func.func @sparse_collapse(
// CHECK-ROUND-SAME:  %[[A:.*]]: tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<100xf64, #sparse_tensor.encoding<{{{.*}}}>>
//      CHECK-ROUND:  %[[C:.*]] = tensor.collapse_shape %[[A]] {{\[\[}}0, 1]] : tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>> into tensor<100xf64, #sparse_tensor.encoding<{{{.*}}}>>
//      CHECK-ROUND:  return %[[C]] : tensor<100xf64, #sparse_tensor.encoding<{{{.*}}}>>
//
// CHECK-LABEL:   func.func @sparse_collapse(
// CHECK-SAME:    %[[S:.*0]]:
// CHECK-DAG:     %[[C10:.*]] = arith.constant 10 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[B:.*]] = bufferization.alloc_tensor()
// CHECK:         %[[P0:.*]] = sparse_tensor.positions %[[S]] {level = 0 : index}
// CHECK:         %[[I0:.*]] = sparse_tensor.coordinates %[[S]] {level = 0 : index}
// CHECK:         %[[P1:.*]] = sparse_tensor.positions %[[S]] {level = 1 : index}
// CHECK:         %[[I1:.*]] = sparse_tensor.coordinates %[[S]] {level = 1 : index}
// CHECK:         %[[V:.*]] = sparse_tensor.values %[[S]]
// CHECK:         %[[S0:.*]] = memref.load %[[P0]]{{\[}}%[[C0]]] : memref<?xindex>
// CHECK:         %[[E0:.*]] = memref.load %[[P0]]{{\[}}%[[C1]]] : memref<?xindex>
// CHECK:         %[[RET:.*]] = scf.for %[[I:.*]] = %[[S0]] to %[[E0]] step %[[C1]] iter_args(%[[A0:.*]] = %[[B]])
// CHECK:           %[[SI0:.*]] = memref.load %[[I0]]{{\[}}%[[I]]] : memref<?xindex>
// CHECK-DAG:       %[[S1:.*]] = memref.load %[[P1]]{{\[}}%[[I]]] : memref<?xindex>
// CHECK-DAG:       %[[PE1:.*]] = arith.addi %[[I]], %[[C1]] : index
// CHECK:           %[[E1:.*]] = memref.load %[[P1]]{{\[}}%[[PE1]]] : memref<?xindex>
// CHECK:           %[[RET_1:.*]] = scf.for %[[J:.*]] = %[[S1]] to %[[E1]] step %[[C1]] iter_args(%[[A1:.*]] = %[[A0]])
// CHECK:             %[[SI1:.*]] = memref.load %[[I1]]{{\[}}%[[J]]] : memref<?xindex>
// CHECK:             %[[SV:.*]] = memref.load %[[V]]{{\[}}%[[J]]] : memref<?xf64>
// CHECK:             %[[T:.*]] = arith.muli %[[SI0]], %[[C10]] : index
// CHECK:             %[[DI:.*]] = arith.addi %[[T]], %[[SI1]] : index
// CHECK:             %[[R1:.*]] = sparse_tensor.insert %[[SV]] into %[[A1]]{{\[}}%[[DI]]]
// CHECK              scf.yield %[[R1]]
// CHECK            }
// CHECK            scf.yield %[[RET_1]]
// CHECK:         }
// CHECK:        %[[NT1:.*]] = sparse_tensor.load %[[RET]] hasInserts
// CHECK-NOT:    sparse_tensor.convert
// CHECK:        return %[[NT1]] : tensor<100xf64, #sparse_tensor.encoding<{{{.*}}}>>
//
func.func @sparse_collapse(%arg0: tensor<10x10xf64, #SparseMatrix>) -> tensor<100xf64, #SparseVector> {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] :
    tensor<10x10xf64, #SparseMatrix> into tensor<100xf64, #SparseVector>
  return %0 : tensor<100xf64, #SparseVector>
}

//
// roundtrip:
//
// CHECK-ROUND-LABEL: func.func @dynamic_sparse_expand(
// CHECK-ROUND-SAME:  %[[A:.*]]: tensor<?xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<?x10xf64, #sparse_tensor.encoding<{{{.*}}}>>
//      CHECK-ROUND:  %[[E:.*]] = tensor.expand_shape %[[A]] {{\[\[}}0, 1]] : tensor<?xf64, #sparse_tensor.encoding<{{{.*}}}>> into tensor<?x10xf64, #sparse_tensor.encoding<{{{.*}}}>>
//      CHECK-ROUND:  return %[[E]] : tensor<?x10xf64, #sparse_tensor.encoding<{{{.*}}}>>
//
// CHECK-LABEL:   func.func @dynamic_sparse_expand(
// CHECK-SAME:    %[[S:.*0]]:
// CHECK-DAG:     %[[C10:.*]] = arith.constant 10 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[SD:.*]] = tensor.dim %[[S]], %[[C0]]
// CHECK:         %[[DD0:.*]] = arith.divui %[[SD]], %[[C10]] : index
// CHECK:         %[[B:.*]] = bufferization.alloc_tensor(%[[DD0]])
// CHECK:         %[[P0:.*]] = sparse_tensor.positions %[[S]] {level = 0 : index}
// CHECK:         %[[I0:.*]] = sparse_tensor.coordinates %[[S]] {level = 0 : index}
// CHECK:         %[[V:.*]] = sparse_tensor.values %[[S]]
// CHECK:         %[[S0:.*]] = memref.load %[[P0]]{{\[}}%[[C0]]] : memref<?xindex>
// CHECK:         %[[E0:.*]] = memref.load %[[P0]]{{\[}}%[[C1]]] : memref<?xindex>
// CHECK:         %[[RET:.*]] = scf.for %[[I:.*]] = %[[S0]] to %[[E0]] step %[[C1]] iter_args(%[[R:.*]] = %[[B]])
// CHECK:           %[[SI:.*]] = memref.load %[[I0]]{{\[}}%[[I]]] : memref<?xindex>
// CHECK:           %[[SV:.*]] = memref.load %[[V]]{{\[}}%[[I]]] : memref<?xf64>
// CHECK:           %[[T1:.*]] = arith.muli %[[DD0]], %[[C10]] : index
// CHECK:           %[[T2:.*]] = arith.divui %[[T1]], %[[DD0]] : index
// CHECK:           %[[DI0:.*]] = arith.divui %[[SI]], %[[T2]] : index
// CHECK:           %[[T3:.*]] = arith.remui %[[SI]], %[[T2]] : index
// CHECK:           %[[T4:.*]] = arith.divui %[[T2]], %[[C10]] : index
// CHECK:           %[[DI1:.*]] = arith.divui %[[T3]], %[[T4]] : index
// CHECK:           %[[NT:.*]] = sparse_tensor.insert %[[SV]] into %[[R]]{{\[}}%[[DI0]], %[[DI1]]]
// CHECK:           scf.yield %[[NT]]
// CHECK:         }
// CHECK:         %[[NT1:.*]] = sparse_tensor.load %[[RET]] hasInserts
// CHECK-NOT:     sparse_tensor.convert
// CHECK:         return %[[NT1]] : tensor<?x10xf64, #sparse_tensor.encoding<{{{.*}}}>>
//
func.func @dynamic_sparse_expand(%arg0: tensor<?xf64, #SparseVector>) -> tensor<?x10xf64, #SparseMatrix> {
  %0 = tensor.expand_shape %arg0 [[0, 1]] :
    tensor<?xf64, #SparseVector> into tensor<?x10xf64, #SparseMatrix>
  return %0 : tensor<?x10xf64, #SparseMatrix>
}

//
// roundtrip:
//
// CHECK-ROUND-LABEL: func.func @dynamic_sparse_collapse(
// CHECK-ROUND-SAME:  %[[A:.*]]: tensor<10x?xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<?xf64, #sparse_tensor.encoding<{{{.*}}}>>
//      CHECK-ROUND:  %[[C:.*]] = tensor.collapse_shape %[[A]] {{\[\[}}0, 1]] : tensor<10x?xf64, #sparse_tensor.encoding<{{{.*}}}>> into tensor<?xf64, #sparse_tensor.encoding<{{{.*}}}>>
//      CHECK-ROUND:  return %[[C]] : tensor<?xf64, #sparse_tensor.encoding<{{{.*}}}>>
//
// CHECK-LABEL:   func.func @dynamic_sparse_collapse(
// CHECK-SAME:    %[[S:.*0]]:
// CHECK-DAG:     %[[C10:.*]] = arith.constant 10 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[SD1:.*]] = tensor.dim %[[S]], %[[C1]]
// CHECK:         %[[DD0:.*]] = arith.muli %[[SD1]], %[[C10]] : index
// CHECK:         %[[B:.*]] = bufferization.alloc_tensor(%[[DD0]])
// CHECK:         %[[P0:.*]] = sparse_tensor.positions %[[S]] {level = 0 : index}
// CHECK:         %[[I0:.*]] = sparse_tensor.coordinates %[[S]] {level = 0 : index}
// CHECK:         %[[P1:.*]] = sparse_tensor.positions %[[S]] {level = 1 : index}
// CHECK:         %[[I1:.*]] = sparse_tensor.coordinates %[[S]] {level = 1 : index}
// CHECK:         %[[V:.*]] = sparse_tensor.values %[[S]]
// CHECK:         %[[S0:.*]] = memref.load %[[P0]]{{\[}}%[[C0]]] : memref<?xindex>
// CHECK:         %[[E0:.*]] = memref.load %[[P0]]{{\[}}%[[C1]]] : memref<?xindex>
// CHECK:         %[[RET:.*]] = scf.for %[[I:.*]] = %[[S0]] to %[[E0]] step %[[C1]] iter_args(%[[R0:.*]] = %[[B]])
// CHECK:           %[[SI0:.*]] = memref.load %[[I0]]{{\[}}%[[I]]] : memref<?xindex>
// CHECK-DAG:       %[[S1:.*]] = memref.load %[[P1]]{{\[}}%[[I]]] : memref<?xindex>
// CHECK-DAG:       %[[PE1:.*]] = arith.addi %[[I]], %[[C1]] : index
// CHECK:           %[[E1:.*]] = memref.load %[[P1]]{{\[}}%[[PE1]]] : memref<?xindex>
// CHECK:           %[[RET_1:.*]] = scf.for %[[J:.*]] = %[[S1]] to %[[E1]] step %[[C1]] iter_args(%[[R1:.*]] = %[[R0]])
// CHECK:             %[[SI1:.*]] = memref.load %[[I1]]{{\[}}%[[J]]] : memref<?xindex>
// CHECK:             %[[SV:.*]] = memref.load %[[V]]{{\[}}%[[J]]] : memref<?xf64>
// CHECK:             %[[T1:.*]] = arith.divui %[[DD0]], %[[C10]] : index
// CHECK:             %[[T2:.*]] = arith.muli %[[SI0]], %[[T1]] : index
// CHECK:             %[[T3:.*]] = arith.divui %[[T1]], %[[SD1]] : index
// CHECK:             %[[T4:.*]] = arith.muli %[[SI1]], %[[T3]] : index
// CHECK:             %[[DI:.*]] = arith.addi %[[T2]], %[[T4]] : index
// CHECK:             %[[NT:.*]] = sparse_tensor.insert %[[SV]] into %[[R1]]{{\[}}%[[DI]]]
// CHECK              scf.yield %[[NT]]
// CHECK            }
// CHECK            scf.yield %[[RET_1]]
// CHECK:        }
// CHECK:        %[[NT1:.*]] = sparse_tensor.load %[[RET]] hasInserts
// CHECK-NOT:    sparse_tensor.convert
// CHECK:        return %[[NT1]] : tensor<?xf64, #sparse_tensor.encoding<{{{.*}}}>>
//
func.func @dynamic_sparse_collapse(%arg0: tensor<10x?xf64, #SparseMatrix>) -> tensor<?xf64, #SparseVector> {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] :
    tensor<10x?xf64, #SparseMatrix> into tensor<?xf64, #SparseVector>
  return %0 : tensor<?xf64, #SparseVector>
}
