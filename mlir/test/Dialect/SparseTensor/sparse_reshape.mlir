// RUN: mlir-opt %s | mlir-opt | FileCheck %s --check-prefix=CHECK-ROUND
// RUN: mlir-opt %s --sparse-tensor-conversion --cse --canonicalize | FileCheck %s --check-prefix=CHECK-CONV
// RUN: mlir-opt %s --post-sparsification-rewrite="enable-runtime-library=false enable-convert=false" \
// RUN: --cse --canonicalize  | FileCheck %s --check-prefix=CHECK-RWT

#SparseVector = #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>
#SparseMatrix = #sparse_tensor.encoding<{ lvlTypes = [ "compressed", "compressed" ] }>

//
// roundtrip:
//
// CHECK-ROUND-LABEL: func.func @sparse_expand(
// CHECK-ROUND-SAME:  %[[A:.*]]: tensor<100xf64, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>>
//      CHECK-ROUND:  %[[E:.*]] = tensor.expand_shape %[[A]] {{\[\[}}0, 1]] : tensor<100xf64, #sparse_tensor.encoding<{{{.*}}}>> into tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>>
//      CHECK-ROUND:  return %[[E]] : tensor<10x10xf64, #sparse_tensor.encoding<{{{.*}}}>>
//
// conversion:
//
// CHECK-CONV-LABEL: func.func @sparse_expand(
// CHECK-CONV-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-CONV-DAG:  %[[C1:.*]] = arith.constant 1 : index
// CHECK-CONV-DAG:  %[[C10:.*]] = arith.constant 10 : index
// CHECK-CONV-DAG:  call @newSparseTensor
// CHECK-CONV-DAG:  call @newSparseTensor
// CHECK-CONV:      scf.while : () -> () {
// CHECK-CONV:        call @getNextF64
// CHECK-CONV:        scf.condition
// CHECK-CONV:      } do {
// CHECK-CONV:        %[[X:.*]] = memref.load %{{.*}}[%[[C0]]] : memref<1xindex>
// CHECK-CONV:        %[[D:.*]] = arith.divui %[[X]], %[[C10]] : index
// CHECK-CONV:        %[[R:.*]] = arith.remui %[[X]], %[[C10]] : index
// CHECK-CONV:        memref.store %[[D]], %{{.*}}[%[[C0]]] : memref<2xindex>
// CHECK-CONV:        memref.store %[[R]], %{{.*}}[%[[C1]]] : memref<2xindex>
// CHECK-CONV:        call @addEltF64
// CHECK-CONV:        scf.yield
// CHECK-CONV:      }
// CHECK-CONV:      %[[N:.*]] = call @newSparseTensor
// CHECK-CONV:      call @delSparseTensorCOOF64
// CHECK-CONV:      call @delSparseTensorIteratorF64
// CHECK-CONV:      return %[[N]] : !llvm.ptr<i8>
//
// rewrite for codegen:
//
// CHECK-RWT-LABEL:   func.func @sparse_expand(
// CHECK-RWT-SAME:    %[[S:.*]]:
// CHECK-RWT-DAG:     %[[C10:.*]] = arith.constant 10 : index
// CHECK-RWT-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-RWT-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-RWT:         %[[B:.*]] = bufferization.alloc_tensor()
// CHECK-RWT:         %[[P0:.*]] = sparse_tensor.positions %[[S]] {level = 0 : index}
// CHECK-RWT:         %[[I0:.*]] = sparse_tensor.coordinates %[[S]] {level = 0 : index}
// CHECK-RWT:         %[[V:.*]] = sparse_tensor.values %[[S]]
// CHECK-RWT:         %[[S0:.*]] = memref.load %[[P0]]{{\[}}%[[C0]]] : memref<?xindex>
// CHECK-RWT:         %[[E0:.*]] = memref.load %[[P0]]{{\[}}%[[C1]]] : memref<?xindex>
// CHECK-RWT:         %[[RET:.*]] = scf.for %[[I:.*]] = %[[S0]] to %[[E0]] step %[[C1]] iter_args(%[[R:.*]] = %[[B]])
// CHECK-RWT:           %[[SI:.*]] = memref.load %[[I0]]{{\[}}%[[I]]] : memref<?xindex>
// CHECK-RWT:           %[[SV:.*]] = memref.load %[[V]]{{\[}}%[[I]]] : memref<?xf64>
// CHECK-RWT:           %[[DI0:.*]] = arith.divui %[[SI]], %[[C10]] : index
// CHECK-RWT:           %[[DI1:.*]] = arith.remui %[[SI]], %[[C10]] : index
// CHECK-RWT:           %[[NT:.*]] = sparse_tensor.insert %[[SV]] into %[[R]]{{\[}}%[[DI0]], %[[DI1]]]
// CHECK-RWT:           scf.yield %[[NT:.*]]
// CHECK-RWT:         }
// CHECK-RWT:         %[[NT1:.*]] = sparse_tensor.load %[[RET]] hasInserts
// CHECK-RWT-NOT:     sparse_tensor.convert
// CHECK-RWT:         return %[[NT1]] : tensor<10x10xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed", "compressed" ] }>>
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
// conversion:
//
// CHECK-CONV-LABEL: func.func @sparse_collapse(
// CHECK-CONV-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-CONV-DAG:  %[[C1:.*]] = arith.constant 1 : index
// CHECK-CONV-DAG:  %[[C10:.*]] = arith.constant 10 : index
// CHECK-CONV-DAG:  call @newSparseTensor
// CHECK-CONV-DAG:  call @newSparseTensor
// CHECK-CONV:      scf.while : () -> () {
// CHECK-CONV:        call @getNextF64
// CHECK-CONV:        scf.condition
// CHECK-CONV:      } do {
// CHECK-CONV:        %[[X:.*]] = memref.load %{{.*}}[%[[C0]]] : memref<2xindex>
// CHECK-CONV:        %[[Y:.*]] = memref.load %{{.*}}[%[[C1]]] : memref<2xindex>
// CHECK-CONV:        %[[M:.*]] = arith.muli %[[X]], %[[C10]] : index
// CHECK-CONV:        %[[A:.*]] = arith.addi %[[M]], %[[Y]] : index
// CHECK-CONV:        memref.store %[[A]], %{{.*}}[%[[C0]]] : memref<1xindex>
// CHECK-CONV:        call @addEltF64
// CHECK-CONV:        scf.yield
// CHECK-CONV:      }
// CHECK-CONV:      %[[N:.*]] = call @newSparseTensor
// CHECK-CONV:      call @delSparseTensorCOOF64
// CHECK-CONV:      call @delSparseTensorIteratorF64
// CHECK-CONV:      return %[[N]] : !llvm.ptr<i8>
//
// rewrite for codegen:
//
// CHECK-RWT-LABEL:   func.func @sparse_collapse(
// CHECK-RWT-SAME:    %[[S:.*]]:
// CHECK-RWT-DAG:     %[[C10:.*]] = arith.constant 10 : index
// CHECK-RWT-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-RWT-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-RWT:         %[[B:.*]] = bufferization.alloc_tensor()
// CHECK-RWT:         %[[P0:.*]] = sparse_tensor.positions %[[S]] {level = 0 : index}
// CHECK-RWT:         %[[I0:.*]] = sparse_tensor.coordinates %[[S]] {level = 0 : index}
// CHECK-RWT:         %[[P1:.*]] = sparse_tensor.positions %[[S]] {level = 1 : index}
// CHECK-RWT:         %[[I1:.*]] = sparse_tensor.coordinates %[[S]] {level = 1 : index}
// CHECK-RWT:         %[[V:.*]] = sparse_tensor.values %[[S]]
// CHECK-RWT:         %[[S0:.*]] = memref.load %[[P0]]{{\[}}%[[C0]]] : memref<?xindex>
// CHECK-RWT:         %[[E0:.*]] = memref.load %[[P0]]{{\[}}%[[C1]]] : memref<?xindex>
// CHECK-RWT:         %[[RET:.*]] = scf.for %[[I:.*]] = %[[S0]] to %[[E0]] step %[[C1]] iter_args(%[[A0:.*]] = %[[B]])
// CHECK-RWT:           %[[SI0:.*]] = memref.load %[[I0]]{{\[}}%[[I]]] : memref<?xindex>
// CHECK-RWT-DAG:       %[[S1:.*]] = memref.load %[[P1]]{{\[}}%[[I]]] : memref<?xindex>
// CHECK-RWT-DAG:       %[[PE1:.*]] = arith.addi %[[I]], %[[C1]] : index
// CHECK-RWT:           %[[E1:.*]] = memref.load %[[P1]]{{\[}}%[[PE1]]] : memref<?xindex>
// CHECK-RWT:           %[[RET_1:.*]] = scf.for %[[J:.*]] = %[[S1]] to %[[E1]] step %[[C1]] iter_args(%[[A1:.*]] = %[[A0]])
// CHECK-RWT:             %[[SI1:.*]] = memref.load %[[I1]]{{\[}}%[[J]]] : memref<?xindex>
// CHECK-RWT:             %[[SV:.*]] = memref.load %[[V]]{{\[}}%[[J]]] : memref<?xf64>
// CHECK-RWT:             %[[T:.*]] = arith.muli %[[SI0]], %[[C10]] : index
// CHECK-RWT:             %[[DI:.*]] = arith.addi %[[T]], %[[SI1]] : index
// CHECK-RWT:             %[[R1:.*]] = sparse_tensor.insert %[[SV]] into %[[A1]]{{\[}}%[[DI]]]
// CHECK-RWT              scf.yield %[[R1]]
// CHECK-RWT            }
// CHECK-RWT            scf.yield %[[RET_1]]
// CHECK-RWT:         }
// CHECK-RWT:        %[[NT1:.*]] = sparse_tensor.load %[[RET]] hasInserts
// CHECK-RWT-NOT:    sparse_tensor.convert
// CHECK-RWT:        return %[[NT1]] : tensor<100xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>
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
// conversion:
//
// CHECK-CONV-LABEL: func.func @dynamic_sparse_expand(
// CHECK-CONV-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-CONV-DAG:  %[[C1:.*]] = arith.constant 1 : index
// CHECK-CONV-DAG:  %[[C10:.*]] = arith.constant 10 : index
// CHECK-CONV-DAG:  %[[D1:.*]] = arith.divui %{{.*}}, %[[C10]] : index
// CHECK-CONV-DAG:  call @newSparseTensor
// CHECK-CONV-DAG:  call @newSparseTensor
// CHECK-CONV:      scf.while : () -> () {
// CHECK-CONV:        call @getNextF64
// CHECK-CONV:        scf.condition
// CHECK-CONV:      } do {
// CHECK-CONV:        %[[L:.*]] = memref.load %{{.*}}[%[[C0]]] : memref<1xindex>
// CHECK-CONV:        %[[M:.*]] = arith.muli %[[D1]], %[[C10]] : index
// CHECK-CONV:        %[[D2:.*]] = arith.divui %[[M]], %[[D1]] : index
// CHECK-CONV:        %[[D3:.*]] = arith.divui %[[L]], %[[D2]] : index
// CHECK-CONV:        %[[R:.*]] = arith.remui %[[L]], %[[D2]] : index
// CHECK-CONV:        %[[D4:.*]] = arith.divui %[[D2]], %[[C10]] : index
// CHECK-CONV:        %[[D5:.*]] = arith.divui %[[R]], %[[D4]] : index
// CHECK-CONV:        memref.store %[[D3]], %{{.*}}[%[[C0]]] : memref<2xindex>
// CHECK-CONV:        memref.store %[[D5]], %{{.*}}[%[[C1]]] : memref<2xindex>
// CHECK-CONV:        call @addEltF64
// CHECK-CONV:        scf.yield
// CHECK-CONV:      }
// CHECK-CONV:      %[[N:.*]] = call @newSparseTensor
// CHECK-CONV:      call @delSparseTensorCOOF64
// CHECK-CONV:      call @delSparseTensorIteratorF64
// CHECK-CONV:      return %[[N]] : !llvm.ptr<i8>
//
// rewrite for codegen:
//
// CHECK-RWT-LABEL:   func.func @dynamic_sparse_expand(
// CHECK-RWT-SAME:    %[[S:.*]]:
// CHECK-RWT-DAG:     %[[C10:.*]] = arith.constant 10 : index
// CHECK-RWT-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-RWT-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-RWT:         %[[SD:.*]] = tensor.dim %[[S]], %[[C0]]
// CHECK-RWT:         %[[DD0:.*]] = arith.divui %[[SD]], %[[C10]] : index
// CHECK-RWT:         %[[B:.*]] = bufferization.alloc_tensor(%[[DD0]])
// CHECK-RWT:         %[[P0:.*]] = sparse_tensor.positions %[[S]] {level = 0 : index}
// CHECK-RWT:         %[[I0:.*]] = sparse_tensor.coordinates %[[S]] {level = 0 : index}
// CHECK-RWT:         %[[V:.*]] = sparse_tensor.values %[[S]]
// CHECK-RWT:         %[[S0:.*]] = memref.load %[[P0]]{{\[}}%[[C0]]] : memref<?xindex>
// CHECK-RWT:         %[[E0:.*]] = memref.load %[[P0]]{{\[}}%[[C1]]] : memref<?xindex>
// CHECK-RWT:         %[[RET:.*]] = scf.for %[[I:.*]] = %[[S0]] to %[[E0]] step %[[C1]] iter_args(%[[R:.*]] = %[[B]])
// CHECK-RWT:           %[[SI:.*]] = memref.load %[[I0]]{{\[}}%[[I]]] : memref<?xindex>
// CHECK-RWT:           %[[SV:.*]] = memref.load %[[V]]{{\[}}%[[I]]] : memref<?xf64>
// CHECK-RWT:           %[[T1:.*]] = arith.muli %[[DD0]], %[[C10]] : index
// CHECK-RWT:           %[[T2:.*]] = arith.divui %[[T1]], %[[DD0]] : index
// CHECK-RWT:           %[[DI0:.*]] = arith.divui %[[SI]], %[[T2]] : index
// CHECK-RWT:           %[[T3:.*]] = arith.remui %[[SI]], %[[T2]] : index
// CHECK-RWT:           %[[T4:.*]] = arith.divui %[[T2]], %[[C10]] : index
// CHECK-RWT:           %[[DI1:.*]] = arith.divui %[[T3]], %[[T4]] : index
// CHECK-RWT:           %[[NT:.*]] = sparse_tensor.insert %[[SV]] into %[[R]]{{\[}}%[[DI0]], %[[DI1]]]
// CHECK-RWT:           scf.yield %[[NT]]
// CHECK-RWT:         }
// CHECK-RWT:         %[[NT1:.*]] = sparse_tensor.load %[[RET]] hasInserts
// CHECK-RWT-NOT:     sparse_tensor.convert
// CHECK-RWT:         return %[[NT1]] : tensor<?x10xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed", "compressed" ] }>>
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
// conversion:
//
// CHECK-CONV-LABEL: func.func @dynamic_sparse_collapse(
// CHECK-CONV-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-CONV-DAG:  %[[C1:.*]] = arith.constant 1 : index
// CHECK-CONV-DAG:  %[[C10:.*]] = arith.constant 10 : index
// CHECK-CONV-DAG:  %[[M1:.*]] = arith.muli %{{.*}}, %[[C10]] : index
// CHECK-CONV-DAG:  call @newSparseTensor
// CHECK-CONV-DAG:  call @newSparseTensor
// CHECK-CONV:      scf.while : () -> () {
// CHECK-CONV:        call @getNextF64
// CHECK-CONV:        scf.condition
// CHECK-CONV:      } do {
// CHECK-CONV:        %[[X:.*]] = memref.load %{{.*}}[%[[C0]]] : memref<2xindex>
// CHECK-CONV:        %[[Y:.*]] = memref.load %{{.*}}[%[[C1]]] : memref<2xindex>
// CHECK-CONV:        %[[D1:.*]] = arith.divui %[[M1]], %[[C10]] : index
// CHECK-CONV:        %[[M2:.*]] = arith.muli %[[X]], %[[D1]] : index
// CHECK-CONV:        %[[D2:.*]] = arith.divui %[[D1]], %{{.*}} : index
// CHECK-CONV:        %[[M3:.*]] = arith.muli %[[Y]], %[[D2]] : index
// CHECK-CONV:        %[[A:.*]] = arith.addi %[[M2]], %[[M3]] : index
// CHECK-CONV:        memref.store %[[A]], %{{.*}}[%[[C0]]] : memref<1xindex>
// CHECK-CONV:        call @addEltF64
// CHECK-CONV:        scf.yield
// CHECK-CONV:      }
// CHECK-CONV:      %[[N:.*]] = call @newSparseTensor
// CHECK-CONV:      call @delSparseTensorCOOF64
// CHECK-CONV:      call @delSparseTensorIteratorF64
// CHECK-CONV:      return %[[N]] : !llvm.ptr<i8>
//
// rewrite for codegen:
//
// CHECK-RWT-LABEL:   func.func @dynamic_sparse_collapse(
// CHECK-RWT-SAME:    %[[S:.*]]:
// CHECK-RWT-DAG:     %[[C10:.*]] = arith.constant 10 : index
// CHECK-RWT-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-RWT-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-RWT:         %[[SD1:.*]] = tensor.dim %[[S]], %[[C1]]
// CHECK-RWT:         %[[DD0:.*]] = arith.muli %[[SD1]], %[[C10]] : index
// CHECK-RWT:         %[[B:.*]] = bufferization.alloc_tensor(%[[DD0]])
// CHECK-RWT:         %[[P0:.*]] = sparse_tensor.positions %[[S]] {level = 0 : index}
// CHECK-RWT:         %[[I0:.*]] = sparse_tensor.coordinates %[[S]] {level = 0 : index}
// CHECK-RWT:         %[[P1:.*]] = sparse_tensor.positions %[[S]] {level = 1 : index}
// CHECK-RWT:         %[[I1:.*]] = sparse_tensor.coordinates %[[S]] {level = 1 : index}
// CHECK-RWT:         %[[V:.*]] = sparse_tensor.values %[[S]]
// CHECK-RWT:         %[[S0:.*]] = memref.load %[[P0]]{{\[}}%[[C0]]] : memref<?xindex>
// CHECK-RWT:         %[[E0:.*]] = memref.load %[[P0]]{{\[}}%[[C1]]] : memref<?xindex>
// CHECK-RWT:         %[[RET:.*]] = scf.for %[[I:.*]] = %[[S0]] to %[[E0]] step %[[C1]] iter_args(%[[R0:.*]] = %[[B]])
// CHECK-RWT:           %[[SI0:.*]] = memref.load %[[I0]]{{\[}}%[[I]]] : memref<?xindex>
// CHECK-RWT-DAG:       %[[S1:.*]] = memref.load %[[P1]]{{\[}}%[[I]]] : memref<?xindex>
// CHECK-RWT-DAG:       %[[PE1:.*]] = arith.addi %[[I]], %[[C1]] : index
// CHECK-RWT:           %[[E1:.*]] = memref.load %[[P1]]{{\[}}%[[PE1]]] : memref<?xindex>
// CHECK-RWT:           %[[RET_1:.*]] = scf.for %[[J:.*]] = %[[S1]] to %[[E1]] step %[[C1]] iter_args(%[[R1:.*]] = %[[R0]])
// CHECK-RWT:             %[[SI1:.*]] = memref.load %[[I1]]{{\[}}%[[J]]] : memref<?xindex>
// CHECK-RWT:             %[[SV:.*]] = memref.load %[[V]]{{\[}}%[[J]]] : memref<?xf64>
// CHECK-RWT:             %[[T1:.*]] = arith.divui %[[DD0]], %[[C10]] : index
// CHECK-RWT:             %[[T2:.*]] = arith.muli %[[SI0]], %[[T1]] : index
// CHECK-RWT:             %[[T3:.*]] = arith.divui %[[T1]], %[[SD1]] : index
// CHECK-RWT:             %[[T4:.*]] = arith.muli %[[SI1]], %[[T3]] : index
// CHECK-RWT:             %[[DI:.*]] = arith.addi %[[T2]], %[[T4]] : index
// CHECK-RWT:             %[[NT:.*]] = sparse_tensor.insert %[[SV]] into %[[R1]]{{\[}}%[[DI]]]
// CHECK-RWT              scf.yield %[[NT]]
// CHECK-RWT            }
// CHECK-RWT            scf.yield %[[RET_1]]
// CHECK-RWT:        }
// CHECK-RWT:        %[[NT1:.*]] = sparse_tensor.load %[[RET]] hasInserts
// CHECK-RWT-NOT:    sparse_tensor.convert
// CHECK-RWT:        return %[[NT1]] : tensor<?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>>
//
func.func @dynamic_sparse_collapse(%arg0: tensor<10x?xf64, #SparseMatrix>) -> tensor<?xf64, #SparseVector> {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] :
    tensor<10x?xf64, #SparseMatrix> into tensor<?xf64, #SparseVector>
  return %0 : tensor<?xf64, #SparseVector>
}
