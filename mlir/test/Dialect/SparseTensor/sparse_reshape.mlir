// RUN: mlir-opt %s | mlir-opt | FileCheck %s --check-prefix=CHECK-ROUND
// RUN: mlir-opt %s --sparse-tensor-conversion --cse --canonicalize | FileCheck %s --check-prefix=CHECK-CONV

#SparseVector = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>
#SparseMatrix = #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>

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
// CHECK-CONV:      call @delSparseTensorCOOF64
// CHECK-CONV:      return %[[N]] : !llvm.ptr<i8>
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
// CHECK-CONV:      call @delSparseTensorCOOF64
// CHECK-CONV:      return %[[N]] : !llvm.ptr<i8>
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
// CHECK-CONV:      call @delSparseTensorCOOF64
// CHECK-CONV:      return %[[N]] : !llvm.ptr<i8>
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
// CHECK-CONV:      call @delSparseTensorCOOF64
// CHECK-CONV:      return %[[N]] : !llvm.ptr<i8>
//
func.func @dynamic_sparse_collapse(%arg0: tensor<10x?xf64, #SparseMatrix>) -> tensor<?xf64, #SparseVector> {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] :
    tensor<10x?xf64, #SparseMatrix> into tensor<?xf64, #SparseVector>
  return %0 : tensor<?xf64, #SparseVector>
}
