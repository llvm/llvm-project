// RUN: mlir-opt %s --sparse-tensor-conversion --canonicalize --cse | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{
  dimLevelType = ["compressed"]
}>

#SparseVector64 = #sparse_tensor.encoding<{
  dimLevelType = ["compressed"],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#SparseVector32 = #sparse_tensor.encoding<{
  dimLevelType = ["compressed"],
  pointerBitWidth = 32,
  indexBitWidth = 32
}>

#SparseMatrix = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "compressed"]
}>

#SparseTensor = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "compressed", "compressed"],
  dimOrdering = affine_map<(i,j,k) -> (k,i,j)>
}>

// CHECK-LABEL: func @sparse_dim1d(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[C:.*]] = arith.constant 0 : index
//       CHECK: %[[D:.*]] = call @sparseDimSize(%[[A]], %[[C]])
//       CHECK: return %[[D]] : index
func @sparse_dim1d(%arg0: tensor<?xf64, #SparseVector>) -> index {
  %c = arith.constant 0 : index
  %0 = tensor.dim %arg0, %c : tensor<?xf64, #SparseVector>
  return %0 : index
}

// CHECK-LABEL: func @sparse_dim3d(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[C:.*]] = arith.constant 2 : index
//       CHECK: %[[D:.*]] = call @sparseDimSize(%[[A]], %[[C]])
//       CHECK: return %[[D]] : index
func @sparse_dim3d(%arg0: tensor<?x?x?xf64, #SparseTensor>) -> index {
  // Querying for dimension 1 in the tensor type needs to be
  // permuted into querying for dimension 2 in the stored sparse
  // tensor scheme, since the latter honors the dimOrdering.
  %c = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c : tensor<?x?x?xf64, #SparseTensor>
  return %0 : index
}

// CHECK-LABEL: func @sparse_dim3d_const(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[C:.*]] = arith.constant 20 : index
//       CHECK: return %[[C]] : index
func @sparse_dim3d_const(%arg0: tensor<10x20x30xf64, #SparseTensor>) -> index {
  // Querying for dimension 1 in the tensor type can be directly
  // folded into the right value (even though it corresponds
  // to dimension 2 in the stored sparse tensor scheme).
  %c = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c : tensor<10x20x30xf64, #SparseTensor>
  return %0 : index
}

// CHECK-LABEL: func @sparse_new1d(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>) -> !llvm.ptr<i8>
//   CHECK-DAG: %[[P:.*]] = memref.alloca() : memref<1xi8>
//   CHECK-DAG: %[[Q:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-DAG: %[[R:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-DAG: %[[X:.*]] = memref.cast %[[P]] : memref<1xi8> to memref<?xi8>
//   CHECK-DAG: %[[Y:.*]] = memref.cast %[[Q]] : memref<1xindex> to memref<?xindex>
//   CHECK-DAG: %[[Z:.*]] = memref.cast %[[R]] : memref<1xindex> to memref<?xindex>
//       CHECK: %[[T:.*]] = call @newSparseTensor(%[[X]], %[[Y]], %[[Z]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[A]])
//       CHECK: return %[[T]] : !llvm.ptr<i8>
func @sparse_new1d(%arg0: !llvm.ptr<i8>) -> tensor<128xf64, #SparseVector> {
  %0 = sparse_tensor.new %arg0 : !llvm.ptr<i8> to tensor<128xf64, #SparseVector>
  return %0 : tensor<128xf64, #SparseVector>
}

// CHECK-LABEL: func @sparse_new2d(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>) -> !llvm.ptr<i8>
//   CHECK-DAG: %[[P:.*]] = memref.alloca() : memref<2xi8>
//   CHECK-DAG: %[[Q:.*]] = memref.alloca() : memref<2xindex>
//   CHECK-DAG: %[[R:.*]] = memref.alloca() : memref<2xindex>
//   CHECK-DAG: %[[X:.*]] = memref.cast %[[P]] : memref<2xi8> to memref<?xi8>
//   CHECK-DAG: %[[Y:.*]] = memref.cast %[[Q]] : memref<2xindex> to memref<?xindex>
//   CHECK-DAG: %[[Z:.*]] = memref.cast %[[R]] : memref<2xindex> to memref<?xindex>
//       CHECK: %[[T:.*]] = call @newSparseTensor(%[[X]], %[[Y]], %[[Z]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[A]])
//       CHECK: return %[[T]] : !llvm.ptr<i8>
func @sparse_new2d(%arg0: !llvm.ptr<i8>) -> tensor<?x?xf32, #SparseMatrix> {
  %0 = sparse_tensor.new %arg0 : !llvm.ptr<i8> to tensor<?x?xf32, #SparseMatrix>
  return %0 : tensor<?x?xf32, #SparseMatrix>
}

// CHECK-LABEL: func @sparse_new3d(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>) -> !llvm.ptr<i8>
//   CHECK-DAG: %[[P:.*]] = memref.alloca() : memref<3xi8>
//   CHECK-DAG: %[[Q:.*]] = memref.alloca() : memref<3xindex>
//   CHECK-DAG: %[[R:.*]] = memref.alloca() : memref<3xindex>
//   CHECK-DAG: %[[X:.*]] = memref.cast %[[P]] : memref<3xi8> to memref<?xi8>
//   CHECK-DAG: %[[Y:.*]] = memref.cast %[[Q]] : memref<3xindex> to memref<?xindex>
//   CHECK-DAG: %[[Z:.*]] = memref.cast %[[R]] : memref<3xindex> to memref<?xindex>
//       CHECK: %[[T:.*]] = call @newSparseTensor(%[[X]], %[[Y]], %[[Z]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[A]])
//       CHECK: return %[[T]] : !llvm.ptr<i8>
func @sparse_new3d(%arg0: !llvm.ptr<i8>) -> tensor<?x?x?xf32, #SparseTensor> {
  %0 = sparse_tensor.new %arg0 : !llvm.ptr<i8> to tensor<?x?x?xf32, #SparseTensor>
  return %0 : tensor<?x?x?xf32, #SparseTensor>
}

// CHECK-LABEL: func @sparse_init(
//  CHECK-SAME: %[[I:.*]]: index,
//  CHECK-SAME: %[[J:.*]]: index) -> !llvm.ptr<i8>
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[P:.*]] = memref.alloca() : memref<2xi8>
//   CHECK-DAG: %[[Q:.*]] = memref.alloca() : memref<2xindex>
//   CHECK-DAG: %[[R:.*]] = memref.alloca() : memref<2xindex>
//   CHECK-DAG: %[[X:.*]] = memref.cast %[[P]] : memref<2xi8> to memref<?xi8>
//   CHECK-DAG: %[[Y:.*]] = memref.cast %[[Q]] : memref<2xindex> to memref<?xindex>
//   CHECK-DAG: %[[Z:.*]] = memref.cast %[[R]] : memref<2xindex> to memref<?xindex>
//   CHECK-DAG: memref.store %[[I]], %[[Q]][%[[C0]]] : memref<2xindex>
//   CHECK-DAG: memref.store %[[J]], %[[Q]][%[[C1]]] : memref<2xindex>
//       CHECK: %[[NP:.*]] = llvm.mlir.null : !llvm.ptr<i8>
//       CHECK: %[[T:.*]] = call @newSparseTensor(%[[X]], %[[Y]], %[[Z]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[NP]])
//       CHECK: return %[[T]] : !llvm.ptr<i8>
func @sparse_init(%arg0: index, %arg1: index) -> tensor<?x?xf64, #SparseMatrix> {
  %0 = sparse_tensor.init [%arg0, %arg1] : tensor<?x?xf64, #SparseMatrix>
  return %0 : tensor<?x?xf64, #SparseMatrix>
}

// CHECK-LABEL: func @sparse_release(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: call @delSparseTensor(%[[A]]) : (!llvm.ptr<i8>) -> ()
//       CHECK: return
func @sparse_release(%arg0: tensor<128xf64, #SparseVector>) {
  sparse_tensor.release %arg0 : tensor<128xf64, #SparseVector>
  return
}

// CHECK-LABEL: func @sparse_nop_convert(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>) -> !llvm.ptr<i8>
//       CHECK: return %[[A]] : !llvm.ptr<i8>
func @sparse_nop_convert(%arg0: tensor<64xf32, #SparseVector>) -> tensor<64xf32, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<64xf32, #SparseVector> to tensor<64xf32, #SparseVector>
  return %0 : tensor<64xf32, #SparseVector>
}

// CHECK-LABEL: func @sparse_hidden_nop_cast(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>) -> !llvm.ptr<i8>
//       CHECK: return %[[A]] : !llvm.ptr<i8>
func @sparse_hidden_nop_cast(%arg0: tensor<32xf32, #SparseVector>) -> tensor<?xf32, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<32xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  return %0 : tensor<?xf32, #SparseVector>
}

// CHECK-LABEL: func @sparse_nop_cast(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>) -> !llvm.ptr<i8>
//       CHECK: return %[[A]] : !llvm.ptr<i8>
func @sparse_nop_cast(%arg0: tensor<64xf32, #SparseVector>) -> tensor<?xf32, #SparseVector> {
  %0 = tensor.cast %arg0 : tensor<64xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  return %0 : tensor<?xf32, #SparseVector>
}

// CHECK-LABEL: func @sparse_convert_1d(
//  CHECK-SAME: %[[A:.*]]: tensor<?xi32>) -> !llvm.ptr<i8>
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[U:.*]] = tensor.dim %[[A]], %[[C0]] : tensor<?xi32>
//   CHECK-DAG: %[[P:.*]] = memref.alloca() : memref<1xi8>
//   CHECK-DAG: %[[Q:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-DAG: %[[R:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-DAG: %[[X:.*]] = memref.cast %[[P]] : memref<1xi8> to memref<?xi8>
//   CHECK-DAG: %[[Y:.*]] = memref.cast %[[Q]] : memref<1xindex> to memref<?xindex>
//   CHECK-DAG: %[[Z:.*]] = memref.cast %[[R]] : memref<1xindex> to memref<?xindex>
//       CHECK: %[[NP:.*]] = llvm.mlir.null : !llvm.ptr<i8>
//       CHECK: %[[C:.*]] = call @newSparseTensor(%[[X]], %[[Y]], %[[Z]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[NP]])
//       CHECK: %[[M:.*]] = memref.alloca() : memref<1xindex>
//       CHECK: %[[T:.*]] = memref.cast %[[M]] : memref<1xindex> to memref<?xindex>
//       CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[U]] step %[[C1]] {
//       CHECK:   %[[E:.*]] = tensor.extract %[[A]][%[[I]]] : tensor<?xi32>
//       CHECK:   memref.store %[[I]], %[[M]][%[[C0]]] : memref<1xindex>
//       CHECK:   call @addEltI32(%[[C]], %[[E]], %[[T]], %[[Z]])
//       CHECK: }
//       CHECK: %[[T:.*]] = call @newSparseTensor(%[[X]], %[[Y]], %[[Z]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[C]])
//       CHECK: return %[[T]] : !llvm.ptr<i8>
func @sparse_convert_1d(%arg0: tensor<?xi32>) -> tensor<?xi32, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<?xi32> to tensor<?xi32, #SparseVector>
  return %0 : tensor<?xi32, #SparseVector>
}

// CHECK-LABEL: func @sparse_convert_1d_ss(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//   CHECK-DAG: %[[P:.*]] = memref.alloca() : memref<1xi8>
//   CHECK-DAG: %[[Q:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-DAG: %[[R:.*]] = memref.alloca() : memref<1xindex>
//   CHECK-DAG: %[[X:.*]] = memref.cast %[[P]] : memref<1xi8> to memref<?xi8>
//   CHECK-DAG: %[[Y:.*]] = memref.cast %[[Q]] : memref<1xindex> to memref<?xindex>
//   CHECK-DAG: %[[Z:.*]] = memref.cast %[[R]] : memref<1xindex> to memref<?xindex>
//       CHECK: %[[C:.*]] = call @newSparseTensor(%[[X]], %[[Y]], %[[Z]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[A]])
//       CHECK: %[[T:.*]] = call @newSparseTensor(%[[X]], %[[Y]], %[[Z]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[C]])
//       CHECK: return %[[T]] : !llvm.ptr<i8>
func @sparse_convert_1d_ss(%arg0: tensor<?xf32, #SparseVector64>) -> tensor<?xf32, #SparseVector32> {
  %0 = sparse_tensor.convert %arg0 : tensor<?xf32, #SparseVector64> to tensor<?xf32, #SparseVector32>
  return %0 : tensor<?xf32, #SparseVector32>
}

// CHECK-LABEL: func @sparse_convert_2d(
//  CHECK-SAME: %[[A:.*]]: tensor<2x4xf64>) -> !llvm.ptr<i8>
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[P:.*]] = memref.alloca() : memref<2xi8>
//   CHECK-DAG: %[[Q:.*]] = memref.alloca() : memref<2xindex>
//   CHECK-DAG: %[[R:.*]] = memref.alloca() : memref<2xindex>
//   CHECK-DAG: %[[X:.*]] = memref.cast %[[P]] : memref<2xi8> to memref<?xi8>
//   CHECK-DAG: %[[Y:.*]] = memref.cast %[[Q]] : memref<2xindex> to memref<?xindex>
//   CHECK-DAG: %[[Z:.*]] = memref.cast %[[R]] : memref<2xindex> to memref<?xindex>
//       CHECK: %[[NP:.*]] = llvm.mlir.null : !llvm.ptr<i8>
//       CHECK: %[[C:.*]] = call @newSparseTensor(%[[X]], %[[Y]], %[[Z]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[NP]])
//       CHECK: %[[M:.*]] = memref.alloca() : memref<2xindex>
//       CHECK: %[[T:.*]] = memref.cast %[[M]] : memref<2xindex> to memref<?xindex>
//       CHECK: scf.for %[[I:.*]] = %[[C0]] to %{{.*}} step %[[C1]] {
//       CHECK:   scf.for %[[J:.*]] = %[[C0]] to %{{.*}} step %[[C1]] {
//       CHECK:     %[[E:.*]] = tensor.extract %[[A]][%[[I]], %[[J]]] : tensor<2x4xf64>
//       CHECK:     memref.store %[[I]], %[[M]][%[[C0]]] : memref<2xindex>
//       CHECK:     memref.store %[[J]], %[[M]][%[[C1]]] : memref<2xindex>
//       CHECK:     call @addEltF64(%[[C]], %[[E]], %[[T]], %[[Z]])
//       CHECK:   }
//       CHECK: }
//       CHECK: %[[T:.*]] = call @newSparseTensor(%[[X]], %[[Y]], %[[Z]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[C]])
//       CHECK: return %[[T]] : !llvm.ptr<i8>
func @sparse_convert_2d(%arg0: tensor<2x4xf64>) -> tensor<2x4xf64, #SparseMatrix> {
  %0 = sparse_tensor.convert %arg0 : tensor<2x4xf64> to tensor<2x4xf64, #SparseMatrix>
  return %0 : tensor<2x4xf64, #SparseMatrix>
}

// CHECK-LABEL: func @sparse_constant() -> !llvm.ptr<i8> {
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG: %[[P:.*]] = memref.alloca() : memref<2xi8>
//   CHECK-DAG: %[[Q:.*]] = memref.alloca() : memref<2xindex>
//   CHECK-DAG: %[[R:.*]] = memref.alloca() : memref<2xindex>
//   CHECK-DAG: %[[X:.*]] = memref.cast %[[P]] : memref<2xi8> to memref<?xi8>
//   CHECK-DAG: %[[Y:.*]] = memref.cast %[[Q]] : memref<2xindex> to memref<?xindex>
//   CHECK-DAG: %[[Z:.*]] = memref.cast %[[R]] : memref<2xindex> to memref<?xindex>
//       CHECK: %[[NP:.*]] = llvm.mlir.null : !llvm.ptr<i8>
//       CHECK: %[[C:.*]] = call @newSparseTensor(%[[X]], %[[Y]], %[[Z]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[NP]])
//       CHECK: %[[M:.*]] = memref.alloca() : memref<2xindex>
//       CHECK: %[[N:.*]] = memref.cast %[[M]] : memref<2xindex> to memref<?xindex>
//       CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
//       CHECK:   memref.store %{{.*}}, %[[M]][%[[C0]]] : memref<2xindex>
//       CHECK:   memref.store %{{.*}}, %[[M]][%[[C1]]] : memref<2xindex>
//       CHECK:   %[[V:.*]] = tensor.extract %{{.*}}[%[[I]]] : tensor<2xf32>
//       CHECK:   call @addEltF32(%{{.*}}, %[[V]], %[[N]], %{{.*}})
//       CHECK: }
//       CHECK: %[[T:.*]] = call @newSparseTensor(%[[X]], %[[Y]], %[[Z]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[C]])
//       CHECK: return %[[T]] : !llvm.ptr<i8>
func @sparse_constant() -> tensor<8x7xf32, #SparseMatrix>{
  // Initialize a tensor.
  %0 = arith.constant sparse<[[0, 0], [1, 6]], [1.0, 5.0]> : tensor<8x7xf32>
  // Convert the tensor to a sparse tensor.
  %1 = sparse_tensor.convert %0 : tensor<8x7xf32> to tensor<8x7xf32, #SparseMatrix>
  return %1 : tensor<8x7xf32, #SparseMatrix>
}

// CHECK-LABEL: func @sparse_convert_3d(
//  CHECK-SAME: %[[A:.*]]: tensor<?x?x?xf64>) -> !llvm.ptr<i8>
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG: %[[U1:.*]] = tensor.dim %[[A]], %[[C0]] : tensor<?x?x?xf64>
//   CHECK-DAG: %[[U2:.*]] = tensor.dim %[[A]], %[[C1]] : tensor<?x?x?xf64>
//   CHECK-DAG: %[[U3:.*]] = tensor.dim %[[A]], %[[C2]] : tensor<?x?x?xf64>
//   CHECK-DAG: %[[P:.*]] = memref.alloca() : memref<3xi8>
//   CHECK-DAG: %[[Q:.*]] = memref.alloca() : memref<3xindex>
//   CHECK-DAG: %[[R:.*]] = memref.alloca() : memref<3xindex>
//   CHECK-DAG: %[[X:.*]] = memref.cast %[[P]] : memref<3xi8> to memref<?xi8>
//   CHECK-DAG: %[[Y:.*]] = memref.cast %[[Q]] : memref<3xindex> to memref<?xindex>
//   CHECK-DAG: %[[Z:.*]] = memref.cast %[[R]] : memref<3xindex> to memref<?xindex>
//       CHECK: %[[NP:.*]] = llvm.mlir.null : !llvm.ptr<i8>
//       CHECK: %[[C:.*]] = call @newSparseTensor(%[[X]], %[[Y]], %[[Z]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[NP]])
//       CHECK: %[[M:.*]] = memref.alloca() : memref<3xindex>
//       CHECK: %[[N:.*]] = memref.cast %[[M]] : memref<3xindex> to memref<?xindex>
//       CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[U1]] step %[[C1]] {
//       CHECK:   scf.for %[[J:.*]] = %[[C0]] to %[[U2]] step %[[C1]] {
//       CHECK:     scf.for %[[K:.*]] = %[[C0]] to %[[U3]] step %[[C1]] {
//       CHECK:       %[[E:.*]] = tensor.extract %[[A]][%[[I]], %[[J]], %[[K]]] : tensor<?x?x?xf64>
//       CHECK:       memref.store %[[I]], %[[M]][%[[C0]]] : memref<3xindex>
//       CHECK:       memref.store %[[J]], %[[M]][%[[C1]]] : memref<3xindex>
//       CHECK:       memref.store %[[K]], %[[M]][%[[C2]]] : memref<3xindex>
//       CHECK:       call @addEltF64(%[[C]], %[[E]], %[[N]], %[[Z]])
//       CHECK:     }
//       CHECK:   }
//       CHECK: }
//       CHECK: %[[T:.*]] = call @newSparseTensor(%[[X]], %[[Y]], %[[Z]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[C]])
//       CHECK: return %[[T]] : !llvm.ptr<i8>
func @sparse_convert_3d(%arg0: tensor<?x?x?xf64>) -> tensor<?x?x?xf64, #SparseTensor> {
  %0 = sparse_tensor.convert %arg0 : tensor<?x?x?xf64> to tensor<?x?x?xf64, #SparseTensor>
  return %0 : tensor<?x?x?xf64, #SparseTensor>
}

// CHECK-LABEL: func @sparse_pointers(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[C:.*]] = arith.constant 0 : index
//       CHECK: %[[T:.*]] = call @sparsePointers(%[[A]], %[[C]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
//       CHECK: return %[[T]] : memref<?xindex>
func @sparse_pointers(%arg0: tensor<128xf64, #SparseVector>) -> memref<?xindex> {
  %c = arith.constant 0 : index
  %0 = sparse_tensor.pointers %arg0, %c : tensor<128xf64, #SparseVector> to memref<?xindex>
  return %0 : memref<?xindex>
}

// CHECK-LABEL: func @sparse_pointers64(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[C:.*]] = arith.constant 0 : index
//       CHECK: %[[T:.*]] = call @sparsePointers64(%[[A]], %[[C]]) : (!llvm.ptr<i8>, index) -> memref<?xi64>
//       CHECK: return %[[T]] : memref<?xi64>
func @sparse_pointers64(%arg0: tensor<128xf64, #SparseVector64>) -> memref<?xi64> {
  %c = arith.constant 0 : index
  %0 = sparse_tensor.pointers %arg0, %c : tensor<128xf64, #SparseVector64> to memref<?xi64>
  return %0 : memref<?xi64>
}

// CHECK-LABEL: func @sparse_pointers32(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[C:.*]] = arith.constant 0 : index
//       CHECK: %[[T:.*]] = call @sparsePointers32(%[[A]], %[[C]]) : (!llvm.ptr<i8>, index) -> memref<?xi32>
//       CHECK: return %[[T]] : memref<?xi32>
func @sparse_pointers32(%arg0: tensor<128xf64, #SparseVector32>) -> memref<?xi32> {
  %c = arith.constant 0 : index
  %0 = sparse_tensor.pointers %arg0, %c : tensor<128xf64, #SparseVector32> to memref<?xi32>
  return %0 : memref<?xi32>
}

// CHECK-LABEL: func @sparse_indices(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[C:.*]] = arith.constant 0 : index
//       CHECK: %[[T:.*]] = call @sparseIndices(%[[A]], %[[C]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
//       CHECK: return %[[T]] : memref<?xindex>
func @sparse_indices(%arg0: tensor<128xf64, #SparseVector>) -> memref<?xindex> {
  %c = arith.constant 0 : index
  %0 = sparse_tensor.indices %arg0, %c : tensor<128xf64, #SparseVector> to memref<?xindex>
  return %0 : memref<?xindex>
}

// CHECK-LABEL: func @sparse_indices64(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[C:.*]] = arith.constant 0 : index
//       CHECK: %[[T:.*]] = call @sparseIndices64(%[[A]], %[[C]]) : (!llvm.ptr<i8>, index) -> memref<?xi64>
//       CHECK: return %[[T]] : memref<?xi64>
func @sparse_indices64(%arg0: tensor<128xf64, #SparseVector64>) -> memref<?xi64> {
  %c = arith.constant 0 : index
  %0 = sparse_tensor.indices %arg0, %c : tensor<128xf64, #SparseVector64> to memref<?xi64>
  return %0 : memref<?xi64>
}

// CHECK-LABEL: func @sparse_indices32(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[C:.*]] = arith.constant 0 : index
//       CHECK: %[[T:.*]] = call @sparseIndices32(%[[A]], %[[C]]) : (!llvm.ptr<i8>, index) -> memref<?xi32>
//       CHECK: return %[[T]] : memref<?xi32>
func @sparse_indices32(%arg0: tensor<128xf64, #SparseVector32>) -> memref<?xi32> {
  %c = arith.constant 0 : index
  %0 = sparse_tensor.indices %arg0, %c : tensor<128xf64, #SparseVector32> to memref<?xi32>
  return %0 : memref<?xi32>
}

// CHECK-LABEL: func @sparse_valuesf64(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[T:.*]] = call @sparseValuesF64(%[[A]]) : (!llvm.ptr<i8>) -> memref<?xf64>
//       CHECK: return %[[T]] : memref<?xf64>
func @sparse_valuesf64(%arg0: tensor<128xf64, #SparseVector>) -> memref<?xf64> {
  %0 = sparse_tensor.values %arg0 : tensor<128xf64, #SparseVector> to memref<?xf64>
  return %0 : memref<?xf64>
}

// CHECK-LABEL: func @sparse_valuesf32(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[T:.*]] = call @sparseValuesF32(%[[A]]) : (!llvm.ptr<i8>) -> memref<?xf32>
//       CHECK: return %[[T]] : memref<?xf32>
func @sparse_valuesf32(%arg0: tensor<128xf32, #SparseVector>) -> memref<?xf32> {
  %0 = sparse_tensor.values %arg0: tensor<128xf32, #SparseVector> to memref<?xf32>
  return %0 : memref<?xf32>
}

// CHECK-LABEL: func @sparse_valuesi32(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[T:.*]] = call @sparseValuesI32(%[[A]]) : (!llvm.ptr<i8>) -> memref<?xi32>
//       CHECK: return %[[T]] : memref<?xi32>
func @sparse_valuesi32(%arg0: tensor<128xi32, #SparseVector>) -> memref<?xi32> {
  %0 = sparse_tensor.values %arg0: tensor<128xi32, #SparseVector> to memref<?xi32>
  return %0 : memref<?xi32>
}

// CHECK-LABEL: func @sparse_valuesi16(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[T:.*]] = call @sparseValuesI16(%[[A]]) : (!llvm.ptr<i8>) -> memref<?xi16>
//       CHECK: return %[[T]] : memref<?xi16>
func @sparse_valuesi16(%arg0: tensor<128xi16, #SparseVector>) -> memref<?xi16> {
  %0 = sparse_tensor.values %arg0: tensor<128xi16, #SparseVector> to memref<?xi16>
  return %0 : memref<?xi16>
}

// CHECK-LABEL: func @sparse_valuesi8(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[T:.*]] = call @sparseValuesI8(%[[A]]) : (!llvm.ptr<i8>) -> memref<?xi8>
//       CHECK: return %[[T]] : memref<?xi8>
func @sparse_valuesi8(%arg0: tensor<128xi8, #SparseVector>) -> memref<?xi8> {
  %0 = sparse_tensor.values %arg0: tensor<128xi8, #SparseVector> to memref<?xi8>
  return %0 : memref<?xi8>
}

// CHECK-LABEL: func @sparse_reconstruct(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>
//       CHECK: return %[[A]] : !llvm.ptr<i8>
func @sparse_reconstruct(%arg0: tensor<128xf32, #SparseVector>) -> tensor<128xf32, #SparseVector> {
  %0 = sparse_tensor.load %arg0 : tensor<128xf32, #SparseVector>
  return %0 : tensor<128xf32, #SparseVector>
}

// CHECK-LABEL: func @sparse_reconstruct_ins(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>
//       CHECK: call @endInsert(%[[A]]) : (!llvm.ptr<i8>) -> ()
//       CHECK: return %[[A]] : !llvm.ptr<i8>
func @sparse_reconstruct_ins(%arg0: tensor<128xf32, #SparseVector>) -> tensor<128xf32, #SparseVector> {
  %0 = sparse_tensor.load %arg0 hasInserts : tensor<128xf32, #SparseVector>
  return %0 : tensor<128xf32, #SparseVector>
}

// CHECK-LABEL: func @sparse_insert(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>,
//  CHECK-SAME: %[[B:.*]]: memref<?xindex>,
//  CHECK-SAME: %[[C:.*]]: f32) {
//       CHECK: call @lexInsertF32(%[[A]], %[[B]], %[[C]]) : (!llvm.ptr<i8>, memref<?xindex>, f32) -> ()
//       CHECK: return
func @sparse_insert(%arg0: tensor<128xf32, #SparseVector>,
                    %arg1: memref<?xindex>,
                    %arg2: f32) {
  sparse_tensor.lex_insert %arg0, %arg1, %arg2 : tensor<128xf32, #SparseVector>, memref<?xindex>, f32
  return
}

// CHECK-LABEL: func @sparse_expansion()
//    %[[S:.*]] = call @sparseDimSize
//    %[[V:.*]] = memref.alloca(%[[S]]) : memref<?xf64>
//    %[[F:.*]] = memref.alloca(%[[S]]) : memref<?xi1>
//    %[[A:.*]] = memref.alloca(%[[S]]) : memref<?xindex>
//    linalg.fill(%{{.*}}, %[[V]]) : f64, memref<?xf64>
//    linalg.fill(%{{.*}}, %[[F]]) : i1, memref<?xi1>
//       CHECK: return
func @sparse_expansion() {
  %c = arith.constant 8 : index
  %0 = sparse_tensor.init [%c, %c] : tensor<8x8xf64, #SparseMatrix>
  %values, %filled, %added, %count = sparse_tensor.expand %0
    : tensor<8x8xf64, #SparseMatrix> to memref<?xf64>, memref<?xi1>, memref<?xindex>, index
  return
}

// CHECK-LABEL: func @sparse_compression(
//  CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>,
//       CHECK: call @expInsertF64(%[[A]],
//       CHECK: return
func @sparse_compression(%arg0: tensor<8x8xf64, #SparseMatrix>,
                         %arg1: memref<?xindex>, %arg2: memref<?xf64>, %arg3: memref<?xi1>,
                         %arg4: memref<?xindex>, %arg5: index) {
  sparse_tensor.compress %arg0, %arg1, %arg2, %arg3, %arg4, %arg5
    : tensor<8x8xf64, #SparseMatrix>, memref<?xindex>, memref<?xf64>, memref<?xi1>, memref<?xindex>, index
  return
}
