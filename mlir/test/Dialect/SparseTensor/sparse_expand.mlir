// RUN: mlir-opt %s -sparsification                           | \
// RUN:   FileCheck %s --check-prefix=CHECK-SPARSE
// RUN: mlir-opt %s -sparsification -sparse-tensor-conversion | \
// RUN:   FileCheck %s --check-prefix=CHECK-CONVERT

#DCSC = #sparse_tensor.encoding<{
  dimLevelType = [  "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>
}>

#SV = #sparse_tensor.encoding<{
  dimLevelType = [  "compressed" ]
}>

#rowsum = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>, // A
    affine_map<(i,j) -> (i)>    // x (out)
  ],
  iterator_types = ["parallel", "reduction"],
  doc = "X(i) = SUM A(i,j)"
}

//
// CHECK-SPARSE-LABEL: func @kernel(
// CHECK-SPARSE: %[[A:.*]], %[[B:.*]], %[[C:.*]], %{{.*}} = sparse_tensor.expand
// CHECK-SPARSE: scf.for
// CHECK-SPARSE:   scf.for
// CHECK-SPARSE: sparse_tensor.compress %{{.*}}, %{{.*}}, %[[A]], %[[B]], %[[C]]
// CHECK-SPARSE: %[[RET:.*]] = sparse_tensor.load %{{.*}} hasInserts
// CHECK-SPARSE: return %[[RET]]
//
// CHECK-CONVERT-LABEL: func @kernel(
// CHECK-CONVERT: %{{.*}} = call @sparseDimSize
// CHECK-CONVERT: %[[S:.*]] = call @sparseDimSize
// CHECK-CONVERT: %[[A:.*]] = memref.alloc(%[[S]]) : memref<?xf64>
// CHECK-CONVERT: %[[B:.*]] = memref.alloc(%[[S]]) : memref<?xi1>
// CHECK-CONVERT: %[[C:.*]] = memref.alloc(%[[S]]) : memref<?xindex>
// CHECK-CONVERT: linalg.fill ins(%{{.*}} : f64) outs(%[[A]] : memref<?xf64>)
// CHECK-CONVERT: linalg.fill ins(%{{.*}} : i1) outs(%[[B]] : memref<?xi1>)
// CHECK-CONVERT: scf.for
// CHECK-CONVERT:   scf.for
// CHECK-CONVERT: call @expInsertF64
// CHECK-CONVERT: memref.dealloc %[[A]] : memref<?xf64>
// CHECK-CONVERT: memref.dealloc %[[B]] : memref<?xi1>
// CHECK-CONVERT: memref.dealloc %[[C]] : memref<?xindex>
// CHECK-CONVERT: call @endInsert
//
func.func @kernel(%arga: tensor<?x?xf64, #DCSC>) -> tensor<?xf64, #SV> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %arga, %c0 : tensor<?x?xf64, #DCSC>
  %v = bufferization.alloc_tensor(%n) : tensor<?xf64, #SV>
  %0 = linalg.generic #rowsum
    ins(%arga: tensor<?x?xf64, #DCSC>)
    outs(%v: tensor<?xf64, #SV>) {
    ^bb(%a: f64, %x: f64):
      %1 = arith.addf %x, %a : f64
      linalg.yield %1 : f64
  } -> tensor<?xf64, #SV>
  return %0 : tensor<?xf64, #SV>
}
