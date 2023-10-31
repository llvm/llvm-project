// RUN: mlir-opt %s --linalg-generalize-named-ops \
// RUN:             --linalg-fuse-elementwise-ops \
// RUN:             --sparsification | \
// RUN:   FileCheck %s --check-prefix=CHECK-SPARSE
// RUN: mlir-opt %s --linalg-generalize-named-ops \
// RUN:             --linalg-fuse-elementwise-ops \
// RUN:             --sparsification --lower-sparse-ops-to-foreach \
// RUN:             --lower-sparse-foreach-to-scf \
// RUN:             --sparse-tensor-conversion --cse | \
// RUN:   FileCheck %s --check-prefix=CHECK-CONVERT

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed)
}>

#DCSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : compressed, d0 : compressed),
}>

#SV = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
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
// CHECK-SPARSE: %[[COUNT:.*]] = scf.for {{.*}} {
// CHECK-SPARSE:   scf.for {{.*}} {
// CHECK-SPARSE:   }
// CHECK-SPARSE: }
// CHECK-SPARSE: sparse_tensor.compress %[[A]], %[[B]], %[[C]], %[[COUNT]] into
// CHECK-SPARSE: %[[RET:.*]] = sparse_tensor.load %{{.*}} hasInserts
// CHECK-SPARSE: return %[[RET]]
//
// CHECK-CONVERT-LABEL: func @kernel(
// CHECK-CONVERT-SAME: %[[A:.*]]: !llvm.ptr) -> !llvm.ptr
// CHECK-CONVERT-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-CONVERT-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-CONVERT: %[[N:.*]] = call @sparseLvlSize(%[[A]], %[[C1]])
// CHECK-CONVERT: %[[V:.*]] = call @newSparseTensor
// CHECK-CONVERT: %[[S:.*]] = call @sparseLvlSize(%[[V]], %[[C0]])
// CHECK-CONVERT: %[[A:.*]] = memref.alloc(%[[S]]) : memref<?xf64>
// CHECK-CONVERT: %[[B:.*]] = memref.alloc(%[[S]]) : memref<?xi1>
// CHECK-CONVERT: %[[C:.*]] = memref.alloc(%[[S]]) : memref<?xindex>
// CHECK-CONVERT: linalg.fill ins(%{{.*}} : f64) outs(%[[A]] : memref<?xf64>)
// CHECK-CONVERT: linalg.fill ins(%{{.*}} : i1) outs(%[[B]] : memref<?xi1>)
// CHECK-CONVERT: scf.for {{.*}} {
// CHECK-CONVERT:   scf.for {{.*}} {
// CHECK-CONVERT:   }
// CHECK-CONVERT: }
// CHECK-CONVERT: call @expInsertF64
// CHECK-CONVERT: memref.dealloc %[[A]] : memref<?xf64>
// CHECK-CONVERT: memref.dealloc %[[B]] : memref<?xi1>
// CHECK-CONVERT: memref.dealloc %[[C]] : memref<?xindex>
// CHECK-CONVERT: call @endLexInsert
//
func.func @kernel(%arga: tensor<?x?xf64, #DCSC>) -> tensor<?xf64, #SV> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %arga, %c0 : tensor<?x?xf64, #DCSC>
  %v = tensor.empty(%n) : tensor<?xf64, #SV>
  %0 = linalg.generic #rowsum
    ins(%arga: tensor<?x?xf64, #DCSC>)
    outs(%v: tensor<?xf64, #SV>) {
    ^bb(%a: f64, %x: f64):
      %1 = arith.addf %x, %a : f64
      linalg.yield %1 : f64
  } -> tensor<?xf64, #SV>
  return %0 : tensor<?xf64, #SV>
}

//
// CHECK-SPARSE-LABEL: func @matmul1(
// CHECK-SPARSE-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-SPARSE-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-SPARSE-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK-SPARSE: %[[T:.*]] = scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]] {{.*}} {
// CHECK-SPARSE:   %[[A:.*]], %[[B:.*]], %[[C:.*]], %{{.*}} = sparse_tensor.expand
// CHECK-SPARSE:   %[[COUNT:.*]] = scf.for {{.*}} {
// CHECK-SPARSE:     scf.for {{.*}} {
// CHECK-SPARSE:     }
// CHECK-SPARSE:   }
// CHECK-SPARSE:   sparse_tensor.compress %[[A]], %[[B]], %[[C]], %[[COUNT]] into
// CHECK-SPARSE: }
// CHECK-SPARSE: %[[RET:.*]] = sparse_tensor.load %[[T]] hasInserts
// CHECK-SPARSE: return %[[RET]]
//
// CHECK-CONVERT-LABEL: func @matmul1(
// CHECK-CONVERT-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-CONVERT-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-CONVERT-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-CONVERT-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK-CONVERT: %[[N:.*]] = call @newSparseTensor
// CHECK-CONVERT: %[[A:.*]] = memref.alloc(%[[C4]]) : memref<?xf64>
// CHECK-CONVERT: %[[B:.*]] = memref.alloc(%[[C4]]) : memref<?xi1>
// CHECK-CONVERT: %[[C:.*]] = memref.alloc(%[[C4]]) : memref<?xindex>
// CHECK-CONVERT: linalg.fill ins(%{{.*}} : f64) outs(%[[A]] : memref<?xf64>)
// CHECK-CONVERT: linalg.fill ins(%{{.*}} : i1) outs(%[[B]] : memref<?xi1>)
// CHECK-CONVERT: scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]] {{.*}} {
// CHECK-CONVERT:   scf.for {{.*}} {
// CHECK-CONVERT:     scf.for {{.*}} {
// CHECK-CONVERT:     }
// CHECK-CONVERT:   }
// CHECK-CONVERT:   call @expInsertF64
// CHECK-CONVERT: }
// CHECK-CONVERT: memref.dealloc %[[A]] : memref<?xf64>
// CHECK-CONVERT: memref.dealloc %[[B]] : memref<?xi1>
// CHECK-CONVERT: memref.dealloc %[[C]] : memref<?xindex>
// CHECK-CONVERT: call @endLexInsert
//
func.func @matmul1(%A: tensor<8x2xf64, #CSR>,
                   %B: tensor<2x4xf64, #CSR>) -> tensor<8x4xf64, #CSR> {
  %C = tensor.empty() : tensor<8x4xf64, #CSR>
  %D = linalg.matmul
    ins(%A, %B: tensor<8x2xf64, #CSR>, tensor<2x4xf64, #CSR>)
       outs(%C: tensor<8x4xf64, #CSR>) -> tensor<8x4xf64, #CSR>
  return %D: tensor<8x4xf64, #CSR>
}

//
// CHECK-SPARSE-LABEL: func @matmul2(
// CHECK-SPARSE-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-SPARSE-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-SPARSE-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-SPARSE: %[[T:.*]] = scf.for %{{.*}} = %[[C0]] to %[[C4]] step %[[C1]] {{.*}} {
// CHECK-SPARSE:   %[[A:.*]], %[[B:.*]], %[[C:.*]], %{{.*}} = sparse_tensor.expand
// CHECK-SPARSE:   %[[COUNT:.*]] = scf.for {{.*}} {
// CHECK-SPARSE:     scf.for {{.*}} {
// CHECK-SPARSE:     }
// CHECK-SPARSE:   }
// CHECK-SPARSE:   sparse_tensor.compress %[[A]], %[[B]], %[[C]], %[[COUNT]]
// CHECK-SPARSE: }
// CHECK-SPARSE: %[[RET:.*]] = sparse_tensor.load %[[T]] hasInserts
// CHECK-SPARSE: return %[[RET]]
//
// CHECK-CONVERT-LABEL: func @matmul2(
// CHECK-CONVERT-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-CONVERT-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-CONVERT-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-CONVERT-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK-CONVERT: %[[N:.*]] = call @newSparseTensor
// CHECK-CONVERT: %[[A:.*]] = memref.alloc(%[[C8]]) : memref<?xf64>
// CHECK-CONVERT: %[[B:.*]] = memref.alloc(%[[C8]]) : memref<?xi1>
// CHECK-CONVERT: %[[C:.*]] = memref.alloc(%[[C8]]) : memref<?xindex>
// CHECK-CONVERT: linalg.fill ins(%{{.*}} : f64) outs(%[[A]] : memref<?xf64>)
// CHECK-CONVERT: linalg.fill ins(%{{.*}} : i1) outs(%[[B]] : memref<?xi1>)
// CHECK-CONVERT: scf.for %{{.*}} = %[[C0]] to %[[C4]] step %[[C1]] {{.*}} {
// CHECK-CONVERT:   scf.for {{.*}} {
// CHECK-CONVERT:     scf.for {{.*}} {
// CHECK-CONVERT:     }
// CHECK-CONVERT:   }
// CHECK-CONVERT:   call @expInsertF64
// CHECK-CONVERT: }
// CHECK-CONVERT: memref.dealloc %[[A]] : memref<?xf64>
// CHECK-CONVERT: memref.dealloc %[[B]] : memref<?xi1>
// CHECK-CONVERT: memref.dealloc %[[C]] : memref<?xindex>
// CHECK-CONVERT: call @endLexInsert
//
func.func @matmul2(%A: tensor<8x2xf64, #CSC>,
                   %B: tensor<2x4xf64, #CSC>) -> tensor<8x4xf64, #CSC> {
  %C = tensor.empty() : tensor<8x4xf64, #CSC>
  %D = linalg.matmul
    ins(%A, %B: tensor<8x2xf64, #CSC>, tensor<2x4xf64, #CSC>)
       outs(%C: tensor<8x4xf64, #CSC>) -> tensor<8x4xf64, #CSC>
  return %D: tensor<8x4xf64, #CSC>
}
