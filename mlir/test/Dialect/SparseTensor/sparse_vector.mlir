// RUN: mlir-opt %s -sparsification -cse -split-input-file | \
// RUN:   FileCheck %s

#DenseVector = #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>

#trait_scale_d = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = a(i) * b"
}

//
// CHECK-LABEL: func @scale_d
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[c1024:.*]] = arith.constant 1024 : index
// CHECK:       scf.for %[[i:.*]] = %[[c0]] to %[[c1024]] step %[[c1]] {
// CHECK:         %[[l:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xf32>
// CHECK:         %[[m:.*]] = arith.mulf %[[l]], %{{.*}} : f32
// CHECK:         store %[[m]], %{{.*}}[%[[i]]] : memref<1024xf32>
// CHECK:       }
// CHECK:       return
//

func.func @scale_d(%arga: tensor<1024xf32, #DenseVector>, %b: f32, %argx: tensor<1024xf32>) -> tensor<1024xf32> {
  %0 = linalg.generic #trait_scale_d
    ins(%arga: tensor<1024xf32, #DenseVector>)
    outs(%argx: tensor<1024xf32>) {
      ^bb(%a: f32, %x: f32):
        %0 = arith.mulf %a, %b : f32
        linalg.yield %0 : f32
  } -> tensor<1024xf32>
  return %0 : tensor<1024xf32>
}

// -----

#SparseVector = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 32,
  indexBitWidth = 32
}>

#trait_mul_s = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>,  // b
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = a(i) * b(i)"
}

//
// CHECK-LABEL: func @mul_s
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK:       %[[p:.*]] = memref.load %{{.*}}[%[[c0]]] : memref<?xi32>
// CHECK:       %[[a:.*]] = arith.extui %[[p]] : i32 to i64
// CHECK:       %[[q:.*]] = arith.index_cast %[[a]] : i64 to index
// CHECK:       %[[r:.*]] = memref.load %{{.*}}[%[[c1]]] : memref<?xi32>
// CHECK:       %[[b:.*]] = arith.extui %[[r]] : i32 to i64
// CHECK:       %[[s:.*]] = arith.index_cast %[[b]] : i64 to index
// CHECK:       scf.for %[[i:.*]] = %[[q]] to %[[s]] step %[[c1]] {
// CHECK:         %[[li:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xi32>
// CHECK:         %[[zi:.*]] = arith.extui %[[li]] : i32 to i64
// CHECK:         %[[ci:.*]] = arith.index_cast %[[zi]] : i64 to index
// CHECK:         %[[la:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xf32>
// CHECK:         %[[lb:.*]] = memref.load %{{.*}}[%[[ci]]] : memref<1024xf32>
// CHECK:         %[[m:.*]] = arith.mulf %[[la]], %[[lb]] : f32
// CHECK:         store %[[m]], %{{.*}}[%[[ci]]] : memref<1024xf32>
// CHECK:       }
// CHECK:       return
//
func.func @mul_s(%arga: tensor<1024xf32, #SparseVector>, %argb: tensor<1024xf32>, %argx: tensor<1024xf32>) -> tensor<1024xf32> {
  %0 = linalg.generic #trait_mul_s
    ins(%arga, %argb: tensor<1024xf32, #SparseVector>, tensor<1024xf32>)
    outs(%argx: tensor<1024xf32>) {
      ^bb(%a: f32, %b: f32, %x: f32):
        %0 = arith.mulf %a, %b : f32
        linalg.yield %0 : f32
  } -> tensor<1024xf32>
  return %0 : tensor<1024xf32>
}

// -----

#DenseVector = #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>

#trait_reduction_d = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>,  // b
    affine_map<(i) -> ()>    // x (out)
  ],
  iterator_types = ["reduction"],
  doc = "x += a(i) * b(i)"
}

//
// CHECK-LABEL: func @reduction_d
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[c1024:.*]] = arith.constant 1024 : index
// CHECK:       %[[red:.*]] = scf.for %[[i:.*]] = %[[c0]] to %[[c1024]] step %[[c1]] iter_args(%[[red_in:.*]] = %{{.*}}) -> (f32) {
// CHECK:         %[[la:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xf32>
// CHECK:         %[[lb:.*]] = memref.load %{{.*}}[%[[i]]] : memref<1024xf32>
// CHECK:         %[[m:.*]] = arith.mulf %[[la]], %[[lb]] : f32
// CHECK:         %[[a:.*]] = arith.addf %[[red_in]], %[[m]] : f32
// CHECK:         scf.yield %[[a]] : f32
// CHECK:       }
// CHECK:       return
//
func.func @reduction_d(%arga: tensor<1024xf32, #DenseVector>, %argb: tensor<1024xf32>, %argx: tensor<f32>) -> tensor<f32> {
  %0 = linalg.generic #trait_reduction_d
    ins(%arga, %argb: tensor<1024xf32, #DenseVector>, tensor<1024xf32>)
    outs(%argx: tensor<f32>) {
      ^bb(%a: f32, %b: f32, %x: f32):
        %0 = arith.mulf %a, %b : f32
        %1 = arith.addf %x, %0 : f32
        linalg.yield %1 : f32
  } -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  pointerBitWidth = 32,
  indexBitWidth = 32
}>

#trait_mul_ds = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (i,j)>,  // B
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) * B(i,j)"
}

//
// CHECK-LABEL: func @mul_ds
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[c512:.*]] = arith.constant 512 : index
// CHECK:       scf.for %[[i:.*]] = %[[c0]] to %[[c512]] step %[[c1]] {
// CHECK:         %[[p:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xi32>
// CHECK:         %[[a:.*]] = arith.extui %[[p]] : i32 to i64
// CHECK:         %[[q:.*]] = arith.index_cast %[[a]] : i64 to index
// CHECK:         %[[a:.*]] = arith.addi %[[i]], %[[c1]] : index
// CHECK:         %[[r:.*]] = memref.load %{{.*}}[%[[a]]] : memref<?xi32>
// CHECK:         %[[b:.*]] = arith.extui %[[r]] : i32 to i64
// CHECK:         %[[s:.*]] = arith.index_cast %[[b]] : i64 to index
// CHECK:         scf.for %[[j:.*]] = %[[q]] to %[[s]] step %[[c1]] {
// CHECK:           %[[lj:.*]] = memref.load %{{.*}}[%[[j]]] : memref<?xi32>
// CHECK:           %[[zj:.*]] = arith.extui %[[lj]] : i32 to i64
// CHECK:           %[[cj:.*]] = arith.index_cast %[[zj]] : i64 to index
// CHECK:           %[[la:.*]] = memref.load %{{.*}}[%[[j]]] : memref<?xf32>
// CHECK:           %[[lb:.*]] = memref.load %{{.*}}[%[[i]], %[[cj]]] : memref<512x1024xf32>
// CHECK:           %[[m:.*]] = arith.mulf %[[la]], %[[lb]] : f32
// CHECK:           store %[[m]], %{{.*}}[%[[i]], %[[cj]]] : memref<512x1024xf32>
// CHECK:         }
// CHECK:       }
// CHECK:       return
//
func.func @mul_ds(%arga: tensor<512x1024xf32, #SparseMatrix>, %argb: tensor<512x1024xf32>, %argx: tensor<512x1024xf32>) -> tensor<512x1024xf32> {
  %0 = linalg.generic #trait_mul_ds
    ins(%arga, %argb: tensor<512x1024xf32, #SparseMatrix>, tensor<512x1024xf32>)
    outs(%argx: tensor<512x1024xf32>) {
      ^bb(%a: f32, %b: f32, %x: f32):
        %0 = arith.mulf %a, %b : f32
        linalg.yield %0 : f32
  } -> tensor<512x1024xf32>
  return %0 : tensor<512x1024xf32>
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{dimLevelType = ["dense","compressed"]}>

#trait_affine = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,
    affine_map<(i,j) -> (i+1,j)>
  ],
  iterator_types = ["parallel","parallel"],
  doc = "X(i+1,j) += A(i,j)"
}

//
// CHECK-LABEL: func @add_dense
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[c32:.*]] = arith.constant 32 : index
// CHECK:       scf.for %[[i:.*]] = %[[c0]] to %[[c32]] step %[[c1]] {
// CHECK:         %[[lo:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xindex>
// CHECK:         %[[i1:.*]] = arith.addi %[[i]], %[[c1]] : index
// CHECK:         %[[hi:.*]] = memref.load %{{.*}}[%[[i1]]] : memref<?xindex>
// CHECK:         scf.for %[[jj:.*]] = %[[lo]] to %[[hi]] step %[[c1]] {
// CHECK:           %[[j:.*]] = memref.load %{{.*}}[%[[jj]]] : memref<?xindex>
// CHECK:           %[[x:.*]] = memref.load %{{.*}}[%[[i1]], %[[j]]] : memref<33x64xf64>
// CHECK:           %[[a:.*]] = memref.load %{{.*}}[%[[jj]]] : memref<?xf64>
// CHECK:           %[[s:.*]] = arith.addf %[[x]], %[[a]] : f64
// CHECK:           memref.store %[[s]], %{{.*}}[%[[i1]], %[[j]]] : memref<33x64xf64>
// CHECK:         }
// CHECK:       }
// CHECK:       return
//
func.func @add_dense(%arga: tensor<32x64xf64, #SparseMatrix>,
                %argx: tensor<33x64xf64>) -> tensor<33x64xf64> {
  %0 = linalg.generic #trait_affine
     ins(%arga: tensor<32x64xf64, #SparseMatrix>)
    outs(%argx: tensor<33x64xf64>) {
      ^bb(%a: f64, %x: f64):
        %0 = arith.addf %x, %a : f64
        linalg.yield %0 : f64
  } -> tensor<33x64xf64>
  return %0 : tensor<33x64xf64>
}
