// RUN: mlir-opt %s -pre-sparsification-rewrite | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>

#SortedCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed_nu", "singleton" ]
}>

#DCSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
}>

#Slice = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed_nu", "singleton" ],
  dimSlices = [ (?, 1, 1), (?, 3, 1) ]
}>

#sel_trait = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // C (in)
    affine_map<(i,j) -> (i,j)>,  // L (in)
    affine_map<(i,j) -> (i,j)>,  // R (in)
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"]
}

// CHECK-LABEL: func @sparse_nop_cast(
//  CHECK-SAME: %[[A:.*]]: tensor<?xf32, #sparse_tensor.encoding<{{{.*}}}>>)
//       CHECK: return %[[A]] : tensor<?xf32, #sparse_tensor.encoding<{{{.*}}}>>
func.func @sparse_nop_cast(%a : tensor<?xf32, #SparseVector>) -> tensor<?xf32, #SparseVector> {
  %0 = tensor.cast %a : tensor<?xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  %1 = tensor.cast %0 : tensor<?xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  %2 = tensor.cast %1 : tensor<?xf32, #SparseVector> to tensor<?xf32, #SparseVector>
  return %2 : tensor<?xf32, #SparseVector>
}

// CHECK-LABEL: func @sparse_repair_cast(
//  CHECK-SAME: %[[A:.*]]: tensor<?xf32>)
//       CHECK: %[[C:.*]] = sparse_tensor.convert %[[A]] : tensor<?xf32> to tensor<?xf32, #sparse_tensor.encoding<{{{.*}}}>
//       CHECK: return %[[C]] : tensor<?xf32, #sparse_tensor.encoding<{{{.*}}}>>
func.func @sparse_repair_cast(%a : tensor<?xf32>) -> tensor<?xf32, #SparseVector> {
  %0 = tensor.cast %a : tensor<?xf32> to tensor<?xf32, #SparseVector>
  return %0 : tensor<?xf32, #SparseVector>
}

// CHECK-LABEL: func @sparse_fuse_slice(
//  CHECK-SAME: %[[A:.*]]: tensor<2x3xi64, #sparse_tensor.encoding<{{{.*}}}>>)
//       CHECK: %[[E:.*]] = tensor.extract_slice %[[A]][1, 0] [1, 3] [1, 1] : tensor<2x3xi64, #sparse_tensor.encoding<{{{.*}}}>> to tensor<1x3xi64, #sparse_tensor.encoding<{{{.*}}}>>
//       CHECK: %[[C:.*]] = sparse_tensor.convert %[[E]] : tensor<1x3xi64, #sparse_tensor.encoding<{{{.*}}}>> to tensor<1x3xi64, #sparse_tensor.encoding<{{{.*}}}>>
//       CHECK: return %[[C]] : tensor<1x3xi64, #sparse_tensor.encoding<{{{.*}}}>>
func.func @sparse_fuse_slice(%a : tensor<2x3xi64, #SortedCOO>) -> tensor<1x3xi64, #SortedCOO> {
  %extracted_slice = tensor.extract_slice %a[1, 0] [1, 3] [1, 1] : tensor<2x3xi64, #SortedCOO> to tensor<1x3xi64>
  %cast = tensor.cast %extracted_slice : tensor<1x3xi64> to tensor<1x3xi64, #Slice>
  %0 = sparse_tensor.convert %cast : tensor<1x3xi64, #Slice> to tensor<1x3xi64, #SortedCOO>
  return %0 : tensor<1x3xi64, #SortedCOO>
}

// CHECK-LABEL:   func.func @sparse_select(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<4x4xi1>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<4x4xf64, #sparse_tensor.encoding<{{.*}}>>,
// CHECK-SAME:      %[[VAL_2:.*]]: tensor<4x4xf64, #sparse_tensor.encoding<{{.*}}>>) -> tensor<4x4xf64, #sparse_tensor.encoding<{{.*}}>> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_4:.*]] = bufferization.alloc_tensor() : tensor<4x4xf64, #sparse_tensor.encoding<{{.*}}>>
// CHECK-NEXT:      %[[VAL_5:.*]] = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:      ins(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]
// CHECK-NEXT:      ^bb0(%[[VAL_6:.*]]: i1, %[[VAL_7:.*]]: f64, %[[VAL_8:.*]]: f64, %[[VAL_9:.*]]: f64):
// CHECK-NEXT:        %[[VAL_10:.*]] = sparse_tensor.binary %[[VAL_7]], %[[VAL_8]] : f64, f64 to f64
// CHECK-NEXT:         overlap = {
// CHECK-NEXT:        ^bb0(%[[VAL_11:.*]]: f64, %[[VAL_12:.*]]: f64):
// CHECK-NEXT:          %[[VAL_13:.*]] = arith.select %[[VAL_6]], %[[VAL_11]], %[[VAL_12]] : f64
// CHECK-NEXT:          sparse_tensor.yield %[[VAL_13]] : f64
// CHECK-NEXT:        }
// CHECK-NEXT:         left = {
// CHECK-NEXT:        ^bb0(%[[VAL_14:.*]]: f64):
// CHECK-NEXT:          %[[VAL_15:.*]] = arith.select %[[VAL_6]], %[[VAL_14]], %[[VAL_3]] : f64
// CHECK-NEXT:          sparse_tensor.yield %[[VAL_15]] : f64
// CHECK-NEXT:        }
// CHECK-NEXT:         right = {
// CHECK-NEXT:        ^bb0(%[[VAL_16:.*]]: f64):
// CHECK-NEXT:          %[[VAL_17:.*]] = arith.select %[[VAL_6]], %[[VAL_3]], %[[VAL_16]] : f64
// CHECK-NEXT:          sparse_tensor.yield %[[VAL_17]] : f64
// CHECK-NEXT:        }
// CHECK-NEXT:        linalg.yield %[[VAL_10]] : f64
// CHECK-NEXT:      } -> tensor<4x4xf64, #sparse_tensor.encoding<{{.*}}>>
// CHECK-NEXT:      return %[[VAL_18:.*]] : tensor<4x4xf64, #sparse_tensor.encoding<{{.*}}>>
// CHECK-NEXT:    }
func.func @sparse_select(%cond: tensor<4x4xi1>,
                         %arga: tensor<4x4xf64, #DCSR>,
                         %argb: tensor<4x4xf64, #DCSR>) -> tensor<4x4xf64, #DCSR> {
  %xv = bufferization.alloc_tensor() : tensor<4x4xf64, #DCSR>
  %0 = linalg.generic #sel_trait
     ins(%cond, %arga, %argb: tensor<4x4xi1>, tensor<4x4xf64, #DCSR>, tensor<4x4xf64, #DCSR>)
      outs(%xv: tensor<4x4xf64, #DCSR>) {
      ^bb(%c: i1, %a: f64, %b: f64, %x: f64):
        %1 = arith.select %c, %a, %b : f64
        linalg.yield %1 : f64
  } -> tensor<4x4xf64, #DCSR>
  return %0 : tensor<4x4xf64, #DCSR>
}
