// RUN: mlir-opt %s -test-vector-to-vector-lowering -split-input-file| FileCheck %s

// CHECK-DAG: #[[$map0:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$map1:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$map2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: cast_away_contraction_leading_one_dims
//  CHECK-NEXT:   %[[R0:.+]] =  vector.extract %{{.*}}[0] : vector<1x16x8xf32>
//  CHECK-NEXT:   %[[R1:.+]] =  vector.extract %{{.*}}[0] : vector<1x8x16xf32>
//  CHECK-NEXT:   %[[R2:.+]] =  vector.extract %{{.*}}[0] : vector<1x16x16xf32>
//  CHECK-NEXT:   %[[R3:.+]] = vector.contract {indexing_maps = [#[[$map0]], #[[$map1]], #[[$map2]]],
//  CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
//  CHECK-SAME:   %[[R0]], %[[R1]], %[[R2]] : vector<16x8xf32>, vector<8x16xf32> into vector<16x16xf32>
//  CHECK-NEXT:   %[[R4:.+]] = vector.broadcast %[[R3]] : vector<16x16xf32> to vector<1x16x16xf32>
//  CHECK-NEXT:  return %[[R4]] : vector<1x16x16xf32>

#contraction_accesses0 = [
  affine_map<(l, i, j, k) -> (l, i, k)>,
  affine_map<(l, i, j, k) -> (l, k, j)>,
  affine_map<(l, i, j, k) -> (l, i, j)>
]
#contraction_trait0 = {
  indexing_maps = #contraction_accesses0,
  iterator_types = ["parallel", "parallel", "parallel", "reduction"]
}

func.func @cast_away_contraction_leading_one_dims(%arg0: vector<1x16x8xf32>, %arg1: vector<1x8x16xf32>, %arg2: vector<1x16x16xf32>) -> vector<1x16x16xf32> {
  %0 = vector.contract #contraction_trait0 %arg0, %arg1, %arg2  : vector<1x16x8xf32>, vector<1x8x16xf32> into vector<1x16x16xf32>
  return %0: vector<1x16x16xf32>
}

// -----
// CHECK-DAG: #[[$map0:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #[[$map1:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG: #[[$map2:.*]] = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: cast_away_contraction_leading_one_dims_transposeneeded
//  CHECK-NEXT:   %[[R0:.+]] =  vector.extract %{{.*}}[0] : vector<1x8x16xf32>
//  CHECK-NEXT:   %[[R1:.+]] =  vector.extract %{{.*}}[0, 0] : vector<1x1x8xf32>
//  CHECK-NEXT:   %[[R2:.+]] =  vector.extract %{{.*}}[0, 0] : vector<1x1x16xf32>
//  CHECK-NEXT:   %[[R3:.+]] = vector.contract {indexing_maps = [#[[$map0]], #[[$map1]], #[[$map2]]],
//  CHECK-SAME:   iterator_types = ["parallel", "reduction"], kind = #vector.kind<mul>}
//  CHECK-SAME:   %[[R1]], %[[R0]], %[[R2]] : vector<8xf32>, vector<8x16xf32> into vector<16xf32>
//  CHECK-NEXT:   %[[R4:.+]] = vector.broadcast %[[R3]] : vector<16xf32> to vector<1x16xf32>
//  CHECK-NEXT:   %[[R5:.+]] = vector.broadcast %[[R4]] : vector<1x16xf32> to vector<1x1x16xf32>
//  CHECK-NEXT:  return %[[R5]] : vector<1x1x16xf32>

#contraction_accesses1 = [
  affine_map<(l, i, j, k) -> (i, l, k)>,
  affine_map<(l, i, j, k) -> (l, k, j)>,
  affine_map<(l, i, j, k) -> (l, i, j)>
]
#contraction_trait1 = {
  indexing_maps = #contraction_accesses1,
  iterator_types = ["parallel", "parallel", "parallel", "reduction"],
  kind = #vector.kind<mul>
}

func.func @cast_away_contraction_leading_one_dims_transposeneeded(%arg0: vector<1x1x8xf32>, %arg1: vector<1x8x16xf32>, %arg2: vector<1x1x16xf32>) -> vector<1x1x16xf32> {
  %0 = vector.contract #contraction_trait1 %arg0, %arg1, %arg2  : vector<1x1x8xf32>, vector<1x8x16xf32> into vector<1x1x16xf32>
  return %0: vector<1x1x16xf32>
}

// -----
// CHECK-DAG: #[[$map0:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$map1:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$map2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: cast_away_contraction_leading_one_dims_transposeneeded2
//  CHECK-NEXT:   %[[R0:.+]] =  vector.transpose %{{.*}}[1, 0, 2] : vector<8x1x16xf32> to vector<1x8x16xf32>
//  CHECK-NEXT:   %[[R1:.+]] =  vector.extract %[[R0]][0] : vector<1x8x16xf32>
//  CHECK-NEXT:   %[[R2:.+]] =  vector.transpose %{{.*}}[2, 0, 1] : vector<2x8x1xf32> to vector<1x2x8xf32>
//  CHECK-NEXT:   %[[R3:.+]] =  vector.extract %[[R2]][0] : vector<1x2x8xf32>
//  CHECK-NEXT:   %[[R4:.+]] =  vector.extract %{{.*}}[0] : vector<1x2x16xf32>
//  CHECK-NEXT:   %[[R5:.+]] = vector.contract {indexing_maps = [#[[$map0]], #[[$map1]], #[[$map2]]],
//  CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
//  CHECK-SAME:   %[[R1]], %[[R3]], %[[R4]] : vector<8x16xf32>, vector<2x8xf32> into vector<2x16xf32>
//  CHECK-NEXT:   %[[R6:.+]] = vector.broadcast %[[R5]] : vector<2x16xf32> to vector<1x2x16xf32>
//  CHECK-NEXT:  return %[[R6]] : vector<1x2x16xf32>

#contraction_accesses2 = [
  affine_map<(l, i, j, k) -> (k, l, j)>,
  affine_map<(l, i, j, k) -> (i, k, l)>,
  affine_map<(l, i, j, k) -> (l, i, j)>
]
#contraction_trait2 = {
  indexing_maps = #contraction_accesses2,
  iterator_types = ["parallel", "parallel", "parallel", "reduction"]
}


func.func @cast_away_contraction_leading_one_dims_transposeneeded2(%arg0: vector<8x1x16xf32>, %arg1: vector<2x8x1xf32>, %arg2: vector<1x2x16xf32>) -> vector<1x2x16xf32> {
  %0 = vector.contract #contraction_trait2 %arg0, %arg1, %arg2  : vector<8x1x16xf32>, vector<2x8x1xf32> into vector<1x2x16xf32>
  return %0: vector<1x2x16xf32>
}

// -----
// CHECK-DAG: #[[$map0:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$map1:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$map2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>


// CHECK-LABEL: cast_away_contraction_leading_one_dims_nonleadingunitdim_rank4
//  CHECK-NEXT:   %[[R0:.+]] =  vector.extract %{{.*}}[0] : vector<1x8x1x16xf32>
//  CHECK-NEXT:   %[[R1:.+]] =  vector.extract %{{.*}}[0] : vector<1x2x8x1xf32>
//  CHECK-NEXT:   %[[R2:.+]] =  vector.transpose %[[R0]], [1, 0, 2] : vector<8x1x16xf32> to vector<1x8x16xf32>
//  CHECK-NEXT:   %[[R3:.+]] =  vector.extract %[[R2]][0] : vector<1x8x16xf32>
//  CHECK-NEXT:   %[[R4:.+]] =  vector.transpose %[[R1]], [2, 0, 1] : vector<2x8x1xf32> to vector<1x2x8xf32>
//  CHECK-NEXT:   %[[R5:.+]] =  vector.extract %[[R4]][0] : vector<1x2x8xf32>
//  CHECK-NEXT:   %[[R6:.+]] =  vector.extract %{{.*}}[0, 0] : vector<1x1x2x16xf32>
//  CHECK-NEXT:   %[[R7:.+]] =  vector.contract {indexing_maps = [#[[$map0]], #[[$map1]], #[[$map2]]],
//  CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
//  CHECK-SAME:   %[[R3]], %[[R5]], %[[R6]] : vector<8x16xf32>, vector<2x8xf32> into vector<2x16xf32>
//  CHECK-NEXT:   %[[R8:.+]] =  vector.broadcast %[[R7]] : vector<2x16xf32> to vector<1x2x16xf32>
//  CHECK-NEXT:   %[[R9:.+]] =  vector.broadcast %[[R8]] : vector<1x2x16xf32> to vector<1x1x2x16xf32>
//  CHECK-NEXT:  return %[[R9]] : vector<1x1x2x16xf32>

#contraction_accesses2 = [
  affine_map<(m, l, i, j, k) -> (m, k, l, j)>,
  affine_map<(m, l, i, j, k) -> (m, i, k, l)>,
  affine_map<(m, l, i, j, k) -> (m, l, i, j)>
]
#contraction_trait2 = {
  indexing_maps = #contraction_accesses2,
  iterator_types = ["parallel","parallel", "parallel", "parallel", "reduction"]
}


func.func @cast_away_contraction_leading_one_dims_nonleadingunitdim_rank4(%arg0: vector<1x8x1x16xf32>, %arg1: vector<1x2x8x1xf32>, %arg2: vector<1x1x2x16xf32>) -> vector<1x1x2x16xf32> {
  %0 = vector.contract #contraction_trait2 %arg0, %arg1, %arg2  : vector<1x8x1x16xf32>, vector<1x2x8x1xf32> into vector<1x1x2x16xf32>
  return %0: vector<1x1x2x16xf32>
}

// -----
// CHECK-DAG: #[[$map0:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$map1:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$map2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: cast_away_contraction_leading_one_dims_nonleadingunitdim_rank4_acctranspose
//  CHECK-NEXT:   %[[R0:.+]] =  vector.transpose %{{.*}}, [2, 0, 1, 3] : vector<1x8x1x16xf32> to vector<1x1x8x16xf32>
//  CHECK-NEXT:   %[[R1:.+]] =  vector.transpose %{{.*}}, [3, 0, 1, 2] : vector<1x2x8x1xf32> to vector<1x1x2x8xf32>
//  CHECK-NEXT:   %[[R2:.+]] =  vector.extract %[[R0]][0, 0] : vector<1x1x8x16xf32>
//  CHECK-NEXT:   %[[R3:.+]] =  vector.extract %[[R1]][0, 0] : vector<1x1x2x8xf32>
//  CHECK-NEXT:   %[[R4:.+]] =  vector.extract %{{.*}}[0, 0] : vector<1x1x2x16xf32>
//  CHECK-NEXT:   %[[R5:.+]] =  vector.contract {indexing_maps = [#[[$map0]], #[[$map1]], #[[$map2]]],
//  CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
//  CHECK-SAME:   %[[R2]], %[[R3]], %[[R4]] : vector<8x16xf32>, vector<2x8xf32> into vector<2x16xf32>
//  CHECK-NEXT:   %[[R6:.+]] =  vector.broadcast %[[R5]] : vector<2x16xf32> to vector<1x2x16xf32>
//  CHECK-NEXT:   %[[R7:.+]] =  vector.broadcast %[[R6]] : vector<1x2x16xf32> to vector<1x1x2x16xf32>
//  CHECK-NEXT:  return %[[R7]] : vector<1x1x2x16xf32>

#contraction_accesses3 = [
  affine_map<(m, l, i, j, k) -> (m, k, l, j)>,
  affine_map<(m, l, i, j, k) -> (m, i, k, l)>,
  affine_map<(m, l, i, j, k) -> (l, m, i, j)>
]
#contraction_trait3 = {
  indexing_maps = #contraction_accesses3,
  iterator_types = ["parallel","parallel", "parallel", "parallel", "reduction"]
}

func.func @cast_away_contraction_leading_one_dims_nonleadingunitdim_rank4_acctranspose(%arg0: vector<1x8x1x16xf32>, %arg1: vector<1x2x8x1xf32>, %arg2: vector<1x1x2x16xf32>) -> vector<1x1x2x16xf32> {
  %0 = vector.contract #contraction_trait3 %arg0, %arg1, %arg2  : vector<1x8x1x16xf32>, vector<1x2x8x1xf32> into vector<1x1x2x16xf32>
  return %0: vector<1x1x2x16xf32>
}

// -----
// CHECK-LABEL: func @cast_away_extract_strided_slice_leading_one_dims
func.func @cast_away_extract_strided_slice_leading_one_dims(%arg0: vector<1x8x8xf16>) -> vector<1x1x8xf16> {
  // CHECK:     %[[SRC:.+]] = vector.extract %{{.*}}[0] : vector<1x8x8xf16>
  // CHECK: %[[EXTRACT:.+]] = vector.extract_strided_slice %[[SRC]] {offsets = [4], sizes = [1], strides = [1]} : vector<8x8xf16> to vector<1x8xf16>
  %0 = vector.extract_strided_slice %arg0 {offsets = [0, 4], sizes = [1, 1], strides = [1, 1]} : vector<1x8x8xf16> to vector<1x1x8xf16>
  // CHECK:     %[[RET:.+]] = vector.broadcast %[[EXTRACT]] : vector<1x8xf16> to vector<1x1x8xf16>
  // CHECK: return %[[RET]]
  return %0: vector<1x1x8xf16>
}

// CHECK-LABEL: func @cast_away_insert_strided_slice_leading_one_dims
func.func @cast_away_insert_strided_slice_leading_one_dims(%arg0: vector<1x8xf16>, %arg1: vector<1x8x8xf16>) -> vector<1x8x8xf16> {
  // CHECK:    %[[SRC:.+]] = vector.extract %{{.*}}[0] : vector<1x8xf16>
  // CHECK:    %[[DST:.+]] = vector.extract %{{.*}}[0] : vector<1x8x8xf16>
  // CHECK: %[[INSERT:.+]] = vector.insert_strided_slice %[[SRC]], %[[DST]] {offsets = [0, 0], strides = [1]} : vector<8xf16> into vector<8x8xf16>
  %0 = vector.insert_strided_slice %arg0, %arg1 {offsets = [0, 0, 0], strides = [1, 1]} : vector<1x8xf16> into vector<1x8x8xf16>
  // CHECK:    %[[RET:.+]] = vector.broadcast %[[INSERT]] : vector<8x8xf16> to vector<1x8x8xf16>
  // CHECK: return %[[RET]]
  return %0: vector<1x8x8xf16>
}

// CHECK-LABEL: func @cast_away_insert_strided_slice_leading_one_dims_one_element
//  CHECK-SAME: %[[ARG0:.+]]: vector<1x1xf16>, %{{.+}}: vector<1x1x1xf16>
func.func @cast_away_insert_strided_slice_leading_one_dims_one_element(%arg0: vector<1x1xf16>, %arg1: vector<1x1x1xf16>) -> vector<1x1x1xf16> {
  // CHECK: %[[EXT:.+]] = vector.extract %{{.*}}[0] : vector<1x1xf16>
  // CHECK: %[[B:.+]] = vector.broadcast %[[EXT]] : vector<1xf16> to vector<1x1x1xf16>
  %0 = vector.insert_strided_slice %arg0, %arg1 {offsets = [0, 0, 0], strides = [1, 1]} : vector<1x1xf16> into vector<1x1x1xf16>
  // CHECK: return %[[B]]
  return %0: vector<1x1x1xf16>
}

// CHECK-LABEL: func @cast_away_transfer_read_leading_one_dims
func.func @cast_away_transfer_read_leading_one_dims(%arg0: memref<1x4x8x16xf16>) -> vector<1x4xf16> {
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK: %[[F0:.+]] = arith.constant 0.000000e+00 : f16
  %f0 = arith.constant 0. : f16
  // CHECK: %[[READ:.+]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]], %[[C0]], %[[C0]]], %[[F0]] {in_bounds = [true]} : memref<1x4x8x16xf16>, vector<4xf16>
  // CHECK: %[[CAST:.+]] = vector.broadcast %[[READ]] : vector<4xf16> to vector<1x4xf16>
  %0 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %f0 {in_bounds = [true, true]} : memref<1x4x8x16xf16>, vector<1x4xf16>
  // CHECK: return %[[CAST]]
  return %0: vector<1x4xf16>
}

// CHECK-LABEL: func @cast_away_transfer_read_leading_one_dims_one_element
func.func @cast_away_transfer_read_leading_one_dims_one_element(%arg0: memref<1x1x1x1xf16>) -> vector<1x1xf16> {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0. : f16
  // CHECK: vector.broadcast %{{.+}} : vector<1xf16> to vector<1x1xf16>
  %0 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %f0 {in_bounds = [true, true]} : memref<1x1x1x1xf16>, vector<1x1xf16>
  return %0: vector<1x1xf16>
}

// CHECK-LABEL: func @cast_away_transfer_write_leading_one_dims
func.func @cast_away_transfer_write_leading_one_dims(%arg0: memref<1x4x8x16xf16>, %arg1: vector<1x4xf16>) {
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK: %[[CAST:.+]] = vector.extract %{{.*}}[0] : vector<1x4xf16>
  // CHECK: vector.transfer_write %[[CAST]], %{{.*}}[%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true]} : vector<4xf16>, memref<1x4x8x16xf16>

  vector.transfer_write %arg1, %arg0[%c0, %c0, %c0, %c0] {in_bounds = [true, true]} : vector<1x4xf16>, memref<1x4x8x16xf16>
  return
}

// CHECK-LABEL: func @cast_away_transfer_write_leading_one_dims_one_element
func.func @cast_away_transfer_write_leading_one_dims_one_element(%arg0: memref<1x1x1x1xf16>, %arg1: vector<1x1xf16>) {
  %c0 = arith.constant 0 : index
  // CHECK: vector.extract %{{.+}}[0] : vector<1x1xf16>
  vector.transfer_write %arg1, %arg0[%c0, %c0, %c0, %c0] {in_bounds = [true, true]} : vector<1x1xf16>, memref<1x1x1x1xf16>
  return
}

// CHECK-LABEL: func @cast_away_elementwise_leading_one_dims
func.func @cast_away_elementwise_leading_one_dims(
  %arg0: vector<1x1x8xf32>, %arg1: f32, %arg2: vector<1x4xf32>,
  %arg3: vector<1x4xf32>, %arg4: i1) ->
  (vector<1x1x8xf32>, vector<1x4xi1>, vector<1x4xf32>, vector<1x4xf32>) {
  // CHECK:  vector.extract %{{.*}}[0, 0] : vector<1x1x8xf32>
  // CHECK:  vector.extract %{{.*}}[0, 0] : vector<1x1x8xf32>
  // CHECK:  arith.addf %{{.*}}, %{{.*}} : vector<8xf32>
  // CHECK:  vector.broadcast %{{.*}} : vector<8xf32> to vector<1x1x8xf32>
  %0 = arith.addf %arg0, %arg0 : vector<1x1x8xf32>
  // CHECK:  vector.extract %{{.*}}[0] : vector<1x4xf32>
  // CHECK:  vector.extract %{{.*}}[0] : vector<1x4xf32>
  // CHECK:  arith.cmpf ogt, %{{.*}}, %{{.*}} : vector<4xf32>
  // CHECK:  vector.broadcast %{{.*}} : vector<4xi1> to vector<1x4xi1>
  %1 = arith.cmpf ogt, %arg2, %arg3 : vector<1x4xf32>
  // CHECK:  vector.extract %{{.*}}[0] : vector<1x4xf32>
  // CHECK:  vector.extract %{{.*}}[0] : vector<1x4xf32>
  // CHECK:  select %{{.*}}, %{{.*}}, %{{.*}} : vector<4xi1>, vector<4xf32>
  // CHECK:  vector.broadcast %{{.*}} : vector<4xf32> to vector<1x4xf32>
  %2 = arith.select %1, %arg3, %arg2 : vector<1x4xi1>, vector<1x4xf32>
  // CHECK:  vector.extract %{{.*}}[0] : vector<1x4xf32>
  // CHECK:  vector.extract %{{.*}}[0] : vector<1x4xf32>
  // CHECK:  select %arg4, %12, %{{.*}} : vector<4xf32>
  // CHECK:  vector.broadcast %{{.*}} : vector<4xf32> to vector<1x4xf32>
  %3 = arith.select %arg4, %arg3, %arg2 : vector<1x4xf32>
  return %0, %1, %2, %3: vector<1x1x8xf32>, vector<1x4xi1>, vector<1x4xf32>, vector<1x4xf32>
}

// CHECK-LABEL: func @cast_away_insert_leading_one_dims_scalar
//  CHECK-SAME: (%[[S:.+]]: f32, %[[V:.+]]: vector<1x1x4xf32>)
//       CHECK:   %[[EXTRACT:.+]] = vector.extract %[[V]][0, 0] : vector<1x1x4xf32>
//       CHECK:   %[[INSERT:.+]] = vector.insert %[[S]], %[[EXTRACT]] [0] : f32 into vector<4xf32>
//       CHECK:   %[[BCAST:.+]] = vector.broadcast %[[INSERT]] : vector<4xf32> to vector<1x1x4xf32>
//       CHECK:   return %[[BCAST]]
func.func @cast_away_insert_leading_one_dims_scalar(%s: f32, %v: vector<1x1x4xf32>) -> vector<1x1x4xf32> {
  %0 = vector.insert %s, %v [0, 0, 0] : f32 into vector<1x1x4xf32>
  return %0: vector<1x1x4xf32>
}

// CHECK-LABEL: func @cast_away_insert_leading_one_dims_rank1
//  CHECK-SAME: (%[[S:.+]]: vector<4xf32>, %[[V:.+]]: vector<1x1x4xf32>)
//       CHECK:   %[[BCAST:.+]] = vector.broadcast %[[S]] : vector<4xf32> to vector<1x1x4xf32>
//       CHECK:   return %[[BCAST]]
func.func @cast_away_insert_leading_one_dims_rank1(%s: vector<4xf32>, %v: vector<1x1x4xf32>) -> vector<1x1x4xf32> {
  %0 = vector.insert %s, %v [0, 0] : vector<4xf32> into vector<1x1x4xf32>
  return %0: vector<1x1x4xf32>
}

// CHECK-LABEL: func @cast_away_insert_leading_one_dims_rank2
//  CHECK-SAME: (%[[S:.+]]: vector<1x4xf32>, %[[V:.+]]: vector<1x1x4xf32>)
//       CHECK:   %[[EXTRACT:.+]] = vector.extract %[[S]][0] : vector<1x4xf32>
//       CHECK:   %[[BCAST:.+]] = vector.broadcast %[[EXTRACT]] : vector<4xf32> to vector<1x1x4xf32>
//       CHECK:   return %[[BCAST]]
func.func @cast_away_insert_leading_one_dims_rank2(%s: vector<1x4xf32>, %v: vector<1x1x4xf32>) -> vector<1x1x4xf32> {
  %0 = vector.insert %s, %v [0] : vector<1x4xf32> into vector<1x1x4xf32>
  return %0: vector<1x1x4xf32>
}

// CHECK-LABEL: func @cast_away_insert_leading_one_dims_non_one_dest
//  CHECK-SAME: (%[[S:.+]]: vector<1x4xf32>, %[[V:.+]]: vector<8x1x4xf32>)
//       CHECK:   %[[EXTRACT:.+]] = vector.extract %[[S]][0] : vector<1x4xf32>
//       CHECK:   %[[INSERT:.+]] = vector.insert %[[EXTRACT]], %[[V]] [5, 0] : vector<4xf32> into vector<8x1x4xf32>
//       CHECK:   return %[[INSERT]]
func.func @cast_away_insert_leading_one_dims_non_one_dest(%s: vector<1x4xf32>, %v: vector<8x1x4xf32>) -> vector<8x1x4xf32> {
  %0 = vector.insert %s, %v [5] : vector<1x4xf32> into vector<8x1x4xf32>
  return %0: vector<8x1x4xf32>
}
