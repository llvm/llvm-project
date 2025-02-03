// RUN: mlir-opt %s --transform-interpreter -canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL: func @vector_transfer_ops_0d_memref(
//  CHECK-SAME:   %[[MEM:.*]]: memref<f32>
//  CHECK-SAME:   %[[VEC:.*]]: vector<1x1x1xf32>
func.func @vector_transfer_ops_0d_memref(%mem: memref<f32>, %vec: vector<1x1x1xf32>) {
    %f0 = arith.constant 0.0 : f32

//  CHECK-NEXT:   %[[S:.*]] = vector.load %[[MEM]][] : memref<f32>, vector<f32>
    %0 = vector.transfer_read %mem[], %f0 : memref<f32>, vector<f32>

//  CHECK-NEXT:   vector.store %[[S]], %[[MEM]][] : memref<f32>, vector<f32>
    vector.transfer_write %0, %mem[] : vector<f32>, memref<f32>

//  CHECK-NEXT:   vector.store %[[VEC]], %[[MEM]][] : memref<f32>, vector<1x1x1xf32>
    vector.store %vec, %mem[] : memref<f32>, vector<1x1x1xf32>

    return
}

// CHECK-LABEL: func @vector_transfer_ops_0d_tensor(
//  CHECK-SAME:   %[[SRC:.*]]: tensor<f32>
func.func @vector_transfer_ops_0d_tensor(%src: tensor<f32>) -> vector<1xf32> {
    %f0 = arith.constant 0.0 : f32

//  CHECK:   %[[S:.*]] = vector.transfer_read %[[SRC]][]
//  CHECK:   %[[V:.*]] = vector.broadcast %[[S]] : vector<f32> to vector<1xf32>
    %res = vector.transfer_read %src[], %f0 {in_bounds = [true], permutation_map = affine_map<()->(0)>} :
      tensor<f32>, vector<1xf32>

//  CHECK-NEXT:   return %[[V]]
    return %res: vector<1xf32>
}

// transfer_read/write are lowered to vector.load/store
// CHECK-LABEL:   func @transfer_to_load(
// CHECK-SAME:      %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:      %[[IDX:.*]]: index) -> vector<4xf32> {
// CHECK-NEXT:      %[[RES:.*]] = vector.load %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32>, vector<4xf32>
// CHECK-NEXT:      vector.store  %[[RES:.*]], %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32>, vector<4xf32>
// CHECK-NEXT:      return %[[RES]] : vector<4xf32>
// CHECK-NEXT:    }

func.func @transfer_to_load(%mem : memref<8x8xf32>, %idx : index) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0 {in_bounds = [true]} : memref<8x8xf32>, vector<4xf32>
  vector.transfer_write %res, %mem[%idx, %idx] {in_bounds = [true]} : vector<4xf32>, memref<8x8xf32>
  return %res : vector<4xf32>
}

// Masked transfer_read/write inside are NOT lowered to vector.load/store
// CHECK-LABEL:   func @masked_transfer_to_load(
//  CHECK-SAME:      %[[MEM:.*]]: memref<8x8xf32>,
//  CHECK-SAME:      %[[IDX:.*]]: index,
//  CHECK-SAME:      %[[MASK:.*]]: vector<4xi1>) -> memref<8x8xf32>
//   CHECK-NOT:      vector.load 
//   CHECK-NOT:      vector.store
//       CHECK:      %[[READ:.*]] = vector.mask %[[MASK]] { vector.transfer_read %arg0[%[[IDX]], %[[IDX]]]{{.*}} : memref<8x8xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
//       CHECK:      vector.mask %[[MASK]] { vector.transfer_write %[[READ]], %[[MEM]][%[[IDX]], %[[IDX]]]{{.*}} : vector<4xf32>, memref<8x8xf32> } : vector<4xi1>

func.func @masked_transfer_to_load(%mem : memref<8x8xf32>, %idx : index, %mask : vector<4xi1>) -> memref<8x8xf32> {
  %cf0 = arith.constant 0.0 : f32
  %read = vector.mask %mask {vector.transfer_read %mem[%idx, %idx], %cf0 {in_bounds = [true]} : memref<8x8xf32>, vector<4xf32>} : vector<4xi1> -> vector<4xf32>
  vector.mask %mask {vector.transfer_write %read, %mem[%idx, %idx] {in_bounds = [true]} : vector<4xf32>, memref<8x8xf32> } : vector<4xi1> 
  return %mem : memref<8x8xf32>
}

// n-D results are also supported.
// CHECK-LABEL:   func @transfer_2D(
// CHECK-SAME:      %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:      %[[IDX:.*]]: index) -> vector<2x4xf32> {
// CHECK-NEXT:      %[[RES:.*]] = vector.load %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32>, vector<2x4xf32>
// CHECK-NEXT:      vector.store %[[RES:.*]], %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32>, vector<2x4xf32>
// CHECK-NEXT:      return %[[RES]] : vector<2x4xf32>
// CHECK-NEXT:    }

func.func @transfer_2D(%mem : memref<8x8xf32>, %idx : index) -> vector<2x4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0 {in_bounds = [true, true]} : memref<8x8xf32>, vector<2x4xf32>
  vector.transfer_write %res, %mem[%idx, %idx] {in_bounds = [true, true]} : vector<2x4xf32>, memref<8x8xf32>
  return %res : vector<2x4xf32>
}

// Vector element types are supported when the result has the same type.
// CHECK-LABEL:   func @transfer_vector_element(
// CHECK-SAME:      %[[MEM:.*]]: memref<8x8xvector<2x4xf32>>,
// CHECK-SAME:      %[[IDX:.*]]: index) -> vector<2x4xf32> {
// CHECK-NEXT:      %[[RES:.*]] = vector.load %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xvector<2x4xf32>>, vector<2x4xf32>
// CHECK-NEXT:      vector.store %[[RES:.*]], %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xvector<2x4xf32>>, vector<2x4xf32>
// CHECK-NEXT:      return %[[RES]] : vector<2x4xf32>
// CHECK-NEXT:    }

func.func @transfer_vector_element(%mem : memref<8x8xvector<2x4xf32>>, %idx : index) -> vector<2x4xf32> {
  %cf0 = arith.constant dense<0.0> : vector<2x4xf32>
  %res = vector.transfer_read %mem[%idx, %idx], %cf0 : memref<8x8xvector<2x4xf32>>, vector<2x4xf32>
  vector.transfer_write %res, %mem[%idx, %idx] : vector<2x4xf32>, memref<8x8xvector<2x4xf32>>
  return %res : vector<2x4xf32>
}

// TODO: Vector element types are not supported yet when the result has a
// different type.
// CHECK-LABEL:   func @transfer_vector_element_different_types(
// CHECK-SAME:      %[[MEM:.*]]: memref<8x8xvector<2x4xf32>>,
// CHECK-SAME:      %[[IDX:.*]]: index) -> vector<1x2x4xf32> {
// CHECK-NEXT:      %[[CF0:.*]] = arith.constant dense<0.000000e+00> : vector<2x4xf32>
// CHECK-NEXT:      %[[RES:.*]] = vector.transfer_read %[[MEM]][%[[IDX]], %[[IDX]]], %[[CF0]] {in_bounds = [true]} : memref<8x8xvector<2x4xf32>>, vector<1x2x4xf32>
// CHECK-NEXT:      vector.transfer_write %[[RES:.*]], %[[MEM]][%[[IDX]], %[[IDX]]] {in_bounds = [true]} : vector<1x2x4xf32>, memref<8x8xvector<2x4xf32>>
// CHECK-NEXT:      return %[[RES]] : vector<1x2x4xf32>
// CHECK-NEXT:    }

func.func @transfer_vector_element_different_types(%mem : memref<8x8xvector<2x4xf32>>, %idx : index) -> vector<1x2x4xf32> {
  %cf0 = arith.constant dense<0.0> : vector<2x4xf32>
  %res = vector.transfer_read %mem[%idx, %idx], %cf0 {in_bounds = [true]} : memref<8x8xvector<2x4xf32>>, vector<1x2x4xf32>
  vector.transfer_write %res, %mem[%idx, %idx] {in_bounds = [true]} : vector<1x2x4xf32>, memref<8x8xvector<2x4xf32>>
  return %res : vector<1x2x4xf32>
}

// TODO: transfer_read/write cannot be lowered because there is a dimension
// that is not guaranteed to be in-bounds.
// CHECK-LABEL:   func @transfer_2D_not_inbounds(
// CHECK-SAME:      %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:      %[[IDX:.*]]: index) -> vector<2x4xf32> {
// CHECK-NEXT:      %[[CF0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %[[RES:.*]] = vector.transfer_read %[[MEM]][%[[IDX]], %[[IDX]]], %[[CF0]] {in_bounds = [true, false]} : memref<8x8xf32>, vector<2x4xf32>
// CHECK-NEXT:      vector.transfer_write %[[RES]], %[[MEM]][%[[IDX]], %[[IDX]]] {in_bounds = [false, true]} : vector<2x4xf32>, memref<8x8xf32>
// CHECK-NEXT:      return %[[RES]] : vector<2x4xf32>
// CHECK-NEXT:    }

func.func @transfer_2D_not_inbounds(%mem : memref<8x8xf32>, %idx : index) -> vector<2x4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0 {in_bounds = [true, false]} : memref<8x8xf32>, vector<2x4xf32>
  vector.transfer_write %res, %mem[%idx, %idx] {in_bounds = [false, true]} : vector<2x4xf32>, memref<8x8xf32>
  return %res : vector<2x4xf32>
}

// TODO: transfer_read/write cannot be lowered because they are not guaranteed
// to be in-bounds.
// CHECK-LABEL:   func @transfer_not_inbounds(
// CHECK-SAME:      %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:      %[[IDX:.*]]: index) -> vector<4xf32> {
// CHECK-NEXT:      %[[CF0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %[[RES:.*]] = vector.transfer_read %[[MEM]][%[[IDX]], %[[IDX]]], %[[CF0]] : memref<8x8xf32>, vector<4xf32>
// CHECK-NEXT:      vector.transfer_write %[[RES]], %[[MEM]][%[[IDX]], %[[IDX]]] : vector<4xf32>, memref<8x8xf32>
// CHECK-NEXT:      return %[[RES]] : vector<4xf32>
// CHECK-NEXT:    }

func.func @transfer_not_inbounds(%mem : memref<8x8xf32>, %idx : index) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0 : memref<8x8xf32>, vector<4xf32>
  vector.transfer_write %res, %mem[%idx, %idx] : vector<4xf32>, memref<8x8xf32>
  return %res : vector<4xf32>
}

// CHECK-LABEL:   func @transfer_nondefault_layout(
// CHECK-SAME:      %[[MEM:.*]]: memref<8x8xf32, #{{.*}}>,
// CHECK-SAME:      %[[IDX:.*]]: index) -> vector<4xf32> {
// CHECK-NEXT:      %[[RES:.*]] = vector.load %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32, #{{.*}}>, vector<4xf32>
// CHECK-NEXT:      vector.store %[[RES]], %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32, #{{.*}}>,  vector<4xf32>
// CHECK-NEXT:      return %[[RES]] : vector<4xf32>
// CHECK-NEXT:    }

#layout = affine_map<(d0, d1) -> (d0*16 + d1)>
func.func @transfer_nondefault_layout(%mem : memref<8x8xf32, #layout>, %idx : index) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0 {in_bounds = [true]} : memref<8x8xf32, #layout>, vector<4xf32>
  vector.transfer_write %res, %mem[%idx, %idx] {in_bounds = [true]} : vector<4xf32>, memref<8x8xf32, #layout>
  return %res : vector<4xf32>
}

// TODO: transfer_read/write cannot be lowered to vector.load/store yet when the
// permutation map is not the minor identity map (up to broadcasting).
// CHECK-LABEL:   func @transfer_perm_map(
// CHECK-SAME:      %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:      %[[IDX:.*]]: index) -> vector<4xf32> {
// CHECK-NEXT:      %[[CF0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %[[RES:.*]] = vector.transfer_read %[[MEM]][%[[IDX]], %[[IDX]]], %[[CF0]] {in_bounds = [true], permutation_map = #{{.*}}} : memref<8x8xf32>, vector<4xf32>
// CHECK-NEXT:      return %[[RES]] : vector<4xf32>
// CHECK-NEXT:    }

func.func @transfer_perm_map(%mem : memref<8x8xf32>, %idx : index) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0 {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<8x8xf32>, vector<4xf32>
  return %res : vector<4xf32>
}

// Lowering of transfer_read with broadcasting is supported (note that a `load`
// is generated instead of a `vector.load`).
// CHECK-LABEL:   func @transfer_broadcasting(
// CHECK-SAME:      %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:      %[[IDX:.*]]: index) -> vector<4xf32> {
// CHECK-NEXT:      %[[LOAD:.*]] = vector.load %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32>, vector<1xf32>
// CHECK-NEXT:      %[[RES:.*]] = vector.broadcast %[[LOAD]] : vector<1xf32> to vector<4xf32>
// CHECK-NEXT:      return %[[RES]] : vector<4xf32>
// CHECK-NEXT:    }

#broadcast_1d = affine_map<(d0, d1) -> (0)>
func.func @transfer_broadcasting(%mem : memref<8x8xf32>, %idx : index) -> vector<4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0
    {in_bounds = [true], permutation_map = #broadcast_1d}
      : memref<8x8xf32>, vector<4xf32>
  return %res : vector<4xf32>
}

// CHECK-LABEL:   func @transfer_scalar(
// CHECK-SAME:      %[[MEM:.*]]: memref<?x?xf32>,
// CHECK-SAME:      %[[IDX:.*]]: index) -> vector<1xf32> {
// CHECK-NEXT:      %[[RES:.*]] = vector.load %[[MEM]][%[[IDX]], %[[IDX]]] : memref<?x?xf32>, vector<1xf32>
// CHECK-NEXT:      return %[[RES]] : vector<1xf32>
// CHECK-NEXT:    }
func.func @transfer_scalar(%mem : memref<?x?xf32>, %idx : index) -> vector<1xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0 {in_bounds = [true]} : memref<?x?xf32>, vector<1xf32>
  return %res : vector<1xf32>
}

// An example with two broadcasted dimensions.
// CHECK-LABEL:   func @transfer_broadcasting_2D(
// CHECK-SAME:      %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:      %[[IDX:.*]]: index) -> vector<4x4xf32> {
// CHECK-NEXT:      %[[LOAD:.*]] = vector.load %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32>, vector<1x1xf32>
// CHECK-NEXT:      %[[RES:.*]] = vector.broadcast %[[LOAD]] : vector<1x1xf32> to vector<4x4xf32>
// CHECK-NEXT:      return %[[RES]] : vector<4x4xf32>
// CHECK-NEXT:    }

#broadcast_2d = affine_map<(d0, d1) -> (0, 0)>
func.func @transfer_broadcasting_2D(%mem : memref<8x8xf32>, %idx : index) -> vector<4x4xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx], %cf0
    {in_bounds = [true, true], permutation_map = #broadcast_2d}
      : memref<8x8xf32>, vector<4x4xf32>
  return %res : vector<4x4xf32>
}

// More complex broadcasting case (here a `vector.load` is generated).
// CHECK-LABEL:   func @transfer_broadcasting_complex(
// CHECK-SAME:      %[[MEM:.*]]: memref<10x20x30x8x8xf32>,
// CHECK-SAME:      %[[IDX:.*]]: index) -> vector<3x2x4x5xf32> {
// CHECK-NEXT:      %[[LOAD:.*]] = vector.load %[[MEM]][%[[IDX]], %[[IDX]], %[[IDX]], %[[IDX]], %[[IDX]]] : memref<10x20x30x8x8xf32>, vector<3x1x1x5xf32>
// CHECK-NEXT:      %[[RES:.*]] = vector.broadcast %[[LOAD]] : vector<3x1x1x5xf32> to vector<3x2x4x5xf32>
// CHECK-NEXT:      return %[[RES]] : vector<3x2x4x5xf32>
// CHECK-NEXT:    }

#broadcast_2d_in_4d = affine_map<(d0, d1, d2, d3, d4) -> (d1, 0, 0, d4)>
func.func @transfer_broadcasting_complex(%mem : memref<10x20x30x8x8xf32>, %idx : index) -> vector<3x2x4x5xf32> {
  %cf0 = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%idx, %idx, %idx, %idx, %idx], %cf0
    {in_bounds = [true, true, true, true], permutation_map = #broadcast_2d_in_4d}
      : memref<10x20x30x8x8xf32>, vector<3x2x4x5xf32>
  return %res : vector<3x2x4x5xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.lower_transfer max_transfer_rank = 99
      transform.apply_patterns.vector.transfer_permutation_patterns
    } : !transform.op<"func.func">
    transform.yield
  }
}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d1, d0, 0, 0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, d0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3, d1, 0, 0)>
#map3 = affine_map<(d0, d1) -> (d1, d0, 0, 0)>
#map4 = affine_map<(d0, d1) -> (0, d1, 0, d0)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3, d0)>
#map6 = affine_map<(d0, d1) -> (0)>

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, 0, d3)>

// CHECK-LABEL: func @transfer_read_permutations
func.func @transfer_read_permutations(%mem_0 : memref<?x?xf32>, %mem_1 : memref<?x?x?x?xf32>, %m: i1)
    -> (vector<7x14x8x16xf32>, vector<7x14x8x16xf32>, vector<7x14x8x16xf32>,
       vector<7x14x8x16xf32>, vector<7x14x8x16xf32>, vector<7x14x8x16xf32>, vector<8xf32>) {
// CHECK-DAG: %[[CF0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index

// CHECK: %[[MASK0:.*]] = vector.splat %{{.*}} : vector<14x7xi1>
  %mask0 = vector.splat %m : vector<14x7xi1>
  %0 = vector.transfer_read %mem_1[%c0, %c0, %c0, %c0], %cst, %mask0 {in_bounds = [true, false, true, true], permutation_map = #map0} : memref<?x?x?x?xf32>, vector<7x14x8x16xf32>
// CHECK: vector.transfer_read {{.*}} %[[MASK0]] {in_bounds = [false, true, true, true], permutation_map = #[[$MAP0]]} : memref<?x?x?x?xf32>, vector<14x7x8x16xf32>
// CHECK: vector.transpose %{{.*}}, [1, 0, 2, 3] : vector<14x7x8x16xf32> to vector<7x14x8x16xf32>

// CHECK: %[[MASK1:.*]] = vector.splat %{{.*}} : vector<16x14xi1>
  %mask1 = vector.splat %m : vector<16x14xi1>
  %1 = vector.transfer_read %mem_1[%c0, %c0, %c0, %c0], %cst, %mask1 {in_bounds = [true, false, true, false], permutation_map = #map1} : memref<?x?x?x?xf32>, vector<7x14x8x16xf32>
// CHECK: vector.transfer_read {{.*}} %[[MASK1]] {in_bounds = [false, false, true, true], permutation_map = #[[$MAP0]]} : memref<?x?x?x?xf32>, vector<16x14x7x8xf32>
// CHECK: vector.transpose %{{.*}}, [2, 1, 3, 0] : vector<16x14x7x8xf32> to vector<7x14x8x16xf32>

// CHECK: %[[MASK3:.*]] = vector.splat %{{.*}} : vector<14x7xi1>
  %mask2 = vector.splat %m : vector<14x7xi1>
  %2 = vector.transfer_read %mem_1[%c0, %c0, %c0, %c0], %cst, %mask2 {in_bounds = [true, false, true, true], permutation_map = #map2} : memref<?x?x?x?xf32>, vector<7x14x8x16xf32>
// CHECK: vector.transfer_read {{.*}} %[[MASK3]] {in_bounds = [false, true, true], permutation_map = #[[$MAP1]]} : memref<?x?x?x?xf32>, vector<14x16x7xf32>
// CHECK: vector.broadcast %{{.*}} : vector<14x16x7xf32> to vector<8x14x16x7xf32>
// CHECK: vector.transpose %{{.*}}, [3, 1, 0, 2] : vector<8x14x16x7xf32> to vector<7x14x8x16xf32>

  %3 = vector.transfer_read %mem_0[%c0, %c0], %cst {in_bounds = [false, false, true, true], permutation_map = #map3} : memref<?x?xf32>, vector<7x14x8x16xf32>
// CHECK: vector.transfer_read %{{.*}}[%[[C0]], %[[C0]]], %[[CF0]] : memref<?x?xf32>, vector<14x7xf32>
// CHECK: vector.broadcast %{{.*}} : vector<14x7xf32> to vector<8x16x14x7xf32>
// CHECK: vector.transpose %{{.*}}, [3, 2, 0, 1] : vector<8x16x14x7xf32> to vector<7x14x8x16xf32>

  %4 = vector.transfer_read %mem_0[%c0, %c0], %cst {in_bounds = [true, false, true, false], permutation_map = #map4} : memref<?x?xf32>, vector<7x14x8x16xf32>
// CHECK: vector.transfer_read %{{.*}}[%[[C0]], %[[C0]]], %[[CF0]] : memref<?x?xf32>, vector<16x14xf32>
// CHECK: vector.broadcast %{{.*}} : vector<16x14xf32> to vector<7x8x16x14xf32>
// CHECK: vector.transpose %{{.*}}, [0, 3, 1, 2] : vector<7x8x16x14xf32> to vector<7x14x8x16xf32>

  %5 = vector.transfer_read %mem_1[%c0, %c0, %c0, %c0], %cst {permutation_map = #map5} : memref<?x?x?x?xf32>, vector<7x14x8x16xf32>
// CHECK: vector.transfer_read %{{.*}}[%[[C0]], %[[C0]], %[[C0]], %[[C0]]], %[[CF0]] : memref<?x?x?x?xf32>, vector<16x14x7x8xf32>
// CHECK: vector.transpose %{{.*}}, [2, 1, 3, 0] : vector<16x14x7x8xf32> to vector<7x14x8x16xf32>

  %6 = vector.transfer_read %mem_0[%c0, %c0], %cst {in_bounds = [true], permutation_map = #map6} : memref<?x?xf32>, vector<8xf32>
// CHECK: vector.load %{{.*}}[%[[C0]], %[[C0]]] : memref<?x?xf32>, vector<1xf32>
// CHECK: vector.broadcast %{{.*}} : vector<1xf32> to vector<8xf32>

  return %0, %1, %2, %3, %4, %5, %6 : vector<7x14x8x16xf32>, vector<7x14x8x16xf32>,
         vector<7x14x8x16xf32>, vector<7x14x8x16xf32>, vector<7x14x8x16xf32>,
         vector<7x14x8x16xf32>, vector<8xf32>
}

// CHECK-LABEL: func @transfer_write_permutations_tensor_masked
// CHECK-SAME:    %[[DST:.*]]: tensor<?x?x?x?xf32>
// CHECK-SAME:    %[[VEC:.*]]: vector<7x14x8x16xf32>
// CHECK-SAME:    %[[M:.*]]: i1
func.func @transfer_write_permutations_tensor_masked(
    %dst : tensor<?x?x?x?xf32>,
    %vec : vector<7x14x8x16xf32>, %m: i1) -> tensor<?x?x?x?xf32> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index

  // CHECK: %[[MASK:.*]] = vector.splat %[[M]] : vector<16x14x7x8xi1>
  %mask0 = vector.splat %m : vector<16x14x7x8xi1>
  %res = vector.transfer_write %vec, %dst[%c0, %c0, %c0, %c0], %mask0 {in_bounds = [true, false, false, true], permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3, d0)>} : vector<7x14x8x16xf32>, tensor<?x?x?x?xf32>
  // CHECK: %[[NEW_VEC0:.*]] = vector.transpose %{{.*}} [3, 1, 0, 2] : vector<7x14x8x16xf32> to vector<16x14x7x8xf32>
  // CHECK: %[[NEW_RES0:.*]] = vector.transfer_write %[[NEW_VEC0]], %[[DST]][%c0, %c0, %c0, %c0], %[[MASK]] {in_bounds = [true, false, true, false]} : vector<16x14x7x8xf32>, tensor<?x?x?x?xf32>

  return %res : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: func @transfer_write_permutations_memref
// CHECK-SAME:    %[[MEM:.*]]: memref<?x?x?x?xf32>
// CHECK-SAME:    %[[VEC:.*]]: vector<8x16xf32>
func.func @transfer_write_permutations_memref(
    %mem : memref<?x?x?x?xf32>, %vec : vector<8x16xf32>) {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index

  vector.transfer_write %vec, %mem[%c0, %c0, %c0, %c0] {permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>} : vector<8x16xf32>, memref<?x?x?x?xf32>
  // CHECK: %[[NEW_VEC0:.*]] = vector.transpose %{{.*}} [1, 0] : vector<8x16xf32> to vector<16x8xf32>
  // CHECK: vector.transfer_write %[[NEW_VEC0]], %[[MEM]][%c0, %c0, %c0, %c0] : vector<16x8xf32>, memref<?x?x?x?xf32>

  return
}

// CHECK-LABEL: func @transfer_write_broadcast_unit_dim_tensor
// CHECK-SAME:    %[[DST0:.*]]: tensor<?x?x?x?xf32>
// CHECK-SAME:    %[[VEC0:.*]]: vector<14x8x16xf32>
func.func @transfer_write_broadcast_unit_dim_tensor(
    %dst_0 : tensor<?x?x?x?xf32>, %vec_0 : vector<14x8x16xf32>) -> tensor<?x?x?x?xf32> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index

  %res = vector.transfer_write %vec_0, %dst_0[%c0, %c0, %c0, %c0] {in_bounds = [false, false, true], permutation_map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>} : vector<14x8x16xf32>, tensor<?x?x?x?xf32>
  // CHECK: %[[NEW_VEC0:.*]] = vector.broadcast %{{.*}} : vector<14x8x16xf32> to vector<1x14x8x16xf32>
  // CHECK: %[[NEW_VEC1:.*]] = vector.transpose %[[NEW_VEC0]], [1, 2, 0, 3] : vector<1x14x8x16xf32> to vector<14x8x1x16xf32>
  // CHECK: %[[NEW_RES0:.*]] = vector.transfer_write %[[NEW_VEC1]], %[[DST0]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {in_bounds = [false, false, true, true]} : vector<14x8x1x16xf32>, tensor<?x?x?x?xf32>

  return %res : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: func @transfer_write_broadcast_unit_dim_memref
// CHECK-SAME:    %[[MEM0:.*]]: memref<?x?x?x?xf32>
// CHECK-SAME:    %[[VEC0:.*]]: vector<8x16xf32>
func.func @transfer_write_broadcast_unit_dim_memref(
    %mem_0 : memref<?x?x?x?xf32>, %vec_0 : vector<8x16xf32>) {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index

  vector.transfer_write %vec_0, %mem_0[%c0, %c0, %c0, %c0] {permutation_map = affine_map<(d0, d1, d2, d3) -> (d1, d2)>} : vector<8x16xf32>, memref<?x?x?x?xf32>
  // CHECK: %[[NEW_VEC0:.*]] = vector.broadcast %{{.*}} : vector<8x16xf32> to vector<1x8x16xf32>
  // CHECK: %[[NEW_VEC1:.*]] = vector.transpose %[[NEW_VEC0]], [1, 2, 0] : vector<1x8x16xf32> to vector<8x16x1xf32>
  // CHECK: vector.transfer_write %[[NEW_VEC1]], %[[MEM0]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {in_bounds = [false, false, true]} : vector<8x16x1xf32>, memref<?x?x?x?xf32>

  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.lower_transfer max_transfer_rank = 99
      transform.apply_patterns.vector.transfer_permutation_patterns
    } : !transform.op<"func.func">
    transform.yield
  }
}

// -----

/// Verify vector.maskedload and vector.maskedstore ops that operate on 1-D
/// vectors aren't generated for rank > 1 transfer ops.

// CHECK-LABEL: @transfer_2D_masked
// CHECK-NOT:     vector.maskedload
// CHECK-NOT:     vector.maskedstore
// CHECK:         vector.transfer_read
// CHECK:         vector.transfer_write
func.func @transfer_2D_masked(%mem : memref<?x?xf32>, %mask : vector<2x4xi1>) -> vector<2x4xf32> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %res = vector.transfer_read %mem[%c0, %c0], %pad, %mask {in_bounds = [true, true]} : memref<?x?xf32>, vector<2x4xf32>
  vector.transfer_write %res, %mem[%c0, %c0], %mask {in_bounds = [true, true]} : vector<2x4xf32>, memref<?x?xf32>
  return %res : vector<2x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.lower_transfer max_transfer_rank = 2
    } : !transform.op<"func.func">
    transform.yield
  }
}
