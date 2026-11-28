// RUN: mlir-opt %s -test-vector-unrolling-patterns -split-input-file | FileCheck %s
// RUN: mlir-opt %s -test-vector-unrolling-patterns="unroll-order=2,0,1" -split-input-file | FileCheck %s --check-prefix=ORDER

func.func @vector_contract_f32(%lhs : vector<8x4xf32>, %rhs : vector<8x4xf32>,
                          %init : vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = vector.contract
         {indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                           affine_map<(i, j, k) -> (j, k)>,
                           affine_map<(i, j, k) -> (i, j)>],
          iterator_types = ["parallel", "parallel", "reduction"]}
       %lhs, %rhs, %init : vector<8x4xf32>, vector<8x4xf32> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}
// CHECK-LABEL: func @vector_contract_f32
// CHECK-SAME: [[arg0:%.+]]: vector<8x4xf32>, [[arg1:%.+]]: vector<8x4xf32>, [[arg2:%.+]]: vector<8x8xf32>

//       CHECK:   [[a:%.+]] = vector.extract_strided_slice [[arg0]]
//  CHECK-SAME:   offsets = [0, 0]
//       CHECK:   [[b:%.+]] = vector.extract_strided_slice [[arg1]]
//  CHECK-SAME:   offsets = [0, 0]
//       CHECK:   [[c:%.+]] = vector.extract_strided_slice [[arg2]]
//  CHECK-SAME:   offsets = [0, 0]
//       CHECK:   [[accum1:%.+]] = vector.contract {{{.*}}} [[a]], [[b]], [[c]]
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>

//       CHECK:   [[a:%.+]] = vector.extract_strided_slice [[arg0]]
//  CHECK-SAME:   offsets = [0, 2]
//       CHECK:   [[b:%.+]] = vector.extract_strided_slice [[arg1]]
//  CHECK-SAME:   offsets = [0, 2]
//       CHECK:   [[accum2:%.+]] = vector.contract {{{.*}}} [[a]], [[b]], [[accum1]]
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>

//       CHECK:   [[a:%.+]] = vector.extract_strided_slice [[arg0]]
//  CHECK-SAME:   offsets = [0, 0]
//       CHECK:   [[b:%.+]] = vector.extract_strided_slice [[arg1]]
//  CHECK-SAME:   offsets = [4, 0]
//       CHECK:   [[c:%.+]] = vector.extract_strided_slice [[arg2]]
//  CHECK-SAME:   offsets = [0, 4]
//       CHECK:   [[accum3:%.+]] = vector.contract {{{.*}}} [[a]], [[b]], [[c]]
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>

//       CHECK:   [[a:%.+]] = vector.extract_strided_slice [[arg0]]
//  CHECK-SAME:   offsets = [0, 2]
//       CHECK:   [[b:%.+]] = vector.extract_strided_slice [[arg1]]
//  CHECK-SAME:   offsets = [4, 2]
//       CHECK:   [[accum4:%.+]] = vector.contract {{{.*}}} [[a]], [[b]], [[accum3]]
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>

//       CHECK:   [[a:%.+]] = vector.extract_strided_slice [[arg0]]
//  CHECK-SAME:   offsets = [4, 0]
//       CHECK:   [[b:%.+]] = vector.extract_strided_slice [[arg1]]
//  CHECK-SAME:   offsets = [0, 0]
//       CHECK:   [[c:%.+]] = vector.extract_strided_slice [[arg2]]
//  CHECK-SAME:   offsets = [4, 0]
//       CHECK:   [[accum5:%.+]] = vector.contract {{{.*}}} [[a]], [[b]], [[c]]
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>

//       CHECK:   [[a:%.+]] = vector.extract_strided_slice [[arg0]]
//  CHECK-SAME:   offsets = [4, 2]
//       CHECK:   [[b:%.+]] = vector.extract_strided_slice [[arg1]]
//  CHECK-SAME:   offsets = [0, 2]
//       CHECK:   [[accum6:%.+]] = vector.contract {{{.*}}} [[a]], [[b]], [[accum5]]
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>

//       CHECK:   [[a:%.+]] = vector.extract_strided_slice [[arg0]]
//  CHECK-SAME:   offsets = [4, 0]
//       CHECK:   [[b:%.+]] = vector.extract_strided_slice [[arg1]]
//  CHECK-SAME:   offsets = [4, 0]
//       CHECK:   [[c:%.+]] = vector.extract_strided_slice [[arg2]]
//  CHECK-SAME:   offsets = [4, 4]
//       CHECK:   [[accum7:%.+]] = vector.contract {{{.*}}} [[a]], [[b]], [[c]]
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>

//       CHECK:   [[a:%.+]] = vector.extract_strided_slice [[arg0]]
//  CHECK-SAME:   offsets = [4, 2]
//       CHECK:   [[b:%.+]] = vector.extract_strided_slice [[arg1]]
//  CHECK-SAME:   offsets = [4, 2]
//       CHECK:   [[accum8:%.+]] = vector.contract {{{.*}}} [[a]], [[b]], [[accum7]]
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>

//       CHECK:   return

// ORDER-LABEL: func @vector_contract_f32
// ORDER-SAME: [[arg0:%.+]]: vector<8x4xf32>, [[arg1:%.+]]: vector<8x4xf32>, [[arg2:%.+]]: vector<8x8xf32>

//       ORDER:   [[a:%.+]] = vector.extract_strided_slice [[arg0]]
//  ORDER-SAME:   offsets = [0, 0]
//       ORDER:   [[b:%.+]] = vector.extract_strided_slice [[arg1]]
//  ORDER-SAME:   offsets = [0, 0]
//       ORDER:   [[c:%.+]] = vector.extract_strided_slice [[arg2]]
//  ORDER-SAME:   offsets = [0, 0]
//       ORDER:   [[accum1:%.+]] = vector.contract {{{.*}}} [[a]], [[b]], [[c]]
//  ORDER-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>

//       ORDER:   [[a:%.+]] = vector.extract_strided_slice [[arg0]]
//  ORDER-SAME:   offsets = [0, 0]
//       ORDER:   [[b:%.+]] = vector.extract_strided_slice [[arg1]]
//  ORDER-SAME:   offsets = [4, 0]
//       ORDER:   [[c:%.+]] = vector.extract_strided_slice [[arg2]]
//  ORDER-SAME:   offsets = [0, 4]
//       ORDER:   [[accum2:%.+]] = vector.contract {{{.*}}} [[a]], [[b]], [[c]]
//  ORDER-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>

//       ORDER:   [[a:%.+]] = vector.extract_strided_slice [[arg0]]
//  ORDER-SAME:   offsets = [4, 0]
//       ORDER:   [[b:%.+]] = vector.extract_strided_slice [[arg1]]
//  ORDER-SAME:   offsets = [0, 0]
//       ORDER:   [[c:%.+]] = vector.extract_strided_slice [[arg2]]
//  ORDER-SAME:   offsets = [4, 0]
//       ORDER:   [[accum3:%.+]] = vector.contract {{{.*}}} [[a]], [[b]], [[c]]
//  ORDER-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>

//       ORDER:   [[a:%.+]] = vector.extract_strided_slice [[arg0]]
//  ORDER-SAME:   offsets = [4, 0]
//       ORDER:   [[b:%.+]] = vector.extract_strided_slice [[arg1]]
//  ORDER-SAME:   offsets = [4, 0]
//       ORDER:   [[c:%.+]] = vector.extract_strided_slice [[arg2]]
//  ORDER-SAME:   offsets = [4, 4]
//       ORDER:   [[accum4:%.+]] = vector.contract {{{.*}}} [[a]], [[b]], [[c]]
//  ORDER-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>

//       ORDER:   [[a:%.+]] = vector.extract_strided_slice [[arg0]]
//  ORDER-SAME:   offsets = [0, 2]
//       ORDER:   [[b:%.+]] = vector.extract_strided_slice [[arg1]]
//  ORDER-SAME:   offsets = [0, 2]
//       ORDER:   [[accum5:%.+]] = vector.contract {{{.*}}} [[a]], [[b]], [[accum1]]
//  ORDER-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>

//       ORDER:   [[a:%.+]] = vector.extract_strided_slice [[arg0]]
//  ORDER-SAME:   offsets = [0, 2]
//       ORDER:   [[b:%.+]] = vector.extract_strided_slice [[arg1]]
//  ORDER-SAME:   offsets = [4, 2]
//       ORDER:   [[accum6:%.+]] = vector.contract {{{.*}}} [[a]], [[b]], [[accum2]]
//  ORDER-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>

//       ORDER:   [[a:%.+]] = vector.extract_strided_slice [[arg0]]
//  ORDER-SAME:   offsets = [4, 2]
//       ORDER:   [[b:%.+]] = vector.extract_strided_slice [[arg1]]
//  ORDER-SAME:   offsets = [0, 2]
//       ORDER:   [[accum7:%.+]] = vector.contract {{{.*}}} [[a]], [[b]], [[accum3]]
//  ORDER-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>

//       ORDER:   [[a:%.+]] = vector.extract_strided_slice [[arg0]]
//  ORDER-SAME:   offsets = [4, 2]
//       ORDER:   [[b:%.+]] = vector.extract_strided_slice [[arg1]]
//  ORDER-SAME:   offsets = [4, 2]
//       ORDER:   [[accum8:%.+]] = vector.contract {{{.*}}} [[a]], [[b]], [[accum4]]
//  ORDER-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>

//       ORDER:   return

// -----

func.func @vector_contract_f16(%lhs : vector<8x8xf16>, %rhs : vector<8x8xf16>,
                          %init : vector<8x8xf16>) -> vector<8x8xf16> {
  %0 = vector.contract
         {indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                           affine_map<(i, j, k) -> (j, k)>,
                           affine_map<(i, j, k) -> (i, j)>],
          iterator_types = ["parallel", "parallel", "reduction"]}
       %lhs, %rhs, %init : vector<8x8xf16>, vector<8x8xf16> into vector<8x8xf16>
  return %0 : vector<8x8xf16>
}
// CHECK-LABEL: func @vector_contract_f16
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x4xf16>, vector<4x4xf16> into vector<4x4xf16>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x4xf16>, vector<4x4xf16> into vector<4x4xf16>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x4xf16>, vector<4x4xf16> into vector<4x4xf16>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x4xf16>, vector<4x4xf16> into vector<4x4xf16>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x4xf16>, vector<4x4xf16> into vector<4x4xf16>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x4xf16>, vector<4x4xf16> into vector<4x4xf16>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x4xf16>, vector<4x4xf16> into vector<4x4xf16>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x4xf16>, vector<4x4xf16> into vector<4x4xf16>
//       CHECK:   return

// -----

// Region-masked ops are not unrolled: the enclosing mask would not be sliced.

func.func @masked_vector_contract(%lhs : vector<8x4xf32>, %rhs : vector<8x4xf32>,
                                  %init : vector<8x8xf32>, %mask : vector<8x8x4xi1>) -> vector<8x8xf32> {
  %0 = vector.mask %mask { vector.contract
         {indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                           affine_map<(i, j, k) -> (j, k)>,
                           affine_map<(i, j, k) -> (i, j)>],
          iterator_types = ["parallel", "parallel", "reduction"]}
       %lhs, %rhs, %init : vector<8x4xf32>, vector<8x4xf32> into vector<8x8xf32> } : vector<8x8x4xi1> -> vector<8x8xf32>
  return %0 : vector<8x8xf32>
}
// CHECK-LABEL: func @masked_vector_contract
//  CHECK-NEXT:   %[[R:.*]] = vector.mask %{{.*}} { vector.contract {{.*}} : vector<8x4xf32>, vector<8x4xf32> into vector<8x8xf32> }
//  CHECK-NEXT:   return %[[R]]

// -----

func.func @masked_vector_multi_reduction(%v : vector<4x6xf32>, %acc: vector<4xf32>, %mask : vector<4x6xi1>) -> vector<4xf32> {
  %0 = vector.mask %mask { vector.multi_reduction #vector.kind<add>, %v, %acc [1] : vector<4x6xf32> to vector<4xf32> } : vector<4x6xi1> -> vector<4xf32>
  return %0 : vector<4xf32>
}
// CHECK-LABEL: func @masked_vector_multi_reduction
//  CHECK-NEXT:   %[[R:.*]] = vector.mask %{{.*}} { vector.multi_reduction <add>, {{.*}} : vector<4x6xf32> to vector<4xf32> }
//  CHECK-NEXT:   return %[[R]]
