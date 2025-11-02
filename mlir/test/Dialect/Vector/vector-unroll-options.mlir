// RUN: mlir-opt %s -test-vector-unrolling-patterns=unroll-based-on-type | FileCheck %s
// RUN: mlir-opt %s -test-vector-unrolling-patterns="unroll-based-on-type unroll-order=2,0,1"  | FileCheck %s --check-prefix=ORDER
// RUN: mlir-opt %s -test-vector-unrolling-patterns="unroll-based-on-type unroll-order=0,3,1,2" | FileCheck %s --check-prefix=BATCHED

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

func.func @vector_fma(%a: vector<4x4xf32>, %b: vector<4x4xf32>, %c: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.fma %a, %b, %c: vector<4x4xf32>
  return %0 : vector<4x4xf32>
}
//   CHECK-LABEL: func @vector_fma
// CHECK-COUNT-4: vector.fma %{{.+}}, %{{.+}}, %{{.+}} : vector<2x2xf32>

func.func @vector_fma_3d(%a: vector<3x2x2xf32>) -> vector<3x2x2xf32>{
  %0 = vector.fma %a, %a, %a : vector<3x2x2xf32>
  return %0 : vector<3x2x2xf32>
}
// CHECK-LABEL: func @vector_fma_3d
//  CHECK-SAME:   (%[[SRC:.*]]: vector<3x2x2xf32>) -> vector<3x2x2xf32> {
//       CHECK:   %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<3x2x2xf32>
//       CHECK:   %[[E_LHS_0:.*]] = vector.extract_strided_slice %[[SRC]] {offsets = [0, 0, 0], sizes = [1, 2, 2], strides = [1, 1, 1]} : vector<3x2x2xf32> to vector<1x2x2xf32>
//       CHECK:   %[[S_LHS_0:.*]] = vector.shape_cast %[[E_LHS_0]] : vector<1x2x2xf32> to vector<2x2xf32>
//       CHECK:   %[[E_RHS_0:.*]] = vector.extract_strided_slice %[[SRC]] {offsets = [0, 0, 0], sizes = [1, 2, 2], strides = [1, 1, 1]} : vector<3x2x2xf32> to vector<1x2x2xf32>
//       CHECK:   %[[S_RHS_0:.*]] = vector.shape_cast %[[E_RHS_0]] : vector<1x2x2xf32> to vector<2x2xf32>
//       CHECK:   %[[E_OUT_0:.*]] = vector.extract_strided_slice %[[SRC]] {offsets = [0, 0, 0], sizes = [1, 2, 2], strides = [1, 1, 1]} : vector<3x2x2xf32> to vector<1x2x2xf32>
//       CHECK:   %[[S_OUT_0:.*]] = vector.shape_cast %[[E_OUT_0]] : vector<1x2x2xf32> to vector<2x2xf32>
//       CHECK:   %[[FMA0:.*]] = vector.fma %[[S_LHS_0]], %[[S_RHS_0]], %[[S_OUT_0]] : vector<2x2xf32>
//       CHECK:   %[[I0:.*]] = vector.insert_strided_slice %[[FMA0]], %[[CST]] {offsets = [0, 0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<3x2x2xf32>
//       CHECK:   %[[E_LHS_1:.*]] = vector.extract_strided_slice %[[SRC]] {offsets = [1, 0, 0], sizes = [1, 2, 2], strides = [1, 1, 1]} : vector<3x2x2xf32> to vector<1x2x2xf32>
//       CHECK:   %[[S_LHS_1:.*]] = vector.shape_cast %[[E_LHS_1]] : vector<1x2x2xf32> to vector<2x2xf32>
//       CHECK:   %[[E_RHS_1:.*]] = vector.extract_strided_slice %[[SRC]] {offsets = [1, 0, 0], sizes = [1, 2, 2], strides = [1, 1, 1]} : vector<3x2x2xf32> to vector<1x2x2xf32>
//       CHECK:   %[[S_RHS_1:.*]] = vector.shape_cast %[[E_RHS_1]] : vector<1x2x2xf32> to vector<2x2xf32>
//       CHECK:   %[[E_OUT_1:.*]] = vector.extract_strided_slice %[[SRC]] {offsets = [1, 0, 0], sizes = [1, 2, 2], strides = [1, 1, 1]} : vector<3x2x2xf32> to vector<1x2x2xf32>
//       CHECK:   %[[S_OUT_1:.*]] = vector.shape_cast %[[E_OUT_1]] : vector<1x2x2xf32> to vector<2x2xf32>
//       CHECK:   %[[FMA1:.*]] = vector.fma %[[S_LHS_1]], %[[S_RHS_1]], %[[S_OUT_1]] : vector<2x2xf32>
//       CHECK:   %[[I1:.*]] = vector.insert_strided_slice %[[FMA1]], %[[I0]] {offsets = [1, 0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<3x2x2xf32>
//       CHECK:   %[[E_LHS_2:.*]] = vector.extract_strided_slice %[[SRC]] {offsets = [2, 0, 0], sizes = [1, 2, 2], strides = [1, 1, 1]} : vector<3x2x2xf32> to vector<1x2x2xf32>
//       CHECK:   %[[S_LHS_2:.*]] = vector.shape_cast %[[E_LHS_2]] : vector<1x2x2xf32> to vector<2x2xf32>
//       CHECK:   %[[E_RHS_2:.*]] = vector.extract_strided_slice %[[SRC]] {offsets = [2, 0, 0], sizes = [1, 2, 2], strides = [1, 1, 1]} : vector<3x2x2xf32> to vector<1x2x2xf32>
//       CHECK:   %[[S_RHS_2:.*]] = vector.shape_cast %[[E_RHS_2]] : vector<1x2x2xf32> to vector<2x2xf32>
//       CHECK:   %[[E_OUT_2:.*]] = vector.extract_strided_slice %[[SRC]] {offsets = [2, 0, 0], sizes = [1, 2, 2], strides = [1, 1, 1]} : vector<3x2x2xf32> to vector<1x2x2xf32>
//       CHECK:   %[[S_OUT_2:.*]] = vector.shape_cast %[[E_OUT_2]] : vector<1x2x2xf32> to vector<2x2xf32>
//       CHECK:   %[[FMA2:.*]] = vector.fma %[[S_LHS_2]], %[[S_RHS_2]], %[[S_OUT_2]] : vector<2x2xf32>
//       CHECK:   %[[I2:.*]] = vector.insert_strided_slice %[[FMA2]], %[[I1]] {offsets = [2, 0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<3x2x2xf32>
//       CHECK:   return %[[I2]] : vector<3x2x2xf32>

func.func @vector_multi_reduction(%v : vector<4x6xf32>, %acc: vector<4xf32>) -> vector<4xf32> {
  %0 = vector.multi_reduction #vector.kind<add>, %v, %acc [1] : vector<4x6xf32> to vector<4xf32>
  return %0 : vector<4xf32>
}
// CHECK-LABEL: func @vector_multi_reduction
//       CHECK:   %[[V0:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
//       CHECK:   %[[E0:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
//       CHECK:   %[[ACC0:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0], sizes = [2], strides = [1]} : vector<4xf32> to vector<2xf32>
//       CHECK:   %[[R0:.*]] = vector.multi_reduction <add>, %[[E0]], %[[ACC0]] [1] : vector<2x2xf32> to vector<2xf32>
//       CHECK:   %[[E1:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
//       CHECK:   %[[R1:.*]] = vector.multi_reduction <add>, %[[E1]], %[[R0]] [1] : vector<2x2xf32> to vector<2xf32>
//       CHECK:   %[[E2:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 4], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
//       CHECK:   %[[R2:.*]] = vector.multi_reduction <add>, %[[E2]], %[[R1]] [1] : vector<2x2xf32> to vector<2xf32>
//       CHECK:   %[[E3:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
//       CHECK:   %[[ACC1:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2], sizes = [2], strides = [1]} : vector<4xf32> to vector<2xf32>
//       CHECK:   %[[R3:.*]] = vector.multi_reduction <add>, %[[E3]], %[[ACC1]] [1] : vector<2x2xf32> to vector<2xf32>
//       CHECK:   %[[E4:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
//       CHECK:   %[[R4:.*]] = vector.multi_reduction <add>, %[[E4]], %[[R3]] [1] : vector<2x2xf32> to vector<2xf32>
//       CHECK:   %[[E5:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 4], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
//       CHECK:   %[[R5:.*]] = vector.multi_reduction <add>, %[[E5]], %[[R4]] [1] : vector<2x2xf32> to vector<2xf32>
//       CHECK:   %[[V1:.*]] = vector.insert_strided_slice %[[R2]], %[[V0]] {offsets = [0], strides = [1]} : vector<2xf32> into vector<4xf32>
//       CHECK:   %[[V2:.*]] = vector.insert_strided_slice %[[R5]], %[[V1]] {offsets = [2], strides = [1]} : vector<2xf32> into vector<4xf32>
//       CHECK:   return %[[V2]] : vector<4xf32>

// This is a negative test case to ensure that further unrolling is not performed. Since the vector.multi_reduction
// operation has already been unrolled, attempting additional unrolling should not be allowed.
func.func @negative_vector_multi_reduction(%v: vector<4x2xf32>, %acc: f32) -> f32 {
  %0 = vector.multi_reduction #vector.kind<add>, %v, %acc [0, 1] : vector<4x2xf32> to f32
  return %0 : f32
}
// CHECK-LABEL: func @negative_vector_multi_reduction
//  CHECK-NEXT:   %[[R0:.*]] = vector.multi_reduction <add>, %{{.*}}, %{{.*}} [0, 1] : vector<4x2xf32> to f32
//  CHECK-NEXT:   return %[[R0]] : f32

func.func @vector_reduction(%v : vector<8xf32>) -> f32 {
  %0 = vector.reduction <add>, %v : vector<8xf32> into f32
  return %0 : f32
}
// CHECK-LABEL: func @vector_reduction(
//  CHECK-SAME:     %[[v:.*]]: vector<8xf32>
//       CHECK:   %[[s0:.*]] = vector.extract_strided_slice %[[v]] {offsets = [0], sizes = [2]
//       CHECK:   %[[r0:.*]] = vector.reduction <add>, %[[s0]]
//       CHECK:   %[[s1:.*]] = vector.extract_strided_slice %[[v]] {offsets = [2], sizes = [2]
//       CHECK:   %[[r1:.*]] = vector.reduction <add>, %[[s1]]
//       CHECK:   %[[add1:.*]] = arith.addf %[[r0]], %[[r1]]
//       CHECK:   %[[s2:.*]] = vector.extract_strided_slice %[[v]] {offsets = [4], sizes = [2]
//       CHECK:   %[[r2:.*]] = vector.reduction <add>, %[[s2]]
//       CHECK:   %[[add2:.*]] = arith.addf %[[add1]], %[[r2]]
//       CHECK:   %[[s3:.*]] = vector.extract_strided_slice %[[v]] {offsets = [6], sizes = [2]
//       CHECK:   %[[r3:.*]] = vector.reduction <add>, %[[s3]]
//       CHECK:   %[[add3:.*]] = arith.addf %[[add2]], %[[r3]]
//       CHECK:   return %[[add3]]

func.func @vector_transpose(%v : vector<2x4x3x8xf32>) -> vector<2x3x8x4xf32> {
  %t = vector.transpose %v, [0, 2, 3, 1] : vector<2x4x3x8xf32> to vector<2x3x8x4xf32>
  return %t : vector<2x3x8x4xf32>
}
// CHECK-LABEL: func @vector_transpose
//       CHECK:   %[[VI:.*]] = arith.constant dense<0.000000e+00> : vector<2x3x8x4xf32>
//       CHECK:   %[[E0:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0, 0, 0], sizes = [1, 2, 3, 4], strides = [1, 1, 1, 1]} : vector<2x4x3x8xf32> to vector<1x2x3x4xf32>
//       CHECK:   %[[T0:.*]] = vector.transpose %[[E0]], [0, 2, 3, 1] : vector<1x2x3x4xf32> to vector<1x3x4x2xf32>
//       CHECK:   %[[V0:.*]] = vector.insert_strided_slice %[[T0]], %[[VI]] {offsets = [0, 0, 0, 0], strides = [1, 1, 1, 1]} : vector<1x3x4x2xf32> into vector<2x3x8x4xf32>
//       CHECK:   %[[E1:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 2, 0, 0], sizes = [1, 2, 3, 4], strides = [1, 1, 1, 1]} : vector<2x4x3x8xf32> to vector<1x2x3x4xf32>
//       CHECK:   %[[T1:.*]] = vector.transpose %[[E1]], [0, 2, 3, 1] : vector<1x2x3x4xf32> to vector<1x3x4x2xf32>
//       CHECK:   %[[V1:.*]] = vector.insert_strided_slice %[[T1]], %[[V0]] {offsets = [0, 0, 0, 2], strides = [1, 1, 1, 1]} : vector<1x3x4x2xf32> into vector<2x3x8x4xf32>
//       CHECK:   %[[E2:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0, 0, 4], sizes = [1, 2, 3, 4], strides = [1, 1, 1, 1]} : vector<2x4x3x8xf32> to vector<1x2x3x4xf32>
//       CHECK:   %[[T2:.*]] = vector.transpose %[[E2]], [0, 2, 3, 1] : vector<1x2x3x4xf32> to vector<1x3x4x2xf32>
//       CHECK:   %[[V2:.*]] = vector.insert_strided_slice %[[T2]], %[[V1]] {offsets = [0, 0, 4, 0], strides = [1, 1, 1, 1]} : vector<1x3x4x2xf32> into vector<2x3x8x4xf32>
//       CHECK:   %[[E3:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 2, 0, 4], sizes = [1, 2, 3, 4], strides = [1, 1, 1, 1]} : vector<2x4x3x8xf32> to vector<1x2x3x4xf32>
//       CHECK:   %[[T3:.*]] = vector.transpose %[[E3]], [0, 2, 3, 1] : vector<1x2x3x4xf32> to vector<1x3x4x2xf32>
//       CHECK:   %[[V3:.*]] = vector.insert_strided_slice %[[T3]], %[[V2]] {offsets = [0, 0, 4, 2], strides = [1, 1, 1, 1]} : vector<1x3x4x2xf32> into vector<2x3x8x4xf32>
//       CHECK:   %[[E4:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [1, 0, 0, 0], sizes = [1, 2, 3, 4], strides = [1, 1, 1, 1]} : vector<2x4x3x8xf32> to vector<1x2x3x4xf32>
//       CHECK:   %[[T4:.*]] = vector.transpose %[[E4]], [0, 2, 3, 1] : vector<1x2x3x4xf32> to vector<1x3x4x2xf32>
//       CHECK:   %[[V4:.*]] = vector.insert_strided_slice %[[T4]], %[[V3]] {offsets = [1, 0, 0, 0], strides = [1, 1, 1, 1]} : vector<1x3x4x2xf32> into vector<2x3x8x4xf32>
//       CHECK:   %[[E5:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [1, 2, 0, 0], sizes = [1, 2, 3, 4], strides = [1, 1, 1, 1]} : vector<2x4x3x8xf32> to vector<1x2x3x4xf32>
//       CHECK:   %[[T5:.*]] = vector.transpose %[[E5]], [0, 2, 3, 1] : vector<1x2x3x4xf32> to vector<1x3x4x2xf32>
//       CHECK:   %[[V5:.*]] = vector.insert_strided_slice %[[T5]], %[[V4]] {offsets = [1, 0, 0, 2], strides = [1, 1, 1, 1]} : vector<1x3x4x2xf32> into vector<2x3x8x4xf32>
//       CHECK:   %[[E6:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [1, 0, 0, 4], sizes = [1, 2, 3, 4], strides = [1, 1, 1, 1]} : vector<2x4x3x8xf32> to vector<1x2x3x4xf32>
//       CHECK:   %[[T6:.*]] = vector.transpose %[[E6]], [0, 2, 3, 1] : vector<1x2x3x4xf32> to vector<1x3x4x2xf32>
//       CHECK:   %[[V6:.*]] = vector.insert_strided_slice %[[T6]], %[[V5]] {offsets = [1, 0, 4, 0], strides = [1, 1, 1, 1]} : vector<1x3x4x2xf32> into vector<2x3x8x4xf32>
//       CHECK:   %[[E7:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [1, 2, 0, 4], sizes = [1, 2, 3, 4], strides = [1, 1, 1, 1]} : vector<2x4x3x8xf32> to vector<1x2x3x4xf32>
//       CHECK:   %[[T7:.*]] = vector.transpose %[[E7]], [0, 2, 3, 1] : vector<1x2x3x4xf32> to vector<1x3x4x2xf32>
//       CHECK:   %[[V7:.*]] = vector.insert_strided_slice %[[T7]], %[[V6]] {offsets = [1, 0, 4, 2], strides = [1, 1, 1, 1]} : vector<1x3x4x2xf32> into vector<2x3x8x4xf32>
//       CHECK:   return %[[V7]] : vector<2x3x8x4xf32>

// -----

func.func @vector_contract_batched(%lhs: vector<8x8x4xf32>, %rhs: vector<8x8x4xf32>, %init: vector<8x8x8xf32>) -> vector<8x8x8xf32> {
  %0 = vector.contract
         {indexing_maps = [affine_map<(d0,d1,d2,c0) -> (d0,d1,c0)>,
                           affine_map<(d0,d1,d2,c0) -> (d0,d2,c0)>,
                           affine_map<(d0,d1,d2,c0) -> (d0,d1,d2)>],
          iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
       %lhs, %rhs, %init : vector<8x8x4xf32>, vector<8x8x4xf32> into vector<8x8x8xf32>
  return %0 : vector<8x8x8xf32>
}


//    CHECK-LABEL: vector_contract_batched
// CHECK-COUNT-16: vector.contract
//      CHECK-NOT: vector.contract
//          CHECK: return

//    UNROLL-LABEL: vector_contract_batched
//  UNROLL-COUNT-1: vector.contract
//      UNROLL-NOT: vector.contract
//          UNROLL: return


//    BATCHED-LABEL: vector_contract_batched
// BATCHED-COUNT-16: vector.contract
//      BATCHED-NOT: vector.contract
//          BATCHED: return


func.func @vector_broadcast(%v: vector<4xf32>) -> vector<4x4xf32> {
  %0 = vector.broadcast %v : vector<4xf32> to vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// CHECK-LABEL: func @vector_broadcast
//  CHECK-SAME: [[arg0:%.+]]: vector<4xf32>
//       CHECK: [[c:%.+]] = arith.constant dense<0.000000e+00> : vector<4x4xf32>
//       CHECK: [[s0:%.+]] = vector.extract_strided_slice [[arg0]] {offsets = [0], sizes = [2], strides = [1]} : vector<4xf32> to vector<2xf32>
//       CHECK: [[b0:%.+]] = vector.broadcast [[s0]] : vector<2xf32> to vector<2x2xf32>
//       CHECK: [[r0:%.+]] = vector.insert_strided_slice [[b0]], [[c]] {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
//       CHECK: [[s1:%.+]] = vector.extract_strided_slice [[arg0]] {offsets = [2], sizes = [2], strides = [1]} : vector<4xf32> to vector<2xf32>
//       CHECK: [[b1:%.+]] = vector.broadcast [[s1]] : vector<2xf32> to vector<2x2xf32>
//       CHECK: [[r1:%.+]] = vector.insert_strided_slice [[b1]], [[r0]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
//       CHECK: [[s2:%.+]] = vector.extract_strided_slice [[arg0]] {offsets = [0], sizes = [2], strides = [1]} : vector<4xf32> to vector<2xf32>
//       CHECK: [[b2:%.+]] = vector.broadcast [[s2]] : vector<2xf32> to vector<2x2xf32>
//       CHECK: [[r2:%.+]] = vector.insert_strided_slice [[b2]], [[r1]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
//       CHECK: [[s3:%.+]] = vector.extract_strided_slice [[arg0]] {offsets = [2], sizes = [2], strides = [1]} : vector<4xf32> to vector<2xf32>
//       CHECK: [[b3:%.+]] = vector.broadcast [[s3]] : vector<2xf32> to vector<2x2xf32>
//       CHECK: [[r3:%.+]] = vector.insert_strided_slice [[b3]], [[r2]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
//       CHECK: return [[r3]] : vector<4x4xf32>

func.func @vector_broadcast_with_leading_unit_dim(%v: vector<1x4xf32>) -> vector<4x4xf32> {
  %0 = vector.broadcast %v : vector<1x4xf32> to vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// CHECK-LABEL: func.func @vector_broadcast_with_leading_unit_dim
//  CHECK-SAME: ([[arg0:%.+]]: vector<1x4xf32>) -> vector<4x4xf32> {
//       CHECK: [[c:%.+]] = arith.constant dense<0.000000e+00> : vector<4x4xf32>
//       CHECK: [[s0:%.+]] = vector.extract_strided_slice [[arg0]] {offsets = [0, 0], sizes = [1, 2], strides = [1, 1]} : vector<1x4xf32> to vector<1x2xf32>
//       CHECK: [[b0:%.+]] = vector.broadcast [[s0]] : vector<1x2xf32> to vector<2x2xf32>
//       CHECK: [[r0:%.+]] = vector.insert_strided_slice [[b0]], [[c]] {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
//       CHECK: [[s1:%.+]] = vector.extract_strided_slice [[arg0]] {offsets = [0, 2], sizes = [1, 2], strides = [1, 1]} : vector<1x4xf32> to vector<1x2xf32>
//       CHECK: [[b1:%.+]] = vector.broadcast [[s1]] : vector<1x2xf32> to vector<2x2xf32>
//       CHECK: [[r1:%.+]] = vector.insert_strided_slice [[b1]], [[r0]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
//       CHECK: [[s2:%.+]] = vector.extract_strided_slice [[arg0]] {offsets = [0, 0], sizes = [1, 2], strides = [1, 1]} : vector<1x4xf32> to vector<1x2xf32>
//       CHECK: [[b2:%.+]] = vector.broadcast [[s2]] : vector<1x2xf32> to vector<2x2xf32>
//       CHECK: [[r2:%.+]] = vector.insert_strided_slice [[b2]], [[r1]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
//       CHECK: [[s3:%.+]] = vector.extract_strided_slice [[arg0]] {offsets = [0, 2], sizes = [1, 2], strides = [1, 1]} : vector<1x4xf32> to vector<1x2xf32>
//       CHECK: [[b3:%.+]] = vector.broadcast [[s3]] : vector<1x2xf32> to vector<2x2xf32>
//       CHECK: [[r3:%.+]] = vector.insert_strided_slice [[b3]], [[r2]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
//       CHECK: return [[r3]] : vector<4x4xf32>

func.func @vector_broadcast_with_tailing_unit_dim(%v: vector<4x1xf32>) -> vector<4x4xf32> {
  %0 = vector.broadcast %v : vector<4x1xf32> to vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// CHECK-LABEL: func.func @vector_broadcast_with_tailing_unit_dim
//  CHECK-SAME: ([[arg0:%.+]]: vector<4x1xf32>) -> vector<4x4xf32> {
//       CHECK: [[c:%.+]] = arith.constant dense<0.000000e+00> : vector<4x4xf32>
//       CHECK: [[s0:%.+]] = vector.extract_strided_slice [[arg0]] {offsets = [0, 0], sizes = [2, 1], strides = [1, 1]} : vector<4x1xf32> to vector<2x1xf32>
//       CHECK: [[b0:%.+]] = vector.broadcast [[s0]] : vector<2x1xf32> to vector<2x2xf32>
//       CHECK: [[r0:%.+]] = vector.insert_strided_slice [[b0]], [[c]] {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
//       CHECK: [[s1:%.+]] = vector.extract_strided_slice [[arg0]] {offsets = [0, 0], sizes = [2, 1], strides = [1, 1]} : vector<4x1xf32> to vector<2x1xf32>
//       CHECK: [[b1:%.+]] = vector.broadcast [[s1]] : vector<2x1xf32> to vector<2x2xf32>
//       CHECK: [[r1:%.+]] = vector.insert_strided_slice [[b1]], [[r0]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
//       CHECK: [[s2:%.+]] = vector.extract_strided_slice [[arg0]] {offsets = [2, 0], sizes = [2, 1], strides = [1, 1]} : vector<4x1xf32> to vector<2x1xf32>
//       CHECK: [[b2:%.+]] = vector.broadcast [[s2]] : vector<2x1xf32> to vector<2x2xf32>
//       CHECK: [[r2:%.+]] = vector.insert_strided_slice [[b2]], [[r1]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
//       CHECK: [[s3:%.+]] = vector.extract_strided_slice [[arg0]] {offsets = [2, 0], sizes = [2, 1], strides = [1, 1]} : vector<4x1xf32> to vector<2x1xf32>
//       CHECK: [[b3:%.+]] = vector.broadcast [[s3]] : vector<2x1xf32> to vector<2x2xf32>
//       CHECK: [[r3:%.+]] = vector.insert_strided_slice [[b3]], [[r2]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x4xf32>
//       CHECK: return [[r3]] : vector<4x4xf32>


func.func @vector_load_2D(%mem: memref<4x4xf16>) -> vector<4x4xf16> {
  %c0 = arith.constant 0 : index
  %0 = vector.load %mem[%c0, %c0] : memref<4x4xf16>, vector<4x4xf16>
  return %0 : vector<4x4xf16>
}

// CHECK-LABEL: func.func @vector_load_2D(
// CHECK-SAME:  %[[ARG:.*]]: memref<4x4xf16>) -> vector<4x4xf16> {
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<4x4xf16>
  // CHECK: %[[V0:.*]] = vector.load %[[ARG]][%[[C0]], %[[C0]]] : memref<4x4xf16>, vector<2x2xf16>
  // CHECK: %[[V1:.*]] = vector.insert_strided_slice %[[V0]], %[[CST]] {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf16> into vector<4x4xf16>
  // CHECK: %[[V2:.*]] = vector.load %[[ARG]][%[[C0]], %[[C2]]] : memref<4x4xf16>, vector<2x2xf16>
  // CHECK: %[[V3:.*]] = vector.insert_strided_slice %[[V2]], %[[V1]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf16> into vector<4x4xf16>
  // CHECK: %[[V4:.*]] = vector.load %[[ARG]][%[[C2]], %[[C0]]] : memref<4x4xf16>, vector<2x2xf16>
  // CHECK: %[[V5:.*]] = vector.insert_strided_slice %[[V4]], %[[V3]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf16> into vector<4x4xf16>
  // CHECK: %[[V6:.*]] = vector.load %[[ARG]][%[[C2]], %[[C2]]] : memref<4x4xf16>, vector<2x2xf16>
  // CHECK: %[[V7:.*]] = vector.insert_strided_slice %[[V6]], %[[V5]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf16> into vector<4x4xf16>
  // CHECK: return %[[V7]] : vector<4x4xf16>


func.func @vector_store_2D(%mem: memref<4x4xf16>, %v: vector<4x4xf16>) {
  %c0 = arith.constant 0 : index
  vector.store %v, %mem[%c0, %c0] : memref<4x4xf16>, vector<4x4xf16>
  return
}

// CHECK-LABEL: func.func @vector_store_2D(
// CHECK-SAME:  %[[ARG0:.*]]: memref<4x4xf16>, %[[ARG1:.*]]: vector<4x4xf16>) {
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[V0:.*]] = vector.extract_strided_slice %[[ARG1]] {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf16> to vector<2x2xf16>
  // CHECK: vector.store %[[V0]], %[[ARG0]][%[[C0]], %[[C0]]] : memref<4x4xf16>, vector<2x2xf16>
  // CHECK: %[[V1:.*]] = vector.extract_strided_slice %[[ARG1]] {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf16> to vector<2x2xf16>
  // CHECK: vector.store %[[V1]], %[[ARG0]][%[[C0]], %[[C2]]] : memref<4x4xf16>, vector<2x2xf16>
  // CHECK: %[[V2:.*]] = vector.extract_strided_slice %[[ARG1]] {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf16> to vector<2x2xf16>
  // CHECK: vector.store %[[V2]], %[[ARG0]][%[[C2]], %[[C0]]] : memref<4x4xf16>, vector<2x2xf16>
  // CHECK: %[[V3:.*]] = vector.extract_strided_slice %[[ARG1]] {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x4xf16> to vector<2x2xf16>
  // CHECK: vector.store %[[V3]], %[[ARG0]][%[[C2]], %[[C2]]] : memref<4x4xf16>, vector<2x2xf16>


func.func @vector_step() -> vector<32xindex> {
    %0 = vector.step : vector<32xindex>
    return %0 : vector<32xindex>
}
// CHECK-LABEL: func @vector_step
// CHECK: %[[CST:.*]] = arith.constant dense<24> : vector<8xindex>
// CHECK: %[[CST0:.*]] = arith.constant dense<16> : vector<8xindex>
// CHECK: %[[CST1:.*]] = arith.constant dense<8> : vector<8xindex>
// CHECK: %[[CST2:.*]] = arith.constant dense<0> : vector<32xindex>
// CHECK: %[[STEP:.*]] = vector.step : vector<8xindex>
// CHECK: %[[INS0:.*]] = vector.insert_strided_slice %[[STEP]], %[[CST2]] {offsets = [0], strides = [1]} : vector<8xindex> into vector<32xindex>
// CHECK: %[[ADD1:.*]] = arith.addi %[[STEP]], %[[CST1]] : vector<8xindex>
// CHECK: %[[INS1:.*]] = vector.insert_strided_slice %[[ADD1]], %[[INS0]] {offsets = [8], strides = [1]} : vector<8xindex> into vector<32xindex>
// CHECK: %[[ADD2:.*]] = arith.addi %[[STEP]], %[[CST0]] : vector<8xindex>
// CHECK: %[[INS2:.*]] = vector.insert_strided_slice %[[ADD2]], %[[INS1]] {offsets = [16], strides = [1]} : vector<8xindex> into vector<32xindex>
// CHECK: %[[ADD3:.*]] = arith.addi %[[STEP]], %[[CST]] : vector<8xindex>
// CHECK: %[[INS3:.*]] = vector.insert_strided_slice %[[ADD3]], %[[INS2]] {offsets = [24], strides = [1]} : vector<8xindex> into vector<32xindex>
// CHECK: return %[[INS3]] : vector<32xindex>


func.func @elementwise_3D_to_2D(%v1: vector<2x2x2xf32>, %v2: vector<2x2x2xf32>) -> vector<2x2x2xf32> {
  %0 = arith.addf %v1, %v2 : vector<2x2x2xf32>
  return %0 : vector<2x2x2xf32>
}
// CHECK-LABEL: func @elementwise_3D_to_2D
//  CHECK-SAME: (%[[ARG0:.*]]: vector<2x2x2xf32>, %[[ARG1:.*]]: vector<2x2x2xf32>) -> vector<2x2x2xf32> {
//       CHECK:   %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<2x2x2xf32>
//       CHECK:   %[[E_LHS_0:.*]] = vector.extract_strided_slice %[[ARG0]] {offsets = [0, 0, 0], sizes = [1, 2, 2], strides = [1, 1, 1]} : vector<2x2x2xf32> to vector<1x2x2xf32>
//       CHECK:   %[[S_LHS_0:.*]] = vector.shape_cast %[[E_LHS_0]] : vector<1x2x2xf32> to vector<2x2xf32>
//       CHECK:   %[[E_RHS_0:.*]] = vector.extract_strided_slice %[[ARG1]] {offsets = [0, 0, 0], sizes = [1, 2, 2], strides = [1, 1, 1]} : vector<2x2x2xf32> to vector<1x2x2xf32>
//       CHECK:   %[[S_RHS_0:.*]] = vector.shape_cast %[[E_RHS_0]] : vector<1x2x2xf32> to vector<2x2xf32>
//       CHECK:   %[[ADD0:.*]] = arith.addf %[[S_LHS_0]], %[[S_RHS_0]] : vector<2x2xf32>
//       CHECK:   %[[I0:.*]] = vector.insert_strided_slice %[[ADD0]], %[[CST]] {offsets = [0, 0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<2x2x2xf32>
//       CHECK:   %[[E_LHS_1:.*]] = vector.extract_strided_slice %[[ARG0]] {offsets = [1, 0, 0], sizes = [1, 2, 2], strides = [1, 1, 1]} : vector<2x2x2xf32> to vector<1x2x2xf32>
//       CHECK:   %[[S_LHS_1:.*]] = vector.shape_cast %[[E_LHS_1]] : vector<1x2x2xf32> to vector<2x2xf32>
//       CHECK:   %[[E_RHS_1:.*]] = vector.extract_strided_slice %[[ARG1]] {offsets = [1, 0, 0], sizes = [1, 2, 2], strides = [1, 1, 1]} : vector<2x2x2xf32> to vector<1x2x2xf32>
//       CHECK:   %[[S_RHS_1:.*]] = vector.shape_cast %[[E_RHS_1]] : vector<1x2x2xf32> to vector<2x2xf32>
//       CHECK:   %[[ADD1:.*]] = arith.addf %[[S_LHS_1]], %[[S_RHS_1]] : vector<2x2xf32>
//       CHECK:   %[[I1:.*]] = vector.insert_strided_slice %[[ADD1]], %[[I0]] {offsets = [1, 0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<2x2x2xf32>
//       CHECK:   return %[[I1]] : vector<2x2x2xf32>


func.func @elementwise_4D_to_2D(%v1: vector<2x2x2x2xf32>, %v2: vector<2x2x2x2xf32>) -> vector<2x2x2x2xf32> {
  %0 = arith.addf %v1, %v2 : vector<2x2x2x2xf32>
  return %0 : vector<2x2x2x2xf32>
}

// CHECK-LABEL: func @elementwise_4D_to_2D
// CHECK-COUNT-4:   arith.addf %{{.*}}, %{{.*}} : vector<2x2xf32>
// CHECK-NOT: arith.addf
// CHECK: return
