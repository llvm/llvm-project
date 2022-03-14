// RUN: mlir-opt %s -test-vector-unrolling-patterns=unroll-based-on-type | FileCheck %s

func @vector_contract_f32(%lhs : vector<8x8xf32>, %rhs : vector<8x8xf32>,
                          %init : vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = vector.contract
         {indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                           affine_map<(i, j, k) -> (j, k)>,
                           affine_map<(i, j, k) -> (i, j)>],
          iterator_types = ["parallel", "parallel", "reduction"]}
       %lhs, %rhs, %init : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}
// CHECK-LABEL: func @vector_contract_f32
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   vector.contract {
//  CHECK-SAME:     vector<4x2xf32>, vector<4x2xf32> into vector<4x4xf32>
//       CHECK:   return

func @vector_contract_f16(%lhs : vector<8x8xf16>, %rhs : vector<8x8xf16>,
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

func @vector_fma(%a: vector<4x4xf32>, %b: vector<4x4xf32>, %c: vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = vector.fma %a, %b, %c: vector<4x4xf32>
  return %0 : vector<4x4xf32>
}
//   CHECK-LABEL: func @vector_fma
// CHECK-COUNT-4: vector.fma %{{.+}}, %{{.+}}, %{{.+}} : vector<2x2xf32>

func @vector_multi_reduction(%v : vector<4x6xf32>) -> vector<4xf32> {
  %0 = vector.multi_reduction #vector.kind<add>, %v [1] : vector<4x6xf32> to vector<4xf32>
  return %0 : vector<4xf32>
}
// CHECK-LABEL: func @vector_multi_reduction
//       CHECK:   %[[V0:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
//       CHECK:   %[[E0:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
//       CHECK:   %[[R0:.*]] = vector.multi_reduction <add>, %[[E0]] [1] : vector<2x2xf32> to vector<2xf32>
//       CHECK:   %[[E1:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
//       CHECK:   %[[R1:.*]] = vector.multi_reduction <add>, %[[E1]] [1] : vector<2x2xf32> to vector<2xf32>
//       CHECK:   %[[A0:.*]] = arith.addf %[[R1]], %[[R0]] : vector<2xf32>
//       CHECK:   %[[E2:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0, 4], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
//       CHECK:   %[[R2:.*]] = vector.multi_reduction <add>, %5 [1] : vector<2x2xf32> to vector<2xf32>
//       CHECK:   %[[A1:.*]] = arith.addf %[[R2]], %[[A0]] : vector<2xf32>
//       CHECK:   %[[E3:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
//       CHECK:   %[[R3:.*]] = vector.multi_reduction <add>, %[[E3]] [1] : vector<2x2xf32> to vector<2xf32>
//       CHECK:   %[[E4:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
//       CHECK:   %[[R4:.*]] = vector.multi_reduction <add>, %[[E4]] [1] : vector<2x2xf32> to vector<2xf32>
//       CHECK:   %[[A2:.*]] = arith.addf %[[R4]], %[[R3]] : vector<2xf32>
//       CHECK:   %[[E5:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [2, 4], sizes = [2, 2], strides = [1, 1]} : vector<4x6xf32> to vector<2x2xf32>
//       CHECK:   %[[R5:.*]] = vector.multi_reduction <add>, %[[E5]] [1] : vector<2x2xf32> to vector<2xf32>
//       CHECK:   %[[A3:.*]] = arith.addf %[[R5]], %[[A2]] : vector<2xf32>
//       CHECK:   %[[V1:.*]] = vector.insert_strided_slice %[[A1]], %[[V0]] {offsets = [0], strides = [1]} : vector<2xf32> into vector<4xf32>
//       CHECK:   %[[V2:.*]] = vector.insert_strided_slice %[[A3]], %[[V1]] {offsets = [2], strides = [1]} : vector<2xf32> into vector<4xf32>
//       CHECK:   return %[[V2]] : vector<4xf32>

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
func @vector_reduction(%v : vector<8xf32>) -> f32 {
  %0 = vector.reduction <add>, %v : vector<8xf32> into f32
  return %0 : f32
}

