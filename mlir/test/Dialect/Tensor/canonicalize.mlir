// RUN: mlir-opt %s -split-input-file -canonicalize | FileCheck %s

// Checks that NOP casts are removed.
// CHECK-LABEL: cast_values
func @cast_values(%arg0: tensor<*xi32>) -> tensor<2xi32> {
  // NOP cast
  %0 = tensor.cast %arg0 : tensor<*xi32> to tensor<*xi32>
  // CHECK-NEXT: %[[RET:.*]] = tensor.cast %arg0 : tensor<*xi32> to tensor<2xi32>
  %2 = tensor.cast %0 : tensor<*xi32> to tensor<2xi32>
  // NOP cast
  %4 = tensor.cast %2 : tensor<2xi32> to tensor<2xi32>
  // CHECK-NEXT: return %[[RET]] : tensor<2xi32>
  return %4 : tensor<2xi32>
}

// -----

// CHECK-LABEL: @tensor.cast_chain_ok
// CHECK-SAME: %[[IN:.*]]: tensor<*xi32>
func @tensor.cast_chain_ok(%input: tensor<*xi32>) -> tensor<4x8xi32> {
  // CHECK-NEXT: %[[RES:.*]] = tensor.cast %[[IN]] : tensor<*xi32> to tensor<4x8xi32>
  %0 = tensor.cast %input : tensor<*xi32> to tensor<4x?xi32>
  %1 = tensor.cast %0 : tensor<4x?xi32> to tensor<4x8xi32>
  // CHECK-NEXT: return %[[RES]]
  return %1 : tensor<4x8xi32>
}

// -----

// CHECK-LABEL: @tensor.cast_chain_regain
// CHECK-SAME: %[[IN:.*]]: tensor<4xi32>
func @tensor.cast_chain_regain(%input: tensor<4xi32>) -> tensor<4xi32> {
  %0 = tensor.cast %input : tensor<4xi32> to tensor<?xi32>
  %1 = tensor.cast %0 : tensor<?xi32> to tensor<4xi32>
  // CHECK-NEXT: return %[[IN]]
  return %1 : tensor<4xi32>
}

// -----

// CHECK-LABEL: @tensor.cast_chain_keep
// CHECK-SAME: %[[IN:.*]]: tensor<?x?xi32>
func @tensor.cast_chain_keep(%input: tensor<?x?xi32>) -> tensor<?x8xi32> {
  // CHECK-NEXT: %[[C1:.*]] = tensor.cast %[[IN]]
  %0 = tensor.cast %input : tensor<?x?xi32> to tensor<4x?xi32>
  // CHECK-NEXT: %[[C2:.*]] = tensor.cast %[[C1]]
  %1 = tensor.cast %0 : tensor<4x?xi32> to tensor<?x8xi32>
  // CHECK-NEXT: return %[[C2]]
  return %1 : tensor<?x8xi32>
}

// -----

// CHECK-LABEL: @tensor.cast_chain_invalid
// CHECK-SAME: %[[IN:.*]]: tensor<4x8xi32>
func @tensor.cast_chain_invalid(%input: tensor<4x8xi32>) -> tensor<8x4xi32> {
  // CHECK-NEXT: %[[C1:.*]] = tensor.cast %[[IN]]
  %0 = tensor.cast %input : tensor<4x8xi32> to tensor<?x?xi32>
  // CHECK-NEXT: %[[C2:.*]] = tensor.cast %[[C1]]
  %1 = tensor.cast %0 : tensor<?x?xi32> to tensor<8x4xi32>
  // CHECK-NEXT: return %[[C2]]
  return %1 : tensor<8x4xi32>
}

// -----

// CHECK-LABEL: func @fold_extract
func @fold_extract(%arg0 : index) -> (f32, f16, f16, i32, complex<f32>) {
  %const_0 = arith.constant 0 : index
  %const_1 = arith.constant 1 : index
  %const_3 = arith.constant 3 : index
  // CHECK-DAG: [[C64:%.+]] = arith.constant 64 : i32
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0.{{0*}}e+00 : f16
  // CHECK-DAG: [[CM2:%.+]] = arith.constant -2.{{0*}}e+00 : f16

  // Fold an extract into a splat.
  // CHECK-DAG: [[C4:%.+]] = arith.constant 4.{{0*}}e+00 : f32
  %0 = arith.constant dense<4.0> : tensor<4xf32>
  %ext_1 = tensor.extract %0[%arg0] : tensor<4xf32>

  // Fold an extract into a sparse with a sparse index.
  %1 = arith.constant sparse<[[0, 0, 0], [1, 1, 1]],  [-5.0, -2.0]> : tensor<4x4x4xf16>
  %ext_2 = tensor.extract %1[%const_1, %const_1, %const_1] : tensor<4x4x4xf16>

  // Fold an extract into a sparse with a non sparse index.
  %2 = arith.constant sparse<[[1, 1, 1]],  [-2.0]> : tensor<2x2x2xf16>
  %ext_3 = tensor.extract %2[%const_0, %const_0, %const_0] : tensor<2x2x2xf16>

  // Fold an extract into a dense tensor.
  %3 = arith.constant dense<[[[1, -2, 1, 36]], [[0, 2, -1, 64]]]> : tensor<2x1x4xi32>
  %ext_4 = tensor.extract %3[%const_1, %const_0, %const_3] : tensor<2x1x4xi32>

  // Fold an extract into a complex constant.
  // CHECK-DAG: [[C5:%.+]] = complex.constant [1.200000e+00 : f32, 2.300000e+00 : f32] : complex<f32>
  %4 = arith.constant dense<(1.2, 2.3)> : tensor<complex<f32>>
  %ext_5 = tensor.extract %4[] : tensor<complex<f32>>

  // CHECK-NEXT: return [[C4]], [[CM2]], [[C0]], [[C64]], [[C5]]
  return %ext_1, %ext_2, %ext_3, %ext_4, %ext_5 : f32, f16, f16, i32, complex<f32>
}

// -----

// CHECK-LABEL: func @fold_insert
func @fold_insert(%arg0 : index) -> (tensor<4xf32>) {
  // Fold an insert into a splat.
  // CHECK-DAG: %[[C4:.+]] = arith.constant dense<4.{{0*}}e+00> : tensor<4xf32>
  %0 = arith.constant dense<4.0> : tensor<4xf32>
  %1 = arith.constant 4.0 : f32
  %ins_1 = tensor.insert %1 into %0[%arg0] : tensor<4xf32>
  // CHECK-NEXT: return %[[C4]]
  return %ins_1 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @extract_from_tensor.cast
// CHECK-SAME: %[[TENSOR:.*]]: tensor<*xf32>
func @extract_from_tensor.cast(%tensor: tensor<*xf32>) -> f32 {
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-NOT: tensor.cast
  %casted = tensor.cast %tensor : tensor<*xf32> to tensor<?xf32>
  // CHECK-NEXT: tensor.extract %[[TENSOR]][%[[C0]]]
  %result = tensor.extract %casted[%c0] : tensor<?xf32>
  return %result : f32
}

// -----

// CHECK-LABEL: func @extract_from_tensor.from_elements
func @extract_from_tensor.from_elements(%element : index) -> index {
  // CHECK-SAME: ([[ARG:%.*]]: index)
  %c0 = arith.constant 0 : index
  %tensor = tensor.from_elements %element : tensor<1xindex>
  %extracted_element = tensor.extract %tensor[%c0] : tensor<1xindex>
  // CHECK: [[ARG]] : index
  return %extracted_element : index
}

// -----

// CHECK-LABEL: func @extract_from_tensor.from_elements_0d
func @extract_from_tensor.from_elements_0d(%element : index) -> index {
  // CHECK-SAME: ([[ARG:%.*]]: index)
  %c0 = arith.constant 0 : index
  %tensor = tensor.from_elements %element : tensor<index>
  %extracted_element = tensor.extract %tensor[] : tensor<index>
  // CHECK: [[ARG]] : index
  return %extracted_element : index
}

// -----

// CHECK-LABEL: func @extract_from_tensor.from_elements_3d
func @extract_from_tensor.from_elements_3d()
    -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
  %f0 = arith.constant 0.0 : f32
  %f1 = arith.constant 1.0 : f32
  %f2 = arith.constant 2.0 : f32
  %f3 = arith.constant 3.0 : f32
  %f4 = arith.constant 4.0 : f32
  %f5 = arith.constant 5.0 : f32
  %f6 = arith.constant 6.0 : f32
  %f7 = arith.constant 7.0 : f32
  %f8 = arith.constant 8.0 : f32
  %f9 = arith.constant 9.0 : f32
  %f10 = arith.constant 10.0 : f32
  %f11 = arith.constant 11.0 : f32

  %tensor = tensor.from_elements %f0,%f1,%f2,%f3,%f4,%f5,%f6,%f7,%f8,%f9,%f10,%f11
         : tensor<3x2x2xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %r0 = tensor.extract %tensor[%c0, %c0, %c0] : tensor<3x2x2xf32>
  %r1 = tensor.extract %tensor[%c0, %c0, %c1] : tensor<3x2x2xf32>
  %r2 = tensor.extract %tensor[%c0, %c1, %c0] : tensor<3x2x2xf32>
  %r3 = tensor.extract %tensor[%c0, %c1, %c1] : tensor<3x2x2xf32>
  %r4 = tensor.extract %tensor[%c1, %c0, %c0] : tensor<3x2x2xf32>
  %r5 = tensor.extract %tensor[%c1, %c0, %c1] : tensor<3x2x2xf32>
  %r6 = tensor.extract %tensor[%c1, %c1, %c0] : tensor<3x2x2xf32>
  %r7 = tensor.extract %tensor[%c1, %c1, %c1] : tensor<3x2x2xf32>
  %r8 = tensor.extract %tensor[%c2, %c0, %c0] : tensor<3x2x2xf32>
  %r9 = tensor.extract %tensor[%c2, %c0, %c1] : tensor<3x2x2xf32>
  %r10 = tensor.extract %tensor[%c2, %c1, %c0] : tensor<3x2x2xf32>
  %r11 = tensor.extract %tensor[%c2, %c1, %c1] : tensor<3x2x2xf32>
  return %r0,%r1,%r2,%r3,%r4,%r5,%r6,%r7,%r8,%r9,%r10,%r11
         : f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32
}
// CHECK-DAG: %[[F0:.*]] = arith.constant 0.0
// CHECK-DAG: %[[F1:.*]] = arith.constant 1.0{{0+}}e+00
// CHECK-DAG: %[[F2:.*]] = arith.constant 2.0
// CHECK-DAG: %[[F3:.*]] = arith.constant 3.0
// CHECK-DAG: %[[F4:.*]] = arith.constant 4.0
// CHECK-DAG: %[[F5:.*]] = arith.constant 5.0
// CHECK-DAG: %[[F6:.*]] = arith.constant 6.0
// CHECK-DAG: %[[F7:.*]] = arith.constant 7.0
// CHECK-DAG: %[[F8:.*]] = arith.constant 8.0
// CHECK-DAG: %[[F9:.*]] = arith.constant 9.0
// CHECK-DAG: %[[F10:.*]] = arith.constant 1.0{{0+}}e+01
// CHECK-DAG: %[[F11:.*]] = arith.constant 1.1{{0+}}e+01

// CHECK: return %[[F0]], %[[F1]], %[[F2]], %[[F3]], %[[F4]], %[[F5]],
// CHECK-SAME:   %[[F6]], %[[F7]], %[[F8]], %[[F9]], %[[F10]], %[[F11]]

// -----

// Ensure the optimization doesn't segfault from bad constants
// CHECK-LABEL: func @extract_negative_from_tensor.from_elements
func @extract_negative_from_tensor.from_elements(%element : index) -> index {
  // CHECK-SAME: ([[ARG:%.*]]: index)
  %c-1 = arith.constant -1 : index
  %tensor = tensor.from_elements %element : tensor<1xindex>
  %extracted_element = tensor.extract %tensor[%c-1] : tensor<1xindex>
  // CHECK: tensor.from_elements
  // CHECK: %[[RESULT:.*]] = tensor.extract
  // CHECK: return %[[RESULT]]
  return %extracted_element : index
}

// -----

// Ensure the optimization doesn't segfault from bad constants
// CHECK-LABEL: func @extract_oob_from_tensor.from_elements
func @extract_oob_from_tensor.from_elements(%element : index) -> index {
  // CHECK-SAME: ([[ARG:%.*]]: index)
  %c1 = arith.constant 1 : index
  %tensor = tensor.from_elements %element : tensor<1xindex>
  %extracted_element = tensor.extract %tensor[%c1] : tensor<1xindex>
  // CHECK: tensor.from_elements
  // CHECK: %[[RESULT:.*]] = tensor.extract
  // CHECK: return %[[RESULT]]
  return %extracted_element : index
}

// -----

// Ensure the optimization doesn't segfault from bad constants
// CHECK-LABEL: func @extract_oob_from_tensor.from_elements
func @extract_oob_from_tensor.from_elements(%element : index) -> index {
  // CHECK-SAME: ([[ARG:%.*]]: index)
  %c2 = arith.constant 2 : index
  %tensor = tensor.from_elements %element : tensor<1xindex>
  %extracted_element = tensor.extract %tensor[%c2] : tensor<1xindex>
  // CHECK: tensor.from_elements
  // CHECK: %[[RESULT:.*]] = tensor.extract
  // CHECK: return %[[RESULT]]
  return %extracted_element : index
}

// -----

// CHECK-LABEL: func @extract_from_tensor.generate
// CHECK-SAME: %[[IDX:.*]]: index, %[[TENSOR:.*]]: tensor<*xf32>
func @extract_from_tensor.generate(%idx: index, %tensor: tensor<*xf32>) -> index {
  %size = tensor.rank %tensor : tensor<*xf32>
  // CHECK-NEXT: %[[RES:.*]] = tensor.dim %[[TENSOR]], %[[IDX]]
  %0 = tensor.generate %size {
    ^bb0(%arg0: index):
    %1 = tensor.dim %tensor, %arg0 : tensor<*xf32>
    tensor.yield %1 : index
  } : tensor<?xindex>
  %1 = tensor.extract %0[%idx] : tensor<?xindex>
  // CHECK-NEXT: return %[[RES]]
  return %1 : index
}

// -----

// CHECK-LABEL: func @extract_from_tensor.generate_2d
// CHECK-SAME: %[[IDX0:.*]]: index, %[[IDX1:.*]]: index, %[[TENSOR:.*]]: tensor<*xf32>
func @extract_from_tensor.generate_2d(%idx0: index, %idx1: index, %tensor: tensor<*xf32>) -> index {
  %size = tensor.rank %tensor : tensor<*xf32>
  // CHECK-NEXT: %[[DIM0:.*]] = tensor.dim %[[TENSOR]], %[[IDX0]]
  // CHECK-NEXT: %[[DIM1:.*]] = tensor.dim %[[TENSOR]], %[[IDX1]]
  // CHECK-NEXT: %[[RES:.*]] = arith.addi %[[DIM0]], %[[DIM1]]
  %0 = tensor.generate %size, %size {
    ^bb0(%arg0: index, %arg1: index):
    %1 = tensor.dim %tensor, %arg0 : tensor<*xf32>
    %2 = tensor.dim %tensor, %arg1 : tensor<*xf32>
    %3 = arith.addi %1, %2 : index
    tensor.yield %3 : index
  } : tensor<?x?xindex>
  %4 = tensor.extract %0[%idx0, %idx1] : tensor<?x?xindex>
  // CHECK-NEXT: return %[[RES]]
  return %4 : index
}

// -----

// CHECK-LABEL: func @extract_from_tensor.generate_sideeffects
// CHECK-SAME: %[[IDX:.*]]: index
func @extract_from_tensor.generate_sideeffects(%idx: index, %tensor: tensor<*xf32>, %mem: memref<?xindex>) -> index {
  %size = tensor.rank %tensor : tensor<*xf32>
  // CHECK: %[[DTENSOR:.*]] = tensor.generate
  %0 = tensor.generate %size {
    ^bb0(%arg0: index):
    %1 = tensor.dim %tensor, %arg0 : tensor<*xf32>
    memref.store %1, %mem[%arg0] : memref<?xindex>
    tensor.yield %1 : index
  } : tensor<?xindex>
  // CHECK: %[[RES:.*]] = tensor.extract %[[DTENSOR]][%[[IDX]]]
  %1 = tensor.extract %0[%idx] : tensor<?xindex>
  // CHECK-NEXT: return %[[RES]]
  return %1 : index
}

// -----

// CHECK-LABEL: @static_tensor.generate
// CHECK-SAME: %[[SIZE1:.*]]: index, %[[SIZE4:.*]]: index)
func @static_tensor.generate(%size1: index, %size4: index) -> tensor<3x?x?x7x?xindex> {
  %c5 = arith.constant 5 : index
  // CHECK: tensor.generate %[[SIZE1]], %[[SIZE4]]
  %0 = tensor.generate %size1, %c5, %size4 {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index):
    %1 = arith.constant 32 : index
    tensor.yield %1 : index
  // CHECK: : tensor<3x?x5x7x?xindex>
  } : tensor<3x?x?x7x?xindex>
  // CHECK: tensor.cast %{{.*}} : tensor<3x?x5x7x?xindex> to tensor<3x?x?x7x?xindex>
  return %0 : tensor<3x?x?x7x?xindex>
}

// -----

// CHECK-LABEL: @from_elements.constant
func @from_elements.constant() -> tensor<3xindex> {
  // CHECK: %[[CST:.*]] = arith.constant dense<[1, 2, 1]> : tensor<3xindex>
  // CHECK: return %[[CST]]
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %tensor = tensor.from_elements %c1, %c2, %c1 : tensor<3xindex>
  return %tensor : tensor<3xindex>
}

// -----

func @slice_canonicalize(%arg0 : tensor<?x?x?xf32>, %arg1 : index,
    %arg2 : index) -> tensor<?x?x?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = tensor.extract_slice %arg0[%c0, %arg1, %c1] [%c4, %c1, %arg2] [%c1, %c1, %c1] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
// CHECK-LABEL: func @slice_canonicalize
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?x?xf32>
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %{{[a-zA-Z0-9_]+}}, 1]
//  CHECK-SAME:      [4, 1, %{{[a-zA-Z0-9_]+}}] [1, 1, 1]
//  CHECK-SAME:      : tensor<?x?x?xf32> to tensor<4x1x?xf32>
//       CHECK:   %[[RESULT:.+]] = tensor.cast %[[SLICE]]
//       CHEKC:   return %[[RESULT]]

// -----

func @rank_reducing_slice_canonicalize(%arg0 : tensor<?x?x?xf32>, %arg1 : index,
    %arg2 : index) -> tensor<?x?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = tensor.extract_slice %arg0[%c0, %arg1, %c1] [%c4, 1, %arg2] [%c1, %c1, %c1] : tensor<?x?x?xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func @rank_reducing_slice_canonicalize
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?x?xf32>
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %{{[a-zA-Z0-9_]+}}, 1]
//  CHECK-SAME:      [4, 1, %{{[a-zA-Z0-9_]+}}] [1, 1, 1]
//  CHECK-SAME:      : tensor<?x?x?xf32> to tensor<4x?xf32>
//       CHECK:   %[[RESULT:.+]] = tensor.cast %[[SLICE]]
//       CHEKC:   return %[[RESULT]]

// -----

// CHECK-LABEL: func @trivial_slice
//  CHECK-SAME:   %[[ARG0:.[a-z0-9A-Z_]+]]: tensor<4x6x16x32xi8>
//   CHECK-NOT:   tensor.extract_slice
//       CHECK:   return %[[ARG0]] :  tensor<4x6x16x32xi8>
func @trivial_slice(%arg0 : tensor<4x6x16x32xi8>) -> tensor<4x6x16x32xi8> {
  %0 = tensor.extract_slice %arg0[0, 0, 0, 0] [4, 6, 16, 32] [1, 1, 1, 1] : tensor<4x6x16x32xi8> to tensor<4x6x16x32xi8>
  return %0 : tensor<4x6x16x32xi8>
}

// -----

// CHECK-LABEL: func @trivial_insert_slice
//  CHECK-SAME:   %[[ARG0:.[a-z0-9A-Z_]+]]: tensor<4x6x16x32xi8>
//   CHECK-NOT:   tensor.extract_slice
//       CHECK:   return %[[ARG0]] :  tensor<4x6x16x32xi8>
func @trivial_insert_slice(%arg0 : tensor<4x6x16x32xi8>, %arg1 : tensor<4x6x16x32xi8>) -> tensor<4x6x16x32xi8> {
  %0 = tensor.insert_slice %arg0 into %arg1[0, 0, 0, 0] [4, 6, 16, 32] [1, 1, 1, 1] : tensor<4x6x16x32xi8> into tensor<4x6x16x32xi8>
  return %0 : tensor<4x6x16x32xi8>
}

// -----

// CHECK-LABEL: func @rank_reducing_tensor_of_cast
//  CHECK-SAME:   %[[ARG0:.[a-z0-9A-Z_]+]]: tensor<4x6x16x32xi8>
//       CHECK:   %[[S:.+]] = tensor.extract_slice %arg0[0, 1, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : tensor<4x6x16x32xi8> to tensor<16x32xi8>
// Tensor cast is moved after slice and then gets canonicalized away.
//   CHECK-NOT:   tensor.cast
//       CHECK:   return %[[S]] : tensor<16x32xi8>
func @rank_reducing_tensor_of_cast(%arg : tensor<4x6x16x32xi8>) -> tensor<16x32xi8> {
  %0 = tensor.cast %arg : tensor<4x6x16x32xi8> to tensor<?x?x16x32xi8>
  %1 = tensor.extract_slice %0[0, 1, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : tensor<?x?x16x32xi8> to tensor<16x32xi8>
  return %1 : tensor<16x32xi8>
}

// -----

// CHECK-LABEL: func @rank_reducing_insert_slice_of_cast
//  CHECK-SAME:   %[[A:.[a-z0-9A-Z_]+]]: tensor<16x32xi8>
//  CHECK-SAME:   %[[B:.[a-z0-9A-Z_]+]]: tensor<4x6x16x32xi8>
//       CHECK:   %[[S:.+]] = tensor.insert_slice %[[A]] into %[[B]][0, 1, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : tensor<16x32xi8> into tensor<4x6x16x32xi8>
// Tensor cast is folded away.
//   CHECK-NOT:   tensor.cast
//       CHECK:   return %[[S]] : tensor<4x6x16x32xi8>
func @rank_reducing_insert_slice_of_cast(%a : tensor<16x32xi8>, %b : tensor<4x6x16x32xi8>) -> tensor<4x6x16x32xi8> {
  %c0 = arith.constant 0: index
  %cast = tensor.cast %a : tensor<16x32xi8> to tensor<?x32xi8>
  %sz = tensor.dim %cast, %c0: tensor<?x32xi8>
  %res = tensor.insert_slice %cast into %b[0, 1, 0, 0] [1, 1, %sz, 32] [1, 1, 1, 1] : tensor<?x32xi8> into tensor<4x6x16x32xi8>
  return %res : tensor<4x6x16x32xi8>
}

// -----

func @insert_slice_canonicalize(%arg0 : tensor<?x?x?xf32>, %arg1 : index,
    %arg2 : index, %arg3 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = tensor.insert_slice %arg0 into %arg3[%c0, %arg1, %c1] [%c4, %c1, %arg2] [%c1, %c1, %c1] : tensor<?x?x?xf32> into tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
// CHECK-LABEL: func @insert_slice_canonicalize
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//       CHECK:   %[[CAST:.+]] = tensor.cast %[[ARG0]] : tensor<?x?x?xf32> to tensor<4x1x?xf32>
//       CHECK:   %[[RESULT:.+]] = tensor.insert_slice %[[CAST]]
//  CHECK-SAME:      [0, %{{.+}}, 1] [4, 1, %{{.+}}] [1, 1, 1]
//  CHECK-SAME:      : tensor<4x1x?xf32> into tensor<?x?x?xf32>
//       CHECK:   return %[[RESULT]]

// -----

func @slice_to_insert_slice_canonicalize(%arg0 : tensor<?x?x?xf32>, %arg1 : index,
    %arg2 : index, %arg3 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = tensor.extract_slice %arg0[%c0, %arg1, %c1] [%c4, %c1, %arg2] [%c1, %c1, %c1] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
  %1 = tensor.insert_slice %0 into %arg3[%c0, %arg1, %c1] [%c4, %c1, %arg2] [%c1, %c1, %c1] : tensor<?x?x?xf32> into tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}
// CHECK-LABEL: func @slice_to_insert_slice_canonicalize
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]]
//  CHECK-SAME:      [0, %{{.+}}, 1] [4, 1, %{{.+}} [1, 1, 1]
//  CHECK-SAME:      : tensor<?x?x?xf32> to tensor<4x1x?xf32>
//       CHECK:   %[[RESULT:.+]] = tensor.insert_slice %[[SLICE]]
//  CHECK-SAME:      [0, %{{.+}}, 1] [4, 1, %{{.+}}] [1, 1, 1]
//  CHECK-SAME:      : tensor<4x1x?xf32> into tensor<?x?x?xf32>
//       CHEKC:   return %[[RESULT]]

// -----

func @rank_reducing_insert_slice_canonicalize(%arg0 : tensor<?x?xf32>, %arg1 : index,
    %arg2 : index, %arg3 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = tensor.insert_slice %arg0 into %arg3[%c0, %arg1, %c1] [%c4, 1, %arg2] [%c1, %c1, %c1] : tensor<?x?xf32> into tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
// CHECK-LABEL: func @rank_reducing_insert_slice_canonicalize
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xf32>
//       CHECK:   %[[CAST:.*]] = tensor.cast %[[ARG0]] : tensor<?x?xf32> to tensor<4x?xf32>
//       CHECK:   %[[RESULT:.+]] = tensor.insert_slice %[[CAST]]
//  CHECK-SAME:      [0, %{{.+}}, 1] [4, 1, %{{.+}}] [1, 1, 1]
//  CHECK-SAME:      : tensor<4x?xf32> into tensor<?x?x?xf32>
//       CHEKC:   return %[[RESULT]]

// -----

func @rank_reducing_slice_to_insert_slice_canonicalize(%arg0 : tensor<?x?x?xf32>, %arg1 : index,
    %arg2 : index, %arg3 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = tensor.extract_slice %arg0[%c0, %arg1, %c1] [%c4, 1, %arg2] [%c1, %c1, %c1] : tensor<?x?x?xf32> to tensor<?x?xf32>
  %1 = tensor.insert_slice %0 into %arg3[%c0, %arg1, %c1] [%c4, 1, %arg2] [%c1, %c1, %c1] : tensor<?x?xf32> into tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}
// CHECK-LABEL: func @rank_reducing_slice_to_insert_slice_canonicalize
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]]
//  CHECK-SAME:     [0, %{{.+}}, 1] [4, 1, %{{.+}}] [1, 1, 1]
//  CHECK-SAME:     : tensor<?x?x?xf32> to tensor<4x?xf32>
//       CHECK:   %[[RESULT:.+]] = tensor.insert_slice %[[SLICE]] into %[[ARG3]]
//  CHECK-SAME:      [0, %{{.+}}, 1] [4, 1, %{{.+}}] [1, 1, 1]
//  CHECK-SAME:      : tensor<4x?xf32> into tensor<?x?x?xf32>
//       CHEKC:   return %[[RESULT]]

// -----

func @insert_slice_propagate_dest_cast(%arg0 : tensor<2x?xi32>, %arg1 : tensor<i32>,
    %arg2 : index, %arg3 : index) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %0 = tensor.dim %arg0, %c1 : tensor<2x?xi32>
  %1 = tensor.extract %arg1[] : tensor<i32>
  %2 = tensor.generate %arg2, %c8 {
  ^bb0(%arg4: index, %arg5: index):
    tensor.yield %1 : i32
  } : tensor<?x?xi32>
  %3 = tensor.insert_slice %arg0 into %2[0, %arg3] [2, %0] [1, 1] : tensor<2x?xi32> into tensor<?x?xi32>
  return %3 : tensor<?x?xi32>
}
// CHECK-LABEL: func @insert_slice_propagate_dest_cast
//       CHECK:   %[[UPDATED:.+]] = tensor.insert_slice %{{.+}} into %{{.+}}[0, %{{.+}}] [2, %{{.+}}] [1, 1]
//  CHECK-SAME:     tensor<2x?xi32> into tensor<?x8xi32>
//       CHECK:   %[[CAST:.+]] = tensor.cast %[[UPDATED]]
//       CHECK:   return %[[CAST]]

// -----

func @insert_slice_output_dest_canonicalize(%arg0 : tensor<2x3xi32>, %arg1 : tensor<i32>) -> tensor<3x9xi32> {
  %c9 = arith.constant 9 : index
  %c3 = arith.constant 3 : index
  %2 = tensor.extract %arg1[] : tensor<i32>
  %4 = tensor.generate %c3, %c9 {
  ^bb0(%arg2: index, %arg3: index):
    tensor.yield %2 : i32
  } : tensor<?x?xi32>
  %5 = tensor.insert_slice %arg0 into %4[0, 1] [2, 3] [1, 1] : tensor<2x3xi32> into tensor<?x?xi32>
  %6 = tensor.cast %5 : tensor<?x?xi32> to tensor<3x9xi32>
  return %6 : tensor<3x9xi32>
}
// CHECK-LABEL: func @insert_slice_output_dest_canonicalize
//  CHECK-SAME:   %[[ARG0:[a-zA-z0-9_]+]]: tensor<2x3xi32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<i32>
//       CHECK:   %[[PAD:.+]] = tensor.extract %[[ARG1]]
//       CHECK:   %[[GENERATE:.+]] = tensor.generate
//       CHECK:   %[[RESULT:.+]] = tensor.insert_slice %[[ARG0]] into %[[GENERATE]]
//       CHECK:   return %[[RESULT]]

// -----

// Test case: Folding of tensor.dim(tensor.generate %idx) -> %idx
// CHECK-LABEL: func @dim_of_tensor.generate(
//  CHECK-SAME:     %[[IDX0:[0-9a-z]+]]: index, %[[IDX1:[0-9a-z]+]]: index
//   CHECK-NOT:   tensor.dim
//       CHECK:   return %[[IDX1]] : index
func @dim_of_tensor.generate(%arg0: index, %arg1: index) -> index {
  %c3 = arith.constant 3 : index
  %0 = tensor.generate %arg0, %arg1 {
  ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index):
    tensor.yield %c3 : index
  } : tensor<2x?x4x?x5xindex>
  %1 = tensor.dim %0, %c3 : tensor<2x?x4x?x5xindex>
  return %1 : index
}

// -----

// Test case: Folding tensor.dim(tensor.cast %0, %idx) -> tensor.dim %0, %idx
// CHECK-LABEL: func @fold_dim_of_tensor.cast
//  CHECK-SAME:   %[[ARG0:.[a-z0-9A-Z_]+]]: tensor<4x?xf32>
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//       CHECK:   %[[T0:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-NEXT:   return %[[C4]], %[[T0]]
func @fold_dim_of_tensor.cast(%arg0 : tensor<4x?xf32>) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.cast %arg0 : tensor<4x?xf32> to tensor<?x?xf32>
  %1 = tensor.dim %0, %c0 : tensor<?x?xf32>
  %2 = tensor.dim %0, %c1 : tensor<?x?xf32>
  return %1, %2: index, index
}

// -----

// CHECK-LABEL: func @insert_tensor_cast_on_insert_slice_src(
// CHECK-SAME:      %[[arg0:.*]]: tensor<?x5x?xf32>, %[[arg1:.*]]: tensor<?x?x?xf32>
//      CHECK:    %[[cast:.*]] = tensor.cast %[[arg0]] : tensor<?x5x?xf32> to tensor<64x5x64xf32>
//      CHECK:    %[[r:.*]] =  tensor.insert_slice %[[cast]] into %[[arg1]][0, 1, 2] [64, 5, 64] [1, 1, 1] : tensor<64x5x64xf32> into tensor<?x?x?xf32>
//      CHECK:    return %[[r]]
func @insert_tensor_cast_on_insert_slice_src(
    %arg0 : tensor<?x5x?xf32>,  %arg1 : tensor<?x?x?xf32>, %sz0: index, %sz2: index) -> tensor<?x?x?xf32> {
  %c64 = arith.constant 64: index
  %r = tensor.insert_slice %arg0 into %arg1[0, 1, 2] [%c64, 5, %c64] [1, 1, 1]
    : tensor<?x5x?xf32> into tensor<?x?x?xf32>
  return %r : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func @fold_extract_insert
//  CHECK-SAME: %{{.+}}: tensor<?x?x?xf32>, %[[SLICE:.+]]: tensor<4x?x8xf32>
func @fold_extract_insert(%input : tensor<?x?x?xf32>, %slice: tensor<4x?x8xf32>, %i: index, %size: index) -> (tensor<4x?x8xf32>) {
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %0 = tensor.insert_slice %slice into %input[%c0, %i, 0] [4, %size, 8] [1, 1, %c1] : tensor<4x?x8xf32> into tensor<?x?x?xf32>
  %1 = tensor.extract_slice %0[%c0, %i, 0] [4, %size, 8] [1, 1, %c1] : tensor<?x?x?xf32> to tensor<4x?x8xf32>
  // CHECK: return %[[SLICE]]
  return %1 : tensor<4x?x8xf32>
}

// -----

// CHECK-LABEL: func @fold_overlapping_insert
//  CHECK-SAME: %[[INPUT:.+]]: tensor<?x?x?xf32>, %{{.+}}: tensor<4x?x8xf32>, %[[SLICE2:.+]]: tensor<4x?x8xf32>
func @fold_overlapping_insert(%input : tensor<?x?x?xf32>, %slice1: tensor<4x?x8xf32>, %slice2: tensor<4x?x8xf32>, %i: index, %size: index) -> (tensor<?x?x?xf32>) {
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %0 = tensor.insert_slice %slice1 into %input[%c0, %i, 0] [4, %size, 8] [1, 1, %c1] : tensor<4x?x8xf32> into tensor<?x?x?xf32>
  // CHECK: %[[INSERT:.+]] = tensor.insert_slice %[[SLICE2]] into %[[INPUT]]
  %1 = tensor.insert_slice %slice2 into %0[%c0, %i, 0] [4, %size, 8] [1, 1, %c1] : tensor<4x?x8xf32> into tensor<?x?x?xf32>
  // CHECK: return %[[INSERT]]
  return %1 : tensor<?x?x?xf32>
}

// -----

func @expanding_tensor_reshapes(%arg0 : tensor<?x?xf32>)
    -> tensor<?x6x4x?x5xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1], [2]]
      : tensor<?x?xf32> into tensor<?x4x?xf32>
  %1 = tensor.expand_shape %0 [[0, 1], [2], [3, 4]]
      : tensor<?x4x?xf32> into tensor<?x6x4x?x5xf32>
  return %1 : tensor<?x6x4x?x5xf32>
}
// CHECK-LABEL: expanding_tensor_reshapes
//       CHECK:   tensor.expand_shape %{{.*}} {{\[}}[0, 1, 2], [3, 4]]
//   CHECK-NOT:   tensor.expand_shape

// -----

func @expanding_tensor_reshapes_to_zero_dim(%arg0 : tensor<f32>)
    -> tensor<1x1x1xf32> {
  %0 = tensor.expand_shape %arg0 [] : tensor<f32> into tensor<1xf32>
  %1 = tensor.expand_shape %0 [[0, 1, 2]]
      : tensor<1xf32> into tensor<1x1x1xf32>
  return %1 : tensor<1x1x1xf32>
}
// CHECK-LABEL: expanding_tensor_reshapes_to_zero
//       CHECK:   tensor.expand_shape %{{.*}} []
//  CHECK-SAME:     tensor<f32> into tensor<1x1x1xf32>

// -----

func @fold_tensor_reshape(%arg0 : tensor<12x4xf32>) -> tensor<12x4xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1], [2]]
      : tensor<12x4xf32> into tensor<3x4x4xf32>
  %1 = tensor.collapse_shape %0 [[0, 1], [2]]
      : tensor<3x4x4xf32> into tensor<12x4xf32>
  return %1 : tensor<12x4xf32>
}
// CHECK-LABEL: @fold_tensor_reshape
//   CHECK-NOT:   linalg.{{.*}}shape

// -----

func @fold_tensor_reshape_dynamic(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1], [2]]
      : tensor<?x?xf32> into tensor<?x4x?xf32>
  %1 = tensor.collapse_shape %0 [[0, 1], [2]]
      : tensor<?x4x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: @fold_tensor_reshape_dynamic
//   CHECK-NOT:   linalg.{{.*}}_shape

// -----
func @reshape_collapse(%arg0 : tensor<2x3x4x5x6x7x8xf32>)
    -> tensor<24x5x42x8xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1, 2, 3, 4, 5, 6]]
      : tensor<2x3x4x5x6x7x8xf32> into tensor<40320xf32>
  %1 = tensor.expand_shape %0 [[0, 1, 2, 3]]
      : tensor<40320xf32> into tensor<24x5x42x8xf32>
  return %1 : tensor<24x5x42x8xf32>
}
//      CHECK: func @reshape_collapse
// CHECK-SAME:   %[[ARG0:.+]]: tensor<2x3x4x5x6x7x8xf32>
//      CHECK:   %[[RESULT:.+]] = tensor.collapse_shape %[[ARG0]]
// CHECK-SAME:     [0, 1, 2], [3], [4, 5], [6]
//      CHECK:   return %[[RESULT]]

// -----

func @reshape_expand(%arg0 : tensor<24x5x42x8xf32>)
    -> tensor<2x3x4x5x6x7x8xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1, 2, 3]]
      : tensor<24x5x42x8xf32> into tensor<40320xf32>
  %1 = tensor.expand_shape %0 [[0, 1, 2, 3, 4, 5, 6]]
      : tensor<40320xf32> into tensor<2x3x4x5x6x7x8xf32>
  return %1 : tensor<2x3x4x5x6x7x8xf32>
}
//      CHECK: func @reshape_expand
// CHECK-SAME:   %[[ARG0:.+]]: tensor<24x5x42x8xf32>
//      CHECK:   %[[RESULT:.+]] = tensor.expand_shape %[[ARG0]]
// CHECK-SAME:     [0, 1, 2], [3], [4, 5], [6]
//      CHECK:   return %[[RESULT]]

// -----

func @expand_reshape_1D(%arg0 : tensor<2048xf32>) -> tensor<4x512xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1, 2, 3]]
    : tensor<2048xf32> into tensor<1x4x1x512xf32>
  %1 = tensor.collapse_shape %0 [[0, 1, 2], [3]]
    : tensor<1x4x1x512xf32> into tensor<4x512xf32>
  return %1 : tensor<4x512xf32>
}
//       CHECK: func @expand_reshape_1D
//       CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1]]
//  CHECK-SAME:   tensor<2048xf32> into tensor<4x512xf32>

// -----

// CHECK-LABEL: zero_rank_reshape_multi
func @zero_rank_reshape_multi(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: return %arg0
  %0 = tensor.expand_shape %arg0 [] : tensor<f32> into tensor<1xf32>
  %1 = tensor.expand_shape %0 [[0, 1]] : tensor<1xf32> into tensor<1x1xf32>
  %2 = tensor.collapse_shape %1 [] : tensor<1x1xf32> into tensor<f32>
  return %2 : tensor<f32>
}

// -----

func @collapsing_tensor_reshapes(%arg0 : tensor<?x?x?x?x?xf32>)
    -> tensor<?x?xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1], [2], [3, 4]]
      : tensor<?x?x?x?x?xf32> into tensor<?x?x?xf32>
  %1 = tensor.collapse_shape %0 [[0, 1], [2]]
      : tensor<?x?x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: collapsing_tensor_reshapes
//       CHECK:   tensor.collapse_shape %{{.*}} {{\[}}[0, 1, 2], [3, 4]]
//   CHECK-NOT:   tensor.collapse_shape

// -----

func @collapsing_tensor_reshapes_to_zero_dim(%arg0 : tensor<1x1x1xf32>)
    -> tensor<f32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1, 2]]
      : tensor<1x1x1xf32> into tensor<1xf32>
  %1 = tensor.collapse_shape %0 [] : tensor<1xf32> into tensor<f32>
  return %1 : tensor<f32>
}
// CHECK-LABEL: collapsing_tensor_reshapes_to_zero
//       CHECK:   tensor.collapse_shape %{{.*}} []
//  CHECK-SAME:     tensor<1x1x1xf32> into tensor<f32>

// -----

func @fold_reshape_1D(%arg0 : tensor<4x512xf32>) -> tensor<2048xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1, 2], [3]]
    : tensor<4x512xf32> into tensor<1x4x1x512xf32>
  %1 = tensor.collapse_shape %0 [[0, 1, 2, 3]]
    : tensor<1x4x1x512xf32> into tensor<2048xf32>
  return %1 : tensor<2048xf32>
}
//       CHECK: func @fold_reshape_1D
//       CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0, 1]]
//  CHECK-SAME:   tensor<4x512xf32> into tensor<2048xf32>

// -----

func @fold_reshape_unit_dims(%arg0 : tensor<2048x1x1xf32>)
    -> tensor<4x512x1x1xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1, 2, 3], [4], [5]]
    : tensor<2048x1x1xf32> into tensor<1x4x1x512x1x1xf32>
  %1 = tensor.collapse_shape %0 [[0, 1, 2], [3], [4], [5]]
    : tensor<1x4x1x512x1x1xf32> into tensor<4x512x1x1xf32>
  return %1 : tensor<4x512x1x1xf32>
}
//       CHECK: func @fold_reshape_unit_dims
//       CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1], [2], [3]]
//  CHECK-SAME:   tensor<2048x1x1xf32> into tensor<4x512x1x1xf32>

// -----

func @expand_reshape_unit_dims(%arg0 : tensor<2048x1x2048xf32>)
    -> tensor<4x512x1x512x4xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4], [5], [6, 7, 8]]
    : tensor<2048x1x2048xf32> into tensor<1x4x1x512x1x1x512x1x4xf32>
  %1 = tensor.collapse_shape %0 [[0, 1, 2], [3, 4], [5], [6, 7], [8]]
    : tensor<1x4x1x512x1x1x512x1x4xf32> into tensor<4x512x1x512x4xf32>
  return %1 : tensor<4x512x1x512x4xf32>
}
//       CHECK: func @expand_reshape_unit_dims
//       CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1], [2], [3, 4]]
//  CHECK-SAME:   tensor<2048x1x2048xf32> into tensor<4x512x1x512x4xf32>

// -----

func @fold_reshape_trailing_unit_dims(%arg0: tensor<2xf32>) -> tensor<2x1xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1, 2]]
      : tensor<2xf32> into tensor<2x1x1xf32>
  %1 = tensor.collapse_shape %0 [[0], [1, 2]]
      : tensor<2x1x1xf32> into tensor<2x1xf32>
  return %1 : tensor<2x1xf32>
}
//       CHECK: func @fold_reshape_trailing_unit_dims
//       CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1]]
//  CHECK-SAME:   tensor<2xf32> into tensor<2x1xf32>

// -----

func @collapse_reshape_unit_dims_dynamic(%arg0 : tensor<?x1x?x1x1x?x?x1x1xf32>)
    -> tensor<?x?x?x?xf32> {
  %0 = tensor.collapse_shape %arg0 [[0], [1, 2], [3], [4], [5], [6, 7, 8]]
    : tensor<?x1x?x1x1x?x?x1x1xf32> into tensor<?x?x1x1x?x?xf32>
  %1 = tensor.collapse_shape %0 [[0], [1], [2, 3, 4], [5]]
    : tensor<?x?x1x1x?x?xf32> into tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}
//       CHECK: func @collapse_reshape_unit_dims_dynamic
//       CHECK: tensor.collapse_shape
//  CHECK-SAME:   [0], [1, 2], [3, 4, 5], [6, 7, 8]
//  CHECK-SAME:   tensor<?x1x?x1x1x?x?x1x1xf32> into tensor<?x?x?x?xf32>

// -----

func @fold_reshape_trailing_unit_dims(%arg0: tensor<2xf32>) -> tensor<2x1xf32>
{
  %0 = tensor.expand_shape %arg0 [[0, 1, 2]]
      : tensor<2xf32> into tensor<2x1x1xf32>
  %1 = tensor.collapse_shape %0 [[0], [1, 2]]
      : tensor<2x1x1xf32> into tensor<2x1xf32>
  return %1 : tensor<2x1xf32>
}
//       CHECK: func @fold_reshape_trailing_unit_dims
//       CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1]]
//  CHECK-SAME:   tensor<2xf32> into tensor<2x1xf32>

// -----

func @fold_reshape_trailing_unit_dims_dynamic(%arg0: tensor<1x1x?x1x1x1xf32>)
    -> tensor<?xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1, 2], [3], [4], [5]]
      : tensor<1x1x?x1x1x1xf32> into tensor<?x1x1x1xf32>
  %1 = tensor.collapse_shape %0 [[0, 1, 2, 3]]
      : tensor<?x1x1x1xf32> into tensor<?xf32>
  return %1 : tensor<?xf32>
}
//       CHECK: func @fold_reshape_trailing_unit_dims_dynamic
//       CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0, 1, 2, 3, 4, 5]]
//  CHECK-SAME:   tensor<1x1x?x1x1x1xf32> into tensor<?xf32>

// -----

func @fold_reshape_trailing_unit_dims(%arg0: tensor<12x42x1x1xf32>)
    -> tensor<12x42xf32> {
  %0 = tensor.expand_shape %arg0 [[0], [1], [2], [3, 4]]
      : tensor<12x42x1x1xf32> into tensor<12x42x1x1x1xf32>
  %1 = tensor.collapse_shape %0 [[0], [1, 2, 3, 4]]
      : tensor<12x42x1x1x1xf32> into tensor<12x42xf32>
  return %1 : tensor<12x42xf32>
}
//       CHECK: func @fold_reshape_trailing_unit_dims
//       CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0], [1, 2, 3]]
//  CHECK-SAME:   tensor<12x42x1x1xf32> into tensor<12x42xf32>

// -----

func @fold_reshapes_unit_dims_in_middle(%arg0 : tensor<?x?x?xf32>) -> tensor<?x?xf32> {
  %0 = tensor.expand_shape %arg0 [[0], [1], [2, 3]]
      : tensor<?x?x?xf32> into tensor<?x?x1x?xf32>
  %1 = tensor.collapse_shape %0 [[0], [1, 2, 3]]
      : tensor<?x?x1x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func @fold_reshapes_unit_dims_in_middle
//  CHECK-SAME: (%[[ARG:.*]]: tensor<?x?x?xf32>
//       CHECK: tensor.collapse_shape %[[ARG]] {{\[}}[0], [1, 2]]
//  CHECK-SAME:   tensor<?x?x?xf32> into tensor<?x?xf32>

// -----

func @no_fold_reshape_incompatible(%arg0 : tensor<4x6x8xf32>)
    -> tensor<2x6x16xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1], [2, 3], [4]]
      : tensor<4x6x8xf32> into tensor<2x2x3x2x8xf32>
  %1 = tensor.collapse_shape %0 [[0], [1, 2], [3, 4]]
      : tensor<2x2x3x2x8xf32> into tensor<2x6x16xf32>
  return %1 : tensor<2x6x16xf32>
}
// CHECK-LABEL: func @no_fold_reshape_incompatible
//       CHECK:   tensor.expand_shape
//       CHECK:   tensor.collapse_shape

// -----

func @no_fold_reshape_empty_expr(%arg0: tensor<3x2x2xf32>) -> tensor<12x1xf32> {
  %0 = tensor.expand_shape %arg0 [[0], [1], [2, 3]]
      : tensor<3x2x2xf32> into tensor<3x2x2x1xf32>
  %1 = tensor.collapse_shape %0 [[0, 1, 2], [3]]
      : tensor<3x2x2x1xf32> into tensor<12x1xf32>
  return %1 : tensor<12x1xf32>
}
//      CHECK: func @no_fold_reshape_empty_expr
// CHECK-SAME:    %[[ARG0:.+]]: tensor<3x2x2xf32>
//      CHECK:    %[[RARG0:.+]] = tensor.expand_shape %[[ARG0]]
// CHECK-SAME:      [0], [1], [2, 3]
//      CHECK:    %[[RES:.+]] = tensor.collapse_shape %[[RARG0]]
// CHECK-SAME:      [0, 1, 2], [3]
//      CHECK:    return %[[RES:.+]] : tensor<12x1xf32>

// -----

func @reshape_splat_constant_int32() -> tensor<2x4x2xi32> {
  %c0 = arith.constant dense<42> : tensor<2x8xi32>
  %0 = tensor.expand_shape %c0 [[0], [1, 2]]
      : tensor<2x8xi32> into tensor<2x4x2xi32>
  return %0 : tensor<2x4x2xi32>
}
// CHECK-LABEL: @reshape_splat_constant_int32
//       CHECK:   %[[CST:.*]] = arith.constant dense<{{.*}}> : tensor<2x4x2xi32>
//   CHECK-NOT:   tensor.expand_shape
//       CHECK:   return %[[CST]]

// -----

func @reshape_splat_constant_int16() -> tensor<2x4x2xi16> {
  %c0 = arith.constant dense<42> : tensor<2x8xi16>
  %0 = tensor.expand_shape %c0 [[0], [1, 2]]
      : tensor<2x8xi16> into tensor<2x4x2xi16>
  return %0 : tensor<2x4x2xi16>
}
// CHECK-LABEL: @reshape_splat_constant_int16
//       CHECK:   %[[CST:.*]] = arith.constant dense<{{.*}}> : tensor<2x4x2xi16>
//   CHECK-NOT:   tensor.expand_shape
//       CHECK:   return %[[CST]]

// -----

func @reshape_splat_constant_float32() -> tensor<2x4x2xf32> {
  %c0 = arith.constant dense<42.0> : tensor<2x8xf32>
  %0 = tensor.expand_shape %c0 [[0], [1, 2]]
      : tensor<2x8xf32> into tensor<2x4x2xf32>
  return %0 : tensor<2x4x2xf32>
}
// CHECK-LABEL: @reshape_splat_constant_float32
//       CHECK:   %[[CST:.*]] = arith.constant dense<{{.*}}> : tensor<2x4x2xf32>
//   CHECK-NOT:   tensor.expand_shape
//       CHECK:   return %[[CST]]

// -----

func @reshape_splat_constant_float64() -> tensor<2x4x2xf64> {
  %c0 = arith.constant dense<42.0> : tensor<2x8xf64>
  %0 = tensor.expand_shape %c0 [[0], [1, 2]]
      : tensor<2x8xf64> into tensor<2x4x2xf64>
  return %0 : tensor<2x4x2xf64>
}
// CHECK-LABEL: @reshape_splat_constant_float64
//       CHECK:   %[[CST:.*]] = arith.constant dense<{{.*}}> : tensor<2x4x2xf64>
//   CHECK-NOT:   tensor.expand_shape
//       CHECK:   return %[[CST]]

// -----

// CHECK-LABEL: func @fold_rank
func @fold_rank() -> (index) {
  %const_0 = arith.constant dense<[[[1, -2, 1, 36]], [[0, 2, -1, 64]]]>
    : tensor<2x1x4xi32>

  // Fold a ank into a constant
  // CHECK-NEXT: [[C3:%.+]] = arith.constant 3 : index
  %rank_0 = tensor.rank %const_0 : tensor<2x1x4xi32>

  // CHECK-NEXT: return [[C3]]
  return %rank_0 : index
}

// -----

// CHECK-LABEL: func @pad_tensor_same_static_shape(
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<5x6xf32>
//   CHECK-NOT:   tensor.pad
//       CHECK:   return %[[ARG0]]
func @pad_tensor_same_static_shape(%arg0: tensor<5x6xf32>, %a: index)
    -> tensor<5x6xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.pad %arg0 low[%a, 0] high[0, %a] {
        ^bb0(%arg1: index, %arg2: index):
          tensor.yield %cst : f32
  } : tensor<5x6xf32> to tensor<5x6xf32>
  return %0 : tensor<5x6xf32>
}

// -----

// CHECK-LABEL: func @pad_tensor_nofold_same_static_shape(
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<5x6xf32>
//       CHECK:   %[[PAD:.*]] = tensor.pad
//       CHECK:   return %[[PAD]]
func @pad_tensor_nofold_same_static_shape(%arg0: tensor<5x6xf32>, %a: index)
    -> tensor<5x6xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.pad %arg0 nofold low[%a, 0] high[0, %a] {
        ^bb0(%arg1: index, %arg2: index):
          tensor.yield %cst : f32
  } : tensor<5x6xf32> to tensor<5x6xf32>
  return %0 : tensor<5x6xf32>
}

// -----

// CHECK-LABEL:   func @pad_tensor_after_cast_different_shape(
// CHECK-SAME:      %[[INPUT:.*]]: tensor<?x64x?x?xf32>) -> tensor<?x?x?x?xf32> {
// CHECK:           %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[PADDED:.*]] = tensor.pad %[[INPUT]]
// CHECK-SAME:        low[0, 0, 1, 1] high[0, 0, 1, 1]  {
// CHECK:           ^bb0(%[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index):
// CHECK:             tensor.yield %[[CST]] : f32
// CHECK:           } : tensor<?x64x?x?xf32> to tensor<?x64x?x?xf32>
// CHECK:           %[[DYNAMIC:.*]] = tensor.cast %[[PADDED:.*]] :
// CHECK-SAME:         tensor<?x64x?x?xf32> to tensor<?x?x?x?xf32>
// CHECK:           return %[[DYNAMIC]] : tensor<?x?x?x?xf32>
// CHECK:         }
func @pad_tensor_after_cast_different_shape(%arg0: tensor<?x64x?x?xf32>)
    -> tensor<?x?x?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %dynamic = tensor.cast %arg0 : tensor<?x64x?x?xf32> to tensor<?x?x?x?xf32>
  %padded = tensor.pad %dynamic low[0, 0, 1, 1] high[0, 0, 1, 1]  {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst: f32
  } : tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32>
  return %padded: tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL:   func @pad_tensor_after_cast_same_shape(
// CHECK-SAME:      %[[INPUT:.*]]: tensor<?x64x?x?xf32>,
// CHECK-SAME:      %[[PADDING:.*]]: index) -> tensor<?x?x?x?xf32> {
// CHECK:           %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[PADDED:.*]] = tensor.pad %[[INPUT]]
// CHECK-SAME:        low[0, %[[PADDING]], 1, 1] high[0, %[[PADDING]], 1, 1]  {
// CHECK:           ^bb0(%[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index):
// CHECK:             tensor.yield %[[CST]] : f32
// CHECK:           } : tensor<?x64x?x?xf32> to tensor<?x?x?x?xf32>
// CHECK:           return %[[PADDED:.*]] : tensor<?x?x?x?xf32>
// CHECK:         }
func @pad_tensor_after_cast_same_shape(%arg0: tensor<?x64x?x?xf32>, %padding : index)
    -> tensor<?x?x?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %dynamic = tensor.cast %arg0 : tensor<?x64x?x?xf32> to tensor<?x?x?x?xf32>
  %padded = tensor.pad %dynamic low[0, %padding, 1, 1] high[0, %padding, 1, 1]  {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst: f32
  } : tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32>
  return %padded: tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func @pad_tensor_of_cast(
// CHECK-NOT:     tensor.cast
// CHECK:         tensor.pad
// CHECK:         tensor<8x?xf32> to tensor<8x32xf32>
func @pad_tensor_of_cast(%t: tensor<8x?xf32>, %s: index) -> tensor<8x32xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.cast %t : tensor<8x?xf32> to tensor<?x?xf32>
  %1 = tensor.pad %0 low[%c0, %c0] high[%c0, %s]  {
  ^bb0(%arg9: index, %arg10: index):
    tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<8x32xf32>
  return %1 : tensor<8x32xf32>
}

// -----

// CHECK-LABEL: @cast_of_pad_more_static
func @cast_of_pad_more_static(%arg0: tensor<?x?xf32>, %padding: index) -> tensor<32x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: %[[PAD:.*]] = tensor.pad
  // CHECK: tensor<?x?xf32> to tensor<32x32xf32>
  %padded = tensor.pad %arg0 low[%padding, %padding] high[0, 0] {
  ^bb0(%arg1: index, %arg2: index):
    tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>
  // CHECK-NOT: tensor.cast
  %casted = tensor.cast %padded : tensor<?x?xf32> to tensor<32x32xf32>
  // CHECK: return %[[PAD]]
  return %casted : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: @cast_of_pad_less_static
func @cast_of_pad_less_static(%arg0: tensor<32x?x?xf32>, %padding: index) -> tensor<?x32x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: tensor.pad
  %padded = tensor.pad %arg0 low[%padding, %padding, %padding] high[0, 0, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index):
    tensor.yield %cst : f32
  } : tensor<32x?x?xf32> to tensor<32x?x?xf32>
  // CHECK: %[[CAST:.*]] = tensor.cast
  %casted = tensor.cast %padded : tensor<32x?x?xf32> to tensor<?x32x32xf32>
  // CHECK: return %[[CAST]]
  return %casted : tensor<?x32x32xf32>
}

// -----

func @tensor_pad_cast_fold(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = tensor.cast %arg0 : tensor<4x4xf32> to tensor<?x?xf32>
  %1 = tensor.pad %0 low[%c0, %c0] high[%c0, %c0]  {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
// CHECK-LABEL: @tensor_pad_cast
// CHECK-SAME: %[[ARG0:.+]]: tensor<4x4xf32>
// CHECK: return %[[ARG0]]

// -----

// CHECK-LABEL: func @fold_pad_tensor_source_cast(
//  CHECK-SAME:                  %[[ARG0:.*]]: tensor<4x?xf32>
//   CHECK-NOT:   tensor.cast
//       CHECK:   %[[RESULT:.*]] = tensor.pad %[[ARG0]]
func @fold_pad_tensor_source_cast(%arg0: tensor<4x?xf32>) -> tensor<4x4xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.cast %arg0 : tensor<4x?xf32> to tensor<?x?xf32>
  %1 = tensor.pad %0 low[0, 0] high[0, 1]  {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @pad_static_zero_cast(
//  CHECK-SAME:                  %[[ARG0:.*]]: tensor<?x?x?xf32>
//   CHECK-NOT:   tensor.pad
//       CHECK:   %[[RESULT:.*]] = tensor.cast %[[ARG0]] : tensor<?x?x?xf32> to tensor<2x3x4xf32>
//       CHECK:   return %[[RESULT]]
func @pad_static_zero_cast(%arg0: tensor<?x?x?xf32>, %pad_value: f32) -> tensor<2x3x4xf32> {
  %c0 = arith.constant 0 : index
  %0 = tensor.pad %arg0 low[0, %c0, 0] high[0, 0, %c0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %pad_value : f32
    } : tensor<?x?x?xf32> to tensor<2x3x4xf32>

  return %0 : tensor<2x3x4xf32>
}

// -----

// CHECK-LABEL: func @pad_nofold_static_zero(
//  CHECK-SAME:                  %[[ARG0:.*]]: tensor<?x?x?xf32>
//       CHECK:   %[[PAD:.*]] = tensor.pad
//       CHECK:   return %[[PAD]]
func @pad_nofold_static_zero(%arg0: tensor<?x?x?xf32>, %pad_value: f32) -> tensor<2x3x4xf32> {
  %c0 = arith.constant 0 : index
  %0 = tensor.pad %arg0 nofold low[0, %c0, 0] high[0, 0, %c0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %pad_value : f32
    } : tensor<?x?x?xf32> to tensor<2x3x4xf32>

  return %0 : tensor<2x3x4xf32>
}

// -----

// CHECK-LABEL: func @fold_collapse_shape_from_elements
func @fold_collapse_shape_from_elements(%arg0: i32) -> tensor<i32> {
  // CHECK: %[[FROM:.+]] = tensor.from_elements %arg0 : tensor<i32>
  // CHECK: return %[[FROM]] : tensor<i32>
  %0 = tensor.from_elements %arg0 : tensor<1xi32>
  %1 = tensor.collapse_shape %0 [] : tensor<1xi32> into tensor<i32>
  return %1 : tensor<i32>
}

// -----

// CHECK-LABEL: func @fold_expand_shape_from_elements
func @fold_expand_shape_from_elements(%arg0: i32) -> tensor<1xi32> {
  // CHECK: %[[FROM:.+]] = tensor.from_elements %arg0 : tensor<1xi32>
  // CHECK: return %[[FROM]] : tensor<1xi32>
  %0 = tensor.from_elements %arg0 : tensor<i32>
  %1 = tensor.expand_shape %0 [] : tensor<i32> into tensor<1xi32>
  return %1 : tensor<1xi32>
}

// -----

// CHECK-LABEL: func @propogate_index_cast
func @propogate_index_cast(%arg0: tensor<1xi32>) -> index {
  // CHECK: %[[IDX:.+]] = arith.constant 0
  // CHECK: %[[EXT:.+]] = tensor.extract %arg0[%[[IDX]]] : tensor<1xi32>
  // CHECK: %[[CAST:.+]] = arith.index_cast %[[EXT]]
  // CHECK: return %[[CAST]] : index
  %c0 = arith.constant 0 : index
  %0 = arith.index_cast %arg0 : tensor<1xi32> to tensor<1xindex>
  %1 = tensor.extract %0[%c0] : tensor<1xindex>
  return %1 : index
}

// -----

// CHECK-LABEL: func @splat_fold
func @splat_fold() -> tensor<4xf32> {
  %c = arith.constant 1.0 : f32
  %t = tensor.splat %c : tensor<4xf32>
  return %t : tensor<4xf32>

  // CHECK-NEXT: [[T:%.*]] = arith.constant dense<1.000000e+00> : tensor<4xf32>
  // CHECK-NEXT: return [[T]] : tensor<4xf32>
}

// -----

// There was an issue in cast + insert_slice folding generating invalid ir.
// https://github.com/llvm/llvm-project/issues/53099
// CHECK-LABEL: func @insert_slice_cast
func @insert_slice_cast(%arg0 : tensor<1x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index, %arg7 : index) -> tensor<?x?xf32> {
  // CHECK: %[[CAST:.*]] = tensor.cast %{{.*}} : tensor<1x?xf32> to tensor<?x?xf32>
  %0 = tensor.cast %arg0 : tensor<1x?xf32> to tensor<?x?xf32>
  // CHECK: %[[RES:.*]] = tensor.insert_slice %[[CAST]]
  // CHECK-SAME: : tensor<?x?xf32> into tensor<?x?xf32>
  %1 = tensor.insert_slice %0 into %arg1[%arg2, %arg3] [%arg4, %arg5] [%arg6, %arg7] : tensor<?x?xf32> into tensor<?x?xf32>
  // CHECK: return %[[RES]] : tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
