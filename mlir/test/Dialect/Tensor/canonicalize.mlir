// RUN: mlir-opt %s -split-input-file -canonicalize="test-convergence" | FileCheck %s


// CHECK-LABEL: expand_shape_identity_fold
// CHECK-NEXT: return
func.func @expand_shape_identity_fold(%arg0 : tensor<5xf32>) -> tensor<5xf32> {
  %0 = tensor.expand_shape %arg0 [[0]] output_shape [5] : tensor<5xf32> into tensor<5xf32>
  return %0 : tensor<5xf32>
}

// -----

// CHECK-LABEL: expand_shape_rank0_identity_fold
// CHECK-NEXT: return
func.func @expand_shape_rank0_identity_fold(%arg0 : tensor<f32>) -> tensor<f32> {
  %0 = tensor.expand_shape %arg0 [] output_shape [] : tensor<f32> into tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: collapse_shape_identity_fold
// CHECK-NEXT: return
func.func @collapse_shape_identity_fold(%arg0 : tensor<5x4xf32>) -> tensor<5x4xf32> {
  %0 = tensor.collapse_shape %arg0 [[0], [1]] : tensor<5x4xf32> into tensor<5x4xf32>
  return %0 : tensor<5x4xf32>
}

// -----

// CHECK-LABEL: collapse_shape_rank0_identity_fold
// CHECK-NEXT: return
func.func @collapse_shape_rank0_identity_fold(%arg0 : tensor<f32>) -> tensor<f32> {
  %0 = tensor.collapse_shape %arg0 [] : tensor<f32> into tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: @tensor_bitcast_chain_ok
// CHECK-SAME: %[[IN:.*]]: tensor<2xi32>
func.func @tensor_bitcast_chain_ok(%input: tensor<2xi32>) -> tensor<2xf32> {
  // CHECK-NEXT: %[[RES:.*]] = tensor.bitcast %[[IN]] : tensor<2xi32> to tensor<2xf32>
  %0 = tensor.bitcast %input : tensor<2xi32> to tensor<2xui32>
  %1 = tensor.bitcast %0 : tensor<2xui32> to tensor<2xf32>
  // CHECK-NEXT: return %[[RES]]
  return %1 : tensor<2xf32>
}

// -----

// CHECK-LABEL: @tensor_bitcast_chain_nop
// CHECK-SAME: %[[IN:.*]]: tensor<4xi32>
func.func @tensor_bitcast_chain_nop(%input: tensor<4xi32>) -> tensor<4xi32> {
  %0 = tensor.bitcast %input : tensor<4xi32> to tensor<4xui32>
  %1 = tensor.bitcast %0 : tensor<4xui32> to tensor<4xi32>
  // CHECK-NEXT: return %[[IN]]
  return %1 : tensor<4xi32>
}

// -----

// Checks that NOP casts are removed.
// CHECK-LABEL: cast_values
func.func @cast_values(%arg0: tensor<*xi32>) -> tensor<2xi32> {
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
func.func @tensor.cast_chain_ok(%input: tensor<*xi32>) -> tensor<4x8xi32> {
  // CHECK-NEXT: %[[RES:.*]] = tensor.cast %[[IN]] : tensor<*xi32> to tensor<4x8xi32>
  %0 = tensor.cast %input : tensor<*xi32> to tensor<4x?xi32>
  %1 = tensor.cast %0 : tensor<4x?xi32> to tensor<4x8xi32>
  // CHECK-NEXT: return %[[RES]]
  return %1 : tensor<4x8xi32>
}

// -----

// CHECK-LABEL: @tensor.cast_chain_regain
// CHECK-SAME: %[[IN:.*]]: tensor<4xi32>
func.func @tensor.cast_chain_regain(%input: tensor<4xi32>) -> tensor<4xi32> {
  %0 = tensor.cast %input : tensor<4xi32> to tensor<?xi32>
  %1 = tensor.cast %0 : tensor<?xi32> to tensor<4xi32>
  // CHECK-NEXT: return %[[IN]]
  return %1 : tensor<4xi32>
}

// -----

// CHECK-LABEL: @tensor.cast_chain_keep
// CHECK-SAME: %[[IN:.*]]: tensor<?x?xi32>
func.func @tensor.cast_chain_keep(%input: tensor<?x?xi32>) -> tensor<?x8xi32> {
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
func.func @tensor.cast_chain_invalid(%input: tensor<4x8xi32>) -> tensor<8x4xi32> {
  // CHECK-NEXT: %[[C1:.*]] = tensor.cast %[[IN]]
  %0 = tensor.cast %input : tensor<4x8xi32> to tensor<?x?xi32>
  // CHECK-NEXT: %[[C2:.*]] = tensor.cast %[[C1]]
  %1 = tensor.cast %0 : tensor<?x?xi32> to tensor<8x4xi32>
  // CHECK-NEXT: return %[[C2]]
  return %1 : tensor<8x4xi32>
}

// -----

// CHECK-LABEL: fold_concat
// CHECK-SAME: %[[ARG0:.*]]: tensor<1x2x?xi32>
func.func @fold_concat(%arg0: tensor<1x2x?xi32>) -> (tensor<1x2x3xi32>, tensor<1x2x?xi32>) {
  %0 = tensor.concat dim(2) %arg0 : (tensor<1x2x?xi32>) -> tensor<1x2x3xi32>
  // CHECK-NEXT: %[[CAST:.*]] = tensor.cast %[[ARG0]] : tensor<1x2x?xi32> to tensor<1x2x3xi32>
  %1 = tensor.concat dim(2) %arg0 : (tensor<1x2x?xi32>) -> tensor<1x2x?xi32>
  // CHECK-NEXT: return %[[CAST]], %[[ARG0]] : tensor<1x2x3xi32>, tensor<1x2x?xi32>
  return %0, %1 : tensor<1x2x3xi32>, tensor<1x2x?xi32>
}

// -----

// CHECK-LABEL: infer_concat_operand_types
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x12xi32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<?x?xi32>
func.func @infer_concat_operand_types(%arg0: tensor<?x12xi32>, %arg1: tensor<?x?xi32>) -> (tensor<?x12xi32>) {
  // CHECK-NEXT: %[[CAST:.+]] = tensor.cast %[[ARG1]] : tensor<?x?xi32> to tensor<?x12xi32>
  %0 = tensor.concat dim(0) %arg0, %arg1: (tensor<?x12xi32>, tensor<?x?xi32>) -> tensor<?x12xi32>
  // CHECK-NEXT: %[[CONCAT:.+]] = tensor.concat dim(0) %[[ARG0]], %[[CAST]] : (tensor<?x12xi32>, tensor<?x12xi32>) -> tensor<?x12xi32>
  return %0 : tensor<?x12xi32>
  // CHECK-NEXT: return %[[CONCAT]] : tensor<?x12xi32>
}

// -----

// CHECK-LABEL: infer_concat_return_type
// CHECK-SAME: %[[ARG0:.+]]: tensor<5x12xi32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<?x12xi32>
func.func @infer_concat_return_type(%arg0: tensor<5x12xi32>, %arg1: tensor<?x12xi32>) -> (tensor<?x?xi32>) {
  %0 = tensor.concat dim(0) %arg0, %arg1: (tensor<5x12xi32>, tensor<?x12xi32>) -> tensor<?x?xi32>
  // CHECK-NEXT: %[[CONCAT:.+]] = tensor.concat dim(0) %[[ARG0]], %[[ARG1]] : (tensor<5x12xi32>, tensor<?x12xi32>) -> tensor<?x12xi32>
  // CHECK-NEXT: %[[CAST:.+]] = tensor.cast %[[CONCAT]] : tensor<?x12xi32> to tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
  // CHECK-NEXT: return %[[CAST]] : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @fold_extract
func.func @fold_extract(%arg0 : index) -> (f32, f16, f16, i32, complex<f32>, i32) {
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

  // Fold an extract after an insert.
  // CHECK-DAG: [[C6:%.+]] = arith.constant 4 : i32
  %c4_i32 = arith.constant 4 : i32
  %5 = arith.constant dense<[[1, 3], [0, 2]]> : tensor<2x2xi32>
  %inserted = tensor.insert %c4_i32 into %5[%const_1, %const_0] : tensor<2x2xi32>
  %ext_6 = tensor.extract %inserted[%const_1, %const_0] : tensor<2x2xi32>

  // CHECK-NEXT: return [[C4]], [[CM2]], [[C0]], [[C64]], [[C5]], [[C6]]
  return %ext_1, %ext_2, %ext_3, %ext_4, %ext_5, %ext_6 : f32, f16, f16, i32, complex<f32>, i32
}

// -----

// Ensure extract dense resource elements not crash.

// CHECK-LABEL: func @extract_dense_resource_nofold
func.func @extract_dense_resource_nofold() -> i64 {
  // CHECK:      %[[EXT:.+]] = tensor.extract
  // CHECK-NEXT:   return %[[EXT]]
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense_resource<__elided__> : tensor<1xi64>
  %extracted = tensor.extract %cst[%c0] : tensor<1xi64>
  return %extracted : i64
}

// -----

// CHECK-LABEL: func @fold_insert
func.func @fold_insert(%arg0 : index) -> (tensor<4xf32>) {
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
// CHECK-SAME: %[[TENSOR:.*]]: tensor<9xf32>
func.func @extract_from_tensor.cast(%tensor: tensor<9xf32>) -> f32 {
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-NOT: tensor.cast
  %casted = tensor.cast %tensor : tensor<9xf32> to tensor<?xf32>
  // CHECK-NEXT: tensor.extract %[[TENSOR]][%[[C0]]]
  %result = tensor.extract %casted[%c0] : tensor<?xf32>
  return %result : f32
}

// -----

// CHECK-LABEL: func @extract_from_tensor.from_elements
func.func @extract_from_tensor.from_elements(%element : index) -> index {
  // CHECK-SAME: ([[ARG:%.*]]: index)
  %c0 = arith.constant 0 : index
  %tensor = tensor.from_elements %element : tensor<1xindex>
  %extracted_element = tensor.extract %tensor[%c0] : tensor<1xindex>
  // CHECK: [[ARG]] : index
  return %extracted_element : index
}

// -----

// CHECK-LABEL: func @extract_from_tensor.from_elements_0d
func.func @extract_from_tensor.from_elements_0d(%element : index) -> index {
  // CHECK-SAME: ([[ARG:%.*]]: index)
  %c0 = arith.constant 0 : index
  %tensor = tensor.from_elements %element : tensor<index>
  %extracted_element = tensor.extract %tensor[] : tensor<index>
  // CHECK: [[ARG]] : index
  return %extracted_element : index
}

// -----

// CHECK-LABEL: func @extract_from_tensor.from_elements_3d
func.func @extract_from_tensor.from_elements_3d()
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

// CHECK-LABEL: func @extract_from_tensor.from_elements_variable_3d
// CHECK-SAME: %[[ARG_0:[a-zA-Z0-9_]+]]: f32
// CHECK-SAME: %[[ARG_1:[a-zA-Z0-9_]+]]: f32
// CHECK-SAME: %[[ARG_2:[a-zA-Z0-9_]+]]: f32
// CHECK-SAME: %[[ARG_3:[a-zA-Z0-9_]+]]: f32
// CHECK-SAME: %[[ARG_4:[a-zA-Z0-9_]+]]: f32
// CHECK-SAME: %[[ARG_5:[a-zA-Z0-9_]+]]: f32
// CHECK-SAME: %[[ARG_6:[a-zA-Z0-9_]+]]: f32
// CHECK-SAME: %[[ARG_7:[a-zA-Z0-9_]+]]: f32
// CHECK-SAME: %[[ARG_8:[a-zA-Z0-9_]+]]: f32
// CHECK-SAME: %[[ARG_9:[a-zA-Z0-9_]+]]: f32
// CHECK-SAME: %[[ARG_10:[a-zA-Z0-9_]+]]: f32
// CHECK-SAME: %[[ARG_11:[a-zA-Z0-9_]+]]: f32
func.func @extract_from_tensor.from_elements_variable_3d(
    %f0: f32, %f1: f32, %f2: f32, %f3: f32, %f4: f32, %f5: f32,
    %f6: f32, %f7: f32, %f8: f32, %f9: f32, %f10: f32, %f11: f32)
    -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {

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
// CHECK: return %[[ARG_0]], %[[ARG_1]], %[[ARG_2]], %[[ARG_3]], %[[ARG_4]], %[[ARG_5]],
// CHECK-SAME: %[[ARG_6]], %[[ARG_7]], %[[ARG_8]], %[[ARG_9]], %[[ARG_10]], %[[ARG_11]]

// -----

// CHECK-LABEL: func.func @extract_from_elements_complex_i() -> tensor<3xcomplex<i32>> {
// CHECK-NEXT:  %cst = arith.constant dense<[(1,2), (3,2), (1,2)]> : tensor<3xcomplex<i32>>
// CHECK-NEXT:  return %cst : tensor<3xcomplex<i32>>
func.func @extract_from_elements_complex_i() -> tensor<3xcomplex<i32>> {
  %c1 = arith.constant dense<(1, 2)> : tensor<complex<i32>>
  %complex1 = tensor.extract %c1[] : tensor<complex<i32>>
  %c2 = arith.constant dense<(3, 2)> : tensor<complex<i32>>
  %complex2 = tensor.extract %c2[] : tensor<complex<i32>>
  %tensor = tensor.from_elements %complex1, %complex2, %complex1 : tensor<3xcomplex<i32>>
  return %tensor : tensor<3xcomplex<i32>>
}

// -----

// CHECK-LABEL:  func.func @extract_from_elements_complex_f() -> tensor<3xcomplex<f32>> {
// CHECK-NEXT:   %cst = arith.constant dense<[(1.200000e+00,2.300000e+00), (3.200000e+00,2.100000e+00), (1.200000e+00,2.300000e+00)]> : tensor<3xcomplex<f32>>
// CHECK-NEXT:   return %cst : tensor<3xcomplex<f32>>
func.func @extract_from_elements_complex_f() -> tensor<3xcomplex<f32>> {
  %c1 = arith.constant dense<(1.2, 2.3)> : tensor<complex<f32>>
  %complex1 = tensor.extract %c1[] : tensor<complex<f32>>
  %c2 = arith.constant dense<(3.2, 2.1)> : tensor<complex<f32>>
  %complex2 = tensor.extract %c2[] : tensor<complex<f32>>
  %tensor = tensor.from_elements %complex1, %complex2, %complex1 : tensor<3xcomplex<f32>>
  return %tensor : tensor<3xcomplex<f32>>
}

// -----

// Ensure the optimization doesn't segfault from bad constants
// CHECK-LABEL: func @extract_negative_from_tensor.from_elements
func.func @extract_negative_from_tensor.from_elements(%element : index) -> index {
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
func.func @extract_oob_from_tensor.from_elements(%element : index) -> index {
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
func.func @extract_oob_from_tensor.from_elements(%element : index) -> index {
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
func.func @extract_from_tensor.generate(%idx: index, %tensor: tensor<*xf32>) -> index {
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
func.func @extract_from_tensor.generate_2d(%idx0: index, %idx1: index, %tensor: tensor<*xf32>) -> index {
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
func.func @extract_from_tensor.generate_sideeffects(%idx: index, %tensor: tensor<*xf32>, %mem: memref<?xindex>) -> index {
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
func.func @static_tensor.generate(%size1: index, %size4: index) -> tensor<3x?x?x7x?xindex> {
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
func.func @from_elements.constant() -> tensor<3xindex> {
  // CHECK: %[[CST:.*]] = arith.constant dense<[1, 2, 1]> : tensor<3xindex>
  // CHECK: return %[[CST]]
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %tensor = tensor.from_elements %c1, %c2, %c1 : tensor<3xindex>
  return %tensor : tensor<3xindex>
}

// -----

func.func @slice_canonicalize(%arg0 : tensor<?x?x?xf32>, %arg1 : index,
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
//       CHECK:   return %[[RESULT]]

// -----

func.func @rank_reducing_slice_canonicalize(%arg0 : tensor<?x?x?xf32>, %arg1 : index,
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
//       CHECK:   return %[[RESULT]]

// -----

// CHECK-LABEL: func @trivial_slice
//  CHECK-SAME:   %[[ARG0:.[a-z0-9A-Z_]+]]: tensor<4x6x16x32xi8>
//   CHECK-NOT:   tensor.extract_slice
//       CHECK:   return %[[ARG0]] :  tensor<4x6x16x32xi8>
func.func @trivial_slice(%arg0 : tensor<4x6x16x32xi8>) -> tensor<4x6x16x32xi8> {
  %0 = tensor.extract_slice %arg0[0, 0, 0, 0] [4, 6, 16, 32] [1, 1, 1, 1] : tensor<4x6x16x32xi8> to tensor<4x6x16x32xi8>
  return %0 : tensor<4x6x16x32xi8>
}

// -----

// CHECK-LABEL: func @trivial_insert_slice
//  CHECK-SAME:   %[[ARG0:.[a-z0-9A-Z_]+]]: tensor<4x6x16x32xi8>
//   CHECK-NOT:   tensor.extract_slice
//       CHECK:   return %[[ARG0]] :  tensor<4x6x16x32xi8>
func.func @trivial_insert_slice(%arg0 : tensor<4x6x16x32xi8>, %arg1 : tensor<4x6x16x32xi8>) -> tensor<4x6x16x32xi8> {
  %0 = tensor.insert_slice %arg0 into %arg1[0, 0, 0, 0] [4, 6, 16, 32] [1, 1, 1, 1] : tensor<4x6x16x32xi8> into tensor<4x6x16x32xi8>
  return %0 : tensor<4x6x16x32xi8>
}

// -----

// CHECK-LABEL: func @empty_insert_slice
//  CHECK-SAME:   %[[ARG0:.[a-z0-9A-Z_]+]]: tensor<0x2xi8>
//  CHECK-SAME:   %[[ARG1:.[a-z0-9A-Z_]+]]: tensor<3x3xi8>
//   CHECK-NOT:   tensor.extract_slice
//       CHECK:   return %[[ARG1]] :  tensor<3x3xi8>
func.func @empty_insert_slice(%arg0 : tensor<0x2xi8>, %arg1 : tensor<3x3xi8>) -> tensor<3x3xi8> {
  %0 = tensor.insert_slice %arg0 into %arg1[0, 0] [0, 2] [1, 1] : tensor<0x2xi8> into tensor<3x3xi8>
  return %0 : tensor<3x3xi8>
}

// -----

// CHECK-LABEL: func @rank_reducing_tensor_of_cast
//  CHECK-SAME:   %[[ARG0:.[a-z0-9A-Z_]+]]: tensor<4x6x16x32xi8>
//       CHECK:   %[[S:.+]] = tensor.extract_slice %arg0[0, 1, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : tensor<4x6x16x32xi8> to tensor<16x32xi8>
// Tensor cast is moved after slice and then gets canonicalized away.
//   CHECK-NOT:   tensor.cast
//       CHECK:   return %[[S]] : tensor<16x32xi8>
func.func @rank_reducing_tensor_of_cast(%arg : tensor<4x6x16x32xi8>) -> tensor<16x32xi8> {
  %0 = tensor.cast %arg : tensor<4x6x16x32xi8> to tensor<?x?x16x32xi8>
  %1 = tensor.extract_slice %0[0, 1, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : tensor<?x?x16x32xi8> to tensor<16x32xi8>
  return %1 : tensor<16x32xi8>
}

// -----

// CHECK-LABEL: func @out_of_bounds_extract_slice
//       CHECK:   tensor.extract_slice %{{.*}}[0] [%{{.*}}] [1] : tensor<5xf32> to tensor<?xf32>
func.func @out_of_bounds_extract_slice(%t: tensor<5xf32>) -> tensor<?xf32> {
  %c10 = arith.constant 10 : index
  %r = tensor.extract_slice %t[0] [%c10] [1] : tensor<5xf32> to tensor<?xf32>
  return %r : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @out_of_bounds_extract_slice
//       CHECK:   tensor.extract_slice %{{.*}}[0] [10] [1] : tensor<?xf32> to tensor<10xf32>
func.func @out_of_bounds_extract_slice(%t: tensor<5xf32>) -> tensor<10xf32> {
  %t2 = tensor.cast %t : tensor<5xf32> to tensor<?xf32>
  %r = tensor.extract_slice %t2 [0][10][1] : tensor<?xf32> to tensor<10xf32>
  return %r : tensor<10xf32>
}

// -----

// CHECK-LABEL: func @out_of_bounds_insert_slice
//       CHECK:   tensor.insert_slice %{{.*}} into %{{.*}}[%{{.*}}] [5] [1] : tensor<5xf32> into tensor<10xf32>
func.func @out_of_bounds_insert_slice(%src: tensor<5xf32>, %dst: tensor<10xf32>) -> tensor<10xf32> {
  %c10 = arith.constant 10 : index
  %r = tensor.insert_slice %src into %dst[%c10] [5] [1] : tensor<5xf32> into tensor<10xf32>
  return %r : tensor<10xf32>
}

// -----

// CHECK-LABEL: func @out_of_bounds_insert_slice
//       CHECK:   tensor.insert_slice %{{.*}} into %{{.*}}[7] [%{{.*}}] [1] : tensor<?xf32> into tensor<10xf32>
func.func @out_of_bounds_insert_slice(%src: tensor<5xf32>, %dst: tensor<10xf32>, %sz: index) -> tensor<10xf32> {
  %src2 = tensor.cast %src : tensor<5xf32> to tensor<?xf32>
  %r = tensor.insert_slice %src2 into %dst[7] [%sz] [1] : tensor<?xf32> into tensor<10xf32>
  return %r : tensor<10xf32>
}

// -----

// CHECK-LABEL: func @out_of_bounds_insert_slice
//       CHECK:   tensor.insert_slice %{{.*}} into %{{.*}}[7] [5] [1] : tensor<5xf32> into tensor<?xf32>
func.func @out_of_bounds_insert_slice(%src: tensor<5xf32>, %dst: tensor<10xf32>, %sz: index) -> tensor<?xf32> {
  %dst2 = tensor.cast %dst : tensor<10xf32> to tensor<?xf32>
  %r = tensor.insert_slice %src into %dst2[7] [5] [1] : tensor<5xf32> into tensor<?xf32>
  return %r : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @rank_reducing_insert_slice_of_cast
//  CHECK-SAME:   %[[A:.[a-z0-9A-Z_]+]]: tensor<16x32xi8>
//  CHECK-SAME:   %[[B:.[a-z0-9A-Z_]+]]: tensor<4x6x16x32xi8>
//       CHECK:   %[[S:.+]] = tensor.insert_slice %[[A]] into %[[B]][0, 1, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : tensor<16x32xi8> into tensor<4x6x16x32xi8>
// Tensor cast is folded away.
//   CHECK-NOT:   tensor.cast
//       CHECK:   return %[[S]] : tensor<4x6x16x32xi8>
func.func @rank_reducing_insert_slice_of_cast(%a : tensor<16x32xi8>, %b : tensor<4x6x16x32xi8>) -> tensor<4x6x16x32xi8> {
  %c0 = arith.constant 0: index
  %cast = tensor.cast %a : tensor<16x32xi8> to tensor<?x32xi8>
  %sz = tensor.dim %cast, %c0: tensor<?x32xi8>
  %res = tensor.insert_slice %cast into %b[0, 1, 0, 0] [1, 1, %sz, 32] [1, 1, 1, 1] : tensor<?x32xi8> into tensor<4x6x16x32xi8>
  return %res : tensor<4x6x16x32xi8>
}

// -----

func.func @insert_slice_canonicalize(%arg0 : tensor<?x?x?xf32>, %arg1 : index,
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

// Do not insert a cast for the following example. The new source type wouldn't be "more static" than the old one.
func.func @insert_slice_canonicalize_encoding(%arg0 : tensor<2x2xf32, "foo">,
                                              %arg1 : tensor<4x4xf32, "foo">) -> tensor<4x4xf32, "foo">
{
  %0 = tensor.insert_slice %arg0 into %arg1[0, 0] [2, 2] [1, 1] : tensor<2x2xf32, "foo"> into tensor<4x4xf32, "foo">
  return %0 : tensor<4x4xf32, "foo">
}
// CHECK-LABEL: func @insert_slice_canonicalize_encoding
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x2xf32, "foo">
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: tensor<4x4xf32, "foo">
//       CHECK-NOT: tensor.cast
//       CHECK:   %[[RESULT:.+]] = tensor.insert_slice %[[ARG0]] into %[[ARG1]]
//  CHECK-SAME:      [0, 0] [2, 2] [1, 1]
//  CHECK-SAME:      : tensor<2x2xf32, "foo"> into tensor<4x4xf32, "foo">
//       CHECK:   return %[[RESULT]]

// -----

func.func @slice_to_insert_slice_canonicalize(%arg0 : tensor<?x?x?xf32>, %arg1 : index,
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
//       CHECK:   return %[[RESULT]]

// -----

func.func @rank_reducing_insert_slice_canonicalize(%arg0 : tensor<?x?xf32>, %arg1 : index,
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
//       CHECK:   return %[[RESULT]]

// -----

func.func @rank_reducing_slice_to_insert_slice_canonicalize(%arg0 : tensor<?x?x?xf32>, %arg1 : index,
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
//       CHECK:   return %[[RESULT]]

// -----

func.func @insert_slice_propagate_dest_cast(%arg0 : tensor<2x?xi32>, %arg1 : tensor<i32>,
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

func.func @insert_slice_output_dest_canonicalize(%arg0 : tensor<2x3xi32>, %arg1 : tensor<i32>) -> tensor<3x9xi32> {
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
func.func @dim_of_tensor.generate(%arg0: index, %arg1: index) -> index {
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
func.func @fold_dim_of_tensor.cast(%arg0 : tensor<4x?xf32>) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.cast %arg0 : tensor<4x?xf32> to tensor<?x?xf32>
  %1 = tensor.dim %0, %c0 : tensor<?x?xf32>
  %2 = tensor.dim %0, %c1 : tensor<?x?xf32>
  return %1, %2: index, index
}

// -----

// CHECK-LABEL: func @insert_slice_cast
func.func @insert_slice_cast(%arg0 : tensor<1x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index, %arg7 : index) -> tensor<?x?xf32> {
  // CHECK-SAME: %[[ARG0:.*]]: tensor<1x?xf32>
  %0 = tensor.cast %arg0 : tensor<1x?xf32> to tensor<?x?xf32>
  // CHECK: %[[RES:.*]] = tensor.insert_slice %[[ARG0]]
  // CHECK-SAME: [{{.*}}, {{.*}}] [1, {{.*}}] [{{.*}}, {{.*}}]
  // CHECK-SAME: : tensor<1x?xf32> into tensor<?x?xf32>
  %1 = tensor.insert_slice %0 into %arg1[%arg2, %arg3] [%arg4, %arg5] [%arg6, %arg7] : tensor<?x?xf32> into tensor<?x?xf32>
  // CHECK: return %[[RES]] : tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_cast_no_fold
func.func @insert_slice_cast_no_fold(%arg0 : tensor<1x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index, %arg7 : index) -> tensor<?x?xf32> {
  %0 = tensor.cast %arg0 : tensor<1x?xf32> to tensor<?x5xf32>
  // CHECK: %[[CAST:.*]] = tensor.cast
  // CHECK: %[[RES:.*]] = tensor.insert_slice %[[CAST]]
  // CHECK-SAME: [{{.*}}, {{.*}}] [{{.*}}, 5] [{{.*}}, {{.*}}]
  // CHECK-SAME: : tensor<?x5xf32> into tensor<?x?xf32>
  %1 = tensor.insert_slice %0 into %arg1[%arg2, %arg3] [%arg4, 5] [%arg6, %arg7] : tensor<?x5xf32> into tensor<?x?xf32>
  // CHECK: return %[[RES]] : tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @insert_tensor_cast_on_insert_slice_src(
// CHECK-SAME:      %[[arg0:.*]]: tensor<?x5x?xf32>, %[[arg1:.*]]: tensor<?x?x?xf32>
//      CHECK:    %[[cast:.*]] = tensor.cast %[[arg0]] : tensor<?x5x?xf32> to tensor<64x5x64xf32>
//      CHECK:    %[[r:.*]] =  tensor.insert_slice %[[cast]] into %[[arg1]][0, 1, 2] [64, 5, 64] [1, 1, 1] : tensor<64x5x64xf32> into tensor<?x?x?xf32>
//      CHECK:    return %[[r]]
func.func @insert_tensor_cast_on_insert_slice_src(
    %arg0 : tensor<?x5x?xf32>,  %arg1 : tensor<?x?x?xf32>, %sz0: index, %sz2: index) -> tensor<?x?x?xf32> {
  %c64 = arith.constant 64: index
  %r = tensor.insert_slice %arg0 into %arg1[0, 1, 2] [%c64, 5, %c64] [1, 1, 1]
    : tensor<?x5x?xf32> into tensor<?x?x?xf32>
  return %r : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func @fold_extract_insert
//  CHECK-SAME: %{{.+}}: tensor<?x?x?xf32>, %[[SLICE:.+]]: tensor<4x?x8xf32>
func.func @fold_extract_insert(%input : tensor<?x?x?xf32>, %slice: tensor<4x?x8xf32>, %i: index, %size: index) -> (tensor<4x?x8xf32>) {
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %0 = tensor.insert_slice %slice into %input[%c0, %i, 0] [4, %size, 8] [1, 1, %c1] : tensor<4x?x8xf32> into tensor<?x?x?xf32>
  %1 = tensor.extract_slice %0[%c0, %i, 0] [4, %size, 8] [1, 1, %c1] : tensor<?x?x?xf32> to tensor<4x?x8xf32>
  // CHECK: return %[[SLICE]]
  return %1 : tensor<4x?x8xf32>
}

// -----

// CHECK-LABEL: func @fold_gather_constant_splat
//   CHECK-NOT: tensor.gather
//       CHECK: arith.constant dense<1.000000e-01> : tensor<1x2x1x1x1xf32>
func.func @fold_gather_constant_splat(%indices : tensor<1x2x3xindex>) -> tensor<1x2x1x1x1xf32> {
  %cst = arith.constant dense<1.000000e-01> : tensor<4x4x4xf32>
  %0 = tensor.gather %cst[%indices] gather_dims([0, 1, 2]) :
    (tensor<4x4x4xf32>, tensor<1x2x 3xindex>) -> tensor<1x2x 1x1x1xf32>
  return %0 : tensor<1x2x 1x1x1xf32>
}

// -----

// CHECK-LABEL: func @fold_reshape_constant_splat
//   CHECK-NOT: tensor.reshape
//       CHECK: arith.constant dense<1.000000e-01> : tensor<4xf32>
func.func @fold_reshape_constant_splat(%shape : tensor<1xi32>) -> tensor<4xf32> {
  %cst = arith.constant dense<1.000000e-01> : tensor<4x1xf32>
  %0 = tensor.reshape %cst(%shape)
             : (tensor<4x1xf32>, tensor<1xi32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @fold_reshape_chain
//  CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: tensor<*xf32>
//  CHECK-SAME: %[[SHAPE_0:[a-zA-Z0-9_]+]]: tensor<?xindex>
//  CHECK-SAME: %[[SHAPE_1:[a-zA-Z0-9_]+]]: tensor<?xindex>
//  CHECK-SAME: %[[SHAPE_2:[a-zA-Z0-9_]+]]: tensor<?xindex>
//       CHECK: %[[RESULT:.*]] = tensor.reshape %[[INPUT]](%[[SHAPE_2]])
//       CHECK: return %[[RESULT]]
func.func @fold_reshape_chain(%input: tensor<*xf32>, %shape_0: tensor<?xindex>, %shape_1: tensor<?xindex>, %shape_2: tensor<?xindex>) -> tensor<*xf32> {
  %0 = tensor.reshape %input(%shape_0) : (tensor<*xf32>, tensor<?xindex>) -> tensor<*xf32>
  %1 = tensor.reshape %0(%shape_1) : (tensor<*xf32>, tensor<?xindex>) -> tensor<*xf32>
  %2 = tensor.reshape %1(%shape_2) : (tensor<*xf32>, tensor<?xindex>) -> tensor<*xf32>
  return %2 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @fold_reshape_1d
//  CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: tensor<?xf32>
//  CHECK-SAME: %[[SHAPE:[a-zA-Z0-9_]+]]: tensor<1xindex>
//       CHECK: return %[[INPUT]]
func.func @fold_reshape_1d(%input: tensor<?xf32>, %shape: tensor<1xindex>) -> tensor<?xf32> {
  %0 = tensor.reshape %input(%shape) : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @fold_reshape_0d
//  CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: tensor<f32>
//  CHECK-SAME: %[[SHAPE:[a-zA-Z0-9_]+]]: tensor<0xindex>
//       CHECK: return %[[INPUT]]
func.func @fold_reshape_0d(%input: tensor<f32>, %shape: tensor<0xindex>) -> tensor<f32> {
  %0 = tensor.reshape %input(%shape) : (tensor<f32>, tensor<0xindex>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @fold_extract_constant_splat
//   CHECK-NOT: tensor.extract_slice
//       CHECK: arith.constant dense<42> : tensor<4x4xi32>
func.func @fold_extract_constant_splat() -> (tensor<4x4xi32>) {
  %cst = arith.constant dense<42> : tensor<1024x1024xi32>
  %1 = tensor.extract_slice %cst[0,0] [4,4] [1, 1] : tensor<1024x1024xi32> to tensor<4x4xi32>
  return %1 : tensor<4x4xi32>
}

// -----

// CHECK-LABEL: func @fold_overlapping_insert
//  CHECK-SAME: %[[INPUT:.+]]: tensor<?x?x?xf32>, %{{.+}}: tensor<4x?x8xf32>, %[[SLICE2:.+]]: tensor<4x?x8xf32>
func.func @fold_overlapping_insert(%input : tensor<?x?x?xf32>, %slice1: tensor<4x?x8xf32>, %slice2: tensor<4x?x8xf32>, %i: index, %size: index) -> (tensor<?x?x?xf32>) {
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %0 = tensor.insert_slice %slice1 into %input[%c0, %i, 0] [4, %size, 8] [1, 1, %c1] : tensor<4x?x8xf32> into tensor<?x?x?xf32>
  // CHECK: %[[INSERT:.+]] = tensor.insert_slice %[[SLICE2]] into %[[INPUT]]
  %1 = tensor.insert_slice %slice2 into %0[0, %i, 0] [4, %size, 8] [1, 1, %c1] : tensor<4x?x8xf32> into tensor<?x?x?xf32>
  // CHECK: return %[[INSERT]]
  return %1 : tensor<?x?x?xf32>
}

// -----

func.func @compose_expand_of_expand(%arg0 : tensor<?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index)
    -> tensor<?x6x4x?x5xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1], [2]] output_shape [%arg1, 4, %arg2]
      : tensor<?x?xf32> into tensor<?x4x?xf32>
  %1 = tensor.expand_shape %0 [[0, 1], [2], [3, 4]] output_shape [%arg3, 6, 4, %arg4, 5] : tensor<?x4x?xf32> into tensor<?x6x4x?x5xf32>
  return %1 : tensor<?x6x4x?x5xf32>
}
// CHECK-LABEL: compose_expand_of_expand
//       CHECK:   tensor.expand_shape %{{.*}} {{\[}}[0, 1, 2], [3, 4]] output_shape [%arg3, 6, 4, %arg4, 5]
//   CHECK-NOT:   tensor.expand_shape

// -----

func.func @compose_expand_of_expand_of_zero_dim(%arg0 : tensor<f32>)
    -> tensor<1x1x1xf32> {
  %0 = tensor.expand_shape %arg0 [] output_shape [1] : tensor<f32> into tensor<1xf32>
  %1 = tensor.expand_shape %0 [[0, 1, 2]] output_shape [1, 1, 1]
      : tensor<1xf32> into tensor<1x1x1xf32>
  return %1 : tensor<1x1x1xf32>
}
// CHECK-LABEL: compose_expand_of_expand_of_zero_dim
//       CHECK:   tensor.expand_shape %{{.*}} [] output_shape [1, 1, 1]
//  CHECK-SAME:     tensor<f32> into tensor<1x1x1xf32>

// -----

// CHECK-LABEL: func.func @collapse_of_cast(
// CHECK-SAME:         %[[IN:.*]]: tensor<8x12x32xf32>) -> tensor<?x32xf32> {
// CHECK-NEXT:    %[[COLLAPSE:.*]] = tensor.collapse_shape %[[IN]] {{\[}}[0, 1], [2]] : tensor<8x12x32xf32> into tensor<96x32xf32>
// CHECK-NEXT:    %[[CAST:.*]] = tensor.cast %[[COLLAPSE]] : tensor<96x32xf32> to tensor<?x32xf32>
// CHECK-NEXT:    return %[[CAST]] : tensor<?x32xf32>
func.func @collapse_of_cast(%t: tensor<8x12x32xf32>) -> tensor<?x32xf32> {
  %0 = tensor.cast %t : tensor<8x12x32xf32> to tensor<?x?x?xf32>
  %1 = tensor.collapse_shape %0 [[0, 1], [2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
  %2 = tensor.cast %1 : tensor<?x?xf32> to tensor<?x32xf32>
  return %2 : tensor<?x32xf32>
}

// -----

func.func @fold_collapse_of_expand(%arg0 : tensor<12x4xf32>) -> tensor<12x4xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1], [2]] output_shape [3, 4, 4]
      : tensor<12x4xf32> into tensor<3x4x4xf32>
  %1 = tensor.collapse_shape %0 [[0, 1], [2]]
      : tensor<3x4x4xf32> into tensor<12x4xf32>
  return %1 : tensor<12x4xf32>
}
// CHECK-LABEL: @fold_collapse_of_expand
//   CHECK-NOT:   tensor.{{.*}}_shape

// -----

func.func @fold_collapse_of_expand_dynamic(%arg0 : tensor<?x?xf32>, %arg1: index, %arg2: index)
    -> tensor<?x?xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1], [2]] output_shape [%arg1, 4, %arg2]
      : tensor<?x?xf32> into tensor<?x4x?xf32>
  %1 = tensor.collapse_shape %0 [[0, 1], [2]]
      : tensor<?x4x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: @fold_collapse_of_expand_dynamic
//   CHECK-NOT:   tensor.{{.*}}_shape

// -----

func.func @fold_collapse_of_expand_fully_dynamic(%arg0 : tensor<?x?xf32>, %arg1: index, %arg2: index, %arg3: index)
    -> tensor<?x?xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1], [2]] output_shape [%arg1, %arg2, %arg3]
      : tensor<?x?xf32> into tensor<?x?x?xf32>
  %1 = tensor.collapse_shape %0 [[0, 1], [2]]
      : tensor<?x?x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: @fold_collapse_of_expand_fully_dynamic
//   CHECK-NOT:   tensor.{{.*}}_shape

// -----

func.func @no_fold_parallel_collapse_of_expand_dynamic(%arg0 : tensor<?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index)
    -> tensor<?x?x?xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1], [2], [3]] output_shape [%arg1, %arg2, %arg3, %arg4]
      : tensor<?x?x?xf32> into tensor<?x?x?x?xf32>
  %1 = tensor.collapse_shape %0 [[0], [1], [2, 3]]
      : tensor<?x?x?x?xf32> into tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}
// CHECK-LABEL: @no_fold_parallel_collapse_of_expand_dynamic
//       CHECK:   tensor.expand_shape
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape
//       CHECK:   return %[[COLLAPSE]]

// -----

func.func @fold_expand_of_collapse(%arg0 : tensor<3x4x4xf32>) -> tensor<3x4x4xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1], [2]]
      : tensor<3x4x4xf32> into tensor<12x4xf32>
  %1 = tensor.expand_shape %0 [[0, 1], [2]] output_shape [3, 4, 4]
      : tensor<12x4xf32> into tensor<3x4x4xf32>
  return %1 : tensor<3x4x4xf32>
}
// CHECK-LABEL: @fold_expand_of_collapse
//   CHECK-NOT:   tensor.{{.*}}_shape

// -----

func.func @fold_expand_of_collapse_mixed_subshape(%arg0 : tensor<?x4x?xf32>, %arg1: index, %arg2: index)
    -> tensor<?x4x?xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1], [2]]
      : tensor<?x4x?xf32> into tensor<?x?xf32>
  %1 = tensor.expand_shape %0 [[0, 1], [2]] output_shape [%arg1, 4, %arg2]
      : tensor<?x?xf32> into tensor<?x4x?xf32>
  return %1 : tensor<?x4x?xf32>
}
// CHECK-LABEL: @fold_expand_of_collapse_mixed_subshape
//   CHECK-NOT:   tensor.{{.*}}_shape

// -----

func.func @fold_expand_of_collapse_mixed_target_subshape(%arg0 : tensor<?x4x?x2xf32>, %arg1: index, %arg2: index)
    -> tensor<?x4x?xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]]
      : tensor<?x4x?x2xf32> into tensor<?x?xf32>
  %1 = tensor.expand_shape %0 [[0, 1], [2]] output_shape [%arg1, 4, %arg2]
      : tensor<?x?xf32> into tensor<?x4x?xf32>
  return %1 : tensor<?x4x?xf32>
}
// CHECK-LABEL: @fold_expand_of_collapse_mixed_target_subshape
//   CHECK-NOT:   tensor.expand_shape
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %arg0 {{\[}}[0], [1], [2, 3]]
//  CHECK-SAME:     : tensor<?x4x?x2xf32> into tensor<?x4x?xf32>
//  CHECK-NEXT:   return %[[COLLAPSE]]

// -----

func.func @no_fold_expand_of_collapse_fully_dynamic(%arg0 : tensor<?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index)
    -> tensor<?x?x?xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1], [2]]
      : tensor<?x?x?xf32> into tensor<?x?xf32>
  %1 = tensor.expand_shape %0 [[0, 1], [2]] output_shape [%arg1, %arg2, %arg3]
      : tensor<?x?xf32> into tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}
// CHECK-LABEL: @no_fold_expand_of_collapse_fully_dynamic
//       CHECK:   tensor.collapse_shape
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape
//       CHECK:   return %[[EXPAND]]

// -----

func.func @no_fold_expand_of_collapse_adjacent_dynamic(%arg0 : tensor<?x?x?xf32>, %arg1: index, %arg2: index)
    -> tensor<?x?xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1, 2]]
      : tensor<?x?x?xf32> into tensor<?xf32>
  %1 = tensor.expand_shape %0 [[0, 1]] output_shape [%arg1, %arg2]
      : tensor<?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: @no_fold_expand_of_collapse_adjacent_dynamic
//       CHECK:   tensor.collapse_shape
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape
//       CHECK:   return %[[EXPAND]]

// -----

func.func @compose_expand_of_collapse_last_two_dims(%arg0: tensor<?x64x1xf32>) -> tensor<?x384xf32> {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2]] : tensor<?x64x1xf32> into tensor<?xf32>
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %collapsed, %c0 : tensor<?xf32>
  %c384= arith.constant 384 : index
  %div = arith.divui %dim, %c384 : index
  %expanded = tensor.expand_shape %collapsed [[0, 1]] output_shape [%div, 384] : tensor<?xf32> into tensor<?x384xf32>
  return %expanded : tensor<?x384xf32>
}
// CHECK-LABEL: @compose_expand_of_collapse_last_two_dims
//  CHECK-SAME: %[[ARG0:.+]]: tensor<?x64x1xf32>
//       CHECK: %[[CONSTANT384:.+]] = arith.constant 384 : index
//       CHECK: %[[CONSTANT0:.+]] = arith.constant 0 : index
//       CHECK: %[[COLLAPSE:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1, 2]] : tensor<?x64x1xf32> into tensor<?xf32>
//       CHECK: %[[DIM:.+]] = tensor.dim %[[COLLAPSE]], %[[CONSTANT0]] : tensor<?xf32>
//       CHECK: %[[DIVUI:.+]] = arith.divui %[[DIM]], %[[CONSTANT384]] : index
//       CHECK: %[[RESULT:.+]] = tensor.expand_shape %[[COLLAPSE]] {{\[}}[0, 1]] output_shape [%[[DIVUI]], 384] : tensor<?xf32> into tensor<?x384xf32>
//       CHECK: return %[[RESULT]]

// -----

func.func @compose_expand_of_collapse(%arg0 : tensor<2x3x4x5x6x7x8xf32>)
    -> tensor<24x5x42x8xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1, 2, 3, 4, 5, 6]]
      : tensor<2x3x4x5x6x7x8xf32> into tensor<40320xf32>
  %1 = tensor.expand_shape %0 [[0, 1, 2, 3]] output_shape [24, 5, 42, 8]
      : tensor<40320xf32> into tensor<24x5x42x8xf32>
  return %1 : tensor<24x5x42x8xf32>
}
//      CHECK: func @compose_expand_of_collapse
// CHECK-SAME:   %[[ARG0:.+]]: tensor<2x3x4x5x6x7x8xf32>
//      CHECK:   %[[RESULT:.+]] = tensor.collapse_shape %[[ARG0]]
// CHECK-SAME:     [0, 1, 2], [3], [4, 5], [6]
//      CHECK:   return %[[RESULT]]

// -----

func.func @compose_expand_of_collapse_7D(%arg0 : tensor<24x5x42x8xf32>)
    -> tensor<2x3x4x5x6x7x8xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1, 2, 3]]
      : tensor<24x5x42x8xf32> into tensor<40320xf32>
  %1 = tensor.expand_shape %0 [[0, 1, 2, 3, 4, 5, 6]] output_shape [2, 3, 4, 5, 6, 7, 8]
      : tensor<40320xf32> into tensor<2x3x4x5x6x7x8xf32>
  return %1 : tensor<2x3x4x5x6x7x8xf32>
}
//      CHECK: func @compose_expand_of_collapse_7D
// CHECK-SAME:   %[[ARG0:.+]]: tensor<24x5x42x8xf32>
//      CHECK:   %[[RESULT:.+]] = tensor.expand_shape %[[ARG0]]
// CHECK-SAME:     [0, 1, 2], [3], [4, 5], [6]
//      CHECK:   return %[[RESULT]]

// -----

func.func @compose_collapse_of_expand(%arg : tensor<?x?x?xi64>, %arg1: index, %arg2: index, %arg3: index)
    -> tensor<?x?xi64> {
  %0 = tensor.expand_shape %arg [[0], [1], [2, 3]] output_shape [%arg1, %arg2, %arg3, 1]
    : tensor<?x?x?xi64> into tensor<?x?x?x1xi64>
  %1 = tensor.collapse_shape %0 [[0, 1], [2, 3]]
    : tensor<?x?x?x1xi64> into tensor<?x?xi64>
  return %1 : tensor<?x?xi64>
}
// CHECK-LABEL: func @compose_collapse_of_expand
//       CHECK:   (%[[ARG:.*]]: tensor<?x?x?xi64>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
//  CHECK-NEXT: tensor.collapse_shape %[[ARG]]
//  CHECK-SAME:   [0, 1], [2]
//  CHECK-SAME:   : tensor<?x?x?xi64> into tensor<?x?xi64>

// -----

func.func @compose_collapse_of_expand_1D(%arg0 : tensor<2048xf32>)
    -> tensor<4x512xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 4, 1, 512]
    : tensor<2048xf32> into tensor<1x4x1x512xf32>
  %1 = tensor.collapse_shape %0 [[0, 1, 2], [3]]
    : tensor<1x4x1x512xf32> into tensor<4x512xf32>
  return %1 : tensor<4x512xf32>
}
//       CHECK: func @compose_collapse_of_expand_1D
//       CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1]] output_shape [4, 512]
//  CHECK-SAME:   tensor<2048xf32> into tensor<4x512xf32>

// -----

func.func @compose_collapse_of_expand_partially_dynamic(%arg0: tensor<?xf16>, %arg1: index, %arg2: index) -> tensor<8x?x?xf16> {
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [4, 2, %arg1, %arg2, 32] : tensor<?xf16> into tensor<4x2x?x?x32xf16>
  %collapsed = tensor.collapse_shape %expanded [[0, 1], [2], [3, 4]] : tensor<4x2x?x?x32xf16> into tensor<8x?x?xf16>
  return %collapsed : tensor<8x?x?xf16>
}
//       CHECK: func @compose_collapse_of_expand_partially_dynamic
//  CHECK-SAME:   %[[SRC:.[a-zA-Z0-9]+]]
//  CHECK-SAME:   %[[ORIG_D2:.[a-zA-Z0-9]+]]
//  CHECK-SAME:   %[[ORIG_D3:.[a-zA-Z0-9]+]]
//   CHECK-DAG:   %[[C32:.+]] = arith.constant 32
//       CHECK:   %[[COLLAPSED_D2:.+]] = arith.muli %[[ORIG_D3]], %[[C32]]
//       CHECK:   %[[RESULT:.+]] = tensor.expand_shape %[[SRC]]
//  CHECK-SAME:     [0, 1, 2]
//  CHECK-SAME:     output_shape [8, %[[ORIG_D2]], %[[COLLAPSED_D2]]]
//       CHECK:   return %[[RESULT]]

// -----

func.func @compose_expand_of_collapse_0_rank_to_expand(%arg0 : tensor<1x1x1xf32>)
    -> tensor<1x1x1x1xf32> {
  %0 = tensor.collapse_shape %arg0 []
      : tensor<1x1x1xf32> into tensor<f32>
  %1 = tensor.expand_shape %0 [] output_shape [1, 1, 1, 1]
      : tensor<f32> into tensor<1x1x1x1xf32>
  return %1 : tensor<1x1x1x1xf32>
}
//      CHECK: func @compose_expand_of_collapse_0_rank_to_expand
// CHECK-SAME:   %[[ARG0:.+]]: tensor<1x1x1xf32>
//      CHECK:   %[[RESULT:.+]] = tensor.expand_shape %[[ARG0]]
// CHECK-SAME:     {{\[}}[0], [1], [2, 3]] output_shape [1, 1, 1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @compose_expand_of_collapse_0_rank_to_collapse(%arg0 : tensor<1x1x1x1xf32>)
    -> tensor<1x1x1xf32> {
  %0 = tensor.collapse_shape %arg0 []
      : tensor<1x1x1x1xf32> into tensor<f32>
  %1 = tensor.expand_shape %0 [] output_shape [1, 1, 1]
      : tensor<f32> into tensor<1x1x1xf32>
  return %1 : tensor<1x1x1xf32>
}
//      CHECK: func @compose_expand_of_collapse_0_rank_to_collapse
// CHECK-SAME:   %[[ARG0:.+]]: tensor<1x1x1x1xf32>
//      CHECK:   %[[RESULT:.+]] = tensor.collapse_shape %[[ARG0]]
// CHECK-SAME:     [0], [1], [2, 3]
//      CHECK:   return %[[RESULT]]

// -----

func.func @compose_expand_of_collapse_static(%arg0 : tensor<4x32x10x64x2xf16>) -> tensor<4x32x10x128xf16> {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2], [3, 4]] : tensor<4x32x10x64x2xf16> into tensor<128x10x128xf16>
  %expanded = tensor.expand_shape %collapsed [[0, 1], [2], [3]] output_shape [4, 32, 10, 128] : tensor<128x10x128xf16> into tensor<4x32x10x128xf16>
  return %expanded : tensor<4x32x10x128xf16>
}

// CHECK-LABEL: func @compose_expand_of_collapse_static
// CHECK-SAME:   %[[ARG0:.+]]: tensor<4x32x10x64x2xf16>
//      CHECK:   %[[RESULT:.+]] = tensor.collapse_shape %[[ARG0]]
// CHECK-SAME:     [0], [1], [2], [3, 4]
//      CHECK:   return %[[RESULT]]

// -----

func.func @compose_expand_of_collapse_dynamic(%arg0 : tensor<4x?x10x64x2xf16>, %arg1 : index) -> tensor<4x?x10x128xf16> {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2], [3, 4]] : tensor<4x?x10x64x2xf16> into tensor<?x10x128xf16>
  %expanded = tensor.expand_shape %collapsed [[0, 1], [2], [3]] output_shape [4, %arg1,  10, 128] : tensor<?x10x128xf16> into tensor<4x?x10x128xf16>
  return %expanded : tensor<4x?x10x128xf16>
}

// CHECK-LABEL: func @compose_expand_of_collapse_dynamic
// CHECK-SAME:   %[[ARG0:.+]]: tensor<4x?x10x64x2xf16>
//      CHECK:   %[[RESULT:.+]] = tensor.collapse_shape %[[ARG0]]
// CHECK-SAME:     [0], [1], [2], [3, 4]
//      CHECK:   return %[[RESULT]]

// -----

func.func @no_compose_collapse_of_expand_dynamic(%arg0 : tensor<?x8x128x?xf16>, %arg1: index) -> tensor<?x128x?xf16> {
  %collapse = tensor.collapse_shape %arg0 [[0, 1, 2, 3]] : tensor<?x8x128x?xf16> into tensor<?xf16>
  %expanded_19 = tensor.expand_shape %collapse [[0, 1, 2]] output_shape [%arg1, 8, %arg1] : tensor<?xf16> into tensor<?x128x?xf16>
  return %expanded_19 : tensor<?x128x?xf16>
}
// CHECK-LABEL: func @no_compose_collapse_of_expand_dynamic
//  CHECK-SAME:   %[[ARG0:.+]]: tensor
//  CHECK-SAME:   %[[ARG1:.+]]: index
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[ARG0]]
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[COLLAPSE]]
//       CHECK:   return %[[EXPAND]]

// -----

// CHECK-LABEL: func @zero_rank_reshape_multi
func.func @zero_rank_reshape_multi(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: return %arg0
  %0 = tensor.expand_shape %arg0 [] output_shape [1] : tensor<f32> into tensor<1xf32>
  %1 = tensor.expand_shape %0 [[0, 1]] output_shape [1, 1] : tensor<1xf32> into tensor<1x1xf32>
  %2 = tensor.collapse_shape %1 [] : tensor<1x1xf32> into tensor<f32>
  return %2 : tensor<f32>
}

// -----

func.func @compose_collapse_of_collapse(%arg0 : tensor<?x?x?x?x?xf32>)
    -> tensor<?x?xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1], [2], [3, 4]]
      : tensor<?x?x?x?x?xf32> into tensor<?x?x?xf32>
  %1 = tensor.collapse_shape %0 [[0, 1], [2]]
      : tensor<?x?x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func @compose_collapse_of_collapse
//       CHECK:   tensor.collapse_shape %{{.*}} {{\[}}[0, 1, 2], [3, 4]]
//   CHECK-NOT:   tensor.collapse_shape

// -----

func.func @compose_collapse_of_collapse_zero_dim(%arg0 : tensor<1x1x1xf32>)
    -> tensor<f32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1, 2]]
      : tensor<1x1x1xf32> into tensor<1xf32>
  %1 = tensor.collapse_shape %0 [] : tensor<1xf32> into tensor<f32>
  return %1 : tensor<f32>
}
// CHECK-LABEL: func @compose_collapse_of_collapse_zero_dim
//       CHECK:   tensor.collapse_shape %{{.*}} []
//  CHECK-SAME:     tensor<1x1x1xf32> into tensor<f32>

// -----

func.func @fold_collapse_of_expand_1D(%arg0 : tensor<4x512xf32>) -> tensor<2048xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1, 2], [3]] output_shape [1, 4, 1, 512]
    : tensor<4x512xf32> into tensor<1x4x1x512xf32>
  %1 = tensor.collapse_shape %0 [[0, 1, 2, 3]]
    : tensor<1x4x1x512xf32> into tensor<2048xf32>
  return %1 : tensor<2048xf32>
}
//       CHECK: func @fold_collapse_of_expand_1D
//       CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0, 1]]
//  CHECK-SAME:   tensor<4x512xf32> into tensor<2048xf32>

// -----

func.func @fold_collapse_of_expand_unit_dims(%arg0 : tensor<2048x1x1xf32>)
    -> tensor<4x512x1x1xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1, 2, 3], [4], [5]] output_shape [1, 4, 1, 512, 1, 1] : tensor<2048x1x1xf32> into tensor<1x4x1x512x1x1xf32>
  %1 = tensor.collapse_shape %0 [[0, 1, 2], [3], [4], [5]]
    : tensor<1x4x1x512x1x1xf32> into tensor<4x512x1x1xf32>
  return %1 : tensor<4x512x1x1xf32>
}
//       CHECK: func @fold_collapse_of_expand_unit_dims
//       CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1], [2], [3]] output_shape [4, 512, 1, 1]
//  CHECK-SAME:   tensor<2048x1x1xf32> into tensor<4x512x1x1xf32>

// -----

func.func @compose_collapse_of_expand_unit_dims(%arg0 : tensor<2048x1x2048xf32>)
    -> tensor<4x512x1x512x4xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4], [5], [6, 7, 8]] output_shape [1, 4, 1, 512, 1, 1, 512, 1, 4] : tensor<2048x1x2048xf32> into tensor<1x4x1x512x1x1x512x1x4xf32>
  %1 = tensor.collapse_shape %0 [[0, 1, 2], [3, 4], [5], [6, 7], [8]]
    : tensor<1x4x1x512x1x1x512x1x4xf32> into tensor<4x512x1x512x4xf32>
  return %1 : tensor<4x512x1x512x4xf32>
}
//       CHECK: func @compose_collapse_of_expand_unit_dims
//       CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1], [2], [3, 4]] output_shape [4, 512, 1, 512, 4]
//  CHECK-SAME:   tensor<2048x1x2048xf32> into tensor<4x512x1x512x4xf32>

// -----

func.func @compose_collapse_of_expand_trailing_unit_dims(%arg0: tensor<2xf32>)
    -> tensor<2x1xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [2, 1, 1]
      : tensor<2xf32> into tensor<2x1x1xf32>
  %1 = tensor.collapse_shape %0 [[0], [1, 2]]
      : tensor<2x1x1xf32> into tensor<2x1xf32>
  return %1 : tensor<2x1xf32>
}
//       CHECK: func @compose_collapse_of_expand_trailing_unit_dims
//       CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1]] output_shape [2, 1]
//  CHECK-SAME:   tensor<2xf32> into tensor<2x1xf32>

// -----

func.func @compose_collapse_of_collapse_unit_dims_dynamic(
    %arg0 : tensor<?x1x?x1x1x?x?x1x1xf32>) -> tensor<?x?x?x?xf32> {
  %0 = tensor.collapse_shape %arg0 [[0], [1, 2], [3], [4], [5], [6, 7, 8]]
    : tensor<?x1x?x1x1x?x?x1x1xf32> into tensor<?x?x1x1x?x?xf32>
  %1 = tensor.collapse_shape %0 [[0], [1], [2, 3, 4], [5]]
    : tensor<?x?x1x1x?x?xf32> into tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}
//       CHECK: func @compose_collapse_of_collapse_unit_dims_dynamic
//       CHECK: tensor.collapse_shape
//  CHECK-SAME:   [0], [1, 2], [3, 4, 5], [6, 7, 8]
//  CHECK-SAME:   tensor<?x1x?x1x1x?x?x1x1xf32> into tensor<?x?x?x?xf32>

// -----

func.func @fold_collapse_of_expand_trailing_unit_dims(%arg0: tensor<2xf32>)
    -> tensor<2x1xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [2, 1, 1] : tensor<2xf32> into tensor<2x1x1xf32>
  %1 = tensor.collapse_shape %0 [[0], [1, 2]]
      : tensor<2x1x1xf32> into tensor<2x1xf32>
  return %1 : tensor<2x1xf32>
}
//       CHECK: func @fold_collapse_of_expand_trailing_unit_dims
//       CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1]] output_shape [2, 1]
//  CHECK-SAME:   tensor<2xf32> into tensor<2x1xf32>

// -----

func.func @fold_collapse_of_collapse_trailing_unit_dims_dynamic(
    %arg0: tensor<1x1x?x1x1x1xf32>) -> tensor<?xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1, 2], [3], [4], [5]]
      : tensor<1x1x?x1x1x1xf32> into tensor<?x1x1x1xf32>
  %1 = tensor.collapse_shape %0 [[0, 1, 2, 3]]
      : tensor<?x1x1x1xf32> into tensor<?xf32>
  return %1 : tensor<?xf32>
}
//       CHECK: func @fold_collapse_of_collapse_trailing_unit_dims_dynamic
//       CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0, 1, 2, 3, 4, 5]]
//  CHECK-SAME:   tensor<1x1x?x1x1x1xf32> into tensor<?xf32>

// -----

func.func @fold_collapse_of_expand_trailing_unit_dims(%arg0: tensor<12x42x1x1xf32>)
    -> tensor<12x42xf32> {
  %0 = tensor.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [12, 42, 1, 1, 1] : tensor<12x42x1x1xf32> into tensor<12x42x1x1x1xf32>
  %1 = tensor.collapse_shape %0 [[0], [1, 2, 3, 4]]
      : tensor<12x42x1x1x1xf32> into tensor<12x42xf32>
  return %1 : tensor<12x42xf32>
}
//       CHECK: func @fold_collapse_of_expand_trailing_unit_dims
//       CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0], [1, 2, 3]]
//  CHECK-SAME:   tensor<12x42x1x1xf32> into tensor<12x42xf32>

// -----

func.func @fold_collapse_of_expand_unit_dims_in_middle(%arg0 : tensor<?x?x?xf32>, %sz0: index, %sz1: index, %sz2: index)
    -> tensor<?x?xf32> {
  %0 = tensor.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [%sz0, %sz1, 1, %sz2]
      : tensor<?x?x?xf32> into tensor<?x?x1x?xf32>
  %1 = tensor.collapse_shape %0 [[0], [1, 2, 3]]
      : tensor<?x?x1x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func @fold_collapse_of_expand_unit_dims_in_middle
//  CHECK-SAME: (%[[ARG:.*]]: tensor<?x?x?xf32>
//       CHECK: tensor.collapse_shape %[[ARG]] {{\[}}[0], [1, 2]]
//  CHECK-SAME:   tensor<?x?x?xf32> into tensor<?x?xf32>

// -----

func.func @no_fold_collapse_of_expand_incompatible(%arg0 : tensor<4x6x8xf32>)
    -> tensor<2x6x16xf32> {
  %0 = tensor.expand_shape %arg0 [[0, 1], [2, 3], [4]] output_shape [2, 2, 3, 2, 8]
      : tensor<4x6x8xf32> into tensor<2x2x3x2x8xf32>
  %1 = tensor.collapse_shape %0 [[0], [1, 2], [3, 4]]
      : tensor<2x2x3x2x8xf32> into tensor<2x6x16xf32>
  return %1 : tensor<2x6x16xf32>
}
// CHECK-LABEL: func @no_fold_collapse_of_expand_incompatible
//       CHECK:   tensor.expand_shape
//       CHECK:   tensor.collapse_shape

// -----

func.func @no_fold_collapse_of_expand_empty_expr(%arg0: tensor<3x2x2xf32>)
    -> tensor<12x1xf32> {
  %0 = tensor.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [3, 2, 2, 1]
      : tensor<3x2x2xf32> into tensor<3x2x2x1xf32>
  %1 = tensor.collapse_shape %0 [[0, 1, 2], [3]]
      : tensor<3x2x2x1xf32> into tensor<12x1xf32>
  return %1 : tensor<12x1xf32>
}
//      CHECK: func @no_fold_collapse_of_expand_empty_expr
// CHECK-SAME:    %[[ARG0:.+]]: tensor<3x2x2xf32>
//      CHECK:    %[[RARG0:.+]] = tensor.expand_shape %[[ARG0]]
// CHECK-SAME:      {{\[}}[0], [1], [2, 3]] output_shape [3, 2, 2, 1]
//      CHECK:    %[[RES:.+]] = tensor.collapse_shape %[[RARG0]]
// CHECK-SAME:      [0, 1, 2], [3]
//      CHECK:    return %[[RES:.+]] : tensor<12x1xf32>

// -----

func.func @reshape_splat_constant_int32() -> tensor<2x4x2xi32> {
  %c0 = arith.constant dense<42> : tensor<2x8xi32>
  %0 = tensor.expand_shape %c0 [[0], [1, 2]] output_shape [2, 4, 2]
      : tensor<2x8xi32> into tensor<2x4x2xi32>
  return %0 : tensor<2x4x2xi32>
}
// CHECK-LABEL: @reshape_splat_constant_int32
//       CHECK:   %[[CST:.*]] = arith.constant dense<{{.*}}> : tensor<2x4x2xi32>
//   CHECK-NOT:   tensor.expand_shape
//       CHECK:   return %[[CST]]
// -----
func.func @expand_shape_splat(%arg : f32) -> tensor<2x2x2xf32> {
  %c0 = tensor.splat %arg : tensor<2x4xf32>
  %0 = tensor.expand_shape %c0 [[0], [1, 2]] output_shape [2, 2, 2]
      : tensor<2x4xf32> into tensor<2x2x2xf32>
  return %0 : tensor<2x2x2xf32>
}
// CHECK-LABEL: @expand_shape_splat
// CHECK-SAME:    %[[ARG0:.+]]: f32
//       CHECK:   %[[CST:.*]] = tensor.splat %[[ARG0:.+]] : tensor<2x2x2xf32>
//   CHECK-NOT:   tensor.expand_shape
//       CHECK:   return %[[CST]]

// -----

// CHECK-LABEL: @expand_shape_splat_dynamic_no_fold
// CHECK-SAME: (%[[F:.+]]: f32, %[[M:.+]]: index, %[[SZ0:.+]]: index)
func.func @expand_shape_splat_dynamic_no_fold(%arg: f32, %m: index, %sz0: index) -> tensor<2x2x?xf32> {
  // CHECK: %[[SPLAT:.+]] = tensor.splat %[[F]][%[[M]]] : tensor<2x?xf32>
  // CHECK: %[[EXPANDED:.+]] = tensor.expand_shape %[[SPLAT]]
  %c0 = tensor.splat %arg[%m] : tensor<2x?xf32>
  %0 = tensor.expand_shape %c0 [[0], [1, 2]] output_shape [2, 2, %sz0] : tensor<2x?xf32> into tensor<2x2x?xf32>
  return %0 : tensor<2x2x?xf32>
}

// -----

func.func @collapse_shape_splat(%arg : f32) -> tensor<2x4xf32> {
  %c0 = tensor.splat %arg : tensor<2x2x2xf32>
  %0 = tensor.collapse_shape %c0 [[0], [1, 2]]
      : tensor<2x2x2xf32> into tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
// CHECK-LABEL: @collapse_shape_splat
// CHECK-SAME:    %[[ARG0:.+]]: f32
//       CHECK:   %[[CST:.*]] = tensor.splat %[[ARG0:.+]] : tensor<2x4xf32>
//   CHECK-NOT:   tensor.collapse_shape
//       CHECK:   return %[[CST]]

// -----

// CHECK-LABEL: @collapse_shape_splat_dynamic_no_fold
// CHECK-SAME: %[[F:.+]]: f32
// CHECK-SAME: %[[M:.+]]: index
func.func @collapse_shape_splat_dynamic_no_fold(%f: f32, %m: index) -> tensor<2x?xf32> {
  // CHECK: %[[SPLAT:.+]] = tensor.splat %[[F]][%[[M]]]
  // CHECK: %[[COLLAPSED:.+]] = tensor.collapse_shape %[[SPLAT]]
  %c0 = tensor.splat %f[%m] : tensor<2x2x?xf32>
  %0 = tensor.collapse_shape %c0 [[0], [1, 2]] : tensor<2x2x?xf32> into tensor<2x?xf32>
  return %0 : tensor<2x?xf32>
}

// -----

func.func @reshape_splat_constant_int16() -> tensor<2x4x2xi16> {
  %c0 = arith.constant dense<42> : tensor<2x8xi16>
  %0 = tensor.expand_shape %c0 [[0], [1, 2]] output_shape [2, 4, 2]
      : tensor<2x8xi16> into tensor<2x4x2xi16>
  return %0 : tensor<2x4x2xi16>
}
// CHECK-LABEL: @reshape_splat_constant_int16
//       CHECK:   %[[CST:.*]] = arith.constant dense<{{.*}}> : tensor<2x4x2xi16>
//   CHECK-NOT:   tensor.expand_shape
//       CHECK:   return %[[CST]]

// -----

func.func @reshape_splat_constant_float32() -> tensor<2x4x2xf32> {
  %c0 = arith.constant dense<42.0> : tensor<2x8xf32>
  %0 = tensor.expand_shape %c0 [[0], [1, 2]] output_shape [2, 4, 2]
      : tensor<2x8xf32> into tensor<2x4x2xf32>
  return %0 : tensor<2x4x2xf32>
}
// CHECK-LABEL: @reshape_splat_constant_float32
//       CHECK:   %[[CST:.*]] = arith.constant dense<{{.*}}> : tensor<2x4x2xf32>
//   CHECK-NOT:   tensor.expand_shape
//       CHECK:   return %[[CST]]

// -----

func.func @reshape_splat_constant_float64() -> tensor<2x4x2xf64> {
  %c0 = arith.constant dense<42.0> : tensor<2x8xf64>
  %0 = tensor.expand_shape %c0 [[0], [1, 2]] output_shape [2, 4, 2]
      : tensor<2x8xf64> into tensor<2x4x2xf64>
  return %0 : tensor<2x4x2xf64>
}
// CHECK-LABEL: @reshape_splat_constant_float64
//       CHECK:   %[[CST:.*]] = arith.constant dense<{{.*}}> : tensor<2x4x2xf64>
//   CHECK-NOT:   tensor.expand_shape
//       CHECK:   return %[[CST]]

// -----

// CHECK-LABEL: func @fold_rank
func.func @fold_rank() -> (index) {
  %const_0 = arith.constant dense<[[[1, -2, 1, 36]], [[0, 2, -1, 64]]]>
    : tensor<2x1x4xi32>

  // Fold a ank into a constant
  // CHECK-NEXT: [[C3:%.+]] = arith.constant 3 : index
  %rank_0 = tensor.rank %const_0 : tensor<2x1x4xi32>

  // CHECK-NEXT: return [[C3]]
  return %rank_0 : index
}

// -----

// CHECK-LABEL: func @pad_same_static_shape(
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<5x6xf32>
//   CHECK-NOT:   tensor.pad
//       CHECK:   return %[[ARG0]]
func.func @pad_same_static_shape(%arg0: tensor<5x6xf32>, %a: index)
    -> tensor<5x6xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.pad %arg0 low[%a, 0] high[0, %a] {
        ^bb0(%arg1: index, %arg2: index):
          tensor.yield %cst : f32
  } : tensor<5x6xf32> to tensor<5x6xf32>
  return %0 : tensor<5x6xf32>
}

// -----

// CHECK-LABEL:   func @pad_fold_static(
// CHECK-SAME:      %[[INPUT:.*]]: tensor<?x64x?x?xf32>) -> tensor<?x?x?x?xf32> {
// CHECK:           %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NOT:       arith.constant 4 : index
// CHECK:           %[[PADDED:.*]] = tensor.pad %[[INPUT]]
// CHECK-SAME:        low[0, 4, 1, 1] high[0, 4, 1, 1]  {
// CHECK:           ^bb0(%[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index):
// CHECK:             tensor.yield %[[CST]] : f32
// CHECK:           } : tensor<?x64x?x?xf32> to tensor<?x72x?x?xf32>
// CHECK:           tensor.cast
func.func @pad_fold_static(%arg0: tensor<?x64x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %padding = arith.constant 4 : index
  %padded = tensor.pad %arg0 low[0, %padding, 1, 1] high[0, %padding, 1, 1]  {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst: f32
  } : tensor<?x64x?x?xf32> to tensor<?x?x?x?xf32>
  return %padded : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func @pad_nofold_same_static_shape(
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<5x6xf32>
//       CHECK:   %[[PAD:.*]] = tensor.pad
//       CHECK:   return %[[PAD]]
func.func @pad_nofold_same_static_shape(%arg0: tensor<5x6xf32>, %a: index)
    -> tensor<5x6xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.pad %arg0 nofold low[%a, 0] high[0, %a] {
        ^bb0(%arg1: index, %arg2: index):
          tensor.yield %cst : f32
  } : tensor<5x6xf32> to tensor<5x6xf32>
  return %0 : tensor<5x6xf32>
}

// -----

// CHECK-LABEL:   func @pad_after_cast_different_shape(
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
func.func @pad_after_cast_different_shape(%arg0: tensor<?x64x?x?xf32>)
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

// CHECK-LABEL:   func @pad_after_cast_same_shape(
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
func.func @pad_after_cast_same_shape(%arg0: tensor<?x64x?x?xf32>, %padding : index)
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

// CHECK-LABEL: func @pad_of_cast(
// CHECK-NOT:     tensor.cast
// CHECK:         tensor.pad
// CHECK:         tensor<8x?xf32> to tensor<8x32xf32>
func.func @pad_of_cast(%t: tensor<8x?xf32>, %s: index) -> tensor<8x32xf32> {
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
func.func @cast_of_pad_more_static(%arg0: tensor<?x?xf32>, %padding: index) -> tensor<32x32xf32> {
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
func.func @cast_of_pad_less_static(%arg0: tensor<32x?x?xf32>, %padding: index) -> tensor<?x32x32xf32> {
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

func.func @pad_cast_fold(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = tensor.cast %arg0 : tensor<4x4xf32> to tensor<?x?xf32>
  %1 = tensor.pad %0 low[%c0, %c0] high[%c0, %c0]  {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
// CHECK-LABEL: @pad_cast
// CHECK-SAME: %[[ARG0:.+]]: tensor<4x4xf32>
// CHECK: return %[[ARG0]]

// -----

// CHECK-LABEL: func @fold_pad_source_cast(
//  CHECK-SAME:                  %[[ARG0:.*]]: tensor<4x?xf32>
//   CHECK-NOT:   tensor.cast
//       CHECK:   %[[RESULT:.*]] = tensor.pad %[[ARG0]]
func.func @fold_pad_source_cast(%arg0: tensor<4x?xf32>) -> tensor<4x4xf32> {
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
func.func @pad_static_zero_cast(%arg0: tensor<?x?x?xf32>, %pad_value: f32) -> tensor<2x3x4xf32> {
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
func.func @pad_nofold_static_zero(%arg0: tensor<?x?x?xf32>, %pad_value: f32) -> tensor<2x3x4xf32> {
  %c0 = arith.constant 0 : index
  %0 = tensor.pad %arg0 nofold low[0, %c0, 0] high[0, 0, %c0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %pad_value : f32
    } : tensor<?x?x?xf32> to tensor<2x3x4xf32>

  return %0 : tensor<2x3x4xf32>
}

// -----

// CHECK-LABEL: func @fold_orthogonal_pad_chains(
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<64x64xf32>,
//  CHECK-SAME:   %[[SZ0:.*]]: index, %[[SZ1:.*]]: index, %[[PW0:.*]]: index, %[[PW1:.*]]: index
func.func @fold_orthogonal_pad_chains(%arg0: tensor<64x64xf32>,
                                      %sz0 : index, %sz1 : index,
                                      %pw0 : index, %pw1 : index) -> tensor<8x4xf32> {
  //       CHECK:   %[[T0:.*]] = tensor.extract_slice %[[ARG0]]
  //  CHECK-SAME:                     [16, 4] [%[[SZ0]], %[[SZ1]]]
  //       CHECK:   %[[PAD:.*]] = tensor.pad %[[T0]] nofold
  //  CHECK-SAME:                     high[%[[PW0]], %[[PW1]]]
  //       CHECK:   return %[[PAD]]
  %pad_value = arith.constant 0.0 : f32
  %0 = tensor.extract_slice %arg0[16, 0] [%sz0, 64] [1, 1] : tensor<64x64xf32> to tensor<?x64xf32>
  %1 = tensor.pad %0 low[0, 0] high[%pw0, 0] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : f32
    } : tensor<?x64xf32> to tensor<8x64xf32>
  %2 = tensor.extract_slice %1[0, 4] [8, %sz1] [1, 1] : tensor<8x64xf32> to tensor<8x?xf32>
  %3 = tensor.pad %2 nofold low[0, 0] high[0, %pw1] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : f32
    } : tensor<8x?xf32> to tensor<8x4xf32>
  func.return %3 : tensor<8x4xf32>
}

// -----

// CHECK-LABEL: func @dont_fold_pad_chains(
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<64x64xf32>,
//  CHECK-SAME:   %[[SZ0:.*]]: index, %[[SZ1:.*]]: index, %[[PW0:.*]]: index, %[[PW1:.*]]: index
func.func @dont_fold_pad_chains(%arg0: tensor<64x64xf32>,
                                %sz0 : index, %sz1 : index,
                                %pw0 : index, %pw1 : index) -> (tensor<8x4xf32>, tensor<4x64xf32>, tensor<8x4xf32>, tensor<6x4xf32>) {
  //       CHECK:   %[[T0:.*]] = tensor.extract_slice %[[ARG0]]
  //       CHECK:   %[[T1:.*]] = tensor.pad %[[T0]]
  %pad_value = arith.constant 0.0 : f32
  %0 = tensor.extract_slice %arg0[16, 0] [%sz0, 64] [1, 1] : tensor<64x64xf32> to tensor<?x64xf32>
  %1 = tensor.pad %0 low[0, 0] high[%pw0, 0] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : f32
    } : tensor<?x64xf32> to tensor<8x64xf32>

  // Don't fold if the padding values are different.
  //       CHECK:   %[[T2:.*]] = tensor.extract_slice %[[T1]]
  //  CHECK-SAME:                     [0, 4] [8, %[[SZ1]]]
  //       CHECK:   %[[PAD0:.*]] = tensor.pad %[[T2]]
  %different_value = arith.constant 1.0 : f32
  %2 = tensor.extract_slice %1[0, 4] [8, %sz1] [1, 1] : tensor<8x64xf32> to tensor<8x?xf32>
  %3 = tensor.pad %2 nofold low[0, 0] high[0, %pw1] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %different_value : f32
    } : tensor<8x?xf32> to tensor<8x4xf32>

  // Don't fold if the pad ops have common padding dimensions.
  //       CHECK:   %[[T3:.*]] = tensor.extract_slice %[[T1]]
  //  CHECK-SAME:                     [4, 0] [%[[SZ1]], 64]
  //       CHECK:   %[[PAD1:.*]] = tensor.pad %[[T3]]
  %4 = tensor.extract_slice %1[4, 0] [%sz1, 64] [1, 1] : tensor<8x64xf32> to tensor<?x64xf32>
  %5 = tensor.pad %4 nofold low[0, 0] high[%pw1, 0] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : f32
    } : tensor<?x64xf32> to tensor<4x64xf32>

  // Don't fold if padded source tensor dimension is accessed at an offset.
  //       CHECK:   %[[T4:.*]] = tensor.extract_slice %[[T1]]
  //  CHECK-SAME:                     [%[[SZ0]], 4] [8, %[[SZ1]]
  //       CHECK:   %[[PAD2:.*]] = tensor.pad %[[T4]]
  %6 = tensor.extract_slice %1[%sz0, 4] [8, %sz1] [1, 1] : tensor<8x64xf32> to tensor<8x?xf32>
  %7 = tensor.pad %6 nofold low[0, 0] high[0, %pw1] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : f32
    } : tensor<8x?xf32> to tensor<8x4xf32>

  // Don't fold if a padded source tensor dimension is sliced.
  //       CHECK:   %[[T5:.*]] = tensor.extract_slice %[[T1]]
  //  CHECK-SAME:                     [0, 4] [6, %[[SZ1]]
  //       CHECK:   %[[PAD3:.*]] = tensor.pad %[[T5]]
  %8 = tensor.extract_slice %1[0, 4] [6, %sz1] [1, 1] : tensor<8x64xf32> to tensor<6x?xf32>
  %9 = tensor.pad %8 nofold low[0, 0] high[0, %pw1] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : f32
    } : tensor<6x?xf32> to tensor<6x4xf32>

  //       CHECK:   return %[[PAD0]], %[[PAD1]], %[[PAD2]], %[[PAD3]]
  func.return %3, %5, %7, %9 : tensor<8x4xf32>, tensor<4x64xf32>, tensor<8x4xf32>, tensor<6x4xf32>
}

// -----

// CHECK-LABEL: func @merge_constant_padding
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<2x3xf32>
//  CHECK-SAME:   %[[PADVAL:[A-Za-z0-9]+]]: f32
//       CHECK:   %[[PAD:.+]] = tensor.pad %[[ARG0]] low[1, 3] high[4, 2]
//       CHECK:     tensor.yield %[[PADVAL]]
//       CHECK:   return %[[PAD]]
func.func @merge_constant_padding(%arg0: tensor<2x3xf32>, %pad_value: f32) -> tensor<7x8xf32> {
  %pad0 = tensor.pad %arg0 low[1, 1] high[1, 0] {
    ^bb0(%b0: index, %b1 : index):
      tensor.yield %pad_value : f32
    } : tensor<2x3xf32> to tensor<4x4xf32>
  %pad1 = tensor.pad %pad0 low[0, 2] high[3, 2] {
    ^bb0(%b2: index, %b3 : index):
      tensor.yield %pad_value : f32
    } : tensor<4x4xf32> to tensor<7x8xf32>
  return %pad1 : tensor<7x8xf32>
}

// -----

//       CHECK: #[[$MAP:.*]] = affine_map<()[s0] -> (s0 + 1)>
// CHECK-LABEL: func @merge_constant_padding_dynamic
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[IDX:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[PADVAL:[A-Za-z0-9]+]]: f32
//       CHECK:   %[[HIGH:.+]] = affine.apply #[[$MAP]]()[%[[IDX]]]
//       CHECK:   %[[PAD:.+]] = tensor.pad %[[ARG0]] low[%[[IDX]], 3] high[%[[HIGH]], 2]
//       CHECK:     tensor.yield %[[PADVAL]]
//       CHECK:   return %[[PAD]]
func.func @merge_constant_padding_dynamic(%arg0: tensor<?x?xf32>, %idx: index, %pad_value: f32) -> tensor<?x?xf32> {
  %pad0 = tensor.pad %arg0 low[%idx, 1] high[1, 0] {
    ^bb0(%b0: index, %b1 : index):
      tensor.yield %pad_value : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
  %pad1 = tensor.pad %pad0 low[0, 2] high[%idx, 2] {
    ^bb0(%b2: index, %b3 : index):
      tensor.yield %pad_value : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
  return %pad1 : tensor<?x?xf32>
}

// -----

// Verify that folding does not happen if it would drop a nofold attribute
// CHECK-LABEL: func @dont_merge_constant_padding_nofold
//       CHECK:   tensor.pad {{.*}} nofold
//       CHECK:   tensor.pad
func.func @dont_merge_constant_padding_nofold(%arg0: tensor<2x3xf32>, %pad_value: f32) -> tensor<7x8xf32> {
  %pad0 = tensor.pad %arg0 nofold low[1, 1] high[1, 0] {
    ^bb0(%b0: index, %b1 : index):
      tensor.yield %pad_value : f32
    } : tensor<2x3xf32> to tensor<4x4xf32>
  %pad1 = tensor.pad %pad0 low[0, 2] high[3, 2] {
    ^bb0(%b2: index, %b3 : index):
      tensor.yield %pad_value : f32
    } : tensor<4x4xf32> to tensor<7x8xf32>
  return %pad1 : tensor<7x8xf32>
}

// -----

// Verify that folding does not happen if it would drop a nofold attribute
// CHECK-LABEL: func @dont_merge_constant_padding_different_vals
//       CHECK:   tensor.pad
//       CHECK:   tensor.pad
func.func @dont_merge_constant_padding_different_vals(
    %arg0: tensor<2x3xf32>,
    %pad_value0: f32,
    %pad_value1: f32) -> tensor<7x8xf32> {
  %pad0 = tensor.pad %arg0 low[1, 1] high[1, 0] {
    ^bb0(%b0: index, %b1 : index):
      tensor.yield %pad_value0 : f32
    } : tensor<2x3xf32> to tensor<4x4xf32>
  %pad1 = tensor.pad %pad0 low[0, 2] high[3, 2] {
    ^bb0(%b2: index, %b3 : index):
      tensor.yield %pad_value1 : f32
    } : tensor<4x4xf32> to tensor<7x8xf32>
  return %pad1 : tensor<7x8xf32>
}

// -----

// CHECK-LABEL: func @fold_collapse_shape_from_elements
func.func @fold_collapse_shape_from_elements(%arg0: i32) -> tensor<i32> {
  // CHECK: %[[FROM:.+]] = tensor.from_elements %arg0 : tensor<i32>
  // CHECK: return %[[FROM]] : tensor<i32>
  %0 = tensor.from_elements %arg0 : tensor<1xi32>
  %1 = tensor.collapse_shape %0 [] : tensor<1xi32> into tensor<i32>
  return %1 : tensor<i32>
}

// -----

// CHECK-LABEL: func @fold_expand_shape_from_elements
func.func @fold_expand_shape_from_elements(%arg0: i32) -> tensor<1xi32> {
  // CHECK: %[[FROM:.+]] = tensor.from_elements %arg0 : tensor<1xi32>
  // CHECK: return %[[FROM]] : tensor<1xi32>
  %0 = tensor.from_elements %arg0 : tensor<i32>
  %1 = tensor.expand_shape %0 [] output_shape [1] : tensor<i32> into tensor<1xi32>
  return %1 : tensor<1xi32>
}

// -----

// CHECK-LABEL: func @propagate_index_cast
func.func @propagate_index_cast(%arg0: tensor<1xi32>) -> index {
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
func.func @splat_fold() -> tensor<4xf32> {
  %c = arith.constant 1.0 : f32
  %t = tensor.splat %c : tensor<4xf32>
  return %t : tensor<4xf32>

  // CHECK-NEXT: [[T:%.*]] = arith.constant dense<1.000000e+00> : tensor<4xf32>
  // CHECK-NEXT: return [[T]] : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @splat_dynamic_no_fold
// CHECK-SAME: %[[M:.+]]: index
func.func @splat_dynamic_no_fold(%m: index) -> tensor<4x?xf32> {
  // CHECK: %[[F:.+]] = arith.constant
  %f = arith.constant 1.0 : f32

  // CHECK: tensor.splat %[[F]][%[[M]]] : tensor<4x?xf32>
  %t = tensor.splat %f[%m] : tensor<4x?xf32>
  return %t : tensor<4x?xf32>
}

// -----

// CHECK-LABEL: func @cast_extract_slice
func.func @cast_extract_slice(%arg0 : tensor<128x512xf32>, %s : index, %o : index)
    -> tensor<16x512xf32> {
// CHECK: %[[E:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}, 0] [16, 512] [1, 1] : tensor<128x512xf32> to tensor<16x512xf32>
  %0 = tensor.extract_slice %arg0[%o, 0] [%s, 512] [1, 1] : tensor<128x512xf32> to tensor<?x512xf32>
  %1 = tensor.cast %0 : tensor<?x512xf32> to tensor<16x512xf32>
// CHECK: return %[[E]] : tensor<16x512xf32>
  return %1 : tensor<16x512xf32>
}

// -----

// CHECK-LABEL: func @cast_extract_slice_rank_reduce
func.func @cast_extract_slice_rank_reduce(%arg0 : tensor<128x512xf32>, %s : index, %o : index)
    -> tensor<16xf32> {
// CHECK: %[[E:.*]]  = tensor.extract_slice %{{.*}}[%{{.*}}, 0] [16, 1] [1, 1] : tensor<128x512xf32> to tensor<16xf32>
  %0 = tensor.extract_slice %arg0[%o, 0] [%s, 1] [1, 1] : tensor<128x512xf32> to tensor<?xf32>
  %1 = tensor.cast %0 : tensor<?xf32> to tensor<16xf32>
// CHECK: return %[[E]] : tensor<16xf32>
  return %1 : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @canonicalize_parallel_insert_slice_indices(
//  CHECK-SAME:     %[[arg0:[0-9a-z]*]]: tensor<1x5xf32>,
//  CHECK-SAME:     %[[arg1:[0-9a-z]*]]: tensor<?x?xf32>,
//  CHECK-SAME:     %[[num_threads:[0-9a-z]*]]: index
func.func @canonicalize_parallel_insert_slice_indices(
    %arg0 : tensor<1x5xf32>, %arg1: tensor<?x?xf32>,
    %num_threads : index) -> tensor<?x?xf32>
{
  %cst = arith.constant 4.200000e+01 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  //  CHECK-NOT: tensor.cast
  //      CHECK: scf.forall (%[[tidx:[0-9a-z]*]]) in (%[[num_threads]]) shared_outs(%[[o:.*]] = %[[arg1]]) -> (tensor<?x?xf32>) {
  // CHECK-NEXT:   scf.forall.in_parallel {
  // CHECK-NEXT:     tensor.parallel_insert_slice %[[arg0]] into %[[o]][%[[tidx]], 0] [1, 5] [1, 1]
  %2 = scf.forall (%tidx) in (%num_threads) shared_outs(%o = %arg1) -> (tensor<?x?xf32>) {
    %3 = tensor.cast %arg0 : tensor<1x5xf32> to tensor<?x5xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %3 into %o[%tidx, %c0] [%c1, 5] [%c1, %c1] : tensor<?x5xf32> into tensor<?x?xf32>
    }
  }
  return %2 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @fold_insert_slice_after_extract_slice
//  CHECK-SAME: (%[[INPUT:.+]]: tensor<1x2x2x4xf32>)
func.func @fold_insert_slice_after_extract_slice(%input: tensor<1x2x2x4xf32>) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %0 = tensor.extract_slice %input[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
  %1 = tensor.insert_slice %0 into %input[%c0, 0, %c0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
  // CHECK: return %[[INPUT]]
  return %1: tensor<1x2x2x4xf32>
}

// -----

// CHECK-LABEL: func.func @dont_fold_mismatched_source_dst
func.func @dont_fold_mismatched_source_dst(%input0: tensor<1x2x2x4xf32>, %input1: tensor<1x2x2x4xf32>) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  // CHECK: tensor.extract_slice
  %0 = tensor.extract_slice %input0[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
  // CHECK: tensor.insert_slice
  %1 = tensor.insert_slice %0 into %input1[%c0, 0, %c0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
  return %1: tensor<1x2x2x4xf32>
}

// -----

// CHECK-LABEL: func.func @dont_fold_mismatched_parameters
func.func @dont_fold_mismatched_parameters(%input: tensor<1x2x2x4xf32>) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  // CHECK: tensor.extract_slice
  %0 = tensor.extract_slice %input[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
  // CHECK: tensor.insert_slice
  %1 = tensor.insert_slice %0 into %input[%c0, 1, %c0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
  return %1: tensor<1x2x2x4xf32>
}

// -----

func.func @empty_canonicalize() -> (tensor<4x5x?xf32>) {
  %c6 = arith.constant 6 : index
  %0 = tensor.empty(%c6) : tensor<4x5x?xf32>
  return %0 : tensor<4x5x?xf32>
}
// CHECK: func @empty_canonicalize
// CHECK:   %[[T0:.+]] = tensor.empty() : tensor<4x5x6xf32>
// CHECK:   %[[T1:.+]] = tensor.cast %[[T0]] : tensor<4x5x6xf32> to tensor<4x5x?xf32>
// CHECK:   return %[[T1]]

// -----

func.func @fold_empty_tensor_with_cast(%arg0 : index) -> tensor<1x12xf32> {
  %0 = tensor.empty(%arg0) : tensor<?x12xf32>
  %1 = tensor.cast %0 : tensor<?x12xf32> to tensor<1x12xf32>
  return %1 : tensor<1x12xf32>
}
//      CHECK: func @fold_empty_tensor_with_cast(%[[ARG0:.+]]: index)
//      CHECK:   %[[T0:.+]] = tensor.empty() : tensor<1x12xf32>
//      CHECK:   return %[[T0]] : tensor<1x12xf32>

// -----

func.func private @some_use(%i : index, %j : index)

// CHECK-LABEL: func @empty_tensor_canonicalize
//  CHECK-SAME:   %[[I:.*]]: index
func.func @empty_tensor_canonicalize(%i : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK-NOT: tensor.empty
  %0 = tensor.empty(%i) : tensor<?x42xf32>

  // CHECK-NOT: tensor.dim
  %1 = tensor.dim %0, %c0: tensor<?x42xf32>
  %2 = tensor.dim %0, %c1: tensor<?x42xf32>

  // CHECK: %[[c42:.*]] = arith.constant 42 : index
  // CHECK: call @some_use(%[[I]], %[[c42]])
  call @some_use(%1, %2) : (index, index) -> ()

  return
}

// -----

// CHECK-LABEL: func @dim_of_expand_shape(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?xf32>
//       CHECK:   %[[c2:.*]] = arith.constant 2 : index
//       CHECK:   %[[expanded:.*]] = tensor.expand_shape %[[t]] {{\[\[}}0], [1, 2, 3, 4, 5]] output_shape [%arg1, 1, %arg2, 5, 1, 8] : tensor<?x?xf32> into tensor<?x1x?x5x1x8xf32>
//       CHECK:   %[[dim:.*]] = tensor.dim %[[expanded]], %[[c2]] : tensor<?x1x?x5x1x8xf32>
//       CHECK:   return %[[dim]]
func.func @dim_of_expand_shape(%t: tensor<?x?xf32>, %sz0: index, %sz1: index) -> index {
  %c2 = arith.constant 2 : index
  %0 = tensor.expand_shape %t [[0], [1, 2, 3, 4, 5]] output_shape [%sz0, 1, %sz1, 5, 1, 8]
      : tensor<?x?xf32> into tensor<?x1x?x5x1x8xf32>
  %1 = tensor.dim %0, %c2 : tensor<?x1x?x5x1x8xf32>
  return %1 : index
}

// -----

// CHECK-LABEL: func @dim_of_collapse_shape(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?x?x7x?xf32>
//   CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[collapsed:.*]] = tensor.collapse_shape %[[t]] {{\[\[}}0], [1, 2, 3, 4]] : tensor<?x?x?x7x?xf32> into tensor<?x?xf32>
//   CHECK-DAG:   %[[dim:.*]] = tensor.dim %[[collapsed]], %[[c1]]
//       CHECK:   return %[[dim]]
func.func @dim_of_collapse_shape(%t: tensor<?x?x?x7x?xf32>) -> index {
  %c1 = arith.constant 1 : index
  %0 = tensor.collapse_shape %t [[0], [1, 2, 3, 4]]
      : tensor<?x?x?x7x?xf32> into tensor<?x?xf32>
  %1 = tensor.dim %0, %c1 : tensor<?x?xf32>
  return %1 : index
}

// -----

// Can't fold when dim is out of bound.
// CHECK-LABEL: func @out_of_bound_dim_of_collapse_shape(
//       CHECK:   %[[DIM:.*]] = tensor.dim
//       CHECK:   return %[[DIM]]
func.func @out_of_bound_dim_of_collapse_shape(%t: tensor<?x?x?x7x?xf32>) -> index {
  %c5 = arith.constant 5 : index
  %0 = tensor.collapse_shape %t [[0], [1, 2, 3, 4]]
      : tensor<?x?x?x7x?xf32> into tensor<?x?xf32>
  %1 = tensor.dim %0, %c5 : tensor<?x?xf32>
  return %1 : index
}

// -----

// CHECK-LABEL: func @collapse_expand_fold_to_cast(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>
//       CHECK:   return %[[t]]
func.func @collapse_expand_fold_to_cast(%t: tensor<?xf32>, %sz0: index) -> (tensor<?xf32>)
{
  %0 = tensor.expand_shape %t [[0, 1]] output_shape [1, %sz0] : tensor<?xf32> into tensor<1x?xf32>
  %1 = tensor.collapse_shape %0 [[0, 1]] : tensor<1x?xf32> into tensor<?xf32>
  return %1 : tensor<?xf32>
}

// -----

// CHECK: func.func @invalid_empty_negative_size
// CHECK: %[[IDX:.*]] = index.constant
// CHECK: %[[T:.*]] = tensor.empty(%[[IDX]]) : tensor<4x5x?xf32>
func.func @invalid_empty_negative_size() -> (tensor<4x5x?xf32>) {
  %c1 = arith.constant 1 : index
  %cn2 = arith.constant 2 : index
  %0 = index.sub %c1, %cn2
  %1 = tensor.empty(%0) : tensor<4x5x?xf32>
  return %1 : tensor<4x5x?xf32>
}

// -----

// The IR in this test case in invalid. This test tests that the canonicalizer
// does not crash.

// CHECK-LABEL: func @invalid_slice_ops(
//       CHECK:   %[[c:.*]] = arith.constant -5 : index
//       CHECK:   tensor.extract_slice {{.*}}%[[c]]
//       CHECK:   tensor.insert_slice {{.*}}%[[c]]
func.func @invalid_slice_ops(%t: tensor<?xf32>, %t2: tensor<?xf32>) -> tensor<?xf32> {
  %c = arith.constant -5 : index
  %0 = tensor.extract_slice %t[0][%c][1] : tensor<?xf32> to tensor<?xf32>
  %1 = tensor.insert_slice %0 into %t2[2][%c][1] : tensor<?xf32> into tensor<?xf32>
  return %1 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @generate_negative_size_verifies(
//       CHECK:   %[[c:.*]] = arith.constant -8 : index
//       CHECK:   tensor.generate %[[c]]
//       CHECK:   : tensor<?x8xi32>
func.func @generate_negative_size_verifies() -> tensor<?x8xi32> {
  %cst = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %size = affine.max affine_map<(d0) -> (d0 mod 64 - 8)>(%c0)
  %tensor = tensor.generate %size {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %cst : i32
  } : tensor<?x8xi32>
  return %tensor : tensor<?x8xi32>
}


// -----

// Test case: Folding of tensor.dim(tensor.reshape %v %shp, %idx) -> tensor.extract %shp[%idx]
// CHECK-LABEL: func @dim_of_reshape(
//  CHECK-SAME:     %[[MEM:[0-9a-z]+]]: tensor<*xf32>,
//  CHECK-SAME:     %[[SHP:[0-9a-z]+]]: tensor<?xindex>
//  CHECK-NEXT:   %[[IDX:.*]] = arith.constant 3
//  CHECK-NEXT:   %[[DIM:.*]] = tensor.extract %[[SHP]][%[[IDX]]]
//   CHECK-NOT:   tensor.store
//   CHECK-NOT:   tensor.dim
//   CHECK-NOT: tensor.reshape
//       CHECK:   return %[[DIM]] : index
func.func @dim_of_reshape(%arg0: tensor<*xf32>, %arg1: tensor<?xindex>)
    -> index {
  %c3 = arith.constant 3 : index
  %0 = tensor.reshape %arg0(%arg1)
      : (tensor<*xf32>, tensor<?xindex>) -> tensor<*xf32>
  // Update the shape to test that the load ends up in the right place.
  tensor.insert %c3 into %arg1[%c3] : tensor<?xindex>
  %1 = tensor.dim %0, %c3 : tensor<*xf32>
  return %1 : index
}

// -----

// Test case: Folding of tensor.dim(tensor.reshape %v %shp, %idx) -> tensor.extract %shp[%idx]
// CHECK-LABEL: func @dim_of_reshape_i32(
//       CHECK:  tensor.extract
//  CHECK-NEXT:  %[[CAST:.*]] = arith.index_cast
//   CHECK-NOT:  tensor.dim
//   CHECK-NOT:  tensor.reshape
//       CHECK:  return %[[CAST]] : index
func.func @dim_of_reshape_i32(%arg0: tensor<*xf32>, %arg1: tensor<?xi32>)
    -> index {
    %c3 = arith.constant 3 : index
    %0 = tensor.reshape %arg0(%arg1)
        : (tensor<*xf32>, tensor<?xi32>) -> tensor<*xf32>
    %1 = tensor.dim %0, %c3 : tensor<*xf32>
    return %1 : index
}

// -----

// Test case: tensor.dim(tensor.reshape %v %shp, %idx) is folded into tensor.extract %shp[%idx]
// CHECK-LABEL: func @dim_of_reshape_for(
//       CHECK: scf.for
//  CHECK-NEXT: tensor.extract
//   CHECK-NOT: tensor.dim
//   CHECK-NOT: tensor.reshape
func.func @dim_of_reshape_for( %arg0: tensor<*xf32>, %arg1: tensor<?xindex>) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    %0 = tensor.reshape %arg0(%arg1) : (tensor<*xf32>, tensor<?xindex>) -> tensor<*xf32>

    %1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %c1) -> (index) {
      %2 = tensor.dim %0, %arg2 : tensor<*xf32>
      %3 = arith.muli %arg3, %2 : index
      scf.yield %3 : index
    }
    return %1 : index
}

// -----

// Test case: tensor.dim(tensor.reshape %v %shp, %idx) is folded into tensor.extract %shp[%idx]
// CHECK-LABEL: func @dim_of_reshape_undominated(
//       CHECK: arith.muli
//  CHECK-NEXT: tensor.extract
//   CHECK-NOT: tensor.dim
//   CHECK-NOT: tensor.reshape
func.func @dim_of_reshape_undominated(%arg0: tensor<*xf32>, %arg1: tensor<?xindex>, %arg2: index) -> index {
    %c4 = arith.constant 4 : index
    %reshape = tensor.reshape %arg0(%arg1) : (tensor<*xf32>, tensor<?xindex>) -> tensor<*xf32>
    %0 = arith.muli %arg2, %c4 : index
    %dim = tensor.dim %reshape, %0 : tensor<*xf32>
    return %dim : index
  }

// -----

// CHECK-LABEL: @reshape_fold_2d
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xi32>
func.func @reshape_fold_2d(%arg0 : tensor<?x?xi32>) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xi32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xi32>
  %ds = tensor.from_elements %d0, %d1 : tensor<2xindex>
  %reshape = tensor.reshape %arg0(%ds) : (tensor<?x?xi32>, tensor<2xindex>) -> tensor<?x?xi32>
  // CHECK: return %[[ARG0]]
  return %reshape : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: @reshape_nofold_2d
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xi32>
func.func @reshape_nofold_2d(%arg0 : tensor<?x?xi32>) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xi32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xi32>
  %ds = tensor.from_elements %d1, %d0 : tensor<2xindex>
  // CHECK: tensor.reshape
  %reshape = tensor.reshape %arg0(%ds) : (tensor<?x?xi32>, tensor<2xindex>) -> tensor<?x?xi32>
  return %reshape : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: @reshape_nofold_2d_ins
func.func @reshape_nofold_2d_ins(%arg0 : tensor<?x?xi32>, %arg1: index, %arg2: index) -> tensor<?x?xi32> {
  %ds = tensor.from_elements %arg1, %arg2 : tensor<2xindex>
  // CHECK: tensor.reshape
  %reshape = tensor.reshape %arg0(%ds) : (tensor<?x?xi32>, tensor<2xindex>) -> tensor<?x?xi32>
  return %reshape : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: @reshape_fold_3d_cst
// CHECK-SAME: %[[ARG0:.+]]: tensor<5x?x?xi32>
func.func @reshape_fold_3d_cst(%arg0 : tensor<5x?x?xi32>) -> tensor<5x?x?xi32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = arith.constant 5 : index
  %d1 = tensor.dim %arg0, %c1 : tensor<5x?x?xi32>
  %d2 = tensor.dim %arg0, %c2 : tensor<5x?x?xi32>
  %ds = tensor.from_elements %d0, %d1, %d2 : tensor<3xindex>
  %reshape = tensor.reshape %arg0(%ds) : (tensor<5x?x?xi32>, tensor<3xindex>) -> tensor<5x?x?xi32>
  // CHECK: return %[[ARG0]]
  return %reshape : tensor<5x?x?xi32>
}

// -----

// Test case: This test fails to fold because the index of tensor.dim is out_of_bounds
// CHECK-LABEL: func @dim_out_of_bounds(
//       CHECK: %[[IDX:.*]] = index.constant 28
//  CHECK-NEXT: bufferization.alloc_tensor
//  CHECK-NEXT: %[[DIM:.*]] = tensor.dim %{{.*}}, %[[IDX]]
//  CHECK-NEXT: memref.alloc
//  CHECK-NEXT: memref.cast
//  CHECK-NEXT: affine.vector_load %{{.*}}[{{.*}}, {{.*}}, symbol(%[[DIM]])]
//  CHECK-NEXT: return
func.func @dim_out_of_bounds() -> vector<7xi32> {
    %c1 = arith.constant 1 : index
    %idx28 = index.constant 28
    %c29 = arith.constant 29 : index
    %3 = bufferization.alloc_tensor(%c29) : tensor<?xi16>
    %dim = tensor.dim %3, %idx28 : tensor<?xi16>
    %alloc_21 = memref.alloc(%c29) : memref<?x26x2xi32>
    %16 = affine.vector_load %alloc_21[%c1, %c1, %dim] : memref<?x26x2xi32>, vector<7xi32>
    return %16 : vector<7xi32>
}

// -----

// CHECK-LABEL:   func.func @fold_cast_multiple_results(
// CHECK-SAME:         %[[ARG1:.*]]: tensor<2x2xf32>,
// CHECK-SAME:         %[[ARG2:.*]]: tensor<2x2xf32>) -> index {
// CHECK:           %[[RES:.*]]:2 = test.destination_style_op ins(%[[ARG1]] : tensor<2x2xf32>)
// CHECK-SAME:      outs(%[[ARG2]] : tensor<2x2xf32>) -> tensor<2x2xf32>, index
// CHECK:           return %[[RES]]#1 : index
func.func @fold_cast_multiple_results(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> index {
  %cast = tensor.cast %arg0 : tensor<2x2xf32> to tensor<?x2xf32>
  %cast_0 = tensor.cast %arg1 : tensor<2x2xf32> to tensor<?x2xf32>
  %0:2 = test.destination_style_op ins(%cast : tensor<?x2xf32>) outs(%cast_0 : tensor<?x2xf32>) -> tensor<?x2xf32>, index
  return %0#1 : index
}


// -----

func.func @fold_expand_of_cast(%arg0 : tensor<10x10xf32>)
    -> tensor<10x1x10xf32> {
  %c1 = arith.constant 1 : index 
  %c10 = arith.constant 10 : index 
  %0 = tensor.cast %arg0 : tensor<10x10xf32> to tensor<?x?xf32>
  %1 = tensor.expand_shape %0 [[0, 1], [2]] output_shape [%c10, %c1, %c10]
      : tensor<?x?xf32> into tensor<?x?x?xf32>
  %2 = tensor.cast %1 : tensor<?x?x?xf32> to tensor<10x1x10xf32>
  return %2 : tensor<10x1x10xf32>
}
// CHECK-LABEL:  func.func @fold_expand_of_cast
//       CHECK:   %[[RES:.+]] = tensor.expand_shape %{{.*}} {{\[}}[0, 1], [2]] output_shape [10, 1, 10]
//       CHECK:   return %[[RES]]

// -----

func.func @sink_expand_of_cast(%arg0 : tensor<?x10xf32>)
    -> tensor<?x?x?xf32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = tensor.cast %arg0 : tensor<?x10xf32> to tensor<?x?xf32>
  %1 = tensor.expand_shape %0 [[0, 1], [2]] output_shape [%c10, %c1, %c10]
      : tensor<?x?xf32> into tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}
// CHECK-LABEL:  func.func @sink_expand_of_cast
//   CHECK-DAG:   %[[C10:.*]] = arith.constant 10
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %{{.*}} {{\[}}[0, 1], [2]] 
//  CHECK-SAME:     output_shape [%[[C10]], %[[C1]], 10]
//       CHECK:   %[[RES:.+]] = tensor.cast %[[EXPAND]]
//       CHECK:   return %[[RES]]

// -----

func.func @partial_sink_expand_of_cast(%arg0 : tensor<10x10xf32>, %arg1 : index, %arg2 : index)
    -> tensor<?x?x?xf32> {
  %c10 = arith.constant 10 : index
  %0 = tensor.cast %arg0 : tensor<10x10xf32> to tensor<?x?xf32>
  %1 = tensor.expand_shape %0 [[0, 1], [2]] output_shape [%arg1, %arg2, %c10]
      : tensor<?x?xf32> into tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}
// CHECK-LABEL:  func.func @partial_sink_expand_of_cast
//       CHECK:   %[[CAST:.+]] = tensor.cast
//  CHECK-SAME:     tensor<10x10xf32> to tensor<?x10xf32>
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %{{.*}} {{\[}}[0, 1], [2]] 
//  CHECK-SAME:     output_shape [%{{.*}}, %{{.*}}, 10]
//       CHECK:   %[[RES:.+]] = tensor.cast %[[EXPAND]]
//  CHECK-SAME:     tensor<?x?x10xf32> to tensor<?x?x?xf32>
//       CHECK:   return %[[RES]]
