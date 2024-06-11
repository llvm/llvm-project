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

// CHECK-LABEL: func @fold_extract
func.func @fold_extract(%arg0 : index) -> (f32, f16, f16, i32, complex<f32>) {
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

// CHECK-LABEL: func @fold_extract_constant_splat
//   CHECK-NOT: tensor.extract_slice
//       CHECK: arith.constant dense<42> : tensor<4x4xi32>
func.func @fold_extract_constant_splat() -> (tensor<4x4xi32>) {
  %cst = arith.constant dense<42> : tensor<1024x1024xi32>
  %1 = tensor.extract_slice %cst[0,0] [4,4] [1, 1] : tensor<1024x1024xi32> to tensor<4x4xi32>
  return %1 : tensor<4x4xi32>
}

// -----

// CHECK-LABEL: func @fold_pack_constant_splat
//   CHECK-NOT: tensor.pack
//       CHECK: arith.constant dense<1.000000e-01> : tensor<8x16x8x32xf32>
func.func @fold_pack_constant_splat(%dest : tensor<8x16x8x32xf32>) -> tensor<8x16x8x32xf32> {
  %cst = arith.constant dense<1.000000e-01> : tensor<64x128xf32>
  %0 = tensor.pack %cst outer_dims_perm = [1, 0] inner_dims_pos = [0, 1]
    inner_tiles = [8, 32] into %dest : tensor<64x128xf32> -> tensor<8x16x8x32xf32>
  return %0 : tensor<8x16x8x32xf32>
}

// -----

// CHECK-LABEL: func @fold_padding_value_pack_constant_splat
//   CHECK-NOT: tensor.pack
//       CHECK: arith.constant dense<1.000000e-01> : tensor<8x16x8x32xf32>
func.func @fold_padding_value_pack_constant_splat(%dest : tensor<8x16x8x32xf32>) -> tensor<8x16x8x32xf32> {
  %pad = arith.constant 1.000000e-01 : f32
  %cst = arith.constant dense<1.000000e-01> : tensor<63x127xf32>
  %0 = tensor.pack %cst
    padding_value(%pad : f32)
    outer_dims_perm = [1, 0] inner_dims_pos = [0, 1]
    inner_tiles = [8, 32] into %dest : tensor<63x127xf32> -> tensor<8x16x8x32xf32>
  return %0 : tensor<8x16x8x32xf32>
}


// -----

// CHECK-LABEL: func @nofold_padding_value_pack_constant_splat
//       CHECK: arith.constant dense<1.000000e-01> : tensor<63x127xf32>
//       CHECK: tensor.pack
func.func @nofold_padding_value_pack_constant_splat(%dest : tensor<8x16x8x32xf32>) -> tensor<8x16x8x32xf32> {
  %pad = arith.constant 0.0 : f32
  %cst = arith.constant dense<1.000000e-01> : tensor<63x127xf32>
  %0 = tensor.pack %cst
    padding_value(%pad : f32)
    outer_dims_perm = [1, 0]
    inner_dims_pos = [0, 1]
    inner_tiles = [8, 32]
    into %dest : tensor<63x127xf32> -> tensor<8x16x8x32xf32>
  return %0 : tensor<8x16x8x32xf32>
}

// -----

func.func @fold_padding_value_pack(%arg0: tensor<1200x500000xf32>) -> tensor<31250x1200x16x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<31250x1200x16x1xf32>
  %pack = tensor.pack %arg0
    padding_value(%cst : f32)
    outer_dims_perm = [1, 0]
    inner_dims_pos = [1, 0]
    inner_tiles = [16, 1]
    into %0 : tensor<1200x500000xf32> -> tensor<31250x1200x16x1xf32>
  return %pack : tensor<31250x1200x16x1xf32>
}
// CHECK-LABEL: func @fold_padding_value_pack
// CHECK-NOT:     padding_value

// -----

func.func @infer_src_shape_pack(%src: tensor<?x?x?x?xf32>, %dest: tensor<10x20x30x40x16xf32>) -> tensor<10x20x30x40x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
   %pack = tensor.pack %src
    padding_value(%cst : f32)
    outer_dims_perm = [2, 1, 3, 0]
    inner_dims_pos = [2]
    inner_tiles = [16]
    into %dest : tensor<?x?x?x?xf32> -> tensor<10x20x30x40x16xf32>
  return %pack : tensor<10x20x30x40x16xf32>
}
// CHECK-LABEL: func.func @infer_src_shape_pack
// CHECK-SAME:    %[[SRC:[0-9a-zA-Z]+]]
// CHECK-SAME:    %[[DEST:[0-9a-zA-Z]+]]
// CHECK:         %[[CAST_SRC:.+]] = tensor.cast %[[SRC]] : tensor<?x?x?x?xf32> to tensor<40x20x?x30xf32>
// CHECK:         %[[PACK:.+]] = tensor.pack %[[CAST_SRC]] {{.+}} into %[[DEST]]
// CHECK:         return %[[PACK]]

// -----

func.func @infer_dest_shape_pack(%src: tensor<30x20x?x10xf32>, %dest: tensor<?x?x?x?x16xf32>) -> tensor<?x?x?x?x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
   %pack = tensor.pack %src
    padding_value(%cst : f32)
    outer_dims_perm = [2, 1, 3, 0]
    inner_dims_pos = [2]
    inner_tiles = [16]
    into %dest : tensor<30x20x?x10xf32> -> tensor<?x?x?x?x16xf32>
  return %pack : tensor<?x?x?x?x16xf32>
}
// CHECK-LABEL: func.func @infer_dest_shape_pack
// CHECK-SAME:    %[[SRC:[0-9a-zA-Z]+]]
// CHECK-SAME:    %[[DEST:[0-9a-zA-Z]+]]
// CHECK:         %[[CAST_DEST:.+]] = tensor.cast %[[DEST]] : tensor<?x?x?x?x16xf32> to tensor<?x20x10x30x16xf32>
// CHECK:         %[[PACK:.+]] = tensor.pack %[[SRC]] {{.+}} into %[[CAST_DEST]]
// CHECK:         %[[CAST_PACK:.+]] = tensor.cast %[[PACK]] : tensor<?x20x10x30x16xf32> to tensor<?x?x?x?x16xf32>
// CHECK:         return %[[CAST_PACK]]

// -----

func.func @no_infer_pack_shape(%arg0: tensor<?x32x100xf32>, %arg1: index) -> tensor<32x7x?x16x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty(%arg1) : tensor<32x7x?x16x1xf32>
  %pack = tensor.pack %arg0 padding_value(%cst : f32) outer_dims_perm = [1, 2, 0] inner_dims_pos = [2, 0] inner_tiles = [16, 1] into %0 : tensor<?x32x100xf32> -> tensor<32x7x?x16x1xf32>
  return %pack : tensor<32x7x?x16x1xf32>
}
// CHECK-LABEL: func.func @no_infer_pack_shape
// CHECK-NOT:     tensor.cast

// -----

func.func @fold_padding_value_pack_negative1(%arg0: tensor<1200x499999xf32>) -> tensor<31250x1200x16x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<31250x1200x16x1xf32>
  %pack = tensor.pack %arg0
    padding_value(%cst : f32)
    outer_dims_perm = [1, 0]
    inner_dims_pos = [1, 0]
    inner_tiles = [16, 1]
    into %0 : tensor<1200x499999xf32> -> tensor<31250x1200x16x1xf32>
  return %pack : tensor<31250x1200x16x1xf32>
}
// CHECK-LABEL: func @fold_padding_value_pack_negative1
// CHECK:         tensor.pack
// CHECK-SAME:      padding_value

// -----

func.func @fold_padding_value_pack_negative2(%arg0: tensor<1200x?xf32>, %arg1: tensor<?x1200x16x1xf32>) -> tensor<?x1200x16x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %pack = tensor.pack %arg0
    padding_value(%cst : f32)
    outer_dims_perm = [1, 0]
    inner_dims_pos = [1, 0]
    inner_tiles = [16, 1]
    into %arg1 : tensor<1200x?xf32> -> tensor<?x1200x16x1xf32>
  return %pack : tensor<?x1200x16x1xf32>
}
// CHECK-LABEL: func @fold_padding_value_pack_negative2
// CHECK:         tensor.pack
// CHECK-SAME:      padding_value

// -----

func.func @fold_padding_value_pack_negative3(%arg0: tensor<1200x500000xf32>, %arg1: tensor<?x1200x?x1xf32>, %tile : index) -> tensor<?x1200x?x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %pack = tensor.pack %arg0
    padding_value(%cst : f32)
    outer_dims_perm = [1, 0]
    inner_dims_pos = [1, 0]
    inner_tiles = [%tile, 1]
    into %arg1 : tensor<1200x500000xf32> -> tensor<?x1200x?x1xf32>
  return %pack : tensor<?x1200x?x1xf32>
}
// CHECK-LABEL: func @fold_padding_value_pack_negative3
// CHECK:         tensor.pack
// CHECK-SAME:      padding_value

// -----

// CHECK-LABEL: func @fold_unpack_constant_splat
//   CHECK-NOT: tensor.unpack
//       CHECK: arith.constant dense<1.000000e-01> : tensor<128x256xf32>
func.func @fold_unpack_constant_splat(%dest : tensor<128x256xf32>) -> tensor<128x256xf32> {
  %cst = arith.constant dense<1.000000e-01> : tensor<16x8x8x32xf32>
  %0 = tensor.unpack %cst inner_dims_pos = [0, 1]
    inner_tiles = [8, 32] into %dest : tensor<16x8x8x32xf32> -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}

// -----

func.func @infer_dest_shape_unpack(%src: tensor<10x20x30x40x16xf32>, %dest: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %unpack = tensor.unpack %src
    outer_dims_perm = [2, 1, 3, 0]
    inner_dims_pos = [2]
    inner_tiles = [16]
    into %dest : tensor<10x20x30x40x16xf32> -> tensor<?x?x?x?xf32>
  return %unpack : tensor<?x?x?x?xf32>
}
// CHECK-LABEL: func.func @infer_dest_shape_unpack
// CHECK-SAME:    %[[SRC:[0-9a-zA-Z]+]]
// CHECK-SAME:    %[[DEST:[0-9a-zA-Z]+]]
// CHECK:         %[[CAST_DEST:.+]] = tensor.cast %[[DEST]] : tensor<?x?x?x?xf32> to tensor<40x20x?x30xf32>
// CHECK:         %[[UNPACK:.+]] = tensor.unpack %[[SRC]] {{.+}} into %[[CAST_DEST]]
// CHECK:         %[[CAST_UNPACK:.+]] = tensor.cast %[[UNPACK]] : tensor<40x20x?x30xf32> to tensor<?x?x?x?xf32>
// CHECK:         return %[[CAST_UNPACK]]

// -----

func.func @infer_src_shape_unpack(%src: tensor<?x?x?x?x16xf32>, %dest: tensor<30x20x?x10xf32>) -> tensor<30x20x?x10xf32> {
  %unpack = tensor.unpack %src
    outer_dims_perm = [2, 1, 3, 0]
    inner_dims_pos = [2]
    inner_tiles = [16]
    into %dest : tensor<?x?x?x?x16xf32> -> tensor<30x20x?x10xf32>
  return %unpack : tensor<30x20x?x10xf32>
}
// CHECK-LABEL: func.func @infer_src_shape_unpack
// CHECK-SAME:    %[[SRC:[0-9a-zA-Z]+]]
// CHECK-SAME:    %[[DEST:[0-9a-zA-Z]+]]
// CHECK:         %[[CAST_SRC:.+]] = tensor.cast %[[SRC]] : tensor<?x?x?x?x16xf32> to tensor<?x20x10x30x16xf32>
// CHECK:         %[[UNPACK:.+]] = tensor.unpack %[[CAST_SRC]]
// CHECK:         return %[[UNPACK]]

// -----

func.func @no_infer_unpack_shape(%arg1: tensor<32x7x?x16x1xf32>, %arg2: index) -> tensor<?x32x100xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty(%arg2) : tensor<?x32x100xf32>
  %unpack = tensor.unpack %arg1 outer_dims_perm = [1, 2, 0] inner_dims_pos = [2, 0] inner_tiles = [16, 1] into %0 : tensor<32x7x?x16x1xf32> -> tensor<?x32x100xf32>
  return %unpack : tensor<?x32x100xf32>
}
// CHECK-LABEL: func.func @no_infer_unpack_shape
// CHECK-NOT:     tensor.cast

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
// CHECK-NEXT     %[[CAST:.*]] = tensor.cast %[[COLLAPSE]] : tensor<96x32xf32> to tensor<?x32xf32>
// CHECK-NEXT     return %[[CAST]] : tensor<?x32xf32>
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
//   CHECK-NOT:   linalg.{{.*}}shape

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
//   CHECK-NOT:   linalg.{{.*}}_shape

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

// CHECK-LABEL: func @propogate_index_cast
func.func @propogate_index_cast(%arg0: tensor<1xi32>) -> index {
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

// CHECK-LABEL: func.func @dont_fold_parallel_insert_slice(
//  CHECK-SAME:     %[[arg0:[0-9a-z]*]]: tensor<1x5xf32>,
//  CHECK-SAME:     %[[arg1:[0-9a-z]*]]: tensor<1x5xf32>)
func.func @dont_fold_parallel_insert_slice(
    %arg0 : tensor<1x5xf32>, %arg1: tensor<1x5xf32>) -> tensor<1x5xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  //      CHECK: scf.forall () in () shared_outs(%[[o:.*]] = %[[arg1]]) -> (tensor<1x5xf32>) {
  // CHECK-NEXT:   scf.forall.in_parallel {
  // CHECK-NEXT:     tensor.parallel_insert_slice %[[arg0]] into %[[o]][0, 0] [1, 5] [1, 1] : tensor<1x5xf32> into tensor<1x5xf32>
  %2 = scf.forall () in () shared_outs(%o = %arg1) -> (tensor<1x5xf32>) {
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %arg0 into %o[%c0, %c0] [1, 5] [%c1, %c1] : tensor<1x5xf32> into tensor<1x5xf32>
    }
  }
  return %2 : tensor<1x5xf32>
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

//       CHECK: #[[$map:.*]] = affine_map<()[s0] -> (s0 floordiv 40)>
// CHECK-LABEL: func @dim_of_expand_shape(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?xf32>
//       CHECK:   %[[c1:.*]] = arith.constant 1 : index
//       CHECK:   %[[dim:.*]] = tensor.dim %[[t]], %[[c1]] : tensor<?x?xf32>
//       CHECK:   %[[apply:.*]] = affine.apply #[[$map]]()[%[[dim]]]
//       CHECK:   return %[[apply]]
func.func @dim_of_expand_shape(%t: tensor<?x?xf32>, %sz0: index, %sz1: index) -> index {
  %c2 = arith.constant 2 : index
  %0 = tensor.expand_shape %t [[0], [1, 2, 3, 4, 5]] output_shape [%sz0, 1, %sz1, 5, 1, 8]
      : tensor<?x?xf32> into tensor<?x1x?x5x1x8xf32>
  %1 = tensor.dim %0, %c2 : tensor<?x1x?x5x1x8xf32>
  return %1 : index
}

// -----

//       CHECK: #[[$map:.*]] = affine_map<()[s0, s1, s2] -> (((s0 * s1) * s2) * 7)>
// CHECK-LABEL: func @dim_of_collapse_shape(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?x?x7x?xf32>
//   CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[c2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[c4:.*]] = arith.constant 4 : index
//   CHECK-DAG:   %[[dim1:.*]] = tensor.dim %[[t]], %[[c1]]
//   CHECK-DAG:   %[[dim2:.*]] = tensor.dim %[[t]], %[[c2]]
//   CHECK-DAG:   %[[dim4:.*]] = tensor.dim %[[t]], %[[c4]]
//       CHECK:   %[[apply:.*]] = affine.apply #[[$map]]()[%[[dim1]], %[[dim2]], %[[dim4]]]
//       CHECK:   return %[[apply]]
func.func @dim_of_collapse_shape(%t: tensor<?x?x?x7x?xf32>) -> index {
  %c1 = arith.constant 1 : index
  %0 = tensor.collapse_shape %t [[0], [1, 2, 3, 4]]
      : tensor<?x?x?x7x?xf32> into tensor<?x?xf32>
  %1 = tensor.dim %0, %c1 : tensor<?x?xf32>
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

// Chain: NC -> NCnc -> NCnc -> NC
// CHECK: func.func @unpack_pack(
// CHECK-SAME: %[[T:.+]]: tensor<128x128xf32>)
// CHECK: return %[[T]] : tensor<128x128xf32>
func.func @unpack_pack(%t: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %tensor_empty = tensor.empty() : tensor<16x16x8x8xf32>
  %packed = tensor.pack %t inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %tensor_empty : tensor<128x128xf32> -> tensor<16x16x8x8xf32>
  %tensor_empty1 = tensor.empty() : tensor<128x128xf32>
  %unpacked = tensor.unpack %packed inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %tensor_empty1 : tensor<16x16x8x8xf32> -> tensor<128x128xf32>
  return %unpacked : tensor<128x128xf32>
}

// -----

// Chain: NC -> NCcn -> NCnc -> NC
// CHECK: func.func @unpack_pack(
// CHECK-SAME: %[[T:.+]]: tensor<128x128xf32>)
// CHECK-NOT: return %[[T]] : tensor<128x128xf32>
func.func @unpack_pack(%t: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %tensor_empty = tensor.empty() : tensor<16x16x8x8xf32>
  %packed = tensor.pack %t inner_dims_pos = [1, 0] inner_tiles = [8, 8] into %tensor_empty : tensor<128x128xf32> -> tensor<16x16x8x8xf32>
  %tensor_empty1 = tensor.empty() : tensor<128x128xf32>
  %unpacked = tensor.unpack %packed inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %tensor_empty1 : tensor<16x16x8x8xf32> -> tensor
<128x128xf32>
  return %unpacked : tensor<128x128xf32>
}

// -----

// Chain: NC -> CNcn -> NCnc -> NC
// CHECK: func.func @unpack_pack(
// CHECK-SAME: %[[T:.+]]: tensor<128x128xf32>)
// CHECK-NOT: return %[[T]] : tensor<128x128xf32>
func.func @unpack_pack(%t: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %tensor_empty = tensor.empty() : tensor<16x16x8x8xf32>
  %packed = tensor.pack %t outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [8, 8] into %tensor_empty : tensor<128x128xf32> -> tensor<16x16x8x8xf32>
  %tensor_empty1 = tensor.empty() : tensor<128x128xf32>
  %unpacked = tensor.unpack %packed inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %tensor_empty1 : tensor<16x16x8x8xf32> -> tensor
<128x128xf32>
  return %unpacked : tensor<128x128xf32>
}

// -----

// Chain: NC -> NCnc -> NCnc -> NC
// CHECK: func.func @unpack_pack(
// CHECK-SAME: %[[T:.+]]: tensor<128x128xf32>,
// CHECK: return %[[T]] : tensor<128x128xf32>
func.func @unpack_pack(%t: tensor<128x128xf32>, %tile1: index, %tile2: index) -> tensor<128x128xf32> {
  %tensor_empty = tensor.empty(%tile1, %tile2) : tensor<16x16x?x?xf32>
  %packed = tensor.pack %t inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty : tensor<128x128xf32> -> tensor<16x16x?x?xf32>
  %tensor_empty1 = tensor.empty() : tensor<128x128xf32>
  %unpacked = tensor.unpack %packed inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty1 : tensor<16x16x?x?xf32> -> tensor
<128x128xf32>
  return %unpacked : tensor<128x128xf32>
}

// -----

// Chain NCnc -> NC -> NC -> NCnc
// CHECK: func.func @pack_unpack(
// CHECK-SAME: %[[T:.+]]: tensor<16x16x?x?xf32>,
// CHECK: return %[[T]] : tensor<16x16x?x?xf32>
func.func @pack_unpack(%t: tensor<16x16x?x?xf32>, %tile1: index, %tile2: index) -> tensor<16x16x?x?xf32> {
  %tensor_empty = tensor.empty() : tensor<128x128xf32>
  %unpacked = tensor.unpack %t inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty : tensor<16x16x?x?xf32> -> tensor<128x128xf32>
  %tensor_empty1 = tensor.empty(%tile1, %tile2) : tensor<16x16x?x?xf32>
  %packed = tensor.pack %unpacked inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty1 : tensor<128x128xf32> -> tensor<16x16x?x?xf32>
  return %packed : tensor<16x16x?x?xf32>
}

// -----

// Chain NCnc -> NC -> NC -> NCnc
// CHECK: func.func @pack_unpack(
// CHECK-SAME: %[[T:.+]]: tensor<16x16x8x8xf32>
// CHECK: return %[[T]] : tensor<16x16x8x8xf32>
func.func @pack_unpack(%t: tensor<16x16x8x8xf32>) -> tensor<16x16x8x8xf32> {
  %tensor_empty = tensor.empty() : tensor<128x128xf32>
  %unpacked = tensor.unpack %t inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %tensor_empty : tensor<16x16x8x8xf32> -> tensor<128x128xf32>
  %tensor_empty1 = tensor.empty() : tensor<16x16x8x8xf32>
  %packed = tensor.pack %unpacked inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %tensor_empty1 : tensor<128x128xf32> -> tensor<16x16x8x8xf32>
  return %packed : tensor<16x16x8x8xf32>
}

// -----

// CHECK: func.func @pack_unpack_same_tiles(
// CHECK-SAME:  %[[T:.+]]: tensor<?x?x?x?xf32>,
// CHECK: return %[[T]] : tensor<?x?x?x?xf32>
func.func @pack_unpack_same_tiles(%t: tensor<?x?x?x?xf32>, %dim1: index, %dim2: index, %dim3: index, %dim4: index, %dim5: index, %dim6: index,
                       %tile1: index, %tile2: index) -> tensor<?x?x?x?xf32> {
  %tensor_empty = tensor.empty(%dim1, %dim2) : tensor<?x?xf32>
  %unpacked = tensor.unpack %t inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  %tensor_empty1 = tensor.empty(%dim3, %dim4, %dim5, %dim6) : tensor<?x?x?x?xf32>
  %packed = tensor.pack %unpacked inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty1 : tensor<?x?xf32> -> tensor<?x?x?x?xf32>
  return %packed : tensor<?x?x?x?xf32>
}

// -----

// CHECK: func.func @pack_unpack_different_tiles(
// CHECK-SAME:  %[[T:.+]]: tensor<?x?x?x?xf32>,
// CHECK-NOT: return %[[T]] : tensor<?x?x?x?xf32>
func.func @pack_unpack_different_tiles(%t: tensor<?x?x?x?xf32>, %dim1: index, %dim2: index, %dim3: index, %dim4: index, %dim5: index, %dim6: index,
                       %tile1: index, %tile2: index) -> tensor<?x?x?x?xf32> {
  %tensor_empty = tensor.empty(%dim1, %dim2) : tensor<?x?xf32>
  %unpacked = tensor.unpack %t inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  %tensor_empty1 = tensor.empty(%dim3, %dim4, %dim5, %dim6) : tensor<?x?x?x?xf32>
  %packed = tensor.pack %unpacked inner_dims_pos = [0, 1] inner_tiles = [%tile2, %tile1] into %tensor_empty1 : tensor<?x?xf32> -> tensor<?x?x?x?xf32>
  return %packed : tensor<?x?x?x?xf32>
}

// -----

// CHECK: func.func @pack_unpack_dynamic_with_padding(
// CHECK-SAME:  %[[T:.+]]: tensor<?x?x?x?xf32>,
// CHECK-NOT: return %[[T]] : tensor<?x?x?x?xf32>
func.func @pack_unpack_dynamic_with_padding(%t: tensor<?x?x?x?xf32>, %dim1: index, %dim2: index, %dim3: index, %dim4: index, %dim5: index, %dim6: index,
                       %tile1: index, %tile2: index, %pad: f32) -> tensor<?x?x?x?xf32> {
  %tensor_empty = tensor.empty(%dim1, %dim2) : tensor<?x?xf32>
  %unpacked = tensor.unpack %t inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  %tensor_empty1 = tensor.empty(%dim3, %dim4, %dim5, %dim6) : tensor<?x?x?x?xf32>
  %packed = tensor.pack %unpacked padding_value(%pad: f32) inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty1 : tensor<?x?xf32> -> tensor<?x?x?x?xf32>
  return %packed : tensor<?x?x?x?xf32>
}

// -----

// CHECK: func.func @pack_outer_dims_unpack_no_outer_dims(
// CHECK-SAME: %[[T:.+]]: tensor<16x16x?x?xf32>,
// CHECK: return %[[T]] : tensor<16x16x?x?xf32>
func.func @pack_outer_dims_unpack_no_outer_dims(%t: tensor<16x16x?x?xf32>, %tile1: index, %tile2: index) -> tensor<16x16x?x?xf32> {
  %tensor_empty = tensor.empty() : tensor<128x128xf32>
  %unpacked = tensor.unpack %t inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty : tensor<16x16x?x?xf32> -> tensor<128x128xf32>
  %tensor_empty1 = tensor.empty(%tile1, %tile2) : tensor<16x16x?x?xf32>
  %packed = tensor.pack %unpacked outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty1 : tensor<128x128xf32> -> tensor<16x16x?x?xf32>
  return %packed : tensor<16x16x?x?xf32>
}

// -----

// CHECK: func.func @pack_no_outer_dims_unpack_outer_dims(
// CHECK-SAME: %[[T:.+]]: tensor<16x16x?x?xf32>,
// CHECK: return %[[T]] : tensor<16x16x?x?xf32>
func.func @pack_no_outer_dims_unpack_outer_dims(%t: tensor<16x16x?x?xf32>, %tile1: index, %tile2: index) -> tensor<16x16x?x?xf32> {
  %tensor_empty = tensor.empty() : tensor<128x128xf32>
  %unpacked = tensor.unpack %t outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty : tensor<16x16x?x?xf32> -> tensor<128x128xf32>
  %tensor_empty1 = tensor.empty(%tile1, %tile2) : tensor<16x16x?x?xf32>
  %packed = tensor.pack %unpacked inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %tensor_empty1 : tensor<128x128xf32> -> tensor<16x16x?x?xf32>
  return %packed : tensor<16x16x?x?xf32>
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

// Fold DstStyleOp -> tensor.unpack operations.
func.func @fold_dst_style_ops_into_unpack(%arg0 : tensor<?x?x16x64xf32>, %init : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %unpack = tensor.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %fill : tensor<?x?x16x64xf32> -> tensor<?x?xf32>
  return %unpack : tensor<?x?xf32>
}
// CHECK-LABEL: func @fold_dst_style_ops_into_unpack
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x16x64xf32>
//  CHECK-SAME:     %[[INIT:.+]]: tensor<?x?xf32>
//       CHECK:   %[[UNPACK:.+]] = tensor.unpack %[[ARG0]]
//  CHECK-SAME:       into %[[INIT]]
//       CHECK:   return %[[UNPACK]]

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

func.func @infer_and_fold_pack_unpack_same_tiles(%t: tensor<10x20x4x4xf32>) -> tensor<10x20x4x4xf32> {
  %dim1 = arith.constant 40 : index
  %dim2 = arith.constant 80 : index
  %tensor_empty = tensor.empty(%dim1, %dim2) : tensor<?x?xf32>
  %unpacked = tensor.unpack %t inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %tensor_empty : tensor<10x20x4x4xf32> -> tensor<?x?xf32>
  %cast = tensor.cast %unpacked : tensor<?x?xf32> to tensor<40x80xf32>
  %tensor_empty1 = tensor.empty() : tensor<10x20x4x4xf32>
  %packed = tensor.pack %cast inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %tensor_empty1 : tensor<40x80xf32> -> tensor<10x20x4x4xf32>
  return %packed : tensor<10x20x4x4xf32>
}
// CHECK-LABEL: func.func @infer_and_fold_pack_unpack_same_tiles
// CHECK-SAME:    %[[SRC:[0-9a-zA-Z]+]]
// CHECK:         return %[[SRC]]

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
