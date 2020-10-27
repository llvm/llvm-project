// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -test-constant-fold | FileCheck %s

// -----

// CHECK-LABEL: @affine_for
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @affine_for(%p : memref<f32>) {
  // CHECK: [[C:%.+]] = constant 6.{{0*}}e+00 : f32
  affine.for %arg1 = 0 to 128 {
    affine.for %arg2 = 0 to 8 { // CHECK: affine.for %{{.*}} = 0 to 8 {
      %0 = constant 4.5 : f32
      %1 = constant 1.5 : f32

      %2 = addf %0, %1 : f32

      // CHECK-NEXT: store [[C]], [[ARG]][]
      store %2, %p[] : memref<f32>
    }
  }
  return
}

// -----

// CHECK-LABEL: func @simple_addf
func @simple_addf() -> f32 {
  %0 = constant 4.5 : f32
  %1 = constant 1.5 : f32

  // CHECK-NEXT: [[C:%.+]] = constant 6.{{0*}}e+00 : f32
  %2 = addf %0, %1 : f32

  // CHECK-NEXT: return [[C]]
  return %2 : f32
}

// -----

// CHECK-LABEL: func @addf_splat_tensor
func @addf_splat_tensor() -> tensor<4xf32> {
  %0 = constant dense<4.5> : tensor<4xf32>
  %1 = constant dense<1.5> : tensor<4xf32>

  // CHECK-NEXT: [[C:%.+]] = constant dense<6.{{0*}}e+00> : tensor<4xf32>
  %2 = addf %0, %1 : tensor<4xf32>

  // CHECK-NEXT: return [[C]]
  return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @addf_dense_tensor
func @addf_dense_tensor() -> tensor<4xf32> {
  %0 = constant dense<[1.5, 2.5, 3.5, 4.5]> : tensor<4xf32>
  %1 = constant dense<[1.5, 2.5, 3.5, 4.5]> : tensor<4xf32>

  // CHECK-NEXT: [[C:%.+]] = constant dense<[3.{{0*}}e+00, 5.{{0*}}e+00, 7.{{0*}}e+00, 9.{{0*}}e+00]> : tensor<4xf32>
  %2 = addf %0, %1 : tensor<4xf32>

  // CHECK-NEXT: return [[C]]
  return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @addf_dense_and_splat_tensors
func @addf_dense_and_splat_tensors() -> tensor<4xf32> {
  %0 = constant dense<[1.5, 2.5, 3.5, 4.5]> : tensor<4xf32>
  %1 = constant dense<1.5> : tensor<4xf32>

  // CHECK-NEXT: [[C:%.+]] = constant dense<[3.{{0*}}e+00, 4.{{0*}}e+00, 5.{{0*}}e+00, 6.{{0*}}e+00]> : tensor<4xf32>
  %2 = addf %0, %1 : tensor<4xf32>

  // CHECK-NEXT: return [[C]]
  return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @simple_addi
func @simple_addi() -> i32 {
  %0 = constant 1 : i32
  %1 = constant 5 : i32

  // CHECK-NEXT: [[C:%.+]] = constant 6 : i32
  %2 = addi %0, %1 : i32

  // CHECK-NEXT: return [[C]]
  return %2 : i32
}

// -----

// CHECK: func @simple_and
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]: i1
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]: i32)
func @simple_and(%arg0 : i1, %arg1 : i32) -> (i1, i32) {
  %c1 = constant 1 : i1
  %cAllOnes_32 = constant 4294967295 : i32

  // CHECK: [[C31:%.*]] = constant 31 : i32
  %c31 = constant 31 : i32
  %1 = and %arg0, %c1 : i1
  %2 = and %arg1, %cAllOnes_32 : i32

  // CHECK: [[VAL:%.*]] = and [[ARG1]], [[C31]]
  %3 = and %2, %c31 : i32

  // CHECK: return [[ARG0]], [[VAL]]
  return %1, %3 : i1, i32
}

// -----

// CHECK-LABEL: func @and_index
//  CHECK-SAME:   [[ARG:%[a-zA-Z0-9]+]]
func @and_index(%arg0 : index) -> (index) {
  // CHECK: [[C31:%.*]] = constant 31 : index
  %c31 = constant 31 : index
  %c_AllOnes = constant -1 : index
  %1 = and %arg0, %c31 : index

  // CHECK: and [[ARG]], [[C31]]
  %2 = and %1, %c_AllOnes : index
  return %2 : index
}

// -----

// CHECK: func @tensor_and
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]: tensor<2xi32>
func @tensor_and(%arg0 : tensor<2xi32>) -> tensor<2xi32> {
  %cAllOnes_32 = constant dense<4294967295> : tensor<2xi32>

  // CHECK: [[C31:%.*]] = constant dense<31> : tensor<2xi32>
  %c31 = constant dense<31> : tensor<2xi32>

  // CHECK: [[CMIXED:%.*]] = constant dense<[31, -1]> : tensor<2xi32>
  %c_mixed = constant dense<[31, 4294967295]> : tensor<2xi32>

  %0 = and %arg0, %cAllOnes_32 : tensor<2xi32>

  // CHECK: [[T1:%.*]] = and [[ARG0]], [[C31]]
  %1 = and %0, %c31 : tensor<2xi32>

  // CHECK: [[T2:%.*]] = and [[T1]], [[CMIXED]]
  %2 = and %1, %c_mixed : tensor<2xi32>

  // CHECK: return [[T2]]
  return %2 : tensor<2xi32>
}

// -----

// CHECK: func @vector_and
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]: vector<2xi32>
func @vector_and(%arg0 : vector<2xi32>) -> vector<2xi32> {
  %cAllOnes_32 = constant dense<4294967295> : vector<2xi32>

  // CHECK: [[C31:%.*]] = constant dense<31> : vector<2xi32>
  %c31 = constant dense<31> : vector<2xi32>

  // CHECK: [[CMIXED:%.*]] = constant dense<[31, -1]> : vector<2xi32>
  %c_mixed = constant dense<[31, 4294967295]> : vector<2xi32>

  %0 = and %arg0, %cAllOnes_32 : vector<2xi32>

  // CHECK: [[T1:%.*]] = and [[ARG0]], [[C31]]
  %1 = and %0, %c31 : vector<2xi32>

  // CHECK: [[T2:%.*]] = and [[T1]], [[CMIXED]]
  %2 = and %1, %c_mixed : vector<2xi32>

  // CHECK: return [[T2]]
  return %2 : vector<2xi32>
}

// -----

// CHECK-LABEL: func @addi_splat_vector
func @addi_splat_vector() -> vector<8xi32> {
  %0 = constant dense<1> : vector<8xi32>
  %1 = constant dense<5> : vector<8xi32>

  // CHECK-NEXT: [[C:%.+]] = constant dense<6> : vector<8xi32>
  %2 = addi %0, %1 : vector<8xi32>

  // CHECK-NEXT: return [[C]]
  return %2 : vector<8xi32>
}

// -----

// CHECK-LABEL: func @simple_subf
func @simple_subf() -> f32 {
  %0 = constant 4.5 : f32
  %1 = constant 1.5 : f32

  // CHECK-NEXT: [[C:%.+]] = constant 3.{{0*}}e+00 : f32
  %2 = subf %0, %1 : f32

  // CHECK-NEXT: return [[C]]
  return %2 : f32
}

// -----

// CHECK-LABEL: func @subf_splat_vector
func @subf_splat_vector() -> vector<4xf32> {
  %0 = constant dense<4.5> : vector<4xf32>
  %1 = constant dense<1.5> : vector<4xf32>

  // CHECK-NEXT: [[C:%.+]] = constant dense<3.{{0*}}e+00> : vector<4xf32>
  %2 = subf %0, %1 : vector<4xf32>

  // CHECK-NEXT: return [[C]]
  return %2 : vector<4xf32>
}

// -----

//      CHECK: func @simple_subi
// CHECK-SAME:   [[ARG0:%[a-zA-Z0-9]+]]
func @simple_subi(%arg0 : i32) -> (i32, i32) {
  %0 = constant 4 : i32
  %1 = constant 1 : i32
  %2 = constant 0 : i32

  // CHECK-NEXT:[[C3:%.+]] = constant 3 : i32
  %3 = subi %0, %1 : i32
  %4 = subi %arg0, %2 : i32

  // CHECK-NEXT: return [[C3]], [[ARG0]]
  return %3, %4 : i32, i32
}

// -----

// CHECK-LABEL: func @subi_splat_tensor
func @subi_splat_tensor() -> tensor<4xi32> {
  %0 = constant dense<4> : tensor<4xi32>
  %1 = constant dense<1> : tensor<4xi32>

  // CHECK-NEXT: [[C:%.+]] = constant dense<3> : tensor<4xi32>
  %2 = subi %0, %1 : tensor<4xi32>

  // CHECK-NEXT: return [[C]]
  return %2 : tensor<4xi32>
}

// -----

// CHECK-LABEL: func @affine_apply
func @affine_apply(%variable : index) -> (index, index, index) {
  %c177 = constant 177 : index
  %c211 = constant 211 : index
  %N = constant 1075 : index

  // CHECK:[[C1159:%.+]] = constant 1159 : index
  // CHECK:[[C1152:%.+]] = constant 1152 : index
  %x0 = affine.apply affine_map<(d0, d1)[S0] -> ( (d0 + 128 * S0) floordiv 128 + d1 mod 128)>
           (%c177, %c211)[%N]
  %x1 = affine.apply affine_map<(d0, d1)[S0] -> (128 * (S0 ceildiv 128))>
           (%c177, %c211)[%N]

  // CHECK:[[C42:%.+]] = constant 42 : index
  %y = affine.apply affine_map<(d0) -> (42)> (%variable)

  // CHECK: return [[C1159]], [[C1152]], [[C42]]
  return %x0, %x1, %y : index, index, index
}

// -----

// CHECK-LABEL: func @simple_mulf
func @simple_mulf() -> f32 {
  %0 = constant 4.5 : f32
  %1 = constant 1.5 : f32

  // CHECK-NEXT: [[C:%.+]] = constant 6.75{{0*}}e+00 : f32
  %2 = mulf %0, %1 : f32

  // CHECK-NEXT: return [[C]]
  return %2 : f32
}

// -----

// CHECK-LABEL: func @mulf_splat_tensor
func @mulf_splat_tensor() -> tensor<4xf32> {
  %0 = constant dense<4.5> : tensor<4xf32>
  %1 = constant dense<1.5> : tensor<4xf32>

  // CHECK-NEXT: [[C:%.+]] = constant dense<6.75{{0*}}e+00> : tensor<4xf32>
  %2 = mulf %0, %1 : tensor<4xf32>

  // CHECK-NEXT: return [[C]]
  return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @simple_divi_signed
func @simple_divi_signed() -> (i32, i32, i32) {
  // CHECK-DAG: [[C0:%.+]] = constant 0
  %z = constant 0 : i32
  // CHECK-DAG: [[C6:%.+]] = constant 6
  %0 = constant 6 : i32
  %1 = constant 2 : i32

  // CHECK-NEXT: [[C3:%.+]] = constant 3 : i32
  %2 = divi_signed %0, %1 : i32

  %3 = constant -2 : i32

  // CHECK-NEXT: [[CM3:%.+]] = constant -3 : i32
  %4 = divi_signed %0, %3 : i32

  // CHECK-NEXT: [[XZ:%.+]] = divi_signed [[C6]], [[C0]]
  %5 = divi_signed %0, %z : i32

  // CHECK-NEXT: return [[C3]], [[CM3]], [[XZ]]
  return %2, %4, %5 : i32, i32, i32
}

// -----

// CHECK-LABEL: func @divi_signed_splat_tensor
func @divi_signed_splat_tensor() -> (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
  // CHECK-DAG: [[C0:%.+]] = constant dense<0>
  %z = constant dense<0> : tensor<4xi32>
  // CHECK-DAG: [[C6:%.+]] = constant dense<6>
  %0 = constant dense<6> : tensor<4xi32>
  %1 = constant dense<2> : tensor<4xi32>

  // CHECK-NEXT: [[C3:%.+]] = constant dense<3> : tensor<4xi32>
  %2 = divi_signed %0, %1 : tensor<4xi32>

  %3 = constant dense<-2> : tensor<4xi32>

  // CHECK-NEXT: [[CM3:%.+]] = constant dense<-3> : tensor<4xi32>
  %4 = divi_signed %0, %3 : tensor<4xi32>

  // CHECK-NEXT: [[XZ:%.+]] = divi_signed [[C6]], [[C0]]
  %5 = divi_signed %0, %z : tensor<4xi32>

  // CHECK-NEXT: return [[C3]], [[CM3]], [[XZ]]
  return %2, %4, %5 : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
}

// -----

// CHECK-LABEL: func @simple_divi_unsigned
func @simple_divi_unsigned() -> (i32, i32, i32) {
  %z = constant 0 : i32
  // CHECK-DAG: [[C6:%.+]] = constant 6
  %0 = constant 6 : i32
  %1 = constant 2 : i32

  // CHECK-DAG: [[C3:%.+]] = constant 3 : i32
  %2 = divi_unsigned %0, %1 : i32

  %3 = constant -2 : i32

  // Unsigned division interprets -2 as 2^32-2, so the result is 0.
  // CHECK-DAG: [[C0:%.+]] = constant 0 : i32
  %4 = divi_unsigned %0, %3 : i32

  // CHECK-NEXT: [[XZ:%.+]] = divi_unsigned [[C6]], [[C0]]
  %5 = divi_unsigned %0, %z : i32

  // CHECK-NEXT: return [[C3]], [[C0]], [[XZ]]
  return %2, %4, %5 : i32, i32, i32
}


// -----

// CHECK-LABEL: func @divi_unsigned_splat_tensor
func @divi_unsigned_splat_tensor() -> (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
  %z = constant dense<0> : tensor<4xi32>
  // CHECK-DAG: [[C6:%.+]] = constant dense<6>
  %0 = constant dense<6> : tensor<4xi32>
  %1 = constant dense<2> : tensor<4xi32>

  // CHECK-DAG: [[C3:%.+]] = constant dense<3> : tensor<4xi32>
  %2 = divi_unsigned %0, %1 : tensor<4xi32>

  %3 = constant dense<-2> : tensor<4xi32>

  // Unsigned division interprets -2 as 2^32-2, so the result is 0.
  // CHECK-DAG: [[C0:%.+]] = constant dense<0> : tensor<4xi32>
  %4 = divi_unsigned %0, %3 : tensor<4xi32>

  // CHECK-NEXT: [[XZ:%.+]] = divi_unsigned [[C6]], [[C0]]
  %5 = divi_unsigned %0, %z : tensor<4xi32>

  // CHECK-NEXT: return [[C3]], [[C0]], [[XZ]]
  return %2, %4, %5 : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
}

// -----

// CHECK-LABEL: func @simple_remi_signed
func @simple_remi_signed(%a : i32) -> (i32, i32, i32) {
  %0 = constant 5 : i32
  %1 = constant 2 : i32
  %2 = constant 1 : i32
  %3 = constant -2 : i32

  // CHECK-NEXT:[[C1:%.+]] = constant 1 : i32
  %4 = remi_signed %0, %1 : i32
  %5 = remi_signed %0, %3 : i32
  // CHECK-NEXT:[[C0:%.+]] = constant 0 : i32
  %6 = remi_signed %a, %2 : i32

  // CHECK-NEXT: return [[C1]], [[C1]], [[C0]] : i32, i32, i32
  return %4, %5, %6 : i32, i32, i32
}

// -----

// CHECK-LABEL: func @simple_remi_unsigned
func @simple_remi_unsigned(%a : i32) -> (i32, i32, i32) {
  %0 = constant 5 : i32
  %1 = constant 2 : i32
  %2 = constant 1 : i32
  %3 = constant -2 : i32

  // CHECK-DAG:[[C1:%.+]] = constant 1 : i32
  %4 = remi_unsigned %0, %1 : i32
  // CHECK-DAG:[[C5:%.+]] = constant 5 : i32
  %5 = remi_unsigned %0, %3 : i32
  // CHECK-DAG:[[C0:%.+]] = constant 0 : i32
  %6 = remi_unsigned %a, %2 : i32

  // CHECK-NEXT: return [[C1]], [[C5]], [[C0]] : i32, i32, i32
  return %4, %5, %6 : i32, i32, i32
}

// -----

// CHECK-LABEL: func @muli
func @muli() -> i32 {
  %0 = constant 4 : i32
  %1 = constant 2 : i32

  // CHECK-NEXT:[[C8:%.+]] = constant 8 : i32
  %2 = muli %0, %1 : i32

  // CHECK-NEXT: return [[C8]]
  return %2 : i32
}

// -----

// CHECK-LABEL: func @muli_splat_vector
func @muli_splat_vector() -> vector<4xi32> {
  %0 = constant dense<4> : vector<4xi32>
  %1 = constant dense<2> : vector<4xi32>

  // CHECK-NEXT: [[C:%.+]] = constant dense<8> : vector<4xi32>
  %2 = muli %0, %1 : vector<4xi32>

  // CHECK-NEXT: return [[C]]
  return %2 : vector<4xi32>
}

// CHECK-LABEL: func @dim
func @dim(%x : tensor<8x4xf32>) -> index {

  // CHECK:[[C4:%.+]] = constant 4 : index
  %c1 = constant 1 : index
  %0 = dim %x, %c1 : tensor<8x4xf32>

  // CHECK-NEXT: return [[C4]]
  return %0 : index
}

// -----

// CHECK-LABEL: func @cmpi
func @cmpi() -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) {
  %c42 = constant 42 : i32
  %cm1 = constant -1 : i32
  // CHECK-DAG: [[F:%.+]] = constant false
  // CHECK-DAG: [[T:%.+]] = constant true
  // CHECK-NEXT: return [[F]],
  %0 = cmpi "eq", %c42, %cm1 : i32
  // CHECK-SAME: [[T]],
  %1 = cmpi "ne", %c42, %cm1 : i32
  // CHECK-SAME: [[F]],
  %2 = cmpi "slt", %c42, %cm1 : i32
  // CHECK-SAME: [[F]],
  %3 = cmpi "sle", %c42, %cm1 : i32
  // CHECK-SAME: [[T]],
  %4 = cmpi "sgt", %c42, %cm1 : i32
  // CHECK-SAME: [[T]],
  %5 = cmpi "sge", %c42, %cm1 : i32
  // CHECK-SAME: [[T]],
  %6 = cmpi "ult", %c42, %cm1 : i32
  // CHECK-SAME: [[T]],
  %7 = cmpi "ule", %c42, %cm1 : i32
  // CHECK-SAME: [[F]],
  %8 = cmpi "ugt", %c42, %cm1 : i32
  // CHECK-SAME: [[F]]
  %9 = cmpi "uge", %c42, %cm1 : i32
  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// -----

// CHECK-LABEL: func @cmpf_normal_numbers
func @cmpf_normal_numbers() -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) {
  %c42 = constant 42. : f32
  %cm1 = constant -1. : f32
  // CHECK-DAG: [[F:%.+]] = constant false
  // CHECK-DAG: [[T:%.+]] = constant true
  // CHECK-NEXT: return [[F]],
  %0 = cmpf "false", %c42, %cm1 : f32
  // CHECK-SAME: [[F]],
  %1 = cmpf "oeq", %c42, %cm1 : f32
  // CHECK-SAME: [[T]],
  %2 = cmpf "ogt", %c42, %cm1 : f32
  // CHECK-SAME: [[T]],
  %3 = cmpf "oge", %c42, %cm1 : f32
  // CHECK-SAME: [[F]],
  %4 = cmpf "olt", %c42, %cm1 : f32
  // CHECK-SAME: [[F]],
  %5 = cmpf "ole", %c42, %cm1 : f32
  // CHECK-SAME: [[T]],
  %6 = cmpf "one", %c42, %cm1 : f32
  // CHECK-SAME: [[T]],
  %7 = cmpf "ord", %c42, %cm1 : f32
  // CHECK-SAME: [[F]],
  %8 = cmpf "ueq", %c42, %cm1 : f32
  // CHECK-SAME: [[T]],
  %9 = cmpf "ugt", %c42, %cm1 : f32
  // CHECK-SAME: [[T]],
  %10 = cmpf "uge", %c42, %cm1 : f32
  // CHECK-SAME: [[F]],
  %11 = cmpf "ult", %c42, %cm1 : f32
  // CHECK-SAME: [[F]],
  %12 = cmpf "ule", %c42, %cm1 : f32
  // CHECK-SAME: [[T]],
  %13 = cmpf "une", %c42, %cm1 : f32
  // CHECK-SAME: [[F]],
  %14 = cmpf "uno", %c42, %cm1 : f32
  // CHECK-SAME: [[T]]
  %15 = cmpf "true", %c42, %cm1 : f32
  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// -----

// CHECK-LABEL: func @cmpf_nan
func @cmpf_nan() -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) {
  %c42 = constant 42. : f32
  %cqnan = constant 0xFFFFFFFF : f32
  // CHECK-DAG: [[F:%.+]] = constant false
  // CHECK-DAG: [[T:%.+]] = constant true
  // CHECK-NEXT: return [[F]],
  %0 = cmpf "false", %c42, %cqnan : f32
  // CHECK-SAME: [[F]]
  %1 = cmpf "oeq", %c42, %cqnan : f32
  // CHECK-SAME: [[F]],
  %2 = cmpf "ogt", %c42, %cqnan : f32
  // CHECK-SAME: [[F]],
  %3 = cmpf "oge", %c42, %cqnan : f32
  // CHECK-SAME: [[F]],
  %4 = cmpf "olt", %c42, %cqnan : f32
  // CHECK-SAME: [[F]],
  %5 = cmpf "ole", %c42, %cqnan : f32
  // CHECK-SAME: [[F]],
  %6 = cmpf "one", %c42, %cqnan : f32
  // CHECK-SAME: [[F]],
  %7 = cmpf "ord", %c42, %cqnan : f32
  // CHECK-SAME: [[T]],
  %8 = cmpf "ueq", %c42, %cqnan : f32
  // CHECK-SAME: [[T]],
  %9 = cmpf "ugt", %c42, %cqnan : f32
  // CHECK-SAME: [[T]],
  %10 = cmpf "uge", %c42, %cqnan : f32
  // CHECK-SAME: [[T]],
  %11 = cmpf "ult", %c42, %cqnan : f32
  // CHECK-SAME: [[T]],
  %12 = cmpf "ule", %c42, %cqnan : f32
  // CHECK-SAME: [[T]],
  %13 = cmpf "une", %c42, %cqnan : f32
  // CHECK-SAME: [[T]],
  %14 = cmpf "uno", %c42, %cqnan : f32
  // CHECK-SAME: [[T]]
  %15 = cmpf "true", %c42, %cqnan : f32
  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// -----

// CHECK-LABEL: func @cmpf_inf
func @cmpf_inf() -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) {
  %c42 = constant 42. : f32
  %cpinf = constant 0x7F800000 : f32
  // CHECK-DAG: [[F:%.+]] = constant false
  // CHECK-DAG: [[T:%.+]] = constant true
  // CHECK-NEXT: return [[F]],
  %0 = cmpf "false", %c42, %cpinf: f32
  // CHECK-SAME: [[F]]
  %1 = cmpf "oeq", %c42, %cpinf: f32
  // CHECK-SAME: [[F]],
  %2 = cmpf "ogt", %c42, %cpinf: f32
  // CHECK-SAME: [[F]],
  %3 = cmpf "oge", %c42, %cpinf: f32
  // CHECK-SAME: [[T]],
  %4 = cmpf "olt", %c42, %cpinf: f32
  // CHECK-SAME: [[T]],
  %5 = cmpf "ole", %c42, %cpinf: f32
  // CHECK-SAME: [[T]],
  %6 = cmpf "one", %c42, %cpinf: f32
  // CHECK-SAME: [[T]],
  %7 = cmpf "ord", %c42, %cpinf: f32
  // CHECK-SAME: [[F]],
  %8 = cmpf "ueq", %c42, %cpinf: f32
  // CHECK-SAME: [[F]],
  %9 = cmpf "ugt", %c42, %cpinf: f32
  // CHECK-SAME: [[F]],
  %10 = cmpf "uge", %c42, %cpinf: f32
  // CHECK-SAME: [[T]],
  %11 = cmpf "ult", %c42, %cpinf: f32
  // CHECK-SAME: [[T]],
  %12 = cmpf "ule", %c42, %cpinf: f32
  // CHECK-SAME: [[T]],
  %13 = cmpf "une", %c42, %cpinf: f32
  // CHECK-SAME: [[F]],
  %14 = cmpf "uno", %c42, %cpinf: f32
  // CHECK-SAME: [[T]]
  %15 = cmpf "true", %c42, %cpinf: f32
  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// -----

// CHECK-LABEL: func @fold_extract_element
func @fold_extract_element(%arg0 : index) -> (f32, f16, f16, i32) {
  %const_0 = constant 0 : index
  %const_1 = constant 1 : index
  %const_3 = constant 3 : index

  // Fold an extract into a splat.
  // CHECK-NEXT: [[C4:%.+]] = constant 4.{{0*}}e+00 : f32
  %0 = constant dense<4.0> : tensor<4xf32>
  %ext_1 = extract_element %0[%arg0] : tensor<4xf32>

  // Fold an extract into a sparse with a sparse index.
  // CHECK-NEXT: [[CM2:%.+]] = constant -2.{{0*}}e+00 : f16
  %1 = constant sparse<[[0, 0, 0], [1, 1, 1]],  [-5.0, -2.0]> : vector<4x4x4xf16>
  %ext_2 = extract_element %1[%const_1, %const_1, %const_1] : vector<4x4x4xf16>

  // Fold an extract into a sparse with a non sparse index.
  // CHECK-NEXT: [[C0:%.+]] = constant 0.{{0*}}e+00 : f16
  %2 = constant sparse<[[1, 1, 1]],  [-2.0]> : vector<1x1x1xf16>
  %ext_3 = extract_element %2[%const_0, %const_0, %const_0] : vector<1x1x1xf16>

  // Fold an extract into a dense tensor.
  // CHECK-NEXT: [[C64:%.+]] = constant 64 : i32
  %3 = constant dense<[[[1, -2, 1, 36]], [[0, 2, -1, 64]]]> : tensor<2x1x4xi32>
  %ext_4 = extract_element %3[%const_1, %const_0, %const_3] : tensor<2x1x4xi32>

  // CHECK-NEXT: return [[C4]], [[CM2]], [[C0]], [[C64]]
  return %ext_1, %ext_2, %ext_3, %ext_4 : f32, f16, f16, i32
}

// -----

// CHECK-LABEL: func @fold_rank
func @fold_rank() -> (index) {
  %const_0 = constant dense<[[[1, -2, 1, 36]], [[0, 2, -1, 64]]]> : tensor<2x1x4xi32>

  // Fold a rank into a constant
  // CHECK-NEXT: [[C3:%.+]] = constant 3 : index
  %rank_0 = rank %const_0 : tensor<2x1x4xi32>

  // CHECK-NEXT: return [[C3]]
  return %rank_0 : index
}

// -----

// CHECK-LABEL: func @fold_rank_memref
func @fold_rank_memref(%arg0 : memref<?x?xf32>) -> (index) {
  // Fold a rank into a constant
  // CHECK-NEXT: [[C2:%.+]] = constant 2 : index
  %rank_0 = rank %arg0 : memref<?x?xf32>

  // CHECK-NEXT: return [[C2]]
  return %rank_0 : index
}

// -----

// CHECK-LABEL: func @nested_isolated_region
func @nested_isolated_region() {
  // CHECK-NEXT: func @isolated_op
  // CHECK-NEXT: constant 2
  func @isolated_op() {
    %0 = constant 1 : i32
    %2 = addi %0, %0 : i32
    "foo.yield"(%2) : (i32) -> ()
  }

  // CHECK: "foo.unknown_region"
  // CHECK-NEXT: constant 2
  "foo.unknown_region"() ({
    %0 = constant 1 : i32
    %2 = addi %0, %0 : i32
    "foo.yield"(%2) : (i32) -> ()
  }) : () -> ()
  return
}

// -----

// CHECK-LABEL: func @custom_insertion_position
func @custom_insertion_position() {
  // CHECK: test.one_region_op
  // CHECK-NEXT: constant 2
  "test.one_region_op"() ({

    %0 = constant 1 : i32
    %2 = addi %0, %0 : i32
    "foo.yield"(%2) : (i32) -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func @splat_fold
func @splat_fold() -> (vector<4xf32>, tensor<4xf32>) {
  %c = constant 1.0 : f32
  %v = splat %c : vector<4xf32>
  %t = splat %c : tensor<4xf32>
  return %v, %t : vector<4xf32>, tensor<4xf32>

  // CHECK-NEXT: [[V:%.*]] = constant dense<1.000000e+00> : vector<4xf32>
  // CHECK-NEXT: [[T:%.*]] = constant dense<1.000000e+00> : tensor<4xf32>
  // CHECK-NEXT: return [[V]], [[T]] : vector<4xf32>, tensor<4xf32>
}
