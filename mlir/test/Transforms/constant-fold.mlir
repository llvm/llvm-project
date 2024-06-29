// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -test-constant-fold | FileCheck %s

// -----

// CHECK-LABEL: @affine_for
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @affine_for(%p : memref<f32>) {
  // CHECK: [[C:%.+]] = arith.constant 6.{{0*}}e+00 : f32
  affine.for %arg1 = 0 to 128 {
    affine.for %arg2 = 0 to 8 { // CHECK: affine.for %{{.*}} = 0 to 8 {
      %0 = arith.constant 4.5 : f32
      %1 = arith.constant 1.5 : f32

      %2 = arith.addf %0, %1 : f32

      // CHECK-NEXT: memref.store [[C]], [[ARG]][]
      memref.store %2, %p[] : memref<f32>
    }
  }
  return
}

// -----

// CHECK-LABEL: func @simple_addf
func.func @simple_addf() -> f32 {
  %0 = arith.constant 4.5 : f32
  %1 = arith.constant 1.5 : f32

  // CHECK-NEXT: [[C:%.+]] = arith.constant 6.{{0*}}e+00 : f32
  %2 = arith.addf %0, %1 : f32

  // CHECK-NEXT: return [[C]]
  return %2 : f32
}

// -----

// CHECK-LABEL: func @addf_splat_tensor
func.func @addf_splat_tensor() -> tensor<4xf32> {
  %0 = arith.constant dense<4.5> : tensor<4xf32>
  %1 = arith.constant dense<1.5> : tensor<4xf32>

  // CHECK-NEXT: [[C:%.+]] = arith.constant dense<6.{{0*}}e+00> : tensor<4xf32>
  %2 = arith.addf %0, %1 : tensor<4xf32>

  // CHECK-NEXT: return [[C]]
  return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @addf_dense_tensor
func.func @addf_dense_tensor() -> tensor<4xf32> {
  %0 = arith.constant dense<[1.5, 2.5, 3.5, 4.5]> : tensor<4xf32>
  %1 = arith.constant dense<[1.5, 2.5, 3.5, 4.5]> : tensor<4xf32>

  // CHECK-NEXT: [[C:%.+]] = arith.constant dense<[3.{{0*}}e+00, 5.{{0*}}e+00, 7.{{0*}}e+00, 9.{{0*}}e+00]> : tensor<4xf32>
  %2 = arith.addf %0, %1 : tensor<4xf32>

  // CHECK-NEXT: return [[C]]
  return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @addf_dense_and_splat_tensors
func.func @addf_dense_and_splat_tensors() -> tensor<4xf32> {
  %0 = arith.constant dense<[1.5, 2.5, 3.5, 4.5]> : tensor<4xf32>
  %1 = arith.constant dense<1.5> : tensor<4xf32>

  // CHECK-NEXT: [[C:%.+]] = arith.constant dense<[3.{{0*}}e+00, 4.{{0*}}e+00, 5.{{0*}}e+00, 6.{{0*}}e+00]> : tensor<4xf32>
  %2 = arith.addf %0, %1 : tensor<4xf32>

  // CHECK-NEXT: return [[C]]
  return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @simple_addi
func.func @simple_addi() -> i32 {
  %0 = arith.constant 1 : i32
  %1 = arith.constant 5 : i32

  // CHECK-NEXT: [[C:%.+]] = arith.constant 6 : i32
  %2 = arith.addi %0, %1 : i32

  // CHECK-NEXT: return [[C]]
  return %2 : i32
}

// -----

// CHECK: func @simple_and
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]: i1
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9]+]]: i32)
func.func @simple_and(%arg0 : i1, %arg1 : i32) -> (i1, i32) {
  %c1 = arith.constant 1 : i1
  %cAllOnes_32 = arith.constant 4294967295 : i32

  // CHECK: [[C31:%.*]] = arith.constant 31 : i32
  %c31 = arith.constant 31 : i32
  %1 = arith.andi %arg0, %c1 : i1
  %2 = arith.andi %arg1, %cAllOnes_32 : i32

  // CHECK: [[VAL:%.*]] = arith.andi [[ARG1]], [[C31]]
  %3 = arith.andi %2, %c31 : i32

  // CHECK: return [[ARG0]], [[VAL]]
  return %1, %3 : i1, i32
}

// -----

// CHECK-LABEL: func @and_index
//  CHECK-SAME:   [[ARG:%[a-zA-Z0-9]+]]
func.func @and_index(%arg0 : index) -> (index) {
  // CHECK: [[C31:%.*]] = arith.constant 31 : index
  %c31 = arith.constant 31 : index
  %c_AllOnes = arith.constant -1 : index
  %1 = arith.andi %arg0, %c31 : index

  // CHECK: arith.andi [[ARG]], [[C31]]
  %2 = arith.andi %1, %c_AllOnes : index
  return %2 : index
}

// -----

// CHECK: func @tensor_and
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]: tensor<2xi32>
func.func @tensor_and(%arg0 : tensor<2xi32>) -> tensor<2xi32> {
  %cAllOnes_32 = arith.constant dense<4294967295> : tensor<2xi32>

  // CHECK: [[C31:%.*]] = arith.constant dense<31> : tensor<2xi32>
  %c31 = arith.constant dense<31> : tensor<2xi32>

  // CHECK: [[CMIXED:%.*]] = arith.constant dense<[31, -1]> : tensor<2xi32>
  %c_mixed = arith.constant dense<[31, 4294967295]> : tensor<2xi32>

  %0 = arith.andi %arg0, %cAllOnes_32 : tensor<2xi32>

  // CHECK: [[T1:%.*]] = arith.andi [[ARG0]], [[C31]]
  %1 = arith.andi %0, %c31 : tensor<2xi32>

  // CHECK: [[T2:%.*]] = arith.andi [[T1]], [[CMIXED]]
  %2 = arith.andi %1, %c_mixed : tensor<2xi32>

  // CHECK: return [[T2]]
  return %2 : tensor<2xi32>
}

// -----

// CHECK: func @vector_and
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]+]]: vector<2xi32>
func.func @vector_and(%arg0 : vector<2xi32>) -> vector<2xi32> {
  %cAllOnes_32 = arith.constant dense<4294967295> : vector<2xi32>

  // CHECK: [[C31:%.*]] = arith.constant dense<31> : vector<2xi32>
  %c31 = arith.constant dense<31> : vector<2xi32>

  // CHECK: [[CMIXED:%.*]] = arith.constant dense<[31, -1]> : vector<2xi32>
  %c_mixed = arith.constant dense<[31, 4294967295]> : vector<2xi32>

  %0 = arith.andi %arg0, %cAllOnes_32 : vector<2xi32>

  // CHECK: [[T1:%.*]] = arith.andi [[ARG0]], [[C31]]
  %1 = arith.andi %0, %c31 : vector<2xi32>

  // CHECK: [[T2:%.*]] = arith.andi [[T1]], [[CMIXED]]
  %2 = arith.andi %1, %c_mixed : vector<2xi32>

  // CHECK: return [[T2]]
  return %2 : vector<2xi32>
}

// -----

// CHECK-LABEL: func @addi_splat_vector
func.func @addi_splat_vector() -> vector<8xi32> {
  %0 = arith.constant dense<1> : vector<8xi32>
  %1 = arith.constant dense<5> : vector<8xi32>

  // CHECK-NEXT: [[C:%.+]] = arith.constant dense<6> : vector<8xi32>
  %2 = arith.addi %0, %1 : vector<8xi32>

  // CHECK-NEXT: return [[C]]
  return %2 : vector<8xi32>
}

// -----

// CHECK-LABEL: func @simple_subf
func.func @simple_subf() -> f32 {
  %0 = arith.constant 4.5 : f32
  %1 = arith.constant 1.5 : f32

  // CHECK-NEXT: [[C:%.+]] = arith.constant 3.{{0*}}e+00 : f32
  %2 = arith.subf %0, %1 : f32

  // CHECK-NEXT: return [[C]]
  return %2 : f32
}

// -----

// CHECK-LABEL: func @subf_splat_vector
func.func @subf_splat_vector() -> vector<4xf32> {
  %0 = arith.constant dense<4.5> : vector<4xf32>
  %1 = arith.constant dense<1.5> : vector<4xf32>

  // CHECK-NEXT: [[C:%.+]] = arith.constant dense<3.{{0*}}e+00> : vector<4xf32>
  %2 = arith.subf %0, %1 : vector<4xf32>

  // CHECK-NEXT: return [[C]]
  return %2 : vector<4xf32>
}

// -----

//      CHECK: func @simple_subi
// CHECK-SAME:   [[ARG0:%[a-zA-Z0-9]+]]
func.func @simple_subi(%arg0 : i32) -> (i32, i32) {
  %0 = arith.constant 4 : i32
  %1 = arith.constant 1 : i32
  %2 = arith.constant 0 : i32

  // CHECK-NEXT:[[C3:%.+]] = arith.constant 3 : i32
  %3 = arith.subi %0, %1 : i32
  %4 = arith.subi %arg0, %2 : i32

  // CHECK-NEXT: return [[C3]], [[ARG0]]
  return %3, %4 : i32, i32
}

// -----

// CHECK-LABEL: func @subi_splat_tensor
func.func @subi_splat_tensor() -> tensor<4xi32> {
  %0 = arith.constant dense<4> : tensor<4xi32>
  %1 = arith.constant dense<1> : tensor<4xi32>

  // CHECK-NEXT: [[C:%.+]] = arith.constant dense<3> : tensor<4xi32>
  %2 = arith.subi %0, %1 : tensor<4xi32>

  // CHECK-NEXT: return [[C]]
  return %2 : tensor<4xi32>
}

// -----

// CHECK-LABEL: func @affine_apply
func.func @affine_apply(%variable : index) -> (index, index, index) {
  %c177 = arith.constant 177 : index
  %c211 = arith.constant 211 : index
  %N = arith.constant 1075 : index

  // CHECK:[[C1159:%.+]] = arith.constant 1159 : index
  // CHECK:[[C1152:%.+]] = arith.constant 1152 : index
  %x0 = affine.apply affine_map<(d0, d1)[S0] -> ( (d0 + 128 * S0) floordiv 128 + d1 mod 128)>
           (%c177, %c211)[%N]
  %x1 = affine.apply affine_map<(d0, d1)[S0] -> (128 * (S0 ceildiv 128))>
           (%c177, %c211)[%N]

  // CHECK:[[C42:%.+]] = arith.constant 42 : index
  %y = affine.apply affine_map<(d0) -> (42)> (%variable)

  // CHECK: return [[C1159]], [[C1152]], [[C42]]
  return %x0, %x1, %y : index, index, index
}

// -----

// CHECK-LABEL: func @simple_mulf
func.func @simple_mulf() -> f32 {
  %0 = arith.constant 4.5 : f32
  %1 = arith.constant 1.5 : f32

  // CHECK-NEXT: [[C:%.+]] = arith.constant 6.75{{0*}}e+00 : f32
  %2 = arith.mulf %0, %1 : f32

  // CHECK-NEXT: return [[C]]
  return %2 : f32
}

// -----

// CHECK-LABEL: func @mulf_splat_tensor
func.func @mulf_splat_tensor() -> tensor<4xf32> {
  %0 = arith.constant dense<4.5> : tensor<4xf32>
  %1 = arith.constant dense<1.5> : tensor<4xf32>

  // CHECK-NEXT: [[C:%.+]] = arith.constant dense<6.75{{0*}}e+00> : tensor<4xf32>
  %2 = arith.mulf %0, %1 : tensor<4xf32>

  // CHECK-NEXT: return [[C]]
  return %2 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @simple_divi_signed
func.func @simple_divi_signed() -> (i32, i32, i32) {
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0
  %z = arith.constant 0 : i32
  // CHECK-DAG: [[C6:%.+]] = arith.constant 6
  %0 = arith.constant 6 : i32
  %1 = arith.constant 2 : i32

  // CHECK-NEXT: [[C3:%.+]] = arith.constant 3 : i32
  %2 = arith.divsi %0, %1 : i32

  %3 = arith.constant -2 : i32

  // CHECK-NEXT: [[CM3:%.+]] = arith.constant -3 : i32
  %4 = arith.divsi %0, %3 : i32

  // CHECK-NEXT: [[XZ:%.+]] = arith.divsi [[C6]], [[C0]]
  %5 = arith.divsi %0, %z : i32

  // CHECK-NEXT: return [[C3]], [[CM3]], [[XZ]]
  return %2, %4, %5 : i32, i32, i32
}

// -----

// CHECK-LABEL: func @divi_signed_splat_tensor
func.func @divi_signed_splat_tensor() -> (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
  // CHECK-DAG: [[C0:%.+]] = arith.constant dense<0>
  %z = arith.constant dense<0> : tensor<4xi32>
  // CHECK-DAG: [[C6:%.+]] = arith.constant dense<6>
  %0 = arith.constant dense<6> : tensor<4xi32>
  %1 = arith.constant dense<2> : tensor<4xi32>

  // CHECK-NEXT: [[C3:%.+]] = arith.constant dense<3> : tensor<4xi32>
  %2 = arith.divsi %0, %1 : tensor<4xi32>

  %3 = arith.constant dense<-2> : tensor<4xi32>

  // CHECK-NEXT: [[CM3:%.+]] = arith.constant dense<-3> : tensor<4xi32>
  %4 = arith.divsi %0, %3 : tensor<4xi32>

  // CHECK-NEXT: [[XZ:%.+]] = arith.divsi [[C6]], [[C0]]
  %5 = arith.divsi %0, %z : tensor<4xi32>

  // CHECK-NEXT: return [[C3]], [[CM3]], [[XZ]]
  return %2, %4, %5 : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
}

// -----

// CHECK-LABEL: func @simple_divi_unsigned
func.func @simple_divi_unsigned() -> (i32, i32, i32) {
  %z = arith.constant 0 : i32
  // CHECK-DAG: [[C6:%.+]] = arith.constant 6
  %0 = arith.constant 6 : i32
  %1 = arith.constant 2 : i32

  // CHECK-DAG: [[C3:%.+]] = arith.constant 3 : i32
  %2 = arith.divui %0, %1 : i32

  %3 = arith.constant -2 : i32

  // Unsigned division interprets -2 as 2^32-2, so the result is 0.
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : i32
  %4 = arith.divui %0, %3 : i32

  // CHECK-NEXT: [[XZ:%.+]] = arith.divui [[C6]], [[C0]]
  %5 = arith.divui %0, %z : i32

  // CHECK-NEXT: return [[C3]], [[C0]], [[XZ]]
  return %2, %4, %5 : i32, i32, i32
}


// -----

// CHECK-LABEL: func @divi_unsigned_splat_tensor
func.func @divi_unsigned_splat_tensor() -> (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
  %z = arith.constant dense<0> : tensor<4xi32>
  // CHECK-DAG: [[C6:%.+]] = arith.constant dense<6>
  %0 = arith.constant dense<6> : tensor<4xi32>
  %1 = arith.constant dense<2> : tensor<4xi32>

  // CHECK-DAG: [[C3:%.+]] = arith.constant dense<3> : tensor<4xi32>
  %2 = arith.divui %0, %1 : tensor<4xi32>

  %3 = arith.constant dense<-2> : tensor<4xi32>

  // Unsigned division interprets -2 as 2^32-2, so the result is 0.
  // CHECK-DAG: [[C0:%.+]] = arith.constant dense<0> : tensor<4xi32>
  %4 = arith.divui %0, %3 : tensor<4xi32>

  // CHECK-NEXT: [[XZ:%.+]] = arith.divui [[C6]], [[C0]]
  %5 = arith.divui %0, %z : tensor<4xi32>

  // CHECK-NEXT: return [[C3]], [[C0]], [[XZ]]
  return %2, %4, %5 : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
}

// -----

// CHECK-LABEL: func @simple_arith.floordivsi
func.func @simple_arith.floordivsi() -> (i32, i32, i32, i32, i32) {
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0
  %z = arith.constant 0 : i32
  // CHECK-DAG: [[C6:%.+]] = arith.constant 7
  %0 = arith.constant 7 : i32
  %1 = arith.constant 2 : i32

  // floor(7, 2) = 3
  // CHECK-NEXT: [[C3:%.+]] = arith.constant 3 : i32
  %2 = arith.floordivsi %0, %1 : i32

  %3 = arith.constant -2 : i32

  // floor(7, -2) = -4
  // CHECK-NEXT: [[CM3:%.+]] = arith.constant -4 : i32
  %4 = arith.floordivsi %0, %3 : i32

  %5 = arith.constant -9 : i32

  // floor(-9, 2) = -5
  // CHECK-NEXT: [[CM4:%.+]] = arith.constant -5 : i32
  %6 = arith.floordivsi %5, %1 : i32

  %7 = arith.constant -13 : i32

  // floor(-13, -2) = 6
  // CHECK-NEXT: [[CM5:%.+]] = arith.constant 6 : i32
  %8 = arith.floordivsi %7, %3 : i32

  // CHECK-NEXT: [[XZ:%.+]] = arith.floordivsi [[C6]], [[C0]]
  %9 = arith.floordivsi %0, %z : i32

  return %2, %4, %6, %8, %9 : i32, i32, i32, i32, i32
}

// -----

// CHECK-LABEL: func @simple_arith.ceildivsi
func.func @simple_arith.ceildivsi() -> (i32, i32, i32, i32, i32) {
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0
  %z = arith.constant 0 : i32
  // CHECK-DAG: [[C6:%.+]] = arith.constant 7
  %0 = arith.constant 7 : i32
  %1 = arith.constant 2 : i32

  // ceil(7, 2) = 4
  // CHECK-NEXT: [[C3:%.+]] = arith.constant 4 : i32
  %2 = arith.ceildivsi %0, %1 : i32

  %3 = arith.constant -2 : i32

  // ceil(7, -2) = -3
  // CHECK-NEXT: [[CM3:%.+]] = arith.constant -3 : i32
  %4 = arith.ceildivsi %0, %3 : i32

  %5 = arith.constant -9 : i32

  // ceil(-9, 2) = -4
  // CHECK-NEXT: [[CM4:%.+]] = arith.constant -4 : i32
  %6 = arith.ceildivsi %5, %1 : i32

  %7 = arith.constant -15 : i32

  // ceil(-15, -2) = 8
  // CHECK-NEXT: [[CM5:%.+]] = arith.constant 8 : i32
  %8 = arith.ceildivsi %7, %3 : i32

  // CHECK-NEXT: [[XZ:%.+]] = arith.ceildivsi [[C6]], [[C0]]
  %9 = arith.ceildivsi %0, %z : i32

  return %2, %4, %6, %8, %9 : i32, i32, i32, i32, i32
}

// -----

// CHECK-LABEL: func @simple_arith.ceildivsi_overflow
func.func @simple_arith.ceildivsi_overflow() -> (i8, i16, i32) {
  // The negative values below are MININTs for the corresponding bit-width. The
  // folder will try to negate them (so that the division operates on two
  // positive numbers), but that would cause overflow (negating MININT
  // overflows). Hence folding should not happen and the original ceildivsi is
  // preserved.

  // TODO: The folder should be able to fold the following by avoiding
  // intermediate operations that overflow.

  // CHECK-DAG: %[[C_1:.*]] = arith.constant 7 : i8
  // CHECK-DAG: %[[MIN_I8:.*]] = arith.constant -128 : i8
  // CHECK-DAG: %[[C_2:.*]] = arith.constant 7 : i16
  // CHECK-DAG: %[[MIN_I16:.*]] = arith.constant -32768 : i16
  // CHECK-DAG: %[[C_3:.*]] = arith.constant 7 : i32
  // CHECK-DAG: %[[MIN_I32:.*]] = arith.constant -2147483648 : i32

  // CHECK-NEXT: %[[CEILDIV_1:.*]] = arith.ceildivsi %[[MIN_I8]], %[[C_1]]  : i8
  %0 = arith.constant 7 : i8
  %min_int_i8 = arith.constant -128 : i8
  %2 = arith.ceildivsi %min_int_i8, %0 : i8

  // CHECK-NEXT: %[[CEILDIV_2:.*]] = arith.ceildivsi %[[MIN_I16]], %[[C_2]]  : i16
  %3 = arith.constant 7 : i16
  %min_int_i16 = arith.constant -32768 : i16
  %5 = arith.ceildivsi %min_int_i16, %3 : i16

  // CHECK-NEXT: %[[CEILDIV_2:.*]] = arith.ceildivsi %[[MIN_I32]], %[[C_3]]  : i32
  %6 = arith.constant 7 : i32
  %min_int_i32 = arith.constant -2147483648 : i32
  %8 = arith.ceildivsi %min_int_i32, %6 : i32

  return %2, %5, %8 : i8, i16, i32
}

// -----

// CHECK-LABEL: func @simple_arith.ceildivui
func.func @simple_arith.ceildivui() -> (i32, i32, i32, i32, i32) {
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0
  %z = arith.constant 0 : i32
  // CHECK-DAG: [[C6:%.+]] = arith.constant 7
  %0 = arith.constant 7 : i32
  %1 = arith.constant 2 : i32

  // ceil(7, 2) = 4
  // CHECK-NEXT: [[C3:%.+]] = arith.constant 4 : i32
  %2 = arith.ceildivui %0, %1 : i32

  %3 = arith.constant -2 : i32

  // ceil(7, -2) = 0
  // CHECK-NEXT: [[CM1:%.+]] = arith.constant 1 : i32
  %4 = arith.ceildivui %0, %3 : i32

  %5 = arith.constant -8 : i32

  // ceil(-8, 2) = 2147483644
  // CHECK-NEXT: [[CM4:%.+]] = arith.constant 2147483644 : i32
  %6 = arith.ceildivui %5, %1 : i32

  %7 = arith.constant -15 : i32

  // ceil(-15, -2) = 0
  // CHECK-NOT: arith.constant 1 : i32
  %8 = arith.ceildivui %7, %3 : i32

  // CHECK-NEXT: [[XZ:%.+]] = arith.ceildivui [[C6]], [[C0]]
  %9 = arith.ceildivui %0, %z : i32

  return %2, %4, %6, %8, %9 : i32, i32, i32, i32, i32
}

// -----

// CHECK-LABEL: func @simple_arith.remsi
func.func @simple_arith.remsi(%a : i32) -> (i32, i32, i32) {
  %0 = arith.constant 5 : i32
  %1 = arith.constant 2 : i32
  %2 = arith.constant 1 : i32
  %3 = arith.constant -2 : i32

  // CHECK-NEXT:[[C1:%.+]] = arith.constant 1 : i32
  %4 = arith.remsi %0, %1 : i32
  %5 = arith.remsi %0, %3 : i32
  // CHECK-NEXT:[[C0:%.+]] = arith.constant 0 : i32
  %6 = arith.remsi %a, %2 : i32

  // CHECK-NEXT: return [[C1]], [[C1]], [[C0]] : i32, i32, i32
  return %4, %5, %6 : i32, i32, i32
}

// -----

// CHECK-LABEL: func @simple_arith.remui
func.func @simple_arith.remui(%a : i32) -> (i32, i32, i32) {
  %0 = arith.constant 5 : i32
  %1 = arith.constant 2 : i32
  %2 = arith.constant 1 : i32
  %3 = arith.constant -2 : i32

  // CHECK-DAG:[[C1:%.+]] = arith.constant 1 : i32
  %4 = arith.remui %0, %1 : i32
  // CHECK-DAG:[[C5:%.+]] = arith.constant 5 : i32
  %5 = arith.remui %0, %3 : i32
  // CHECK-DAG:[[C0:%.+]] = arith.constant 0 : i32
  %6 = arith.remui %a, %2 : i32

  // CHECK-NEXT: return [[C1]], [[C5]], [[C0]] : i32, i32, i32
  return %4, %5, %6 : i32, i32, i32
}

// -----

// CHECK-LABEL: func @muli
func.func @muli() -> i32 {
  %0 = arith.constant 4 : i32
  %1 = arith.constant 2 : i32

  // CHECK-NEXT:[[C8:%.+]] = arith.constant 8 : i32
  %2 = arith.muli %0, %1 : i32

  // CHECK-NEXT: return [[C8]]
  return %2 : i32
}

// -----

// CHECK-LABEL: func @muli_splat_vector
func.func @muli_splat_vector() -> vector<4xi32> {
  %0 = arith.constant dense<4> : vector<4xi32>
  %1 = arith.constant dense<2> : vector<4xi32>

  // CHECK-NEXT: [[C:%.+]] = arith.constant dense<8> : vector<4xi32>
  %2 = arith.muli %0, %1 : vector<4xi32>

  // CHECK-NEXT: return [[C]]
  return %2 : vector<4xi32>
}

// CHECK-LABEL: func @dim
func.func @dim(%x : tensor<8x4xf32>) -> index {

  // CHECK:[[C4:%.+]] = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %x, %c1 : tensor<8x4xf32>

  // CHECK-NEXT: return [[C4]]
  return %0 : index
}

// -----

// CHECK-LABEL: func @cmpi
func.func @cmpi() -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) {
  %c42 = arith.constant 42 : i32
  %cm1 = arith.constant -1 : i32
  // CHECK-DAG: [[F:%.+]] = arith.constant false
  // CHECK-DAG: [[T:%.+]] = arith.constant true
  // CHECK-NEXT: return [[F]],
  %0 = arith.cmpi eq, %c42, %cm1 : i32
  // CHECK-SAME: [[T]],
  %1 = arith.cmpi ne, %c42, %cm1 : i32
  // CHECK-SAME: [[F]],
  %2 = arith.cmpi slt, %c42, %cm1 : i32
  // CHECK-SAME: [[F]],
  %3 = arith.cmpi sle, %c42, %cm1 : i32
  // CHECK-SAME: [[T]],
  %4 = arith.cmpi sgt, %c42, %cm1 : i32
  // CHECK-SAME: [[T]],
  %5 = arith.cmpi sge, %c42, %cm1 : i32
  // CHECK-SAME: [[T]],
  %6 = arith.cmpi ult, %c42, %cm1 : i32
  // CHECK-SAME: [[T]],
  %7 = arith.cmpi ule, %c42, %cm1 : i32
  // CHECK-SAME: [[F]],
  %8 = arith.cmpi ugt, %c42, %cm1 : i32
  // CHECK-SAME: [[F]]
  %9 = arith.cmpi uge, %c42, %cm1 : i32
  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// -----

// CHECK-LABEL: func @cmpf_normal_numbers
func.func @cmpf_normal_numbers() -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) {
  %c42 = arith.constant 42. : f32
  %cm1 = arith.constant -1. : f32
  // CHECK-DAG: [[F:%.+]] = arith.constant false
  // CHECK-DAG: [[T:%.+]] = arith.constant true
  // CHECK-NEXT: return [[F]],
  %0 = arith.cmpf false, %c42, %cm1 : f32
  // CHECK-SAME: [[F]],
  %1 = arith.cmpf oeq, %c42, %cm1 : f32
  // CHECK-SAME: [[T]],
  %2 = arith.cmpf ogt, %c42, %cm1 : f32
  // CHECK-SAME: [[T]],
  %3 = arith.cmpf oge, %c42, %cm1 : f32
  // CHECK-SAME: [[F]],
  %4 = arith.cmpf olt, %c42, %cm1 : f32
  // CHECK-SAME: [[F]],
  %5 = arith.cmpf ole, %c42, %cm1 : f32
  // CHECK-SAME: [[T]],
  %6 = arith.cmpf one, %c42, %cm1 : f32
  // CHECK-SAME: [[T]],
  %7 = arith.cmpf ord, %c42, %cm1 : f32
  // CHECK-SAME: [[F]],
  %8 = arith.cmpf ueq, %c42, %cm1 : f32
  // CHECK-SAME: [[T]],
  %9 = arith.cmpf ugt, %c42, %cm1 : f32
  // CHECK-SAME: [[T]],
  %10 = arith.cmpf uge, %c42, %cm1 : f32
  // CHECK-SAME: [[F]],
  %11 = arith.cmpf ult, %c42, %cm1 : f32
  // CHECK-SAME: [[F]],
  %12 = arith.cmpf ule, %c42, %cm1 : f32
  // CHECK-SAME: [[T]],
  %13 = arith.cmpf une, %c42, %cm1 : f32
  // CHECK-SAME: [[F]],
  %14 = arith.cmpf uno, %c42, %cm1 : f32
  // CHECK-SAME: [[T]]
  %15 = arith.cmpf true, %c42, %cm1 : f32
  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// -----

// CHECK-LABEL: func @cmpf_nan
func.func @cmpf_nan() -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) {
  %c42 = arith.constant 42. : f32
  %cqnan = arith.constant 0xFFFFFFFF : f32
  // CHECK-DAG: [[F:%.+]] = arith.constant false
  // CHECK-DAG: [[T:%.+]] = arith.constant true
  // CHECK-NEXT: return [[F]],
  %0 = arith.cmpf false, %c42, %cqnan : f32
  // CHECK-SAME: [[F]]
  %1 = arith.cmpf oeq, %c42, %cqnan : f32
  // CHECK-SAME: [[F]],
  %2 = arith.cmpf ogt, %c42, %cqnan : f32
  // CHECK-SAME: [[F]],
  %3 = arith.cmpf oge, %c42, %cqnan : f32
  // CHECK-SAME: [[F]],
  %4 = arith.cmpf olt, %c42, %cqnan : f32
  // CHECK-SAME: [[F]],
  %5 = arith.cmpf ole, %c42, %cqnan : f32
  // CHECK-SAME: [[F]],
  %6 = arith.cmpf one, %c42, %cqnan : f32
  // CHECK-SAME: [[F]],
  %7 = arith.cmpf ord, %c42, %cqnan : f32
  // CHECK-SAME: [[T]],
  %8 = arith.cmpf ueq, %c42, %cqnan : f32
  // CHECK-SAME: [[T]],
  %9 = arith.cmpf ugt, %c42, %cqnan : f32
  // CHECK-SAME: [[T]],
  %10 = arith.cmpf uge, %c42, %cqnan : f32
  // CHECK-SAME: [[T]],
  %11 = arith.cmpf ult, %c42, %cqnan : f32
  // CHECK-SAME: [[T]],
  %12 = arith.cmpf ule, %c42, %cqnan : f32
  // CHECK-SAME: [[T]],
  %13 = arith.cmpf une, %c42, %cqnan : f32
  // CHECK-SAME: [[T]],
  %14 = arith.cmpf uno, %c42, %cqnan : f32
  // CHECK-SAME: [[T]]
  %15 = arith.cmpf true, %c42, %cqnan : f32
  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// -----

// CHECK-LABEL: func @cmpf_inf
func.func @cmpf_inf() -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) {
  %c42 = arith.constant 42. : f32
  %cpinf = arith.constant 0x7F800000 : f32
  // CHECK-DAG: [[F:%.+]] = arith.constant false
  // CHECK-DAG: [[T:%.+]] = arith.constant true
  // CHECK-NEXT: return [[F]],
  %0 = arith.cmpf false, %c42, %cpinf: f32
  // CHECK-SAME: [[F]]
  %1 = arith.cmpf oeq, %c42, %cpinf: f32
  // CHECK-SAME: [[F]],
  %2 = arith.cmpf ogt, %c42, %cpinf: f32
  // CHECK-SAME: [[F]],
  %3 = arith.cmpf oge, %c42, %cpinf: f32
  // CHECK-SAME: [[T]],
  %4 = arith.cmpf olt, %c42, %cpinf: f32
  // CHECK-SAME: [[T]],
  %5 = arith.cmpf ole, %c42, %cpinf: f32
  // CHECK-SAME: [[T]],
  %6 = arith.cmpf one, %c42, %cpinf: f32
  // CHECK-SAME: [[T]],
  %7 = arith.cmpf ord, %c42, %cpinf: f32
  // CHECK-SAME: [[F]],
  %8 = arith.cmpf ueq, %c42, %cpinf: f32
  // CHECK-SAME: [[F]],
  %9 = arith.cmpf ugt, %c42, %cpinf: f32
  // CHECK-SAME: [[F]],
  %10 = arith.cmpf uge, %c42, %cpinf: f32
  // CHECK-SAME: [[T]],
  %11 = arith.cmpf ult, %c42, %cpinf: f32
  // CHECK-SAME: [[T]],
  %12 = arith.cmpf ule, %c42, %cpinf: f32
  // CHECK-SAME: [[T]],
  %13 = arith.cmpf une, %c42, %cpinf: f32
  // CHECK-SAME: [[F]],
  %14 = arith.cmpf uno, %c42, %cpinf: f32
  // CHECK-SAME: [[T]]
  %15 = arith.cmpf true, %c42, %cpinf: f32
  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}

// -----

// CHECK-LABEL: func @nested_isolated_region
func.func @nested_isolated_region() {
  // CHECK-NEXT: builtin.module {
  // CHECK-NEXT: func @isolated_op
  // CHECK-NEXT: arith.constant 2
  builtin.module {
    func.func @isolated_op() {
      %0 = arith.constant 1 : i32
      %2 = arith.addi %0, %0 : i32
      "foo.yield"(%2) : (i32) -> ()
    }
  }

  // CHECK: "foo.unknown_region"
  // CHECK-NEXT: arith.constant 2
  "foo.unknown_region"() ({
    %0 = arith.constant 1 : i32
    %2 = arith.addi %0, %0 : i32
    "foo.yield"(%2) : (i32) -> ()
  }) : () -> ()
  return
}

// -----

// CHECK-LABEL: func @custom_insertion_position
func.func @custom_insertion_position() {
  // CHECK: test.one_region_op
  // CHECK-NEXT: arith.constant 2
  "test.one_region_op"() ({

    %0 = arith.constant 1 : i32
    %2 = arith.addi %0, %0 : i32
    "foo.yield"(%2) : (i32) -> ()
  }) : () -> ()
  return
}

// -----

// CHECK-LABEL: func @subview_scalar_fold
func.func @subview_scalar_fold(%arg0: memref<f32>) -> memref<f32> {
  // CHECK-NOT: memref.subview
  %c = memref.subview %arg0[] [] [] : memref<f32> to memref<f32>
  return %c : memref<f32>
}
