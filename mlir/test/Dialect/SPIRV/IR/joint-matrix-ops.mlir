// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @joint_matrix_load
spirv.func @joint_matrix_load(%ptr : !spirv.ptr<i32, Workgroup>, %stride : i32) "None" {
  // CHECK: {{%.*}} = spirv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> {{%.*}}, {{%.*}} : (!spirv.ptr<i32, Workgroup>, i32) -> !spirv.jointmatrix<16x8xi32, RowMajor, Workgroup>
  %0 = spirv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> %ptr, %stride : (!spirv.ptr<i32, Workgroup>, i32) -> !spirv.jointmatrix<16x8xi32, RowMajor, Workgroup>
  spirv.Return
}

// -----
// CHECK-LABEL: @joint_matrix_load_memaccess
spirv.func @joint_matrix_load_memaccess(%ptr : !spirv.ptr<i32, CrossWorkgroup>, %stride : i32) "None" {
  // CHECK: {{%.*}} = spirv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> {{%.*}}, {{%.*}} {memory_access = #spirv.memory_access<Volatile>} : (!spirv.ptr<i32, CrossWorkgroup>, i32) -> !spirv.jointmatrix<8x16xi32, ColumnMajor, Subgroup>
  %0 = spirv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> %ptr, %stride {memory_access = #spirv.memory_access<Volatile>} : (!spirv.ptr<i32, CrossWorkgroup>, i32) -> !spirv.jointmatrix<8x16xi32, ColumnMajor, Subgroup>
  spirv.Return
}

// CHECK-LABEL: @joint_matrix_load_diff_ptr_type
spirv.func @joint_matrix_load_diff_ptr_type(%ptr : !spirv.ptr<vector<4xi32>, Workgroup>, %stride : i32) "None" {
  // CHECK: {{%.*}} = spirv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> {{%.*}}, {{%.*}} {memory_access = #spirv.memory_access<Volatile>} : (!spirv.ptr<vector<4xi32>, Workgroup>, i32) -> !spirv.jointmatrix<8x16xi32, RowMajor, Workgroup>
  %0 = spirv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> %ptr, %stride {memory_access = #spirv.memory_access<Volatile>} : (!spirv.ptr<vector<4xi32>, Workgroup>, i32) -> !spirv.jointmatrix<8x16xi32, RowMajor, Workgroup>
  spirv.Return
}

// CHECK-LABEL: @joint_matrix_store
spirv.func @joint_matrix_store(%ptr : !spirv.ptr<i32, Workgroup>, %stride : i32, %m : !spirv.jointmatrix<8x16xi32, RowMajor, Workgroup>) "None" {
  // CHECK: spirv.INTEL.JointMatrixStore <Subgroup> <RowMajor> {{%.*}}, {{%.*}}, {{%.*}} : (!spirv.ptr<i32, Workgroup>, !spirv.jointmatrix<8x16xi32, RowMajor, Workgroup>, i32)
  spirv.INTEL.JointMatrixStore <Subgroup> <RowMajor> %ptr, %m, %stride : (!spirv.ptr<i32, Workgroup>, !spirv.jointmatrix<8x16xi32, RowMajor, Workgroup>, i32)
  spirv.Return
}

// CHECK-LABEL: @joint_matrix_store_memaccess
spirv.func @joint_matrix_store_memaccess(%ptr : !spirv.ptr<i32, Workgroup>, %m : !spirv.jointmatrix<8x16xi32, RowMajor, Subgroup>, %stride : i32) "None" {
  // CHECK: spirv.INTEL.JointMatrixStore <Subgroup> <ColumnMajor> {{%.*}}, {{%.*}}, {{%.*}} {Volatile} : (!spirv.ptr<i32, Workgroup>, !spirv.jointmatrix<8x16xi32, RowMajor, Subgroup>, i32)
  spirv.INTEL.JointMatrixStore <Subgroup> <ColumnMajor> %ptr, %m, %stride {Volatile} : (!spirv.ptr<i32, Workgroup>, !spirv.jointmatrix<8x16xi32, RowMajor, Subgroup>, i32)
  spirv.Return
}

// CHECK-LABEL: @joint_matrix_length
spirv.func @joint_matrix_length() -> i32 "None" {
  // CHECK: {{%.*}} = spirv.INTEL.JointMatrixWorkItemLength : !spirv.jointmatrix<8x16xi32, PackedB, Subgroup>
  %0 = spirv.INTEL.JointMatrixWorkItemLength : !spirv.jointmatrix<8x16xi32, PackedB, Subgroup>
  spirv.ReturnValue %0 : i32
}

// CHECK-LABEL: @joint_matrix_muladd
spirv.func @joint_matrix_muladd(%a : !spirv.jointmatrix<8x32xi8, RowMajor, Subgroup>, %b : !spirv.jointmatrix<32x8xi8, ColumnMajor, Subgroup>, %c : !spirv.jointmatrix<8x8xi32, RowMajor, Subgroup>) "None" {
  // CHECK: {{%.*}} = spirv.INTEL.JointMatrixMad <Subgroup> {{%.*}}, {{%.*}}, {{%.*}}  : !spirv.jointmatrix<8x32xi8, RowMajor, Subgroup>, !spirv.jointmatrix<32x8xi8, ColumnMajor, Subgroup> -> !spirv.jointmatrix<8x8xi32, RowMajor, Subgroup>
  %r = spirv.INTEL.JointMatrixMad <Subgroup> %a, %b, %c : !spirv.jointmatrix<8x32xi8, RowMajor, Subgroup>, !spirv.jointmatrix<32x8xi8, ColumnMajor, Subgroup> -> !spirv.jointmatrix<8x8xi32,  RowMajor, Subgroup>
  spirv.Return
}

// -----

spirv.func @joint_matrix_muladd(%a : !spirv.jointmatrix<16x16xi32, RowMajor, Subgroup>, %b : !spirv.jointmatrix<16x8xi32, RowMajor, Subgroup>, %c : !spirv.jointmatrix<8x8xi32, RowMajor, Subgroup>) "None" {
  // expected-error @+1 {{'spirv.INTEL.JointMatrixMad' op matrix size must match}}
  %r = spirv.INTEL.JointMatrixMad <Subgroup> %a, %b, %c : !spirv.jointmatrix<16x16xi32, RowMajor, Subgroup>, !spirv.jointmatrix<16x8xi32, RowMajor, Subgroup> -> !spirv.jointmatrix<8x8xi32, RowMajor, Subgroup>
  spirv.Return
}

// -----

spirv.func @joint_matrix_muladd(%a : !spirv.jointmatrix<8x16xi32, RowMajor, Subgroup>, %b : !spirv.jointmatrix<8x8xi32, RowMajor, Subgroup>, %c : !spirv.jointmatrix<8x8xi32, RowMajor, Subgroup>) "None" {
  // expected-error @+1 {{'spirv.INTEL.JointMatrixMad' op matrix size must match}}
  %r = spirv.INTEL.JointMatrixMad <Subgroup> %a, %b, %c : !spirv.jointmatrix<8x16xi32, RowMajor, Subgroup>, !spirv.jointmatrix<8x8xi32, RowMajor, Subgroup> -> !spirv.jointmatrix<8x8xi32, RowMajor, Subgroup>
  spirv.Return
}

// -----

spirv.func @joint_matrix_muladd(%a : !spirv.jointmatrix<8x16xi32, RowMajor, Subgroup>, %b : !spirv.jointmatrix<16x8xi32, RowMajor, Workgroup>, %c : !spirv.jointmatrix<8x8xi32, RowMajor, Subgroup>) "None" {
  // expected-error @+1 {{'spirv.INTEL.JointMatrixMad' op matrix scope must match}}
  %r = spirv.INTEL.JointMatrixMad <Subgroup> %a, %b, %c : !spirv.jointmatrix<8x16xi32, RowMajor, Subgroup>, !spirv.jointmatrix<16x8xi32, RowMajor, Workgroup> -> !spirv.jointmatrix<8x8xi32, RowMajor, Subgroup>
  spirv.Return
}

// -----

spirv.func @joint_matrix_muladd(%a : !spirv.jointmatrix<8x16xf32, RowMajor, Subgroup>, %b : !spirv.jointmatrix<16x8xi32, RowMajor, Subgroup>, %c : !spirv.jointmatrix<8x8xi32, RowMajor, Subgroup>) "None" {
  // expected-error @+1 {{matrix element type must match}}
  %r = spirv.INTEL.JointMatrixMad <Subgroup> %a, %b, %c : !spirv.jointmatrix<8x16xf32, RowMajor, Subgroup>, !spirv.jointmatrix<16x8xi32, RowMajor, Subgroup> -> !spirv.jointmatrix<8x8xi32, RowMajor, Subgroup>
  spirv.Return
}

// -----

spirv.func @joint_matrix_load_memaccess(%ptr : !spirv.ptr<!spirv.struct<(f32 [0])>, Workgroup>, %stride : i32) "None" {
  // expected-error @+1 {{Pointer must point to a scalar or vector type}}
  %0 = spirv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> %ptr, %stride : (!spirv.ptr<!spirv.struct<(f32 [0])>, Workgroup>, i32)-> !spirv.jointmatrix<8x16xi32, RowMajor, Subgroup>
  spirv.Return
}

// -----

spirv.func @joint_matrix_load_memaccess(%ptr : !spirv.ptr<i32, Function>, %stride : i32) "None" {
  // expected-error @+1 {{Pointer storage class must be Workgroup or CrossWorkgroup}}
  %0 = spirv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> %ptr, %stride : (!spirv.ptr<i32, Function>, i32) -> !spirv.jointmatrix<8x16xi32, RowMajor, Subgroup>
  spirv.Return
}
