// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @joint_matrix_load
spv.func @joint_matrix_load(%ptr : !spv.ptr<i32, Workgroup>, %stride : i32) "None" {
  // CHECK: {{%.*}} = spv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> {{%.*}}, {{%.*}} : (!spv.ptr<i32, Workgroup>, i32) -> !spv.jointmatrix<16x8xi32, RowMajor, Workgroup>
  %0 = spv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> %ptr, %stride : (!spv.ptr<i32, Workgroup>, i32) -> !spv.jointmatrix<16x8xi32, RowMajor, Workgroup>
  spv.Return
}

// -----
// CHECK-LABEL: @joint_matrix_load_memaccess
spv.func @joint_matrix_load_memaccess(%ptr : !spv.ptr<i32, CrossWorkgroup>, %stride : i32) "None" {
  // CHECK: {{%.*}} = spv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> {{%.*}}, {{%.*}} {memory_access = #spv.memory_access<Volatile>} : (!spv.ptr<i32, CrossWorkgroup>, i32) -> !spv.jointmatrix<8x16xi32, ColumnMajor, Subgroup>
  %0 = spv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> %ptr, %stride {memory_access = #spv.memory_access<Volatile>} : (!spv.ptr<i32, CrossWorkgroup>, i32) -> !spv.jointmatrix<8x16xi32, ColumnMajor, Subgroup>
  spv.Return
}

// CHECK-LABEL: @joint_matrix_load_diff_ptr_type
spv.func @joint_matrix_load_diff_ptr_type(%ptr : !spv.ptr<vector<4xi32>, Workgroup>, %stride : i32) "None" {
  // CHECK: {{%.*}} = spv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> {{%.*}}, {{%.*}} {memory_access = #spv.memory_access<Volatile>} : (!spv.ptr<vector<4xi32>, Workgroup>, i32) -> !spv.jointmatrix<8x16xi32, RowMajor, Workgroup>
  %0 = spv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> %ptr, %stride {memory_access = #spv.memory_access<Volatile>} : (!spv.ptr<vector<4xi32>, Workgroup>, i32) -> !spv.jointmatrix<8x16xi32, RowMajor, Workgroup>
  spv.Return
}

// CHECK-LABEL: @joint_matrix_store
spv.func @joint_matrix_store(%ptr : !spv.ptr<i32, Workgroup>, %stride : i32, %m : !spv.jointmatrix<8x16xi32, RowMajor, Workgroup>) "None" {
  // CHECK: spv.INTEL.JointMatrixStore <Subgroup> <RowMajor> {{%.*}}, {{%.*}}, {{%.*}} : (!spv.ptr<i32, Workgroup>, !spv.jointmatrix<8x16xi32, RowMajor, Workgroup>, i32)
  spv.INTEL.JointMatrixStore <Subgroup> <RowMajor> %ptr, %m, %stride : (!spv.ptr<i32, Workgroup>, !spv.jointmatrix<8x16xi32, RowMajor, Workgroup>, i32)
  spv.Return
}

// CHECK-LABEL: @joint_matrix_store_memaccess
spv.func @joint_matrix_store_memaccess(%ptr : !spv.ptr<i32, Workgroup>, %m : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, %stride : i32) "None" {
  // CHECK: spv.INTEL.JointMatrixStore <Subgroup> <ColumnMajor> {{%.*}}, {{%.*}}, {{%.*}} {Volatile} : (!spv.ptr<i32, Workgroup>, !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, i32)
  spv.INTEL.JointMatrixStore <Subgroup> <ColumnMajor> %ptr, %m, %stride {Volatile} : (!spv.ptr<i32, Workgroup>, !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, i32)
  spv.Return
}

// CHECK-LABEL: @joint_matrix_length
spv.func @joint_matrix_length() -> i32 "None" {
  // CHECK: {{%.*}} = spv.INTEL.JointMatrixWorkItemLength : !spv.jointmatrix<8x16xi32, PackedB, Subgroup>
  %0 = spv.INTEL.JointMatrixWorkItemLength : !spv.jointmatrix<8x16xi32, PackedB, Subgroup>
  spv.ReturnValue %0 : i32
}

// CHECK-LABEL: @joint_matrix_muladd
spv.func @joint_matrix_muladd(%a : !spv.jointmatrix<8x32xi8, RowMajor, Subgroup>, %b : !spv.jointmatrix<32x8xi8, ColumnMajor, Subgroup>, %c : !spv.jointmatrix<8x8xi32, RowMajor, Subgroup>) "None" {
  // CHECK: {{%.*}} = spv.INTEL.JointMatrixMad <Subgroup> {{%.*}}, {{%.*}}, {{%.*}}  : !spv.jointmatrix<8x32xi8, RowMajor, Subgroup>, !spv.jointmatrix<32x8xi8, ColumnMajor, Subgroup> -> !spv.jointmatrix<8x8xi32, RowMajor, Subgroup>
  %r = spv.INTEL.JointMatrixMad <Subgroup> %a, %b, %c : !spv.jointmatrix<8x32xi8, RowMajor, Subgroup>, !spv.jointmatrix<32x8xi8, ColumnMajor, Subgroup> -> !spv.jointmatrix<8x8xi32,  RowMajor, Subgroup>
  spv.Return
}

// -----

spv.func @joint_matrix_muladd(%a : !spv.jointmatrix<16x16xi32, RowMajor, Subgroup>, %b : !spv.jointmatrix<16x8xi32, RowMajor, Subgroup>, %c : !spv.jointmatrix<8x8xi32, RowMajor, Subgroup>) "None" {
  // expected-error @+1 {{'spv.INTEL.JointMatrixMad' op matrix size must match}}
  %r = spv.INTEL.JointMatrixMad <Subgroup> %a, %b, %c : !spv.jointmatrix<16x16xi32, RowMajor, Subgroup>, !spv.jointmatrix<16x8xi32, RowMajor, Subgroup> -> !spv.jointmatrix<8x8xi32, RowMajor, Subgroup>
  spv.Return
}

// -----

spv.func @joint_matrix_muladd(%a : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, %b : !spv.jointmatrix<8x8xi32, RowMajor, Subgroup>, %c : !spv.jointmatrix<8x8xi32, RowMajor, Subgroup>) "None" {
  // expected-error @+1 {{'spv.INTEL.JointMatrixMad' op matrix size must match}}
  %r = spv.INTEL.JointMatrixMad <Subgroup> %a, %b, %c : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, !spv.jointmatrix<8x8xi32, RowMajor, Subgroup> -> !spv.jointmatrix<8x8xi32, RowMajor, Subgroup>
  spv.Return
}

// -----

spv.func @joint_matrix_muladd(%a : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, %b : !spv.jointmatrix<16x8xi32, RowMajor, Workgroup>, %c : !spv.jointmatrix<8x8xi32, RowMajor, Subgroup>) "None" {
  // expected-error @+1 {{'spv.INTEL.JointMatrixMad' op matrix scope must match}}
  %r = spv.INTEL.JointMatrixMad <Subgroup> %a, %b, %c : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, !spv.jointmatrix<16x8xi32, RowMajor, Workgroup> -> !spv.jointmatrix<8x8xi32, RowMajor, Subgroup>
  spv.Return
}

// -----

spv.func @joint_matrix_muladd(%a : !spv.jointmatrix<8x16xf32, RowMajor, Subgroup>, %b : !spv.jointmatrix<16x8xi32, RowMajor, Subgroup>, %c : !spv.jointmatrix<8x8xi32, RowMajor, Subgroup>) "None" {
  // expected-error @+1 {{matrix element type must match}}
  %r = spv.INTEL.JointMatrixMad <Subgroup> %a, %b, %c : !spv.jointmatrix<8x16xf32, RowMajor, Subgroup>, !spv.jointmatrix<16x8xi32, RowMajor, Subgroup> -> !spv.jointmatrix<8x8xi32, RowMajor, Subgroup>
  spv.Return
}

// -----

spv.func @joint_matrix_load_memaccess(%ptr : !spv.ptr<!spv.struct<(f32 [0])>, Workgroup>, %stride : i32) "None" {
  // expected-error @+1 {{Pointer must point to a scalar or vector type}}
  %0 = spv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> %ptr, %stride : (!spv.ptr<!spv.struct<(f32 [0])>, Workgroup>, i32)-> !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>
  spv.Return
}

// -----

spv.func @joint_matrix_load_memaccess(%ptr : !spv.ptr<i32, Function>, %stride : i32) "None" {
  // expected-error @+1 {{Pointer storage class must be Workgroup or CrossWorkgroup}}
  %0 = spv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> %ptr, %stride : (!spv.ptr<i32, Function>, i32) -> !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>
  spv.Return
}
