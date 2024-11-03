// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @cooperative_matrix_load
spirv.func @cooperative_matrix_load(%ptr : !spirv.ptr<i32, StorageBuffer>, %stride : i32, %b : i1) "None" {
  // CHECK: {{%.*}} = spirv.NV.CooperativeMatrixLoad {{%.*}}, {{%.*}}, {{%.*}} : !spirv.ptr<i32, StorageBuffer> as !spirv.coopmatrix<16x8xi32, Workgroup>
  %0 = spirv.NV.CooperativeMatrixLoad %ptr, %stride, %b : !spirv.ptr<i32, StorageBuffer> as !spirv.coopmatrix<16x8xi32, Workgroup>
  spirv.Return
}

// -----
// CHECK-LABEL: @cooperative_matrix_load_memaccess
spirv.func @cooperative_matrix_load_memaccess(%ptr : !spirv.ptr<i32, StorageBuffer>, %stride : i32, %b : i1) "None" {
  // CHECK: {{%.*}} = spirv.NV.CooperativeMatrixLoad {{%.*}}, {{%.*}}, {{%.*}} ["Volatile"] : !spirv.ptr<i32, StorageBuffer> as !spirv.coopmatrix<8x16xi32, Subgroup>
  %0 = spirv.NV.CooperativeMatrixLoad %ptr, %stride, %b ["Volatile"] : !spirv.ptr<i32, StorageBuffer> as !spirv.coopmatrix<8x16xi32, Subgroup>
  spirv.Return
}

// CHECK-LABEL: @cooperative_matrix_load_diff_ptr_type
spirv.func @cooperative_matrix_load_diff_ptr_type(%ptr : !spirv.ptr<vector<4xi32>, StorageBuffer>, %stride : i32, %b : i1) "None" {
  // CHECK: {{%.*}} = spirv.NV.CooperativeMatrixLoad {{%.*}}, {{%.*}}, {{%.*}} ["Volatile"] : !spirv.ptr<vector<4xi32>, StorageBuffer> as !spirv.coopmatrix<8x16xi32, Subgroup>
  %0 = spirv.NV.CooperativeMatrixLoad %ptr, %stride, %b ["Volatile"] : !spirv.ptr<vector<4xi32>, StorageBuffer> as !spirv.coopmatrix<8x16xi32, Subgroup>
  spirv.Return
}

// CHECK-LABEL: @cooperative_matrix_store
spirv.func @cooperative_matrix_store(%ptr : !spirv.ptr<i32, StorageBuffer>, %stride : i32, %m : !spirv.coopmatrix<8x16xi32, Workgroup>, %b : i1) "None" {
  // CHECK: spirv.NV.CooperativeMatrixStore {{%.*}}, {{%.*}}, {{%.*}} : !spirv.ptr<i32, StorageBuffer>, !spirv.coopmatrix<8x16xi32, Workgroup>
  spirv.NV.CooperativeMatrixStore %ptr, %m, %stride, %b : !spirv.ptr<i32, StorageBuffer>, !spirv.coopmatrix<8x16xi32, Workgroup>
  spirv.Return
}

// CHECK-LABEL: @cooperative_matrix_store_memaccess
spirv.func @cooperative_matrix_store_memaccess(%ptr : !spirv.ptr<i32, StorageBuffer>, %m : !spirv.coopmatrix<8x16xi32, Subgroup>, %stride : i32, %b : i1) "None" {
  // CHECK: spirv.NV.CooperativeMatrixStore {{%.*}}, {{%.*}}, {{%.*}} ["Volatile"] : !spirv.ptr<i32, StorageBuffer>, !spirv.coopmatrix<8x16xi32, Subgroup>
  spirv.NV.CooperativeMatrixStore %ptr, %m, %stride, %b ["Volatile"] : !spirv.ptr<i32, StorageBuffer>, !spirv.coopmatrix<8x16xi32, Subgroup>
  spirv.Return
}

// CHECK-LABEL: @cooperative_matrix_length
spirv.func @cooperative_matrix_length() -> i32 "None" {
  // CHECK: {{%.*}} = spirv.NV.CooperativeMatrixLength : !spirv.coopmatrix<8x16xi32, Subgroup>
  %0 = spirv.NV.CooperativeMatrixLength : !spirv.coopmatrix<8x16xi32, Subgroup>
  spirv.ReturnValue %0 : i32
}

// CHECK-LABEL: @cooperative_matrix_muladd
spirv.func @cooperative_matrix_muladd(%a : !spirv.coopmatrix<8x32xi8, Subgroup>, %b : !spirv.coopmatrix<32x8xi8, Subgroup>, %c : !spirv.coopmatrix<8x8xi32, Subgroup>) "None" {
  // CHECK: {{%.*}} = spirv.NV.CooperativeMatrixMulAdd {{%.*}}, {{%.*}}, {{%.*}}  : !spirv.coopmatrix<8x32xi8, Subgroup>, !spirv.coopmatrix<32x8xi8, Subgroup> -> !spirv.coopmatrix<8x8xi32, Subgroup>
  %r = spirv.NV.CooperativeMatrixMulAdd %a, %b, %c : !spirv.coopmatrix<8x32xi8, Subgroup>, !spirv.coopmatrix<32x8xi8, Subgroup> -> !spirv.coopmatrix<8x8xi32, Subgroup>
  spirv.Return
}

// CHECK-LABEL: @cooperative_matrix_add
spirv.func @cooperative_matrix_add(%a : !spirv.coopmatrix<8x16xi32, Subgroup>, %b : !spirv.coopmatrix<8x16xi32, Subgroup>) "None" {
  // CHECK: {{%.*}} = spirv.IAdd {{%.*}}, {{%.*}} : !spirv.coopmatrix<8x16xi32, Subgroup>
  %r = spirv.IAdd %a, %b : !spirv.coopmatrix<8x16xi32, Subgroup>
  spirv.Return
}

// CHECK-LABEL: @cooperative_matrix_sub
spirv.func @cooperative_matrix_sub(%a : !spirv.coopmatrix<8x16xi32, Subgroup>, %b : !spirv.coopmatrix<8x16xi32, Subgroup>) "None" {
  // CHECK: {{%.*}} = spirv.ISub {{%.*}}, {{%.*}} : !spirv.coopmatrix<8x16xi32, Subgroup>
  %r = spirv.ISub %a, %b : !spirv.coopmatrix<8x16xi32, Subgroup>
  spirv.Return
}

// CHECK-LABEL: @cooperative_matrix_sdiv
spirv.func @cooperative_matrix_sdiv(%a : !spirv.coopmatrix<8x16xi32, Subgroup>, %b : !spirv.coopmatrix<8x16xi32, Subgroup>) "None" {
  // CHECK: {{%.*}} = spirv.SDiv {{%.*}}, {{%.*}} : !spirv.coopmatrix<8x16xi32, Subgroup>
  %r = spirv.SDiv %a, %b : !spirv.coopmatrix<8x16xi32, Subgroup>
  spirv.Return
}

// CHECK-LABEL: @cooperative_matrix_udiv
spirv.func @cooperative_matrix_udiv(%a : !spirv.coopmatrix<8x16xi32, Subgroup>, %b : !spirv.coopmatrix<8x16xi32, Subgroup>) "None" {
  // CHECK: {{%.*}} = spirv.UDiv {{%.*}}, {{%.*}} : !spirv.coopmatrix<8x16xi32, Subgroup>
  %r = spirv.UDiv %a, %b : !spirv.coopmatrix<8x16xi32, Subgroup>
  spirv.Return
}

// CHECK-LABEL: @cooperative_matrix_fadd
spirv.func @cooperative_matrix_fadd(%a : !spirv.coopmatrix<8x16xf32, Subgroup>, %b : !spirv.coopmatrix<8x16xf32, Subgroup>) "None" {
  // CHECK: {{%.*}} = spirv.FAdd {{%.*}}, {{%.*}} : !spirv.coopmatrix<8x16xf32, Subgroup>
  %r = spirv.FAdd %a, %b : !spirv.coopmatrix<8x16xf32, Subgroup>
  spirv.Return
}

// CHECK-LABEL: @cooperative_matrix_fsub
spirv.func @cooperative_matrix_fsub(%a : !spirv.coopmatrix<8x16xf32, Subgroup>, %b : !spirv.coopmatrix<8x16xf32, Subgroup>) "None" {
  // CHECK: {{%.*}} = spirv.FSub {{%.*}}, {{%.*}} : !spirv.coopmatrix<8x16xf32, Subgroup>
  %r = spirv.FSub %a, %b : !spirv.coopmatrix<8x16xf32, Subgroup>
  spirv.Return
}

// CHECK-LABEL: @cooperative_matrix_fdiv
spirv.func @cooperative_matrix_fdiv(%a : !spirv.coopmatrix<8x16xf32, Subgroup>, %b : !spirv.coopmatrix<8x16xf32, Subgroup>) "None" {
  // CHECK: {{%.*}} = spirv.FDiv {{%.*}}, {{%.*}} : !spirv.coopmatrix<8x16xf32, Subgroup>
  %r = spirv.FDiv %a, %b : !spirv.coopmatrix<8x16xf32, Subgroup>
  spirv.Return
}

// -----

// CHECK-LABEL: @cooperative_matrix_access_chain
spirv.func @cooperative_matrix_access_chain(%a : !spirv.ptr<!spirv.coopmatrix<8x16xf32, Subgroup>, Function>) -> !spirv.ptr<f32, Function> "None" {
  %0 = spirv.Constant 0: i32
  // CHECK: {{%.*}} = spirv.AccessChain {{%.*}}[{{%.*}}] : !spirv.ptr<!spirv.coopmatrix<8x16xf32, Subgroup>, Function>, i32
  %1 = spirv.AccessChain %a[%0] : !spirv.ptr<!spirv.coopmatrix<8x16xf32, Subgroup>, Function>, i32
  spirv.ReturnValue %1 : !spirv.ptr<f32, Function>
}

// -----

spirv.func @cooperative_matrix_muladd(%a : !spirv.coopmatrix<16x16xi32, Subgroup>, %b : !spirv.coopmatrix<16x8xi32, Subgroup>, %c : !spirv.coopmatrix<8x8xi32, Subgroup>) "None" {
  // expected-error @+1 {{'spirv.NV.CooperativeMatrixMulAdd' op matrix size must match}}
  %r = spirv.NV.CooperativeMatrixMulAdd %a, %b, %c : !spirv.coopmatrix<16x16xi32, Subgroup>, !spirv.coopmatrix<16x8xi32, Subgroup> -> !spirv.coopmatrix<8x8xi32, Subgroup>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_muladd(%a : !spirv.coopmatrix<8x16xi32, Subgroup>, %b : !spirv.coopmatrix<8x8xi32, Subgroup>, %c : !spirv.coopmatrix<8x8xi32, Subgroup>) "None" {
  // expected-error @+1 {{'spirv.NV.CooperativeMatrixMulAdd' op matrix size must match}}
  %r = spirv.NV.CooperativeMatrixMulAdd %a, %b, %c : !spirv.coopmatrix<8x16xi32, Subgroup>, !spirv.coopmatrix<8x8xi32, Subgroup> -> !spirv.coopmatrix<8x8xi32, Subgroup>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_muladd(%a : !spirv.coopmatrix<8x16xi32, Subgroup>, %b : !spirv.coopmatrix<16x8xi32, Workgroup>, %c : !spirv.coopmatrix<8x8xi32, Subgroup>) "None" {
  // expected-error @+1 {{'spirv.NV.CooperativeMatrixMulAdd' op matrix scope must match}}
  %r = spirv.NV.CooperativeMatrixMulAdd %a, %b, %c : !spirv.coopmatrix<8x16xi32, Subgroup>, !spirv.coopmatrix<16x8xi32, Workgroup> -> !spirv.coopmatrix<8x8xi32, Subgroup>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_muladd(%a : !spirv.coopmatrix<8x16xf32, Subgroup>, %b : !spirv.coopmatrix<16x8xi32, Subgroup>, %c : !spirv.coopmatrix<8x8xi32, Subgroup>) "None" {
  // expected-error @+1 {{matrix A and B non-integer element types must match}}
  %r = spirv.NV.CooperativeMatrixMulAdd %a, %b, %c : !spirv.coopmatrix<8x16xf32, Subgroup>, !spirv.coopmatrix<16x8xi32, Subgroup> -> !spirv.coopmatrix<8x8xi32, Subgroup>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_muladd(%a : !spirv.coopmatrix<8x16xui8, Subgroup>, %b : !spirv.coopmatrix<16x8xsi32, Subgroup>, %c : !spirv.coopmatrix<8x8xi32, Subgroup>) "None" {
  // expected-error @+1 {{matrix A and B integer element types must be the same bit width}}
  %r = spirv.NV.CooperativeMatrixMulAdd %a, %b, %c : !spirv.coopmatrix<8x16xui8, Subgroup>, !spirv.coopmatrix<16x8xsi32, Subgroup> -> !spirv.coopmatrix<8x8xi32, Subgroup>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_load_memaccess(%ptr : !spirv.ptr<!spirv.struct<(f32 [0])>, StorageBuffer>, %stride : i32, %b : i1) "None" {
  // expected-error @+1 {{Pointer must point to a scalar or vector type}}
  %0 = spirv.NV.CooperativeMatrixLoad %ptr, %stride, %b : !spirv.ptr<!spirv.struct<(f32 [0])>, StorageBuffer> as !spirv.coopmatrix<8x16xi32, Subgroup>
  spirv.Return
}

// -----

spirv.func @cooperative_matrix_load_memaccess(%ptr : !spirv.ptr<i32, Function>, %stride : i32, %b : i1) "None" {
  // expected-error @+1 {{Pointer storage class must be Workgroup, StorageBuffer or PhysicalStorageBufferEXT}}
  %0 = spirv.NV.CooperativeMatrixLoad %ptr, %stride, %b : !spirv.ptr<i32, Function> as !spirv.coopmatrix<8x16xi32, Subgroup>
  spirv.Return
}
