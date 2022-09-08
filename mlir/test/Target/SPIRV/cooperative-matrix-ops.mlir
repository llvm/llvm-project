// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [CooperativeMatrixNV], [SPV_NV_cooperative_matrix]> {
  // CHECK-LABEL: @cooperative_matrix_load
  spv.func @cooperative_matrix_load(%ptr : !spv.ptr<i32, StorageBuffer>, %stride : i32, %b : i1) "None" {
    // CHECK: {{%.*}} = spv.NV.CooperativeMatrixLoad {{%.*}}, {{%.*}}, {{%.*}} : !spv.ptr<i32, StorageBuffer> as !spv.coopmatrix<16x8xi32, Workgroup>
    %0 = spv.NV.CooperativeMatrixLoad %ptr, %stride, %b : !spv.ptr<i32, StorageBuffer> as !spv.coopmatrix<16x8xi32, Workgroup>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_load_memaccess
  spv.func @cooperative_matrix_load_memaccess(%ptr : !spv.ptr<i32, StorageBuffer>, %stride : i32, %b : i1) "None" {
    // CHECK: {{%.*}} = spv.NV.CooperativeMatrixLoad {{%.*}}, {{%.*}}, {{%.*}} ["Volatile"] : !spv.ptr<i32, StorageBuffer> as !spv.coopmatrix<8x16xi32, Subgroup>
    %0 = spv.NV.CooperativeMatrixLoad %ptr, %stride, %b ["Volatile"] : !spv.ptr<i32, StorageBuffer> as !spv.coopmatrix<8x16xi32, Subgroup>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_store
  spv.func @cooperative_matrix_store(%ptr : !spv.ptr<i32, StorageBuffer>, %stride : i32, %m : !spv.coopmatrix<16x8xi32, Workgroup>, %b : i1) "None" {
    // CHECK: spv.NV.CooperativeMatrixStore {{%.*}}, {{%.*}}, {{%.*}} : !spv.ptr<i32, StorageBuffer>, !spv.coopmatrix<16x8xi32, Workgroup>
    spv.NV.CooperativeMatrixStore %ptr, %m, %stride, %b : !spv.ptr<i32, StorageBuffer>, !spv.coopmatrix<16x8xi32, Workgroup>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_store_memaccess
  spv.func @cooperative_matrix_store_memaccess(%ptr : !spv.ptr<i32, StorageBuffer>, %m : !spv.coopmatrix<8x16xi32, Subgroup>, %stride : i32, %b : i1) "None" {
    // CHECK: spv.NV.CooperativeMatrixStore {{%.*}}, {{%.*}}, {{%.*}} ["Volatile"] : !spv.ptr<i32, StorageBuffer>, !spv.coopmatrix<8x16xi32, Subgroup>
    spv.NV.CooperativeMatrixStore %ptr, %m, %stride, %b ["Volatile"] : !spv.ptr<i32, StorageBuffer>, !spv.coopmatrix<8x16xi32, Subgroup>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_length
  spv.func @cooperative_matrix_length() -> i32 "None" {
    // CHECK: {{%.*}} = spv.NV.CooperativeMatrixLength : !spv.coopmatrix<8x16xi32, Subgroup>
    %0 = spv.NV.CooperativeMatrixLength : !spv.coopmatrix<8x16xi32, Subgroup>
    spv.ReturnValue %0 : i32
  }

  // CHECK-LABEL: @cooperative_matrix_muladd
  spv.func @cooperative_matrix_muladd(%a : !spv.coopmatrix<8x16xi32, Subgroup>, %b : !spv.coopmatrix<16x8xi32, Subgroup>, %c : !spv.coopmatrix<8x8xi32, Subgroup>) "None" {
    // CHECK: {{%.*}} = spv.NV.CooperativeMatrixMulAdd {{%.*}}, {{%.*}}, {{%.*}}  : !spv.coopmatrix<8x16xi32, Subgroup>, !spv.coopmatrix<16x8xi32, Subgroup> -> !spv.coopmatrix<8x8xi32, Subgroup>
    %r = spv.NV.CooperativeMatrixMulAdd %a, %b, %c : !spv.coopmatrix<8x16xi32, Subgroup>, !spv.coopmatrix<16x8xi32, Subgroup> -> !spv.coopmatrix<8x8xi32, Subgroup>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_add
  spv.func @cooperative_matrix_add(%a : !spv.coopmatrix<8x16xi32, Subgroup>, %b : !spv.coopmatrix<8x16xi32, Subgroup>) "None" {
    // CHECK: {{%.*}} = spv.IAdd {{%.*}}, {{%.*}} : !spv.coopmatrix<8x16xi32, Subgroup>
    %r = spv.IAdd %a, %b : !spv.coopmatrix<8x16xi32, Subgroup>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_sub
  spv.func @cooperative_matrix_sub(%a : !spv.coopmatrix<8x16xi32, Subgroup>, %b : !spv.coopmatrix<8x16xi32, Subgroup>) "None" {
    // CHECK: {{%.*}} = spv.ISub {{%.*}}, {{%.*}} : !spv.coopmatrix<8x16xi32, Subgroup>
    %r = spv.ISub %a, %b : !spv.coopmatrix<8x16xi32, Subgroup>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_sdiv
  spv.func @cooperative_matrix_sdiv(%a : !spv.coopmatrix<8x16xi32, Subgroup>, %b : !spv.coopmatrix<8x16xi32, Subgroup>) "None" {
    // CHECK: {{%.*}} = spv.SDiv {{%.*}}, {{%.*}} : !spv.coopmatrix<8x16xi32, Subgroup>
    %r = spv.SDiv %a, %b : !spv.coopmatrix<8x16xi32, Subgroup>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_udiv
  spv.func @cooperative_matrix_udiv(%a : !spv.coopmatrix<8x16xi32, Subgroup>, %b : !spv.coopmatrix<8x16xi32, Subgroup>) "None" {
    // CHECK: {{%.*}} = spv.UDiv {{%.*}}, {{%.*}} : !spv.coopmatrix<8x16xi32, Subgroup>
    %r = spv.UDiv %a, %b : !spv.coopmatrix<8x16xi32, Subgroup>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_fadd
  spv.func @cooperative_matrix_fadd(%a : !spv.coopmatrix<8x16xf32, Subgroup>, %b : !spv.coopmatrix<8x16xf32, Subgroup>) "None" {
    // CHECK: {{%.*}} = spv.FAdd {{%.*}}, {{%.*}} : !spv.coopmatrix<8x16xf32, Subgroup>
    %r = spv.FAdd %a, %b : !spv.coopmatrix<8x16xf32, Subgroup>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_fsub
  spv.func @cooperative_matrix_fsub(%a : !spv.coopmatrix<8x16xf32, Subgroup>, %b : !spv.coopmatrix<8x16xf32, Subgroup>) "None" {
    // CHECK: {{%.*}} = spv.FSub {{%.*}}, {{%.*}} : !spv.coopmatrix<8x16xf32, Subgroup>
    %r = spv.FSub %a, %b : !spv.coopmatrix<8x16xf32, Subgroup>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_fdiv
  spv.func @cooperative_matrix_fdiv(%a : !spv.coopmatrix<8x16xf32, Subgroup>, %b : !spv.coopmatrix<8x16xf32, Subgroup>) "None" {
    // CHECK: {{%.*}} = spv.FDiv {{%.*}}, {{%.*}} : !spv.coopmatrix<8x16xf32, Subgroup>
    %r = spv.FDiv %a, %b : !spv.coopmatrix<8x16xf32, Subgroup>
    spv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_access_chain
  spv.func @cooperative_matrix_access_chain(%a : !spv.ptr<!spv.coopmatrix<8x16xf32, Subgroup>, Function>) -> !spv.ptr<f32, Function> "None" {
    %0 = spv.Constant 0: i32
    // CHECK: {{%.*}} = spv.AccessChain {{%.*}}[{{%.*}}] : !spv.ptr<!spv.coopmatrix<8x16xf32, Subgroup>, Function>, i32
    %1 = spv.AccessChain %a[%0] : !spv.ptr<!spv.coopmatrix<8x16xf32, Subgroup>, Function>, i32
    spv.ReturnValue %1 : !spv.ptr<f32, Function>
  }
}
