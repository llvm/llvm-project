// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [CooperativeMatrixNV], [SPV_NV_cooperative_matrix]> {
  // CHECK-LABEL: @cooperative_matrix_load
  spirv.func @cooperative_matrix_load(%ptr : !spirv.ptr<i32, StorageBuffer>, %stride : i32, %b : i1) "None" {
    // CHECK: {{%.*}} = spirv.NV.CooperativeMatrixLoad {{%.*}}, {{%.*}}, {{%.*}} : !spirv.ptr<i32, StorageBuffer> as !spirv.NV.coopmatrix<16x8xi32, Workgroup>
    %0 = spirv.NV.CooperativeMatrixLoad %ptr, %stride, %b : !spirv.ptr<i32, StorageBuffer> as !spirv.NV.coopmatrix<16x8xi32, Workgroup>
    spirv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_load_memaccess
  spirv.func @cooperative_matrix_load_memaccess(%ptr : !spirv.ptr<i32, StorageBuffer>, %stride : i32, %b : i1) "None" {
    // CHECK: {{%.*}} = spirv.NV.CooperativeMatrixLoad {{%.*}}, {{%.*}}, {{%.*}} ["Volatile"] : !spirv.ptr<i32, StorageBuffer> as !spirv.NV.coopmatrix<8x16xi32, Subgroup>
    %0 = spirv.NV.CooperativeMatrixLoad %ptr, %stride, %b ["Volatile"] : !spirv.ptr<i32, StorageBuffer> as !spirv.NV.coopmatrix<8x16xi32, Subgroup>
    spirv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_store
  spirv.func @cooperative_matrix_store(%ptr : !spirv.ptr<i32, StorageBuffer>, %stride : i32, %m : !spirv.NV.coopmatrix<16x8xi32, Workgroup>, %b : i1) "None" {
    // CHECK: spirv.NV.CooperativeMatrixStore {{%.*}}, {{%.*}}, {{%.*}} : !spirv.ptr<i32, StorageBuffer>, !spirv.NV.coopmatrix<16x8xi32, Workgroup>
    spirv.NV.CooperativeMatrixStore %ptr, %m, %stride, %b : !spirv.ptr<i32, StorageBuffer>, !spirv.NV.coopmatrix<16x8xi32, Workgroup>
    spirv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_store_memaccess
  spirv.func @cooperative_matrix_store_memaccess(%ptr : !spirv.ptr<i32, StorageBuffer>, %m : !spirv.NV.coopmatrix<8x16xi32, Subgroup>, %stride : i32, %b : i1) "None" {
    // CHECK: spirv.NV.CooperativeMatrixStore {{%.*}}, {{%.*}}, {{%.*}} ["Volatile"] : !spirv.ptr<i32, StorageBuffer>, !spirv.NV.coopmatrix<8x16xi32, Subgroup>
    spirv.NV.CooperativeMatrixStore %ptr, %m, %stride, %b ["Volatile"] : !spirv.ptr<i32, StorageBuffer>, !spirv.NV.coopmatrix<8x16xi32, Subgroup>
    spirv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_length
  spirv.func @cooperative_matrix_length() -> i32 "None" {
    // CHECK: {{%.*}} = spirv.NV.CooperativeMatrixLength : !spirv.NV.coopmatrix<8x16xi32, Subgroup>
    %0 = spirv.NV.CooperativeMatrixLength : !spirv.NV.coopmatrix<8x16xi32, Subgroup>
    spirv.ReturnValue %0 : i32
  }

  // CHECK-LABEL: @cooperative_matrix_muladd
  spirv.func @cooperative_matrix_muladd(%a : !spirv.NV.coopmatrix<8x16xi32, Subgroup>, %b : !spirv.NV.coopmatrix<16x8xi32, Subgroup>, %c : !spirv.NV.coopmatrix<8x8xi32, Subgroup>) "None" {
    // CHECK: {{%.*}} = spirv.NV.CooperativeMatrixMulAdd {{%.*}}, {{%.*}}, {{%.*}}  : !spirv.NV.coopmatrix<8x16xi32, Subgroup>, !spirv.NV.coopmatrix<16x8xi32, Subgroup> -> !spirv.NV.coopmatrix<8x8xi32, Subgroup>
    %r = spirv.NV.CooperativeMatrixMulAdd %a, %b, %c : !spirv.NV.coopmatrix<8x16xi32, Subgroup>, !spirv.NV.coopmatrix<16x8xi32, Subgroup> -> !spirv.NV.coopmatrix<8x8xi32, Subgroup>
    spirv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_add
  spirv.func @cooperative_matrix_add(%a : !spirv.NV.coopmatrix<8x16xi32, Subgroup>, %b : !spirv.NV.coopmatrix<8x16xi32, Subgroup>) "None" {
    // CHECK: {{%.*}} = spirv.IAdd {{%.*}}, {{%.*}} : !spirv.NV.coopmatrix<8x16xi32, Subgroup>
    %r = spirv.IAdd %a, %b : !spirv.NV.coopmatrix<8x16xi32, Subgroup>
    spirv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_sub
  spirv.func @cooperative_matrix_sub(%a : !spirv.NV.coopmatrix<8x16xi32, Subgroup>, %b : !spirv.NV.coopmatrix<8x16xi32, Subgroup>) "None" {
    // CHECK: {{%.*}} = spirv.ISub {{%.*}}, {{%.*}} : !spirv.NV.coopmatrix<8x16xi32, Subgroup>
    %r = spirv.ISub %a, %b : !spirv.NV.coopmatrix<8x16xi32, Subgroup>
    spirv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_sdiv
  spirv.func @cooperative_matrix_sdiv(%a : !spirv.NV.coopmatrix<8x16xi32, Subgroup>, %b : !spirv.NV.coopmatrix<8x16xi32, Subgroup>) "None" {
    // CHECK: {{%.*}} = spirv.SDiv {{%.*}}, {{%.*}} : !spirv.NV.coopmatrix<8x16xi32, Subgroup>
    %r = spirv.SDiv %a, %b : !spirv.NV.coopmatrix<8x16xi32, Subgroup>
    spirv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_udiv
  spirv.func @cooperative_matrix_udiv(%a : !spirv.NV.coopmatrix<8x16xi32, Subgroup>, %b : !spirv.NV.coopmatrix<8x16xi32, Subgroup>) "None" {
    // CHECK: {{%.*}} = spirv.UDiv {{%.*}}, {{%.*}} : !spirv.NV.coopmatrix<8x16xi32, Subgroup>
    %r = spirv.UDiv %a, %b : !spirv.NV.coopmatrix<8x16xi32, Subgroup>
    spirv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_fadd
  spirv.func @cooperative_matrix_fadd(%a : !spirv.NV.coopmatrix<8x16xf32, Subgroup>, %b : !spirv.NV.coopmatrix<8x16xf32, Subgroup>) "None" {
    // CHECK: {{%.*}} = spirv.FAdd {{%.*}}, {{%.*}} : !spirv.NV.coopmatrix<8x16xf32, Subgroup>
    %r = spirv.FAdd %a, %b : !spirv.NV.coopmatrix<8x16xf32, Subgroup>
    spirv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_fsub
  spirv.func @cooperative_matrix_fsub(%a : !spirv.NV.coopmatrix<8x16xf32, Subgroup>, %b : !spirv.NV.coopmatrix<8x16xf32, Subgroup>) "None" {
    // CHECK: {{%.*}} = spirv.FSub {{%.*}}, {{%.*}} : !spirv.NV.coopmatrix<8x16xf32, Subgroup>
    %r = spirv.FSub %a, %b : !spirv.NV.coopmatrix<8x16xf32, Subgroup>
    spirv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_fdiv
  spirv.func @cooperative_matrix_fdiv(%a : !spirv.NV.coopmatrix<8x16xf32, Subgroup>, %b : !spirv.NV.coopmatrix<8x16xf32, Subgroup>) "None" {
    // CHECK: {{%.*}} = spirv.FDiv {{%.*}}, {{%.*}} : !spirv.NV.coopmatrix<8x16xf32, Subgroup>
    %r = spirv.FDiv %a, %b : !spirv.NV.coopmatrix<8x16xf32, Subgroup>
    spirv.Return
  }

  // CHECK-LABEL: @cooperative_matrix_access_chain
  spirv.func @cooperative_matrix_access_chain(%a : !spirv.ptr<!spirv.NV.coopmatrix<8x16xf32, Subgroup>, Function>) -> !spirv.ptr<f32, Function> "None" {
    %0 = spirv.Constant 0: i32
    // CHECK: {{%.*}} = spirv.AccessChain {{%.*}}[{{%.*}}] : !spirv.ptr<!spirv.NV.coopmatrix<8x16xf32, Subgroup>, Function>, i32
    %1 = spirv.AccessChain %a[%0] : !spirv.ptr<!spirv.NV.coopmatrix<8x16xf32, Subgroup>, Function>, i32
    spirv.ReturnValue %1 : !spirv.ptr<f32, Function>
  }
}
