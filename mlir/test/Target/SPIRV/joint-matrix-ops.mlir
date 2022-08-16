// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [JointMatrixINTEL], [SPV_INTEL_joint_matrix]> {
  // CHECK-LABEL: @joint_matrix_load
  spv.func @joint_matrix_load(%ptr : !spv.ptr<i32, Workgroup>, %stride : i32) "None" {
    // CHECK: {{%.*}} = spv.JointMatrixLoadINTEL <Subgroup> <RowMajor> {{%.*}}, {{%.*}} : (!spv.ptr<i32, Workgroup>, i32) -> !spv.jointmatrix<16x8xi32, RowMajor, Workgroup>
    %0 = spv.JointMatrixLoadINTEL <Subgroup> <RowMajor> %ptr, %stride : (!spv.ptr<i32, Workgroup>, i32) -> !spv.jointmatrix<16x8xi32, RowMajor, Workgroup>
    spv.Return
  }

  // CHECK-LABEL: @joint_matrix_load_memaccess
  spv.func @joint_matrix_load_memaccess(%ptr : !spv.ptr<i32, Workgroup>, %stride : i32) "None" {
    // CHECK: {{%.*}} = spv.JointMatrixLoadINTEL <Subgroup> <RowMajor> {{%.*}}, {{%.*}} {memory_access = #spv.memory_access<Volatile>} : (!spv.ptr<i32, Workgroup>, i32) -> !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>
    %0 = spv.JointMatrixLoadINTEL <Subgroup> <RowMajor> %ptr, %stride {memory_access = #spv.memory_access<Volatile>} : (!spv.ptr<i32, Workgroup>, i32) -> !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>
    spv.Return
  }

  // CHECK-LABEL: @joint_matrix_store
  spv.func @joint_matrix_store(%ptr : !spv.ptr<i32, Workgroup>, %stride : i32, %m : !spv.jointmatrix<16x8xi32, RowMajor, Workgroup>) "None" {
    // CHECK: spv.JointMatrixStoreINTEL <Subgroup> <RowMajor> {{%.*}}, {{%.*}}, {{%.*}} : (!spv.ptr<i32, Workgroup>, !spv.jointmatrix<16x8xi32, RowMajor, Workgroup>, i32)
    spv.JointMatrixStoreINTEL <Subgroup> <RowMajor> %ptr, %m, %stride : (!spv.ptr<i32, Workgroup>, !spv.jointmatrix<16x8xi32, RowMajor, Workgroup>, i32)
    spv.Return
  }

  // CHECK-LABEL: @joint_matrix_store_memaccess
  spv.func @joint_matrix_store_memaccess(%ptr : !spv.ptr<i32, Workgroup>, %m : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, %stride : i32) "None" {
    // CHECK: spv.JointMatrixStoreINTEL <Subgroup> <RowMajor> {{%.*}}, {{%.*}}, {{%.*}} {memory_access = #spv.memory_access<Volatile>} : (!spv.ptr<i32, Workgroup>, !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, i32)
    spv.JointMatrixStoreINTEL <Subgroup> <RowMajor> %ptr, %m, %stride {memory_access = #spv.memory_access<Volatile>} : (!spv.ptr<i32, Workgroup>, !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, i32)
    spv.Return
  }

  // CHECK-LABEL: @joint_matrix_length
  spv.func @joint_matrix_length() -> i32 "None" {
    // CHECK: {{%.*}} = spv.JointMatrixWorkItemLengthINTEL : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>
    %0 = spv.JointMatrixWorkItemLengthINTEL : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>
    spv.ReturnValue %0 : i32
  }

  // CHECK-LABEL: @joint_matrix_muladd
  spv.func @joint_matrix_muladd(%a : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, %b : !spv.jointmatrix<16x8xi32, RowMajor, Subgroup>, %c : !spv.jointmatrix<8x8xi32, RowMajor, Subgroup>) "None" {
    // CHECK: {{%.*}} = spv.JointMatrixMadINTEL <Subgroup> {{%.*}}, {{%.*}}, {{%.*}}  : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, !spv.jointmatrix<16x8xi32, RowMajor, Subgroup> -> !spv.jointmatrix<8x8xi32, RowMajor, Subgroup>
    %r = spv.JointMatrixMadINTEL <Subgroup> %a, %b, %c : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, !spv.jointmatrix<16x8xi32, RowMajor, Subgroup> -> !spv.jointmatrix<8x8xi32, RowMajor, Subgroup>
    spv.Return
  }

  // CHECK-LABEL: @joint_matrix_add
  spv.func @joint_matrix_add(%a : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, %b : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>) "None" {
    // CHECK: {{%.*}} = spv.IAdd {{%.*}}, {{%.*}} : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>
    %r = spv.IAdd %a, %b : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>
    spv.Return
  }

  // CHECK-LABEL: @joint_matrix_sub
  spv.func @joint_matrix_sub(%a : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, %b : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>) "None" {
    // CHECK: {{%.*}} = spv.ISub {{%.*}}, {{%.*}} : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>
    %r = spv.ISub %a, %b : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>
    spv.Return
  }

  // CHECK-LABEL: @joint_matrix_sdiv
  spv.func @joint_matrix_sdiv(%a : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, %b : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>) "None" {
    // CHECK: {{%.*}} = spv.SDiv {{%.*}}, {{%.*}} : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>
    %r = spv.SDiv %a, %b : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>
    spv.Return
  }

  // CHECK-LABEL: @joint_matrix_udiv
  spv.func @joint_matrix_udiv(%a : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, %b : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>) "None" {
    // CHECK: {{%.*}} = spv.UDiv {{%.*}}, {{%.*}} : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>
    %r = spv.UDiv %a, %b : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>
    spv.Return
  }

  // CHECK-LABEL: @joint_matrix_fadd
  spv.func @joint_matrix_fadd(%a : !spv.jointmatrix<8x16xf32, RowMajor, Subgroup>, %b : !spv.jointmatrix<8x16xf32, RowMajor, Subgroup>) "None" {
    // CHECK: {{%.*}} = spv.FAdd {{%.*}}, {{%.*}} : !spv.jointmatrix<8x16xf32, RowMajor, Subgroup>
    %r = spv.FAdd %a, %b : !spv.jointmatrix<8x16xf32, RowMajor, Subgroup>
    spv.Return
  }

  // CHECK-LABEL: @joint_matrix_fsub
  spv.func @joint_matrix_fsub(%a : !spv.jointmatrix<8x16xf32, RowMajor, Subgroup>, %b : !spv.jointmatrix<8x16xf32, RowMajor, Subgroup>) "None" {
    // CHECK: {{%.*}} = spv.FSub {{%.*}}, {{%.*}} : !spv.jointmatrix<8x16xf32, RowMajor, Subgroup>
    %r = spv.FSub %a, %b : !spv.jointmatrix<8x16xf32, RowMajor, Subgroup>
    spv.Return
  }

  // CHECK-LABEL: @joint_matrix_fdiv
  spv.func @joint_matrix_fdiv(%a : !spv.jointmatrix<8x16xf32, RowMajor, Subgroup>, %b : !spv.jointmatrix<8x16xf32, RowMajor, Subgroup>) "None" {
    // CHECK: {{%.*}} = spv.FDiv {{%.*}}, {{%.*}} : !spv.jointmatrix<8x16xf32, RowMajor, Subgroup>
    %r = spv.FDiv %a, %b : !spv.jointmatrix<8x16xf32, RowMajor, Subgroup>
    spv.Return
  }

  // CHECK-LABEL: @joint_matrix_access_chain
  spv.func @joint_matrix_access_chain(%a : !spv.ptr<!spv.jointmatrix<8x16xf32, RowMajor, Subgroup>, Function>) -> !spv.ptr<f32, Function> "None" {
    %0 = spv.Constant 0: i32
    // CHECK: {{%.*}} = spv.AccessChain {{%.*}}[{{%.*}}] : !spv.ptr<!spv.jointmatrix<8x16xf32, RowMajor, Subgroup>, Function>, i32
    %1 = spv.AccessChain %a[%0] : !spv.ptr<!spv.jointmatrix<8x16xf32, RowMajor, Subgroup>, Function>, i32
    spv.ReturnValue %1 : !spv.ptr<f32, Function>
  }
}
