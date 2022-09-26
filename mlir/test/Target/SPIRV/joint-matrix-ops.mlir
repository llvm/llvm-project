// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [JointMatrixINTEL], [SPIRV_INTEL_joint_matrix]> {
  // CHECK-LABEL: @joint_matrix_load
  spirv.func @joint_matrix_load(%ptr : !spirv.ptr<i32, Workgroup>, %stride : i32) "None" {
    // CHECK: {{%.*}} = spirv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> {{%.*}}, {{%.*}} : (!spirv.ptr<i32, Workgroup>, i32) -> !spirv.jointmatrix<16x8xi32, RowMajor, Workgroup>
    %0 = spirv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> %ptr, %stride : (!spirv.ptr<i32, Workgroup>, i32) -> !spirv.jointmatrix<16x8xi32, RowMajor, Workgroup>
    spirv.Return
  }

  // CHECK-LABEL: @joint_matrix_load_memaccess
  spirv.func @joint_matrix_load_memaccess(%ptr : !spirv.ptr<i32, Workgroup>, %stride : i32) "None" {
    // CHECK: {{%.*}} = spirv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> {{%.*}}, {{%.*}} {memory_access = #spirv.memory_access<Volatile>} : (!spirv.ptr<i32, Workgroup>, i32) -> !spirv.jointmatrix<8x16xi32, RowMajor, Subgroup>
    %0 = spirv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> %ptr, %stride {memory_access = #spirv.memory_access<Volatile>} : (!spirv.ptr<i32, Workgroup>, i32) -> !spirv.jointmatrix<8x16xi32, RowMajor, Subgroup>
    spirv.Return
  }

  // CHECK-LABEL: @joint_matrix_store
  spirv.func @joint_matrix_store(%ptr : !spirv.ptr<i32, Workgroup>, %stride : i32, %m : !spirv.jointmatrix<16x8xi32, RowMajor, Workgroup>) "None" {
    // CHECK: spirv.INTEL.JointMatrixStore <Subgroup> <RowMajor> {{%.*}}, {{%.*}}, {{%.*}} : (!spirv.ptr<i32, Workgroup>, !spirv.jointmatrix<16x8xi32, RowMajor, Workgroup>, i32)
    spirv.INTEL.JointMatrixStore <Subgroup> <RowMajor> %ptr, %m, %stride : (!spirv.ptr<i32, Workgroup>, !spirv.jointmatrix<16x8xi32, RowMajor, Workgroup>, i32)
    spirv.Return
  }

  // CHECK-LABEL: @joint_matrix_store_memaccess
  spirv.func @joint_matrix_store_memaccess(%ptr : !spirv.ptr<i32, Workgroup>, %m : !spirv.jointmatrix<8x16xi32, RowMajor, Subgroup>, %stride : i32) "None" {
    // CHECK: spirv.INTEL.JointMatrixStore <Subgroup> <RowMajor> {{%.*}}, {{%.*}}, {{%.*}} {memory_access = #spirv.memory_access<Volatile>} : (!spirv.ptr<i32, Workgroup>, !spirv.jointmatrix<8x16xi32, RowMajor, Subgroup>, i32)
    spirv.INTEL.JointMatrixStore <Subgroup> <RowMajor> %ptr, %m, %stride {memory_access = #spirv.memory_access<Volatile>} : (!spirv.ptr<i32, Workgroup>, !spirv.jointmatrix<8x16xi32, RowMajor, Subgroup>, i32)
    spirv.Return
  }

  // CHECK-LABEL: @joint_matrix_length
  spirv.func @joint_matrix_length() -> i32 "None" {
    // CHECK: {{%.*}} = spirv.INTEL.JointMatrixWorkItemLength : !spirv.jointmatrix<8x16xi32, RowMajor, Subgroup>
    %0 = spirv.INTEL.JointMatrixWorkItemLength : !spirv.jointmatrix<8x16xi32, RowMajor, Subgroup>
    spirv.ReturnValue %0 : i32
  }

  // CHECK-LABEL: @joint_matrix_muladd
  spirv.func @joint_matrix_muladd(%a : !spirv.jointmatrix<8x16xi32, RowMajor, Subgroup>, %b : !spirv.jointmatrix<16x8xi32, RowMajor, Subgroup>, %c : !spirv.jointmatrix<8x8xi32, RowMajor, Subgroup>) "None" {
    // CHECK: {{%.*}} = spirv.INTEL.JointMatrixMad <Subgroup> {{%.*}}, {{%.*}}, {{%.*}}  : !spirv.jointmatrix<8x16xi32, RowMajor, Subgroup>, !spirv.jointmatrix<16x8xi32, RowMajor, Subgroup> -> !spirv.jointmatrix<8x8xi32, RowMajor, Subgroup>
    %r = spirv.INTEL.JointMatrixMad <Subgroup> %a, %b, %c : !spirv.jointmatrix<8x16xi32, RowMajor, Subgroup>, !spirv.jointmatrix<16x8xi32, RowMajor, Subgroup> -> !spirv.jointmatrix<8x8xi32, RowMajor, Subgroup>
    spirv.Return
  }
}
