// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [JointMatrixINTEL], [SPV_INTEL_joint_matrix]> {
  // CHECK-LABEL: @joint_matrix_load
  spv.func @joint_matrix_load(%ptr : !spv.ptr<i32, Workgroup>, %stride : i32) "None" {
    // CHECK: {{%.*}} = spv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> {{%.*}}, {{%.*}} : (!spv.ptr<i32, Workgroup>, i32) -> !spv.jointmatrix<16x8xi32, RowMajor, Workgroup>
    %0 = spv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> %ptr, %stride : (!spv.ptr<i32, Workgroup>, i32) -> !spv.jointmatrix<16x8xi32, RowMajor, Workgroup>
    spv.Return
  }

  // CHECK-LABEL: @joint_matrix_load_memaccess
  spv.func @joint_matrix_load_memaccess(%ptr : !spv.ptr<i32, Workgroup>, %stride : i32) "None" {
    // CHECK: {{%.*}} = spv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> {{%.*}}, {{%.*}} {memory_access = #spv.memory_access<Volatile>} : (!spv.ptr<i32, Workgroup>, i32) -> !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>
    %0 = spv.INTEL.JointMatrixLoad <Subgroup> <RowMajor> %ptr, %stride {memory_access = #spv.memory_access<Volatile>} : (!spv.ptr<i32, Workgroup>, i32) -> !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>
    spv.Return
  }

  // CHECK-LABEL: @joint_matrix_store
  spv.func @joint_matrix_store(%ptr : !spv.ptr<i32, Workgroup>, %stride : i32, %m : !spv.jointmatrix<16x8xi32, RowMajor, Workgroup>) "None" {
    // CHECK: spv.INTEL.JointMatrixStore <Subgroup> <RowMajor> {{%.*}}, {{%.*}}, {{%.*}} : (!spv.ptr<i32, Workgroup>, !spv.jointmatrix<16x8xi32, RowMajor, Workgroup>, i32)
    spv.INTEL.JointMatrixStore <Subgroup> <RowMajor> %ptr, %m, %stride : (!spv.ptr<i32, Workgroup>, !spv.jointmatrix<16x8xi32, RowMajor, Workgroup>, i32)
    spv.Return
  }

  // CHECK-LABEL: @joint_matrix_store_memaccess
  spv.func @joint_matrix_store_memaccess(%ptr : !spv.ptr<i32, Workgroup>, %m : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, %stride : i32) "None" {
    // CHECK: spv.INTEL.JointMatrixStore <Subgroup> <RowMajor> {{%.*}}, {{%.*}}, {{%.*}} {memory_access = #spv.memory_access<Volatile>} : (!spv.ptr<i32, Workgroup>, !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, i32)
    spv.INTEL.JointMatrixStore <Subgroup> <RowMajor> %ptr, %m, %stride {memory_access = #spv.memory_access<Volatile>} : (!spv.ptr<i32, Workgroup>, !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, i32)
    spv.Return
  }

  // CHECK-LABEL: @joint_matrix_length
  spv.func @joint_matrix_length() -> i32 "None" {
    // CHECK: {{%.*}} = spv.INTEL.JointMatrixWorkItemLength : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>
    %0 = spv.INTEL.JointMatrixWorkItemLength : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>
    spv.ReturnValue %0 : i32
  }

  // CHECK-LABEL: @joint_matrix_muladd
  spv.func @joint_matrix_muladd(%a : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, %b : !spv.jointmatrix<16x8xi32, RowMajor, Subgroup>, %c : !spv.jointmatrix<8x8xi32, RowMajor, Subgroup>) "None" {
    // CHECK: {{%.*}} = spv.INTEL.JointMatrixMad <Subgroup> {{%.*}}, {{%.*}}, {{%.*}}  : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, !spv.jointmatrix<16x8xi32, RowMajor, Subgroup> -> !spv.jointmatrix<8x8xi32, RowMajor, Subgroup>
    %r = spv.INTEL.JointMatrixMad <Subgroup> %a, %b, %c : !spv.jointmatrix<8x16xi32, RowMajor, Subgroup>, !spv.jointmatrix<16x8xi32, RowMajor, Subgroup> -> !spv.jointmatrix<8x8xi32, RowMajor, Subgroup>
    spv.Return
  }
}
