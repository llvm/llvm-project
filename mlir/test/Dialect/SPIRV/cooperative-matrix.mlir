// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @cooperative_matrix_load
spv.func @cooperative_matrix_load(%ptr : !spv.ptr<i32, StorageBuffer>, %stride : i32, %b : i1) "None" {
  // CHECK: {{%.*}} = spv.CooperativeMatrixLoadNV "StorageBuffer" {{%.*}}, {{%.*}}, {{%.*}} : !spv.coopmatrix<16x8xi32, Workgroup>
  %0 = spv.CooperativeMatrixLoadNV "StorageBuffer" %ptr, %stride, %b : !spv.coopmatrix<16x8xi32, Workgroup>
  spv.Return
}

// -----
// CHECK-LABEL: @cooperative_matrix_load_memaccess
spv.func @cooperative_matrix_load_memaccess(%ptr : !spv.ptr<i32, StorageBuffer>, %stride : i32, %b : i1) "None" {
  // CHECK: {{%.*}} = spv.CooperativeMatrixLoadNV "StorageBuffer" {{%.*}}, {{%.*}}, {{%.*}} ["Volatile"] : !spv.coopmatrix<8x16xi32, Subgroup>
  %0 = spv.CooperativeMatrixLoadNV "StorageBuffer" %ptr, %stride, %b ["Volatile"] : !spv.coopmatrix<8x16xi32, Subgroup>
  spv.Return
}
